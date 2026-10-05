"""Full-response audit using the existing generation and steering path."""

from dataclasses import asdict
import time
import uuid

from research.code_review import review_python
from research.generation_trace import GenerationTrace, feature_ids

from . import model_runtime as runtime
from .artifact_store import record_event
from .workflow import require_profile, validate_strengths


MODES = ["Base only", "Base + current steering"]
DEFAULT_PROMPT = (
    "Write a Python sqlite3 function lookup_user(conn, user_id) that returns a user "
    "by ID. Use a bound SQL parameter. Return only Python code."
)


def run_generation(
    session, settings, prompt, mode, layer, observed, strengths, progress=None
):
    settings.validate()
    observed = (observed or "").strip()
    ids = feature_ids(observed)
    if not (prompt or "").strip():
        raise ValueError("Enter a generation prompt.")
    if mode not in MODES or layer not in runtime.LAYERS:
        raise ValueError("Select a generation mode and an available layer.")
    compare = mode == MODES[1]
    with session.lock, runtime.MODEL_LOCK:
        if compare:
            require_profile(session, settings, vectors=True)
            validate_strengths(strengths)
            missing = [
                k for k, v in strengths.items() if v and k not in session.vectors
            ]
            if missing:
                raise ValueError(f"Missing current vectors for layers {missing}.")
        if progress:
            progress(0, desc="Loading model and SAE dictionaries")
        runtime.ensure_models_loaded(sae_layers=[layer])
        width = runtime.saes[layer].W_dec.shape[0]
        if any(j >= width for j in ids):
            raise ValueError(f"Layer {layer} has feature IDs 0–{width - 1}.")
        inputs, input_ids, length = runtime.prepare_inputs(None, prompt)
        kwargs = dict(
            inputs=inputs,
            input_len=length,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            seed=settings.seed,
        )
        runs = []
        for name in ["Base", "Current steering"] if compare else ["Base"]:
            if progress:
                progress(len(runs) / (2 if compare else 1), desc=f"Generating {name}")
            trace = GenerationTrace(layer)
            steering = (
                {}
                if name == "Base"
                else dict(
                    steering_directions={
                        k: v.direction for k, v in session.vectors.items()
                    },
                    strengths=strengths,
                )
            )
            response = runtime.generate_answer(**kwargs, **steering, trace=trace)
            if name == "Base" and not observed.strip():
                ids = trace.select(runtime)
            if progress:
                progress(
                    (len(runs) + 0.5) / (2 if compare else 1),
                    desc=f"Measuring {name} activations",
                )
            runs.append(
                {
                    "name": name,
                    "response": response,
                    "review": review_python(response),
                    "trace": trace.measure(
                        runtime, ids, settings.max_new_tokens, response
                    ),
                }
            )
        result = {
            "kind": "generation_audit",
            "protocol": 1,
            "id": uuid.uuid4().hex,
            "created_unix": time.time(),
            "model": runtime.MODEL_ID,
            "sae_release": runtime.SAE_RELEASE,
            "sae_id": runtime.SAE_IDS[layer],
            "layer": layer,
            "dictionary_size": width,
            "features": ids,
            "selection": "Manual observation"
            if observed.strip()
            else "Largest activation range in Base (ties: peak activity, then ID)",
            "settings": asdict(settings),
            "prompt": prompt,
            "input_ids": input_ids.tolist(),
            "mode": mode,
            "profile_id": session.profile_id if compare else None,
            "vector_id": session.vector_id if compare else None,
            "strengths": {str(k): v for k, v in strengths.items()} if compare else {},
            "vector_metadata": {str(k): v.metadata for k, v in session.vectors.items()}
            if compare
            else {},
            "steer_last_token_only": runtime.STEER_LAST_TOKEN_ONLY,
            "runs": runs,
        }
        record_event(session, "generation_audit", result)
        if progress:
            progress(1, desc="Audit complete")
        return result
