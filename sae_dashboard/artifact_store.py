"""Explicit experiment persistence and portable VLM export."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile
from safetensors.torch import save_file
from . import model_runtime as runtime
from .session import Session, Settings

ARTIFACT_ROOT = Path(
    os.getenv("GEMMA_SAE_RUNS_DIR", Path(__file__).resolve().parents[1] / "runs")
).resolve()


def record_event(session: Session, kind: str, payload: dict):
    event = {
        "kind": kind,
        "session_id": session.id,
        "profile_id": session.profile_id,
        "vector_id": session.vector_id,
        "created_unix": time.time(),
        **payload,
    }
    session.results.append(event)
    if session.save_enabled:
        directory = ARTIFACT_ROOT / session.id / "experiments"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{kind}_{uuid.uuid4().hex}.json"
        # Unique names and exclusive creation prevent accidental replacement.
        with path.open("x", encoding="utf-8") as file:
            json.dump(event, file, indent=2, ensure_ascii=False)
    return event


def export_model(
    session: Session,
    settings: Settings,
    strengths: dict[int, float],
    include_weights: bool,
) -> str:
    if not session.vectors:
        raise ValueError("Create steering vectors before exporting the model.")
    root = ARTIFACT_ROOT / session.id / "exports"
    directory = root / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory.with_suffix(".zip")
    try:
        metadata = runtime.runtime_metadata()
        config = {
            "format_version": 1,
            "model_id": runtime.MODEL_ID,
            "model_revision": metadata["model_commit"] or runtime.MODEL_REVISION,
            "profile_id": session.profile_id,
            "vector_id": session.vector_id,
            "steer_last_token_only": runtime.STEER_LAST_TOKEN_ONLY,
            "strengths": {str(k): v for k, v in strengths.items()},
            "generation": {
                key: asdict(settings)[key]
                for key in ("seed", "temperature", "max_new_tokens")
            },
            "capture": {
                "token_scope": settings.token_scope,
                "aggregation": settings.aggregation,
            },
            "vector_metadata": {str(k): v.metadata for k, v in session.vectors.items()},
            "manifests": [
                {"name": m.name, "fingerprint": m.fingerprint}
                for m in session.manifests
            ],
            "runtime": metadata,
        }
        (directory / "steering_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        save_file(
            {
                str(k): v.direction.detach().float().cpu().contiguous()
                for k, v in session.vectors.items()
            },
            str(directory / "steering_vectors.safetensors"),
        )
        shutil.copyfile(
            Path(__file__).with_name("portable_model.py"),
            directory / "steered_model.py",
        )
        packages = metadata["packages"]
        (directory / "requirements.txt").write_text(
            "\n".join(
                f"{p}=={packages[p].split('+', 1)[0]}"
                for p in (
                    "torch",
                    "torchvision",
                    "transformers",
                    "accelerate",
                    "safetensors",
                    "pillow",
                    "sentencepiece",
                )
            )
            + "\n"
        )
        (directory / "README.md").write_text(
            """# Steered Gemma 3

Install requirements, then use the loader in this directory:

```python
from steered_model import SteeredVLM
vlm = SteeredVLM(".")
print(vlm.generate("Explain this code."))
```

This is activation steering: the loader installs residual hooks during generation
and removes them afterwards. The vectors cannot be merged into ordinary model
weights. Loading base_model with AutoModel alone will not apply steering.
If base_model is absent, the pinned base model is downloaded from Hugging Face;
its access requirements still apply. The loader does not require SAEs or this repo.
""",
            encoding="utf-8",
        )
        if include_weights:
            with runtime.MODEL_LOCK:
                runtime.ensure_models_loaded(with_saes=False)
                runtime.model.save_pretrained(
                    directory / "base_model", safe_serialization=True
                )
                runtime.processor.save_pretrained(directory / "base_model")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(directory))
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(directory)
    return str(archive)
