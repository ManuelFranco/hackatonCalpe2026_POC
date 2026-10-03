"""Conditional-steering controls. Gate math lives in research/conditional_steering.py."""

import gradio as gr
from . import conditional_gate as gating, model_runtime as runtime
from .gate_manifests import repository_gate_manifests
from .manifests import DATA_ROOT
from .session import Session

GATE_HEADERS = [
    "Split",
    "Manifest",
    "Target",
    "Target images",
    "Non-target images",
    "Hard negatives",
]
METRIC_HEADERS = ["Split", "TP", "FP", "TN", "FN", "TPR", "FPR", "TNR"]
LEAK_HEADERS = [
    "Example",
    "Category",
    "Role",
    "Gate score",
    "g",
    "Global damage shift",
    "Gated damage shift",
]


def build_gate_section(session, common, layer_strengths, settings, strengths, friendly):
    with gr.Accordion("Conditional steering · vehicle gate", open=False):
        gr.Markdown(
            "Global steering adds the same vector to every input: "
            "`h' = h + αv`. With the gate enabled, a linear probe reads the mean "
            "**image-token** residual at the gate layer during prefill and the "
            "steering becomes `h' = h + α·g·v`. In hard mode `g` is 0 or 1; a closed "
            "gate means zero intervention. The probe is calibrated from its own "
            "labelled manifests (`\"kind\": \"binary_gate\"`) and survives profile "
            "and vector rebuilds. See `docs/conditional_steering.md`."
        )
        choices = [
            (str(p.relative_to(DATA_ROOT)), str(p)) for p in repository_gate_manifests()
        ]
        with gr.Row():
            train = gr.Dropdown(choices, value=None, label="Gate training manifest")
            validation = gr.Dropdown(
                choices, value=None, label="Gate validation manifest"
            )
        with gr.Row():
            train_upload = gr.File(
                label="Upload gate training manifest",
                file_types=[".json"],
                type="filepath",
            )
            validation_upload = gr.File(
                label="Upload gate validation manifest",
                file_types=[".json"],
                type="filepath",
            )
        load_btn = gr.Button("Load gate manifests")
        manifests = gr.Dataframe(headers=GATE_HEADERS, interactive=False)
        with gr.Row():
            layer = gr.Dropdown(
                runtime.LAYERS, value=runtime.LAYERS[0], label="Gate layer"
            )
            mode = gr.Radio(["hard", "soft"], value="hard", label="Gate mode")
            temperature = gr.Number(
                value=1.0, minimum=1e-6, label="Soft-gate temperature"
            )
            max_fpr = gr.Slider(
                0, 0.5, value=0, step=0.01, label="Max validation false-positive rate"
            )
            l2 = gr.Number(value=0.01, minimum=0, label="Probe L2 penalty")
        build_btn = gr.Button("Build vehicle gate", variant="primary")
        status = gr.Markdown(gating.gate_status(Session()))
        metrics = gr.Dataframe(headers=METRIC_HEADERS, interactive=False)
        enabled = gr.Checkbox(
            value=False, label="Enable vehicle gate (conditional steering)"
        )
        with gr.Accordion("Leakage evaluation", open=False):
            gr.Markdown(
                "For each labelled image: next-token log-odds of *Yes* vs *No* for "
                "a damage question and an intact-control question, base vs. global "
                "vs. gated steering. Damage shift = Δ(damaged?) − Δ(intact?), so a "
                "plain *Yes* bias cancels. Uses the strengths above."
            )
            eval_manifest = gr.Dropdown(choices, value=None, label="Evaluation manifest")
            eval_upload = gr.File(
                label="Upload evaluation manifest", file_types=[".json"], type="filepath"
            )
            open_text = gr.Checkbox(
                value=False, label="Also generate open-ended descriptions (slow)"
            )
            eval_btn = gr.Button("Evaluate leakage")
            eval_summary = gr.Markdown()
            eval_table = gr.Dataframe(headers=LEAK_HEADERS, interactive=False)

    @friendly
    def load_gate(state, train_path, validation_path, train_file, validation_file):
        with state.lock:
            table = gating.set_gate_manifests(
                state, train_file or train_path, validation_file or validation_path
            )
            return table, gating.gate_status(state), [], False

    load_btn.click(
        load_gate,
        [session, train, validation, train_upload, validation_upload],
        [manifests, status, metrics, enabled],
    )

    def invalidate_gate(state):
        with state.lock:
            if state.conditional_gate is not None:
                state.invalidate_gate()
            return gating.gate_status(state), [], False

    for control in (layer, mode, temperature, max_fpr, l2):
        control.input(invalidate_gate, session, [status, metrics, enabled])

    @friendly
    def build_gate(
        state, gate_layer, gate_mode, temp, fpr, penalty, progress=gr.Progress(), *values
    ):
        with state.lock:
            gating.build_gate(
                state,
                int(gate_layer),
                gate_mode,
                float(temp),
                float(fpr),
                float(penalty),
                settings(values).seed,
                progress,
            )
            return gating.gate_status(state), gating.gate_metrics_table(state), False

    build_btn.click(
        build_gate,
        [session, layer, mode, temperature, max_fpr, l2, *common],
        [status, metrics, enabled],
    )

    def toggle_gate(state, value):
        try:
            return gating.set_gate_enabled(state, value), gr.update()
        except ValueError as error:
            return f"{gating.gate_status(state)}  \n⚠️ {error}", False

    enabled.input(toggle_gate, [session, enabled], [status, enabled])

    @friendly
    def evaluate(state, path, upload, with_text, progress=gr.Progress(), *values):
        summary, rows, _ = gating.evaluate_leakage(
            state,
            settings(values),
            strengths(values[5:]),
            upload or path,
            bool(with_text),
            progress,
        )
        return summary, rows

    eval_btn.click(
        evaluate,
        [session, eval_manifest, eval_upload, open_text, *common, *layer_strengths],
        [eval_summary, eval_table],
    )
