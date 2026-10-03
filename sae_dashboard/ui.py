"""Compact English UI. Keep model and research logic outside this module."""

from functools import wraps
import json
from pathlib import Path
import gradio as gr
from .benchmark_ui import build_benchmark_section
from .causal_ui import build_causal_lab
from .gate_ui import build_gate_section
from research.vector_builders import BUILDERS
from . import conditional_gate as gating, model_runtime as runtime, workflow
from .feature_explorer import FEATURE_CSS, NEURONPEDIA_JS, empty_profile, render_profile
from .manifests import DATA_ROOT
from .manual_input_ui import build_manual_editor, input_summary, wire_manual_editor
from .input_preview import build_input_preview, refresh_preview, wire_input_preview
from .session import Session, Settings

CSS = (
    """
.benchmark-meta {display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;}
.benchmark-meta > div {border:1px solid rgba(127,127,127,.2);border-radius:8px;padding:8px 12px;}
.benchmark-meta small {display:block;opacity:.7;font-size:.75rem;}
.benchmark-meta strong {display:block;font-size:.9rem;overflow-wrap:anywhere;}
.input-summary {overflow-x:auto;}
.input-summary table {width:100%;border-collapse:collapse;}
.input-summary th, .input-summary td {padding:10px 12px;text-align:left;border-bottom:1px solid rgba(127,127,127,.2);}
.input-preview {margin-top:18px;}
.input-preview-meta {display:flex;flex-wrap:wrap;gap:8px 20px;padding:8px 0;color:var(--body-text-color-subdued);font-size:.875rem;}
.input-preview img {object-fit:contain;}
gradio-app {width: 100%; min-width: 0;}
.gradio-container {max-width: 1440px !important; width: 100% !important; min-width: 0 !important; margin: auto;}
.gradio-container .main {width: 100%; min-width: 0;}
.gradio-container [role="tablist"] {flex-wrap: wrap; min-width: 0;}
.gradio-container .row > * {min-width: min(160px, 100%) !important;}
@media (max-width: 600px) {
  .gradio-container .main {padding: 16px !important;}
  .gradio-container [role="tab"] {white-space: normal;}
}
#app-title h1 {font-size: clamp(1.75rem, 4vw, 2.35rem); letter-spacing: -.04em; margin-bottom: .1rem;}
.section-title h2 {border-left: 5px solid #6366f1; padding: 10px 16px; margin: 12px 0 20px;
 background: color-mix(in srgb, #6366f1 8%, transparent); border-radius: 0 10px 10px 0;}
.section-title h2 {font-size: 1.7rem !important; font-weight: 750 !important;}
"""
    + FEATURE_CSS
)


def english_widgets():
    """Keep Gradio upload/error widgets in English for every browser locale.

    Widget strings originate from Gradio 6.17.3 (Apache-2.0).
    Use the supported translation API instead of changing browser preferences.
    """
    translations = json.loads(
        Path(__file__).with_name("english_widgets.json").read_text()
    )
    locales = "ar ca ckb de en es et eu fa fi fr he hi id ja ko lt nb nl pl pt-BR pt ro ru sv ta th tr uk ur uz zh-CN zh-TW".split()
    flat = {
        f"{group}.{key}": value
        for group, values in translations.items()
        for key, value in values.items()
    }
    return gr.I18n(**{locale: flat for locale in locales})


def friendly(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, OSError, KeyError) as error:
            raise gr.Error(str(error)) from error

    return wrapped


def settings(values):
    seed, temperature, tokens, scope, aggregation = values[:5]
    result = Settings(seed, float(temperature), int(tokens), scope, aggregation)
    result.validate()
    return result


def strengths(values):
    return dict(zip(runtime.LAYERS, map(float, values)))


def response_panel(title):
    with gr.Column():
        gr.Markdown(f"### {title}")
        return gr.Markdown(
            value="",
            label=title,
            min_height=240,
            max_height=700,
            container=True,
            padding=True,
            buttons=["copy"],
            sanitize_html=True,
            line_breaks=True,
        )


def build_demo():
    with gr.Blocks(
        title="Hackathon 2026", analytics_enabled=False, delete_cache=(3600, 14400)
    ) as demo:
        session = gr.State(Session(), time_to_live=14400)
        help_visible = gr.State(False)
        gr.Markdown("# Hackathon 2026", elem_id="app-title")
        gr.Markdown("Gemma 3 · Contrastive steering workspace")
        with gr.Row():
            save_btn = gr.Button("Enable experiment saving", size="sm")
            more_info = gr.Button("More info", size="sm")
            save_status = gr.Markdown(
                "**Experiment saving off** · Base benchmark cache is automatic."
            )
        with gr.Group(visible=False) as help_panel:
            gr.Markdown("""**Quick guide**

Load JSON manifests or add manual pairs → build a common profile → create vectors → compare and evaluate → export.

Each browser session has its own data, profile and results. Model jobs share a queue;
base and steered runs use the same seed. Experiment saving starts only when enabled.
Base benchmark answers are cached automatically; steered benchmark answers stay in memory.
Exporting a model always requires the final export button. Upload/model/dataset caches
are managed separately by Gradio and Hugging Face.
""")
        with gr.Group():
            gr.Markdown("### Shared parameters")
            with gr.Row():
                seed = gr.Number(
                    value=0, precision=0, label="Seed", minimum=0, maximum=2**32 - 1
                )
                temperature = gr.Slider(0, 2, value=0, step=0.05, label="Temperature")
                tokens = gr.Slider(1, 1024, value=256, step=1, label="Max new tokens")
                scope = gr.Dropdown(
                    ["all", "last", "non_image"],
                    value=runtime.FEATURE_TOKEN_SCOPE,
                    label="Profile tokens",
                )
                aggregation = gr.Dropdown(
                    ["mean", "max"],
                    value=runtime.FEATURE_AGGREGATION,
                    label="Token aggregation",
                )
            with gr.Accordion("Steering strengths", open=False):
                with gr.Row():
                    layer_strengths = [
                        gr.Slider(-10, 10, value=0, step=0.25, label=f"Layer {layer}")
                        for layer in runtime.LAYERS
                    ]
                gr.Markdown(
                    "Positive values add B − A. Zero disables steering for that layer."
                )
        common = [seed, temperature, tokens, scope, aggregation]

        with gr.Tabs(selected="load"):
            with gr.Tab("0 · Try Gemma 3", id="probe"):
                gr.Markdown("## 0. Try Gemma 3", elem_classes="section-title")
                with gr.Row():
                    with gr.Column():
                        probe_prompt = gr.Textbox(
                            label="Prompt",
                            lines=4,
                            placeholder="Ask Gemma 3 a question…",
                        )
                        with gr.Accordion("Optional image", open=False):
                            probe_image = gr.Image(type="pil", label="Image")
                        probe_btn = gr.Button("Run Gemma 3", variant="primary")
                    probe_answer = response_panel("Gemma 3 response")
            with gr.Tab("1 · Load inputs", id="load"):
                gr.Markdown("## 1. Load inputs", elem_classes="section-title")
                repo_paths = [
                    p
                    for p in sorted(DATA_ROOT.rglob("*manifest*.json"))
                    if "cwes" not in json.loads(p.read_text())
                ]
                repository = gr.Dropdown(
                    choices=[
                        (str(p.relative_to(DATA_ROOT)), str(p)) for p in repo_paths
                    ],
                    value=str(repo_paths[0]) if repo_paths else None,
                    label="Repository manifest",
                    multiselect=False,
                )
                uploads = gr.File(
                    label="Upload JSON manifests",
                    file_count="multiple",
                    file_types=[".json"],
                    type="filepath",
                )
                with gr.Row():
                    load_repo = gr.Button("Load repository manifest", variant="primary")
                    load_uploads = gr.Button("Load uploaded manifests")
                manual_editor = build_manual_editor()
                manifest_status = gr.Markdown(
                    "Load JSON manifests or add manual pairs to begin."
                )
                manifest_table = gr.HTML()
                input_preview = build_input_preview()
                with gr.Accordion("Manifest format", open=False):
                    gr.Markdown("""Each manifest defines one use case with matched A/B conditions.
Use `text` or `text_file`; `image` is optional. Each condition needs at least one modality.
Asset paths resolve inside the server's data directory. `asset_root` keeps uploaded
JSON paths portable. Loading manifests replaces the JSON selection and preserves manual pairs.
Changing either input source clears the previous profile and vectors.

```json
{"version": 1, "name": "Code safety", "asset_root": "scripts/cwe_287",
 "pairs": [{"id": "CWE-287",
   "A": {"text_file": "cwe_287_A_01.txt", "image": ""},
   "B": {"text_file": "cwe_287_B_01.txt", "image": ""}}]}
```
""")
            with gr.Tab("2 · Common profile", id="profile"):
                gr.Markdown("## 2. Build common profile", elem_classes="section-title")
                gr.Markdown("Capture A/B feature activations for every loaded pair.")
                profile_btn = gr.Button("Build common profile", variant="primary")
                profile_status = gr.Markdown()
                profile_explorers = []
                with gr.Tabs():
                    for layer in runtime.LAYERS:
                        with gr.Tab(f"Layer {layer}"):
                            profile_explorers.append(
                                gr.HTML(
                                    value=empty_profile(layer),
                                    label=f"Layer {layer} features",
                                    js_on_load=NEURONPEDIA_JS,
                                )
                            )
                with gr.Accordion("Capture details", open=False):
                    profile_table = gr.Dataframe(
                        headers=["Layer", "Pairs", "Features", "Reference norm"],
                        interactive=False,
                    )
                    gr.Markdown(
                        "The profile includes the assistant prefix and excludes generated tokens. "
                        "All tokens are selected by default; last and non_image remain available. "
                        "Changing token scope or aggregation requires rebuilding the profile."
                    )
            with gr.Tab("3 · Steering vectors", id="vectors"):
                gr.Markdown(
                    "## 3. Create steering vectors", elem_classes="section-title"
                )
                method = gr.Dropdown(
                    [(b.name, key) for key, b in BUILDERS.items()],
                    value="mean_difference",
                    label="Method",
                )
                vector_btn = gr.Button("Create vectors", variant="primary")
                vector_status = gr.Markdown()
                vector_table = gr.Dataframe(
                    headers=["Layer", "Nonzero features", "Direction norm"],
                    interactive=False,
                )
                with gr.Accordion("Method details", open=False):
                    gr.Markdown(
                        "The baseline averages B − A across pairs and retains all features. "
                        "The SAE decoder projects the result into the residual stream; scaling uses the calibration norm. "
                        "Research strategies can be added through the vector-builder interface."
                    )
            with gr.Tab("4 · Base vs. steered", id="compare"):
                gr.Markdown(
                    "## 4. Compare base and steered", elem_classes="section-title"
                )
                prompt = gr.Textbox(label="Prompt", lines=4)
                with gr.Accordion("Optional image", open=False):
                    query_image = gr.Image(type="pil", label="Image")
                compare_btn = gr.Button("Run base + steered", variant="primary")
                with gr.Row():
                    base_answer = response_panel("Base")
                    steered_answer = response_panel("Steered")
                gate_trace = gr.Markdown()
                build_gate_section(
                    session, common, layer_strengths, settings, strengths, friendly
                )
            with gr.Tab("5 · Coherence benchmarks", id="benchmarks"):
                benchmark_outputs, benchmark_cleared = build_benchmark_section(
                    session, common, layer_strengths, settings, strengths, friendly
                )
            with gr.Tab("6 · Export VLM", id="export"):
                gr.Markdown("## 6. Export steered VLM", elem_classes="section-title")
                gr.Markdown(
                    "Download vectors, settings and a standalone inference loader."
                )
                include_weights = gr.Checkbox(
                    value=False,
                    label="Include base model weights and processor (large export)",
                )
                export_btn = gr.Button("Export steered VLM", variant="primary")
                export_file = gr.File(label="Model package", interactive=False)
                with gr.Accordion("Load elsewhere", open=False):
                    gr.Markdown("""Extract the ZIP, install its requirements, then run:
```python
from steered_model import SteeredVLM
model = SteeredVLM("path/to/extracted/package")
print(model.generate("Your prompt"))
```
The loader applies activation steering during generation. Without bundled base weights,
it downloads the referenced Gemma model. SAEs are not needed for inference.
When the vehicle gate is enabled (section 4), the package also contains the gate
probe and steering is applied only when the gate opens.
""")

            with gr.Tab("7 · Extra", id="extra"):
                causal_selector, causal_outputs, causal_cleared = build_causal_lab(
                    session,
                    common,
                    settings,
                    friendly,
                    response_panel,
                    prompt,
                    query_image,
                )

        downstream = [
            profile_status,
            profile_table,
            vector_status,
            vector_table,
            base_answer,
            steered_answer,
            *benchmark_outputs,
            export_file,
            causal_selector,
            *causal_outputs,
        ]
        cleared = [
            "",
            [],
            "",
            [],
            "",
            "",
            *benchmark_cleared,
            None,
            gr.update(choices=[], value=None),
            *causal_cleared,
        ]
        empty_explorers = [empty_profile(layer) for layer in runtime.LAYERS]

        def profile_views(state):
            labels = [
                f"{manifest.name} / {pair.id}"
                for manifest in state.manifests
                for pair in manifest.pairs
            ]
            linked = (
                runtime.MODEL_ID == "google/gemma-3-4b-it"
                and runtime.SAE_RELEASE == "gemma-scope-2-4b-it-res"
            )
            return [
                render_profile(
                    layer,
                    state.profile[layer],
                    state.vectors.get(layer),
                    labels,
                    neuronpedia=linked,
                )
                if layer in state.profile
                else empty_profile(layer)
                for layer in runtime.LAYERS
            ]

        def toggle_help(visible):
            return not visible, gr.update(visible=not visible)

        more_info.click(
            toggle_help, help_visible, [help_visible, help_panel], queue=False
        )

        def toggle_saving(state):
            with state.lock:
                state.save_enabled = not state.save_enabled
                enabled = state.save_enabled
            return gr.update(
                value="Disable experiment saving"
                if enabled
                else "Enable experiment saving"
            ), (
                "**Experiment saving on** · Steered benchmark responses stay in memory."
                if enabled
                else "**Experiment saving off** · Base benchmark cache is automatic."
            )

        save_btn.click(toggle_saving, session, [save_btn, save_status])

        @friendly
        def load(state, paths):
            with state.lock:
                table = workflow.set_manifests(
                    state,
                    paths if isinstance(paths, list) else ([paths] if paths else []),
                )
                return (
                    f"Loaded {len(table)} manifest(s) · {sum(row[1] for row in table)} pairs.",
                    input_summary(table),
                    *cleared,
                    *refresh_preview(state),
                    *empty_explorers,
                )

        for button, source in ((load_repo, repository), (load_uploads, uploads)):
            button.click(
                load,
                [session, source],
                [
                    manifest_status,
                    manifest_table,
                    *downstream,
                    *input_preview,
                    *profile_explorers,
                ],
            )

        wire_manual_editor(
            session,
            manual_editor,
            [manifest_status, manifest_table, *downstream, *profile_explorers],
            [*cleared, *empty_explorers],
            friendly,
            preview_outputs=input_preview,
        )
        wire_input_preview(session, input_preview, friendly)

        def invalidate(state):
            with state.lock:
                state.invalidate_profile()
            return [*cleared, *empty_explorers]

        for control in (scope, aggregation):
            control.input(invalidate, session, [*downstream, *profile_explorers])

        @friendly
        def capture(state, progress=gr.Progress(), *values):
            with state.lock:
                table = workflow.build_profile(state, settings(values), progress)
                return "Profile ready.", table, *cleared[2:], *profile_views(state)

        profile_btn.click(
            capture, [session, *common], [*downstream, *profile_explorers]
        )

        @friendly
        def vectors(state, selected_method, *values):
            with state.lock:
                table = workflow.create_vectors(
                    state, settings(values), selected_method
                )
                return "Vectors ready.", table, *cleared[4:], *profile_views(state)

        vector_btn.click(
            vectors, [session, method, *common], [*downstream[2:], *profile_explorers]
        )

        @friendly
        def run_probe(state, text, image, *values):
            return workflow.probe(state, settings(values), text, image)

        probe_btn.click(
            run_probe, [session, probe_prompt, probe_image, *common], probe_answer
        )

        @friendly
        def run_comparison(state, text, image, *values):
            if state.gate_enabled:
                return gating.compare_gated(
                    state, settings(values), strengths(values[5:]), text, image
                )
            base, steered = workflow.compare(
                state, settings(values), strengths(values[5:]), text, image
            )
            return base, steered, gating.trace_summary(None, None)

        compare_btn.click(
            run_comparison,
            [session, prompt, query_image, *common, *layer_strengths],
            [base_answer, steered_answer, gate_trace],
        )

        @friendly
        def run_export(state, weights, *values):
            return workflow.export(
                state, settings(values), strengths(values[5:]), weights
            )

        export_btn.click(
            run_export,
            [session, include_weights, *common, *layer_strengths],
            export_file,
        )
    return demo
