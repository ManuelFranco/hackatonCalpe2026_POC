"""Compact English UI. Keep model and research logic outside this module."""

from functools import wraps
import json
from pathlib import Path
import gradio as gr
from research.benchmarks import BENCHMARKS
from research.vector_builders import BUILDERS
from . import model_runtime as runtime, workflow
from .manifests import DATA_ROOT
from .session import Session, Settings

CSS = """
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
.section-title {border-left: 5px solid #6366f1; padding: 10px 16px; margin: 12px 0 20px;
 background: color-mix(in srgb, #6366f1 8%, transparent); border-radius: 0 10px 10px 0;}
.section-title h2 {font-size: 1.7rem !important; font-weight: 750 !important;}
"""


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
            save_status = gr.Markdown("**Saving off** · This session stays in memory.")
        with gr.Group(visible=False) as help_panel:
            gr.Markdown("""**Quick guide**

Load JSON manifests → build a common profile → create vectors → compare and evaluate → export.

Each browser session has its own data, profile and results. Model jobs share a queue;
base and steered runs use the same seed. Experiment saving starts only when enabled.
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
                        gr.Slider(-10, 10, value=1, step=0.25, label=f"Layer {layer}")
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
                    probe_answer = gr.Textbox(
                        label="Gemma 3 response", lines=12, interactive=False
                    )
            with gr.Tab("1 · Load manifests", id="load"):
                gr.Markdown("## 1. Load manifests", elem_classes="section-title")
                repo_paths = sorted(DATA_ROOT.rglob("*manifest*.json"))
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
                manifest_status = gr.Markdown("Load one or more manifests to begin.")
                manifest_table = gr.Dataframe(
                    headers=["Manifest", "Pairs", "Fingerprint"], interactive=False
                )
                with gr.Accordion("Manifest format", open=False):
                    gr.Markdown("""Each manifest defines one use case with matched A/B conditions.
Use `text` or `text_file`; `image` is optional. Each condition needs at least one modality.
Asset paths resolve inside the server's data directory. `asset_root` keeps uploaded
JSON paths portable. Loading new manifests replaces this session's profile and vectors.

```json
{"version": 1, "name": "Code safety", "asset_root": "cwe_c_pairs_dataset",
 "pairs": [{"id": "CWE-287",
   "A": {"text_file": "cwe_287_A.txt", "image": ""},
   "B": {"text_file": "cwe_287_B.txt", "image": ""}}]}
```
""")
            with gr.Tab("2 · Common profile", id="profile"):
                gr.Markdown("## 2. Build common profile", elem_classes="section-title")
                gr.Markdown("Capture A/B feature activations for every loaded pair.")
                profile_btn = gr.Button("Build common profile", variant="primary")
                profile_status = gr.Markdown()
                profile_table = gr.Dataframe(
                    headers=["Layer", "Pairs", "Features", "Reference norm"],
                    interactive=False,
                )
                with gr.Accordion("Capture details", open=False):
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
                    base_answer = gr.Textbox(label="Base", lines=14, interactive=False)
                    steered_answer = gr.Textbox(
                        label="Steered", lines=14, interactive=False
                    )
            with gr.Tab("5 · Coherence benchmarks", id="benchmarks"):
                gr.Markdown("## 5. Evaluate coherence", elem_classes="section-title")
                with gr.Row():
                    benchmark_name = gr.Dropdown(
                        [(a.name, k) for k, a in BENCHMARKS.items()],
                        value="mmlu_pro",
                        label="Benchmark",
                    )
                    split = gr.Dropdown(
                        list(BENCHMARKS["mmlu_pro"].splits), value="test", label="Split"
                    )
                    count = gr.Number(
                        value=20, minimum=1, maximum=10000, precision=0, label="Items"
                    )
                    category = gr.Textbox(
                        label="Subject (optional)", placeholder="Blank = all subjects"
                    )
                benchmark_btn = gr.Button("Run benchmark", variant="primary")
                benchmark_status = gr.Markdown()
                benchmark_summary = gr.Dataframe(
                    headers=["Metric", "Value"], interactive=False
                )
                with gr.Accordion("Item results", open=False):
                    benchmark_details = gr.Dataframe(
                        headers=[
                            "ID",
                            "Base correct",
                            "Steered correct",
                            "Base",
                            "Steered",
                        ],
                        interactive=False,
                    )
                with gr.Accordion("Scoring and sources", open=False):
                    gr.Markdown("""Paired zero-shot evaluation with a reproducible sample and shared generation settings.
- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro): multiple-choice accuracy.
- [MMMU](https://huggingface.co/datasets/MMMU/MMMU): multiple images and open or multiple-choice answers.
- [IFEval](https://huggingface.co/datasets/google/IFEval): official strict/loose prompt and instruction accuracy.

Preservation measures how often a correct base answer stays correct. These dashboard
runs use their stated protocol and sample; they are not leaderboard reproductions.
IFEval may need a larger shared token limit for long-response instructions.
""")
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
""")

        downstream = [
            profile_status,
            profile_table,
            vector_status,
            vector_table,
            base_answer,
            steered_answer,
            benchmark_status,
            benchmark_summary,
            benchmark_details,
            export_file,
        ]
        cleared = ["", [], "", [], "", "", "", [], [], None]

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
                "**Saving on** · New results are saved for this session."
                if enabled
                else "**Saving off** · This session stays in memory."
            )

        save_btn.click(toggle_saving, session, [save_btn, save_status])

        @friendly
        def load(state, paths):
            table = workflow.set_manifests(
                state, paths if isinstance(paths, list) else ([paths] if paths else [])
            )
            return (
                f"Loaded {len(table)} manifest(s) · {sum(row[1] for row in table)} pairs.",
                table,
                *cleared,
            )

        for button, source in ((load_repo, repository), (load_uploads, uploads)):
            button.click(
                load, [session, source], [manifest_status, manifest_table, *downstream]
            )

        def invalidate(state):
            with state.lock:
                state.invalidate_profile()
            return cleared

        for control in (scope, aggregation):
            control.input(invalidate, session, downstream)

        @friendly
        def capture(state, progress=gr.Progress(), *values):
            table = workflow.build_profile(state, settings(values), progress)
            return "Profile ready.", table, *cleared[2:]

        profile_btn.click(capture, [session, *common], downstream)

        @friendly
        def vectors(state, selected_method, *values):
            table = workflow.create_vectors(state, settings(values), selected_method)
            return "Vectors ready.", table, *cleared[4:]

        vector_btn.click(vectors, [session, method, *common], downstream[2:])

        @friendly
        def run_probe(state, text, image, *values):
            return workflow.probe(state, settings(values), text, image)

        probe_btn.click(
            run_probe, [session, probe_prompt, probe_image, *common], probe_answer
        )

        @friendly
        def run_comparison(state, text, image, *values):
            return workflow.compare(
                state, settings(values), strengths(values[5:]), text, image
            )

        compare_btn.click(
            run_comparison,
            [session, prompt, query_image, *common, *layer_strengths],
            [base_answer, steered_answer],
        )

        def change_benchmark(name):
            adapter = BENCHMARKS[name]
            return (
                gr.update(choices=list(adapter.splits), value=adapter.default_split),
                gr.update(value="", interactive=name != "ifeval"),
                "",
                [],
                [],
            )

        benchmark_name.change(
            change_benchmark,
            benchmark_name,
            [split, category, benchmark_status, benchmark_summary, benchmark_details],
        )

        @friendly
        def run_benchmark(
            state, name, selected_split, items, subject, progress=gr.Progress(), *values
        ):
            result = workflow.benchmark(
                state,
                settings(values),
                strengths(values[5:]),
                name,
                selected_split,
                items,
                subject,
                progress,
            )
            summary = [
                [
                    key.replace("_", " ").capitalize(),
                    "n/a"
                    if value is None
                    else str(round(value, 4))
                    if isinstance(value, float)
                    else str(value),
                ]
                for key, value in result["summary"].items()
            ]
            rows = [
                [
                    r["id"],
                    r["base_score"]["correct"],
                    r["steered_score"]["correct"],
                    r["base"],
                    r["steered"],
                ]
                for r in result["rows"]
            ]
            return (
                f"{result['benchmark']} · {result['summary']['num_items']} items · {result['metric']}",
                summary,
                rows,
            )

        benchmark_btn.click(
            run_benchmark,
            [
                session,
                benchmark_name,
                split,
                count,
                category,
                *common,
                *layer_strengths,
            ],
            [benchmark_status, benchmark_summary, benchmark_details],
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
