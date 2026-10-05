"""Transfer evaluation alongside existing coherence benchmarks."""

from copy import deepcopy
import gradio as gr

from . import transfer_lab as lab, transfer_view as view
from .manifests import DATA_ROOT


def build_transfer_section(
    session, common, layer_strengths, settings, strengths, friendly, repo_paths
):
    gr.Markdown("### Transfer across vulnerabilities")
    gr.Markdown(
        "Freeze vectors from section 3, then apply the same intervention to reserved code families. **All sampled cases are evaluated**, including base failures."
    )
    with gr.Row():
        source_name = gr.Textbox(
            label="Source name", placeholder="e.g. SQL · code only · L17 +1", scale=3
        )
        freeze = gr.Button("Freeze current vectors", scale=1)
        clear = gr.Button("Clear frozen sources", scale=1)
    source_table = gr.HTML(view.sources(None))
    source_ids = gr.Dropdown(
        [], label="Frozen sources to evaluate", multiselect=True, interactive=True
    )
    targets = gr.Dropdown(
        [(str(p.relative_to(DATA_ROOT)), str(p)) for p in repo_paths],
        value=[
            str(DATA_ROOT / p) for p in lab.DEFAULT_TARGETS if (DATA_ROOT / p).exists()
        ],
        multiselect=True,
        label="Evaluation manifests",
    )
    with gr.Accordion("Upload evaluation manifests (optional)", open=False):
        uploads = gr.File(
            label="Additional evaluation manifests",
            file_types=[".json"],
            file_count="multiple",
            type="filepath",
        )
        gr.Markdown(
            "Use the same A/B manifest format, with `labels.A` and `labels.B` describing the two paths. Names must be distinct. Uploaded asset paths use the existing data-root rules."
        )
    count = gr.Slider(1, 20, value=4, step=1, label="Pairs per evaluation set")
    with gr.Row():
        prepare = gr.Button("1 · Prepare transfer sample", variant="primary")
        run = gr.Button("2 · Run transfer matrix", interactive=False)
        export = gr.Button("Export transfer report", interactive=False)
    ready = gr.HTML()
    sample = gr.Dropdown([], label="Preview transfer input", interactive=False)
    prompt_preview = gr.HTML()
    status = gr.Markdown()
    matrix = gr.HTML(view.empty())
    with gr.Row():
        cell = gr.Dropdown([], label="Inspect transfer cell", interactive=False)
        case = gr.Dropdown([], label="Inspect transfer case", interactive=False)
    details = gr.HTML()
    files = gr.File(
        label="Transfer report · HTML + JSON", file_count="multiple", interactive=False
    )
    with gr.Accordion("How to use transfer", open=False):
        gr.Markdown("""1. Load a **training** manifest, choose Profile tokens, build the profile and create vectors.
2. Set nonzero layer strengths and **Freeze current vectors** with a distinct source name.
3. Repeat for another family or capture scope if desired. Frozen sources survive loading new inputs; later strength changes do not modify them.
4. Select frozen sources and **validation** manifests. Prepare the sample and inspect the exact prompts and expected labels.
5. Run the matrix. Compare corrections, regressions, false alarms and misses, then inspect individual cases.
6. Freeze your choices before selecting **test** manifests. Export the HTML/JSON report.

Use **Code only** for comparable SQL, command and XSS source profiles. SQL-specific scopes require Python database calls and are useful for comparing SQL capture regions.
The matrix uses one next-token A/B forward per condition; temperature and output length do not affect this readout. BASE is computed once per input and reused across sources in the run. Transfer inference does not use SAEs.
Exact calibration overlap is marked exploratory. Small synthetic sets and related templates do not establish generalization to real applications. Use the coherence benchmarks separately to assess preserved capabilities.
""")

    outputs = [
        ready,
        sample,
        prompt_preview,
        status,
        matrix,
        cell,
        case,
        details,
        files,
        run,
        export,
    ]
    cleared = [
        "",
        gr.update(choices=[], value=None, interactive=False),
        "",
        "",
        view.empty(),
        gr.update(choices=[], value=None, interactive=False),
        gr.update(choices=[], value=None, interactive=False),
        "",
        gr.update(value=None),
        gr.update(interactive=False),
        gr.update(interactive=False),
    ]

    @friendly
    def freeze_transfer_source(state, name, selected, *values):
        identifier = lab.freeze_current(
            state, settings(values), strengths(values[5:]), name
        )
        selected = [k for k in (selected or []) if k in state.transfer_sources] + [
            identifier
        ]
        return (
            view.sources(state),
            gr.update(choices=view.source_choices(state), value=selected),
            *deepcopy(cleared),
        )

    freeze.click(
        freeze_transfer_source,
        [session, source_name, source_ids, *common, *layer_strengths],
        [source_table, source_ids, *outputs],
    )

    def clear_transfer_sources(state):
        lab.clear_sources(state)
        return view.sources(state), gr.update(choices=[], value=[]), *deepcopy(cleared)

    clear.click(clear_transfer_sources, session, [source_table, source_ids, *outputs])

    @friendly
    def prepare_transfer_sample(state, selected, paths, uploaded, pairs, seed):
        plan = lab.prepare_transfer(
            state, selected, [*(paths or []), *(uploaded or [])], pairs, seed
        )
        choices = [
            (f"{t['name']} · {c['pair_id']} · {c['expected']}", c["case_id"])
            for t in plan["targets"]
            for c in t["cases"]
        ]
        return (
            view.prepared(plan, state),
            gr.update(choices=choices, value=choices[0][1], interactive=True),
            view.preview(plan, choices[0][1]),
            "Sample ready. Review the exact prompts before running.",
            view.empty(),
            *deepcopy(cleared[5:9]),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    prepare.click(
        prepare_transfer_sample,
        [session, source_ids, targets, uploads, count, common[0]],
        outputs,
    )
    sample.input(
        lambda state, identifier: view.preview(state.transfer_plan, identifier),
        [session, sample],
        prompt_preview,
    )

    @friendly
    def run_transfer_matrix(state, progress=gr.Progress()):
        result = lab.run_transfer(state, progress)
        cells = view.cell_choices(result)
        cases = view.case_choices(result, cells[0][1])
        return (
            f"Matrix complete · {result['forwards']} forwards · all sampled inputs evaluated.",
            view.matrix(result),
            gr.update(choices=cells, value=cells[0][1], interactive=True),
            gr.update(choices=cases, value=cases[0][1], interactive=True),
            view.detail(result, cells[0][1], cases[0][1]),
            gr.update(value=None),
            gr.update(interactive=True),
        )

    run_event = run.click(run_transfer_matrix, session, outputs[3:9] + [export])

    @friendly
    def select_transfer_cell(state, identifier):
        choices = view.case_choices(state.transfer_result, identifier)
        first = choices[0][1] if choices else None
        return gr.update(
            choices=choices, value=first, interactive=bool(choices)
        ), view.detail(state.transfer_result, identifier, first)

    cell.input(select_transfer_cell, [session, cell], [case, details])
    case.input(
        lambda state, identifier, item: view.detail(
            state.transfer_result, identifier, item
        ),
        [session, cell, case],
        details,
    )
    export.click(friendly(lab.export_report), session, files)

    def reset_transfer(state):
        with state.lock:
            lab.reset_evaluation(state)
        return tuple(deepcopy(cleared))

    for control in (source_ids, targets, count, common[0]):
        control.input(reset_transfer, session, outputs, cancels=[run_event])
    for event in (uploads.upload, uploads.clear):
        event(reset_transfer, session, outputs, cancels=[run_event])
    return outputs, cleared
