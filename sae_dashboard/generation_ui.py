"""Small generation-audit surface; model calls stay in generation_lab."""

from copy import deepcopy

import gradio as gr

from . import generation_lab as lab, generation_view as view, model_runtime as runtime
from .evidence_lab import export_report


def build_generation_lab(
    session, common, layer_strengths, settings, friendly, candidate_controls=None
):
    result = gr.State(None)
    gr.HTML(view.hero())
    prompt = gr.Textbox(value=lab.DEFAULT_PROMPT, label="Generation prompt", lines=3)
    with gr.Row():
        mode = gr.Dropdown(
            lab.MODES, value=lab.MODES[0], label="Run conditions", scale=2
        )
        layer = gr.Dropdown(runtime.LAYERS, value=17, label="Observed layer")
        observed = gr.Textbox(
            value="",
            label="Observed feature IDs · up to 3",
            placeholder="e.g. 3995, 14237 · blank = automatic",
            scale=2,
        )
    if candidate_controls is not None:
        candidate_controls.update(layer=layer, observed=observed)
    gr.Markdown(
        "Blank IDs: observe the **3 largest activation ranges in Base**. Manual IDs are observed together. **Current steering** uses section 3 vectors and the shared strengths above."
    )
    with gr.Row():
        run = gr.Button("Generate & inspect", variant="primary")
        export = gr.Button("Export generation report", interactive=False)
    summary = gr.HTML(view.empty())
    with gr.Row():
        condition = gr.Dropdown(
            [], label="Activation map · response", interactive=False
        )
        position = gr.Slider(
            1, 2, value=1, step=1, label="First token in view", interactive=False
        )
    timeline = gr.HTML()
    identity = gr.HTML()
    with gr.Accordion("Generation report downloads", open=False) as downloads:
        files = gr.File(
            label="Generation report · HTML + JSON",
            file_count="multiple",
            interactive=False,
        )
    outputs = [
        result,
        summary,
        condition,
        position,
        timeline,
        identity,
        files,
        export,
        downloads,
    ]
    cleared = [
        None,
        view.empty(),
        gr.update(choices=[], value=None, interactive=False),
        gr.update(value=1, maximum=2, interactive=False),
        "",
        "",
        gr.update(value=None),
        gr.update(interactive=False),
        gr.update(open=False),
    ]

    @friendly
    def inspect_generation(
        state, text, run_mode, selected_layer, ids, progress=gr.Progress(), *values
    ):
        measured = lab.run_generation(
            state,
            settings(values),
            text,
            run_mode,
            selected_layer,
            ids,
            dict(zip(runtime.LAYERS, map(float, values[5:]))),
            progress,
        )
        return (
            measured,
            view.overview(measured),
            gr.update(
                choices=[r["name"] for r in measured["runs"]],
                value="Base",
                interactive=True,
            ),
            gr.update(
                value=view.activity_start(measured),
                maximum=max(2, len(measured["runs"][0]["trace"]["tokens"])),
                interactive=len(measured["runs"][0]["trace"]["tokens"]) > 1,
            ),
            view.timeline(measured),
            view.metadata(measured),
            gr.update(value=None),
            gr.update(interactive=True),
            gr.update(open=False),
        )

    run_event = run.click(
        inspect_generation,
        [session, prompt, mode, layer, observed, *common, *layer_strengths],
        outputs,
    )

    def change_response(measured, name):
        if not measured or not name:
            return "", gr.update(value=1, maximum=2, interactive=False)
        chosen = next(r for r in measured["runs"] if r["name"] == name)
        return view.timeline(measured, name), gr.update(
            value=view.activity_start(measured, name),
            maximum=max(2, len(chosen["trace"]["tokens"])),
            interactive=len(chosen["trace"]["tokens"]) > 1,
        )

    condition.input(change_response, [result, condition], [timeline, position])
    position.release(
        lambda r, name, pos: view.timeline(r, name, pos) if r and name else "",
        [result, condition, position],
        timeline,
    )

    @friendly
    def export_generation(state, measured):
        return export_report(state, measured), gr.update(open=True)

    export.click(
        export_generation,
        [session, result],
        [files, downloads],
        queue=False,
        show_progress="hidden",
    )
    for control in (prompt, mode, layer, observed, *common, *layer_strengths):
        control.input(
            lambda: tuple(deepcopy(cleared)), None, outputs, cancels=[run_event]
        )
    return outputs, cleared
