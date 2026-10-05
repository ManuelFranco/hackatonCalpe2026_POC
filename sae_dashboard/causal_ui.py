"""Extra: compact visual evidence for single-feature causal effects."""

from copy import deepcopy

import gradio as gr

from . import evidence_lab as lab, evidence_view as view, model_runtime as runtime
from .generation_ui import build_generation_lab


def build_causal_lab(session, common, settings, friendly, layer_strengths):
    with gr.Tabs():
        with gr.Tab("Full generation"):
            generation_outputs, generation_cleared = build_generation_lab(
                session, common, layer_strengths, settings, friendly
            )
        with gr.Tab("A/B diagnostic"):
            candidates, outputs, cleared = build_decision_lab(
                session, common, settings, friendly
            )
    return candidates, [*outputs, *generation_outputs], [*cleared, *generation_cleared]


def build_decision_lab(session, common, settings, friendly):
    prepared = gr.State(None)
    result = gr.State(None)
    with gr.Column(elem_id="evidence-lab"):
        gr.HTML(view.hero())
        with gr.Row():
            source = gr.Dropdown(
                lab.SOURCES, value=lab.SOURCES[0], label="Evaluation examples", scale=3
            )
            layer = gr.Dropdown(runtime.LAYERS, value=17, label="Study layer", scale=1)
            prepare = gr.Button("1 · Prepare study", scale=1)
        ready = gr.HTML()
        candidates = gr.Dropdown(
            [],
            value=[],
            multiselect=True,
            max_choices=3,
            label="Features to compare · up to 3",
            interactive=True,
        )
        with gr.Row():
            definition_a = gr.Textbox(
                label="A means", lines=1, placeholder="Description of condition A"
            )
            definition_b = gr.Textbox(
                label="B means", lines=1, placeholder="Description of condition B"
            )
        with gr.Accordion("Study settings", open=False) as study_settings:
            with gr.Row():
                pairs = gr.Slider(1, 6, value=4, step=1, label="Evaluation pairs")
                activity = gr.Slider(
                    0,
                    100,
                    value=50,
                    step=25,
                    label="Change current feature activity · ±%",
                )
            manual = gr.Textbox(
                label="Additional feature IDs (optional)",
                placeholder="Comma-separated IDs; at most 3 features total",
            )
            uploaded = gr.File(
                label="Evaluation manifest",
                file_types=[".json"],
                type="filepath",
            )
            gr.Markdown(
                "One feature at a time, at the **final input position**. Current activity is rescaled in its decoder direction, with an intended **2% residual budget**. "
                "Two random directions match each input's perturbation norm. An inactive feature stays inactive. "
                "A/B descriptions are supplied by you or the dataset; confirm that they match the pairs."
            )
        with gr.Row():
            run = gr.Button(
                "2 · Compare features", variant="primary", elem_classes="el-primary"
            )
            export = gr.Button("Export presentation & data", interactive=False)
        status = gr.Markdown()
        summary = gr.HTML(view.empty())
        with gr.Row():
            selection = gr.Dropdown(
                [], label="Inspect an intervention", interactive=False, scale=2
            )
            example = gr.Dropdown(
                [], label="Inspect an example", interactive=False, scale=3
            )
        details = gr.HTML()
        identity = gr.HTML()
        with gr.Accordion("3 · Report downloads", open=False) as downloads:
            files = gr.File(
                label="Study report · HTML + JSON",
                file_count="multiple",
                interactive=False,
            )

    result_outputs = [
        result,
        summary,
        selection,
        example,
        details,
        identity,
        files,
        status,
        export,
        downloads,
    ]
    clear_results = [
        None,
        view.empty(),
        gr.update(choices=[], value=None, interactive=False),
        gr.update(choices=[], value=None, interactive=False),
        "",
        "",
        gr.update(value=None),
        "",
        gr.update(interactive=False),
        gr.update(open=False),
    ]

    @friendly
    def prepare_evidence(state, data_source, upload, selected_layer, *values):
        study = lab.prepare_study(
            state, settings(values), data_source, upload, selected_layer
        )
        choices = [
            (
                f"#{c['feature']} · {c['same_sign']}/{c['pairs']} same sign",
                str(c["feature"]),
            )
            for c in study["candidates"]
        ]
        return (
            study,
            view.prepared_card(study),
            gr.update(choices=choices, value=[c[1] for c in choices[:3]]),
            study["definitions"]["A"],
            study["definitions"]["B"],
            "",
            *deepcopy(clear_results),
        )

    prepare.click(
        prepare_evidence,
        [session, source, uploaded, layer, *common],
        [
            prepared,
            ready,
            candidates,
            definition_a,
            definition_b,
            manual,
            *result_outputs,
        ],
    )

    @friendly
    def compare_evidence(
        state,
        study,
        selected,
        extra,
        a,
        b,
        count,
        percent,
        progress=gr.Progress(),
        *values,
    ):
        measured = lab.run_study(
            state,
            settings(values),
            study,
            selected,
            extra,
            a,
            b,
            count,
            percent,
            progress,
        )
        ranked = lab.feature_groups(measured)
        group_choices = [
            (f"#{g['feature']} · {g['mode']} activity", g["key"]) for g in ranked
        ]
        case_choices = [(c["label"], c["case_id"]) for c in measured["cases"]]
        first_group, first_case = group_choices[0][1], case_choices[0][1]
        return (
            measured,
            view.overview(measured),
            gr.update(choices=group_choices, value=first_group, interactive=True),
            gr.update(choices=case_choices, value=first_case, interactive=True),
            view.detail(measured, first_group, first_case),
            view.metadata(measured),
            gr.update(value=None),
            f"**Study complete** · {measured['forwards']} forwards · A/B decision readout, not free-response generation.",
            gr.update(interactive=True),
            gr.update(open=False),
        )

    run.click(
        compare_evidence,
        [
            session,
            prepared,
            candidates,
            manual,
            definition_a,
            definition_b,
            pairs,
            activity,
            *common,
        ],
        result_outputs,
    )

    @friendly
    def inspect_evidence(measured, group_key, case_id):
        if not measured or not group_key or not case_id:
            return ""
        return view.detail(measured, group_key, case_id)

    for control in (selection, example):
        control.input(inspect_evidence, [result, selection, example], details)

    @friendly
    def export_evidence(state, measured):
        return lab.export_report(state, measured), gr.update(open=True)

    export.click(
        export_evidence,
        [session, result],
        [files, downloads],
        queue=False,
        show_progress="hidden",
    )

    def reset_setup(data_source):
        return (
            None,
            "",
            gr.update(choices=[], value=[]),
            "",
            "",
            "",
            *deepcopy(clear_results),
        )

    for control in (source, layer):
        control.input(
            reset_setup,
            source,
            [
                prepared,
                ready,
                candidates,
                definition_a,
                definition_b,
                manual,
                *result_outputs,
            ],
        )
    for event in (uploaded.upload, uploaded.clear):
        event(
            reset_setup,
            source,
            [
                prepared,
                ready,
                candidates,
                definition_a,
                definition_b,
                manual,
                *result_outputs,
            ],
        )
    source.input(
        lambda value: gr.update(open=value == lab.SOURCES[2]), source, study_settings
    )
    for control in (
        candidates,
        manual,
        definition_a,
        definition_b,
        pairs,
        activity,
        common[0],
    ):
        control.input(lambda: tuple(deepcopy(clear_results)), None, result_outputs)

    # Reuse the existing app-wide invalidation contract when input/profile changes.
    outputs = [prepared, ready, definition_a, definition_b, manual, *result_outputs]
    cleared = [None, "", "", "", "", *clear_results]
    return candidates, outputs, cleared
