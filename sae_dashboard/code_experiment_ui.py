"""Common-profile integration; generated-code experiments do not change vectors."""

from copy import deepcopy

import gradio as gr

from . import code_experiment_lab as lab, code_experiment_view as view
from .evidence_lab import export_report
from .artifact_store import record_event


def cleared(ready=False):
    return [
        None,
        view.empty(),
        gr.update(choices=[], value=None, interactive=False),
        gr.update(choices=[], value=None, interactive=False),
        "",
        gr.update(interactive=ready),
        gr.update(interactive=False),
        gr.update(value=None),
        gr.update(open=False),
    ]


def refresh(session):
    with session.lock:
        return cleared(bool(session.profile))


def build_code_experiment_panel(session, common, layer, settings, friendly):
    with gr.Accordion("Test candidates on generated SQL", open=False):
        gr.Markdown(
            "Automatically test features, pairs and timing. Check ordinary lookups and one SQL probe; validate a frozen candidate on reserved tasks. Uses **Discovery layer** above."
        )
        with gr.Row():
            polarity = gr.Dropdown(
                lab.POLARITIES, value=lab.POLARITIES[0], label="Candidate contrasts"
            )
            goal = gr.Dropdown(
                lab.GOALS, value=lab.GOALS[0], label="SQL experiment goal"
            )
        with gr.Accordion("Experiment settings", open=False):
            with gr.Row():
                candidates = gr.Slider(
                    1, 6, value=6, step=1, label="Candidates to test"
                )
                cases = gr.Slider(1, 4, value=2, step=1, label="Examples per split")
                budget = gr.Slider(
                    0.5, 5, value=2, step=0.5, label="Residual budget (%)"
                )
                window = gr.Slider(
                    1, 64, value=8, step=1, label="Initial continuation window"
                )
            gr.Markdown(
                "Default: up to **47 generations**. Uses shared seed, temperature and token limit. Search starts from the same function prefix; reserved full generation also tests without it."
            )
        with gr.Row():
            run = gr.Button("Discover & test", variant="primary", interactive=False)
            stop = gr.Button("Stop / clear", size="sm")
            reevaluate = gr.Button("Re-evaluate displayed code", size="sm")
            export = gr.Button("Export experiment report", interactive=False)
        gr.Markdown(
            "Re-evaluation updates the checks without generating again; the frozen candidate stays unchanged."
        )
        result = gr.State(None)
        summary = gr.HTML(value=view.empty(), label="Generated code experiment results")
        with gr.Row():
            comparison = gr.Dropdown(
                [], label="Code comparison", interactive=False, scale=2
            )
            example = gr.Dropdown([], label="SQL task", interactive=False)
        detail = gr.HTML(label="Generated code experiment comparison")
        with gr.Accordion("Experiment report downloads", open=False) as downloads:
            files = gr.File(
                label="Code experiment · HTML + JSON",
                file_count="multiple",
                interactive=False,
            )
    outputs = [
        result,
        summary,
        comparison,
        example,
        detail,
        run,
        export,
        files,
        downloads,
    ]

    @friendly
    def run_code_experiment(
        state,
        selected_layer,
        sign,
        target,
        count,
        examples,
        percent,
        tokens,
        progress=gr.Progress(),
        *values,
    ):
        measured = lab.run_experiment(
            state,
            settings(values),
            selected_layer,
            sign,
            target,
            count,
            examples,
            percent,
            tokens,
            progress,
        )
        return published(measured)

    def published(measured):
        selected = measured["selected_id"]
        choices = view.case_choices(measured, selected)
        return (
            measured,
            view.overview(measured),
            gr.update(choices=view.choices(measured), value=selected, interactive=True),
            gr.update(choices=choices, value=choices[0][1], interactive=True),
            view.comparison(measured, selected, choices[0][1]),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=None),
            gr.update(open=False),
        )

    event = run.click(
        run_code_experiment,
        [session, layer, polarity, goal, candidates, cases, budget, window, *common],
        outputs,
    )

    @friendly
    def reevaluate_code_experiment(state, measured):
        with state.lock:
            if not measured or measured["profile_id"] != state.profile_id:
                raise ValueError("Run an experiment for the current profile first.")
            if state.code_experiment_job is not None:
                raise ValueError(
                    "Wait for the current generation to finish before re-evaluating."
                )
            updated = lab.reevaluate_result(measured)
            record_event(state, "code_experiment", updated)
            return published(updated)

    reevaluation_event = reevaluate.click(
        reevaluate_code_experiment, [session, result], outputs
    )

    def change_code_comparison(measured, selection):
        if not measured or not selection:
            return gr.update(choices=[], value=None, interactive=False), ""
        choices = view.case_choices(measured, selection)
        return gr.update(
            choices=choices, value=choices[0][1], interactive=True
        ), view.comparison(measured, selection, choices[0][1])

    comparison.input(
        change_code_comparison, [result, comparison], [example, detail], queue=False
    )
    example.input(
        lambda r, s, c: view.comparison(r, s, c) if r and s and c else "",
        [result, comparison, example],
        detail,
        queue=False,
    )

    def clear_code_experiment(state):
        lab.cancel(state)
        return deepcopy(refresh(state))

    for control in (layer, polarity, goal, candidates, cases, budget, window, *common):
        control.input(
            clear_code_experiment,
            session,
            outputs,
            cancels=[event, reevaluation_event],
            queue=False,
        )
    stop.click(
        clear_code_experiment,
        session,
        outputs,
        cancels=[event, reevaluation_event],
        queue=False,
    )

    @friendly
    def export_code_experiment(state, measured):
        return export_report(state, measured), gr.update(open=True)

    export.click(
        export_code_experiment,
        [session, result],
        [files, downloads],
        queue=False,
        show_progress="hidden",
    )
    return outputs
