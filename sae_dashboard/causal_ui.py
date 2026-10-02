"""Optional single-feature laboratory with no dependency on the dashboard layout."""

import gradio as gr
import pandas as pd
from . import causal_lab, model_runtime as runtime
from .feature_explorer import neuronpedia_url


def build_causal_lab(
    session,
    common,
    settings,
    friendly,
    response_panel,
    comparison_prompt,
    comparison_image,
):
    gr.Markdown("## 7. Extra · Single-feature causal lab", elem_classes="section-title")
    gr.Markdown(
        "Measure one SAE feature with a dose sweep, ablation and an equal-norm random control."
    )
    with gr.Row():
        source = gr.Radio(
            ["Common profile", "Manual"], value="Common profile", label="Feature source"
        )
        refresh = gr.Button("Load features from common profile")
    candidates = gr.Dropdown(
        choices=[], label="Profile features · largest mean contrasts", interactive=True
    )
    with gr.Row():
        layer = gr.Dropdown(
            runtime.LAYERS, value=runtime.LAYERS[0], label="Feature layer"
        )
        feature = gr.Number(value=0, minimum=0, precision=0, label="Feature ID")
        coefficient = gr.Number(
            value=1, label="Manual native coefficient per dose", interactive=False
        )
    feature_info = gr.Markdown(
        "Choose a profile feature or switch to Manual. Layer steering strengths do not apply here."
    )
    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(
                label="Causal prompt",
                lines=5,
                placeholder="Enter the task whose response you want to study.",
            )
        with gr.Column():
            image = gr.Image(type="pil", label="Optional causal image", height=220)
    copy_query = gr.Button("Copy input from Base vs. steered", size="sm")
    with gr.Accordion("Shared assistant prefix (optional)", open=False):
        prefix = gr.Textbox(
            label="Exact assistant prefix",
            lines=3,
            info="Replayed for every condition. Only newly generated continuation is shown below.",
        )
    with gr.Accordion("Decision readout settings", open=True):
        with gr.Row():
            positive = gr.Textbox(value="YES", label="Positive label")
            negative = gr.Textbox(value="NO", label="Negative label")
            spaced = gr.Checkbox(value=False, label="Prepend a space to both labels")
            maximum = gr.Slider(1, 4, value=2, step=1, label="Maximum sweep dose (±)")
        decision_prompt = gr.Textbox(
            label="Separate decision prompt (optional)",
            lines=2,
            info="Blank uses the causal prompt above. The image and prefix remain shared.",
        )
    measure = gr.Button("Measure sweep + ablation + random control", variant="primary")
    sweep_status = gr.Markdown()
    curve = gr.LinePlot(
        x="dose",
        y="log_odds",
        color="intervention",
        sort="x",
        x_title="Signed dose × native coefficient",
        y_title="log P(positive) − log P(negative)",
        color_map={"SAE feature": "#0e7490", "Random control": "#9ca3af"},
        height=320,
    )
    with gr.Accordion("Exact measurements", open=False):
        measurements = gr.Dataframe(
            headers=[
                "Intervention",
                "Dose",
                "Preferred label",
                "Log odds",
                "P(positive | labels)",
                "P(labels)",
                "Residual change %",
                "Greedy next token",
            ],
            interactive=False,
        )
    with gr.Accordion("Input activation map", open=False):
        heatmap = gr.HTML()
    gr.Markdown("### Full response comparison")
    with gr.Row():
        dose = gr.Slider(
            -4, 4, value=0, step=0.05, label="Single-feature response dose"
        )
        schedule = gr.Radio(
            ["Every decoding step", "First continuation step only"],
            value="Every decoding step",
            label="Intervention schedule",
        )
    generate = gr.Button(
        "Generate base + feature + random responses", variant="primary"
    )
    report_status = gr.Markdown()
    with gr.Row():
        base = response_panel("Base · causal lab")
        feature_response = response_panel("Single feature")
        random_response = response_panel("Random control")
    with gr.Accordion("Method and interpretation", open=False):
        gr.Markdown("""The selected decoder direction is added to the residual stream without replacing its reconstruction error.
A sweep intervenes only at the **final input position**, then measures the next-token logits.
Ablation subtracts the query's current decoded feature contribution. The random direction has the same norm.

A profile-derived unit dose uses the 95th percentile of the feature activation magnitude,
capped at 5% of the profile's reference residual norm and signed toward B − A.
Manual mode uses the native coefficient entered above. This is separate from layer steering strengths.

The decision labels must each tokenize to one distinct token. Conditional label probability is not
absolute probability: inspect **P(labels)** and the unconstrained next token too.
Test inverse questions and different cases before assigning a semantic interpretation.
Selecting a profile feature does not imply independent causal validation.

Full responses share the top-level seed, temperature and token limit. The same seed generates the random control.
At dose 0 all three conditions have no intervention. A shared prefix is replayed, so its preservation is by construction.
Causal results follow the experiment-saving toggle. This lab does not modify the common profile or steering vectors.""")

    @friendly
    def load_candidates(state, *values):
        choices = causal_lab.profile_candidates(state, settings(values))
        selected_layer, f = map(int, choices[0][1].split(":"))
        return (
            gr.update(choices=choices, value=choices[0][1]),
            "Common profile",
            selected_layer,
            f,
            describe(selected_layer, f, "Common profile"),
        )

    refresh.click(
        load_candidates,
        [session, *common],
        [candidates, source, layer, feature, feature_info],
    )
    candidates.input(
        lambda key: tuple(map(int, key.split(":"))) if key else (runtime.LAYERS[0], 0),
        candidates,
        [layer, feature],
    )
    source.change(
        lambda selected: gr.update(interactive=selected == "Manual"),
        source,
        coefficient,
    )

    def describe(selected_layer, f, selected):
        if selected_layer is None or f is None or int(f) != f or f < 0:
            return "Choose a valid layer and feature ID."
        link = (
            f" · [Inspect in Neuronpedia]({neuronpedia_url(int(selected_layer), int(f))})"
            if runtime.MODEL_ID == "google/gemma-3-4b-it"
            and runtime.SAE_RELEASE == "gemma-scope-2-4b-it-res"
            else ""
        )
        return f"**Layer {selected_layer} · feature #{int(f)}** · {selected}{link}"

    for component in (layer, feature, source):
        component.change(describe, [layer, feature, source], feature_info)
    copy_query.click(
        lambda text, picture: (text, picture),
        [comparison_prompt, comparison_image],
        [query, image],
    )
    selection = [source, layer, feature, coefficient]

    @friendly
    def run_sweep(
        state,
        selected,
        selected_layer,
        f,
        scale,
        text,
        picture,
        shared_prefix,
        pos,
        neg,
        space,
        extent,
        decision,
        progress=gr.Progress(),
        *values,
    ):
        case = causal_lab.make_case(
            decision or text, picture, shared_prefix, pos, neg, space
        )
        result, tokens = causal_lab.sweep(
            state,
            settings(values),
            selected,
            selected_layer,
            f,
            scale,
            case,
            extent,
            progress,
        )
        plotted = pd.DataFrame(
            [
                {k: row[k] for k in ("dose", "intervention", "log_odds")}
                for row in result["measurements"]
            ]
        )
        rows = [
            [
                row["intervention"],
                row["dose"],
                row["choice"],
                row["log_odds"],
                row["p_positive_given_labels"],
                row["label_probability_mass"],
                100 * row["relative_residual_change"],
                row["top_token"],
            ]
            for row in [
                *result["measurements"],
                {
                    "intervention": "Feature ablation",
                    "dose": None,
                    **result["ablation"],
                },
            ]
        ]
        c = result["candidate"]
        message = (
            f"**Layer {c['layer']} · #{c['feature_id']}** · Native coefficient per dose: **{c['intervention_step']:+.6g}** · "
            f"Zero control: **{'PASS' if result['zero_control_passed'] else 'FAIL'}** · Ablation Δ log odds: **{result['ablation']['log_odds'] - result['base']['log_odds']:+.4f}**."
        )
        if decision:
            message += "\n\nThis curve uses the separate decision prompt; full responses use the causal prompt."
        return message, plotted, rows, tokens

    measure.click(
        run_sweep,
        [
            session,
            *selection,
            query,
            image,
            prefix,
            positive,
            negative,
            spaced,
            maximum,
            decision_prompt,
            *common,
        ],
        [sweep_status, curve, measurements, heatmap],
    )

    @friendly
    def run_responses(
        state,
        selected,
        selected_layer,
        f,
        scale,
        text,
        picture,
        shared_prefix,
        selected_dose,
        selected_schedule,
        *values,
    ):
        case = causal_lab.make_case(text, picture, shared_prefix)
        results = causal_lab.responses(
            state,
            settings(values),
            selected,
            selected_layer,
            f,
            scale,
            case,
            selected_dose,
            selected_schedule,
        )
        return (
            f"**Dose {selected_dose:+g}** · {selected_schedule} · Shared seed, temperature and token limit.",
            *[r["answer"] for r in results],
        )

    generate.click(
        run_responses,
        [session, *selection, query, image, prefix, dose, schedule, *common],
        [report_status, base, feature_response, random_response],
    )
    # Outputs are measurements of a specific input. Clear them whenever that input changes.
    result_outputs = [
        sweep_status,
        curve,
        measurements,
        heatmap,
        report_status,
        base,
        feature_response,
        random_response,
    ]
    for control in [
        *selection,
        query,
        image,
        prefix,
        positive,
        negative,
        spaced,
        maximum,
        decision_prompt,
        dose,
        schedule,
        *common,
    ]:
        control.change(lambda: ("", None, [], "", "", "", "", ""), None, result_outputs)
    return candidates, result_outputs
