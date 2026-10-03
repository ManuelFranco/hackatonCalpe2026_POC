"""Single-feature research workspace; inference lives outside the UI."""

import gradio as gr
import pandas as pd
import re
from research.feature_studies import VIEWS, summarize_ratings
from . import causal_lab, feature_study, model_runtime as runtime
from .manifests import load_manifest
from .artifact_store import record_event
from .feature_explorer import neuronpedia_url


def response_markdown(answer):
    """Close a truncated code fence for display without changing the stored answer."""
    opened = None
    for line in answer.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if match:
            fence, suffix = match.groups()
            if opened is None:
                opened = fence
            elif (
                fence[0] == opened[0]
                and len(fence) >= len(opened)
                and not suffix.strip()
            ):
                opened = None
    return answer + ("\n\n" + opened if opened else "")


def build_causal_lab(
    session,
    common,
    settings,
    friendly,
    response_panel,
    comparison_prompt,
    comparison_image,
):
    gr.Markdown("## 7. Extra · Feature research", elem_classes="section-title")
    gr.Markdown(
        "Study **where a feature activates** and **what it changes**. Use matched A/B inputs and independent evaluation cases."
    )
    with gr.Row():
        source = gr.Radio(
            ["Common profile", "Manual"], value="Common profile", label="Feature source"
        )
        refresh = gr.Button("Load profile candidates")
        reference = gr.Button("Study reference · L22 / #11749 / 262k")
    candidates = gr.Dropdown(
        choices=[],
        label="Candidates · ranked by absolute mean contrast",
        interactive=True,
    )
    with gr.Row():
        layer = gr.Dropdown(runtime.LAYERS, value=22, label="Layer")
        feature = gr.Number(value=11749, minimum=0, precision=0, label="Feature ID")
        coefficient = gr.Number(
            value=1, label="Manual coefficient per unit dose", interactive=False
        )
        expected = gr.Number(
            value=262144,
            minimum=0,
            precision=0,
            label="Expected dictionary size",
            info="262144 for the reference. 0 accepts the loaded dictionary.",
        )
    identity = gr.Markdown(
        "Reference: **security vulnerabilities and exploits**. Verify the loaded SAE before interpreting this feature."
    )
    inspect = gr.Button("Inspect feature and verify dictionary", size="sm")
    with gr.Accordion(
        "Neuronpedia · external feature evidence",
        open=False,
        elem_classes="optional-section",
    ):
        neuron = gr.HTML()

    gr.Markdown("### 1 · Separate code from explanatory text")
    gr.Markdown(
        "Compare mean activation on full inputs, code with its preamble, and text without code. A contrast that exists only in the explanations is not evidence of code understanding."
    )
    dataset = gr.State(())
    with gr.Row():
        manifest_file = gr.File(
            label="Optional research manifest · leaves the common profile intact",
            file_types=[".json"],
            type="filepath",
        )
        load_inputs = gr.Button("Load research manifest / current section 1 inputs")
    pair = gr.Dropdown(
        choices=[],
        label="Pair to inspect and use for response experiments",
        interactive=True,
    )
    with gr.Row():
        view = gr.Dropdown(
            list(VIEWS), value="Full input", label="Preview / response input view"
        )
        views = gr.CheckboxGroup(
            list(VIEWS), value=["Full input"], label="Activation study views"
        )
        limit = gr.Slider(
            1, 24, value=12, step=1, label="First N pairs · activation study"
        )
    instruction = gr.Textbox(
        label="Shared response instruction (optional)",
        lines=2,
        placeholder="Explain the code's behavior, security implications and any necessary fix.",
        info="Added to response prompts only. Activation measurements use the selected input view without this instruction.",
    )
    with gr.Accordion(
        "Exact response inputs · inspect before running",
        open=False,
        elem_classes="explanatory-section",
    ):
        with gr.Row():
            with gr.Column():
                preview_a = response_panel("A · prompt")
                image_a = gr.Image(
                    type="pil", label="A · image", height=180, interactive=False
                )
            with gr.Column():
                preview_b = response_panel("B · prompt")
                image_b = gr.Image(
                    type="pil", label="B · image", height=180, interactive=False
                )
    activate = gr.Button("Measure paired input activations", variant="primary")
    activation_status = gr.Markdown()
    plot = gr.LinePlot(
        x="pair_index",
        y="delta",
        color="view",
        x_title="Pair index",
        y_title="Mean activation B − A",
        height=280,
    )
    table = gr.Dataframe(
        headers=[
            "Dataset",
            "Pair",
            "View",
            "Mean A",
            "Mean B",
            "B − A",
            "Active A %",
            "Active B %",
            "Tokens A",
            "Tokens B",
        ],
        interactive=False,
    )

    gr.Markdown("### 2 · Test causal effects on complete responses")
    with gr.Row():
        case_source = gr.Radio(
            ["Selected A/B pair", "Custom input"],
            value="Selected A/B pair",
            label="Response inputs",
        )
        dose = gr.Slider(
            0,
            4,
            value=0,
            step=0.05,
            label="Symmetric dose magnitude · ±",
            info="0 is the no-intervention check; try 0.5, then 1.",
        )
    with gr.Accordion(
        "Custom input / token activation map",
        open=False,
        elem_classes="optional-section",
    ):
        query = gr.Textbox(label="Custom prompt", lines=5)
        image = gr.Image(type="pil", label="Optional custom image", height=200)
        copy_query = gr.Button("Copy input from Base vs. steered", size="sm")
        map_button = gr.Button("Inspect custom input token activations", size="sm")
        heatmap = gr.HTML()
    with gr.Accordion(
        "Intervention controls", open=False, elem_classes="optional-section"
    ):
        schedule = gr.Radio(
            ["Every decoding step", "First continuation step only"],
            value="Every decoding step",
            label="Intervention schedule",
        )
        ablation = gr.Checkbox(
            value=False,
            label="Include dynamic feature ablation (active even when dose is 0)",
        )
        random_count = gr.Slider(
            1, 3, value=1, step=1, label="Independent equal-norm random directions"
        )
    gr.Markdown(
        "Each input produces **base, +dose, −dose and random-control** responses; optional ablation removes the current decoded contribution. A/B plus one random direction costs **8 generations**; ablation adds 2. Shared generation settings apply."
    )
    generate = gr.Button("Run controlled response study", variant="primary")
    response_status = gr.Markdown()
    diagnostics = gr.Dataframe(
        headers=[
            "Case",
            "Condition",
            "Dose",
            "Random seed",
            "Intervened steps",
            "Largest residual change %",
        ],
        interactive=False,
    )
    responses = gr.Markdown(sanitize_html=True, line_breaks=True)
    with gr.Accordion(
        "Score response quality · manual paired means",
        open=False,
        elem_classes="optional-section",
    ):
        gr.Markdown(
            "Score mechanism, consequence and recommendation **0–2**. Leave uncertain rows blank; they are excluded, never counted as zero. Keep the case and condition columns unchanged. These are human ratings, not automated benchmark scores."
        )
        rating_context = gr.State(None)
        ratings = gr.Dataframe(
            headers=["Case", "Condition", "Mechanism", "Consequence", "Recommendation"],
            datatype=["str", "str", "number", "number", "number"],
            type="array",
            interactive=True,
        )
        summarize = gr.Button("Compute mean quality and paired change")
        rating_summary = gr.Dataframe(
            headers=[
                "Condition",
                "Rated cases",
                "Mean quality / 2",
                "Pairs with rated base",
                "Mean change vs. base",
            ],
            interactive=False,
        )
    with gr.Accordion(
        "Interpretation and scoring protocol",
        open=False,
        elem_classes="explanatory-section",
    ):
        gr.Markdown("""**Hypothesis:** this feature changes grounded security analysis, rather than merely increasing security vocabulary.

1. Use all-token mean capture. Compare the three input views on calibration pairs. Different token counts and context mean these are diagnostic controls, not a causal decomposition.
2. Load an independent validation manifest here, retaining the training profile. For code review, use **Code + preamble** and a shared instruction; do not give the model a reference answer. Include safe code that mentions security terms and vulnerable code without suggestive names.
3. Start at dose **0**, ablation off. All additive conditions should reproduce base. Then try **0.5** and **1**, both signs, with 3 random directions. Repeat on held-out pairs and more seeds. In profile mode +dose follows mean B − A; manual mode follows the entered coefficient.
4. Score each response manually: **mechanism**, **consequence**, **recommendation**, each 0–2 (wrong/missing, partial, correct and grounded). Average the three scores, then average equally over cases. Compare paired changes against base and random controls. Track false vulnerability claims on safe code separately; more security words alone earns no credit. Freeze the rubric before evaluating the test split.

**Ablation** subtracts the current SAE activation × decoder direction at the final position of each selected forward. It preserves the reconstruction error, but does not guarantee a zero re-encoded activation or remove the feature at every prompt position. Negative additive dose is not ablation. Ablation has its own magnitude; random controls match the additive intervention, not ablation.

The unit dose is the calibration activation p95, capped using 5% of the calibration residual norm. This is not a per-step safety bound: inspect the measured residual changes. Generated continuations can diverge, so equal seeds and equal-norm controls do not imply identical trajectories. One feature and a small sample do not establish generalization.

Activation studies always report the arithmetic **mean**, using the shared token scope; the profile's aggregation still determines the calibrated coefficient. No Boolean labels or LLM judge are required. Neuronpedia is external evidence, not ground truth. Model hooks are removed after each run; this tab does not modify steering vectors. Results stay in session memory unless experiment saving is enabled at the top.""")

    selection = [source, layer, feature, coefficient]

    def selected_pair(manifests, key):
        pairs = [(m, p) for m in manifests for p in m.pairs]
        if key is None or not 0 <= int(key) < len(pairs):
            raise ValueError("Load research inputs and select a pair first.")
        return pairs[int(key)]

    @friendly
    def load(state, uploaded):
        with state.lock:
            manifests = (
                (load_manifest(uploaded),) if uploaded else tuple(state.manifests)
            )
        if not manifests:
            raise ValueError(
                "Load a manifest in section 1 or choose a research manifest here."
            )
        choices = [
            (f"{m.name} / {p.id}", str(i))
            for i, (m, p) in enumerate((m, p) for m in manifests for p in m.pairs)
        ]
        return manifests, gr.update(choices=choices, value="0")

    load_inputs.click(load, [session, manifest_file], [dataset, pair])

    @friendly
    def preview(manifests, key, selected_view, task):
        if not manifests or key is None:
            return "", None, "", None
        _, p = selected_pair(manifests, key)
        cases = feature_study.cases_for_pair(p, selected_view, task)
        return (
            cases[0][1]["prompt"],
            cases[0][1]["image"],
            cases[1][1]["prompt"],
            cases[1][1]["image"],
        )

    for control in (dataset, pair, view, instruction):
        control.change(
            preview,
            [dataset, pair, view, instruction],
            [preview_a, image_a, preview_b, image_b],
        )

    @friendly
    def load_candidates(state, *values):
        choices = causal_lab.profile_candidates(state, settings(values))
        selected_layer, f = map(int, choices[0][1].split(":"))
        return (
            gr.update(choices=choices, value=choices[0][1]),
            "Common profile",
            selected_layer,
            f,
            0,
        )

    refresh.click(
        load_candidates,
        [session, *common],
        [candidates, source, layer, feature, expected],
    )
    candidates.input(
        lambda key: (*map(int, key.split(":")), 0) if key else (22, 0, 0),
        candidates,
        [layer, feature, expected],
    )
    reference.click(lambda: (22, 11749, 262144), None, [layer, feature, expected])
    source.change(lambda s: gr.update(interactive=s == "Manual"), source, coefficient)

    @friendly
    def inspect_feature(state, selected, selected_layer, f, scale, width, *values):
        with state.lock, runtime.MODEL_LOCK:
            c = causal_lab.select_candidate(
                state, settings(values), selected, selected_layer, f, scale
            )
            feature_study.check_dictionary(c, width)
        embed = ""
        if (
            c["model_id"] == "google/gemma-3-4b-it"
            and c["sae_release"] == "gemma-scope-2-4b-it-res"
            and c["dictionary_size"] in (16384, 262144)
        ):
            url = neuronpedia_url(int(selected_layer), int(f), c["dictionary_size"])
            embed = f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open in Neuronpedia</a><iframe src="{url}" title="Neuronpedia feature" loading="lazy" style="width:100%;height:520px;border:0"></iframe>'
        return feature_study.candidate_info(c), embed

    inspect.click(
        inspect_feature, [session, *selection, expected, *common], [identity, neuron]
    )

    @friendly
    def run_activation(
        state,
        selected,
        selected_layer,
        f,
        scale,
        width,
        manifests,
        chosen_views,
        count,
        progress=gr.Progress(),
        *values,
    ):
        c, rows = feature_study.activation_study(
            state,
            settings(values),
            (selected, selected_layer, f, scale),
            width,
            manifests,
            chosen_views,
            count,
            progress,
        )
        summary = []
        points = []
        for v in chosen_views:
            subset = [r for r in rows if r["view"] == v]
            delta = sum(r["delta"] for r in subset) / len(subset)
            same = sum(r["delta"] * delta > 0 for r in subset)
            summary.append(
                f"**{v}**: mean paired Δ **{delta:+.4g}** · same sign **{same}/{len(subset)}**"
            )
            points.extend(
                {"pair_index": i + 1, "delta": r["delta"], "view": v}
                for i, r in enumerate(subset)
            )
        values = [
            [
                r["dataset"],
                r["pair"],
                r["view"],
                r["mean_a"],
                r["mean_b"],
                r["delta"],
                100 * r["active_a"],
                100 * r["active_b"],
                r["tokens_a"],
                r["tokens_b"],
            ]
            for r in rows
        ]
        return (
            feature_study.candidate_info(c),
            "\n\n".join(summary),
            pd.DataFrame(points),
            values,
        )

    activate.click(
        run_activation,
        [session, *selection, expected, dataset, views, limit, *common],
        [identity, activation_status, plot, table],
    )
    copy_query.click(
        lambda text, picture: (text, picture),
        [comparison_prompt, comparison_image],
        [query, image],
    )

    @friendly
    def map_input(
        state, selected, selected_layer, f, scale, width, text, picture, *values
    ):
        with state.lock, runtime.MODEL_LOCK:
            c = causal_lab.select_candidate(
                state, settings(values), selected, selected_layer, f, scale
            )
            feature_study.check_dictionary(c, width)
            return causal_lab.token_map(causal_lab.make_case(text, picture, ""), c)

    map_button.click(
        map_input, [session, *selection, expected, query, image, *common], heatmap
    )

    @friendly
    def run_responses(
        state,
        selected,
        selected_layer,
        f,
        scale,
        width,
        origin,
        manifests,
        key,
        selected_view,
        task,
        text,
        picture,
        magnitude,
        selected_schedule,
        remove,
        controls,
        progress=gr.Progress(),
        *values,
    ):
        if origin == "Selected A/B pair":
            m, p = selected_pair(manifests, key)
            cases = [
                (f"{m.name} / {p.id} / {side}", case)
                for side, case in feature_study.cases_for_pair(p, selected_view, task)
            ]
        else:
            cases = [("Custom input", causal_lab.make_case(text, picture, ""))]
        c, results, zero = feature_study.response_study(
            state,
            settings(values),
            (selected, selected_layer, f, scale),
            width,
            cases,
            magnitude,
            selected_schedule,
            remove,
            controls,
            progress,
        )
        status = (
            f"**{len(results)} responses** · dose ±{magnitude:g} · {selected_schedule}."
        )
        if zero is not None:
            status += f" Zero-dose reproducibility: **{'PASS' if zero else 'FAIL — inspect runtime nondeterminism before interpreting effects'}**."
        if selected == "Manual":
            status += " In Manual mode, Toward B/A labels mean +/− the entered coefficient; no semantic direction is established."
        blocks = []
        for name, _ in cases:
            blocks.append(f"## {name}")
            for result in results:
                if result["case"] == name:
                    blocks.append(
                        f"### {result['condition']}\n\n{response_markdown(result['answer'])}"
                    )
        rows = [
            [
                r["case"],
                r["condition"],
                None if r["ablation"] else r["dose"],
                r["direction_seed"],
                r["intervention_count"],
                100 * r["maximum_relative_residual_change"],
            ]
            for r in results
        ]
        keys = [(r["case"], r["condition"]) for r in results]
        context = {
            "study_id": results[0]["study_id"],
            "candidate": c,
            "dose": magnitude,
            "schedule": selected_schedule,
            "keys": keys,
        }
        return (
            feature_study.candidate_info(c),
            status,
            rows,
            "\n\n---\n\n".join(blocks),
            [[*key, None, None, None] for key in keys],
            context,
            [],
        )

    generate.click(
        run_responses,
        [
            session,
            *selection,
            expected,
            case_source,
            dataset,
            pair,
            view,
            instruction,
            query,
            image,
            dose,
            schedule,
            ablation,
            random_count,
            *common,
        ],
        [
            identity,
            response_status,
            diagnostics,
            responses,
            ratings,
            rating_context,
            rating_summary,
        ],
    )

    @friendly
    def score(state, context, rows):
        if not context:
            raise ValueError("Generate responses before rating them.")
        summary = summarize_ratings(rows, context["keys"])
        with state.lock:
            record_event(
                state,
                "feature_manual_ratings",
                {**context, "ratings": rows, "summary": summary},
            )
        return [
            [
                r["condition"],
                r["rated_cases"],
                r["mean_quality"],
                r["paired_cases"],
                r["mean_delta_vs_base"],
            ]
            for r in summary
        ]

    summarize.click(score, [session, rating_context, ratings], rating_summary)
    ratings.input(lambda: [], None, rating_summary)
    result_outputs = [
        identity,
        neuron,
        activation_status,
        plot,
        table,
        heatmap,
        response_status,
        diagnostics,
        responses,
        ratings,
        rating_context,
        rating_summary,
    ]
    clear_results = ["", "", "", None, [], "", "", [], "", [], None, []]
    for control in [
        *selection,
        expected,
        dataset,
        pair,
        view,
        views,
        limit,
        instruction,
        case_source,
        query,
        image,
        dose,
        schedule,
        ablation,
        random_count,
        *common,
    ]:
        control.change(lambda: tuple(clear_results), None, result_outputs)
    # Profile changes also discard the research snapshot and its previews.
    outputs = [*result_outputs, dataset, pair, preview_a, image_a, preview_b, image_b]
    cleared = [
        *clear_results,
        (),
        gr.update(choices=[], value=None),
        "",
        None,
        "",
        None,
    ]
    return candidates, outputs, cleared
