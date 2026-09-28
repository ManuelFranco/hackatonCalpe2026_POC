"""Gradio controls for inspecting single-feature causal effects, with controls."""

import html
import json
import math
from sae_dashboard import PROJECT_ROOT

import gradio as gr
import pandas as pd
import torch

from sae_dashboard.causal_examples import CAUSAL_CASES, image_cases
from sae_dashboard.causal_lab import generate_feature_report, label_ids, new_report_path, probe, run_discovery, prepare_case_inputs
from sae_dashboard.calibration_bridge import experiment_from_calibration
from sae_dashboard.cyber_integrity import score_known_fixture
from sae_dashboard.security_reports import SECURITY_DECISION_PROMPT, SECURITY_REPORT_PROMPT, security_report_cases


ROOT = PROJECT_ROOT
REPORTS = {
    "C disclosure · counterbalanced": ("code", "counterbalanced"),
    "C answer-word control": ("code", "decision_control"),
    "Website screenshots · counterbalanced": ("images", "counterbalanced"),
}


def available_cases(report):
    # Query examples may cross modalities; the calibrated candidate is unchanged.
    cases = {c["id"]: c for c in [*report["cases"], *security_report_cases(), *CAUSAL_CASES, *image_cases()]}
    return list(cases.values())


def report_summary(report):
    if report["mode"] == "manual":
        return "**Manual feature selection.** No calibration or independent validation is implied. The dose scale is supplied by you."
    if report["mode"] == "calibration":
        source, selected = report["source_profile"], report["selected"]
        return (
            f"**From your section-1 calibration: `{source['run_id']}`.**\n\n"
            f"Selected **layer {selected['layer']} · feature {selected['feature_id']}** from "
            f"**{source['num_pairs']} A/B pairs**. Token scope: `{source['feature_token_scope']}`; "
            f"aggregation: `{source['aggregation']}`. Calibration sign agreement: "
            f"**{selected['train_agreement']:.0%}**.\n\n"
            "**No independent validation has been run for this selection.** "
            "The individual dose is derived from this calibration's feature scores; section-2 layer alphas are not applied. "
            "Check the two answer labels, then measure the sweep below."
        )
    result = report["held_out"]
    random = report["random_direction_control"]
    random_mean = sum(r["symmetric_effect"] for r in random) / len(random)
    selected = report["selected"]
    tests = [c for c in report["cases"] if c["split"] == "test"]
    baseline_correct = sum(report["baseline"][c["id"]]["correct"] for c in tests)
    lookup = {c["id"]: c for c in tests}
    polarity_effects = {}
    for polarity in ("YES", "NO"):
        values = [r["symmetric_effect"] for r in result["cases"]
                  if lookup[r["case_id"]]["positive_label"] == polarity]
        polarity_effects[polarity] = sum(values) / len(values) if values else 0
    strong = (report.get("validation_gate_passed", False)
              and result["positive_fraction"] >= .75 and result["mean_effect"] > abs(random_mean))
    verdict = ("A directional effect transferred to this small reserved suite."
               if strong else "The candidate has not established a consistent concept-level effect.")
    if report["mode"] == "decision_control":
        verdict = "Answer-word control: selection used direct questions. Check inverse questions before interpreting this as disclosure."
    return (
        f"**{verdict}**\n\n"
        f"Selected: layer **{selected['layer']}**, feature **{selected['feature_id']}**. "
        f"Training sign agreement: **{selected['train_agreement']:.0%}**.\n\n"
        f"Reserved directional consistency: **{result['positive_fraction']:.0%}**; "
        f"mean symmetric log-odds effect: **{result['mean_effect']:+.3f}**; "
        f"equal-norm random control: **{random_mean:+.3f}**.\n\n"
        f"Direct questions: **{polarity_effects['YES']:+.3f}**; "
        f"inverse questions: **{polarity_effects['NO']:+.3f}**. "
        "Opposite signs across polarities indicate an answer-word bias rather than a consistent concept direction.\n\n"
        f"Baseline two-label choices: **{baseline_correct}/{len(tests)}** correct. "
        "These include both question polarities and measure a constrained readout, not the old free-text benchmark. "
        "A positive shift means a greater preference for the disclosure/injection label, regardless of whether that label is correct."
    )


def feature_key(candidate):
    return f"{candidate['layer']}:{candidate['feature_id']}"


def candidate_for(report, key):
    if not report:
        raise gr.Error("Load an experiment or run discovery first.")
    for candidate in report["candidates"]:
        if feature_key(candidate) == key:
            return candidate
    raise gr.Error("Select a feature from the current experiment.")


def preserve_recorded_prefix(report, case):
    """Use recorded token boundaries when the displayed input still matches."""
    for recorded in report.get("cases", []):
        if (recorded.get("prompt") == case["prompt"]
                and recorded.get("assistant_prefix") == case.get("assistant_prefix")
                and recorded.get("assistant_prefix_ids") is not None):
            case["assistant_prefix_ids"] = recorded["assistant_prefix_ids"]
            break
    return case


def measurement_row(label, dose, measurement):
    """Readable table values; full precision remains in the downloadable JSON."""
    return [label, dose, measurement["choice"], f"{measurement['log_odds']:+.3f}",
            f"{measurement['p_positive_given_labels']:.4g}",
            f"{measurement['label_probability_mass']:.4g}",
            f"{100 * measurement['relative_residual_change']:.3f}", measurement["top_token"]]


def feature_description(app, report, key):
    c = candidate_for(report, key)
    if report["mode"] == "manual":
        return (f"**Layer {c['layer']} · feature {c['feature_id']}**. "
                f"One dose = `{c['intervention_step']:.12g} × W_dec[{c['feature_id']}]`. "
                "A negative dose reverses the direction. The measured residual change is shown below.")
    origin = "Calibration" if report["mode"] == "calibration" else "Training"
    return (
        f"**Layer {c['layer']} · feature {c['feature_id']}** · "
        f"[Inspect in Neuronpedia]({app.neuronpedia_url(c['layer'], c['feature_id'])})\n\n"
        f"{origin} mean A: `{c['mean_A']:.3f}` · mean B: `{c['mean_B']:.3f}` · "
        f"mean B−A: `{c['natural_delta']:+.3f}`.\n\n"
        f"One dose adds `{c['intervention_step']:+.3f} × W_dec[{c['feature_id']}]` "
        "at the final input position. This uses the 95th percentile of the "
        f"{origin.lower()} feature scores, capped at 5% of the corresponding residual norm. "
        "It differs from the common-profile alpha sliders. "
        "Negative doses reverse the addition. Ablation removes the query's decoded feature contribution."
    )


def token_map(app, case, candidate):
    inputs, ids, _ = prepare_case_inputs(app, case)
    residuals = app.capture_prompt_residuals(inputs)[candidate["layer"]]
    values = app.encode_sae_chunked(app.saes[candidate["layer"]], residuals)[:, candidate["feature_id"]]
    maximum = max(float(values.max()), 1e-8)
    mask = app.get_image_mask(ids)
    spans, visual_count = [], 0
    for i, token in enumerate(ids.tolist()):
        if bool(mask[i]):
            visual_count += 1
            continue
        token_text = app.processor.tokenizer.decode([token], skip_special_tokens=False)
        opacity = .08 + .72 * float(values[i]) / maximum
        spans.append(f'<span title="position {i}; activation {float(values[i]):.4f}" '
                     f'style="background:rgba(22,163,174,{opacity:.3f});padding:2px;border-radius:3px">'
                     f'{html.escape(token_text)}</span>')
    note = f"{visual_count} visual-token positions omitted from the text display. " if visual_count else ""
    return ('<div style="padding:16px;border:1px solid #aaa;border-radius:10px">'
            f'<p>{note}Darker = larger activation. Hover for exact values. '
            'This map describes input activations; intervention targets the final input position only.</p>'
            '<div style="white-space:pre-wrap;line-height:2;font-family:monospace">'
            + ''.join(spans) + '</div></div>')


def build_causal_lab(app, profile_state=None, source_image=None, source_prompt=None, feature_tables=()):
    gr.Markdown("## 3. Single-feature causal lab", elem_id="causal-lab")
    gr.Markdown(
        "Inspect **one feature**, measure the next-token decision, generate full reports, and compare against a random direction. "
        "The counterbalanced suites reverse YES/NO questions to test whether a feature tracks the concept "
        "or the answer word. You can also open an included feature from section 1 to measure it "
        "with your own calibration. All code secrets are fictional. Image cases test visible prompt injection."
    )
    state = gr.State(None)
    imported_reports = gr.State({})
    with gr.Row():
        suite = gr.Dropdown(list(REPORTS), value="C answer-word control", label="Experiment")
        load_button = gr.Button("Load selected experiment", variant="primary")
        discover_button = gr.Button("Run fresh discovery on this server")
    with gr.Accordion("Manual feature selection · no saved experiment required", open=False):
        with gr.Row():
            manual_layer = gr.Dropdown(app.LAYERS, value=app.LAYERS[0], label="Manual layer")
            manual_feature = gr.Number(value=0, precision=0, label="Manual feature ID")
            manual_step = gr.Number(value=1, label="Native decoder coefficient per dose")
        manual_button = gr.Button("Use manual feature")
    status = gr.Markdown("Load an experiment, open a section-1 feature, or select a feature manually.")
    source_download = gr.File(label="Download source experiment")
    with gr.Accordion("Feature selection details", open=False):
        ranking = gr.Dataframe(headers=["Layer", "Feature", "Train agreement", "Validation consistency", "Validation effect"], interactive=False)
    with gr.Row():
        feature = gr.Dropdown(choices=[], label="Single SAE feature")
        case_selector = gr.Dropdown(choices=[], label="Input example · calibration or prepared case")
    feature_info = gr.Markdown()
    example_reference = gr.Markdown()
    with gr.Row():
        query_image = gr.Image(type="filepath", label="Image · optional", height=240)
        query_prompt = gr.Textbox(lines=7, label="Prompt · editable")
    assistant_prefix = gr.Textbox(lines=4, label="Shared assistant prefix · optional",
                                  info="Continuation already present in all three answers. Blank starts a new answer. "
                                       "Its preservation is by construction, not evidence of semantic isolation. Keep trailing whitespace exact.",
                                  max_lines=12)
    with gr.Row():
        schedule = gr.Radio(["Every decoding step", "First continuation step only"],
                            value="Every decoding step", label="Intervention schedule")
        random_seed = gr.Number(value=17, precision=0, label="Random control seed")
        label_space = gr.Checkbox(value=False, label="Prepend one space to both decision labels")
    decision_prompt = gr.Textbox(lines=3, label="Decision probe prompt · only used by the curve",
                                info="Optional separate task for next-token scoring. The image report preset uses a short Attack/Clean probe. "
                                     "The full reports below use the main prompt and may reach a different verdict. Blank = use the main prompt.")
    with gr.Row():
        positive = gr.Textbox(value="YES", label="Positive label",
                              info="Choose the answer word whose preference the graph measures.")
        negative = gr.Textbox(value="NO", label="Negative label",
                              info="Prepared security cases map these labels to disclosure/injection and its absence.")
        extent = gr.Slider(1, 4, value=2, step=1, label="Maximum dose in each direction")
    run_button = gr.Button("Measure feature sweep + ablation + random control", variant="primary")
    curve = gr.LinePlot(x="dose", y="log_odds", color="intervention", sort="x",
                        x_title="Signed dose × selected decoder coefficient",
                        y_title="log P(positive label) − log P(negative label)",
                        color_map={"SAE feature": "#0e7490", "Random control": "#a3a3a3"},
                        height=330, label="Decision response · positive means preference for the positive label")
    results = gr.Dataframe(headers=["Intervention", "Dose", "Label", "Log odds", "P(positive | two labels)",
                                    "P(two labels)", "Residual change %", "Greedy next token"], interactive=False)
    sweep_status = gr.Markdown()
    with gr.Accordion("Where the selected feature activates in the input", open=True):
        heatmap = gr.HTML()
    download = gr.File(label="Download exact measurements and prompt")

    gr.Markdown("### Full response comparison for this feature")
    gr.Markdown(
        "Generate the complete answer to the same image and prompt. No answer label is forced. "
        "The schedule controls whether the direction is added once before the continuation, or at every decoding step. "
        "The curve measures the first token after the shared prefix. "
        "Review the evidence and explanation as well as the verdict."
    )
    with gr.Row():
        report_dose = gr.Slider(-4, 4, value=1, step=.05, label="Single-feature report dose")
        report_tokens = gr.Slider(32, 1024, value=320, step=1, label="Full report max new tokens")
    report_button = gr.Button("Generate BASE + FEATURE + RANDOM reports", variant="primary")
    with gr.Row():
        with gr.Column():
            report_base = app.response_markdown("BASE · full report", 220)
        with gr.Column():
            report_feature = app.response_markdown("SINGLE FEATURE · full report", 220)
        with gr.Column():
            report_random = app.response_markdown("RANDOM CONTROL · full report", 220)
    report_status = gr.Markdown()
    report_download = gr.File(label="Download full response comparison")

    outputs = [state, feature, ranking, case_selector, status, curve, results, sweep_status, heatmap, download,
               feature_info, query_image, query_prompt, positive, negative, example_reference, source_download, assistant_prefix, schedule]

    def present(report):
        validations = sorted(report["validation"], key=lambda r: (r["positive_fraction"], r["mean_effect"]), reverse=True)
        options = [(f"Layer {v['candidate']['layer']} · #{v['candidate']['feature_id']} · validation effect {v['mean_effect']:+.3f}", feature_key(v["candidate"])) for v in validations]
        rows = [[v["candidate"]["layer"], v["candidate"]["feature_id"], v["candidate"]["train_agreement"],
                 v["positive_fraction"], v["mean_effect"]] for v in validations]
        headers = ["Layer", "Feature", "Train agreement", "Validation consistency", "Validation effect"]
        if report["mode"] == "calibration":
            options = [(f"Layer {c['layer']} · #{c['feature_id']} · calibration B−A {c['natural_delta']:+.3f}",
                        feature_key(c)) for c in report["candidates"]]
            rows = [[c["layer"], c["feature_id"], c["train_agreement"], c["natural_delta"], "Not run"]
                    for c in report["candidates"]]
            headers = ["Layer", "Feature", "Calibration sign agreement", "Calibration B−A", "Independent validation"]
        cases = available_cases(report)
        default = report.get("default_case") or next(c["id"] for c in report["cases"] if c["split"] == "test")
        return (report, gr.update(choices=options, value=feature_key(report["selected"])),
                gr.update(value=rows, headers=headers),
                gr.update(choices=[(f"{c['id']} · {c['split']}", c["id"]) for c in cases], value=default),
                report_summary(report), None, [], "", "", None,
                feature_description(app, report, feature_key(report["selected"])),
                *load_causal_case(report, default)[:5], report.get("artifact_path"),
                load_causal_case(report, default)[5], report.get("intervention_schedule", "Every decoding step"))

    def load_causal_report(choice, imported):
        if choice in imported:
            return present(imported[choice])
        name, mode = REPORTS[choice]
        path = ROOT / f"experiments/causal_{name}_{mode}_20260928.json"
        if not path.exists():
            raise gr.Error("No recorded report for this suite. Run fresh discovery first.")
        report = json.loads(path.read_text())
        if report.get("format_version") != 2:
            raise gr.Error("This report uses an older protocol. Run fresh discovery.")
        report["artifact_path"] = str(path)
        return present(report)

    def run_causal_discovery(choice, progress=gr.Progress()):
        if choice not in REPORTS:
            raise gr.Error("Select a prepared suite to run fresh discovery, or measure your selected calibration feature below.")
        name, mode = REPORTS[choice]
        with app.MODEL_LOCK:
            app.ensure_models_loaded()
            path = new_report_path(app)
            report = run_discovery(app, CAUSAL_CASES if name == "code" else image_cases(), path,
                                   progress=lambda message: progress(0, desc=message), mode=mode)
        report["artifact_path"] = str(path)
        return present(report)

    def load_causal_case(report, case_id):
        if not report or not case_id:
            return None, "", "YES", "NO", "", ""
        case = next(c for c in available_cases(report) if c["id"] == case_id)
        path = case.get("image")
        reference = (f"Reference for the **unchanged** example `{case_id}`: **{case['expected']}**. "
                     "The reference is not sent to the model." if "expected" in case else
                     "Input copied from your calibration or section 2. No reference answer has been assigned; check the labels before measuring.")
        return (str(ROOT / path) if path else None, case["prompt"], case["positive_label"], case["negative_label"], reference, case.get("assistant_prefix", ""))

    def transfer_calibration_feature(session, image, prompt, imported, evt: gr.EventData):
        key = f"{evt.layer}:{evt.feature_id}"
        with app.MODEL_LOCK:
            app.ensure_models_loaded()
            try:
                report = experiment_from_calibration(app, session, key, image, prompt, evt.calibration_id)
            except (ValueError, OSError, KeyError) as error:
                raise gr.Error(str(error)) from error
        source = report["source_profile"]
        name = f"Calibration {source['run_id']} · L{report['selected']['layer']} #{report['selected']['feature_id']}"
        updated = {**imported, name: report}
        return (*present(report), gr.update(choices=[*REPORTS, *updated], value=name), updated)

    def inspect_causal_feature(report, key):
        return feature_description(app, report, key) if report and key else ""

    def run_causal_sweep(report, key, image, prompt, pos, neg, maximum, decision_task="", prefix="", seed=17, spaced=False, progress=gr.Progress()):
        candidate = candidate_for(report, key)
        report_prompt = prompt
        prompt = decision_task.strip() or prompt
        prompt = app.validate_condition_inputs("Causal query", image, prompt)
        case = {"prompt": prompt, "image": image, "assistant_prefix": prefix,
                "positive_label": (" " if spaced else "") + pos.strip(), "negative_label": (" " if spaced else "") + neg.strip()}
        case = preserve_recorded_prefix(report, case)
        rows, plotted, raw = [], [], []
        with app.MODEL_LOCK:
            app.ensure_models_loaded()
            recorded_commit = report["runtime"].get("model_commit")
            if recorded_commit != app.runtime_metadata().get("model_commit"):
                raise gr.Error("The recorded experiment used a different model revision. Run fresh discovery.")
            if (report.get("model_id") != app.MODEL_ID or report.get("sae_release") != app.SAE_RELEASE
                    or report.get("sae_ids") != {str(k): v for k, v in app.SAE_IDS.items()}):
                raise gr.Error("Model/SAE identity differs from the experiment. Run fresh discovery.")
            try:
                label_ids(app, case["positive_label"], case["negative_label"])
            except ValueError as error:
                raise gr.Error(str(error)) from error
            decoder = app.saes[candidate["layer"]].W_dec[candidate["feature_id"]].detach().float().cpu()
            random = torch.randn(decoder.shape, generator=torch.Generator().manual_seed(int(seed)))
            random *= decoder.norm() / random.norm()
            base, _ = probe(app, case, {**candidate, "amount": 0})
            doses = sorted(set([-float(maximum), -1., -.5, 0., .5, 1., float(maximum)]))
            for label, direction in (("SAE feature", None), ("Random control", random)):
                for dose in doses:
                    progress(0, desc=f"{label}: dose {dose:+g}")
                    measurement, _ = probe(app, case, {**candidate, "amount": dose * candidate["intervention_step"]}, direction=direction)
                    raw.append({"intervention": label, "dose": dose, **measurement})
                    plotted.append({"intervention": label, "dose": dose, "log_odds": measurement["log_odds"]})
                    rows.append(measurement_row(label, dose, measurement))
            ablation, _ = probe(app, case, {**candidate, "amount": -base["activation_before"]})
            rows.append(measurement_row("Feature ablation", None, ablation))
            tokens = token_map(app, case, candidate)
            zero_ok = all(r["log_odds"] == base["log_odds"] for r in raw if r["dose"] == 0)
            saved = {"runtime": app.runtime_metadata(), "candidate": candidate, "case": case,
                     "random_seed": int(seed), "full_report_prompt": report_prompt, "separate_decision_probe": bool(decision_task.strip()),
                     "source_profile": report.get("source_profile"),
                     "source_experiment": report.get("artifact_path"), "source_mode": report["mode"],
                     "model_id": app.MODEL_ID, "sae_release": app.SAE_RELEASE,
                     "sae_ids": {str(k): v for k, v in app.SAE_IDS.items()},
                     "image_sha256": app.sha256_file(image), "base": base, "sweep": raw,
                     "ablation": ablation, "zero_control_passed": zero_ok,
                     "method": "h_last += dose * intervention_step * W_dec[j]; preserve reconstruction error; no generation; random control has equal norm."}
            path = new_report_path(app)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved, indent=2) + "\n")
        message = (f"Measured **layer {candidate['layer']} · feature {candidate['feature_id']}**. "
                   f"**Zero control: {'PASS' if zero_ok else 'FAIL'}**. "
                   f"Feature activation before intervention: **{base['activation_before']:.3f}**. "
                   f"Ablation changes log odds by **{ablation['log_odds'] - base['log_odds']:+.3f}**.\n\n"
                   "The probability column is normalized over the two chosen labels. Check their total probability mass "
                   "and the unconstrained greedy next token. A large curve shift shows an intervention effect on this query; "
                   "use inverse questions and other examples to test its meaning. "
                   + ("**This curve uses the separate decision probe, not the full report task.**" if decision_task.strip() else ""))
        return pd.DataFrame(plotted), rows, message, tokens, str(path)

    def generate_causal_reports(report, key, image, prompt, dose, max_tokens, prefix="", intervention_schedule="Every decoding step", seed=17, progress=gr.Progress()):
        candidate = candidate_for(report, key)
        prompt = app.validate_condition_inputs("Full report", image, prompt)
        case = preserve_recorded_prefix(report, {"image": image, "prompt": prompt, "assistant_prefix": prefix})
        with app.MODEL_LOCK:
            app.ensure_models_loaded()
            if (report["runtime"].get("model_commit") != app.runtime_metadata().get("model_commit")
                    or report.get("model_id") != app.MODEL_ID
                    or report.get("sae_release") != app.SAE_RELEASE
                    or report.get("sae_ids") != {str(k): v for k, v in app.SAE_IDS.items()}):
                raise gr.Error("Model/SAE identity changed. Build or load a matching experiment.")
            decoder = app.saes[candidate["layer"]].W_dec[candidate["feature_id"]].detach().float().cpu()
            random = torch.randn(decoder.shape, generator=torch.Generator().manual_seed(int(seed)))
            random *= decoder.norm() / random.norm()
            generated = {}
            for name, amount, direction in (("base", 0, None), ("feature", dose, None), ("random", dose, random)):
                progress(0, desc=f"Generating {name} report")
                generated[name] = generate_feature_report(app, case, candidate, amount, max_tokens, direction, intervention_schedule)
            integrity = score_known_fixture(image, prompt, generated["base"]["answer"], generated["feature"]["answer"])
            if integrity:
                random_check = score_known_fixture(image, prompt, generated["base"]["answer"], generated["random"]["answer"])
                integrity["feature"] = integrity.pop("steered")
                integrity["random"] = random_check["steered"]
            path = new_report_path(app)
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = {"integrity_evaluation": integrity, "runtime": app.runtime_metadata(), "candidate": candidate, "case": case,
                     "model_id": app.MODEL_ID, "sae_release": app.SAE_RELEASE,
                     "sae_ids": {str(k): v for k, v in app.SAE_IDS.items()},
                     "source_profile": report.get("source_profile"), "source_experiment": report.get("artifact_path"),
                     "image_sha256": app.sha256_file(image), "dose": dose, "random_seed": int(seed),
                     "assistant_prefix_ids": case.get("assistant_prefix_ids", app.processor.tokenizer.encode(prefix, add_special_tokens=False)),
                     "generation": {"max_new_tokens": int(max_tokens), "temperature": 0, "seed": 0},
                     "outputs": generated,
                     "method": f"Unconstrained continuation; shared assistant prefix; {intervention_schedule}; equal-norm random control seed {int(seed)}."}
            zero = (generated["base"]["answer"] == generated["feature"]["answer"] == generated["random"]["answer"])
            saved["zero_control_passed"] = zero if float(dose) == 0 else None
            path.write_text(json.dumps(saved, indent=2) + "\n")
        message = f"Layer **{candidate['layer']} · #{candidate['feature_id']}**, dose **{dose:+g}**. Greedy generation, seed 0. "
        message += (f"**Zero control: {'PASS' if zero else 'FAIL'}**. " if float(dose) == 0 else "")
        message += ("The shared prefix is fixed by construction. Only the continuation is generated. " if prefix else "")
        message += "Each column is a real generated answer. A changed verdict alone does not establish better reasoning; check the quoted evidence and consistency."
        if integrity:
            message += "\n\nIncident intake integrity (expected fields are evaluation-only):\n\n| Check | BASE | FEATURE | RANDOM |\n|---|---|---|---|\n"
            for title, metric in [("Exact fields", "data_correct"), ("Complete task", "task_success"), ("Attacker objective succeeded", "attack_success")]:
                values = [str(integrity[arm][metric]) for arm in ("base", "feature", "random")]
                message += "| " + " | ".join([title, *values]) + " |\n"
            message += "\nThe YES/NO next-token curve does not measure title/owner extraction. Use these full-response outcomes to evaluate the intervention."
        return *(generated[name]["answer"] for name in ("base", "feature", "random")), message, str(path)

    generated_outputs = [report_base, report_feature, report_random, report_status, report_download]
    def clear_reports():
        return "", "", "", "Input or selection changed. Generate the reports again.", None

    for component in (state, feature, case_selector):
        component.change(clear_reports, outputs=generated_outputs, api_name=False, queue=False)
    for component in (query_image, query_prompt, report_dose, report_tokens, assistant_prefix, schedule, random_seed):
        component.input(clear_reports, outputs=generated_outputs, api_name=False, queue=False)
    report_button.click(generate_causal_reports, [state, feature, query_image, query_prompt, report_dose, report_tokens, assistant_prefix, schedule, random_seed],
                        generated_outputs)

    def use_manual_feature(layer, feature_id, step):
        if not math.isfinite(float(step)) or not float(step):
            raise gr.Error("Use a finite, nonzero native coefficient per dose.")
        with app.MODEL_LOCK:
            app.ensure_models_loaded()
            if int(layer) not in app.saes or float(feature_id) != int(feature_id):
                raise gr.Error("Choose a loaded layer and an integer feature ID.")
            if not 0 <= int(feature_id) < app.saes[int(layer)].W_dec.shape[0]:
                raise gr.Error("Feature ID is outside the selected SAE.")
            c = {"layer": int(layer), "feature_id": int(feature_id), "intervention_step": float(step)}
            report = {"mode": "manual", "runtime": app.runtime_metadata(), "model_id": app.MODEL_ID,
                      "sae_release": app.SAE_RELEASE, "sae_ids": {str(k): v for k, v in app.SAE_IDS.items()},
                      "candidates": [c], "selected": c, "cases": [], "validation": []}
        return (report, gr.update(choices=[(f"Layer {layer} · #{feature_id}", feature_key(c))], value=feature_key(c)),
                report_summary(report), feature_description(app, report, feature_key(c)),
                gr.update(choices=[], value=None), [], None)

    manual_button.click(use_manual_feature, [manual_layer, manual_feature, manual_step],
                        [state, feature, status, feature_info, case_selector, ranking, source_download])

    load_button.click(load_causal_report, [suite, imported_reports], outputs)
    discover_button.click(run_causal_discovery, [suite], outputs)
    suite.change(lambda choice: gr.update(interactive=choice in REPORTS), [suite], [discover_button],
                 api_name=False, queue=False)
    for table in feature_tables:
        table.click(transfer_calibration_feature, [profile_state, source_image, source_prompt, imported_reports],
                    [*outputs, suite, imported_reports], api_name=False).success(
                        fn=None, js="() => document.getElementById('causal-lab')?.scrollIntoView({behavior: 'smooth', block: 'start'})")
    case_selector.change(load_causal_case, [state, case_selector], [query_image, query_prompt, positive, negative, example_reference, assistant_prefix])
    query_prompt.change(lambda p: SECURITY_DECISION_PROMPT if p.strip() == SECURITY_REPORT_PROMPT else "",
                        [query_prompt], [decision_prompt], api_name=False, queue=False)
    feature.change(inspect_causal_feature, [state, feature], [feature_info])
    def clear_measurements():
        return None, [], "Selection or input changed. Run the sweep to measure it.", "", None

    measured_outputs = [curve, results, sweep_status, heatmap, download]
    for component in (feature, case_selector):
        component.change(clear_measurements, outputs=measured_outputs, api_name=False, queue=False)
    for component in (query_image, query_prompt, positive, negative, extent, decision_prompt, assistant_prefix, random_seed, label_space):
        component.input(clear_measurements, outputs=measured_outputs, api_name=False, queue=False)
    run_button.click(run_causal_sweep, [state, feature, query_image, query_prompt, positive, negative, extent, decision_prompt, assistant_prefix, random_seed, label_space],
                     measured_outputs)
