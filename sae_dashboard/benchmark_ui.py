"""Benchmark preview and base-gated results, kept separate from the main layout."""

import html
import gradio as gr
from research.benchmarks import BENCHMARKS
from . import workflow


def result_tables(result):
    summary = [
        [
            key.replace("_", " ").capitalize(),
            "n/a"
            if value is None
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value),
        ]
        for key, value in result["summary"].items()
    ]
    eligible, excluded = [], []
    for row in result["rows"]:
        shared = [
            str(row["id"]),
            row["category"],
            row["prompt"],
            row["reference"],
            row["base"],
            "Cache" if row["base_cached"] else "Inference",
        ]
        if row["base_score"]["correct"]:
            score = row["steered_score"]
            eligible.append(
                [
                    *shared,
                    row["steered"] or "",
                    "Not evaluated"
                    if score is None
                    else "Preserved"
                    if score["correct"]
                    else "Regressed",
                ]
            )
        else:
            excluded.append(shared)
    s = result["summary"]
    status = f"**{s['base_correct']}/{s['num_items']} base-correct** · {s['base_failed_excluded']} excluded · {s['base_cache_hits']} cached base answers · {s['steered_evaluated']} steered evaluations."
    return status, summary, eligible, excluded


def build_benchmark_section(
    session, common, layer_strengths, settings, strengths, friendly
):
    gr.Markdown("## 5. Evaluate coherence", elem_classes="section-title")
    gr.Markdown(
        "Review the exact inputs and expected answers. Only base-correct cases reach the steered model. "
        "For simple visual questions, choose POPE with the random split (object presence, yes/no)."
    )
    with gr.Row():
        name = gr.Dropdown(
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
    prepare = gr.Button("1 · Prepare sample", variant="primary")
    preview_status = gr.Markdown("Prepare a sample before running inference.")
    item_selector = gr.Dropdown(choices=[], label="Preview case", interactive=True)
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Markdown(
                label="Exact model prompt",
                sanitize_html=True,
                line_breaks=True,
                buttons=["copy"],
            )
        with gr.Column():
            expected = gr.Markdown(
                label="Correct answer / requirements",
                sanitize_html=True,
                line_breaks=True,
            )
            metadata = gr.HTML()
    images = gr.Gallery(
        label="Prompt images",
        columns=3,
        height=280,
        visible=False,
        interactive=False,
        object_fit="contain",
        preview=True,
        selected_index=0,
        buttons=["fullscreen"],
    )
    with gr.Accordion("Sample overview", open=False):
        overview = gr.Dataframe(
            headers=[
                "ID",
                "Category",
                "Type",
                "Images",
                "Correct answer / requirements",
            ],
            datatype=["str", "str", "str", "number", "markdown"],
            wrap=True,
            interactive=False,
        )
    with gr.Row():
        base_btn = gr.Button("2 · Evaluate base", variant="primary")
        steered_btn = gr.Button(
            "3 · Evaluate steered on base-correct cases", variant="primary"
        )
    gr.Markdown(
        "Base responses are cached automatically as JSON. Steered responses are recomputed on every click and never saved to disk."
    )
    status = gr.Markdown()
    with gr.Accordion("Detailed metrics", open=False):
        summary = gr.Dataframe(headers=["Metric", "Value"], interactive=False)
    gr.Markdown("### Base-correct cases · steering evaluation")
    headers = [
        "ID",
        "Category",
        "Prompt",
        "Correct answer / requirements",
        "Base response",
        "Base source",
    ]
    datatypes = ["str", "str", "markdown", "markdown", "markdown", "str"]
    eligible = gr.Dataframe(
        headers=[*headers, "Steered response", "Outcome"],
        datatype=[*datatypes, "markdown", "str"],
        column_widths=[90, 110, 220, 150, 180, 80, 180, 100],
        wrap=True,
        interactive=False,
    )
    with gr.Accordion("Base failures · excluded from steering", open=False):
        excluded = gr.Dataframe(
            headers=headers,
            datatype=datatypes,
            column_widths=[90, 110, 300, 180, 300, 100],
            wrap=True,
            interactive=False,
        )
    with gr.Accordion("Protocol and cache details", open=False):
        gr.Markdown("""The preservation rate uses **base-correct cases only**. Base accuracy uses the full sample.
Base failures are never sent to the steered model. A sample with no base-correct cases has no preservation rate.

- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro): correct option letter and text.
- [MMMU](https://huggingface.co/datasets/MMMU/MMMU): labeled dev/validation cases, including all prompt images.
- [POPE](https://huggingface.co/datasets/lmms-lab-encoder/POPE): one photo and an object-presence question. Start with `random`; `popular` and `adversarial` select harder negative objects. No subject filter is needed. Answers must be only yes/no (case-insensitive, optional final period or exclamation mark); explanations, empty and ambiguous answers count as incorrect. This strict scoring differs from the original POPE evaluator.
- [IFEval](https://huggingface.co/datasets/google/IFEval): all required instructions must pass strict evaluation; there is no single reference answer.

The expected answer is shown here for inspection and is **not added to the model prompt**.
JSON base-cache keys include the exact processed input, image tensors, seed, temperature, token limit,
model identity, precision and software versions. Cache location: `.cache/base_responses` by default
(`GEMMA_BASE_CACHE_DIR` overrides it). This automatic cache is separate from the experiment-saving button.
The model may still load to validate its identity and prepare inputs on a cache hit; generation is skipped.
These sampled zero-shot runs are not leaderboard reproductions.""")
    outputs = [status, summary, eligible, excluded]
    empty = ["", [], [], []]
    preview_outputs = [
        item_selector,
        preview_status,
        prompt,
        expected,
        metadata,
        images,
        overview,
    ]
    blank_preview = [
        gr.update(choices=[], value=None),
        "Prepare a sample before running inference.",
        "",
        "",
        "",
        gr.update(value=[], visible=False),
        [],
    ]

    def show_item(state, index):
        with state.lock:
            sample = state.benchmark_sample
            if sample is None or index is None:
                return "", "", "", gr.update(value=[], visible=False)
            item = sample.items[int(index)]
            gallery = [(image, label) for label, image in item["images"]]
            info = {
                "Benchmark": sample.adapter.name,
                "Split": sample.request.split,
                "ID": item["id"],
                "Category": item["category"] or "Not provided",
                "Type": item["question_type"],
                "Difficulty": item["difficulty"] or "Not provided",
                "Images": len(gallery),
            }
            metadata_text = (
                '<div class="benchmark-meta">'
                + "".join(
                    f"<div><small>{html.escape(key)}</small><strong>{html.escape(str(value))}</strong></div>"
                    for key, value in info.items()
                    if value != "Not provided"
                )
                + "</div>"
            )
            return (
                "### Exact model prompt\n\n" + item["prompt"],
                "### Correct answer / requirements\n\n" + item["reference"],
                metadata_text,
                gr.update(
                    value=gallery,
                    visible=bool(gallery),
                    selected_index=0 if gallery else None,
                ),
            )

    @friendly
    def prepare_sample(state, selected, selected_split, items, subject, *values):
        sample = workflow.prepare_evaluation(
            state, selected, selected_split, items, subject, settings(values)
        )
        options = [
            (f"{i + 1} · {item['id']} · {item['category'] or item['question_type']}", i)
            for i, item in enumerate(sample.items)
        ]
        rows = [
            [
                str(item["id"]),
                item["category"],
                item["question_type"],
                len(item["images"]),
                item["reference"],
            ]
            for item in sample.items
        ]
        return (
            gr.update(choices=options, value=0),
            f"**{len(sample.items)} cases ready** · Review cases below, then evaluate the base model.",
            *show_item(state, 0),
            rows,
            *empty,
        )

    prepare.click(
        prepare_sample,
        [session, name, split, count, category, *common],
        [*preview_outputs, *outputs],
    )
    item_selector.input(
        show_item, [session, item_selector], [prompt, expected, metadata, images]
    )

    def invalidate_sample(state):
        with state.lock:
            state.benchmark_sample = None
            state.benchmark_base = []
            state.benchmark_settings = None
            state.benchmark_result = None
        return [*blank_preview, *empty]

    def change_benchmark(state, selected):
        adapter = BENCHMARKS[selected]
        return (
            gr.update(choices=list(adapter.splits), value=adapter.default_split),
            gr.update(value="", interactive=adapter.supports_category),
            *invalidate_sample(state),
        )

    name.input(
        change_benchmark, [session, name], [split, category, *preview_outputs, *outputs]
    )
    for control in (split, count, category, common[0]):
        control.input(invalidate_sample, session, [*preview_outputs, *outputs])

    def invalidate_answers(state):
        with state.lock:
            state.benchmark_base = []
            state.benchmark_settings = None
            state.benchmark_result = None
        return empty

    for control in common[1:3]:
        control.input(invalidate_answers, session, outputs)

    @friendly
    def run_base(state, progress=gr.Progress(), *values):
        return result_tables(workflow.benchmark_base(state, settings(values), progress))

    @friendly
    def run_steered(state, progress=gr.Progress(), *values):
        return result_tables(
            workflow.benchmark_steered(
                state, settings(values), strengths(values[5:]), progress
            )
        )

    base_btn.click(run_base, [session, *common], outputs)
    steered_btn.click(run_steered, [session, *common, *layer_strengths], outputs)
    return outputs, empty
