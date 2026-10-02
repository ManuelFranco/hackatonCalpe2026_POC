"""Session-local, lazy previews of the exact inputs used for profile capture."""

import html
import re
import gradio as gr


def build_input_preview():
    with gr.Column(visible=False, elem_classes="input-preview") as panel:
        gr.Markdown("### Loaded pairs")
        with gr.Row():
            source = gr.Dropdown(label="Input source", choices=[], interactive=True)
            pair = gr.Dropdown(label="Pair", choices=[], interactive=True)
        details = gr.HTML()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### A · Reference")
                a_image = gr.Image(
                    label="Reference image",
                    interactive=False,
                    visible=False,
                    height=320,
                    format="png",
                )
                a_text = gr.Markdown(
                    label="Reference text",
                    sanitize_html=True,
                    line_breaks=True,
                    max_height=440,
                    buttons=["copy"],
                    visible=False,
                )
            with gr.Column():
                gr.Markdown("### B · Target")
                b_image = gr.Image(
                    label="Target image",
                    interactive=False,
                    visible=False,
                    height=320,
                    format="png",
                )
                b_text = gr.Markdown(
                    label="Target text",
                    sanitize_html=True,
                    line_breaks=True,
                    max_height=440,
                    buttons=["copy"],
                    visible=False,
                )
    return [panel, source, pair, details, a_text, a_image, b_text, b_image]


def _source(state, name):
    manifest = next((m for m in state.manifests if m.name == name), None)
    if manifest is None:
        raise ValueError("This input source changed. Select a currently loaded source.")
    return manifest


def _condition_preview(condition, code=False):
    text = condition.text
    if code and text.strip():
        # Keep C snippets literal, including preprocessor lines and Markdown symbols.
        fence = "`" * max(
            3, max((len(s) for s in re.findall(r"`+", text)), default=0) + 1
        )
        text = f"{fence}c\n{text}\n{fence}"
    image = condition.load_image()
    if image is not None:
        # Resize only the display copy; profile capture always uses the original.
        image.thumbnail((1600, 1600))
    return (
        gr.update(value=text, visible=bool(condition.text.strip())),
        gr.update(value=image, visible=image is not None),
    )


def _modality(condition):
    has_image = condition.image is not None or condition.image_bytes is not None
    return (
        "Text + image"
        if has_image and condition.text.strip()
        else "Image only"
        if has_image
        else "Text only"
    )


def pair_preview(state, source_name, pair_id):
    with state.lock:
        source = _source(state, source_name)
        index = next((i for i, p in enumerate(source.pairs) if p.id == pair_id), None)
        if index is None:
            raise ValueError("This pair changed. Select a currently loaded pair.")
        pair = source.pairs[index]
        details = (
            '<div class="input-preview-meta">'
            f"<span>Pair <strong>{index + 1} / {len(source.pairs)}</strong></span>"
            f"<span>ID <strong>{html.escape(pair.id)}</strong></span>"
            f"<span>A · {_modality(pair.a)}</span><span>B · {_modality(pair.b)}</span>"
            "</div>"
        )
        return (
            details,
            *_condition_preview(pair.a, code=source.name.startswith("CWE-")),
            *_condition_preview(pair.b, code=source.name.startswith("CWE-")),
        )


def source_preview(state, source_name):
    with state.lock:
        source = _source(state, source_name)
        first = source.pairs[0].id
        return (
            gr.update(
                choices=[
                    (f"{i} · {p.id}", p.id) for i, p in enumerate(source.pairs, 1)
                ],
                value=first,
            ),
            *pair_preview(state, source_name, first),
        )


def refresh_preview(state):
    with state.lock:
        if not state.manifests:
            return (
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                "",
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
            )
        first = state.manifests[0].name
        return (
            gr.update(visible=True),
            gr.update(choices=[m.name for m in state.manifests], value=first),
            *source_preview(state, first),
        )


def wire_input_preview(session, controls, friendly):
    _, source, pair, *content = controls
    source.input(friendly(source_preview), [session, source], [pair, *content])
    pair.input(friendly(pair_preview), [session, source, pair], content)
