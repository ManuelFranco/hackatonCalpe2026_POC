"""Optional manual pair editor for section 1."""

import html
import gradio as gr
from .manual_pairs import change_pair, read_pair


def input_summary(rows):
    """Render every input source without virtualized rows hiding new additions."""
    if not rows:
        return ""
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="input-summary"><table><thead><tr>'
        '<th scope="col">Input source</th><th scope="col">Pairs</th>'
        '<th scope="col">Fingerprint</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def build_manual_editor():
    with gr.Accordion("Manual image–text pairs (optional)", open=False):
        gr.Markdown(
            "Add A/B pairs directly, on their own or alongside JSON manifests. Each side needs text, an image, or both."
        )
        selected = gr.Dropdown(
            choices=[], label="Edit an existing manual pair", interactive=True
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### A · Reference")
                a_text = gr.Textbox(
                    label="A · Text",
                    lines=4,
                    placeholder="Reference text (optional if an image is provided)",
                )
                a_image = gr.Image(
                    type="pil",
                    label="A · Image (optional)",
                    height=220,
                    sources=["upload", "clipboard"],
                )
            with gr.Column():
                gr.Markdown("### B · Target")
                b_text = gr.Textbox(
                    label="B · Text",
                    lines=4,
                    placeholder="Target text (optional if an image is provided)",
                )
                b_image = gr.Image(
                    type="pil",
                    label="B · Image (optional)",
                    height=220,
                    sources=["upload", "clipboard"],
                )
        with gr.Row():
            add = gr.Button("Add pair", variant="primary")
            update = gr.Button("Update selected pair", interactive=False)
            remove = gr.Button("Remove selected pair", interactive=False)
        notice = gr.Markdown(
            "Manual pairs stay in this session. Changes require rebuilding the common profile."
        )
    return dict(
        selected=selected,
        a_text=a_text,
        a_image=a_image,
        b_text=b_text,
        b_image=b_image,
        add=add,
        update=update,
        remove=remove,
        notice=notice,
    )


def wire_manual_editor(session, controls, data_outputs, reset_values, friendly):
    selected = controls["selected"]
    fields = [controls[k] for k in ("a_text", "a_image", "b_text", "b_image")]
    buttons = [controls[k] for k in ("update", "remove")]
    editor_outputs = [selected, *fields, *buttons, controls["notice"]]

    def handle(state, action, pair_id, *values):
        with state.lock:
            table = change_pair(state, action, pair_id, *values)
            count = len(state.manual_pairs)
            total = sum(row[1] for row in table)
            return (
                f"**{total} {'pair' if total == 1 else 'pairs'} ready** · {count} manual. Build the common profile in section 2.",
                input_summary(table),
                *reset_values,
                gr.update(choices=[p.id for p in state.manual_pairs], value=None),
                "",
                None,
                "",
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
                f"Pair {'added' if action == 'add' else 'updated' if action == 'update' else 'removed'}. **{count} manual {'pair' if count == 1 else 'pairs'}** · Previous profile and vectors cleared.",
            )

    @friendly
    def add_manual_pair(state, *values):
        return handle(state, "add", None, *values)

    @friendly
    def update_manual_pair(state, pair_id, *values):
        return handle(state, "update", pair_id, *values)

    @friendly
    def remove_manual_pair(state, pair_id):
        return handle(state, "remove", pair_id)

    @friendly
    def select_manual_pair(state, pair_id):
        if pair_id is None:
            return (
                "",
                None,
                "",
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        return (
            *read_pair(state, pair_id),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    selected.input(select_manual_pair, [session, selected], [*fields, *buttons])
    controls["add"].click(
        add_manual_pair, [session, *fields], [*data_outputs, *editor_outputs]
    )
    controls["update"].click(
        update_manual_pair,
        [session, selected, *fields],
        [*data_outputs, *editor_outputs],
    )
    controls["remove"].click(
        remove_manual_pair, [session, selected], [*data_outputs, *editor_outputs]
    )
