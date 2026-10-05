"""Feature discovery in Common profile without manual IDs or generation."""

import gradio as gr

from . import profile_discovery_lab as lab, profile_discovery_view as view
from . import model_runtime as runtime


def build_discovery_panel():
    gr.Markdown("### Discover features before steering")
    gr.Markdown(
        "Automatically shortlist consistent A/B contrasts, then inspect their input-token locations together."
    )
    with gr.Row():
        layer = gr.Dropdown(
            runtime.LAYERS, value=17, label="Discovery layer", interactive=False
        )
        pair = gr.Dropdown([], label="A/B pair to inspect", interactive=False, scale=3)
    ranked = gr.HTML(value=view.empty(), label="Automatic feature candidates")
    with gr.Row():
        inspect = gr.Button(
            "Inspect candidate locations", variant="primary", interactive=False
        )
        follow = gr.Button("Follow top 3 in response viewer", interactive=False)
    result = gr.State(None)
    observation = gr.HTML(label="Candidate input locations")
    return [layer, pair, ranked, inspect, result, observation, follow]


def cleared():
    return [
        gr.update(choices=runtime.LAYERS, value=17, interactive=False),
        gr.update(choices=[], value=None, interactive=False),
        view.empty(),
        gr.update(interactive=False),
        None,
        "",
        gr.update(interactive=False),
    ]


def refresh(session, layer=None, pair_index=None):
    with session.lock:
        if not session.profile:
            return cleared()
        available = [k for k in runtime.LAYERS if k in session.profile]
        if not available:
            return cleared()
        if layer not in available:
            layer = 17 if 17 in available else available[0]
        rows = lab.rank_candidates(session.profile[layer])
        loaded = lab.pairs(session)
        choices = [
            (f"{i + 1} · {label}", str(i)) for i, (label, _) in enumerate(loaded)
        ]
        valid_pair = str(pair_index).isdigit() and int(pair_index) < len(loaded)
        selected = str(pair_index) if valid_pair else "0" if loaded else None
        return [
            gr.update(choices=available, value=layer, interactive=True),
            gr.update(choices=choices, value=selected, interactive=bool(choices)),
            view.render_candidates(rows, layer, [label for label, _ in loaded]),
            gr.update(interactive=bool(rows and choices)),
            None,
            "",
            gr.update(interactive=bool(rows)),
        ]


def wire_discovery(
    session,
    controls,
    common,
    settings,
    friendly,
    generation_controls,
    generation_outputs,
    generation_cleared,
    tabs,
):
    layer, pair, _, inspect, result, observation, follow = controls

    @friendly
    def discover_inputs(
        state, selected_layer, selected_pair, progress=gr.Progress(), *values
    ):
        measured = lab.inspect_pair(
            state, settings(values), selected_layer, selected_pair, progress
        )
        return measured, view.render_observation(measured)

    event = inspect.click(
        discover_inputs, [session, layer, pair, *common], [result, observation]
    )
    layer.input(friendly(refresh), [session, layer, pair], controls, cancels=[event])
    pair.input(lambda: (None, ""), None, [result, observation], cancels=[event])
    for control in common[3:]:
        control.input(lambda: (None, ""), None, [result, observation], cancels=[event])

    @friendly
    def follow_candidates(state, selected_layer, *values):
        with state.lock:
            rows = lab.candidates(state, settings(values), selected_layer)
            if not rows:
                raise ValueError("No candidates in this discovery layer.")
            return (
                selected_layer,
                ", ".join(str(row["feature"]) for row in rows[:3]),
                *generation_cleared,
                gr.update(selected="extra"),
            )

    follow.click(
        follow_candidates,
        [session, layer, *common],
        [
            generation_controls["layer"],
            generation_controls["observed"],
            *generation_outputs,
            tabs,
        ],
    )
