#!/usr/bin/env python3
"""Hackathon 2026: Gemma 3 steering dashboard."""

import argparse
import json
import os
import gradio as gr
from sae_dashboard import model_runtime as runtime
from sae_dashboard.ui import CSS, build_demo, english_widgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and UI without downloading weights",
    )
    parser.add_argument(
        "--preload", action="store_true", help="Load Gemma and SAEs before serving"
    )
    args = parser.parse_args()
    runtime.validate_configuration()
    runtime.validate_sae_registry()
    demo = build_demo()
    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "model_weights_loaded": False,
                    "runtime": runtime.runtime_metadata(),
                },
                indent=2,
            )
        )
        demo.close()
        return
    if args.preload:
        with runtime.MODEL_LOCK:
            runtime.ensure_models_loaded()
    demo.queue(default_concurrency_limit=8, max_size=64)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        i18n=english_widgets(),
        footer_links=[],
        css=CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
