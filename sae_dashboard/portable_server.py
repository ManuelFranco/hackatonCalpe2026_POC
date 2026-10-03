"""OpenAI-compatible chat API; also copied to standalone exports as serve_model.py."""

import argparse
import base64
import binascii
import hmac
import io
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[dict] = Field(min_length=1, max_length=200)
    stream: bool = False
    stream_options: dict | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=1)
    presence_penalty: float = Field(default=0, ge=0, le=0)
    frequency_penalty: float = Field(default=0, ge=0, le=0)
    tools: list | None = None
    tool_choice: str | dict | None = None
    user: str | None = None


def image_from_data_url(url):
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError(
            "Attach images as base64 data URLs; remote URLs and file paths are unsupported."
        )
    header, separator, encoded = url.partition(",")
    if not separator or not header.endswith(";base64") or len(encoded) > 28_000_000:
        raise ValueError("Invalid image data URL or image exceeds 20 MB.")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 20_000_000:
            raise ValueError("Image exceeds 20 MB.")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("Image exceeds 25 megapixels.")
            return image.convert("RGB")
    except (
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("Invalid image attachment.") from exc


def prepare_messages(messages):
    """Preserve roles, history and images; combine adjacent turns for Gemma's template."""
    prepared = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"} or message.get("tool_calls"):
            raise ValueError(
                "Use Continue Chat mode. Tool calls and tool messages are unsupported."
            )
        if role == "system" and any(m["role"] != "system" for m in prepared):
            raise ValueError("System instructions must precede the conversation.")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            raise ValueError("Each message needs text or image content.")
        blocks = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Invalid message content block.")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url" and role == "user":
                value = part.get("image_url")
                url = value.get("url") if isinstance(value, dict) else value
                blocks.append({"type": "image", "image": image_from_data_url(url)})
            else:
                raise ValueError("Only text and user image_url content are supported.")
        if prepared and prepared[-1]["role"] == role:
            prepared[-1]["content"].extend([{"type": "text", "text": "\n\n"}, *blocks])
        else:
            prepared.append({"role": role, "content": blocks})
    turns = [m for m in prepared if m["role"] != "system"]
    if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != "user":
        raise ValueError("The conversation must start and end with a user turn.")
    return prepared


def create_app(vlm, model_id="gemma-steered", api_key=None, context_length=8192):
    app = FastAPI(title="Steered Gemma chat", docs_url=None, redoc_url=None)

    def authorize(authorization: str | None = Header(default=None)):
        if api_key and not hmac.compare_digest(
            authorization or "", f"Bearer {api_key}"
        ):
            raise HTTPException(status_code=401, detail="Invalid API key.")

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    def completions(request: ChatRequest):
        if request.model != model_id:
            raise HTTPException(status_code=404, detail=f"Use model ID: {model_id}")
        if request.tools or request.tool_choice not in (None, "none"):
            raise HTTPException(
                status_code=400,
                detail="Use Continue Chat mode; tool calling is unsupported.",
            )
        overrides = {
            name: getattr(request, name)
            for name in ("temperature", "top_p", "seed")
            if getattr(request, name) is not None
        }
        if request.max_tokens is not None and request.max_completion_tokens is not None:
            raise HTTPException(
                status_code=400, detail="Specify only one output token limit."
            )
        limit = request.max_completion_tokens or request.max_tokens
        if limit is not None:
            overrides["max_new_tokens"] = limit
        stop = [request.stop] if isinstance(request.stop, str) else request.stop or []
        if len(stop) > 4 or any(not s or len(s) > 256 for s in stop):
            raise HTTPException(
                status_code=400,
                detail="Use at most four non-empty stop strings of up to 256 characters.",
            )
        overrides["stop"] = stop
        try:
            messages = prepare_messages(request.messages)
            result = vlm.chat(messages, max_context_tokens=context_length, **overrides)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        identity = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": model_id,
        }
        if not request.stream:
            return JSONResponse(
                {
                    **identity,
                    "object": "chat.completion",
                    "usage": result["usage"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result["text"]},
                            "finish_reason": result["finish_reason"],
                        }
                    ],
                }
            )

        def events():
            # Buffered SSE: generation finishes first, then a valid OpenAI stream is sent.
            for delta, finish in (
                ({"role": "assistant", "content": ""}, None),
                ({"content": result["text"]}, None),
                ({}, result["finish_reason"]),
            ):
                chunk = {
                    **identity,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            if (request.stream_options or {}).get("include_usage"):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            **identity,
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "usage": result["usage"],
                        }
                    )
                    + "\n\n"
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Serve an extracted steering export for Continue Chat."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Extracted ZIP directory containing steering_config.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-id", default="gemma-steered")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--context-length", type=int, default=8192)
    args = parser.parse_args()
    if not (args.directory / "steering_config.json").is_file():
        parser.error("Extract the export ZIP first; steering_config.json is missing.")
    if args.context_length < 1:
        parser.error("--context-length must be positive.")
    key = os.getenv("STEERED_API_KEY")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not key:
        parser.error(
            "Set STEERED_API_KEY when binding outside localhost, or use an SSH tunnel."
        )
    if __package__:
        from .portable_model import SteeredVLM
    else:
        from steered_model import SteeredVLM
    import uvicorn

    vlm = SteeredVLM(args.directory, device_map=args.device_map)
    print(
        f"Serving {args.model_id} with exported steering at http://{args.host}:{args.port}/v1",
        flush=True,
    )
    uvicorn.run(
        create_app(vlm, args.model_id, key, args.context_length),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
