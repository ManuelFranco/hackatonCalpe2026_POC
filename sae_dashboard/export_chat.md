# Use the exported model in VS Code with Continue

The server loads the base model plus the exported directions and strengths. It
applies activation steering on every generation. No SAE or dashboard is needed.
Use Continue **Chat** mode: tool calling, Agent mode and autocomplete are not
implemented. Chat supports conversation history, code context supplied as text,
and attached images (base64 image data URLs).

## Start the model server

Extract the ZIP. In the extracted directory, using Python 3.12:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-server.txt
.venv/bin/python serve_model.py . --port 8001
```

The API becomes available after the model has loaded. If `base_model/` is absent,
the loader downloads the referenced model; Hugging Face access to Gemma is required.
On the hackathon NVIDIA server, prefer the repository command below to reuse its
locked CUDA 12.4 dependencies. Loading this server while the dashboard holds the
base model uses memory for another model instance; stop the dashboard first if
the GPU cannot fit both. The model server runs in the foreground; keep it running.

### Existing exports and the hackathon repository

Old ZIPs do not include this API server. They still work with the updated project
loader; there is no need to export or download the model weights again:

```bash
cd /home/andromeda/hackathonCalpe2026_POC
uv sync --locked
uv run --locked python serve_export.py /absolute/path/to/extracted/export --port 8001
```

Replace the export path with the folder containing `steering_config.json` and
`steering_vectors.safetensors`. The repository loader is used, so an old
`steered_model.py` inside that folder does not need updating.

## Connect VS Code on your computer

If the model runs on `fl-andromeda`, open this tunnel in a terminal on your computer
and leave it running:

```bash
ssh -N -L 8001:127.0.0.1:8001 fl-andromeda
```

If Continue runs on the server itself through VS Code Remote SSH, its localhost
already refers to the server; no additional tunnel is needed. A locally running
model server also needs no tunnel.

Install the **Continue** VS Code extension. Open its local `config.yaml` from
Continue's configuration controls. For a new configuration, use the bundled
`continue.example.yaml`. For an existing configuration, append just its model
entry under your existing `models:` list; preserve your other models/settings.

Select **Gemma Steered** and use **Chat** mode. Attach an image or ask about selected
code. Continue sends the conversation to `http://127.0.0.1:8001/v1` with model ID
`gemma-steered`. The `openai` provider selects the API protocol; it does not send
requests to OpenAI when this local `apiBase` is configured.

The sample sets temperature 0 and an output limit of 512 tokens; adjust these to
match your experiments. Exported strengths and vectors remain fixed. There is no
API parameter for changing or disabling steering. The 8,192-token server limit
includes prompt/image tokens plus requested output; increase `--context-length`
and Continue's `contextLength` together only if your model and memory permit it.

## Check the connection

Run from the machine where Continue runs:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-steered","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

The first command should list `gemma-steered`; the second runs actual steered
inference. The API supports JSON and SSE responses. **SSE is buffered:** the answer
arrives after generation finishes, rather than token by token. Continue may show
its loading indicator during this time. Unsupported tool requests and nonzero
frequency/presence penalties are rejected instead of silently ignored.

The server binds to localhost by default. The placeholder `local-steering` key in
the example is sufficient for this SSH-tunnel setup. If you set `STEERED_API_KEY`
on the server, set the same value in Continue's `apiKey` (and pass it as a Bearer
token when testing). Binding to other interfaces requires that environment variable.

Provider configuration follows the [Continue OpenAI-compatible API documentation](https://docs.continue.dev/customize/model-providers/top-level/openai)
and [capability reference](https://docs.continue.dev/customize/deep-dives/model-capabilities).
