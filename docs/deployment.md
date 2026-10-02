# Run and deploy

Use Python 3.12 and uv 0.9.15 or later. Install from the committed lockfile:

```bash
make setup
make benchmark-setup
make check
make test
make login
```

Accept the access terms for [Gemma 3](https://huggingface.co/google/gemma-3-4b-it)
before authenticating. Start with `make mac` on Apple Silicon or `make server` on
NVIDIA; `make run` selects a device automatically. Append `PRELOAD=1` to load Gemma
and four SAEs before accepting requests. Without preload, section 0 loads only Gemma.

Linux/Windows x86_64 use the CUDA 12.4 PyTorch index pinned in `pyproject.toml`;
macOS uses native wheels. Copy `pyproject.toml` and `uv.lock` together when deploying.
Do not copy a `.venv` between machines. Copy `dashboard.py`, `sae_dashboard/`,
`research/`, `data/` and the setup/configuration files. Run tests on the target machine.

Copy `.env.example` to `.env` to configure the host, port, device, data root, cache or
artifact root. Make loads `.env`; direct Python execution does not. Environment variables
override `.env`. `make mac` forces MPS and `make server` forces CUDA.

The default address is http://127.0.0.1:7860. For remote access through SSH:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@server
```

For shared access on a trusted network set `GRADIO_SERVER_NAME=0.0.0.0`. Place public
instances behind authentication and HTTPS: session isolation is not authentication.
Uploaded manifests can read assets only under `GEMMA_DATA_ROOT`; mount a shared data
directory there for the team's images/text assets. Uploads contain JSON only.

## Persistent Linux service

After setup and login as the service user, adapt this systemd unit:

```ini
[Unit]
Description=Hackathon 2026 steering dashboard
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/PATH/TO/PROJECT
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/make server PRELOAD=1 UV=/ABSOLUTE/PATH/TO/uv
Restart=on-failure
RestartSec=10
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

Use one app process per GPU/model allocation. The process serializes model access and
holds session profiles in CPU memory; additional browser sessions do not duplicate model
weights. Reloads/restarts lose unsaved session data. Results are saved only after the user
enables saving, and model exports only after the final export button. `runs/` is ignored
by Git. Budget disk space before including full model weights in an export.
