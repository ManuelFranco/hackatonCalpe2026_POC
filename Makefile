.DEFAULT_GOAL := help

UV ?= uv
PRELOAD ?= 0
ENV_FILE ?= .env
ENV_OPTION = $(if $(wildcard $(ENV_FILE)),--env-file "$(ENV_FILE)",)
RUN = $(UV) run --locked $(ENV_OPTION)
APP = python dashboard.py
PRELOAD_OPTION = $(if $(filter 1,$(PRELOAD)),--preload,)

.PHONY: help require-uv setup login check run mac server stop lint

help:
	@printf '%s\n' \
	  'make setup            Install locked dependencies with uv' \
	  'make benchmark-setup  Download IFEval tokenizer data' \
	  'make login            Authenticate with Hugging Face' \
	  'make check            Validate configuration and UI without model weights' \
	  'make test             Run regression tests' \
	  'make mac              Start on Apple Silicon (MPS)' \
	  'make server           Start on NVIDIA (CUDA)' \
	  'make run              Start with automatic device selection' \
	  'make stop             Stop the dashboard' \
	  'make lint             Check Python code with Ruff' \
	  '' \
	  'Add PRELOAD=1 to load models before serving. Configure .env as needed.' \
	  'Default URL: http://127.0.0.1:7860. See docs/deployment.md.'

require-uv:
	@command -v "$(UV)" >/dev/null 2>&1 || { printf '%s\n' 'uv is missing. Install it from https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }

setup: require-uv
	$(UV) sync --locked

login: require-uv
	$(RUN) hf auth login

check: require-uv
	$(RUN) $(APP) --check

# uv preserves existing environment variables over values in the env file.
# Set the loopback fallback after uv loads .env so an explicit host takes effect.
run: require-uv
	$(RUN) sh -c 'export GRADIO_SERVER_NAME="$${GRADIO_SERVER_NAME:-127.0.0.1}"; exec python dashboard.py "$$@"' sh $(PRELOAD_OPTION)

mac:
	@GEMMA_DEVICE=mps $(MAKE) run PRELOAD=$(PRELOAD)

server:
	@GEMMA_DEVICE=cuda $(MAKE) run PRELOAD=$(PRELOAD)

stop:
	@pkill -TERM -f '^([^ ]*/)?python([0-9.]+)? dashboard[.]py( --preload)?$$'; status=$$?; \
	case $$status in \
	  0) printf '%s\n' 'Stop signal sent.' ;; \
	  1) printf '%s\n' 'The dashboard is not running.' ;; \
	  *) exit $$status ;; \
	esac

lint: require-uv
	$(RUN) ruff check dashboard.py sae_dashboard research tests

.PHONY: test benchmark-setup
test: require-uv
	$(RUN) python -m unittest discover -s tests -v

benchmark-setup: require-uv
	$(RUN) python -m nltk.downloader -d .cache/nltk punkt_tab
