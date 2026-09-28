.DEFAULT_GOAL := help

UV ?= uv
PRELOAD ?= 0
ENV_FILE ?= .env
ENV_OPTION = $(if $(wildcard $(ENV_FILE)),--env-file "$(ENV_FILE)",)
RUN = $(UV) run --locked $(ENV_OPTION)
APP = python demo_gradio_hackaton.py
PRELOAD_OPTION = $(if $(filter 1,$(PRELOAD)),--preload,)

.PHONY: help require-uv setup login check run mac server stop lint

help:
	@printf '%s\n' \
	  'make setup   Instala Python 3.12 y las dependencias fijadas con uv' \
	  'make login   Inicia sesión en Hugging Face (acepta antes la licencia de Gemma)' \
	  'make check   Comprueba imports, registro SAE e interfaz sin descargar pesos' \
	  'make mac     Arranca en Apple Silicon con MPS' \
	  'make server  Arranca en NVIDIA con CUDA' \
	  'make run     Arranca con selección automática de dispositivo o lo indicado en .env' \
	  'make stop    Detiene la app arrancada con run, mac o server' \
	  'make lint    Comprueba el código con Ruff' \
	  '' \
	  'Opcional: cp .env.example .env y edita puerto, dispositivo y memoria.' \
	  'Añade PRELOAD=1 para cargar los modelos antes de abrir la interfaz.' \
	  'Acceso por defecto: http://127.0.0.1:7860. Detener: Ctrl+C o make stop.' \
	  'Guía de servidor persistente: DEPLOY_ES.md'

require-uv:
	@command -v "$(UV)" >/dev/null 2>&1 || { printf '%s\n' 'Falta uv. Instálalo siguiendo https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }

setup: require-uv
	$(UV) sync --locked

login: require-uv
	$(RUN) hf auth login

check: require-uv
	$(RUN) $(APP) --check

# uv preserves existing environment variables over values in the env file.
# Set the loopback fallback after uv loads .env so an explicit host takes effect.
run: require-uv
	$(RUN) sh -c 'export GRADIO_SERVER_NAME="$${GRADIO_SERVER_NAME:-127.0.0.1}"; exec python demo_gradio_hackaton.py "$$@"' sh $(PRELOAD_OPTION)

mac:
	@GEMMA_DEVICE=mps $(MAKE) run PRELOAD=$(PRELOAD)

server:
	@GEMMA_DEVICE=cuda $(MAKE) run PRELOAD=$(PRELOAD)

stop:
	@pkill -TERM -f '^([^ ]*/)?python([0-9.]+)? demo_gradio_hackaton[.]py( --preload)?$$'; status=$$?; \
	case $$status in \
	  0) printf '%s\n' 'Señal de parada enviada a la app.' ;; \
	  1) printf '%s\n' 'La app no está en ejecución.' ;; \
	  *) exit $$status ;; \
	esac

lint: require-uv
	$(RUN) ruff check demo_gradio_hackaton.py sae_dashboard tests
