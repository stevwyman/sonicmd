.PHONY: help models devices up down logs sidecar sidecar-meet sidecar-venv capture-test app-dev

help:
	@echo "StenoPod"
	@echo "  make models        Download the Qwen 3B GGUF into ./models"
	@echo "  make up            Build and start app + llama.cpp (Podman)"
	@echo "  make down          Stop the stack"
	@echo "  make logs          Follow container logs"
	@echo "  make devices       List macOS audio capture devices"
	@echo "  make sidecar-venv  Create host venv (Whisper + pyannote.audio)"
	@echo "  make capture-test  Record 3s from BlackHole and check for real audio"
	@echo "  make sidecar       Capture system audio (BlackHole) + transcribe"
	@echo "  make sidecar-meet  Mix BlackHole + microphone (Webex / Meet)"
	@echo "  make app-dev       Run the web app on the host (no Podman)"

models:
	./scripts/download-llm.sh

capture-test:
	./scripts/capture-test.sh

up:
	podman compose up --build -d

down:
	podman compose down

logs:
	podman compose logs -f

sidecar-venv:
	python3 -m venv .venv-host
	.venv-host/bin/pip install -U pip
	.venv-host/bin/pip install -r host/requirements.txt

sidecar:
	./.venv-host/bin/python host/sidecar.py

sidecar-meet:
	./.venv-host/bin/python host/sidecar.py --source both

app-dev:
	DATA_DIR="$(PWD)/data" LLM_URL="$${LLM_URL:-http://127.0.0.1:8081}" \
		./.venv-app/bin/python -m uvicorn stenopod.main:app --app-dir app --reload --host 127.0.0.1 --port 8780
