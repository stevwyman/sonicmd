.PHONY: help models devices up down logs status sidecar sidecar-meet sidecar-venv app-venv capture-test

# Extra sidecar flags, e.g.
#   make sidecar-meet MIC_DEVICE='MacBook Air-Mikrofon'
#   make sidecar ARGS='--source both --mic-device "MacBook Air-Mikrofon"'
ARGS ?=
MIC_DEVICE ?=
mic_flag = $(if $(MIC_DEVICE),--mic-device "$(MIC_DEVICE)")

help:
	@echo "StenoPod (native macOS, local MLX notes)"
	@echo "  make models        Download Whisper, pyannote 3.1, and Qwen MLX (once, online)"
	@echo "  make app-venv      Create app venv (FastAPI)"
	@echo "  make sidecar-venv  Create host venv (Whisper + pyannote.audio + MLX notes)"
	@echo "  make up            Start the web app"
	@echo "  make down          Stop the app (and capture if the UI started it)"
	@echo "  make logs          Follow app + sidecar logs"
	@echo "  make status        Show app / sidecar"
	@echo "  make devices       List macOS audio capture devices"
	@echo "  make capture-test  Record 3s from BlackHole, then 3s from the Mac mic (speak during the mic part)"
	@echo "  make sidecar       Capture system audio (BlackHole) from the terminal"
	@echo "  make sidecar-meet  Stereo: left Mac mic, right BlackHole (or use Start capture in the UI)"
	@echo "  Extra sidecar flags: make sidecar-meet MIC_DEVICE='MacBook Air-Mikrofon'"

models:
	@if [ ! -x .venv-host/bin/python ]; then $(MAKE) sidecar-venv; fi
	.venv-host/bin/python scripts/download-models.py

app-venv:
	python3 -m venv .venv-app
	.venv-app/bin/pip install -U pip
	.venv-app/bin/pip install -r app/requirements.txt

sidecar-venv:
	python3 -m venv .venv-host
	.venv-host/bin/pip install -U pip
	.venv-host/bin/pip install -r host/requirements.txt

capture-test:
	./scripts/capture-test.sh

up:
	@if [ ! -x .venv-app/bin/python ]; then $(MAKE) app-venv; fi
	./scripts/stack.sh start

down:
	./scripts/stack.sh stop

logs:
	./scripts/stack.sh logs

status:
	./scripts/stack.sh status

devices:
	./.venv-host/bin/python host/sidecar.py --list-devices

sidecar:
	./.venv-host/bin/python host/sidecar.py $(ARGS) $(mic_flag)

sidecar-meet:
	./.venv-host/bin/python host/sidecar.py --source both $(ARGS) $(mic_flag)
