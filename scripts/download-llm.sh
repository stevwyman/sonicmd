#!/usr/bin/env bash
# Download Whisper, pyannote 3.1, segmentation-3.0, and the MLX Qwen notes model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv-host/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing host venv. Run: make sidecar-venv" >&2
  exit 1
fi

exec "$PY" "$ROOT/scripts/download-models.py"
