#!/usr/bin/env bash
# Record 3 seconds from BlackHole and from the Mac microphone.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv-host/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Host venv missing. Run: make sidecar-venv" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. brew install ffmpeg" >&2
  exit 1
fi
exec "$PY" "$ROOT/host/sidecar.py" --capture-test
