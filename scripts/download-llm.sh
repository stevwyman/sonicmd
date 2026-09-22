#!/usr/bin/env bash
# Download the default notes model into ./models (Qwen2.5-3B Instruct, Q4_K_M).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT/models"
FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
DEST="$DEST_DIR/$FILE"
URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/${FILE}"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST" ]]; then
  echo "Already present: $DEST"
  ls -lh "$DEST"
  exit 0
fi

echo "Downloading $FILE (~2.0 GB) …"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$DEST.partial" "$URL"
else
  echo "curl is required" >&2
  exit 1
fi
mv "$DEST.partial" "$DEST"
ls -lh "$DEST"
echo "Done. Start the stack with: podman compose up --build"
