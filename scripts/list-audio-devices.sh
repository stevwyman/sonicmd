#!/usr/bin/env bash
# List macOS AVFoundation capture devices (ffmpeg prints them on stderr).
set -euo pipefail

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install with: brew install ffmpeg" >&2
  exit 1
fi

ffmpeg -hide_banner -f avfoundation -list_devices true -i "" || true
