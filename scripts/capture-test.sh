#!/usr/bin/env bash
# Record 3 seconds from BlackHole and report whether system audio is actually arriving.
set -euo pipefail

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. brew install ffmpeg" >&2
  exit 1
fi

OUT="${TMPDIR:-/tmp}/stenopod-capture-test.wav"
echo "Recording 3s from BlackHole 2ch …"
if ! ffmpeg -y -hide_banner -loglevel error -f avfoundation -i ":BlackHole 2ch" -t 3 -ac 1 -ar 16000 "$OUT"; then
  echo "Could not open BlackHole. Quit extra sidecar/ffmpeg processes, and allow Microphone access for Terminal (System Settings → Privacy & Security → Microphone)." >&2
  exit 1
fi

python3 - "$OUT" <<'PY'
import struct, sys, wave
path = sys.argv[1]
with wave.open(path) as w:
    n = w.getnframes()
    samples = struct.unpack("<" + "h" * n, w.readframes(n))
peak = max((abs(x) for x in samples), default=0)
rms = (sum(x * x for x in samples) / max(len(samples), 1)) ** 0.5
print(f"peak={peak/32768:.4f}  rms={rms/32768:.4f}")
if peak < 50:
    print("SILENT. YouTube is not reaching BlackHole.")
    print("Audio MIDI Setup → Multiausgangsgerät → enable BlackHole 2ch AND your speakers/monitor.")
    print("System Settings → Sound → Output → Multiausgangsgerät, then pause/play the video.")
else:
    print("Audio is reaching BlackHole. Restart: make sidecar")
PY
