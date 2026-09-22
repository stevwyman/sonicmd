# sonicmd

StenoPod captures whatever is playing on your Mac, transcribes it locally with **Metal Whisper**, and folds the rolling transcript into **markdown notes** with a small LLM running in **Podman (LibKrun / Vulkan)**.

There is no hook into Zoom, Teams, or the browser. The sidecar listens to a virtual audio device (and optionally a file). English and German are both handled by multilingual Whisper; notes follow the dominant language of the speech.

```
Mac speakers / headphones ─┐
Apps  →  Multi-Output  ────┼─► BlackHole 2ch ─► host sidecar (mlx-whisper, Metal)
Mic  ──────────────────────┘                         │
                                                     ▼ JSON segments
                                            stenopod-app :8780
                                                     │
                                            stenopod-llm :8081
                                            llama.cpp Vulkan via /dev/dri
```

## What runs where

| Piece | Where | Why |
|---|---|---|
| BlackHole + ffmpeg | macOS | Containers cannot see Core Audio |
| Whisper `large-v3-turbo` | macOS Metal (`mlx-whisper`) | Live EN/DE captions |
| Web UI + session store | Podman `stenopod-app` | Transcript + notes at http://127.0.0.1:8780 |
| Qwen2.5-3B Instruct Q4 | Podman `stenopod-llm` | Markdown summaries on the LibKrun GPU |

On a 16 GB M1, keep the Podman machine at **8 GB**. Whisper stays on the host so the VM is free for the 3B notes model.

## Prerequisites

- Apple Silicon Mac
- [Podman Desktop](https://podman-desktop.io/) with **LibKrun** / Default GPU (you already have this)
- [Homebrew](https://brew.sh)
- Python 3.11+

```bash
brew install ffmpeg
brew install --cask blackhole-2ch
```

### Route system audio into BlackHole

1. Open **Audio MIDI Setup**.
2. Click **+** → **Create Multi-Output Device**.
3. Enable your speakers/headphones **and** **BlackHole 2ch**.
4. Enable **Drift Correction** on BlackHole.
5. Set that multi-output as the Mac’s sound output so you still hear audio while StenoPod records a copy.

List capture devices any time with `make devices`. The sidecar picks BlackHole automatically if it is present.

### Meetings (Webex / Google Meet): add your microphone

Remote people arrive through the speakers → BlackHole. Your own voice only arrives if we also capture the mic.

```bash
make sidecar-meet
# same as: ./.venv-host/bin/python host/sidecar.py --source both
```

Use **headphones** so the mic does not re-record the other side. pyannote still splits speakers (you vs them) as A/B — rename your chip after the first turns.

If Meet/Webex has exclusive lock on the microphone, ffmpeg will fail to open it. Quit extra capture processes, or in Audio MIDI Setup create an **Aggregate Device** of BlackHole + your mic and run:

```bash
./.venv-host/bin/python host/sidecar.py --source system --device "StenoPod In"
```

Pick the mic explicitly with `--mic-device "MacBook Air-Mikrofon"` if `make devices` shows several.

## Run

```bash
# 1. Notes model (~2 GB, once)
make models

# 2. App + llama.cpp in Podman (needs the LibKrun machine running)
make up

# 3. Metal Whisper + pyannote sidecar on the Mac
export HF_TOKEN=hf_...          # see Hugging Face steps below
make sidecar-venv
make sidecar
```

Open [http://127.0.0.1:8780](http://127.0.0.1:8780).

Play anything through the multi-output. **pyannote.audio** (local `community-1` pipeline) cuts speaker turns; labels stay **Speaker A, B, C…** across windows. Click a chip to rename. Notes update after a pause, after enough new text, or when you click **Update notes**.

`Ctrl+C` on the sidecar stops capture, transcribes whatever is still in the buffer, then exits. Press `Ctrl+C` again if you need to abort immediately.

Capture stays on the main thread; pyannote and Whisper run on a worker so ffmpeg is not paused during inference. Each window logs `timing … pyannote Xs / whisper Ys`. If inference falls more than four windows behind live audio, the oldest queued window is dropped.

### Hugging Face token (required for pyannote)

1. Accept the model terms: <https://huggingface.co/pyannote/speaker-diarization-community-1>
2. Create a read token: <https://huggingface.co/settings/tokens>
3. `export HF_TOKEN=hf_...` in the shell that runs `make sidecar`

Without a token, use the lighter embedder instead: `./.venv-host/bin/python host/sidecar.py --diarize ecapa`

Transcribe a file instead of live audio:

```bash
./.venv-host/bin/python host/sidecar.py --file ~/interview.wav
```

Stop the stack with `make down`. Logs: `make logs`.

## Configuration

| Knob | Default | Meaning |
|---|---|---|
| `host/sidecar.py --device` | BlackHole | AVFoundation name or index |
| `host/sidecar.py --model` | `mlx-community/whisper-large-v3-turbo` | MLX Whisper repo |
| `host/sidecar.py --source` | `system` | `system` (BlackHole), `mic`, or `both` (meetings) |
| `--mic-device` | built-in mic | Microphone name or index |
| `LLM_URL` | `http://llm:8080` | llama.cpp server inside compose |
| `AUTO_SUMMARY_CHARS` | `1800` | Fold notes when this much new text lands |
| `AUTO_SUMMARY_SECONDS` | `90` | Fold notes after this much quiet |
| `--diarize` | `pyannote` | `pyannote` (local) or `ecapa` |
| `--num-speakers` | unset | Optional hint for pyannote |
| `--min-seconds` / `--max-seconds` | `6` / `18` | Live window size for pyannote |

Swap the GGUF in `compose.yaml` if you later try Qwen2.5-7B Q4. That is tighter on an 8 GB VM.

## Troubleshooting

**Sidecar lamp stays dark.** The app is up but `make sidecar` is not running, or it cannot reach `http://127.0.0.1:8780`.

**No speech in the transcript.** System output is not the multi-output device, or ffmpeg is capturing the wrong index (`make devices`, then `--device 1`). Raise `--vad` if fan noise triggers junk, lower it if quiet speech is dropped.

**LLM lamp is red.** The GGUF is missing (`make models`) or `llama-server` failed to start. Check `podman logs stenopod-llm`. Confirm the machine is LibKrun and the container has `/dev/dri`.

**GPU not used.** Inside the VM this is Vulkan forwarded to Metal, not native Metal. `llama.cpp` must run with `-ngl 99`. CPU fallback will still summarize, just slowly.

**Everything is Speaker A.** The sidecar now keeps pyannote’s per-voice embeddings across windows (ads vs host, you vs them). Restart `make sidecar` after updating. English vs German is a hint, not a hard split — a bilingual speaker can stay one label if the voice match is strong. Rename chips when you know who is who.

**Overlapping speech.** pyannote.audio segments turns locally (`community-1` on MPS when available). Overlap and heavy compression still confuse labels.

**HF token errors.** Accept the community-1 terms and export `HF_TOKEN`. First sidecar start downloads the pipeline.

**Out of memory.** pyannote (PyTorch) and Whisper both sit on the Mac. If the machine swaps, shrink the Podman VM while capturing, or use `--diarize ecapa`.

## Layout

```
app/           FastAPI UI + incremental notes
host/sidecar.py  AVFoundation capture + mlx-whisper + pyannote.audio
models/        GGUF (gitignored)
data/          session JSON
compose.yaml   app + ramalama llama.cpp (Vulkan)
```
