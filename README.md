# sonicmd

StenoPod captures whatever is playing on your Mac, transcribes it locally with **Metal Whisper**, and folds the rolling transcript into **markdown notes** with a local **MLX Qwen** process on this Mac. No Podman, no Ollama, no extra llama-server.

There is no hook into Zoom, Teams, or the browser. The sidecar listens to a virtual audio device (and optionally a file). English and German are both handled by multilingual Whisper; notes follow the dominant language of the speech.

```
Speakers:   Meet → StenoPod Speakers → BlackHole ──────────────► one channel (meeting)
Headphones: Meet → StenoPod Headphones → BlackHole ── right ──► stereo wav
            MacBook mic ───────────────────────────── left  ──► sidecar
                                                     ▼ JSON segments
                                            FastAPI UI :8780
                                                     │
                                            MLX Qwen (local process, no server)
```

## What runs where

| Piece | Where | Why |
|---|---|---|
| BlackHole + ffmpeg | macOS | Core Audio is host-only |
| Whisper `large-v3-turbo` | macOS Metal (`mlx-whisper`) | Transcript after recording stops |
| Web UI + session store | `make up` → FastAPI on :8780 | Transcript + notes + Start/Stop |
| Qwen2.5 7B Instruct, MLX 4-bit | `mlx-lm` in the host venv | Markdown summaries |

On a 16 GB Mac, Whisper, pyannote, and Qwen share unified memory. Notes load after transcription finishes. Keep other heavy apps closed while capturing.

## Prerequisites

- Apple Silicon Mac
- [Homebrew](https://brew.sh)
- Python 3.11+

```bash
brew install ffmpeg
brew install --cask blackhole-2ch
```

### Route system audio into BlackHole

A Multi-Output Device plays to **every box you tick, at the same time**. If Mac speakers and BlackHole are both ticked, the speakers stay on even after you connect Bluetooth headphones. Connecting headphones does not change that list.

Make **two** Multi-Outputs and switch between them. Do not put speakers and headphones in the same one.

1. Connect the Bluetooth headphones first. Open **Audio MIDI Setup** (`/Applications/Utilities/Audio MIDI Setup.app`).
2. Select **BlackHole 2ch**, **MacBook speakers**, and the headphones. Set each to **48 000 Hz**, 2 channels.
3. **+** → **Create Multi-Output Device**. Rename it `StenoPod Speakers`. Tick **only** MacBook speakers + **BlackHole 2ch**. Enable **Drift Correction** on BlackHole.
4. **+** → **Create Multi-Output Device** again. Rename it `StenoPod Headphones`. Tick **only** the Bluetooth headphones + **BlackHole 2ch**. Drift Correction on BlackHole again.
5. In **System Settings → Sound** (not in Meet — Meet’s “system audio” follows this):
   - **Output** = `StenoPod Headphones` or `StenoPod Speakers`
   - **Input** = **MacBook Air-Mikrofon** (not AirPods)
6. Never pick the headphones or speakers as Output by themselves — then BlackHole gets silence and StenoPod records nothing.

That two-device layout is correct: you only switch where **you** hear audio; BlackHole is ticked on both, so the sidecar should always get a copy. The sidecar still records **BlackHole**, not the Multi-Output name. If you hear Meet in the buds but BlackHole levels stay ~0, the Multi-Output is not actually copying (common with AirPods). Restart Chrome/Meet after switching Output, confirm 48 kHz and Drift Correction on BlackHole, and that BlackHole is not muted in Audio MIDI Setup when `StenoPod Headphones` is selected.

`make sidecar` records **only BlackHole**, one channel (YouTube, or a meeting you are not speaking in). Your voice needs `make sidecar-meet`, which writes a stereo file: left is the Mac mic, right is BlackHole.

**AirPods cannot be the listen device and the talk device at the same time.** If Input is AirPods, macOS switches them to Hands-Free: the Multi-Output copy into BlackHole goes silent, which is why the sidecar hears nothing. Keep Input on the MacBook mic.

To switch output: change the Sound **Output** picker. Leave the sidecar running.

Bluetooth in a Multi-Output can glitch (dropouts, device missing until you reconnect). If the headphones do not appear as a tick box, disconnect/reconnect them, or use a wired jack. Sample rate mismatches (44.1 vs 48 kHz) are the usual cause of crackle.

List capture devices with `make devices`. The sidecar picks BlackHole automatically if it is present.

### Meetings (Webex / Google Meet): add your microphone

Remote people arrive through the Multi-Output → BlackHole. Your own voice only arrives if we also capture a mic.

```bash
make sidecar-meet MIC_DEVICE='MacBook Air-Mikrofon'
# same as:
./.venv-host/bin/python host/sidecar.py --source both --mic-device "MacBook Air-Mikrofon"
```

There is no `sidecar` binary. `make sidecar --source both` also does not work (Make eats the flags). `--source both` keeps the two sides apart in one stereo file: **left** is the Mac microphone, **right** is BlackHole. Transcription reads each side on its own. `--source system` (and the Speakers choice in the UI) records BlackHole only.

Prefer the **MacBook microphone** while you *listen* on Bluetooth. Do not set Sound → Input to AirPods.

If you select the headphone mic as the Mac’s input, macOS switches the headset to Hands-Free: playback gets thin and BlackHole usually goes silent. Keep **Input** on the MacBook mic, **Output** on `StenoPod Headphones`.

There is no reliable way to use the AirPods mic with this stack. `make devices` may show `AirPods Pro - Find My`; that is not a microphone — ignore it.

Use headphones for the stereo recording so the Mac mic does not also hear the meeting. The left channel is you; the right channel is everyone else. Rename a speaker chip after the first turns if you want a name instead of a letter.

If Meet/Webex has exclusive lock on the microphone, ffmpeg will fail to open it. Quit extra capture processes, or in Audio MIDI Setup create an **Aggregate Device** of BlackHole + your mic and run:

```bash
./.venv-host/bin/python host/sidecar.py --source system --device "StenoPod In"
```

## Run

```bash
# 1. Python envs (once)
make app-venv
make sidecar-venv
cp .env.example .env            # then put your Hugging Face token in .env

# 2. Download models once, while online (Whisper, pyannote 3.1, segmentation-3.0, Qwen MLX)
make models

# 3. Web UI. Later runs stay offline (HF_HUB_OFFLINE=1).
make up
```

Open [http://127.0.0.1:8780](http://127.0.0.1:8780). Press **New meeting**, choose where you are listening, then **Start recording**.

- **StenoPod Speakers · meeting only** records one channel: what is playing through the Mac speakers. Set Sound → Output to `StenoPod Speakers`. The microphone is not opened.
- **StenoPod Headphones · you + meeting** records stereo. Left is the Mac microphone, right is what you hear in the headphones. Set Sound → Output to `StenoPod Headphones`, and Input to the Mac microphone.

While recording, Transcribe stays off. **Stop recording** closes the wav under `.run/captures/` and starts transcription on its own: pyannote labels speakers on the whole recording, Whisper writes each turn with its language, then notes are written. Min and Max next to the source choose the speaker count (leave them blank for automatic, or set them equal for an exact count). Those values are saved with the meeting and used again by **Transcribe again**. The **Recordings** list opens any saved wav and can rerun speakers, transcript, and notes. **Download** saves the transcript or the notes. Rename a speaker to store that voice; a later meeting can reuse the name.

You can still start capture from a terminal with `make sidecar` or `make sidecar-meet` if you want the logs in that window. The web app is the path that transcribes after stop.

`Ctrl+C` / **Stop recording** closes the wav. Retry a saved wav with `./.venv-host/bin/python host/sidecar.py --file .run/captures/stenopod-….wav`.

### Hugging Face token (required for speakers)

1. Accept the model terms: <https://huggingface.co/pyannote/speaker-diarization-3.1> and <https://huggingface.co/pyannote/segmentation-3.0>
2. Create a read token: <https://huggingface.co/settings/tokens>
3. Copy `.env.example` to `.env` and set `HF_TOKEN=hf_...`. That file is gitignored. **Start recording**, `make sidecar`, and `host/sidecar.py` load it automatically.

Without a token, Transcribe cannot run. Speaker labels come from pyannote on the whole recording.

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
| `host/sidecar.py --source` | `system` | `system` or `speakers` (BlackHole only), `mic`, or `both` (stereo: left mic, right BlackHole) |
| `--mic-device` | built-in mic | Microphone name or index |
| `HF_HUB_OFFLINE` | `1` | Inference uses the local cache only. `make models` is the online step. |
| `LLM_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Local MLX notes model |
| `--min-speakers` / `--max-speakers` | none | Header Min/Max. Equal values are an exact count. The transcript will not exceed Max. |

`Qwen/Qwen2.5-7B-Instruct-MLX` is the name in the Qwen docs, and that repo is gone. The default is the MLX 4-bit conversion of the same instruct model.

## Troubleshooting

**Sidecar lamp stays dark.** Press **Start recording**, or run `make sidecar`. The app must be up at http://127.0.0.1:8780. Model download happens when you press **Transcribe**, and the first run can take a few minutes.

**No speech in the transcript.** System output is not the multi-output device, or ffmpeg is capturing the wrong index (`make devices`, then `--device 1`). Capture can be running with a silent BlackHole — check the sidecar log for levels. Stop recording (transcription starts on its own), or press **Transcribe** on that row.

**LLM lamp is red.** The Qwen weights are not in the Hugging Face cache. Run `make models` once while online.

**Too many speakers / hopping A/B mid-sentence.** Set Min and Max before **Transcribe again**. An exact count is Min and Max set to the same number. pyannote labels the whole wav once, Whisper writes each turn, and the transcript is kept inside that maximum.

**Overlapping speech.** Speakers come from `pyannote/segmentation-3.0` with the `speaker-diarization-3.1` clustering settings, on MPS when available. Overlap and heavy compression still confuse labels.

**HF token errors.** Accept the 3.1 and segmentation-3.0 terms, put `HF_TOKEN` in `.env`, and run `make models`. Transcription does not download models.

**Out of memory.** pyannote (PyTorch), Whisper, and the MLX notes model all sit on the Mac. Notes start after Whisper has exited. If the machine swaps, close other apps.

## Layout

```
app/                    FastAPI UI, session store, notes
app/static/             Web UI
host/sidecar.py         AVFoundation capture (wav) + control API :8781
host/chunk_transcribe.py  pyannote on the full wav, then Whisper
host/pyannote_diarize.py  Speaker diarization
host/notes_llm.py       Local MLX Qwen notes
host/hf_offline.py      Cached Hugging Face models
scripts/stack.sh        Start/stop the web app
scripts/download-models.py  One-time model download
data/                   Session JSON (gitignored except .gitkeep)
.env                    Local secrets (`HF_TOKEN`; gitignored)
.run/                   Logs, pid files, and capture wavs
```
