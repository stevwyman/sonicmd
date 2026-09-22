#!/usr/bin/env python3
"""Capture macOS system audio (or a file) and transcribe it with Metal Whisper."""

from __future__ import annotations

import argparse
import math
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np

HOST_DIR = Path(__file__).resolve().parent
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from diarize import SpeakerBank, load_embedder
from pyannote_diarize import PyannoteDiarizer, SpeakerTurn

SAMPLE_RATE = 16000
FRAME_MS = 30
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
MIN_FLUSH_SAMPLES = SAMPLE_RATE // 2
INFER_QUEUE_MAX = 4
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_audio_devices() -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    text = proc.stderr or proc.stdout
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in text.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if "AVFoundation video devices" in line:
            in_audio = False
            continue
        if not in_audio:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def pick_device(requested: str | None) -> tuple[str, str]:
    devices = list_audio_devices()
    if not devices:
        raise SystemExit(
            "No AVFoundation audio devices found. Is ffmpeg installed? brew install ffmpeg"
        )
    if requested:
        if requested.isdigit():
            for idx, name in devices:
                if idx == int(requested):
                    return str(idx), name
        needle = requested.lower()
        for idx, name in devices:
            if needle in name.lower():
                return str(idx), name
        raise SystemExit(f"No audio device matching {requested!r}. Found: {devices}")

    for idx, name in devices:
        if "blackhole" in name.lower():
            return str(idx), name
    idx, name = devices[0]
    print(f"BlackHole not found; using {name!r} [{idx}]", file=sys.stderr)
    return str(idx), name


def pick_mic_device(requested: str | None) -> tuple[str, str]:
    devices = list_audio_devices()
    skip = ("blackhole", "multi-output", "multiausgang", "teams audio")
    if requested:
        return pick_device(requested)
    for idx, name in devices:
        low = name.lower()
        if any(s in low for s in skip):
            continue
        if any(s in low for s in ("mikrofon", "microphone", "mic")):
            return str(idx), name
    for idx, name in devices:
        if not any(s in name.lower() for s in skip):
            return str(idx), name
    raise SystemExit("No microphone found. Run make devices and pass --mic-device.")


FFMPEG_SHUTDOWN_MARKERS = (
    "Immediate exit requested",
    "Error submitting a packet to the muxer",
    "Error muxing a packet",
    "Error writing trailer",
    "Error closing file",
    "Terminating thread with return code",
    "Task finished with error code: -1414092869",
    "Last message repeated",
)


def ffmpeg_message_is_shutdown(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker in stripped for marker in FFMPEG_SHUTDOWN_MARKERS):
            continue
        return False
    return True


def stop_ffmpeg(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass


def ffmpeg_pcm(device_index: str, mic_index: str | None = None) -> subprocess.Popen[bytes]:
    if mic_index is None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "avfoundation",
            "-i",
            f":{device_index}",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "avfoundation",
            "-i",
            f":{device_index}",
            "-f",
            "avfoundation",
            "-i",
            f":{mic_index}",
            "-filter_complex",
            "[0:a]aresample=16000,aformat=channel_layouts=mono[bh];"
            "[1:a]aresample=16000,aformat=channel_layouts=mono[mic];"
            "[bh][mic]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]",
            "-map",
            "[a]",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _stderr() -> None:
        assert proc.stderr is not None
        err = proc.stderr.read().decode("utf-8", errors="replace").strip()
        if err and not ffmpeg_message_is_shutdown(err):
            print(f"ffmpeg error:\n{err}", file=sys.stderr)

    threading.Thread(target=_stderr, daemon=True).start()
    threading.Event().wait(0.8)
    if proc.poll() is not None:
        raise SystemExit(
            "ffmpeg could not open the capture device. "
            "BlackHole or the mic may be locked (quit extra sidecar/ffmpeg, or the meeting app may have exclusive mic access). "
            "Allow Microphone in System Settings → Privacy & Security."
        )
    return proc


def frames_from_proc(proc: subprocess.Popen[bytes]) -> Iterator[np.ndarray]:
    assert proc.stdout is not None
    nbytes = SAMPLES_PER_FRAME * 2
    first = True
    while True:
        raw = proc.stdout.read(nbytes)
        if not raw:
            if first:
                print("No audio frames from ffmpeg — device busy or capture failed.", file=sys.stderr)
            break
        if len(raw) < nbytes:
            if len(raw) >= 2:
                if len(raw) % 2:
                    raw = raw[:-1]
                yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            break
        frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if first:
            peak = float(np.max(np.abs(frame)))
            print(f"Capture started (first-frame peak {peak:.4f}).", flush=True)
            first = False
        yield frame


def decode_file(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode("utf-8", errors="replace"))
    return np.frombuffer(proc.stdout, dtype=np.float32)


class EnergyVAD:
    def __init__(self, start_frames: int = 3, stop_frames: int = 18, threshold: float = 0.012) -> None:
        self.start_frames = start_frames
        self.stop_frames = stop_frames
        self.threshold = threshold
        self.voiced = 0
        self.silence = 0
        self.in_speech = False

    def push(self, frame: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        if rms >= self.threshold:
            self.voiced += 1
            self.silence = 0
        else:
            self.silence += 1
            self.voiced = 0

        if not self.in_speech and self.voiced >= self.start_frames:
            self.in_speech = True
            return "start"
        if self.in_speech and self.silence >= self.stop_frames:
            self.in_speech = False
            self.voiced = 0
            return "end"
        return "speech" if self.in_speech else "silence"


class AppClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=10.0)
        self._lock = threading.Lock()

    def heartbeat(self, listening: bool, device: str | None, source: str, model: str) -> None:
        try:
            with self._lock:
                self.http.post(
                    f"{self.base_url}/api/sidecar",
                    json={
                        "listening": listening,
                        "device": device,
                        "source": source,
                        "model": model,
                    },
                )
        except httpx.HTTPError as exc:
            print(f"sidecar heartbeat failed: {exc}", file=sys.stderr)

    def send_segment(
        self,
        text: str,
        lang: str,
        source: str,
        speaker: str,
        duration_ms: int,
        speaker_score: float | None = None,
    ) -> None:
        payload = {
            "text": text,
            "lang": lang,
            "source": source,
            "t": iso_now(),
            "speaker": speaker,
            "duration_ms": int(duration_ms),
            "speaker_score": None,
        }
        if speaker_score is not None:
            score = float(speaker_score)
            if math.isfinite(score):
                payload["speaker_score"] = score
        with self._lock:
            r = self.http.post(f"{self.base_url}/api/segments", json=payload)
        if r.is_error:
            detail = r.text
            raise httpx.HTTPStatusError(
                f"{r.status_code} posting segment: {detail}",
                request=r.request,
                response=r,
            )


def transcribe_audio(audio: np.ndarray, model: str) -> tuple[str, str]:
    import mlx_whisper

    if audio.size < SAMPLE_RATE // 2:
        return "", "und"
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model,
        word_timestamps=False,
    )
    text = (result.get("text") or "").strip()
    lang = result.get("language") or "und"
    return text, lang


def turn_embedding(turn: SpeakerTurn, embedder: object) -> np.ndarray:
    if turn.embedding is not None:
        vec = np.asarray(turn.embedding, dtype=np.float64).reshape(-1)
        if vec.size and float(np.linalg.norm(vec)) > 1e-6:
            return vec
    return embedder.embed(turn.audio)


def process_clip(
    audio: np.ndarray,
    client: AppClient,
    whisper_model: str,
    source: str,
    bank: SpeakerBank,
    embedder: object,
    pipeline: PyannoteDiarizer | None,
) -> None:
    audio_s = audio.size / SAMPLE_RATE if audio.size else 0.0
    t0 = time.perf_counter()
    if pipeline is not None:
        turns = pipeline.segment(audio)
    else:
        turns = [SpeakerTurn(audio, "utt", None, 0.0, audio_s)]
    diarize_s = time.perf_counter() - t0

    window_map: dict[str, str] = {}
    used: set[str] = set()
    whisper_s = 0.0
    n_text = 0
    for turn in turns:
        try:
            tw = time.perf_counter()
            text, lang = transcribe_audio(turn.audio, whisper_model)
            whisper_s += time.perf_counter() - tw
            if not text:
                continue
            n_text += 1
            vec = turn_embedding(turn, embedder)
            if turn.local_id in window_map:
                speaker = window_map[turn.local_id]
                score = bank.reinforce(speaker, vec, lang=lang)
            else:
                speaker, score = bank.assign(vec, lang=lang, forbidden=used)
                window_map[turn.local_id] = speaker
                used.add(speaker)
            duration_ms = int(round(1000 * turn.audio.size / SAMPLE_RATE))
            sim = f"{score:.2f}" if math.isfinite(float(score)) else "?"
            print(f"[{speaker} {lang} {duration_ms}ms sim={sim}] {text}")
            client.send_segment(text, lang, source, speaker, duration_ms, score)
        except httpx.HTTPError as exc:
            print(f"failed to post segment: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"utterance failed ({type(exc).__name__}): {exc}", file=sys.stderr)

    n_spk = len({turn.local_id for turn in turns}) if turns else 0
    print(
        f"  timing {audio_s:.1f}s audio: pyannote {diarize_s:.2f}s / whisper {whisper_s:.2f}s "
        f"({len(turns)} turn{'s' if len(turns) != 1 else ''}, {n_spk} speaker{'s' if n_spk != 1 else ''}, {n_text} posted)",
        flush=True,
    )


class InferenceWorker:
    """Capture stays on the main thread; pyannote + Whisper run here."""

    def __init__(
        self,
        client: AppClient,
        whisper_model: str,
        source: str,
        bank: SpeakerBank,
        embedder: object,
        pipeline: PyannoteDiarizer | None,
    ) -> None:
        self.client = client
        self.whisper_model = whisper_model
        self.source = source
        self.bank = bank
        self.embedder = embedder
        self.pipeline = pipeline
        self.q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=INFER_QUEUE_MAX)
        self.ready = threading.Event()
        self.warmup_error: str | None = None
        self.thread = threading.Thread(target=self._loop, name="stenopod-infer", daemon=True)

    def start(self) -> None:
        print("Warming up Metal Whisper (first run downloads the model) …")
        self.thread.start()
        if not self.ready.wait(timeout=300):
            raise SystemExit("Whisper warmup timed out.")
        if self.warmup_error:
            raise SystemExit(self.warmup_error)

    def submit(self, audio: np.ndarray) -> None:
        clip = np.array(audio, dtype=np.float32, copy=True)
        while True:
            try:
                self.q.put_nowait(clip)
                pending = self.q.qsize()
                if pending > 1:
                    print(f"  queued {pending} windows", flush=True)
                return
            except queue.Full:
                try:
                    dropped = self.q.get_nowait()
                    self.q.task_done()
                    if dropped is None:
                        self.q.put_nowait(None)
                        print("  dropped live window (shutting down)", flush=True)
                        return
                    print(
                        f"  dropped {dropped.size / SAMPLE_RATE:.1f}s queued window (inference behind live)",
                        flush=True,
                    )
                except queue.Empty:
                    pass

    def close(self) -> None:
        if not self.thread.is_alive():
            return
        try:
            self.q.put(None, timeout=60)
        except queue.Full:
            print("Inference worker did not accept stop; skipping drain.", file=sys.stderr)
            return
        self.thread.join(timeout=180)
        if self.thread.is_alive():
            print("Inference still running; leftover windows may be skipped.", file=sys.stderr)

    def _loop(self) -> None:
        try:
            transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32), self.whisper_model)
        except Exception as exc:
            self.warmup_error = f"Whisper warmup failed ({type(exc).__name__}): {exc}"
            self.ready.set()
            return
        self.ready.set()
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                process_clip(
                    item,
                    self.client,
                    self.whisper_model,
                    self.source,
                    self.bank,
                    self.embedder,
                    self.pipeline,
                )
            except Exception as exc:
                print(f"inference failed ({type(exc).__name__}): {exc}", file=sys.stderr)
            finally:
                self.q.task_done()


def emit_window(audio: np.ndarray, on_window, reason: str = "") -> None:
    if audio.size < MIN_FLUSH_SAMPLES:
        return
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    note = f" ({reason})" if reason else ""
    print(f"Window {audio.size / SAMPLE_RATE:.1f}s peak={peak:.4f}{note}", flush=True)
    if peak < 0.008:
        print("  (near silence — is the Multi-Output sending audio to BlackHole?)", flush=True)
        return
    on_window(audio)


def consume_windows(
    frames: Iterator[np.ndarray],
    vad: EnergyVAD,
    min_samples: int,
    max_samples: int,
    silence_frames: int,
    on_window,
) -> None:
    buf: list[np.ndarray] = []
    silent = 0
    heard = False
    try:
        for frame in frames:
            buf.append(frame)
            event = vad.push(frame)
            if event == "silence":
                silent += 1
            else:
                heard = True
                silent = 0
            n = sum(len(x) for x in buf)
            flush = False
            if n >= max_samples:
                flush = True
            elif heard and n >= min_samples and silent >= silence_frames:
                flush = True
            if flush:
                audio = np.concatenate(buf)
                buf = []
                heard = False
                silent = 0
                vad.in_speech = False
                vad.voiced = 0
                vad.silence = 0
                emit_window(audio, on_window)
    finally:
        leftover = np.concatenate(buf) if buf and heard else None
        if leftover is not None:
            print("Flushing leftover capture …", flush=True)
            emit_window(leftover, on_window, reason="stop")


def consume_frames(frames: Iterator[np.ndarray], vad: EnergyVAD, max_samples: int, on_utt) -> None:
    preroll: deque[np.ndarray] = deque(maxlen=10)
    uttered: list[np.ndarray] = []
    try:
        for frame in frames:
            event = vad.push(frame)
            if event == "silence":
                preroll.append(frame)
                continue
            if event == "start":
                uttered = list(preroll) + [frame]
                preroll.clear()
                continue
            uttered.append(frame)
            if event == "end" or (uttered and sum(len(x) for x in uttered) >= max_samples):
                audio = np.concatenate(uttered)
                uttered = []
                vad.in_speech = False
                try:
                    on_utt(audio)
                except httpx.HTTPError as exc:
                    print(f"failed to post segment: {exc}", file=sys.stderr)
    finally:
        if uttered and sum(len(x) for x in uttered) >= MIN_FLUSH_SAMPLES:
            print("Flushing leftover capture …", flush=True)
            try:
                on_utt(np.concatenate(uttered))
            except httpx.HTTPError as exc:
                print(f"failed to post segment: {exc}", file=sys.stderr)


def frames_from_array(audio: np.ndarray) -> Iterator[np.ndarray]:
    for i in range(0, max(audio.size - SAMPLES_PER_FRAME, 0) + 1, SAMPLES_PER_FRAME):
        yield audio[i : i + SAMPLES_PER_FRAME]


def heartbeat_loop(client: AppClient, stop: threading.Event, **kwargs: str | None) -> None:
    while not stop.wait(3.0):
        client.heartbeat(True, kwargs.get("device"), kwargs["source"], kwargs["model"])
    client.heartbeat(False, kwargs.get("device"), kwargs["source"], kwargs["model"])


def prepare_speakers(args: argparse.Namespace) -> tuple[SpeakerBank, object, PyannoteDiarizer | None]:
    embedder, kind = load_embedder()
    print(f"Speaker embeddings: {kind}")
    pipeline = PyannoteDiarizer(num_speakers=args.num_speakers) if args.diarize == "pyannote" else None
    return SpeakerBank(), embedder, pipeline


def run_live(args: argparse.Namespace, client: AppClient) -> None:
    source = "both" if args.mic else args.source
    if source == "mic":
        device_index, device_name = pick_mic_device(args.mic_device)
        mic_index = None
        sys_index = device_index
    elif source == "both":
        sys_index, sys_name = pick_device(args.device)
        mic_index, mic_name = pick_mic_device(args.mic_device)
        device_name = f"{sys_name} + {mic_name}"
        print(f"Mixing system {sys_name!r} [{sys_index}] with mic {mic_name!r} [{mic_index}]")
    else:
        sys_index, device_name = pick_device(args.device)
        mic_index = None

    print(f"Will listen on {device_name!r} ({source}) → {args.app}")
    print(f"Whisper model: {args.model}")
    print(f"Diarization: {args.diarize}")
    bank, embedder, pipeline = prepare_speakers(args)
    worker = InferenceWorker(client, args.model, source, bank, embedder, pipeline)
    worker.start()
    print("Ready. Opening capture …")
    print("Ctrl+C finishes the current buffer, then exits. Press again to quit immediately.")
    proc = ffmpeg_pcm(sys_index, mic_index)

    vad = EnergyVAD(threshold=args.vad)
    stop = threading.Event()
    hb = threading.Thread(
        target=heartbeat_loop,
        args=(client, stop),
        kwargs={"device": device_name, "source": source, "model": args.model},
        daemon=True,
    )
    hb.start()
    client.heartbeat(True, device_name, source, args.model)

    def request_stop(signum: int, _frame) -> None:
        if stop.is_set():
            print("\nForce exit.", flush=True)
            sys.exit(128 + signum)
        print(
            "\nStopping — finishing the current buffer. Press Ctrl+C again to quit immediately.",
            flush=True,
        )
        stop.set()
        stop_ffmpeg(proc)

    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)

    max_samples = int(args.max_seconds * SAMPLE_RATE)
    min_samples = int(args.min_seconds * SAMPLE_RATE)
    silence_frames = max(1, int(args.flush_silence * 1000 / FRAME_MS))
    try:
        if pipeline is None:
            consume_frames(
                frames_from_proc(proc),
                vad,
                max_samples,
                worker.submit,
            )
        else:
            consume_windows(
                frames_from_proc(proc),
                vad,
                min_samples,
                max_samples,
                silence_frames,
                worker.submit,
            )
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        stop.set()
        stop_ffmpeg(proc)
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        print("Waiting for inference queue to drain …", flush=True)
        worker.close()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        hb.join(timeout=2)


def run_file(args: argparse.Namespace, client: AppClient) -> None:
    path = Path(args.file)
    print(f"Transcribing {path} with {args.model}")
    print(f"Diarization: {args.diarize}")
    bank, embedder, pipeline = prepare_speakers(args)
    audio = decode_file(path)
    client.heartbeat(True, str(path), "file", args.model)
    if pipeline is not None:
        process_clip(audio, client, args.model, "file", bank, embedder, pipeline)
    else:
        consume_frames(
            frames_from_array(audio),
            EnergyVAD(threshold=args.vad),
            int(args.max_seconds * SAMPLE_RATE),
            lambda piece: process_clip(piece, client, args.model, "file", bank, embedder, None),
        )
    client.heartbeat(False, str(path), "file", args.model)


def main() -> None:
    parser = argparse.ArgumentParser(description="StenoPod Metal Whisper sidecar")
    parser.add_argument("--app", default="http://127.0.0.1:8780", help="StenoPod app URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", help="AVFoundation device name or index (default: BlackHole)")
    parser.add_argument(
        "--source",
        choices=("system", "mic", "both"),
        default="system",
        help="system = BlackHole, mic = microphone, both = mix for meetings",
    )
    parser.add_argument("--mic-device", help="Microphone name or index (default: built-in mic)")
    parser.add_argument("--mic", action="store_true", help="Shortcut for --source both")
    parser.add_argument("--file", help="Transcribe a local audio file instead of live capture")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--vad", type=float, default=0.012, help="RMS speech threshold")
    parser.add_argument("--max-seconds", type=float, default=18.0, help="Max audio window before diarize/transcribe")
    parser.add_argument("--min-seconds", type=float, default=6.0, help="Min window for pyannote (live)")
    parser.add_argument("--flush-silence", type=float, default=0.7, help="Silence seconds that close a pyannote window")
    parser.add_argument(
        "--diarize",
        choices=("pyannote", "ecapa"),
        default="pyannote",
        help="pyannote.audio turns (default) or per-utterance ECAPA clustering",
    )
    parser.add_argument("--num-speakers", type=int, help="Optional speaker count hint for pyannote")
    args = parser.parse_args()

    if args.list_devices:
        devices = list_audio_devices()
        if not devices:
            print("No audio devices found.")
            sys.exit(1)
        for idx, name in devices:
            print(f"[{idx}] {name}")
        return

    if sys.platform != "darwin":
        raise SystemExit("The sidecar is macOS-only (Metal / AVFoundation).")

    client = AppClient(args.app)
    if args.file:
        run_file(args, client)
    else:
        run_live(args, client)


if __name__ == "__main__":
    main()
