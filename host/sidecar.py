#!/usr/bin/env python3
"""Capture macOS system audio (or a file) and transcribe it with Metal Whisper."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np

HOST_DIR = Path(__file__).resolve().parent
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from hf_offline import WHISPER_REPO, force_hf_offline, require_cached

force_hf_offline()

from chunk_transcribe import (
    load_wav,
    process_range,
    process_split_channels,
    read_wav_range,
    speech_seconds,
    trim_to_speech,
)
from diarize import configure_ssl
from pyannote_diarize import PyannoteDiarizer

configure_ssl()

SAMPLE_RATE = 16000
FRAME_MS = 30
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
DEFAULT_MODEL = WHISPER_REPO
CAPTURE_DIR = HOST_DIR.parent / ".run" / "captures"
CONTROL_PORT = 8781


def new_capture_path(session_id: str | None = None) -> Path:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sid = re.sub(r"[^a-zA-Z0-9_-]", "", session_id or "")[:12]
    name = f"stenopod-{sid}-{stamp}.wav" if sid else f"stenopod-{stamp}.wav"
    return CAPTURE_DIR / name


def pcm_wav_header(data_bytes: int, sr: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    block = max(1, int(channels)) * 2
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + data_bytes),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sr, sr * block, block, 16),
            b"data",
            struct.pack("<I", data_bytes),
        ]
    )


class PcmWavWriter:
    """16-bit WAV that can be read while capture is still writing.

    Dual capture is stereo: left = mic (you), right = BlackHole (them).
    """

    def __init__(self, path: Path, append: bool = False, channels: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.channels = max(1, int(channels))
        if append and self.path.is_file() and self.path.stat().st_size >= 44:
            import wave as _wave

            with _wave.open(str(self.path), "rb") as handle:
                self.channels = handle.getnchannels() or 1
            self.file = open(self.path, "r+b")
            self.file.seek(0, os.SEEK_END)
            self.samples = max(0, (self.file.tell() - 44) // (2 * self.channels))
        else:
            self.file = open(self.path, "wb")
            self.file.write(pcm_wav_header(0, channels=self.channels))
            self.samples = 0
        self._header_at = self.samples
        self.rewrite_header()

    def write_audio(self, audio: np.ndarray) -> None:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != self.channels:
            if arr.shape[1] == 1 and self.channels == 2:
                arr = np.repeat(arr, 2, axis=1)
            else:
                arr = arr[:, : self.channels]
        pcm = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
        self.file.write(np.ascontiguousarray(pcm).tobytes())
        self.samples += int(arr.shape[0])
        if self.samples - self._header_at >= SAMPLE_RATE:
            self.rewrite_header()
            self._header_at = self.samples

    def flush(self) -> None:
        self.file.flush()
        try:
            os.fsync(self.file.fileno())
        except OSError:
            pass

    def rewrite_header(self) -> None:
        self.file.seek(0)
        self.file.write(pcm_wav_header(self.samples * 2 * self.channels, channels=self.channels))
        self.file.seek(0, os.SEEK_END)
        self.flush()

    def close(self) -> None:
        try:
            self.rewrite_header()
        finally:
            self.file.close()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def macos_default_devices() -> tuple[str | None, str | None]:
    """Return (default output name, default input name) from System Settings."""
    try:
        import plistlib

        raw = subprocess.check_output(
            ["system_profiler", "SPAudioDataType", "-xml"],
            timeout=12,
        )
        data = plistlib.loads(raw)
    except Exception:
        return None, None
    output = None
    inp = None

    def walk(node: object) -> None:
        nonlocal output, inp
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        name = node.get("_name")
        if isinstance(name, str):
            if node.get("coreaudio_default_audio_output_device") in ("spaudio_yes", True, "yes"):
                output = name
            if node.get("coreaudio_default_audio_input_device") in ("spaudio_yes", True, "yes"):
                inp = name
        for value in node.values():
            if isinstance(value, (list, dict)):
                walk(value)

    walk(data)
    return output, inp


def describe_system_route() -> None:
    output, inp = macos_default_devices()
    if output:
        print(f"macOS Sound Output: {output}")
    if inp:
        print(f"macOS Sound Input:  {inp}")
    print(
        "The sidecar records BlackHole 2ch (and the mic if --source both). "
        "It never opens StenoPod Headphones / Speakers — those are outputs. "
        "Picking one of them in System Settings → Sound is how Meet/YouTube send a copy into BlackHole."
    )
    if output and "blackhole" not in output.lower() and "steno" not in output.lower():
        print(
            f"  Note: Output is {output!r}, not a StenoPod Multi-Output. "
            "BlackHole will stay silent. Switch Sound → Output to StenoPod Headphones or StenoPod Speakers.",
            flush=True,
        )
    if inp and "airpod" in inp.lower():
        print(
            "  Note: Sound Input is AirPods. macOS often switches this back when the buds connect.\n"
            "  Set System Settings → Sound → Input to MacBook Air-Mikrofon (not AirPods),\n"
            "  and turn the Input volume up. Then Start recording again.",
            flush=True,
        )


def macos_input_volume() -> float | None:
    proc = subprocess.run(
        ["osascript", "-e", "input volume of (get volume settings)"],
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "").strip()
    if not text or text.lower() == "missing value":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def macos_set_input_volume(percent: float) -> None:
    level = max(0, min(100, int(round(percent))))
    subprocess.run(
        ["osascript", "-e", f"set volume input volume {level}"],
        capture_output=True,
        text=True,
    )


def prepare_mac_microphone() -> None:
    """The Mac mic is unusable at a 5–10% Input slider; raise it before capture."""
    vol = macos_input_volume()
    if vol is None:
        return
    print(f"macOS Input volume: {vol:.0f}%", flush=True)
    if vol < 50:
        macos_set_input_volume(80)
        print(
            "  Raised Input volume to 80%. Speech was being captured at almost nothing.",
            flush=True,
        )


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
    skip = ("blackhole", "multi-output", "multiausgang", "teams audio", "find my")
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


def stop_ffmpeg(proc: subprocess.Popen[bytes]) -> None:
    """Tell ffmpeg to exit. Do not close stdout here — the capture loop may be reading it."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass


def ffmpeg_pcm(system_spec: str, mic_spec: str | None = None) -> subprocess.Popen[bytes]:
    if mic_spec is None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-thread_queue_size",
            "1024",
            "-f",
            "avfoundation",
            "-i",
            system_spec,
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
            "warning",
            "-nostdin",
            "-thread_queue_size",
            "1024",
            "-f",
            "avfoundation",
            "-i",
            system_spec,
            "-thread_queue_size",
            "1024",
            "-f",
            "avfoundation",
            "-i",
            mic_spec,
            "-filter_complex",
            "[0:a]aresample=16000,aformat=channel_layouts=mono[bh];"
            "[1:a]aresample=16000,aformat=channel_layouts=mono[mic];"
            "[bh][mic]join=inputs=2:channel_layout=stereo:map=1.0-FL|0.0-FR[a]",
            "-map",
            "[a]",
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    time.sleep(0.2)
    if proc.poll() is not None:
        raise SystemExit(
            "ffmpeg could not open the capture device. "
            "BlackHole or the mic may be locked (quit extra sidecar/ffmpeg, or the meeting app may have exclusive mic access). "
            "Allow Microphone in System Settings → Privacy & Security."
        )
    print("ffmpeg is running. Waiting for the first audio frame …", flush=True)
    return proc


def _read_pipe(stream, nbytes: int) -> bytes:
    """Read until nbytes or EOF. A pipe read may return a short chunk without ending capture."""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = stream.read(nbytes - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


class MicGain:
    """Lift a quiet Mac mic on the left channel. The right channel stays untouched."""

    def __init__(self) -> None:
        self.gain = 6.0
        self.speech_peak = 0.0

    def apply(self, mic: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32)
        raw_peak = float(np.max(np.abs(mic))) if mic.size else 0.0
        self.speech_peak = max(self.speech_peak, raw_peak)
        if raw_peak >= 0.012:
            desired = min(12.0, max(2.0, 0.22 / raw_peak))
            self.gain = 0.88 * self.gain + 0.12 * desired
            return np.clip(mic * self.gain, -1.0, 1.0)
        return np.clip(mic, -1.0, 1.0)


def frames_from_proc(proc: subprocess.Popen[bytes], channels: int = 1) -> Iterator[np.ndarray]:
    """Yield frames. Dual capture is stereo: L=mic (you), R=BlackHole (them)."""
    assert proc.stdout is not None
    sample_bytes = 2 * channels
    nbytes = SAMPLES_PER_FRAME * sample_bytes
    first = True
    bh_peak = 0.0
    mic_peak = 0.0
    n_frames = 0
    log_every = max(1, int(5.0 * 1000 / FRAME_MS))
    warned_bh = False
    warned_mic = False
    mic_gain = MicGain()
    while True:
        raw = _read_pipe(proc.stdout, nbytes)
        if not raw:
            if first:
                print("No audio frames from ffmpeg — device busy or capture failed.", file=sys.stderr)
            break
        usable = raw[: len(raw) - (len(raw) % sample_bytes)]
        if len(usable) < sample_bytes:
            break
        pcm = np.frombuffer(usable, dtype=np.int16).astype(np.float32) / 32768.0
        if channels == 2:
            stereo = pcm.reshape(-1, 2)
            mic = stereo[:, 0]
            bh = stereo[:, 1]
            bh_p = float(np.max(np.abs(bh))) if bh.size else 0.0
            mic_p = float(np.max(np.abs(mic))) if mic.size else 0.0
            bh_peak = max(bh_peak, bh_p)
            mic_peak = max(mic_peak, mic_p)
            n_frames += 1
            if first:
                print(
                    f"Capture started (BlackHole peak {bh_p:.4f}, mic peak {mic_p:.4f}). "
                    "Writing stereo wav: left=you, right=them.",
                    flush=True,
                )
                first = False
            if n_frames % log_every == 0:
                print(
                    f"  levels ~5s: meeting/right {bh_peak:.4f}  you/left {mic_peak:.4f}  "
                    f"(mic×{mic_gain.gain:.0f})",
                    flush=True,
                )
                if bh_peak < 0.008 and mic_peak >= 0.008 and not warned_bh:
                    print(
                        "  BlackHole is silent while the mic has audio. Hearing Meet in the headphones\n"
                        "  does not mean BlackHole is getting a copy. Keep Sound → Output on\n"
                        "  StenoPod Headphones (Meet uses system audio). Then restart the Meet tab.\n"
                        "  In Audio MIDI Setup, select StenoPod Headphones: BlackHole 2ch must be\n"
                        "  ticked, not muted, 48 kHz, Drift Correction on.",
                        flush=True,
                    )
                    warned_bh = True
                if n_frames >= log_every * 2 and mic_gain.speech_peak < 0.02 and not warned_mic:
                    print(
                        "  Mac microphone is still barely audible. Set Sound → Input to\n"
                        "  MacBook Air-Mikrofon (not AirPods) and turn the Input volume up.",
                        flush=True,
                    )
                    warned_mic = True
                bh_peak = 0.0
                mic_peak = 0.0
            frame = np.stack([mic_gain.apply(mic), bh], axis=1)
        else:
            frame = pcm
            if first:
                peak = float(np.max(np.abs(frame))) if frame.size else 0.0
                print(f"Capture started (first-frame peak {peak:.4f}).", flush=True)
                first = False
        yield frame
        if len(raw) < nbytes:
            break


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


class AppClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=10.0)
        self._lock = threading.Lock()

    def send_levels(self, channels: int, left: float, right: float) -> None:
        try:
            with self._lock:
                self.http.post(
                    f"{self.base_url}/api/sidecar/levels",
                    json={
                        "channels": int(channels),
                        "left": float(left),
                        "right": float(right),
                    },
                )
        except httpx.HTTPError:
            pass

    def heartbeat(
        self,
        listening: bool,
        device: str | None,
        source: str,
        model: str,
        phase: str | None = None,
        recorded_s: float | None = None,
        wav: str | None = None,
    ) -> None:
        try:
            with self._lock:
                self.http.post(
                    f"{self.base_url}/api/sidecar",
                    json={
                        "listening": listening,
                        "device": device,
                        "source": source,
                        "model": model,
                        "phase": phase,
                        "recorded_s": recorded_s,
                        "wav": wav,
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

    def send_voices(self, speakers: list[dict]) -> None:
        if not speakers:
            return
        try:
            with self._lock:
                r = self.http.post(
                    f"{self.base_url}/api/speakers/voices",
                    json={"speakers": speakers},
                )
            if r.is_error:
                print(f"voice profiles rejected: {r.status_code} {r.text}", file=sys.stderr)
        except httpx.HTTPError as exc:
            print(f"voice profiles failed: {exc}", file=sys.stderr)

    def send_progress(self, payload: dict) -> None:
        try:
            with self._lock:
                r = self.http.post(f"{self.base_url}/api/progress", json=payload)
            if r.is_error:
                print(f"progress rejected: {r.status_code} {r.text}", file=sys.stderr)
        except httpx.HTTPError as exc:
            print(f"progress failed: {exc}", file=sys.stderr)


def transcribe_clip(audio: np.ndarray, model: str) -> list[dict]:
    """One Whisper window. Segment times are relative to this clip.

    Isolated slices of music come back as "Thank you". Callers should pass a
    normal Whisper window of the recording, then keep words that land inside
    a pyannote turn.
    """
    import mlx_whisper

    skip, audio = trim_to_speech(audio)
    skip_s = skip / float(SAMPLE_RATE)
    if audio.size < SAMPLE_RATE // 4:
        return []
    require_cached([model])
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=2.0,
        word_timestamps=True,
        verbose=None,
    )
    segments = result.get("segments") or []
    lang = str(result.get("language") or "").strip().lower() or "und"
    stamped = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if not str(seg.get("language") or "").strip():
            seg["language"] = lang
        if skip_s:
            try:
                seg["start"] = float(seg.get("start") or 0.0) + skip_s
                seg["end"] = float(seg.get("end") or 0.0) + skip_s
            except (TypeError, ValueError):
                pass
            words = seg.get("words") or []
            if isinstance(words, list):
                for item in words:
                    if not isinstance(item, dict):
                        continue
                    try:
                        if item.get("start") is not None:
                            item["start"] = float(item["start"]) + skip_s
                        if item.get("end") is not None:
                            item["end"] = float(item["end"]) + skip_s
                    except (TypeError, ValueError):
                        pass
        stamped.append(seg)
    return stamped

def heartbeat_loop(
    client: AppClient,
    shutdown: threading.Event,
    runtime: "CaptureRuntime",
    device: str | None,
    source: str,
    model: str,
) -> None:
    while True:
        snap = runtime.snapshot()
        client.heartbeat(
            True,
            device,
            source,
            model,
            phase=snap["phase"],
            recorded_s=snap["recorded_s"],
            wav=snap["wav"],
        )
        if shutdown.wait(1.0):
            break
    client.heartbeat(False, device, source, model)


def level_loop(client: AppClient, shutdown: threading.Event, runtime: "CaptureRuntime") -> None:
    """Push meter levels often enough for the UI bars to follow speech."""
    while not shutdown.is_set():
        snap = runtime.levels()
        client.send_levels(int(snap["channels"]), float(snap["left"]), float(snap["right"]))
        if shutdown.wait(0.08):
            break
    client.send_levels(1, 0.0, 0.0)


class CaptureRuntime:
    """One session wav on disk. Capture writes; transcription runs after stop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.writer: PcmWavWriter | None = None
        self.wav_path: Path | None = None
        self.session_id = ""
        self.recording = True
        self._channels = 1
        self.level_left = 0.0
        self.level_right = 0.0
        self._level_at = 0.0
        self.stop_cb = None

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            if self.writer is not None:
                self.writer.rewrite_header()
            samples = self.writer.samples if self.writer is not None else 0
            phase = "recording" if self.recording else "idle"
            return {
                "ok": True,
                "wav": str(self.wav_path) if self.wav_path else None,
                "samples": samples,
                "recorded_s": samples / SAMPLE_RATE if samples else 0.0,
                "session_id": self.session_id,
                "phase": phase,
            }

    def levels(self) -> dict[str, float]:
        with self.lock:
            return {
                "channels": self._channels,
                "left": self.level_left,
                "right": self.level_right,
            }

    def _observe_locked(self, frame: np.ndarray) -> None:
        """Fast attack, slow release, so the meter falls like a recorder."""
        arr = np.asarray(frame, dtype=np.float32)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            peaks = [float(np.max(np.abs(arr)))]
        else:
            peaks = [float(np.max(np.abs(arr[:, i]))) for i in range(min(2, arr.shape[1]))]
        now = time.monotonic()
        dt = 0.03 if self._level_at <= 0 else max(0.0, now - self._level_at)
        self._level_at = now
        release = math.exp(-dt / 0.28) if dt else 1.0

        def follow(prev: float, peak: float) -> float:
            peak = max(0.0, min(1.0, peak))
            return peak if peak >= prev else prev * release

        self.level_left = follow(self.level_left, peaks[0] if peaks else 0.0)
        self.level_right = follow(self.level_right, peaks[1] if len(peaks) > 1 else 0.0)

    def open_wav(
        self,
        session_id: str,
        path: Path | None = None,
        append: bool = False,
        channels: int = 1,
    ) -> Path:
        with self.lock:
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception:
                    pass
                self.writer = None
            self.session_id = session_id or "session"
            self._channels = 2 if channels >= 2 else 1
            self.level_left = 0.0
            self.level_right = 0.0
            self._level_at = 0.0
            if path is None:
                path = new_capture_path(self.session_id)
            self.wav_path = Path(path)
            use_append = append and self.wav_path.is_file()
            self.writer = PcmWavWriter(self.wav_path, append=use_append, channels=self._channels)
            kind = "stereo L=you R=them" if self._channels == 2 else "mono"
            print(
                f"{'Appending' if use_append else 'Recording'} {kind} wav {self.wav_path} "
                f"({self.writer.samples / SAMPLE_RATE:.0f}s already on disk)",
                flush=True,
            )
            return self.wav_path

    def write_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            if self.writer is None:
                return
            self.writer.write_audio(frame)
            self._observe_locked(frame)

    def close_wav(self) -> None:
        with self.lock:
            if self.writer is None:
                return
            try:
                self.writer.close()
            except Exception:
                pass
            dur = self.writer.samples / SAMPLE_RATE if self.writer.samples else 0.0
            print(f"Wav ({dur:.0f}s) → {self.wav_path}", flush=True)
            self.writer = None


def start_control_server(runtime: CaptureRuntime, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            print(f"control: {fmt % args}", flush=True)

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def _send(self, code: int, obj: dict[str, object]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/status":
                self._send(200, runtime.snapshot())
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            body = self._read_json()
            if path == "/rotate":
                session_id = str(body.get("session_id") or "session")
                wav = runtime.open_wav(session_id)
                snap = runtime.snapshot()
                snap["ok"] = True
                snap["wav"] = str(wav)
                self._send(200, snap)
                return
            if path == "/open":
                session_id = str(body.get("session_id") or "session")
                given = body.get("wav")
                path_arg = Path(str(given)) if given else None
                append = bool(path_arg and path_arg.is_file())
                wav = runtime.open_wav(session_id, path=path_arg, append=append)
                snap = runtime.snapshot()
                snap["ok"] = True
                snap["wav"] = str(wav)
                self._send(200, snap)
                return
            if path == "/stop":
                cb = runtime.stop_cb
                if cb is not None:
                    cb()
                self._send(200, {"ok": True, "stopped": True})
                return
            self._send(404, {"ok": False, "error": "not found"})

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, name="stenopod-ctl", daemon=True).start()
    print(f"Control API on http://127.0.0.1:{port}  (GET /status, POST /rotate, POST /stop)", flush=True)
    return httpd


def _speaker_ceiling(args: argparse.Namespace) -> int | None:
    """Upper bound the transcript must not exceed. An exact count is that bound."""
    exact = getattr(args, "num_speakers", None)
    if exact:
        return int(exact)
    cap = getattr(args, "max_speakers", None)
    return int(cap) if cap else None


def _speaker_floor(args: argparse.Namespace) -> int | None:
    """Minimum used only when the user did not ask for an exact count."""
    if getattr(args, "num_speakers", None):
        return None
    floor = getattr(args, "min_speakers", None)
    return int(floor) if floor else None


def prepare_pipeline(args: argparse.Namespace) -> PyannoteDiarizer:
    exact = args.num_speakers
    return PyannoteDiarizer(
        num_speakers=exact,
        min_speakers=None if exact else args.min_speakers,
        max_speakers=None if exact else args.max_speakers,
    )


def run_live(args: argparse.Namespace, client: AppClient) -> None:
    source = "both" if args.mic else args.source
    if source == "speakers":
        source = "system"
    if source == "mic":
        sys_index, device_name = pick_mic_device(args.mic_device)
        mic_index = None
        sys_spec = f":{device_name}"
        mic_spec = None
        describe_system_route()
        prepare_mac_microphone()
    elif source == "both":
        sys_index, sys_name = pick_device(args.device)
        mic_index, mic_name = pick_mic_device(args.mic_device)
        device_name = f"{sys_name} + {mic_name}"
        print(f"Stereo capture {sys_name!r} [{sys_index}] + mic {mic_name!r} [{mic_index}] (L=you, R=them)")
        describe_system_route()
        prepare_mac_microphone()
        sys_spec = f":{sys_name}"
        mic_spec = f":{mic_name}"
    else:
        sys_index, device_name = pick_device(args.device)
        mic_index = None
        sys_spec = f":{device_name}"
        mic_spec = None
        describe_system_route()

    print(f"Will listen on {device_name!r} ({source}) → {args.app}")
    print(f"Whisper model: {args.model}")
    if mic_spec is not None:
        print(
            "Headphones capture writes a stereo wav: left is you (Mac mic), "
            "right is the meeting (BlackHole). Transcription runs after you stop.",
            flush=True,
        )
    elif source == "mic":
        print(
            "Mic capture writes one channel: the Mac microphone. Transcription runs after you stop.",
            flush=True,
        )
    else:
        print(
            "Speakers capture writes one channel: the meeting on BlackHole. "
            "The microphone is not recorded. Transcription runs after you stop.",
            flush=True,
        )

    runtime = CaptureRuntime()
    session_id = args.session_id or os.environ.get("STENOPOD_SESSION_ID") or "session"
    wav_arg = Path(args.wav) if args.wav else None
    runtime.open_wav(
        session_id,
        path=wav_arg,
        append=bool(wav_arg and wav_arg.is_file()),
        channels=2 if mic_spec is not None else 1,
    )
    httpd = start_control_server(runtime, args.control_port)

    print(f"  saved wav: {runtime.wav_path}")
    print("Ready. Opening capture …")
    print("Ctrl+C / Stop capture closes the wav. Press again to quit immediately.")
    proc = ffmpeg_pcm(sys_spec, mic_spec)

    shutdown = threading.Event()
    hb = threading.Thread(
        target=heartbeat_loop,
        args=(client, shutdown, runtime, device_name, source, args.model),
        daemon=True,
    )
    hb.start()
    levels = threading.Thread(
        target=level_loop,
        args=(client, shutdown, runtime),
        daemon=True,
    )
    levels.start()

    stopping = threading.Event()

    def request_stop(signum: int | None = None, _frame=None) -> None:
        if stopping.is_set():
            if signum:
                print("\nForce exit. The wav is already on disk.", flush=True)
                sys.exit(128 + signum)
            return
        print("\nStopping capture — closing the wav.", flush=True)
        stopping.set()
        stop_ffmpeg(proc)

    runtime.stop_cb = lambda: request_stop(None, None)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)

    last_log = 0
    try:
        for frame in frames_from_proc(proc, channels=2 if mic_spec is not None else 1):
            runtime.write_frame(frame)
            with runtime.lock:
                samples = runtime.writer.samples if runtime.writer is not None else 0
            if samples - last_log >= 10 * SAMPLE_RATE:
                last_log = samples
                print(f"  recording {samples / SAMPLE_RATE:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("\nStopped recording.", flush=True)
    finally:
        if not stopping.is_set():
            print("ffmpeg closed the capture stream.", flush=True)
        stopping.set()
        runtime.recording = False
        stop_ffmpeg(proc)
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        runtime.close_wav()
        try:
            httpd.shutdown()
        except Exception:
            pass
        shutdown.set()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        hb.join(timeout=2)


def run_file(args: argparse.Namespace, client: AppClient) -> None:
    path = Path(args.file)
    print(f"Transcribing {path} with {args.model}")
    client.send_progress({"stage": "load", "chunk": 0, "chunks": 0, "progress": 0})
    start_s = max(0.0, float(args.start_s or 0.0))
    if path.suffix.lower() == ".wav":
        audio = load_wav(path, end_s=args.end_s)
    elif args.end_s is not None:
        audio = read_wav_range(path, 0.0, float(args.end_s))
    else:
        audio = decode_file(path)
    stereo = audio.ndim == 2 and audio.shape[1] >= 2
    if stereo:
        left = np.ascontiguousarray(audio[:, 0])
        right = np.ascontiguousarray(audio[:, 1])
        left_s = speech_seconds(left)
        right_s = speech_seconds(right)
        print(
            f"  channel speech: you/mic {left_s:.0f}s  them/system {right_s:.0f}s",
            flush=True,
        )
        split_ok = left_s >= 4.0 and right_s >= 4.0
        if not split_ok:
            channel_label = "mixed"
            if right_s >= 4.0 and right_s >= left_s * 2:
                print(
                    "Mic channel is quiet. Diarizing system audio with the Min/Max speaker count.",
                    flush=True,
                )
                mono = right
                channel_label = "them"
            elif left_s >= 4.0:
                print(
                    "System channel is too quiet to split (BlackHole got little or no Meet/YouTube). "
                    "Diarizing the mic with the Min/Max speaker count so you and a leaked remote "
                    "voice can be two people, not one.",
                    flush=True,
                )
                if right_s >= 0.5:
                    mono = np.clip(left + right, -1.0, 1.0)
                    channel_label = "mixed"
                else:
                    mono = left
                    channel_label = "you"
            else:
                print(
                    "Both channels are quiet. Mixing left+right and diarizing with Min/Max.",
                    flush=True,
                )
                mono = np.clip(left + right, -1.0, 1.0)
            pipeline = prepare_pipeline(args)
            client.heartbeat(False, str(path), "file", args.model, phase="transcribing", wav=str(path))
            process_range(
                np.ascontiguousarray(mono),
                0.0,
                transcribe_clip,
                client,
                args.model,
                "file",
                pipeline,
                on_progress=client.send_progress,
                transcribe_from=start_s,
                min_speakers=_speaker_floor(args),
                max_speakers=_speaker_ceiling(args),
                label=channel_label,
            )
        else:
            print("Pipeline: left=you (mic), right=them (system). Whisper on each side, then merge.")
            has_local = left_s >= 4.0
            remote_pipe = _remote_pipeline(args, has_local)
            local_pipe = PyannoteDiarizer(num_speakers=1)
            client.heartbeat(False, str(path), "file", args.model, phase="transcribing", wav=str(path))
            process_split_channels(
                left,
                right,
                0.0,
                transcribe_clip,
                client,
                args.model,
                "file",
                remote_pipe,
                local_pipe,
                on_progress=client.send_progress,
                transcribe_from=start_s,
                min_speakers=_speaker_floor(args),
                max_speakers=_speaker_ceiling(args),
                local_present=has_local,
            )
    else:
        print("Pipeline: pyannote on the full recording, then Whisper")
        pipeline = prepare_pipeline(args)
        client.heartbeat(False, str(path), "file", args.model, phase="transcribing", wav=str(path))
        mono = np.ascontiguousarray(audio.reshape(-1))
        process_range(
            mono,
            0.0,
            transcribe_clip,
            client,
            args.model,
            "file",
            pipeline,
            on_progress=client.send_progress,
            transcribe_from=start_s,
            min_speakers=_speaker_floor(args),
            max_speakers=_speaker_ceiling(args),
        )
    client.heartbeat(False, str(path), "file", args.model, phase="idle", wav=str(path))


def _remote_pipeline(args: argparse.Namespace, has_local: bool) -> PyannoteDiarizer:
    """Min/max include you. On a stereo file you are the left channel, so the right side is N-1."""
    exact = getattr(args, "num_speakers", None)
    if exact:
        n = max(1, int(exact) - 1) if has_local else int(exact)
        return PyannoteDiarizer(num_speakers=n)
    min_s = getattr(args, "min_speakers", None)
    max_s = getattr(args, "max_speakers", None)
    if has_local:
        if min_s:
            min_s = max(1, int(min_s) - 1)
        if max_s:
            max_s = max(1, int(max_s) - 1)
        if min_s and max_s and min_s > max_s:
            min_s = max_s
    return PyannoteDiarizer(num_speakers=None, min_speakers=min_s, max_speakers=max_s)


def _wav_peak_rms(path: Path) -> tuple[float, float]:
    import wave

    with wave.open(str(path), "rb") as handle:
        n = handle.getnframes()
        samples = np.frombuffer(handle.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples * samples)))
    return peak, rms


def _record_device_wav(spec: str, seconds: float, dest: Path) -> tuple[float, float]:
    dest.unlink(missing_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
        "-i",
        spec,
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 10)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"ffmpeg timed out opening {spec}. Quit extra sidecar/ffmpeg processes, "
            "and allow Microphone for Terminal."
        ) from exc
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size < 44:
        err = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"Could not record {spec}: {err or proc.returncode}")
    return _wav_peak_rms(dest)


def run_capture_test() -> None:
    print("StenoPod capture test")
    output, inp = macos_default_devices()
    vol = macos_input_volume()
    if output:
        print(f"Sound Output: {output}")
    if inp:
        print(f"Sound Input:  {inp}")
    if vol is not None:
        print(f"Input volume: {vol:.0f}%")
        if vol < 50:
            macos_set_input_volume(80)
            print("  Input volume was too low for speech. Raised it to 80%.")
    if inp and "airpod" in inp.lower():
        print("  WARNING: Sound Input is AirPods. Switch Input to MacBook Air-Mikrofon.")
    devices = list_audio_devices()
    print("AVFoundation audio devices:")
    for idx, name in devices:
        print(f"  [{idx}] {name}")
    _, sys_name = pick_device(None)
    _, mic_name = pick_mic_device(None)
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    bh_wav = tmp / "stenopod-capture-test-bh.wav"
    mic_wav = tmp / "stenopod-capture-test-mic.wav"
    print(f"\n1/2 Recording 3s from {sys_name} (system audio). Play YouTube if you want to test BlackHole.")
    peak, rms = _record_device_wav(f":{sys_name}", 3.0, bh_wav)
    print(f"   BlackHole peak={peak:.4f}  rms={rms:.4f}")
    if peak < 0.002:
        print("   SILENT. Meet/YouTube is not reaching BlackHole. Keep Output on StenoPod Headphones.")
    else:
        print("   System audio is reaching BlackHole.")
    print(f"\n2/2 Recording 3s from {mic_name}. Speak now.")
    peak, rms = _record_device_wav(f":{mic_name}", 3.0, mic_wav)
    print(f"   Mic peak={peak:.4f}  rms={rms:.4f}")
    if peak < 0.02:
        print("   SILENT / too quiet. Set Input to MacBook Air-Mikrofon (not AirPods)")
        print("   and turn the Input volume up. Meet can exclusive-lock the mic.")
    else:
        print("   Microphone is capturing speech.")


def main() -> None:
    parser = argparse.ArgumentParser(description="StenoPod Metal Whisper sidecar")
    parser.add_argument("--app", default="http://127.0.0.1:8780", help="StenoPod app URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", help="AVFoundation device name or index (default: BlackHole)")
    parser.add_argument(
        "--source",
        choices=("system", "mic", "both", "speakers"),
        default="system",
        help="system/speakers = BlackHole only (meeting). both = stereo, left Mac mic, right BlackHole",
    )
    parser.add_argument("--mic-device", help="Microphone name or index (default: built-in mic)")
    parser.add_argument("--mic", action="store_true", help="Shortcut for --source both")
    parser.add_argument("--file", help="Transcribe a local audio file instead of live capture")
    parser.add_argument("--start-s", type=float, default=0.0, help="With --file, start offset in seconds")
    parser.add_argument("--end-s", type=float, help="With --file, end offset in seconds")
    parser.add_argument("--session-id", help="Session id used in the wav filename")
    parser.add_argument("--wav", help="Reuse/append this wav (same session after Stop/Start)")
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--capture-test", action="store_true", help="Record 3s from BlackHole and the Mac mic")
    # Older app processes still pass these. Capture ignores them.
    parser.add_argument("--diarize", default="pyannote", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-seconds", type=float, default=55.0, help=argparse.SUPPRESS)
    parser.add_argument("--vad", type=float, default=0.012, help=argparse.SUPPRESS)
    parser.add_argument("--silero-threshold", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--max-seconds", type=float, default=12.0, help=argparse.SUPPRESS)
    parser.add_argument("--min-seconds", type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument("--flush-silence", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--num-speakers", type=int, help="Exact speaker count for pyannote")
    parser.add_argument("--min-speakers", type=int, help="Minimum speakers for pyannote")
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional upper bound for pyannote clustering. Omit to leave the count uncapped.",
    )
    args = parser.parse_args()

    if args.list_devices:
        devices = list_audio_devices()
        if not devices:
            print("No audio devices found.")
            sys.exit(1)
        for idx, name in devices:
            print(f"[{idx}] {name}")
        return

    if args.capture_test:
        if sys.platform != "darwin":
            raise SystemExit("The sidecar is macOS-only (Metal / AVFoundation).")
        run_capture_test()
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
