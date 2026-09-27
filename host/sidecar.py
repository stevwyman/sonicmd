#!/usr/bin/env python3
"""Capture macOS system audio (or a file) and transcribe it with Metal Whisper."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque
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

from chunk_transcribe import DEFAULT_CHUNK_S, process_range, read_wav_range
from diarize import SpeakerBank, SpectralEmbedder, configure_ssl, load_embedder
from pyannote_diarize import PyannoteDiarizer, SpeakerTurn

configure_ssl()

SAMPLE_RATE = 16000
FRAME_MS = 30
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
MIN_FLUSH_SAMPLES = SAMPLE_RATE // 2
DEFAULT_MODEL = WHISPER_REPO
CAPTURE_DIR = HOST_DIR.parent / ".run" / "captures"
SLICE_SAMPLES = 8 * 60 * SAMPLE_RATE
CONTROL_PORT = 8781
RECORD_TAIL_S = 1.0


def new_capture_path(session_id: str | None = None) -> Path:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sid = re.sub(r"[^a-zA-Z0-9_-]", "", session_id or "")[:12]
    name = f"stenopod-{sid}-{stamp}.wav" if sid else f"stenopod-{stamp}.wav"
    return CAPTURE_DIR / name


def pcm_wav_header(data_bytes: int, sr: int = SAMPLE_RATE) -> bytes:
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + data_bytes),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16),
            b"data",
            struct.pack("<I", data_bytes),
        ]
    )


class PcmWavWriter:
    """Mono 16-bit WAV that can be read while capture is still writing."""

    def __init__(self, path: Path, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if append and self.path.is_file() and self.path.stat().st_size >= 44:
            self.file = open(self.path, "r+b")
            self.file.seek(0, os.SEEK_END)
            self.samples = max(0, (self.file.tell() - 44) // 2)
        else:
            self.file = open(self.path, "wb")
            self.file.write(pcm_wav_header(0))
            self.samples = 0
        self.rewrite_header()

    def write_audio(self, audio: np.ndarray) -> None:
        pcm = np.clip(np.asarray(audio, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)
        self.file.write(pcm.tobytes())
        self.samples += int(pcm.size)

    def flush(self) -> None:
        self.file.flush()
        try:
            os.fsync(self.file.fileno())
        except OSError:
            pass

    def rewrite_header(self) -> None:
        self.file.seek(0)
        self.file.write(pcm_wav_header(self.samples * 2))
        self.file.seek(0, os.SEEK_END)
        self.flush()

    def close(self) -> None:
        try:
            self.rewrite_header()
        finally:
            self.file.close()


def write_pcm16(handle: wave.Wave_write, audio: np.ndarray) -> None:
    pcm = np.clip(np.asarray(audio, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)
    handle.writeframes(pcm.tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


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
            "  Note: Sound Input is AirPods. Use MacBook Air-Mikrofon or BlackHole may go silent.",
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
            "[bh][mic]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[a]",
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


def frames_from_proc(proc: subprocess.Popen[bytes], channels: int = 1) -> Iterator[np.ndarray]:
    """Yield mono float32 frames. If channels==2, L=BlackHole R=mic; mix after logging levels."""
    assert proc.stdout is not None
    nbytes = SAMPLES_PER_FRAME * 2 * channels
    first = True
    bh_peak = 0.0
    mic_peak = 0.0
    n_frames = 0
    log_every = max(1, int(5.0 * 1000 / FRAME_MS))
    warned_bh = False
    while True:
        raw = proc.stdout.read(nbytes)
        if not raw:
            if first:
                print("No audio frames from ffmpeg — device busy or capture failed.", file=sys.stderr)
            break
        if len(raw) < nbytes:
            usable = raw[: len(raw) - (len(raw) % (2 * channels))]
            if len(usable) < 2 * channels:
                break
            pcm = np.frombuffer(usable, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels == 2:
            stereo = pcm.reshape(-1, 2)
            bh = stereo[:, 0]
            mic = stereo[:, 1]
            bh_p = float(np.max(np.abs(bh))) if bh.size else 0.0
            mic_p = float(np.max(np.abs(mic))) if mic.size else 0.0
            bh_peak = max(bh_peak, bh_p)
            mic_peak = max(mic_peak, mic_p)
            n_frames += 1
            if first:
                print(
                    f"Capture started (BlackHole peak {bh_p:.4f}, mic peak {mic_p:.4f}).",
                    flush=True,
                )
                first = False
            if n_frames % log_every == 0:
                print(
                    f"  levels ~5s: BlackHole {bh_peak:.4f}  mic {mic_peak:.4f}",
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
                bh_peak = 0.0
                mic_peak = 0.0
            frame = np.clip(bh + mic, -1.0, 1.0)
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


def transcribe_audio(audio: np.ndarray, model: str) -> tuple[str, str]:
    text, lang, _words = transcribe_window(audio, model, word_timestamps=False)
    return text, lang


def transcribe_clip(audio: np.ndarray, model: str) -> list[dict]:
    """One Whisper window. Segment times are relative to this clip.

    Isolated slices of music come back as "Thank you". Callers should pass a
    normal Whisper window of the recording, then keep words that land inside
    a pyannote turn.
    """
    import mlx_whisper

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
        stamped.append(seg)
    return stamped


def transcribe_window(
    audio: np.ndarray,
    model: str,
    word_timestamps: bool = True,
) -> tuple[str, str, list[dict[str, float | str]]]:
    import mlx_whisper

    if audio.size < SAMPLE_RATE // 4:
        return "", "und", []
    require_cached([model])
    kwargs = dict(
        path_or_hf_repo=model,
        condition_on_previous_text=False,
        no_speech_threshold=0.9,
        word_timestamps=word_timestamps,
    )
    try:
        result = mlx_whisper.transcribe(audio, **kwargs)
    except Exception:
        if word_timestamps:
            result = mlx_whisper.transcribe(audio, **{**kwargs, "word_timestamps": False})
        else:
            raise
    text = (result.get("text") or "").strip()
    lang = result.get("language") or "und"
    words: list[dict[str, float | str]] = []
    for seg in result.get("segments") or []:
        for item in seg.get("words") or []:
            start, end = item.get("start"), item.get("end")
            token = item.get("word") or item.get("text") or ""
            if start is None or end is None:
                continue
            words.append({"word": str(token), "start": float(start), "end": float(end)})
    return text, lang, words


def _join_words(words: list[dict[str, float | str]]) -> str:
    parts = []
    for item in words:
        tok = str(item.get("word") or "").strip()
        if tok:
            parts.append(tok)
    return " ".join(parts).replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")


def _turn_distance(mid: float, turn: SpeakerTurn) -> float:
    if turn.start <= mid <= turn.end:
        return 0.0
    if mid < turn.start:
        return turn.start - mid
    return mid - turn.end


def assign_words_to_turns(
    words: list[dict[str, float | str]],
    turns: list[SpeakerTurn],
    audio: np.ndarray,
) -> list[tuple[SpeakerTurn, str]]:
    if not words:
        return []
    if not turns:
        text = _join_words(words)
        if not text:
            return []
        dummy = SpeakerTurn(audio, "SPEAKER_00", None, float(words[0]["start"]), float(words[-1]["end"]))
        return [(dummy, text)]
    ordered = sorted(turns, key=lambda t: t.start)
    groups: list[list[dict[str, float | str]]] = [[] for _ in ordered]
    for item in words:
        mid = 0.5 * (float(item["start"]) + float(item["end"]))
        idx = next(
            (
                i
                for i, turn in enumerate(ordered)
                if turn.start <= mid < turn.end
                or (turn.start == turn.end and abs(mid - turn.start) < 0.05)
            ),
            None,
        )
        if idx is None:
            idx = min(range(len(ordered)), key=lambda i: _turn_distance(mid, ordered[i]))
        groups[idx].append(item)
    labeled: list[tuple[SpeakerTurn, str]] = []
    for turn, picked in zip(ordered, groups):
        text = _join_words(picked)
        if text:
            labeled.append((turn, text))
    return stitch_same_local(labeled, audio)


def stitch_same_local(
    labeled: list[tuple[SpeakerTurn, str]],
    audio: np.ndarray,
    gap: float = 2.0,
) -> list[tuple[SpeakerTurn, str]]:
    if not labeled:
        return []
    items = sorted(labeled, key=lambda row: row[0].start)
    out: list[tuple[SpeakerTurn, str]] = []
    for turn, text in items:
        if (
            out
            and out[-1][0].local_id == turn.local_id
            and turn.start - out[-1][0].end <= gap
        ):
            prev, prev_text = out[-1]
            start = prev.start
            end = max(prev.end, turn.end)
            a = max(0, int(start * SAMPLE_RATE))
            b = min(audio.size, int(end * SAMPLE_RATE))
            clip = audio[a:b] if b > a else prev.audio
            merged = SpeakerTurn(clip, prev.local_id, prev.embedding, start, end)
            out[-1] = (merged, f"{prev_text} {text}".strip())
        else:
            out.append((turn, text))
    return out


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

    tw = time.perf_counter()
    full_text, lang, words = transcribe_window(audio, whisper_model, word_timestamps=True)
    whisper_s = time.perf_counter() - tw

    labeled = assign_words_to_turns(words, turns, audio) if words else []
    if not labeled and full_text:
        labeled = [(turns[0] if turns else SpeakerTurn(audio, "utt", None, 0.0, audio_s), full_text)]

    window_map: dict[str, str] = {}
    used: set[str] = set()
    n_text = 0
    pending: list[tuple[str, float, float, float, str]] = []
    for turn, text in labeled:
        try:
            vec = turn_embedding(turn, embedder)
            if turn.local_id in window_map:
                speaker = window_map[turn.local_id]
                score = bank.reinforce(speaker, vec)
            else:
                # Voice only — language must not mint extra letters for the same person.
                speaker, score = bank.assign(vec, forbidden=used)
                window_map[turn.local_id] = speaker
                used.add(speaker)
            pending.append((speaker, float(score), turn.start, turn.end, text))
        except Exception as exc:
            print(f"utterance failed ({type(exc).__name__}): {exc}", file=sys.stderr)

    stitched: list[tuple[str, float, float, float, str]] = []
    for speaker, score, start, end, text in pending:
        if stitched and stitched[-1][0] == speaker and start - stitched[-1][3] <= 2.0:
            ps, pscore, pstart, _pend, ptext = stitched[-1]
            stitched[-1] = (ps, pscore, pstart, end, f"{ptext} {text}".strip())
        else:
            stitched.append((speaker, score, start, end, text))

    for speaker, score, start, end, text in stitched:
        try:
            duration_ms = int(round(1000 * max(end - start, 0.05)))
            sim = f"{score:.2f}" if math.isfinite(float(score)) else "?"
            print(f"[{speaker} {lang} {duration_ms}ms sim={sim}] {text}")
            client.send_segment(text, lang, source, speaker, duration_ms, score)
            n_text += 1
        except httpx.HTTPError as exc:
            print(f"failed to post segment: {exc}", file=sys.stderr)

    n_spk = len({turn.local_id for turn, _t in labeled}) if labeled else 0
    print(
        f"  timing {audio_s:.1f}s audio: pyannote {diarize_s:.2f}s / whisper {whisper_s:.2f}s "
        f"({len(turns)} turn{'s' if len(turns) != 1 else ''}, {n_spk} speaker{'s' if n_spk != 1 else ''}, "
        f"{n_text} posted, 1 whisper pass, {len(words)} words)",
        flush=True,
    )


def process_utterance(
    audio: np.ndarray,
    client: AppClient,
    whisper_model: str,
    source: str,
    bank: SpeakerBank,
    embedder: object,
) -> None:
    """One VAD utterance → one Whisper pass → one speaker label."""
    audio_s = audio.size / SAMPLE_RATE if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if audio.size < int(0.3 * SAMPLE_RATE) or peak < 0.008:
        return
    t0 = time.perf_counter()
    text, lang = transcribe_audio(audio, whisper_model)
    whisper_s = time.perf_counter() - t0
    if not text:
        print(f"  skip {audio_s:.1f}s (no text, peak={peak:.4f})", flush=True)
        return
    vec = embedder.embed(audio)
    speaker, score = bank.assign(vec)
    duration_ms = int(round(1000 * audio_s))
    sim = f"{score:.2f}" if math.isfinite(float(score)) else "?"
    print(f"[{speaker} {lang} {duration_ms}ms sim={sim}] {text}", flush=True)
    try:
        client.send_segment(text, lang, source, speaker, duration_ms, score)
    except httpx.HTTPError as exc:
        print(f"failed to post segment: {exc}", file=sys.stderr)
    print(f"  timing {audio_s:.1f}s audio: vad-utt / whisper {whisper_s:.2f}s", flush=True)


def process_recording(
    audio: np.ndarray,
    client: AppClient,
    whisper_model: str,
    source: str,
    bank: SpeakerBank,
    embedder: object,
    pipeline: PyannoteDiarizer | None,
) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    dur = audio.size / SAMPLE_RATE if audio.size else 0.0
    if audio.size < MIN_FLUSH_SAMPLES:
        print("Recording is too short to transcribe.", flush=True)
        return
    if peak < 0.008:
        print("  (near silence — is the Multi-Output sending audio to BlackHole?)", flush=True)
        return
    print(f"Transcribing {dur:.0f}s recording (peak={peak:.4f}) …", flush=True)
    if pipeline is None:
        process_clip(audio, client, whisper_model, source, bank, embedder, None)
        return
    try:
        process_clip(audio, client, whisper_model, source, bank, embedder, pipeline)
        return
    except Exception as exc:
        if dur <= 8 * 60:
            raise
        print(
            f"Full-file pass failed ({type(exc).__name__}: {exc}); "
            "slicing the saved recording into 8-minute pieces.",
            flush=True,
        )
    for start in range(0, audio.size, SLICE_SAMPLES):
        piece = audio[start : start + SLICE_SAMPLES]
        if piece.size < MIN_FLUSH_SAMPLES:
            continue
        print(
            f"  slice {start / SAMPLE_RATE:.0f}s–{(start + piece.size) / SAMPLE_RATE:.0f}s",
            flush=True,
        )
        try:
            process_clip(piece, client, whisper_model, source, bank, embedder, pipeline)
        except Exception as exc:
            print(f"  slice failed ({type(exc).__name__}): {exc}", file=sys.stderr)


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
        self.q: queue.Queue[np.ndarray | None] = queue.Queue()
        self.ready = threading.Event()
        self.warmup_error: str | None = None
        self.waiting_s = 0.0
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
        self.waiting_s += clip.size / SAMPLE_RATE
        self.q.put(clip)
        pending = self.q.qsize()
        if pending > 1:
            print(
                f"  queued {pending} chunks ({self.waiting_s:.0f}s waiting — nothing dropped, catching up)",
                flush=True,
            )

    def close(self) -> None:
        if not self.thread.is_alive():
            return
        self.q.put(None)
        self.thread.join(timeout=3600)
        if self.thread.is_alive():
            print("Inference still running; wait for the queue to drain or stop the app.", file=sys.stderr)

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
                audio_s = item.size / SAMPLE_RATE if item.size else 0.0
                self.waiting_s = max(0.0, self.waiting_s - audio_s)
                t0 = time.perf_counter()
                if self.pipeline is None:
                    process_utterance(
                        item,
                        self.client,
                        self.whisper_model,
                        self.source,
                        self.bank,
                        self.embedder,
                    )
                else:
                    process_clip(
                        item,
                        self.client,
                        self.whisper_model,
                        self.source,
                        self.bank,
                        self.embedder,
                        self.pipeline,
                    )
                work_s = time.perf_counter() - t0
                rtf = work_s / audio_s if audio_s else 0.0
                pending = self.q.qsize()
                if rtf <= 1.0:
                    pace = "KEEP UP"
                else:
                    pace = "behind live — queued, nothing dropped"
                print(
                    f"  realtime {work_s:.1f}s work / {audio_s:.1f}s audio = {rtf:.2f}x ({pace}) "
                    f"queue={pending} waiting={self.waiting_s:.0f}s",
                    flush=True,
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
    print(f"Stream chunk {audio.size / SAMPLE_RATE:.1f}s peak={peak:.4f}{note}", flush=True)
    if peak < 0.008:
        print("  (near silence — is the Multi-Output sending audio to BlackHole?)", flush=True)
        return
    on_window(audio)


def consume_stream(
    frames: Iterator[np.ndarray],
    chunk_samples: int,
    on_chunk,
) -> None:
    """Keep capturing PCM; only hand audio to Whisper/pyannote in long chunks."""
    chunk_s = chunk_samples / SAMPLE_RATE
    print(
        f"Recording a continuous stream. First transcript after {chunk_s:.0f}s "
        "(not realtime — pyannote needs a long stretch to tell voices apart).",
        flush=True,
    )
    buf: list[np.ndarray] = []
    n = 0
    logged = 0
    try:
        for frame in frames:
            buf.append(frame)
            n += frame.size
            if n - logged >= 10 * SAMPLE_RATE:
                logged = n
                if n < chunk_samples:
                    remain = (chunk_samples - n) / SAMPLE_RATE
                    print(f"  recording {n / SAMPLE_RATE:.0f}s — transcribe in {remain:.0f}s", flush=True)
                else:
                    print(f"  recording {n / SAMPLE_RATE:.0f}s", flush=True)
            if n >= chunk_samples:
                audio = np.concatenate(buf)
                buf = []
                n = 0
                logged = 0
                emit_window(audio, on_chunk, reason="chunk")
    finally:
        leftover = np.concatenate(buf) if buf else None
        if leftover is not None:
            print("Flushing leftover stream …", flush=True)
            emit_window(leftover, on_chunk, reason="stop")


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
                overlap = int(0.4 * SAMPLE_RATE) if n >= max_samples else 0
                if overlap and audio.size > overlap:
                    buf = [np.array(audio[-overlap :], copy=True)]
                else:
                    buf = []
                heard = False
                silent = 0
                vad.in_speech = False
                vad.voiced = 0
                vad.silence = 0
                emit_window(audio, on_window)
    finally:
        leftover = np.concatenate(buf) if buf else None
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


def heartbeat_loop(
    client: AppClient,
    shutdown: threading.Event,
    runtime: "CaptureRuntime",
    device: str | None,
    source: str,
    model: str,
) -> None:
    while not shutdown.wait(3.0):
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
    client.heartbeat(False, device, source, model)


class CaptureRuntime:
    """One session wav on disk. Capture writes; Update notes pulls a range."""

    def __init__(self, args: argparse.Namespace, client: AppClient, source: str) -> None:
        self.args = args
        self.client = client
        self.source = source
        self.lock = threading.Lock()
        self.models_lock = threading.Lock()
        self.writer: PcmWavWriter | None = None
        self.wav_path: Path | None = None
        self.session_id = ""
        self.processing = False
        self.models_ready = False
        self.models_error: str | None = None
        self.bank: SpeakerBank | None = None
        self.embedder: object | None = None
        self.pipeline: PyannoteDiarizer | None = None
        self.recording = True

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            if self.writer is not None:
                self.writer.rewrite_header()
            samples = self.writer.samples if self.writer is not None else 0
            phase = "transcribing" if self.processing else ("recording" if self.recording else "idle")
            return {
                "ok": True,
                "wav": str(self.wav_path) if self.wav_path else None,
                "samples": samples,
                "recorded_s": samples / SAMPLE_RATE if samples else 0.0,
                "processing": self.processing,
                "session_id": self.session_id,
                "phase": phase,
                "models_ready": self.models_ready,
                "error": self.models_error,
            }

    def open_wav(self, session_id: str, path: Path | None = None, append: bool = False) -> Path:
        with self.lock:
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception:
                    pass
                self.writer = None
            self.session_id = session_id or "session"
            if path is None:
                path = new_capture_path(self.session_id)
            self.wav_path = Path(path)
            use_append = append and self.wav_path.is_file()
            self.writer = PcmWavWriter(self.wav_path, append=use_append)
            self.bank = SpeakerBank()
            print(
                f"{'Appending' if use_append else 'Recording'} wav {self.wav_path} "
                f"({self.writer.samples / SAMPLE_RATE:.0f}s already on disk)",
                flush=True,
            )
            return self.wav_path

    def write_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            if self.writer is None:
                return
            self.writer.write_audio(frame)

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

    def ensure_models(self) -> None:
        with self.models_lock:
            with self.lock:
                if self.models_ready:
                    return
                if self.models_error:
                    raise RuntimeError(self.models_error)
            try:
                print("Loading pyannote + Whisper (used when you click Update notes) …", flush=True)
                bank, embedder, pipeline = prepare_speakers(self.args)
                transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32), self.args.model)
                with self.lock:
                    if self.bank is None:
                        self.bank = bank
                    self.embedder = embedder
                    self.pipeline = pipeline
                    self.models_ready = True
                print("Models ready.", flush=True)
            except BaseException as exc:
                err = str(exc) or type(exc).__name__
                self.models_error = err
                print(f"Model load failed: {err}", file=sys.stderr)
                raise

    def process(self, start_s: float, end_s: float, chunk_s: float) -> dict[str, object]:
        with self.lock:
            if self.processing:
                return {"ok": False, "error": "already processing"}
            self.processing = True
            if self.writer is not None:
                self.writer.rewrite_header()
            path = self.wav_path
            recorded_s = self.writer.samples / SAMPLE_RATE if self.writer is not None else 0.0
        try:
            self.ensure_models()
            if path is None or not Path(path).is_file():
                return {"ok": False, "error": "no wav on disk yet"}
            start_s = max(0.0, float(start_s))
            if end_s is None or end_s <= 0:
                tail = RECORD_TAIL_S if self.recording else 0.0
                end_s = max(start_s, recorded_s - tail)
            end_s = max(start_s, float(end_s))
            if self.recording:
                end_s = min(end_s, max(start_s, recorded_s - RECORD_TAIL_S))
            if end_s - start_s < 1.5:
                return {"ok": True, "posted": 0, "start_s": start_s, "end_s": end_s, "skipped": True}
            print(
                f"Transcribe {start_s:.1f}s–{end_s:.1f}s of {path} "
                "(pyannote on the whole recording, then Whisper)",
                flush=True,
            )
            audio = read_wav_range(Path(path), 0.0, end_s)
            actual_end = audio.size / SAMPLE_RATE if audio.size else 0.0
            posted = process_range(
                audio,
                0.0,
                chunk_s,
                transcribe_clip,
                self.client,
                self.args.model,
                self.source,
                self.bank or SpeakerBank(),
                self.embedder,
                self.pipeline,
                on_progress=self.client.send_progress,
                transcribe_from=start_s,
                min_speakers=_speaker_floor(self.args),
                max_speakers=_speaker_ceiling(self.args),
            )
            return {
                "ok": True,
                "posted": posted,
                "start_s": start_s,
                "end_s": actual_end,
                "wav": str(path),
            }
        except Exception as exc:
            print(f"process failed ({type(exc).__name__}): {exc}", file=sys.stderr)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            with self.lock:
                self.processing = False


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
            if path == "/process":
                start_s = float(body.get("start_s") or 0.0)
                end_raw = body.get("end_s")
                end_s = float(end_raw) if end_raw is not None else 0.0
                chunk_s = float(body.get("chunk_s") or DEFAULT_CHUNK_S)
                result = runtime.process(start_s, end_s, chunk_s)
                self._send(200 if result.get("ok") else 500, result)
                return
            self._send(404, {"ok": False, "error": "not found"})

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, name="stenopod-ctl", daemon=True).start()
    print(f"Control API on http://127.0.0.1:{port}  (GET /status, POST /process, POST /rotate)", flush=True)
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


def prepare_speakers(args: argparse.Namespace) -> tuple[SpeakerBank, object, PyannoteDiarizer | None]:
    if args.diarize == "pyannote":
        exact = args.num_speakers
        pipeline = PyannoteDiarizer(
            num_speakers=exact,
            min_speakers=None if exact else args.min_speakers,
            max_speakers=None if exact else args.max_speakers,
        )
        print("Speaker embeddings: pyannote (spectral backup if a turn has none)")
        return SpeakerBank(max_speakers=args.max_speakers), SpectralEmbedder(), pipeline
    embedder, kind = load_embedder()
    print(f"Speaker embeddings: {kind}")
    return (
        SpeakerBank(threshold=0.45, stay_bonus=0.10, max_speakers=args.max_speakers),
        embedder,
        None,
    )


def run_live(args: argparse.Namespace, client: AppClient) -> None:
    source = "both" if args.mic else args.source
    if source == "mic":
        sys_index, device_name = pick_mic_device(args.mic_device)
        mic_index = None
    elif source == "both":
        sys_index, sys_name = pick_device(args.device)
        mic_index, mic_name = pick_mic_device(args.mic_device)
        device_name = f"{sys_name} + {mic_name}"
        print(f"Mixing {sys_name!r} [{sys_index}] with mic {mic_name!r} [{mic_index}]")
        describe_system_route()
    else:
        sys_index, device_name = pick_device(args.device)
        mic_index = None

    print(f"Will listen on {device_name!r} ({source}) → {args.app}")
    if source != "both":
        describe_system_route()
    print(f"Whisper model: {args.model}")
    print(
        "Live capture only writes a wav. Transcription runs after you stop: "
        "pyannote on the whole recording, then Whisper.",
        flush=True,
    )

    runtime = CaptureRuntime(args, client, source)
    session_id = args.session_id or os.environ.get("STENOPOD_SESSION_ID") or "session"
    wav_arg = Path(args.wav) if args.wav else None
    runtime.open_wav(
        session_id,
        path=wav_arg,
        append=bool(wav_arg and wav_arg.is_file()),
    )
    httpd = start_control_server(runtime, args.control_port)

    print(f"  saved wav: {runtime.wav_path}")
    print("Ready. Opening capture …")
    print("Ctrl+C / Stop capture closes the wav. Press again to quit immediately.")
    proc = ffmpeg_pcm(sys_index, mic_index)

    shutdown = threading.Event()
    hb = threading.Thread(
        target=heartbeat_loop,
        args=(client, shutdown, runtime, device_name, source, args.model),
        daemon=True,
    )
    hb.start()
    snap = runtime.snapshot()
    client.heartbeat(
        True,
        device_name,
        source,
        args.model,
        phase="recording",
        recorded_s=float(snap["recorded_s"] or 0),
        wav=str(snap["wav"]) if snap["wav"] else None,
    )

    stopping = threading.Event()

    def request_stop(signum: int, _frame) -> None:
        if stopping.is_set():
            print("\nForce exit. The wav is already on disk.", flush=True)
            sys.exit(128 + signum)
        print("\nStopping capture — closing the wav.", flush=True)
        stopping.set()
        stop_ffmpeg(proc)

    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)

    last_log = 0
    try:
        for frame in frames_from_proc(proc, channels=2 if mic_index is not None else 1):
            runtime.write_frame(frame)
            with runtime.lock:
                samples = runtime.writer.samples if runtime.writer is not None else 0
            if samples - last_log >= 10 * SAMPLE_RATE:
                last_log = samples
                busy = " (transcribing)" if runtime.processing else ""
                print(f"  recording {samples / SAMPLE_RATE:.0f}s{busy}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped recording.", flush=True)
    finally:
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
        deadline = time.time() + 3600
        if runtime.processing:
            print("Waiting for the current transcription job to finish …", flush=True)
        while runtime.processing and time.time() < deadline:
            time.sleep(0.3)
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
    print("Pipeline: pyannote on the full recording, then Whisper")
    client.send_progress({"stage": "load", "chunk": 0, "chunks": 0, "progress": 0})
    bank, embedder, pipeline = prepare_speakers(args)
    client.heartbeat(False, str(path), "file", args.model, phase="transcribing", wav=str(path))
    start_s = max(0.0, float(args.start_s or 0.0))
    if args.end_s is not None:
        audio = read_wav_range(path, 0.0, float(args.end_s))
    else:
        audio = decode_file(path)
    process_range(
        audio,
        0.0,
        args.chunk_seconds,
        transcribe_clip,
        client,
        args.model,
        "file",
        bank,
        embedder,
        pipeline,
        on_progress=client.send_progress,
        transcribe_from=start_s,
        min_speakers=_speaker_floor(args),
        max_speakers=_speaker_ceiling(args),
    )
    client.heartbeat(False, str(path), "file", args.model, phase="idle", wav=str(path))


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
    parser.add_argument("--start-s", type=float, default=0.0, help="With --file, start offset in seconds")
    parser.add_argument("--end-s", type=float, help="With --file, end offset in seconds")
    parser.add_argument("--session-id", help="Session id used in the wav filename")
    parser.add_argument("--wav", help="Reuse/append this wav (same session after Stop/Start)")
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--vad", type=float, default=0.012, help="Legacy energy VAD threshold")
    parser.add_argument(
        "--silero-threshold",
        type=float,
        default=0.5,
        help="Unused. Kept so older commands still parse.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_S,
        help="Unused. Speakers come from one pyannote pass on the whole file.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=12.0,
        help="Unused. Kept so older commands still parse.",
    )
    parser.add_argument("--min-seconds", type=float, default=6.0, help="Unused (kept for old scripts)")
    parser.add_argument(
        "--flush-silence",
        type=float,
        default=0.5,
        help="Unused. Kept so older commands still parse.",
    )
    parser.add_argument(
        "--diarize",
        choices=("vad", "pyannote", "ecapa"),
        default="pyannote",
        help="pyannote (default) labels the whole recording; vad/ecapa are legacy embedders",
    )
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

    if sys.platform != "darwin":
        raise SystemExit("The sidecar is macOS-only (Metal / AVFoundation).")

    client = AppClient(args.app)
    if args.file:
        run_file(args, client)
    else:
        run_live(args, client)


if __name__ == "__main__":
    main()
