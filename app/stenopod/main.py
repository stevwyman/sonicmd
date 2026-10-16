from __future__ import annotations

import asyncio
import logging
import os
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from stenopod.models import Segment, iso, utc_now
from stenopod.prompt import notes_language
from stenopod.store import SessionStore
from stenopod.summarizer import Summarizer
from stenopod.voices import VoiceBank
from stenopod import sidecar_ctl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stenopod")

def _load_repo_env() -> None:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_repo_env()

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8780")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

store = SessionStore(DATA_DIR)
voices = VoiceBank(DATA_DIR / "voices.sqlite")
summarizer = Summarizer(model=LLM_MODEL)
hub: set[WebSocket] = set()
lock = asyncio.Lock()
sidecar_state: dict[str, Any] = {
    "listening": False,
    "device": None,
    "source": None,
    "model": None,
    "phase": None,
    "recorded_s": None,
    "wav": None,
    "last_seen": None,
    "levels": None,
}
# True only while a UI recording is in progress.
ui_recording = False
# Drop "still listening" heartbeats while Stop is tearing the capture process down.
suppress_listening = False
_finish_task: asyncio.Task | None = None
_notes_task: asyncio.Task | None = None
_rerun_backup: dict[str, dict[str, Any]] = {}
_duration_cache: dict[str, tuple[int, int, float]] = {}
_job_cancel = False
STAGE_NAMES = {
    "pyannote": "Speakers",
    "whisper": "Whisper",
    "notes": "Notes",
    "load": "Loading",
}
_job_clock: dict[str, Any] = {
    "stage": None,
    "channel": None,
    "started": None,
    "log": [],
}


class IngestSegment(BaseModel):
    text: str
    lang: str = "und"
    source: str = "system"
    t: str | None = None
    speaker: str = "A"
    duration_ms: int = 0
    speaker_score: float | None = None

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        allowed = {"system", "mic", "both", "file"}
        key = (value or "system").strip().lower()
        return key if key in allowed else "system"

    @field_validator("speaker_score")
    @classmethod
    def finite_score(cls, value: float | None) -> float | None:
        if value is None or not math.isfinite(float(value)):
            return None
        return float(value)


class SidecarStatus(BaseModel):
    listening: bool
    device: str | None = None
    source: str | None = None
    model: str | None = None
    phase: str | None = None
    recorded_s: float | None = None
    wav: str | None = None


class LevelBody(BaseModel):
    channels: int = 1
    left: float = 0.0
    right: float = 0.0

    @field_validator("left", "right")
    @classmethod
    def unit_level(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            return 0.0
        return max(0.0, min(1.0, float(value)))


class SpeakerNameBody(BaseModel):
    id: str = Field(min_length=1, max_length=2)
    name: str = Field(min_length=1, max_length=40)


class TitleBody(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class VoiceVector(BaseModel):
    id: str = Field(min_length=1, max_length=2)
    embedding: list[float]

    @field_validator("embedding")
    @classmethod
    def finite_embedding(cls, value: list[float]) -> list[float]:
        if not 8 <= len(value) <= 4096:
            raise ValueError("voice embedding has an unexpected length")
        if any(not math.isfinite(float(item)) for item in value):
            raise ValueError("voice embedding must be finite")
        return [float(item) for item in value]


class VoiceBody(BaseModel):
    speakers: list[VoiceVector]


class ProgressBody(BaseModel):
    stage: str = "pyannote"
    chunk: int = 1
    chunks: int = 1
    progress: float = 0.0
    channel: str | None = None


class SpeakerBounds(BaseModel):
    """Blank fields mean pyannote chooses the count. Equal min and max means that exact count."""

    min_speakers: int | None = None
    max_speakers: int | None = None

    @field_validator("min_speakers", "max_speakers")
    @classmethod
    def speaker_count(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if not 1 <= value <= 12:
            raise ValueError("speaker count must be from 1 to 12")
        return value

    @model_validator(mode="after")
    def min_not_above_max(self) -> SpeakerBounds:
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("Minimum speakers cannot be higher than the maximum.")
        return self


class RecordingBody(SpeakerBounds):
    path: str = Field(min_length=1)
    restart: bool = False


class CaptureStartBody(BaseModel):
    source: str = "both"
    mic_device: str | None = None

    @field_validator("source")
    @classmethod
    def capture_source(cls, value: str) -> str:
        key = (value or "both").strip().lower()
        if key not in {"system", "mic", "both", "speakers"}:
            raise ValueError("source must be system, mic, both, or speakers")
        return key


async def broadcast(event: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    payload = event
    for ws in list(hub):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        hub.discard(ws)


def require_local(request: Request) -> None:
    host = (request.client.host if request.client else "") or ""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(403, "Capture start/stop is only allowed from this Mac")


def clear_job(session) -> None:
    _close_job_stage()
    session.job_kind = None
    session.job_stage = None
    session.job_progress = None
    session.job_label = None
    session.job_chunk = 0
    session.job_chunks = 0


def _channel_phrase(channel: str | None) -> str:
    if channel == "left":
        return "left channel (you)"
    if channel == "right":
        return "right channel (them)"
    if channel == "mixed":
        return "mixed left+right"
    return ""


def _fmt_secs(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{int(round(seconds))}s"


def reset_job_clock() -> None:
    _job_clock["stage"] = None
    _job_clock["channel"] = None
    _job_clock["started"] = None
    _job_clock["log"] = []


def _close_job_stage() -> None:
    stage = _job_clock.get("stage")
    started = _job_clock.get("started")
    if not stage or started is None:
        _job_clock["stage"] = None
        _job_clock["started"] = None
        return
    elapsed = time.monotonic() - float(started)
    name = STAGE_NAMES.get(str(stage), str(stage))
    phrase = _channel_phrase(_job_clock.get("channel"))
    line = f"{name} · {phrase} · {_fmt_secs(elapsed)}" if phrase else f"{name} · {_fmt_secs(elapsed)}"
    lines = list(_job_clock.get("log") or [])
    lines.append(line)
    _job_clock["log"] = lines[-12:]
    _job_clock["stage"] = None
    _job_clock["started"] = None


def _open_job_stage(stage: str, channel: str | None) -> None:
    if (
        _job_clock.get("stage") == stage
        and _job_clock.get("channel") == channel
        and _job_clock.get("started") is not None
    ):
        return
    _close_job_stage()
    _job_clock["stage"] = stage
    _job_clock["channel"] = channel
    _job_clock["started"] = time.monotonic()


def job_clock_public() -> dict[str, Any]:
    started = _job_clock.get("started")
    elapsed = time.monotonic() - float(started) if started is not None else None
    return {
        "channel": _job_clock.get("channel"),
        "log": list(_job_clock.get("log") or []),
        "elapsed_s": elapsed,
    }


def _live_job_label(session, extra: dict[str, Any]) -> str | None:
    if not session.job_kind:
        return None
    stage = session.job_stage
    phrase = _channel_phrase(extra.get("channel"))
    elapsed = extra.get("elapsed_s")
    tail = f" · {_fmt_secs(elapsed)}" if elapsed is not None else ""
    chunk = int(session.job_chunk or 0)
    chunks = int(session.job_chunks or 0)
    if stage == "load":
        return f"Loading speakers and Whisper{tail}"
    if stage == "notes":
        return f"Notes{tail}"
    if stage == "pyannote":
        base = f"Speakers · {phrase}" if phrase else "Speakers"
        return f"{base}{tail}"
    if stage == "whisper":
        base = f"Whisper · {phrase}" if phrase else "Whisper"
        if chunks > 0:
            base = f"{base} · {chunk} of {chunks}"
        return f"{base}{tail}"
    return (session.job_label or STAGE_NAMES.get(stage or "", stage or "")) + tail


def recording_active() -> bool:
    return sidecar_ctl.running_pid() is not None


def sync_ui_recording() -> None:
    """If the capture process died, the Stop button must not stay latched on."""
    global ui_recording
    if ui_recording and not recording_active():
        ui_recording = False


def captures_dir() -> Path:
    return (sidecar_ctl.REPO_ROOT / ".run" / "captures").resolve()


def cached_wav_duration(path: Path) -> float:
    st = path.stat()
    key = str(path.resolve())
    hit = _duration_cache.get(key)
    if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    dur = sidecar_ctl.wav_duration_s(path)
    _duration_cache[key] = (st.st_mtime_ns, st.st_size, dur)
    return dur


def recording_label(wav: Path, title: str | None) -> str:
    cleaned = (title or "").strip()
    if cleaned and cleaned != "Untitled session":
        return cleaned
    parts = wav.stem.split("-")
    if (
        len(parts) >= 3
        and parts[-2].isdigit()
        and len(parts[-2]) == 8
        and parts[-1].isdigit()
        and len(parts[-1]) == 6
    ):
        day, clock = parts[-2], parts[-1]
        return f"{day[0:4]}-{day[4:6]}-{day[6:8]} {clock[0:2]}:{clock[2:4]}"
    return wav.stem


def capture_wav(path: str) -> Path:
    root = captures_dir()
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.suffix.lower() != ".wav":
        raise HTTPException(400, "That recording is not on disk.")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, "Recording must live in the captures folder.") from exc
    return target


def list_recordings() -> list[dict[str, Any]]:
    root = captures_dir()
    if not root.is_dir():
        return []
    index = store.sessions_by_audio()
    rows: list[dict[str, Any]] = []
    wavs = sorted(root.glob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True)
    for wav in wavs:
        try:
            if wav.stat().st_size <= 44:
                continue
            dur = cached_wav_duration(wav)
        except OSError:
            continue
        except Exception:
            log.exception("Could not read %s", wav)
            continue
        key = str(wav.resolve())
        meta = index.get(key) or {}
        cursor = float(meta.get("cursor") or 0.0)
        segments = int(meta.get("segments") or 0)
        too_short = dur < 1.0
        transcribed = (not too_short) and segments > 0 and cursor >= dur - 1.0
        recorded_at = datetime.fromtimestamp(wav.stat().st_mtime, timezone.utc).isoformat()
        rows.append(
            {
                "path": key,
                "name": wav.name,
                "title": recording_label(wav, meta.get("title")),
                "session_id": meta.get("id"),
                "duration_s": dur,
                "segments": segments,
                "cursor_s": cursor,
                "transcribed": transcribed,
                "too_short": too_short,
                "current": meta.get("id") == store.current.id,
                "recorded_at": recorded_at,
            }
        )
    return rows


def current_has_recording() -> bool:
    if store.current.segments:
        return True
    path = store.current.audio_path
    if not path:
        return False
    wav = Path(path)
    if not wav.is_file():
        return False
    dur = float(store.current.audio_duration_s or 0.0)
    if dur <= 0:
        try:
            dur = cached_wav_duration(wav)
        except Exception:
            return False
    return dur >= 1.0


def activate_recording(wav: Path, duration_s: float, restart: bool) -> None:
    found = store.find_id_for_audio(wav)
    if found and found != store.current.id:
        store.load(found)
    elif not found and current_has_recording():
        store.bind_wav(wav, duration_s, recording_label(wav, None))
    elif not found:
        store.current.audio_path = str(wav.resolve())
        store.current.audio_cursor_s = 0.0
        if store.current.title in {"", "Untitled session"}:
            store.current.title = recording_label(wav, None)
    store.current.audio_path = str(wav.resolve())
    store.current.audio_duration_s = duration_s
    if restart:
        session = store.current
        _rerun_backup[session.id] = {
            "segments": list(session.segments),
            "notes": session.notes,
            "notes_updated_at": session.notes_updated_at,
            "last_summarized_index": session.last_summarized_index,
            "speaker_names": dict(session.speaker_names),
            "speaker_embeddings": {key: list(value) for key, value in session.speaker_embeddings.items()},
            "audio_cursor_s": float(session.audio_cursor_s or 0.0),
        }
        session.segments.clear()
        session.audio_cursor_s = 0.0
        session.notes = ""
        session.notes_updated_at = None
        session.last_summarized_index = 0
        session.speaker_names = {}
        session.speaker_embeddings = {}
        session.summary_error = None
        session.transcribe_error = None
        clear_job(session)
    store.save()


def transcribe_running() -> bool:
    return _finish_task is not None and not _finish_task.done()


def notes_running() -> bool:
    return _notes_task is not None and not _notes_task.done()


def job_busy() -> bool:
    return bool(
        ui_recording
        or recording_active()
        or store.current.transcribing
        or store.current.summarizing
        or transcribe_running()
        or notes_running()
    )


def remember_wav_duration() -> None:
    session = store.current
    path = session.audio_path
    if session.audio_duration_s > 0 or not path or recording_active():
        return
    wav = Path(path)
    if not wav.is_file():
        return
    try:
        session.audio_duration_s = sidecar_ctl.wav_duration_s(wav)
        store.save()
    except Exception:
        log.exception("Could not read wav duration")


def public_state() -> dict[str, Any]:
    sync_ui_recording()
    remember_wav_duration()
    listening = False
    last = sidecar_state.get("last_seen")
    if sidecar_state.get("listening") and isinstance(last, datetime) and recording_active():
        listening = (utc_now() - last).total_seconds() < 8
    recorded = sidecar_state.get("recorded_s")
    cursor = float(store.current.audio_cursor_s or 0.0)
    duration = float(store.current.audio_duration_s or 0.0)
    if listening and isinstance(recorded, (int, float)):
        duration = float(recorded)
    pending_audio_s = max(0.0, duration - cursor) if duration else None
    snap = store.current.snapshot()
    snap["pending_audio_s"] = pending_audio_s
    snap["audio_duration_s"] = duration or snap.get("audio_duration_s") or 0.0
    try:
        recordings = list_recordings()
    except Exception:
        log.exception("Could not list recordings")
        recordings = []
    out = {
        **snap,
        "recordings": recordings,
        "sidecar": {
            "listening": listening,
            "device": sidecar_state.get("device"),
            "source": sidecar_state.get("source"),
            "model": sidecar_state.get("model"),
            "phase": sidecar_state.get("phase"),
            "recorded_s": recorded if listening else snap["audio_duration_s"],
            "wav": sidecar_state.get("wav") or store.current.audio_path,
            "levels": sidecar_state.get("levels") if listening else None,
        },
        "capture": sidecar_ctl.status(),
    }
    job = out.get("job")
    if isinstance(job, dict):
        extra = job_clock_public()
        job.update(extra)
        live = _live_job_label(store.current, extra)
        if live:
            job["label"] = live
    return out


async def transcribe_pending(
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    """Run pyannote on the whole wav, then Whisper on each turn."""
    async with lock:
        session = store.current
        if session.transcribing:
            raise RuntimeError("already transcribing")
        if recording_active():
            raise RuntimeError("stop recording first")
        session_id = session.id
        cursor = float(session.audio_cursor_s or 0.0)
        audio_path = session.audio_path
        if not audio_path or not Path(audio_path).is_file():
            raise RuntimeError("no recording on disk")
        session.transcribing = True
        session.transcribe_error = None
        session.min_speakers = min_speakers
        session.max_speakers = max_speakers
        session.speaker_bounds_saved = True
        session.job_kind = "transcribe"
        session.job_stage = "load"
        session.job_progress = None
        session.job_label = "Loading speakers and Whisper"
        session.job_chunk = 0
        session.job_chunks = 0
        _open_job_stage("load", None)
        store.save()

    await broadcast({"type": "state", "state": public_state()})
    try:
        recorded_s = await asyncio.to_thread(sidecar_ctl.wav_duration_s, audio_path)
        async with lock:
            if store.current.id == session_id:
                store.current.audio_duration_s = recorded_s
                store.save()
        if recorded_s - cursor < 1.0:
            async with lock:
                if store.current.id == session_id:
                    clear_job(store.current)
                    store.current.audio_duration_s = recorded_s
                    if recorded_s < 1.0:
                        store.current.transcribe_error = (
                            "Recording captured no audio. Set Sound → Output to the StenoPod device "
                            "you selected. For Headphones, Input must be the MacBook microphone."
                        )
                    elif not store.current.segments:
                        store.current.transcribe_error = "No speech found in this recording."
                    store.save()
            return {"ok": True, "skipped": True, "start_s": cursor, "end_s": recorded_s, "wav": audio_path}
        if _job_cancel:
            async with lock:
                if store.current.id == session_id:
                    store.current.transcribe_error = None
                    clear_job(store.current)
                    store.save()
            return {"ok": True, "stopped": True, "skipped": True, "start_s": cursor, "end_s": recorded_s, "wav": audio_path}
        log.info("Transcribing wav %s from %.1fs to %.1fs", audio_path, cursor, recorded_s)
        result = await asyncio.to_thread(
            sidecar_ctl.process_file,
            audio_path,
            cursor,
            recorded_s,
            APP_URL,
            min_speakers,
            max_speakers,
        )
        if result.get("stopped") or _job_cancel:
            async with lock:
                if store.current.id == session_id:
                    store.current.transcribe_error = None
                    clear_job(store.current)
                    store.save()
            return {"ok": True, "stopped": True, "skipped": True, "start_s": cursor, "end_s": recorded_s, "wav": audio_path}
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "transcription failed"))
        used_end = float(result.get("end_s") or recorded_s)
        async with lock:
            if store.current.id != session_id:
                return result
            if used_end > cursor:
                store.current.audio_cursor_s = used_end
            store.current.audio_path = str(result.get("wav") or audio_path)
            store.current.audio_duration_s = recorded_s
            store.save()
        return result
    except Exception as exc:
        err = str(exc)
        log.exception("Transcription failed")
        async with lock:
            if store.current.id == session_id:
                store.current.transcribe_error = err
                clear_job(store.current)
                store.save()
        raise
    finally:
        async with lock:
            if store.current.id == session_id:
                store.current.transcribing = False
                store.save()
        await broadcast({"type": "state", "state": public_state()})


async def maybe_summarize(reason: str, force: bool = False, rewrite: bool = False) -> None:
    async with lock:
        session = store.current
        if session.summarizing:
            return
        if session.transcribing:
            return
        if recording_active():
            return
        pending = session.pending_text(0) if rewrite else session.pending_text()
        if not pending:
            return
        if not force and not rewrite and len(pending) < 200:
            return
        session.summarizing = True
        session.summary_error = None
        session.job_kind = "notes"
        session.job_stage = "notes"
        session.job_progress = None
        session.job_label = "Notes"
        session.job_chunk = 0
        session.job_chunks = 0
        _open_job_stage("notes", None)
        # A full rewrite uses the current speaker names and ignores stale notes.
        notes = "" if rewrite else session.notes
        session_id = session.id
        summarized_index = len(session.segments)
        language = notes_language(session.segments)

    await broadcast({"type": "state", "state": public_state()})
    log.info("Summarizing (%s) pending_chars=%s rewrite=%s", reason, len(pending), rewrite)
    try:
        notes_out = await summarizer.summarize(notes, pending, language)
        async with lock:
            if store.current.id != session_id:
                return
            store.current.summarizing = False
            clear_job(store.current)
            store.persist_notes(notes_out, summarized_index)
    except Exception as exc:
        if _job_cancel or str(exc) == "stopped":
            log.info("Notes stopped")
            async with lock:
                if store.current.id == session_id:
                    store.current.summarizing = False
                    clear_job(store.current)
                    store.save()
            await broadcast({"type": "state", "state": public_state()})
            return
        log.exception("Summary failed")
        err = str(exc)
        if "not in the local Hugging Face cache" in err or "Host venv missing" in err:
            err = (
                f"Notes model {LLM_MODEL} is not ready. "
                "Run `make models` once while online, then transcribe again."
            )
        async with lock:
            if store.current.id == session_id:
                store.current.summarizing = False
                store.current.summary_error = err
                clear_job(store.current)
                store.save()
    await broadcast({"type": "state", "state": public_state()})


async def _store_transcribe_error(message: str) -> None:
    if message == "already transcribing":
        return
    async with lock:
        if not store.current.transcribe_error:
            store.current.transcribe_error = message
        clear_job(store.current)
        store.save()
    await broadcast({"type": "state", "state": public_state()})


async def _clear_transcribe_job() -> None:
    async with lock:
        if store.current.job_kind == "transcribe" and not store.current.transcribing:
            clear_job(store.current)
            store.save()
    await broadcast({"type": "state", "state": public_state()})


async def _restore_rerun_if_empty() -> None:
    """Put the previous transcript back when a rerun exits before any new turns."""
    async with lock:
        session = store.current
        backup = _rerun_backup.get(session.id)
        if backup is None:
            return
        if session.segments:
            _rerun_backup.pop(session.id, None)
            return
        session.segments = backup["segments"]
        session.notes = backup["notes"]
        session.notes_updated_at = backup["notes_updated_at"]
        session.last_summarized_index = backup["last_summarized_index"]
        session.speaker_names = backup["speaker_names"]
        session.speaker_embeddings = backup.get("speaker_embeddings") or {}
        session.audio_cursor_s = backup["audio_cursor_s"]
        store.save()
        _rerun_backup.pop(session.id, None)
    await broadcast({"type": "state", "state": public_state()})


async def finish_recording(
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> None:
    try:
        result = await transcribe_pending(min_speakers, max_speakers)
    except RuntimeError as exc:
        await _store_transcribe_error(str(exc))
        await _restore_rerun_if_empty()
        return
    except Exception:
        log.exception("Transcription failed")
        await _clear_transcribe_job()
        await _restore_rerun_if_empty()
        return
    if result.get("stopped"):
        await _restore_rerun_if_empty()
        await _clear_transcribe_job()
        return
    if result.get("skipped"):
        await _clear_transcribe_job()
        await _restore_rerun_if_empty()
        return
    if _job_cancel:
        await _restore_rerun_if_empty()
        await _clear_transcribe_job()
        return
    async with lock:
        if not store.current.segments and not store.current.transcribe_error:
            store.current.transcribe_error = "No speech found in this recording."
            store.save()
    await _restore_rerun_if_empty()
    await _clear_transcribe_job()


async def begin_transcription(
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    global _finish_task, _job_cancel
    if ui_recording or recording_active():
        raise HTTPException(409, "Stop recording before transcribing.")
    if store.current.summarizing or notes_running():
        raise HTTPException(409, "Notes are still being written.")
    if store.current.transcribing or transcribe_running():
        return public_state()
    path = store.current.audio_path
    if not path or not Path(path).is_file():
        raise HTTPException(400, "No recording on disk for this meeting.")
    _job_cancel = False
    reset_job_clock()
    async with lock:
        session = store.current
        dur = float(session.audio_duration_s or 0.0)
        cursor = float(session.audio_cursor_s or 0.0)
        if dur >= 1.0 and not session.segments and cursor >= dur - 1.0:
            session.audio_cursor_s = 0.0
            session.transcribe_error = None
            store.save()
    log.info(
        "Starting transcription for %s (speakers min=%s max=%s)",
        path,
        min_speakers,
        max_speakers,
    )
    _finish_task = asyncio.create_task(finish_recording(min_speakers, max_speakers))
    for _ in range(50):
        if store.current.transcribing or store.current.transcribe_error or _finish_task.done():
            break
        await asyncio.sleep(0.05)
    return public_state()


async def resume_unfinished_recording() -> None:
    """A stop that died after the wav closed never saved the duration or started transcription."""
    await asyncio.sleep(0.3)
    if ui_recording or recording_active() or transcribe_running():
        return
    session = store.current
    path = session.audio_path
    if not path or not Path(path).is_file():
        return
    try:
        dur = await asyncio.to_thread(sidecar_ctl.wav_duration_s, path)
    except Exception:
        log.exception("Could not read wav duration on startup")
        return
    cursor = float(session.audio_cursor_s or 0.0)
    if dur < 1.0 or dur - cursor < 1.0:
        return
    if session.segments and cursor >= dur - 1.0:
        return
    async with lock:
        current = store.current
        if current.id != session.id:
            return
        cur = float(current.audio_cursor_s or 0.0)
        if current.segments and cur >= dur - 1.0:
            return
        current.audio_duration_s = dur
        current.transcribe_error = None
        store.save()
    log.info("Recording %s was closed but not transcribed (%.0fs). Press Transcribe.", path, dur)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(sidecar_ctl.running_pid)
    resume_task = asyncio.create_task(resume_unfinished_recording())
    yield
    resume_task.cancel()
    await asyncio.to_thread(sidecar_ctl.stop, True)


app = FastAPI(title="StenoPod", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    logo = STATIC_DIR / "logo.jpg"
    if not logo.is_file():
        logo = sidecar_ctl.REPO_ROOT / "doc" / "logo.jpg"
    if not logo.is_file():
        raise HTTPException(404, "logo missing")
    return FileResponse(logo, media_type="image/jpeg")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    llm_ok = await summarizer.healthy()
    return {"app": True, "llm": llm_ok, "sidecar": public_state()["sidecar"], "capture": sidecar_ctl.status()}


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return public_state()


@app.post("/api/segments")
async def add_segment(body: IngestSegment) -> dict[str, Any]:
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty transcript")
    stamp = body.t or iso(utc_now())
    speaker = (body.speaker or "A").strip().upper()[:1] or "A"
    segment = Segment(
        t=stamp,
        lang=body.lang or "und",
        source=body.source,
        text=text,
        speaker=speaker,
        duration_ms=max(0, body.duration_ms),
        speaker_score=body.speaker_score,
    )
    async with lock:
        store.current.segments.append(segment)
        if len(store.current.segments) == 1 and store.current.title == "Untitled session":
            store.current.title = text[:72]
        store.save()
    await broadcast({"type": "segment", "segment": segment.model_dump(), "state": public_state()})
    return {"ok": True, "id": segment.id}


@app.post("/api/sidecar")
async def sidecar_heartbeat(body: SidecarStatus) -> dict[str, Any]:
    if body.listening and suppress_listening:
        if body.wav:
            sidecar_state["wav"] = body.wav
        return {"ok": True}
    sidecar_state.update(
        {
            "listening": body.listening,
            "device": body.device,
            "source": body.source,
            "model": body.model,
            "phase": body.phase,
            "recorded_s": body.recorded_s,
            "wav": body.wav,
            "last_seen": utc_now(),
        }
    )
    if not body.listening:
        sidecar_state["levels"] = None
    await broadcast({"type": "state", "state": public_state()})
    return {"ok": True}


def _clip_level(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


@app.post("/api/sidecar/levels")
async def sidecar_levels(body: LevelBody) -> dict[str, Any]:
    if suppress_listening or not sidecar_state.get("listening"):
        return {"ok": True, "ignored": True}
    levels = {
        "channels": 2 if int(body.channels) >= 2 else 1,
        "left": _clip_level(body.left),
        "right": _clip_level(body.right),
    }
    sidecar_state["levels"] = levels
    await broadcast({"type": "levels", "levels": levels})
    return {"ok": True}


@app.post("/api/progress")
async def job_progress(request: Request, body: ProgressBody) -> dict[str, Any]:
    require_local(request)
    stage = (body.stage or "pyannote").strip().lower()
    name = STAGE_NAMES.get(stage, stage)
    chunks = max(0, int(body.chunks or 0))
    chunk = max(0, int(body.chunk or 0))
    progress = body.progress
    if progress is None or not math.isfinite(float(progress)):
        progress = None
    else:
        progress = max(0.0, min(1.0, float(progress)))
    channel = (body.channel or "").strip().lower() or None
    if channel not in {"left", "right", "mixed"}:
        channel = None
    _open_job_stage(stage, channel)
    extra = job_clock_public()
    if stage == "load":
        progress = None
    async with lock:
        session = store.current
        session.job_kind = "transcribe" if stage != "notes" else "notes"
        session.job_stage = stage
        session.job_progress = progress
        session.job_chunk = chunk
        session.job_chunks = chunks
        session.job_label = _live_job_label(session, extra) or name
    await broadcast({"type": "state", "state": public_state()})
    return {"ok": True}


@app.post("/api/capture/start")
async def capture_start(request: Request, body: CaptureStartBody | None = None) -> dict[str, Any]:
    global ui_recording
    require_local(request)
    payload = body or CaptureStartBody()
    if recording_active():
        ui_recording = True
        await broadcast({"type": "state", "state": public_state()})
        return {"ok": True, "already": True, "state": public_state()}
    if ui_recording:
        ui_recording = False
    if job_busy():
        raise HTTPException(409, "Wait for transcription and notes to finish.")
    existing = store.current.audio_path
    if existing and Path(existing).is_file():
        dur = store.current.audio_duration_s
        if dur <= 0:
            try:
                dur = sidecar_ctl.wav_duration_s(existing)
            except Exception:
                dur = 0
        if dur > 1.0:
            raise HTTPException(409, "This meeting already has a recording. Start a new meeting first.")
    try:
        # Speakers is the meeting mix only. Headphones is the stereo mic + meeting capture.
        capture_source = "system" if payload.source in {"speakers", "system"} else payload.source
        result = await asyncio.to_thread(
            sidecar_ctl.start,
            capture_source,
            payload.mic_device if capture_source == "both" else None,
            APP_URL,
            store.current.id,
            None,
        )
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    if result.get("already"):
        raise HTTPException(409, "A recording is already running.")
    ui_recording = True
    wav = result.get("wav")
    async with lock:
        store.current.capture_source = payload.source
        store.current.audio_cursor_s = 0.0
        store.current.audio_duration_s = 0.0
        store.current.transcribe_error = None
        if wav:
            store.current.audio_path = str(wav)
        store.save()
    await broadcast({"type": "state", "state": public_state()})
    return {**result, "state": public_state()}


@app.post("/api/capture/stop")
async def capture_stop(request: Request, body: SpeakerBounds | None = None) -> dict[str, Any]:
    global ui_recording, suppress_listening
    require_local(request)
    ui_recording = False
    suppress_listening = True
    sidecar_state["listening"] = False
    sidecar_state["phase"] = None
    sidecar_state["last_seen"] = None
    path = store.current.audio_path or sidecar_state.get("wav")
    try:
        result = await asyncio.to_thread(sidecar_ctl.stop)
    finally:
        suppress_listening = False
        sidecar_state["listening"] = False
        sidecar_state["phase"] = None
        sidecar_state["last_seen"] = None
    async with lock:
        if path:
            store.current.audio_path = str(path)
        wav = store.current.audio_path
        dur = 0.0
        if wav and Path(wav).is_file():
            try:
                dur = sidecar_ctl.wav_duration_s(wav)
                store.current.audio_duration_s = dur
                resolved = str(Path(wav).resolve())
                st = Path(wav).stat()
                _duration_cache[resolved] = (st.st_mtime_ns, st.st_size, dur)
            except Exception:
                log.exception("Could not read wav duration after stop")
        if dur < 1.0:
            store.current.transcribe_error = (
                "Recording captured no audio. Set Sound → Output to the StenoPod device "
                "you selected. For Headphones, Input must be the MacBook microphone."
            )
        else:
            store.current.transcribe_error = None
        store.save()
    await broadcast({"type": "state", "state": public_state()})
    return {**result, "state": public_state(), "transcribe_started": False}


@app.post("/api/transcribe")
async def transcribe_now(body: SpeakerBounds | None = None) -> dict[str, Any]:
    bounds = body or SpeakerBounds()
    return await begin_transcription(bounds.min_speakers, bounds.max_speakers)


@app.post("/api/transcribe/stop")
async def transcribe_stop() -> dict[str, Any]:
    global _job_cancel
    _job_cancel = True
    await asyncio.to_thread(sidecar_ctl.stop_file)
    summarizer.cancel()
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.post("/api/recordings/open")
async def open_recording(body: RecordingBody) -> dict[str, Any]:
    if job_busy():
        raise HTTPException(409, "Wait for recording, transcription, and notes to finish.")
    wav = capture_wav(body.path)
    try:
        dur = await asyncio.to_thread(cached_wav_duration, wav)
    except Exception as exc:
        raise HTTPException(400, f"Could not read the recording: {exc}") from exc
    async with lock:
        activate_recording(wav, dur, restart=False)
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.post("/api/recordings/transcribe")
async def transcribe_recording(body: RecordingBody) -> dict[str, Any]:
    if job_busy():
        raise HTTPException(409, "Wait for recording, transcription, and notes to finish.")
    wav = capture_wav(body.path)
    try:
        dur = await asyncio.to_thread(cached_wav_duration, wav)
    except Exception as exc:
        raise HTTPException(400, f"Could not read the recording: {exc}") from exc
    if dur < 1.0:
        raise HTTPException(400, "Recording is too short to transcribe.")
    async with lock:
        activate_recording(wav, dur, restart=body.restart)
        done = (
            not body.restart
            and len(store.current.segments) > 0
            and float(store.current.audio_cursor_s or 0.0) >= dur - 1.0
        )
    await broadcast({"type": "state", "state": public_state()})
    if done:
        return public_state()
    return await begin_transcription(body.min_speakers, body.max_speakers)


@app.post("/api/recordings/delete")
async def delete_recording(body: RecordingBody) -> dict[str, Any]:
    if job_busy():
        raise HTTPException(409, "Wait for recording, transcription, and notes to finish.")
    wav = capture_wav(body.path)
    async with lock:
        removed = store.forget_wav(wav)
        _duration_cache.pop(str(wav.resolve()), None)
        for session_id in removed:
            _rerun_backup.pop(session_id, None)
        pointed = sidecar_state.get("wav")
        if pointed:
            try:
                same = Path(str(pointed)).resolve() == wav.resolve()
            except OSError:
                same = False
            if same:
                sidecar_state["wav"] = store.current.audio_path
                sidecar_state["recorded_s"] = None
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.post("/api/summarize")
async def summarize_now() -> dict[str, Any]:
    global _notes_task, _job_cancel
    if ui_recording or recording_active():
        raise HTTPException(409, "Stop recording before updating notes.")
    if store.current.transcribing or transcribe_running():
        raise HTTPException(409, "Wait for transcription to finish.")
    if not store.current.segments:
        raise HTTPException(400, "Transcribe the recording before writing notes.")
    if store.current.summarizing or notes_running():
        return public_state()
    _job_cancel = False
    reset_job_clock()
    _notes_task = asyncio.create_task(maybe_summarize("manual", force=True, rewrite=True))
    for _ in range(50):
        if store.current.summarizing or store.current.summary_error or _notes_task.done():
            break
        await asyncio.sleep(0.05)
    return public_state()


@app.post("/api/session")
async def new_session() -> dict[str, Any]:
    if ui_recording or recording_active():
        raise HTTPException(409, "Stop recording before starting a new meeting.")
    async with lock:
        if job_busy():
            raise HTTPException(409, "Wait for transcription and notes to finish.")
        store.new_session()
    sidecar_state["listening"] = False
    sidecar_state["wav"] = None
    sidecar_state["recorded_s"] = None
    sidecar_state["phase"] = None
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.post("/api/speakers/voices")
async def match_voices(request: Request, body: VoiceBody) -> dict[str, Any]:
    require_local(request)
    async with lock:
        session_id = store.current.id
        pairs: list[tuple[str, list[float]]] = []
        for item in body.speakers:
            letter = item.id.strip().upper()[:1]
            if not letter.isalpha():
                continue
            store.current.speaker_embeddings[letter] = list(item.embedding)
            pairs.append((letter, list(item.embedding)))
        store.save()
    matched: dict[str, str] = {}
    if pairs:
        matched = await asyncio.to_thread(voices.identify, pairs)
    async with lock:
        if store.current.id == session_id:
            for letter, name in matched.items():
                if not str(store.current.speaker_names.get(letter) or "").strip():
                    store.current.speaker_names[letter] = name
            store.save()
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.post("/api/speakers")
async def rename_speaker(body: SpeakerNameBody) -> dict[str, Any]:
    speaker_id = body.id.strip().upper()[:1]
    if not speaker_id.isalpha():
        raise HTTPException(400, "invalid speaker id")
    async with lock:
        name = body.name.strip()
        store.current.speaker_names[speaker_id] = name
        embedding = store.current.speaker_embeddings.get(speaker_id) or []
        store.save()
    if embedding:
        await asyncio.to_thread(voices.remember, name, embedding)
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.post("/api/title")
async def set_title(body: TitleBody) -> dict[str, Any]:
    async with lock:
        store.current.title = body.title.strip()
        store.save()
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.get("/api/transcript.txt")
async def download_transcript() -> PlainTextResponse:
    return PlainTextResponse(
        store.current.transcript_text(),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=transcript.txt"},
    )


@app.get("/api/notes.md")
async def download_notes() -> PlainTextResponse:
    body = store.current.notes or "# Notes\n\n_No notes yet._\n"
    return PlainTextResponse(
        body,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=notes.md"},
    )


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    hub.add(ws)
    await ws.send_json({"type": "state", "state": public_state()})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.discard(ws)
