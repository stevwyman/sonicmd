from __future__ import annotations

import asyncio
import logging
import os
import math
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from stenopod.models import Segment, iso, utc_now
from stenopod.store import SessionStore
from stenopod.summarizer import Summarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stenopod")

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8081")
AUTO_SUMMARY_CHARS = int(os.environ.get("AUTO_SUMMARY_CHARS", "1800"))
AUTO_SUMMARY_SECONDS = int(os.environ.get("AUTO_SUMMARY_SECONDS", "90"))
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

store = SessionStore(DATA_DIR)
summarizer = Summarizer(LLM_URL)
hub: set[WebSocket] = set()
lock = asyncio.Lock()
sidecar_state: dict[str, Any] = {
    "listening": False,
    "device": None,
    "source": None,
    "model": None,
    "last_seen": None,
}
idle_task: asyncio.Task[None] | None = None


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


class SpeakerNameBody(BaseModel):
    id: str = Field(min_length=1, max_length=2)
    name: str = Field(min_length=1, max_length=40)


class TitleBody(BaseModel):
    title: str = Field(min_length=1, max_length=120)


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


def public_state() -> dict[str, Any]:
    listening = False
    last = sidecar_state.get("last_seen")
    if sidecar_state.get("listening") and isinstance(last, datetime):
        listening = (utc_now() - last).total_seconds() < 8
    return {
        **store.current.snapshot(),
        "sidecar": {
            "listening": listening,
            "device": sidecar_state.get("device"),
            "source": sidecar_state.get("source"),
            "model": sidecar_state.get("model"),
        },
    }


async def maybe_summarize(reason: str, force: bool = False) -> None:
    async with lock:
        session = store.current
        if session.summarizing:
            return
        pending = session.pending_text()
        if not pending:
            return
        if not force and len(pending) < 200:
            return
        if not force and len(pending) < AUTO_SUMMARY_CHARS:
            return
        session.summarizing = True
        session.summary_error = None
        notes = session.notes
        session_id = session.id
        summarized_index = len(session.segments)

    await broadcast({"type": "state", "state": public_state()})
    log.info("Summarizing (%s) pending_chars=%s", reason, len(pending))
    try:
        notes_out = await summarizer.summarize(notes, pending)
        async with lock:
            if store.current.id != session_id:
                return
            store.current.summarizing = False
            store.persist_notes(notes_out, summarized_index)
    except Exception as exc:
        log.exception("Summary failed")
        err = str(exc)
        if "All connection attempts failed" in err or "ConnectError" in type(exc).__name__:
            err = f"Cannot reach the notes model at {LLM_URL}. Start it with `make up`."
        async with lock:
            store.current.summarizing = False
            store.current.summary_error = err
            store.save()
    await broadcast({"type": "state", "state": public_state()})


async def idle_watchdog() -> None:
    while True:
        await asyncio.sleep(5)
        async with lock:
            session = store.current
            if session.summarizing or not session.segments:
                continue
            pending = session.pending_text()
            if len(pending) < 200:
                continue
            last_seg = session.segments[-1]
            try:
                last_dt = datetime.fromisoformat(last_seg.t.replace("Z", "+00:00"))
            except ValueError:
                continue
            quiet = (utc_now() - last_dt).total_seconds()
        if quiet >= AUTO_SUMMARY_SECONDS:
            await maybe_summarize("idle")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global idle_task
    idle_task = asyncio.create_task(idle_watchdog())
    yield
    idle_task.cancel()


app = FastAPI(title="StenoPod", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    llm_ok = await summarizer.healthy()
    return {"app": True, "llm": llm_ok, "sidecar": public_state()["sidecar"]}


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
        pending_len = len(store.current.pending_text())
    await broadcast({"type": "segment", "segment": segment.model_dump(), "state": public_state()})
    if pending_len >= AUTO_SUMMARY_CHARS:
        asyncio.create_task(maybe_summarize("chars"))
    return {"ok": True, "id": segment.id}


@app.post("/api/sidecar")
async def sidecar_heartbeat(body: SidecarStatus) -> dict[str, Any]:
    sidecar_state.update(
        {
            "listening": body.listening,
            "device": body.device,
            "source": body.source,
            "model": body.model,
            "last_seen": utc_now(),
        }
    )
    await broadcast({"type": "state", "state": public_state()})
    return {"ok": True}


@app.post("/api/summarize")
async def summarize_now() -> dict[str, Any]:
    await maybe_summarize("manual", force=True)
    return public_state()


@app.post("/api/session")
async def new_session() -> dict[str, Any]:
    async with lock:
        store.new_session()
    await broadcast({"type": "state", "state": public_state()})
    return public_state()


@app.post("/api/speakers")
async def rename_speaker(body: SpeakerNameBody) -> dict[str, Any]:
    speaker_id = body.id.strip().upper()[:1]
    if not speaker_id.isalpha():
        raise HTTPException(400, "invalid speaker id")
    async with lock:
        store.current.speaker_names[speaker_id] = body.name.strip()
        store.save()
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
