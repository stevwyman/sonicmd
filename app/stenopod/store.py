from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stenopod.models import Session, utc_now


class SessionStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current = Session()
        self._load_latest()

    def _session_path(self, session: Session) -> Path:
        return self.data_dir / f"{session.id}.json"

    def _load_latest(self) -> None:
        files = sorted(self.data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return
        try:
            payload = json.loads(files[0].read_text())
            self.current = Session.model_validate(payload)
            # Runtime-only: a restart never has an in-flight LLM job.
            self.current.summarizing = False
            self.current.transcribing = False
        except Exception:
            self.current = Session()

    def save(self) -> None:
        path = self._session_path(self.current)
        path.write_text(self.current.model_dump_json(indent=2))

    def new_session(self) -> Session:
        self.save()
        self.current = Session()
        self.save()
        return self.current

    def load(self, session_id: str) -> Session:
        path = self.data_dir / f"{session_id}.json"
        if not path.is_file():
            raise FileNotFoundError(session_id)
        if self.current.id != session_id:
            self.save()
        payload = json.loads(path.read_text())
        self.current = Session.model_validate(payload)
        self.current.summarizing = False
        self.current.transcribing = False
        return self.current

    def find_id_for_audio(self, wav: Path) -> str | None:
        try:
            target = wav.resolve()
        except OSError:
            return None
        if self.current.audio_path:
            try:
                if Path(self.current.audio_path).resolve() == target:
                    return self.current.id
            except OSError:
                pass
        best_id = None
        best_mtime = -1.0
        for path in self.data_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
                audio = payload.get("audio_path") or ""
                if not audio or Path(audio).resolve() != target:
                    continue
            except Exception:
                continue
            mtime = path.stat().st_mtime
            if mtime >= best_mtime:
                best_mtime = mtime
                best_id = str(payload.get("id") or path.stem)
        return best_id

    def sessions_by_audio(self) -> dict[str, dict[str, Any]]:
        best: dict[str, tuple[float, dict[str, Any]]] = {}
        for path in self.data_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
                audio = payload.get("audio_path") or ""
                if not audio:
                    continue
                key = str(Path(audio).resolve())
                segments = payload.get("segments") or []
                meta = {
                    "id": str(payload.get("id") or path.stem),
                    "title": payload.get("title") or "Untitled session",
                    "segments": len(segments) if isinstance(segments, list) else 0,
                    "cursor": float(payload.get("audio_cursor_s") or 0.0),
                }
            except Exception:
                continue
            mtime = path.stat().st_mtime
            prev = best.get(key)
            if prev is None or mtime >= prev[0]:
                best[key] = (mtime, meta)
        found = {key: item[1] for key, item in best.items()}
        if self.current.audio_path:
            try:
                key = str(Path(self.current.audio_path).resolve())
            except OSError:
                return found
            found[key] = {
                "id": self.current.id,
                "title": self.current.title,
                "segments": len(self.current.segments),
                "cursor": float(self.current.audio_cursor_s or 0.0),
            }
        return found

    def bind_wav(self, wav: Path, duration_s: float, title: str) -> Session:
        self.save()
        self.current = Session()
        self.current.audio_path = str(wav.resolve())
        self.current.audio_duration_s = duration_s
        self.current.audio_cursor_s = 0.0
        self.current.title = title
        self.save()
        return self.current

    def forget_wav(self, wav: Path) -> list[str]:
        """Remove the wav and any meeting whose audio file is that path."""
        try:
            target = wav.resolve()
        except OSError:
            target = wav
        doomed: list[Path] = []
        for path in self.data_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
                audio = payload.get("audio_path") or ""
                if audio and Path(audio).resolve() == target:
                    doomed.append(path)
            except Exception:
                continue
        doomed_ids = {path.stem for path in doomed}
        current_doomed = self.current.id in doomed_ids
        if self.current.audio_path:
            try:
                if Path(self.current.audio_path).resolve() == target:
                    current_doomed = True
            except OSError:
                pass
        if current_doomed:
            others = [path for path in self.data_dir.glob("*.json") if path.stem not in doomed_ids]
            others.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            loaded = False
            for path in others:
                try:
                    self.current = Session.model_validate(json.loads(path.read_text()))
                    self.current.summarizing = False
                    self.current.transcribing = False
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                self.current = Session()
        for path in doomed:
            if path.stem == self.current.id:
                continue
            path.unlink(missing_ok=True)
        if current_doomed and not (self.data_dir / f"{self.current.id}.json").is_file():
            self.save()
        try:
            target.unlink()
        except OSError:
            pass
        return sorted(doomed_ids)

    def persist_notes(self, notes: str, summarized_index: int | None = None) -> None:
        self.current.notes = notes.strip()
        self.current.notes_updated_at = utc_now()
        self.current.last_summarized_index = (
            summarized_index if summarized_index is not None else len(self.current.segments)
        )
        self.current.summary_error = None
        self.save()
