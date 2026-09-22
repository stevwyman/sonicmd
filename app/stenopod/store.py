from __future__ import annotations

import json
from pathlib import Path

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

    def persist_notes(self, notes: str, summarized_index: int | None = None) -> None:
        self.current.notes = notes.strip()
        self.current.notes_updated_at = utc_now()
        self.current.last_summarized_index = (
            summarized_index if summarized_index is not None else len(self.current.segments)
        )
        self.current.summary_error = None
        self.save()
