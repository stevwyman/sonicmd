"""Local voice profiles so a renamed speaker can be recognized in a later meeting.

This is a small SQLite file under data/, the same role as a local key-value
store: one row per person, looked up by voice embedding.
"""

from __future__ import annotations

import math
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Same-person clips in this stack land near 0.6. Different people land near 0.
MATCH_COSINE = 0.55


def _unit(values: object) -> list[float] | None:
    if not isinstance(values, (list, tuple)):
        return None
    try:
        vec = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    if len(vec) < 8 or any(not math.isfinite(item) for item in vec):
        return None
    norm = math.sqrt(sum(item * item for item in vec))
    if norm < 1e-6:
        return None
    return [item / norm for item in vec]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _blend(left: list[float], right: list[float]) -> list[float] | None:
    if len(left) != len(right):
        return None
    return _unit([a + b for a, b in zip(left, right)])


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes, dim: int) -> list[float] | None:
    if dim < 8 or len(blob) < dim * 4:
        return None
    raw = list(struct.unpack(f"<{dim}f", blob[: dim * 4]))
    return _unit(raw)


class VoiceBank:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voices (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _load(self, conn: sqlite3.Connection) -> list[tuple[str, str, list[float]]]:
        rows = []
        for row in conn.execute("SELECT id, name, embedding, dim FROM voices"):
            vec = _unpack(row["embedding"], int(row["dim"]))
            name = str(row["name"] or "").strip()
            if vec is None or not name:
                continue
            rows.append((str(row["id"]), name, vec))
        return rows

    def identify(self, speakers: list[tuple[str, list[float]]]) -> dict[str, str]:
        """Map speaker letters to saved names. One letter gets the closest profile."""
        with self._lock:
            with self._connect() as conn:
                profiles = self._load(conn)
                found: dict[str, str] = {}
                touched: dict[str, list[float]] = {}
                for letter, raw in speakers:
                    vec = _unit(raw)
                    if vec is None:
                        continue
                    best_id = None
                    best_name = None
                    best_score = MATCH_COSINE
                    best_vec = None
                    for profile_id, name, stored in profiles:
                        if len(stored) != len(vec):
                            continue
                        score = _dot(vec, stored)
                        if score > best_score:
                            best_score = score
                            best_id = profile_id
                            best_name = name
                            best_vec = stored
                    if best_id is None or best_name is None or best_vec is None:
                        continue
                    found[letter] = best_name
                    mixed = _blend(best_vec, vec)
                    if mixed is None:
                        continue
                    touched[best_id] = mixed
                    profiles = [
                        (profile_id, name, mixed if profile_id == best_id else stored)
                        for profile_id, name, stored in profiles
                    ]
                now = datetime.now(timezone.utc).isoformat()
                for profile_id, mixed in touched.items():
                    conn.execute(
                        "UPDATE voices SET embedding = ?, dim = ?, updated_at = ? WHERE id = ?",
                        (_pack(mixed), len(mixed), now, profile_id),
                    )
                conn.commit()
                return found

    def remember(self, name: str, embedding: list[float]) -> None:
        """Save a name the user typed for this voice, or correct the stored name."""
        label = name.strip()
        vec = _unit(embedding)
        if not label or vec is None:
            return
        with self._lock:
            with self._connect() as conn:
                profiles = self._load(conn)
                best_id = None
                best_vec = None
                best_score = MATCH_COSINE
                for profile_id, _stored_name, stored in profiles:
                    if len(stored) != len(vec):
                        continue
                    score = _dot(vec, stored)
                    if score > best_score:
                        best_score = score
                        best_id = profile_id
                        best_vec = stored
                now = datetime.now(timezone.utc).isoformat()
                if best_id is not None and best_vec is not None:
                    mixed = _blend(best_vec, vec) or vec
                    conn.execute(
                        "UPDATE voices SET name = ?, embedding = ?, dim = ?, updated_at = ? WHERE id = ?",
                        (label, _pack(mixed), len(mixed), now, best_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO voices (id, name, embedding, dim, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (uuid4().hex, label, _pack(vec), len(vec), now),
                    )
                conn.commit()
