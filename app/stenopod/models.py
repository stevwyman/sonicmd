from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def format_duration(ms: int) -> str:
    total = max(0, int(round(ms / 1000)))
    minutes, seconds = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}:{seconds:02d}"


class Segment(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:10])
    t: str
    lang: str = "und"
    source: str = "system"
    text: str
    speaker: str = "A"
    duration_ms: int = 0
    speaker_score: float | None = None


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    title: str = "Untitled session"
    started_at: datetime = Field(default_factory=utc_now)
    segments: list[Segment] = Field(default_factory=list)
    speaker_names: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    notes_updated_at: datetime | None = None
    last_summarized_index: int = 0
    summarizing: bool = Field(default=False, exclude=True)
    summary_error: str | None = None

    def speaker_label(self, speaker: str) -> str:
        custom = self.speaker_names.get(speaker, "").strip()
        return custom or f"Speaker {speaker}"

    def pending_text(self) -> str:
        pending = self.segments[self.last_summarized_index :]
        lines = []
        for s in pending:
            lines.append(f"[{self.speaker_label(s.speaker)} | {s.t} | {s.lang}] {s.text}")
        return "\n".join(lines).strip()

    def speaker_stats(self) -> list[dict[str, Any]]:
        totals: dict[str, dict[str, Any]] = {}
        for s in self.segments:
            row = totals.setdefault(
                s.speaker,
                {"id": s.speaker, "duration_ms": 0, "turns": 0, "utterances": 0},
            )
            row["duration_ms"] += s.duration_ms
            row["utterances"] += 1
        last = None
        for s in self.segments:
            if s.speaker != last:
                totals[s.speaker]["turns"] += 1
                last = s.speaker
        spoken = sum(r["duration_ms"] for r in totals.values()) or 1
        out = []
        for speaker, row in totals.items():
            out.append(
                {
                    **row,
                    "name": self.speaker_label(speaker),
                    "duration": format_duration(row["duration_ms"]),
                    "share": round(row["duration_ms"] / spoken, 3),
                }
            )
        out.sort(key=lambda r: r["duration_ms"], reverse=True)
        return out

    def blocks(self, segments: list[Segment] | None = None) -> list[dict[str, Any]]:
        items = segments if segments is not None else self.segments
        grouped: list[dict[str, Any]] = []
        for s in items:
            if grouped and grouped[-1]["speaker"] == s.speaker:
                block = grouped[-1]
                block["texts"].append(s.text)
                block["duration_ms"] += s.duration_ms
                block["end"] = s.t
                if s.lang not in block["langs"]:
                    block["langs"].append(s.lang)
            else:
                grouped.append(
                    {
                        "speaker": s.speaker,
                        "name": self.speaker_label(s.speaker),
                        "start": s.t,
                        "end": s.t,
                        "langs": [s.lang],
                        "source": s.source,
                        "texts": [s.text],
                        "duration_ms": s.duration_ms,
                    }
                )
        for block in grouped:
            block["duration"] = format_duration(block["duration_ms"])
            block["text"] = " ".join(block["texts"])
        return grouped

    def transcript_text(self) -> str:
        lines = [
            f"# Transcript — {self.title}",
            f"Started: {iso(self.started_at)}",
            "",
        ]
        stats = self.speaker_stats()
        if stats:
            lines.append("## Speakers")
            for row in stats:
                lines.append(f"- {row['name']}: {row['duration']} ({row['turns']} turns)")
            lines.append("")
        for block in self.blocks():
            langs = ", ".join(block["langs"])
            lines.append(f"## {block['name']} · {block['duration']} · {langs}")
            lines.append(block["text"])
            lines.append("")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        recent = self.segments[-80:]
        return {
            "id": self.id,
            "title": self.title,
            "started_at": iso(self.started_at),
            "segment_count": len(self.segments),
            "notes": self.notes,
            "notes_updated_at": iso(self.notes_updated_at) if self.notes_updated_at else None,
            "pending_chars": len(self.pending_text()),
            "summarizing": self.summarizing,
            "summary_error": self.summary_error,
            "speaker_names": self.speaker_names,
            "speakers": self.speaker_stats(),
            "blocks": self.blocks(recent),
            "segments": [s.model_dump() for s in recent],
        }
