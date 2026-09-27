from __future__ import annotations

SYSTEM_PROMPT = """You are StenoPod, a local notes editor for live speech transcripts.

You receive a rolling transcript in English and/or German, plus any previous markdown notes.
Write structured meeting notes in Markdown.

Rules:
- Write the notes in the language named in the request (German or English). Use mixed only when the request says the meeting uses both. Keep proper nouns as spoken.
- The Language line must be a single code: de, en, or mixed. Do not print a list of choices.
- Use the speaker labels from the transcript (Speaker A, Speaker B, or any renamed names). Do not invent real identities. If a label is still Speaker A/B, keep it that way.
- Attribute decisions and action items to the labeled speaker when the transcript makes that clear.
- If a point is incomplete or unclear, list it under Open questions.
- Prefer short bullets over long prose.
- Output markdown only. No preamble, no code fences around the whole document.

Use this structure:

# {short title}

**Language:** de

## Speakers
- Speaker A — …

## Summary
- …

## Decisions
- … (or _None yet._)

## Action items
- [ ] Owner — task — due if mentioned

## Open questions
- …
"""


def notes_language(segments: list) -> str:
    """Dominant transcript language. Empty when every turn is still unlabeled."""
    totals: dict[str, int] = {}
    for segment in segments:
        if isinstance(segment, dict):
            lang = str(segment.get("lang") or "")
            weight = int(segment.get("duration_ms") or 0)
        else:
            lang = str(getattr(segment, "lang", "") or "")
            weight = int(getattr(segment, "duration_ms", 0) or 0)
        code = lang.strip().lower()
        if code in {"", "und", "unknown", "mixed"}:
            continue
        totals[code] = totals.get(code, 0) + max(weight, 1)
    if not totals:
        return ""
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    top, top_n = ranked[0]
    total = sum(totals.values()) or 1
    if len(ranked) > 1 and top_n < 0.75 * total:
        return "mixed"
    return top


def build_user_prompt(previous_notes: str, new_transcript: str, language: str = "") -> str:
    notes = previous_notes.strip() or "(none yet)"
    delta = new_transcript.strip() or "(no new speech)"
    code = (language or "").strip().lower()
    if code == "de":
        instruction = "Write the notes in German. Set the Language line to de."
    elif code == "en":
        instruction = "Write the notes in English. Set the Language line to en."
    elif code == "mixed":
        instruction = (
            "This meeting uses both German and English. "
            "Write the notes in whichever language has more speech. Set the Language line to mixed."
        )
    else:
        instruction = (
            "Choose German or English from the words in the transcript. "
            "Set the Language line to de or en. Use mixed only when both languages take real stretches of the meeting."
        )
    return (
        f"{instruction}\n\n"
        "Previous notes:\n"
        f"{notes}\n\n"
        "New transcript to fold in:\n"
        f"{delta}\n"
    )
