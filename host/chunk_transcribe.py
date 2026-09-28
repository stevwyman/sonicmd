"""Diarize a whole wav with pyannote, then transcribe each turn with Whisper."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from diarize import LETTERS, SAMPLE_RATE
from pyannote_diarize import SpeakerTurn

MIN_TURN_S = 0.45
MIN_SPEECH_S = 0.35
# Whisper's own window. Overlapping windows were merging two guesses of the
# same sentence ("can't cannot"). A repeated token from a music sting is
# dropped instead.
WHISPER_WINDOW_S = 30.0
WHISPER_HOP_S = 30.0


def _report(on_progress, stage: str, chunk: int, chunks: int, progress: float) -> None:
    if on_progress is None:
        return
    try:
        on_progress(
            {
                "stage": stage,
                "chunk": int(chunk),
                "chunks": int(chunks),
                "progress": max(0.0, min(1.0, float(progress))),
            }
        )
    except Exception as exc:
        print(f"progress failed: {exc}", flush=True)


def load_wav(path: Path, end_s: float | None = None) -> np.ndarray:
    """Mono float32, or stereo (n, 2) with left=mic and right=system."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16-bit PCM")
        sr = handle.getframerate()
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path} sample rate {sr} != {SAMPLE_RATE}")
        channels = handle.getnchannels()
        n = handle.getnframes()
        take = n if end_s is None else min(n, max(0, int(round(float(end_s) * sr))))
        handle.setpos(0)
        raw = handle.readframes(take)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels <= 1:
        return samples
    return samples.reshape(-1, channels)


def read_wav_range(path: Path, start_s: float, end_s: float) -> np.ndarray:
    start_s = max(0.0, start_s)
    end_s = max(start_s, end_s)
    audio = load_wav(path, end_s=end_s)
    a = int(round(start_s * SAMPLE_RATE))
    if audio.ndim == 1:
        return audio[a:]
    return audio[a:]


def speaker_spans(turns: list[SpeakerTurn]) -> list[tuple[float, float, str]]:
    """Map pyannote's own labels to A/B/C in order of first appearance.

    The letter is the diarizer's cluster, not a second embedding match.
    Adjacent turns from the same person are one block.
    """
    letters: dict[str, str] = {}
    spans: list[tuple[float, float, str]] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        local_id = str(turn.local_id or "SPEAKER_00")
        if local_id not in letters:
            letters[local_id] = LETTERS[min(len(letters), len(LETTERS) - 1)]
        start, end = float(turn.start), float(turn.end)
        if end - start < MIN_SPEECH_S:
            continue
        letter = letters[local_id]
        if spans and spans[-1][2] == letter and start - spans[-1][1] <= 1.25:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end), letter)
        else:
            spans.append((start, end, letter))
    return spans


def _clean_text(text: str) -> str:
    text = " ".join(str(text).split())
    for punct in (",", ".", "?", "!", ";", ":"):
        text = text.replace(f" {punct}", punct)
    return text.strip()


def _segment_usable(segment: dict) -> bool:
    """Drop windows where Whisper filled silence or music with a few words."""
    text = _clean_text(segment.get("text") or "")
    if not text:
        return False
    try:
        no_speech = float(segment.get("no_speech_prob") or 0.0)
        logprob = float(segment.get("avg_logprob") or 0.0)
        compression = float(segment.get("compression_ratio") or 1.0)
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
    except (TypeError, ValueError):
        return False
    if no_speech > 0.6 or compression > 2.4 or logprob < -1.0:
        return False
    duration = max(0.0, end - start)
    words = [tok for tok in text.split() if any(ch.isalpha() for ch in tok)]
    if duration >= 3.0 and words and len(words) / duration < 0.8:
        return False
    return True


def _token_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _drop_repeated_segments(segments: list[dict]) -> list[dict]:
    """A sting transcribed as the same token over and over is not speech."""
    keys = [_token_key(_clean_text(segment.get("text") or "")) for segment in segments]
    kept: list[dict] = []
    for index, segment in enumerate(segments):
        key = keys[index]
        if not key:
            continue
        nearby = [item for item in keys[max(0, index - 8) : index + 9] if item]
        if nearby.count(key) >= 3:
            continue
        kept.append(segment)
    return kept


def _lang_code(value: object) -> str:
    code = str(value or "").strip().lower().replace("_", "-")
    if not code or code in {"unknown", "none", "null", "und"}:
        return "und"
    if "-" in code:
        code = code.split("-", 1)[0]
    if not code.isalpha() or not 2 <= len(code) <= 8:
        return "und"
    return code


def _majority_lang(langs: list[str]) -> str:
    counts: dict[str, int] = {}
    for lang in langs:
        code = _lang_code(lang)
        if code == "und":
            continue
        counts[code] = counts.get(code, 0) + 1
    if not counts:
        return "und"
    return max(counts, key=counts.get)


def _boundary_mark(token: str) -> str:
    """Punctuation Whisper split off the word ("afraid" + "..."). Empty if it is not a boundary."""
    stripped = token.strip()
    if not stripped or any(ch.isalpha() or ch.isdigit() for ch in stripped):
        return ""
    core = stripped.strip("\"'”’()[]")
    if core in {"-", "—", "–", "...", "…", ".", "?", "!"} or core.endswith("...") or core.endswith("…"):
        return core
    return ""


def _words_from_segment(segment: dict, offset_s: float) -> list[tuple[float, float, str, float, str]]:
    lang = _lang_code(segment.get("language"))
    raw_words = segment.get("words") or []
    out: list[tuple[float, float, str, float, str]] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        token = _clean_text(item.get("word") or item.get("text") or "")
        if not token:
            continue
        mark = _boundary_mark(token)
        if not mark and not any(ch.isalpha() for ch in token):
            continue
        try:
            start = offset_s + float(item["start"])
            end = offset_s + float(item["end"])
            prob = float(item.get("probability") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            continue
        if mark:
            if not out:
                continue
            prev_start, prev_end, prev, prev_prob, prev_lang = out[-1]
            out[-1] = (prev_start, max(prev_end, end), f"{prev}{mark}", prev_prob, prev_lang)
            continue
        if end <= start or prob < 0.15:
            continue
        out.append((start, end, token, prob, lang))
    if out:
        return out
    text = _clean_text(segment.get("text") or "")
    if not text:
        return []
    try:
        start = offset_s + float(segment.get("start") or 0.0)
        end = offset_s + float(segment.get("end") or 0.0)
    except (TypeError, ValueError):
        return []
    if end <= start:
        return []
    return [(start, end, text, 0.5, lang)]


def reconcile_window_words(
    windows: list[tuple[float, float, list[tuple[float, float, str, float, str]]]],
) -> list[tuple[float, float, str, str]]:
    """Keep a word only when the window with the most leading audio also heard it.

    A window that opens on a music sting repeats a nonsense token. The window
    that started earlier, during real speech, has that same stretch as silence.
    """
    survivors: list[tuple[float, float, str, float, float, str]] = []
    for win_start, win_end, words in windows:
        del win_end
        for start, end, token, prob, lang in words:
            mid = 0.5 * (start + end)
            context = mid - win_start
            drop = False
            for other_start, other_end, other_words in windows:
                if other_end <= mid or other_start > mid:
                    continue
                if mid - other_start <= context + 0.4:
                    continue
                heard = any(abs(0.5 * (item[0] + item[1]) - mid) <= 0.45 for item in other_words)
                if not heard:
                    drop = True
                    break
            if not drop:
                survivors.append((start, end, token, prob, context, _lang_code(lang)))
    survivors.sort(key=lambda item: (-item[3], -item[4], item[0]))
    claimed: list[tuple[float, float, str, float, float, str]] = []
    for start, end, token, prob, context, lang in survivors:
        key = _token_key(token)
        mid = 0.5 * (start + end)
        duplicate = False
        for kept_start, kept_end, kept_token, _kept_prob, _kept_context, _kept_lang in claimed:
            same = key and key == _token_key(kept_token)
            overlaps = min(end, kept_end) - max(start, kept_start) > 0
            close = abs(mid - 0.5 * (kept_start + kept_end)) <= 0.5
            if same and (overlaps or close):
                duplicate = True
                break
        if duplicate:
            continue
        claimed.append((start, end, token, prob, context, lang))
    claimed.sort(key=lambda item: item[0])
    return [(start, end, token, lang) for start, end, token, _prob, _context, lang in claimed]


def _letter_at(moment: float, spans: list[tuple[float, float, str]]) -> str | None:
    for start, end, letter in spans:
        if start - 0.05 <= moment <= end + 0.05:
            return letter
    nearest: str | None = None
    nearest_dist = 0.35
    for start, end, letter in spans:
        dist = start - moment if moment < start else moment - end if moment > end else 0.0
        if dist < nearest_dist:
            nearest = letter
            nearest_dist = dist
    return nearest


# A short reply is a different person when it does not match the voice around it.
# Same-speaker pairs in this stack land near 0.6; the disputed "Yes, of course."
# landed near 0.
_SAME_VOICE = 0.40
_MOVE_MARGIN = 0.15
_PEEL_VOICE = 0.30
_MERGE_VOICE = 0.35
_SHORT_CLAUSE_S = 2.2
_PEEL_MAX_S = 1.35
_PEEL_MIN_S = 0.70
_PAUSE_SPLIT_S = 0.45


def _clause_end(token: str) -> bool:
    """A word ends a sentence, including an interruption ("afraid...", "a—")."""
    stripped = token.strip().rstrip("\"'”’)")
    if not stripped:
        return False
    if stripped.endswith("...") or stripped.endswith("…"):
        return True
    if stripped in {"-", "—", "–", "...", "…"}:
        return True
    return stripped[-1] in ".?!-—–"


def _overlap_letter(start: float, end: float, spans: list[tuple[float, float, str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for span_start, span_end, letter in spans:
        overlap = min(end, span_end) - max(start, span_start)
        if overlap > 0:
            scores[letter] = scores.get(letter, 0.0) + overlap
    return scores


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _can_add_speaker(
    current: set[str],
    pyannote_n: int,
    min_speakers: int | None,
    max_speakers: int | None,
) -> bool:
    """A new letter is allowed only while the user's count still has room.

    With no min and no max, pyannote's own clusters are the cast. The repair
    may move a sentence between those people, and it does not invent another.
    """
    n = len(current)
    if max_speakers is not None:
        return n < max_speakers and pyannote_n < max_speakers
    if min_speakers is not None:
        return n < min_speakers and pyannote_n < min_speakers
    return False


def _limit_speakers(
    rows: list[tuple[str, float, float, str, np.ndarray | None, str]],
    max_speakers: int | None,
) -> list[tuple[str, float, float, str, np.ndarray | None, str]]:
    """Fold the closest voices together until the cast fits the maximum."""
    if max_speakers is None or max_speakers < 1:
        return rows
    limited = list(rows)
    for _ in range(40):
        letters = {row[0] for row in limited}
        if len(letters) <= max_speakers:
            return limited
        duration: dict[str, float] = {}
        centroids: dict[str, np.ndarray] = {}
        for letter, start, end, _text, vec, _lang in limited:
            duration[letter] = duration.get(letter, 0.0) + max(0.0, end - start)
            if vec is None:
                continue
            centroids[letter] = vec if letter not in centroids else centroids[letter] + vec
        for letter, vec in list(centroids.items()):
            norm = float(np.linalg.norm(vec))
            centroids[letter] = vec / norm if norm else vec
        victim = min(letters, key=lambda letter: (duration.get(letter, 0.0), letter))
        target = None
        best = -2.0
        if victim in centroids:
            for other, centroid in centroids.items():
                if other == victim:
                    continue
                score = _cosine(centroids[victim], centroid)
                if score > best:
                    best = score
                    target = other
        if target is None:
            for index, row in enumerate(limited):
                if row[0] != victim:
                    continue
                if index > 0 and limited[index - 1][0] != victim:
                    target = limited[index - 1][0]
                    break
                if index + 1 < len(limited) and limited[index + 1][0] != victim:
                    target = limited[index + 1][0]
                    break
        if target is None or target == victim:
            target = next(letter for letter in letters if letter != victim)
        limited = [
            (target if letter == victim else letter, start, end, text, vec, lang)
            for letter, start, end, text, vec, lang in limited
        ]
    return limited


def _profile_vectors(
    blocks: list[tuple[str, float, float, str, str]],
    audio: np.ndarray | None,
    origin_s: float,
    embed,
) -> dict[str, list[float]]:
    """One voice vector per letter, from that speaker's longest turn."""
    if audio is None or embed is None:
        return {}
    best: dict[str, tuple[float, float, float]] = {}
    for letter, start, end, _text, _lang in blocks:
        duration = end - start
        if duration < 1.0:
            continue
        prev = best.get(letter)
        if prev is None or duration > prev[0]:
            best[letter] = (duration, start, min(end, start + 8.0))
    vectors: dict[str, list[float]] = {}
    for letter, (_duration, start, end) in best.items():
        a = max(0, int((start - origin_s) * SAMPLE_RATE))
        b = min(int(audio.size), int((end - origin_s) * SAMPLE_RATE))
        if b - a < int(0.8 * SAMPLE_RATE):
            continue
        try:
            vec = embed(audio[a:b])
        except Exception:
            vec = None
        if vec is None:
            continue
        flat = np.asarray(vec, dtype=np.float64).reshape(-1)
        if flat.size == 0 or not np.isfinite(flat).all():
            continue
        vectors[letter] = [float(x) for x in flat]
    return vectors


def words_to_blocks(
    words: list[tuple],
    spans: list[tuple[float, float, str]],
    transcribe_from: float,
    audio: np.ndarray | None = None,
    origin_s: float = 0.0,
    embed=None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[tuple[str, float, float, str, str]]:
    """Group words into sentences, then give each sentence one speaker.

    A sentence is split on punctuation, on an interruption ("afraid..."), and
    on a pause. A short reply glued to the next sentence is peeled off when
    its voice matches a different person. That reply takes the matching voice
    when it sits on a speaker change. A new letter is created only when
    min/max still allows another speaker.
    """
    pending = []
    for word in words:
        if len(word) < 3 or float(word[1]) < transcribe_from - 0.05:
            continue
        lang = _lang_code(word[3]) if len(word) > 3 else "und"
        pending.append((float(word[0]), float(word[1]), str(word[2]), lang))
    clauses: list[list[tuple[float, float, str, str]]] = []
    current: list[tuple[float, float, str, str]] = []
    for word in pending:
        if current and word[0] - current[-1][1] >= _PAUSE_SPLIT_S:
            clauses.append(current)
            current = []
        current.append(word)
        if _clause_end(word[2]):
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)

    # Long turns of one person, kept whole. A short reply is scored against the
    # closest of these, because averaging them hides the sample it actually matches.
    exemplars: dict[str, list[tuple[float, np.ndarray]]] = {}
    assigned: list[tuple[str, float, float, str, np.ndarray | None, str]] = []
    extra_n = 0
    pyannote_n = len({letter for _start, _end, letter in spans})

    def remember(letter: str, vec: np.ndarray | None, duration: float) -> None:
        # Only a turn too long to be a short reply trains that speaker's voice.
        # Otherwise the reply we are about to move becomes the evidence for the wrong person.
        if vec is None or duration <= _SHORT_CLAUSE_S:
            return
        rows = exemplars.setdefault(letter, [])
        rows.append((duration, vec))
        rows.sort(key=lambda item: item[0], reverse=True)
        del rows[6:]

    def match_voice(vec: np.ndarray) -> tuple[str | None, float, dict[str, float]]:
        scores: dict[str, float] = {}
        best_letter = None
        best_score = -1.0
        for known, rows in exemplars.items():
            score = max(_cosine(vec, item[1]) for item in rows)
            scores[known] = score
            if score > best_score:
                best_letter, best_score = known, score
        return best_letter, best_score, scores

    def voice(start: float, end: float) -> np.ndarray | None:
        if audio is None or embed is None:
            return None
        if end - start > 12.0:
            return None
        a = max(0, int((start - origin_s) * SAMPLE_RATE))
        b = min(int(audio.size), int((end - origin_s) * SAMPLE_RATE))
        if b - a < int(0.25 * SAMPLE_RATE):
            return None
        try:
            return embed(audio[a:b])
        except Exception:
            return None

    # Pyannote's long turns are the voice samples. Skip the opening second so a
    # short reply stuck on the front of a turn is not the sample for that person.
    ranked_spans = sorted(spans, key=lambda row: row[1] - row[0], reverse=True)
    for span_start, span_end, letter in ranked_spans:
        if len(exemplars.get(letter, ())) >= 4:
            continue
        dur = span_end - span_start
        if dur < 2.5:
            continue
        sample_start = span_start + min(1.0, 0.15 * dur) if dur > 5.0 else span_start
        sample_end = min(span_end, sample_start + 5.0)
        remember(letter, voice(sample_start, sample_end), sample_end - sample_start)

    def near_change(start: float, end: float) -> bool:
        for span_start, span_end, _letter in spans:
            for edge in (span_start, span_end):
                if start - 1.2 <= edge <= end + 1.2:
                    return True
        return False

    def peel(clause: list[tuple[float, float, str, str]]) -> list[list[tuple[float, float, str, str]]]:
        """Cut a short reply off the start or end of a longer sentence when the voice changes.

        "No, I'm afraid" and "Let's not rush" share no pause and no period, so
        punctuation never separates them. The two halves still do not match.
        """
        if len(clause) < 3:
            return [clause]
        start = clause[0][0]
        end = clause[-1][1]
        if end - start < 1.3:
            return [clause]
        cuts: list[int] = []
        for index in range(1, len(clause)):
            left_dur = clause[index - 1][1] - start
            right_dur = end - clause[index][0]
            if left_dur > _PEEL_MAX_S:
                break
            if _PEEL_MIN_S <= left_dur <= _PEEL_MAX_S and right_dur >= 1.0:
                cuts.append(index)
        for index in range(len(clause) - 1, 0, -1):
            right_dur = end - clause[index][0]
            left_dur = clause[index - 1][1] - start
            if right_dur > _PEEL_MAX_S:
                break
            if _PEEL_MIN_S <= right_dur <= _PEEL_MAX_S and left_dur >= 1.0:
                cuts.append(index)
        best_score = 1.0
        best_cut = None
        for index in sorted(set(cuts)):
            left_end = clause[index - 1][1]
            right_start = clause[index][0]
            left_dur = left_end - start
            if left_dur <= _PEEL_MAX_S:
                left = voice(max(start, left_end - 1.6), left_end)
                right = voice(right_start, min(end, right_start + 6.0))
            else:
                left = voice(max(start, left_end - 6.0), left_end)
                right = voice(right_start, min(end, right_start + 1.6))
            if left is None or right is None:
                continue
            score = _cosine(left, right)
            if score >= _PEEL_VOICE:
                continue
            short_vec, long_vec = (left, right) if left_dur <= _PEEL_MAX_S else (right, left)
            short_letter, short_score, _short_scores = match_voice(short_vec)
            long_letter, _long_score, _long_scores = match_voice(long_vec)
            # A noisy first couple of words can look unlike the rest of the same
            # sentence. Cut only when the short side is actually someone else.
            if (
                not exemplars
                or short_letter is None
                or long_letter is None
                or short_letter == long_letter
                or short_score < _SAME_VOICE
            ):
                continue
            if score < best_score:
                best_score = score
                best_cut = index
        if best_cut is None or best_score >= _PEEL_VOICE:
            return [clause]
        return [clause[:best_cut], clause[best_cut:]]

    peeled: list[list[tuple[float, float, str, str]]] = []
    for clause in clauses:
        peeled.extend(peel(clause))

    drafted: list[tuple[str, float, float, str, np.ndarray | None, str]] = []
    for clause in peeled:
        start = clause[0][0]
        end = clause[-1][1]
        duration = max(0.0, end - start)
        scores: dict[str, float] = {}
        for word_start, word_end, _token, _lang in clause:
            for letter, overlap in _overlap_letter(word_start, word_end, spans).items():
                scores[letter] = scores.get(letter, 0.0) + overlap
        letter = max(scores, key=scores.get) if scores else _letter_at(0.5 * (start + end), spans)
        if letter is None:
            continue
        text = _clean_text(" ".join(token for _start, _end, token, _lang in clause))
        lang = _majority_lang([item[3] for item in clause])
        vec = voice(start, end)
        drafted.append((letter, start, end, text, vec, lang))
        remember(letter, vec, duration)

    for letter, start, end, text, vec, lang in drafted:
        duration = max(0.0, end - start)
        if vec is not None and duration <= _SHORT_CLAUSE_S and exemplars:
            best_letter, best_score, scores = match_voice(vec)
            own = scores.get(letter)
            # Override the diarizer only at a turn boundary, where a short reply
            # gets glued to the neighbour, or when the voice match is unmistakable.
            if (
                best_letter is not None
                and best_score >= _SAME_VOICE
                and (own is None or best_score >= own + _MOVE_MARGIN)
                and (near_change(start, end) or best_score >= 0.55)
            ):
                letter = best_letter
            elif (
                duration >= 0.35
                and (own is None or own < 0.25)
                and best_score < _SAME_VOICE
                and _can_add_speaker(
                    {row[0] for row in assigned},
                    pyannote_n,
                    min_speakers,
                    max_speakers,
                )
            ):
                extra_n += 1
                letter = f"extra-{extra_n}"
        assigned.append((letter, start, end, text, vec, lang))

    assigned = _limit_speakers(assigned, max_speakers)
    order: list[str] = []
    for letter, _start, _end, _text, _vec, _lang in assigned:
        if letter not in order:
            order.append(letter)
    kept = order[:max_speakers] if max_speakers is not None else order
    mapping = {old: LETTERS[index] for index, old in enumerate(kept) if index < len(LETTERS)}
    fallback = mapping[kept[-1]] if kept else LETTERS[0]
    blocks: list[tuple[str, float, float, str, str, np.ndarray | None]] = []
    for letter, start, end, text, vec, lang in assigned:
        if not text:
            continue
        letter = mapping.get(letter, fallback)
        if blocks and blocks[-1][0] == letter and start - blocks[-1][2] <= 1.0:
            prev_letter, prev_start, prev_end_block, prev_text, prev_lang, prev_vec = blocks[-1]
            short = (end - start) < 1.3 or (prev_end_block - prev_start) < 1.3
            sim = _cosine(prev_vec, vec) if prev_vec is not None and vec is not None else None
            # A short acknowledgement can share a letter with the next sentence
            # and still be a different person. Keep it as its own turn.
            if short and sim is not None and sim < _MERGE_VOICE:
                blocks.append((letter, start, end, text, lang, vec))
            else:
                prev_dur = prev_end_block - prev_start
                new_dur = end - start
                if prev_lang != "und" and (lang == "und" or prev_dur >= new_dur):
                    lang = prev_lang
                mixed = prev_vec if vec is None else vec if prev_vec is None else prev_vec + vec
                if mixed is not None:
                    norm = float(np.linalg.norm(mixed))
                    mixed = mixed / norm if norm else mixed
                blocks[-1] = (
                    prev_letter,
                    prev_start,
                    end,
                    _clean_text(f"{prev_text} {text}"),
                    lang,
                    mixed,
                )
        else:
            blocks.append((letter, start, end, text, lang, vec))
    return [(letter, start, end, text, lang) for letter, start, end, text, lang, _vec in blocks]


def process_range(
    audio: np.ndarray,
    origin_s: float,
    transcribe_clip,
    client,
    whisper_model: str,
    source: str,
    pipeline,
    on_progress=None,
    transcribe_from: float = 0.0,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> int:
    """Pyannote labels the whole buffer once. Whisper then transcribes it in
    30s windows, and each word keeps the speaker who was talking at that time.

    ``transcribe_from`` skips words already posted by an earlier pass.
    Speaker letters still come from the full buffer.
    ``max_speakers`` is a hard cap, including any sentence moved to a new voice.
    """
    blocks = blocks_from_audio(
        audio,
        origin_s,
        transcribe_clip,
        whisper_model,
        pipeline,
        on_progress=on_progress,
        transcribe_from=transcribe_from,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return emit_blocks(blocks, client, source, audio, origin_s, pipeline)


def process_split_channels(
    left: np.ndarray,
    right: np.ndarray,
    origin_s: float,
    transcribe_clip,
    client,
    whisper_model: str,
    source: str,
    remote_pipeline,
    local_pipeline,
    on_progress=None,
    transcribe_from: float = 0.0,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    local_present: bool = True,
) -> int:
    """Transcribe stereo capture without mixing: left=you, right=them."""
    print("Speakers: right channel is them (system), left channel is you (mic)", flush=True)
    them_min, them_max = min_speakers, max_speakers
    if local_present:
        if them_min:
            them_min = max(1, int(them_min) - 1)
        if them_max:
            them_max = max(1, int(them_max) - 1)
    them_raw: list[tuple[str, float, float, str, str]] = []
    you_raw: list[tuple[str, float, float, str, str]] = []
    right_peak = float(np.max(np.abs(right))) if right.size else 0.0
    left_peak = float(np.max(np.abs(left))) if left.size else 0.0
    if right_peak >= 0.008:
        them_blocks = blocks_from_audio(
            right,
            origin_s,
            transcribe_clip,
            whisper_model,
            remote_pipeline,
            on_progress=on_progress,
            transcribe_from=transcribe_from,
            min_speakers=them_min,
            max_speakers=them_max,
            label="them",
        )
        them_raw = [(f"them-{letter}", start, end, text, lang) for letter, start, end, text, lang in them_blocks]
    if local_present and left_peak >= 0.012:
        you_blocks = blocks_from_audio(
            left,
            origin_s,
            transcribe_clip,
            whisper_model,
            local_pipeline,
            on_progress=on_progress,
            transcribe_from=transcribe_from,
            min_speakers=1,
            max_speakers=1,
            label="you",
        )
        you_raw = [("you", start, end, text, lang) for _letter, start, end, text, lang in you_blocks]
    tagged = you_raw + them_raw
    mapping = _appearance_map(tagged)
    merged = [(mapping[key], start, end, text, lang) for key, start, end, text, lang in sorted(tagged, key=lambda row: (row[1], row[2]))]
    posted = emit_blocks(merged, client, source, None, origin_s, None)
    try:
        you_final = [(mapping[key], start, end, text, lang) for key, start, end, text, lang in you_raw]
        them_final = [(mapping[key], start, end, text, lang) for key, start, end, text, lang in them_raw]
        voices = {}
        voices.update(_profile_vectors(you_final, left, origin_s, getattr(local_pipeline, "embed", None)))
        voices.update(_profile_vectors(them_final, right, origin_s, getattr(remote_pipeline, "embed", None)))
        if voices and hasattr(client, "send_voices"):
            client.send_voices([{"id": letter, "embedding": vec} for letter, vec in voices.items()])
    except Exception as exc:
        print(f"voice profiles skipped: {exc}", flush=True)
    return posted


def _appearance_map(rows: list[tuple[str, float, float, str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    order: list[str] = []
    for key, _start, _end, _text, _lang in sorted(rows, key=lambda row: (row[1], row[2])):
        if key in mapping:
            continue
        mapping[key] = LETTERS[min(len(order), len(LETTERS) - 1)]
        order.append(key)
    return mapping


def blocks_from_audio(
    audio: np.ndarray,
    origin_s: float,
    transcribe_clip,
    whisper_model: str,
    pipeline,
    on_progress=None,
    transcribe_from: float = 0.0,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    label: str | None = None,
) -> list[tuple[str, float, float, str, str]]:
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    audio_s = audio.size / SAMPLE_RATE if audio.size else 0.0
    tag = f"{label} " if label else ""
    if audio.size < int(MIN_SPEECH_S * SAMPLE_RATE):
        _report(on_progress, "whisper", 1, 1, 1.0)
        return []
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 0.008:
        print(f"  skip {tag}{origin_s:.0f}s–{origin_s + audio_s:.0f}s (near silence)", flush=True)
        _report(on_progress, "whisper", 1, 1, 1.0)
        return []
    if pipeline is None:
        raise RuntimeError("Speaker labels need pyannote on the full recording.")

    print(
        f"Speakers: {tag}pyannote on {audio_s:.0f}s, one pass "
        f"(min={min_speakers or 'auto'} max={max_speakers or 'auto'})",
        flush=True,
    )
    _report(on_progress, "pyannote", 0, 0, 0.08)
    turns = pipeline.segment(audio, min_duration=MIN_TURN_S, fill_gaps=False)
    spans = speaker_spans(turns)
    speakers = sorted({letter for _s, _e, letter in spans})
    print(
        f"  {tag}{len(turns)} pyannote turns → {len(spans)} blocks, "
        f"{len(speakers)} speakers ({', '.join(speakers) or 'none'})",
        flush=True,
    )
    _report(on_progress, "pyannote", 0, 0, 0.35)
    if not spans:
        _report(on_progress, "whisper", 1, 1, 1.0)
        return []

    window = int(WHISPER_WINDOW_S * SAMPLE_RATE)
    hop = int(WHISPER_HOP_S * SAMPLE_RATE)
    starts = list(range(0, max(audio.size, 1), hop)) if audio.size else [0]
    total = len(starts)
    decoded: list[tuple[float, float, list[tuple[float, float, str, float, str]]]] = []
    for index, offset in enumerate(starts, start=1):
        _report(on_progress, "whisper", index, total, 0.35 + 0.55 * ((index - 1) / total))
        clip = audio[offset : offset + window]
        if clip.size < int(MIN_SPEECH_S * SAMPLE_RATE):
            continue
        offset_s = origin_s + offset / SAMPLE_RATE
        win_end = offset_s + clip.size / SAMPLE_RATE
        print(f"Whisper {tag}{index}/{total} ({offset_s:.0f}s–{win_end:.0f}s)", flush=True)
        try:
            segments = transcribe_clip(clip, whisper_model) or []
        except Exception as exc:
            print(f"whisper failed ({type(exc).__name__}): {exc}", flush=True)
            decoded.append((offset_s, win_end, []))
            continue
        words: list[tuple[float, float, str, float, str]] = []
        for segment in _drop_repeated_segments(segments):
            if not _segment_usable(segment):
                continue
            try:
                seg_dur = float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)
            except (TypeError, ValueError):
                seg_dur = 0.0
            picked = _words_from_segment(segment, offset_s)
            if seg_dur < 0.35 and len(picked) <= 2:
                continue
            words.extend(picked)
        decoded.append((offset_s, win_end, words))
        _report(on_progress, "whisper", index, total, 0.35 + 0.55 * (index / total))

    words = reconcile_window_words(decoded)
    return words_to_blocks(
        words,
        [(origin_s + start, origin_s + end, letter) for start, end, letter in spans],
        transcribe_from,
        audio,
        origin_s,
        getattr(pipeline, "embed", None),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def emit_blocks(
    blocks: list[tuple[str, float, float, str, str]],
    client,
    source: str,
    audio,
    origin_s: float,
    pipeline,
) -> int:
    posted = 0
    for letter, start, end, text, lang in blocks:
        if not text:
            continue
        duration_ms = int(round(1000 * max(end - start, 0.05)))
        print(f"[{letter} {lang} {duration_ms}ms @ {start:.1f}s] {text}", flush=True)
        try:
            client.send_segment(text, lang, source, letter, duration_ms, None)
            posted += 1
        except Exception as exc:
            print(f"failed to post segment: {exc}", flush=True)
    try:
        vectors = _profile_vectors(blocks, audio, origin_s, getattr(pipeline, "embed", None))
        if vectors and hasattr(client, "send_voices"):
            client.send_voices([{"id": letter, "embedding": vec} for letter, vec in vectors.items()])
    except Exception as exc:
        print(f"voice profiles skipped: {exc}", flush=True)
    letters = sorted({letter for letter, _start, _end, _text, _lang in blocks})
    print(f"  {len(letters)} speakers in the transcript ({', '.join(letters) or 'none'})", flush=True)
    print(f"  posted {posted} turns", flush=True)
    return posted
