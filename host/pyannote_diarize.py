from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hf_offline import (
    DIARIZATION_31_CLUSTERING,
    DIARIZATION_31_ONSET,
    DIARIZATION_REPO,
    EMBEDDING_REPO,
    SEGMENTATION_REPO,
    force_hf_offline,
    require_cached,
)

SAMPLE_RATE = 16000


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_repo_env() -> None:
    """Load KEY=VALUE pairs from the repo .env without overriding the shell."""
    env_path = REPO_ROOT / ".env"
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


load_repo_env()


def hf_token() -> str | None:
    load_repo_env()
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = np.clip(np.asarray(audio, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


@dataclass
class SpeakerTurn:
    audio: np.ndarray
    local_id: str
    embedding: np.ndarray | None
    start: float
    end: float


def iter_turns(output) -> list[tuple[float, float, str]]:
    candidates = []
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    primary = getattr(output, "speaker_diarization", output)
    if exclusive is not None:
        candidates.append(exclusive)
    candidates.append(primary)

    for dia in candidates:
        turns: list[tuple[float, float, str]] = []
        if hasattr(dia, "itertracks"):
            for turn, _, speaker in dia.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), str(speaker)))
        else:
            for item in dia:
                if isinstance(item, tuple) and len(item) == 2:
                    turn, speaker = item
                    turns.append((float(turn.start), float(turn.end), str(speaker)))
        if turns:
            return turns
    return []


def merge_turns(
    turns: list[tuple[float, float, str]], gap: float = 1.25
) -> list[tuple[float, float, str]]:
    merged: list[tuple[float, float, str]] = []
    for start, end, speaker in sorted(turns, key=lambda row: row[0]):
        if merged and merged[-1][2] == speaker and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), speaker)
        else:
            merged.append((start, end, speaker))
    return merged


def resolve_overlaps(
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Longer turns win. Drops 00/01 islands nested inside a dominant speaker."""
    by_dur = sorted(turns, key=lambda row: (row[1] - row[0], -row[0]), reverse=True)
    kept: list[tuple[float, float, str]] = []
    for start, end, speaker in by_dur:
        if end - start <= 0:
            continue
        if any(start >= ks - 1e-3 and end <= ke + 1e-3 for ks, ke, _spk in kept):
            continue
        kept.append((start, end, speaker))
    kept.sort(key=lambda row: row[0])
    trimmed: list[tuple[float, float, str]] = []
    for start, end, speaker in kept:
        if trimmed and start < trimmed[-1][1]:
            prev_s, prev_e, prev_spk = trimmed[-1]
            if speaker == prev_spk:
                trimmed[-1] = (prev_s, max(prev_e, end), speaker)
                continue
            mid = 0.5 * (start + prev_e)
            if prev_e - prev_s >= end - start:
                start = max(start, prev_e)
            else:
                trimmed[-1] = (prev_s, min(prev_e, mid), prev_spk)
                start = max(start, trimmed[-1][1])
            if start >= end:
                continue
        trimmed.append((start, end, speaker))
    return trimmed


def absorb_short_turns(
    turns: list[tuple[float, float, str]],
    min_on: float = 1.0,
    merge_gap: float = 1.25,
) -> list[tuple[float, float, str]]:
    """Collapse word-scale 00/01 flicker into the surrounding speaker.

    Real speaker changes of about a second or more are kept. Brief islands
    (and UNLABELED gap fills) take the neighboring label.
    """
    items = merge_turns(turns, gap=merge_gap)
    if len(items) <= 1:
        return items
    seen: set[tuple[tuple[float, float, str], ...]] = set()
    for _ in range(len(items) * 3 + 8):
        key = tuple(items)
        if key in seen:
            break
        seen.add(key)
        short_i = None
        short_dur = None
        for i, (start, end, speaker) in enumerate(items):
            dur = end - start
            if speaker == "UNLABELED" or dur < min_on:
                if short_dur is None or dur < short_dur - 1e-9:
                    short_i, short_dur = i, dur
        if short_i is None or len(items) == 1:
            break
        left = items[short_i - 1] if short_i > 0 else None
        right = items[short_i + 1] if short_i + 1 < len(items) else None
        if left is None and right is None:
            break

        def _pick(*cands: tuple[float, float, str] | None) -> str | None:
            labeled = [c for c in cands if c is not None and c[2] != "UNLABELED"]
            if not labeled:
                return None
            if len(labeled) == 1:
                return labeled[0][2]
            if labeled[0][2] == labeled[1][2]:
                return labeled[0][2]
            left_dur = labeled[0][1] - labeled[0][0]
            right_dur = labeled[1][1] - labeled[1][0]
            return labeled[0][2] if left_dur >= right_dur else labeled[1][2]

        speaker = _pick(left, right)
        if speaker is None:
            break
        start, end, current = items[short_i]
        if current == speaker and (end - start) >= min_on:
            break
        items[short_i] = (start, end, speaker)
        items = merge_turns(items, gap=merge_gap)
    return items


def fill_uncovered_turns(
    audio: np.ndarray,
    turns: list[SpeakerTurn],
    sr: int = SAMPLE_RATE,
    min_gap: float = 0.35,
    peak_min: float = 0.01,
) -> list[SpeakerTurn]:
    """Keep speech pyannote did not label (short asides, overlaps, quiet talk)."""
    duration = audio.size / sr if audio.size else 0.0
    if duration <= 0:
        return turns
    covered = sorted((t.start, t.end) for t in turns)
    extra: list[SpeakerTurn] = []
    cursor = 0.0
    gaps: list[tuple[float, float]] = []
    for start, end in covered:
        if start - cursor >= min_gap:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= min_gap:
        gaps.append((cursor, duration))
    for start, end in gaps:
        a = max(0, int(start * sr))
        b = min(audio.size, int(end * sr))
        clip = audio[a:b]
        if clip.size < int(min_gap * sr):
            continue
        if float(np.max(np.abs(clip))) < peak_min:
            continue
        extra.append(SpeakerTurn(clip, "UNLABELED", None, start, end))
    if not extra:
        return turns
    filled = turns + extra
    filled.sort(key=lambda t: t.start)
    return filled


def embeddings_by_label(output) -> dict[str, np.ndarray]:
    dia = getattr(output, "speaker_diarization", output)
    embs = getattr(output, "speaker_embeddings", None)
    if embs is None or not hasattr(dia, "labels"):
        return {}
    labels = [str(label) for label in dia.labels()]
    matrix = np.asarray(embs, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    mapped: dict[str, np.ndarray] = {}
    for index, label in enumerate(labels):
        if index >= matrix.shape[0]:
            break
        vec = np.asarray(matrix[index], dtype=np.float64).reshape(-1)
        if float(np.linalg.norm(vec)) > 1e-6:
            mapped[label] = vec
    return mapped


class PyannoteDiarizer:
    def __init__(
        self,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        force_hf_offline()
        token = hf_token()
        if not token:
            raise SystemExit(
                "pyannote.audio needs a Hugging Face token.\n"
                "1. Accept terms:\n"
                "   https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "   https://huggingface.co/pyannote/segmentation-3.0\n"
                "2. Create a read token: https://huggingface.co/settings/tokens\n"
                "3. Put HF_TOKEN=hf_... in a .env file at the repo root (copy .env.example).\n"
                "4. Run `make models` once. Later runs stay offline."
            )
        require_cached([DIARIZATION_REPO, SEGMENTATION_REPO, EMBEDDING_REPO])

        import torch
        from pyannote.audio.pipelines import SpeakerDiarization
        import pyannote.audio.pipelines.speaker_diarization as speaker_diarization

        # pyannote.audio 4 still constructs a PLDA for every pipeline. The 3.1
        # recipe uses agglomerative clustering, which never reads that PLDA.
        # A sentinel skips the community-1 PLDA download.
        skip_plda = object()
        real_get_plda = speaker_diarization.get_plda

        def _get_plda(plda, token=None, cache_dir=None):
            if plda is skip_plda:
                return None
            return real_get_plda(plda, token=token, cache_dir=cache_dir)

        speaker_diarization.get_plda = _get_plda
        try:
            pipeline = SpeakerDiarization(
                segmentation=SEGMENTATION_REPO,
                embedding=EMBEDDING_REPO,
                embedding_exclude_overlap=True,
                embedding_batch_size=32,
                segmentation_batch_size=32,
                clustering="AgglomerativeClustering",
                plda=skip_plda,
                token=token,
            )
        finally:
            speaker_diarization.get_plda = real_get_plda
        segmentation = {"min_duration_off": 0.0}
        if "threshold" in pipeline.segmentation._parameters:
            segmentation["threshold"] = DIARIZATION_31_ONSET
        pipeline.instantiate(
            {
                "clustering": dict(DIARIZATION_31_CLUSTERING),
                "segmentation": segmentation,
            }
        )
        loaded = f"{DIARIZATION_REPO} segmentation={SEGMENTATION_REPO}"

        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            device = "mps"
        else:
            device = "cpu"
        self.pipeline = pipeline
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        if self.num_speakers:
            hint = f"num_speakers={self.num_speakers}"
        elif self.min_speakers or self.max_speakers:
            hint = f"min_speakers={self.min_speakers or 'auto'} max_speakers={self.max_speakers or 'auto'}"
        else:
            hint = "no speaker cap"
        print(f"pyannote pipeline: {loaded} on {device} ({hint})")

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Unit voice vector for a short slice. None when the slice is unusable."""
        import torch

        clip = np.asarray(audio, dtype=np.float32).reshape(-1)
        model = self.pipeline._embedding
        need = int(getattr(model, "min_num_samples", 0) or 0)
        if clip.size < need:
            clip = np.pad(clip, (0, need - clip.size))
        if clip.size < int(0.25 * SAMPLE_RATE):
            return None
        vec = np.asarray(model(torch.from_numpy(clip[None, None, :])), dtype=np.float64).reshape(-1)
        if vec.size == 0 or not np.isfinite(vec).all():
            return None
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return vec / norm

    def _run(self, audio: np.ndarray):
        import torch

        kwargs = {}
        if self.num_speakers:
            kwargs["num_speakers"] = self.num_speakers
        else:
            if self.min_speakers:
                kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers:
                kwargs["max_speakers"] = self.max_speakers
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)[None, :])
        payload = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        # Keep the requested speaker count. A failed call must not retry uncapped.
        try:
            return self.pipeline(payload, **kwargs)
        except Exception as first:
            path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    path = Path(tmp.name)
                write_wav(path, audio)
                return self.pipeline(str(path), **kwargs)
            except Exception as second:
                raise RuntimeError(f"pyannote failed: {second}") from first
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)

    def segment(self, audio: np.ndarray, min_duration: float = 1.0, fill_gaps: bool = False) -> list[SpeakerTurn]:
        duration = audio.size / SAMPLE_RATE if audio.size else 0.0
        if audio.size < int(0.25 * SAMPLE_RATE):
            return [SpeakerTurn(audio, "SPEAKER_00", None, 0.0, duration)] if audio.size else []
        output = self._run(audio)
        emb_map = embeddings_by_label(output)
        spans = absorb_short_turns(
            resolve_overlaps(merge_turns(iter_turns(output))),
            min_on=min_duration,
        )
        turns: list[SpeakerTurn] = []
        for start, end, speaker in spans:
            a = max(0, int(start * SAMPLE_RATE))
            b = min(audio.size, int(end * SAMPLE_RATE))
            clip = audio[a:b]
            if clip.size < int(0.2 * SAMPLE_RATE):
                continue
            turns.append(
                SpeakerTurn(
                    audio=clip,
                    local_id=str(speaker),
                    embedding=emb_map.get(str(speaker)),
                    start=start,
                    end=end,
                )
            )
        if not turns:
            return [SpeakerTurn(audio, "SPEAKER_00", None, 0.0, duration)]
        if not fill_gaps:
            return turns
        # Optional: paint silent holes with the neighboring speaker. That pulls
        # music and room tone into a turn, and Whisper then invents words for them.
        filled = fill_uncovered_turns(audio, turns, min_gap=0.8)
        spans = absorb_short_turns(
            resolve_overlaps([(t.start, t.end, t.local_id) for t in filled]),
            min_on=min_duration,
        )
        rebuilt: list[SpeakerTurn] = []
        for start, end, speaker in spans:
            a = max(0, int(start * SAMPLE_RATE))
            b = min(audio.size, int(end * SAMPLE_RATE))
            clip = audio[a:b]
            if clip.size < int(0.2 * SAMPLE_RATE):
                continue
            rebuilt.append(
                SpeakerTurn(
                    audio=clip,
                    local_id=str(speaker),
                    embedding=emb_map.get(str(speaker)),
                    start=start,
                    end=end,
                )
            )
        return rebuilt or [SpeakerTurn(audio, "SPEAKER_00", None, 0.0, duration)]
