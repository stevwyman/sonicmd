from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
PIPELINE_CANDIDATES = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
)


def hf_token() -> str | None:
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
    turns: list[tuple[float, float, str]], gap: float = 0.35
) -> list[tuple[float, float, str]]:
    merged: list[tuple[float, float, str]] = []
    for start, end, speaker in sorted(turns, key=lambda row: row[0]):
        if merged and merged[-1][2] == speaker and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), speaker)
        else:
            merged.append((start, end, speaker))
    return merged


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
    def __init__(self, num_speakers: int | None = None) -> None:
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        token = hf_token()
        if not token:
            raise SystemExit(
                "pyannote.audio needs a Hugging Face token.\n"
                "1. Accept terms: https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "2. Create a read token: https://huggingface.co/settings/tokens\n"
                "3. export HF_TOKEN=hf_...\n"
                "Or run with --diarize ecapa to keep the lighter embedder."
            )

        import torch
        from pyannote.audio import Pipeline

        last_error: Exception | None = None
        pipeline = None
        loaded = ""
        for name in PIPELINE_CANDIDATES:
            try:
                pipeline = Pipeline.from_pretrained(name, token=token)
                loaded = name
                break
            except Exception as exc:  # noqa: BLE001 — try the legacy pipeline next
                last_error = exc
        if pipeline is None:
            raise SystemExit(f"Could not load a pyannote pipeline: {last_error}") from last_error

        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            device = "mps"
        else:
            device = "cpu"
        self.pipeline = pipeline
        self.num_speakers = num_speakers
        print(f"pyannote pipeline: {loaded} on {device}")

    def _run(self, audio: np.ndarray):
        import torch

        kwargs = {}
        if self.num_speakers:
            kwargs["num_speakers"] = self.num_speakers
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)[None, :])
        try:
            return self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = Path(tmp.name)
            try:
                write_wav(path, audio)
                return self.pipeline(str(path), **kwargs)
            finally:
                path.unlink(missing_ok=True)

    def segment(self, audio: np.ndarray, min_duration: float = 0.6) -> list[SpeakerTurn]:
        duration = audio.size / SAMPLE_RATE if audio.size else 0.0
        if audio.size < int(min_duration * SAMPLE_RATE):
            return [SpeakerTurn(audio, "SPEAKER_00", None, 0.0, duration)] if audio.size else []
        output = self._run(audio)
        emb_map = embeddings_by_label(output)
        turns: list[SpeakerTurn] = []
        for start, end, speaker in merge_turns(iter_turns(output)):
            a = max(0, int(start * SAMPLE_RATE))
            b = min(audio.size, int(end * SAMPLE_RATE))
            clip = audio[a:b]
            if clip.size < int(min_duration * SAMPLE_RATE):
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
        return turns
