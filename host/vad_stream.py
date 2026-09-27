"""Streaming Silero VAD — split on silence, not on imagined speaker changes."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 512  # Silero 16 kHz window

_SILERO = None


def silero_model():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad

        _SILERO = load_silero_vad(onnx=True)
    return _SILERO


class SileroStream:
    """Feed 16 kHz mono frames; get back finished speech utterances."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 150,
        min_speech_s: float = 0.35,
        max_speech_s: float = 12.0,
    ) -> None:
        from silero_vad import VADIterator, load_silero_vad

        self.model = load_silero_vad(onnx=True)
        self.iterator = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.min_samples = int(min_speech_s * SAMPLE_RATE)
        self.max_samples = int(max_speech_s * SAMPLE_RATE)
        self.pending = np.zeros(0, dtype=np.float32)
        self.speech: list[np.ndarray] = []
        self.in_speech = False
        print(
            f"Silero VAD (onnx) threshold={threshold} "
            f"min_silence={min_silence_ms}ms max_utt={max_speech_s:.0f}s"
        )

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        self.pending = np.concatenate([self.pending, np.asarray(frame, dtype=np.float32).reshape(-1)])
        out: list[np.ndarray] = []
        while self.pending.size >= WINDOW:
            chunk = np.array(self.pending[:WINDOW], dtype=np.float32, copy=True)
            self.pending = self.pending[WINDOW:]
            out.extend(self._step(chunk))
        return out

    def flush(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        if self.pending.size:
            pad = np.zeros(WINDOW, dtype=np.float32)
            pad[: self.pending.size] = self.pending
            out.extend(self._step(pad))
            self.pending = np.zeros(0, dtype=np.float32)
        leftover = self._take()
        if leftover is not None:
            out.append(leftover)
        self.iterator.reset_states()
        self.in_speech = False
        self.speech = []
        return out

    def _step(self, chunk: np.ndarray) -> list[np.ndarray]:
        result = self.iterator(chunk, return_seconds=False)
        out: list[np.ndarray] = []
        started = bool(result and "start" in result)
        ended = bool(result and "end" in result)
        if started:
            self.in_speech = True
            self.speech = [chunk]
        elif self.in_speech:
            self.speech.append(chunk)
        if ended:
            taken = self._take()
            self.in_speech = False
            if taken is not None:
                out.append(taken)
        elif self.in_speech and sum(x.size for x in self.speech) >= self.max_samples:
            taken = self._take()
            self.in_speech = True
            if taken is not None:
                out.append(taken)
        return out

    def _take(self) -> np.ndarray | None:
        if not self.speech:
            return None
        audio = np.concatenate(self.speech)
        self.speech = []
        if audio.size < self.min_samples:
            return None
        return audio


def speech_clips(audio: np.ndarray, min_silence_ms: int = 500, min_speech_ms: int = 250) -> list[np.ndarray]:
    """Offline: cut a whole recording into Silero speech regions."""
    from silero_vad import get_speech_timestamps

    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    stamps = get_speech_timestamps(
        wav,
        silero_model(),
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=150,
        return_seconds=False,
        threshold=0.5,
    )
    clips: list[np.ndarray] = []
    for ts in stamps:
        start = int(ts["start"])
        end = int(ts["end"])
        clip = wav[start:end]
        if clip.size >= int(0.35 * SAMPLE_RATE):
            clips.append(clip)
    return clips


def speech_timestamps(
    audio: np.ndarray,
    min_silence_ms: int = 500,
    min_speech_ms: int = 250,
) -> list[tuple[float, float]]:
    """Offline Silero spans as (start_s, end_s) in the given clip."""
    from silero_vad import get_speech_timestamps

    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    stamps = get_speech_timestamps(
        wav,
        silero_model(),
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=150,
        return_seconds=False,
        threshold=0.5,
    )
    out: list[tuple[float, float]] = []
    min_s = 0.35
    for ts in stamps:
        start = int(ts["start"]) / SAMPLE_RATE
        end = int(ts["end"]) / SAMPLE_RATE
        if end - start >= min_s:
            out.append((start, end))
    return out
