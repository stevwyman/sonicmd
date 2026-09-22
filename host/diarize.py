from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
ECAPA_URL = (
    "https://huggingface.co/pranjal-pravesh/ecapa_tdnn_onnx/resolve/main/ecapa_tdnn.onnx"
)
ECAPA_NAME = "ecapa_tdnn.onnx"


def _l2(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return vec
    return vec / n


def logmel_embed(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fixed-size spectral embedding (must not depend on clip length)."""
    sig = np.asarray(audio, dtype=np.float32)
    if sig.size < 512:
        sig = np.pad(sig, (0, 512 - sig.size))
    win = 512
    hop = 160
    window = np.hanning(win).astype(np.float32)
    frames = []
    for start in range(0, sig.size - win + 1, hop):
        frame = sig[start : start + win] * window
        spec = np.abs(np.fft.rfft(frame))
        frames.append(np.log(spec + 1e-6))
    mat = np.stack(frames, axis=0)
    bands = np.array_split(mat, 16, axis=1)
    band_means = np.array([float(np.mean(b)) for b in bands], dtype=np.float64)
    band_stds = np.array([float(np.std(b)) for b in bands], dtype=np.float64)
    time_chunks = np.array_split(mat, 3, axis=0)
    region_parts = []
    for chunk in time_chunks:
        mean_f = np.mean(chunk, axis=0)
        if mean_f.size < 32:
            mean_f = np.pad(mean_f, (0, 32 - mean_f.size))
        region_parts.append(mean_f[:32])
    region = np.concatenate(region_parts).astype(np.float64)
    return _l2(np.concatenate([band_means, band_stds, region]))


class EcapaEmbedder:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]

    def embed(self, audio: np.ndarray) -> np.ndarray:
        sig = np.asarray(audio, dtype=np.float32).reshape(-1)
        min_len = SAMPLE_RATE
        if sig.size < min_len:
            sig = np.pad(sig, (0, min_len - sig.size))
        max_len = SAMPLE_RATE * 8
        if sig.size > max_len:
            extra = sig.size - max_len
            sig = sig[extra // 2 : extra // 2 + max_len]
        name = self.input.name
        shape = self.input.shape
        if len(shape) == 1:
            arr = sig
        elif len(shape) == 2:
            arr = sig[np.newaxis, :]
        else:
            arr = sig[np.newaxis, np.newaxis, :]
        out = np.asarray(self.session.run(None, {name: arr})[0], dtype=np.float64)
        while out.ndim > 1:
            out = out.mean(axis=0)
        return _l2(out.reshape(-1))


@dataclass
class SpeakerRecord:
    letter: str
    centroid: np.ndarray
    count: int = 1
    langs: Counter = field(default_factory=Counter)


class SpeakerBank:
    """Stable A/B/C labels across capture windows.

    Cosine 0.62 matches pyannote's own clustering threshold. A language switch
    (German host vs English ad) opens a new label unless the voice match is
    already very strong — so a bilingual speaker can still stay one person.
    """

    def __init__(
        self,
        threshold: float = 0.62,
        stay_bonus: float = 0.03,
        max_speakers: int = 8,
        lang_override: float = 0.88,
    ) -> None:
        self.threshold = threshold
        self.stay_bonus = stay_bonus
        self.max_speakers = max_speakers
        self.lang_override = lang_override
        self.speakers: list[SpeakerRecord] = []
        self.last: str | None = None

    def _reset_if_dim_changed(self, embedding: np.ndarray) -> None:
        if self.speakers and self.speakers[0].centroid.shape != embedding.shape:
            self.speakers = []
            self.last = None

    def _lang_mismatch(self, rec: SpeakerRecord, lang: str | None) -> bool:
        if not lang or lang == "und" or not rec.langs:
            return False
        dominant, n = rec.langs.most_common(1)[0]
        return n >= 1 and dominant not in {"und", lang}

    def _score(self, embedding: np.ndarray, rec: SpeakerRecord) -> float:
        sim = float(np.dot(embedding, rec.centroid))
        return sim if np.isfinite(sim) else -1.0

    def _needed(self, rec: SpeakerRecord, lang: str | None) -> float:
        return self.lang_override if self._lang_mismatch(rec, lang) else self.threshold

    def _new_letter(self, forbidden: set[str]) -> str:
        used = {rec.letter for rec in self.speakers} | forbidden
        for letter in LETTERS:
            if letter not in used:
                return letter
        return LETTERS[min(len(self.speakers), len(LETTERS) - 1)]

    def _remember(self, rec: SpeakerRecord, embedding: np.ndarray, lang: str | None) -> None:
        n = rec.count + 1
        rec.centroid = _l2(rec.centroid * (rec.count / n) + embedding * (1 / n))
        rec.count = n
        if lang and lang != "und":
            rec.langs[lang] += 1
        self.last = rec.letter

    def reinforce(self, letter: str, embedding: np.ndarray, lang: str | None = None) -> float:
        embedding = _l2(np.asarray(embedding, dtype=np.float64).reshape(-1))
        self._reset_if_dim_changed(embedding)
        for rec in self.speakers:
            if rec.letter == letter:
                sim = self._score(embedding, rec)
                self._remember(rec, embedding, lang)
                return sim
        _, score = self.assign(embedding, lang=lang)
        return score

    def assign(
        self,
        embedding: np.ndarray,
        lang: str | None = None,
        forbidden: set[str] | None = None,
    ) -> tuple[str, float]:
        embedding = _l2(np.asarray(embedding, dtype=np.float64).reshape(-1))
        forbidden = set(forbidden or ())
        self._reset_if_dim_changed(embedding)

        if not self.speakers:
            letter = self._new_letter(forbidden)
            langs: Counter = Counter()
            if lang and lang != "und":
                langs[lang] += 1
            self.speakers.append(SpeakerRecord(letter, embedding, 1, langs))
            self.last = letter
            return letter, 1.0

        ranked: list[tuple[float, float, SpeakerRecord]] = []
        fallback: SpeakerRecord | None = None
        fallback_sim = -1.0
        for rec in self.speakers:
            if rec.letter in forbidden:
                continue
            sim = self._score(embedding, rec)
            if sim > fallback_sim:
                fallback = rec
                fallback_sim = sim
            if sim < self._needed(rec, lang):
                continue
            boosted = sim
            if rec.letter == self.last:
                boosted += self.stay_bonus
            if lang and rec.langs and rec.langs.most_common(1)[0][0] == lang:
                boosted += 0.04
            ranked.append((boosted, sim, rec))

        if ranked:
            ranked.sort(key=lambda row: row[0], reverse=True)
            _boosted, raw, chosen = ranked[0]
            self._remember(chosen, embedding, lang)
            return chosen.letter, raw

        if fallback is not None and len(self.speakers) >= self.max_speakers:
            self._remember(fallback, embedding, lang)
            return fallback.letter, fallback_sim

        letter = self._new_letter(forbidden)
        langs = Counter()
        if lang and lang != "und":
            langs[lang] += 1
        self.speakers.append(SpeakerRecord(letter, embedding, 1, langs))
        self.last = letter
        return letter, fallback_sim


def default_model_path() -> Path:
    root = Path(__file__).resolve().parent.parent / "models"
    return Path(os.environ.get("ECAPA_PATH", root / ECAPA_NAME))


def download_ecapa(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    import urllib.request

    print(f"Downloading speaker model → {dest} (~80 MB, once) …")
    tmp = dest.with_suffix(".partial")
    urllib.request.urlretrieve(ECAPA_URL, tmp)
    tmp.replace(dest)
    return dest


def load_embedder() -> tuple[object, str]:
    path = default_model_path()
    try:
        download_ecapa(path)
        return EcapaEmbedder(path), "ecapa"
    except Exception as exc:
        print(f"ECAPA unavailable ({exc}); using spectral fallback.", flush=True)

        class Fallback:
            def embed(self, audio: np.ndarray) -> np.ndarray:
                return logmel_embed(audio)

        return Fallback(), "spectral"
