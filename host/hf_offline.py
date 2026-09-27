"""Local Hugging Face cache. Inference never calls the hub."""

from __future__ import annotations

import os
from pathlib import Path

# Qwen/Qwen2.5-7B-Instruct-MLX is the name in the Qwen MLX docs, and that repo
# returns 404. This is the Apple MLX 4-bit build of the same instruct model.
QWEN_REPO = "mlx-community/Qwen2.5-7B-Instruct-4bit"
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
DIARIZATION_REPO = "pyannote/speaker-diarization-3.1"
SEGMENTATION_REPO = "pyannote/segmentation-3.0"
EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Published speaker-diarization-3.1 hyperparameters. segmentation-3.0 has no
# threshold knob; the 3.1 pipeline binarized at a fixed 0.5 onset instead.
DIARIZATION_31_CLUSTERING = {
    "method": "centroid",
    "min_cluster_size": 12,
    "threshold": 0.7045654963945799,
}
DIARIZATION_31_ONSET = 0.5


def force_hf_offline() -> None:
    """Block hub lookups. `make models` is the only path that may download."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def notes_model() -> str:
    raw = os.environ.get("LLM_MODEL", "").strip()
    if "/" in raw:
        return raw
    return QWEN_REPO


def hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return root / "hub"


def repo_cached(repo_id: str) -> bool:
    snaps = hub_dir() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return False
    return any(path.is_dir() for path in snaps.iterdir())


def missing_repos(repo_ids: list[str]) -> list[str]:
    return [repo_id for repo_id in repo_ids if not repo_cached(repo_id)]


def require_cached(repo_ids: list[str]) -> None:
    missing = missing_repos(repo_ids)
    if not missing:
        return
    raise SystemExit(
        "These models are not in the local Hugging Face cache:\n"
        + "\n".join(f"  {repo_id}" for repo_id in missing)
        + "\nRun `make models` once while online. After that, inference stays offline "
        "(HF_HUB_OFFLINE=1)."
    )
