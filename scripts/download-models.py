#!/usr/bin/env python3
"""Download the local models once. Inference itself stays offline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOST_DIR = Path(__file__).resolve().parent.parent / "host"
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from pyannote_diarize import hf_token, load_repo_env

load_repo_env()
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from hf_offline import (  # noqa: E402
    DIARIZATION_REPO,
    EMBEDDING_REPO,
    QWEN_REPO,
    SEGMENTATION_REPO,
    WHISPER_REPO,
    notes_model,
)

REPOS = (
    notes_model(),
    WHISPER_REPO,
    DIARIZATION_REPO,
    SEGMENTATION_REPO,
    EMBEDDING_REPO,
)


def main() -> int:
    token = hf_token()
    if not token:
        print(
            "HF_TOKEN is missing. Accept the pyannote terms, then put a read token in .env.",
            file=sys.stderr,
        )
        return 1
    from huggingface_hub import snapshot_download

    failed: list[str] = []
    for repo in dict.fromkeys(REPOS):
        print(f"Downloading {repo} …", flush=True)
        try:
            snapshot_download(repo, token=token)
        except Exception as exc:
            failed.append(repo)
            print(f"  failed {repo}: {exc}", flush=True)
            continue
        print(f"  cached {repo}", flush=True)
    if failed:
        print(
            "Still missing: "
            + ", ".join(failed)
            + "\nOpen each model page, accept the terms with this token's account, then run `make models` again.",
            file=sys.stderr,
        )
        return 1
    print("Done. Inference uses HF_HUB_OFFLINE=1 and will not contact Hugging Face.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
