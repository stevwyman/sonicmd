#!/usr/bin/env python3
"""Local notes model. Replaces the Ollama server.

One process loads the MLX Qwen weights, writes markdown to stdout, and exits.
The web app calls this; nothing listens on a port.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HOST_DIR = Path(__file__).resolve().parent
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from hf_offline import QWEN_REPO, force_hf_offline, notes_model, require_cached
from pyannote_diarize import load_repo_env

load_repo_env()
force_hf_offline()


def _check() -> int:
    model = notes_model()
    try:
        require_cached([model])
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(model)
    return 0


def _generate(system: str, user: str) -> str:
    model_id = notes_model()
    require_cached([model_id])
    from mlx_lm import load
    from mlx_lm.generate import generate

    model, tokenizer = load(model_id)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.2)
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1200,
            sampler=sampler,
            verbose=False,
        )
    except TypeError:
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1200,
            temp=0.2,
            verbose=False,
        )
    return str(text).strip()


def main() -> int:
    if "--check" in sys.argv:
        return _check()
    payload = json.loads(sys.stdin.read() or "{}")
    system = str(payload.get("system") or "")
    user = str(payload.get("user") or "")
    if not user.strip():
        print("notes request was empty", file=sys.stderr)
        return 1
    text = _generate(system, user)
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("markdown"):
            text = text[len("markdown") :].lstrip()
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{QWEN_REPO}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
