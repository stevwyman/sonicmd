from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from pathlib import Path

from stenopod.prompt import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger("stenopod.summarizer")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOST_PYTHON = REPO_ROOT / ".venv-host" / "bin" / "python"
NOTES_SCRIPT = REPO_ROOT / "host" / "notes_llm.py"


class Summarizer:
    def __init__(self, model: str, timeout: float = 300.0) -> None:
        self.model = model
        self.timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["LLM_MODEL"] = self.model
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def cancel(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
        except OSError:
            pass

    def _run(self, args: list[str], payload: str | None = None) -> subprocess.CompletedProcess[str]:
        if not HOST_PYTHON.is_file():
            raise RuntimeError("Host venv missing. Run `make sidecar-venv`.")
        proc = subprocess.Popen(
            [str(HOST_PYTHON), str(NOTES_SCRIPT), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
            env=self._env(),
        )
        with self._lock:
            self._proc = proc
        try:
            stdout, stderr = proc.communicate(input=payload, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return subprocess.CompletedProcess(proc.args, proc.returncode or 1, stdout, stderr)
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

    async def healthy(self) -> bool:
        try:
            proc = await asyncio.to_thread(self._run, ["--check"])
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0

    async def summarize(self, previous_notes: str, new_transcript: str, language: str = "") -> str:
        payload = json.dumps(
            {
                "system": SYSTEM_PROMPT,
                "user": build_user_prompt(previous_notes, new_transcript, language),
            }
        )
        proc = await asyncio.to_thread(self._run, [], payload)
        if proc.returncode != 0:
            if proc.returncode is not None and proc.returncode < 0:
                raise RuntimeError("stopped")
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(detail or f"notes model exited {proc.returncode}")
        return (proc.stdout or "").strip()
