from __future__ import annotations

import logging

import httpx

from stenopod.prompt import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger("stenopod.summarizer")


class Summarizer:
    def __init__(self, llm_url: str, timeout: float = 120.0) -> None:
        self.llm_url = llm_url.rstrip("/")
        self.timeout = timeout

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                r = await client.get(f"{self.llm_url}/health")
                return r.status_code < 500
        except httpx.HTTPError:
            return False

    async def summarize(self, previous_notes: str, new_transcript: str) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(previous_notes, new_transcript),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        timeout = httpx.Timeout(self.timeout, connect=2.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{self.llm_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("markdown"):
                text = text[len("markdown") :].lstrip()
        return text
