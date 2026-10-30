"""Minimal async streaming client for the Ollama chat API.

Only the pieces the reasoning agents need. Two details matter, both measured on
this box against ``qwen3.8:27b``:

  * The model returns a separate ``message.thinking`` channel from
    ``message.content`` when ``think`` is enabled. Keeping the monologue out of
    the answer channel is what makes the answer parseable at all.
  * Grammar-constrained decoding (``format`` as a JSON schema) works, and works
    well: the transcription step that overran its budget as free-form text came
    back complete, valid, and in a third of the tokens. This was *not* true of
    the ``gemma4:26b`` this stack previously ran, which degenerated into
    repetition under the same constraint — hence ``format`` is a per-call
    argument the harness decides on, never a hardcoded default.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


@dataclass
class Chunk:
    thinking: str = ""
    content: str = ""
    done: bool = False
    meta: dict[str, Any] | None = None


class OllamaClient:
    def __init__(self, host: str, *, timeout_s: float = 600.0, keep_alive: str = "30m") -> None:
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive

    async def health(self) -> dict[str, Any]:
        """Return {"ok": bool, "models": [...], "error": str|None}."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.host}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return {"ok": True, "models": models, "error": None}
        except Exception as exc:
            return {"ok": False, "models": [], "error": f"{type(exc).__name__}: {exc}"}

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool = True,
        options: dict[str, Any] | None = None,
        format: dict[str, Any] | None = None,
    ) -> AsyncIterator[Chunk]:
        """Stream chat completion chunks (thinking + content deltas).

        ``format`` is a JSON schema; when given, Ollama constrains decoding so
        the answer channel can only be an instance of it.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": options or {},
        }
        if format is not None:
            payload["format"] = format
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout_s) as client,
                client.stream("POST", f"{self.host}/api/chat", json=payload) as resp,
            ):
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise OllamaError(f"Ollama HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        raise OllamaError(str(obj["error"]))
                    msg = obj.get("message", {}) or {}
                    done = bool(obj.get("done"))
                    yield Chunk(
                        thinking=msg.get("thinking") or "",
                        content=msg.get("content") or "",
                        done=done,
                        meta=obj if done else None,
                    )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({exc}). "
                "Is it running?  Try:  ollama serve   (and:  ollama pull " + model + ")"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
