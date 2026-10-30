"""The local model as a cheap meta-controller.

The same Ollama installation that serves the user-serving master agent also
serves the evolver's controller — but for different work. The division is a
competence judgement, not a cost one:

**Qwen does** the jobs where the answer is already in the data and the model's
value is compression and routing: summarising a metrics table, picking which
failure cluster to open next, deciding whether an expensive investigation looks
warranted, folding a long log into structured context, keeping the experiment
state machine moving.

**Qwen does not** do the jobs where being wrong is expensive and hard to notice:
architectural reasoning about the platform, critique of a candidate, or writing
code. Those escalate. The controller is allowed to *ask* for an escalation; it is
not allowed to substitute for one.

Model discovery is by probe, never by assumption. The brief's spelling of the
local model was approximate, so :func:`resolve_model` asks the server what it
actually has and matches loosely.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from .base import BackendResult, ReasoningBackend

DEFAULT_HOST = "http://127.0.0.1:11434"

#: Preference order when the caller does not name a model. These are families,
#: matched as substrings against what the server reports.
CONTROLLER_PREFERENCES = ("qwen3.8", "qwen3.5", "qwen3", "gpt-oss", "gemma4", "gemma3")


class OllamaBackend(ReasoningBackend):
    """A thin, synchronous client for the local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        *,
        host: str = "",
        model: str = "",
        timeout_s: float = 600.0,
        temperature: float = 0.3,
        num_predict: int = 4000,
        max_calls: int | None = None,
    ) -> None:
        self.host = (host or _default_host()).rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.num_predict = num_predict
        self.max_calls = max_calls
        self.n_calls = 0
        self.total_seconds = 0.0
        self.total_output_tokens = 0
        self.last_error: str | None = None

    # -- discovery -------------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            r.raise_for_status()
            return [m.get("name", "") for m in (r.json().get("models") or [])]
        except Exception:
            return []

    def resolve_model(self, requested: str = "") -> str:
        """Match a requested name against what the server really has.

        Exact match wins; then prefix (``qwen3.8`` → ``qwen3.8:27b``); then the
        preference order. Returns ``""`` when the server has nothing, which the
        caller reports rather than papering over with a name that will 404.
        """
        available = self.list_models()
        if not available:
            return ""
        want = (requested or self.model or "").strip()
        if want:
            if want in available:
                return want
            folded = want.lower().replace(" ", "")
            for name in available:
                if name.lower() == folded or name.lower().startswith(folded.split(":")[0]):
                    return name
            # "qwen3.8:27b" typed as "Qwen3.8:27b" or "qwen3.8-27b"
            loose = re.sub(r"[^a-z0-9.]", "", folded)
            for name in available:
                if re.sub(r"[^a-z0-9.]", "", name.lower()).startswith(loose[:8]):
                    return name
        for family in CONTROLLER_PREFERENCES:
            for name in available:
                if name.lower().startswith(family):
                    return name
        return available[0]

    def available(self) -> tuple[bool, str]:
        if self.max_calls is not None and self.n_calls >= self.max_calls:
            return False, f"call budget exhausted ({self.n_calls}/{self.max_calls})"
        models = self.list_models()
        if not models:
            return False, f"no Ollama server answering at {self.host}"
        if not self.resolve_model():
            return False, "Ollama has no usable model installed"
        return True, ""

    # -- the call --------------------------------------------------------
    def complete(
        self,
        *,
        system: str,
        user: str,
        role: str = "",
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        think: bool = False,
    ) -> BackendResult:
        started = time.time()
        usable, reason = self.available()
        resolved = self.resolve_model(model or "")
        if not usable:
            self.last_error = reason
            name = resolved or model or self.model
            return self._fail(role, name, started, f"ollama unavailable: {reason}")

        payload: dict[str, Any] = {
            "model": resolved,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": user}]
            ),
            "stream": False,
            "think": think,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 20,
                # Structured output is inherently repetitive; a high penalty
                # fights valid JSON. Same reasoning as the user-serving harness.
                "repeat_penalty": 1.05,
                "num_predict": self.num_predict,
            },
        }
        if schema:
            payload["format"] = schema

        self.n_calls += 1
        try:
            r = httpx.post(f"{self.host}/api/chat", json=payload, timeout=timeout_s or self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self._fail(role, resolved, started, self.last_error)
        finally:
            self.total_seconds += time.time() - started

        message = data.get("message") or {}
        text = str(message.get("content") or "")
        parsed: Any = None
        if schema and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        out_tokens = data.get("eval_count")
        if isinstance(out_tokens, int):
            self.total_output_tokens += out_tokens
        return BackendResult(
            text=text,
            parsed=parsed,
            ok=bool(text),
            error=None if text else "empty reply",
            backend=self.name,
            model=resolved,
            role=role,
            elapsed_s=round(time.time() - started, 3),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=out_tokens,
            raw={
                "done_reason": data.get("done_reason"),
                "thinking_chars": len(str(message.get("thinking") or "")),
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "host": self.host,
            "model": self.resolve_model(),
            "n_calls": self.n_calls,
            "seconds": round(self.total_seconds, 3),
            "output_tokens": self.total_output_tokens,
            "last_error": self.last_error,
        }


def _default_host() -> str:
    import os

    return os.environ.get("LLM_AGENT_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST


__all__ = ["CONTROLLER_PREFERENCES", "DEFAULT_HOST", "OllamaBackend"]
