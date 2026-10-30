"""LLM provider interface.

The platform never sources a scientific *number* from an LLM. Providers are
used only for planning, natural-language -> typed-request translation, evidence
summarization, hypothesis phrasing, and critique. The default ``FakeProvider``
is fully deterministic so CI never depends on an external API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..security.egress import EgressPolicy
from ..util import stable_hash_int


@dataclass
class LLMResponse:
    text: str
    tokens: int
    model: str
    structured: dict[str, Any] | None = None


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512, **kw: Any
    ) -> LLMResponse: ...

    def summarize(self, texts: list[str], *, focus: str = "") -> LLMResponse:
        joined = "\n".join(texts)
        return self.complete(f"Summarize (focus={focus}):\n{joined}")


class FakeProvider(LLMProvider):
    """Deterministic, offline, template-based provider.

    Output depends only on the prompt content (via a stable hash), so runs are
    byte-reproducible. It performs *no* scientific computation — it composes
    text from structured inputs handed to it.
    """

    name = "fake"

    def __init__(self, model: str = "deterministic-fake-1") -> None:
        self.model = model

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512, **kw: Any
    ) -> LLMResponse:
        # Length-bounded, deterministic pseudo-summary.
        tokens = min(max_tokens, 32 + stable_hash_int(prompt, modulo=64))
        head = prompt.strip().splitlines()[0][:160] if prompt.strip() else ""
        text = f"[fake-llm] {head}".strip()
        return LLMResponse(text=text, tokens=tokens, model=self.model)

    def plan(self, goal: str, steps: list[str]) -> LLMResponse:
        """Return a deterministic ordered plan phrased from provided steps."""
        payload = {"goal": goal, "steps": list(steps)}
        text = "Plan: " + " -> ".join(steps)
        return LLMResponse(text=text, tokens=16 + len(steps), model=self.model, structured=payload)

    def summarize(self, texts: list[str], *, focus: str = "") -> LLMResponse:
        # Deterministic extractive summary: first sentence of each text.
        picks = []
        for t in texts:
            first = t.strip().split(". ")[0].strip()
            if first:
                picks.append(first if first.endswith(".") else first + ".")
        summary = " ".join(picks[:5]) or "No source text provided."
        tokens = sum(len(p.split()) for p in picks)
        return LLMResponse(
            text=summary,
            tokens=tokens,
            model=self.model,
            structured={"n_sources": len(texts), "focus": focus},
        )


class StubOpenAIProvider(LLMProvider):
    """Real-provider *seam*. Requires network + credentials; never used offline.

    It intentionally refuses to run unless network egress is explicitly enabled,
    so it can be wired in production without being reachable in the default
    build or tests.
    """

    name = "stub_openai"

    def __init__(self, model: str = "gpt-stub", api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512, **kw: Any
    ) -> LLMResponse:
        import os

        EgressPolicy.from_env().check("https://api.openai.com/v1/chat/completions")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} not set; real provider unavailable (this is a seam).")
        raise RuntimeError(
            "StubOpenAIProvider is an integration seam and is not implemented for "
            "offline use. Provide a concrete client here in production."
        )


def make_provider(name: str, model: str | None = None) -> LLMProvider:
    if name == "stub_openai":
        return StubOpenAIProvider(model=model or "gpt-stub")
    return FakeProvider(model=model or "deterministic-fake-1")
