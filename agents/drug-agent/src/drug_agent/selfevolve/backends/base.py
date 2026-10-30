"""The backend contract.

One method, :meth:`ReasoningBackend.complete`, returning a
:class:`BackendResult` that always records how much it cost and how it failed.
The loop is long-running and partly unattended, so a backend that raises on a
rate limit is worse than one that returns a result saying "unavailable, here is
why, try later" — the controller can then persist state and stop cleanly rather
than losing a cycle's work.

Two exception types exist and the distinction is load-bearing:
:class:`BackendUnavailable` means *this will not work right now* (no binary, not
authenticated, out of quota, network down) and is recoverable by waiting or
falling back; :class:`BackendError` means *this call was wrong* and retrying it
unchanged is pointless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class BackendError(RuntimeError):
    """The call itself was bad. Retrying it unchanged will not help."""


class BackendUnavailable(BackendError):
    """The backend cannot serve right now: absent, unauthenticated, or throttled."""


@dataclass
class BackendResult:
    """One completion, plus everything needed to audit and budget it."""

    text: str = ""
    parsed: Any = None
    ok: bool = True
    error: str | None = None
    backend: str = ""
    model: str = ""
    role: str = ""
    elapsed_s: float = 0.0
    #: Provider-reported usage where available. Never invented.
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: Client-side cost estimate, when the provider reports one. For a
    #: subscription-backed Claude Code session this is an *estimate of list
    #: price*, not a charge; it is recorded for budgeting, never as a bill.
    cost_estimate_usd: float | None = None
    session_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def require(self) -> str:
        if not self.ok:
            raise BackendError(self.error or "backend call failed")
        return self.text


class ReasoningBackend:
    """Base class. Subclasses implement :meth:`complete` and :meth:`available`."""

    name: str = "abstract"

    def available(self) -> tuple[bool, str]:
        """``(usable, reason_if_not)``. Must never raise and must be cheap."""
        raise NotImplementedError

    def complete(
        self,
        *,
        system: str,
        user: str,
        role: str = "",
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> BackendResult:
        raise NotImplementedError

    # -- shared helpers --------------------------------------------------
    @staticmethod
    def _timed() -> float:
        return time.time()

    def _fail(self, role: str, model: str, started: float, error: str) -> BackendResult:
        return BackendResult(
            ok=False,
            error=error,
            backend=self.name,
            model=model,
            role=role,
            elapsed_s=round(time.time() - started, 3),
        )


__all__ = ["BackendError", "BackendResult", "BackendUnavailable", "ReasoningBackend"]
