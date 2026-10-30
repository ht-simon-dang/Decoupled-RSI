"""Roles → backends, resolved once per run and reported.

The loop asks for a *role*. Which process serves it is configuration, and the
resolution is recorded in the cycle log so a later reader can tell which model
produced which hypothesis.

Roles
-----
``cheap_controller``  summarise, route, schedule, compress. Local model.
``deep_researcher``   architectural reasoning, literature work, candidate design.
``critic``            try to falsify a hypothesis or a candidate.
``implementer``       write the change.
``evaluator``         **no model at all.** Scoring is deterministic and lives in
                      :mod:`~drug_agent.selfevolve.evaluation.kernel`. The role
                      exists in this table only so that the absence is explicit:
                      nothing in this system asks a model whether it succeeded.
"""

from __future__ import annotations

from typing import Any

from ..config import EvolveConfig
from .base import BackendResult, ReasoningBackend
from .claude_cli_backend import READ_ONLY_TOOLS, ClaudeCLIBackend
from .ollama_backend import OllamaBackend

ROLES = ("cheap_controller", "deep_researcher", "critic", "implementer", "evaluator")


class BackendRegistry:
    """Resolves roles, enforces budgets, and degrades gracefully."""

    def __init__(self, cfg: EvolveConfig | None = None) -> None:
        self.cfg = cfg or EvolveConfig()
        self.ollama = OllamaBackend(model=self.cfg.controller_model, max_calls=self.cfg.ollama_call_budget)
        self.claude = ClaudeCLIBackend(
            default_model=self.cfg.deep_model,
            max_calls=self.cfg.claude_call_budget,
            allowed_tools=READ_ONLY_TOOLS,
            cwd=str(self.cfg.repo_root()),
        )
        self.escalations_declined: list[str] = []

    # -- resolution ------------------------------------------------------
    def backend_for(self, role: str) -> ReasoningBackend:
        if role == "cheap_controller":
            return self.ollama if self.cfg.controller_backend == "ollama" else self.claude
        if role == "critic":
            return self.claude if self.cfg.critic_backend == "claude_cli" else self.ollama
        if role in ("deep_researcher", "implementer"):
            return self.claude if self.cfg.deep_backend == "claude_cli" else self.ollama
        raise ValueError(f"role {role!r} is not served by a model backend")

    def model_for(self, role: str) -> str:
        if role == "cheap_controller":
            return self.cfg.controller_model or ""
        if role == "critic":
            return self.cfg.critic_model
        return self.cfg.deep_model

    def status(self) -> dict[str, Any]:
        o_ok, o_why = self.ollama.available()
        c_ok, c_why = self.claude.available()
        return {
            "ollama": {**self.ollama.stats(), "available": o_ok, "reason": o_why},
            "claude_cli": {**self.claude.stats(), "available": c_ok, "reason": c_why},
            "roles": {
                role: (
                    "deterministic kernel (no model)" if role == "evaluator" else self.backend_for(role).name
                )
                for role in ROLES
            },
            "escalations_declined": list(self.escalations_declined),
        }

    # -- calling ---------------------------------------------------------
    def complete(
        self,
        role: str,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> BackendResult:
        """Call the backend for ``role``, falling back to local when configured.

        A declined escalation is *recorded*, not hidden: a hypothesis written by
        the 27B local model when the strong model was unavailable is a different
        artefact from one written by the strong model, and the archive says which.
        """
        backend = self.backend_for(role)
        usable, reason = backend.available()
        if not usable and backend is self.claude and self.cfg.escalation_fallback:
            self.escalations_declined.append(f"{role}: {reason}")
            fallback = self.ollama
            f_ok, f_why = fallback.available()
            if not f_ok:
                return BackendResult(
                    ok=False,
                    role=role,
                    backend="none",
                    error=f"no backend available for {role}: claude={reason}; ollama={f_why}",
                )
            result = fallback.complete(
                system=system, user=user, role=role, schema=schema, timeout_s=timeout_s
            )
            result.raw["escalation_declined"] = reason
            return result
        return backend.complete(
            system=system,
            user=user,
            role=role,
            schema=schema,
            model=self.model_for(role) or None,
            timeout_s=timeout_s,
            **({} if backend is self.ollama else kwargs),
        )

    def budget(self) -> dict[str, Any]:
        return {
            "claude_calls": f"{self.claude.n_calls}/{self.cfg.claude_call_budget}",
            "ollama_calls": f"{self.ollama.n_calls}/{self.cfg.ollama_call_budget}",
            "claude_cost_estimate_usd": round(self.claude.total_cost_estimate_usd, 6),
            "claude_seconds": round(self.claude.total_seconds, 3),
            "ollama_seconds": round(self.ollama.total_seconds, 3),
        }


__all__ = ["ROLES", "BackendRegistry"]
