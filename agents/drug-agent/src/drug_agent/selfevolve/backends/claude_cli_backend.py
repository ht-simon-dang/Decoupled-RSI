"""Claude Code CLI as a programmatic worker — subscription only, never the API.

The operator has a Claude Max subscription and an authenticated ``claude`` CLI.
Anthropic API usage is separate and separately billed. So this backend does
exactly one thing: it shells out to the officially supported non-interactive
mode of the installed CLI and reads its documented JSON result.

What it deliberately does not do
--------------------------------
No reverse-engineered endpoints, no browser cookies, no unofficial APIs, no
attempt to work around a rate limit, and no second account. If the CLI says the
quota is spent, this backend reports that and the controller stops; that is the
correct behaviour, not an obstacle.

Billing protection
------------------
``ANTHROPIC_API_KEY`` (and the Bedrock/Vertex/Foundry switches, which would route
the same call to a different billed provider) are **removed from the child
environment**. Nothing is written to the operator's shell configuration and no
credential is read, moved, or deleted. The one global that is *set* is
``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` when a caller asks for a cap.

``--bare`` is specifically **not** used: the documented behaviour of bare mode is
that OAuth and the keychain are never read and authentication is strictly
``ANTHROPIC_API_KEY``, which is the opposite of what is wanted here. The cost of
not using it is that the child session loads the project's ``CLAUDE.md``, hooks
and MCP config; that is acceptable in a repository the operator owns and is
documented in ``docs/self_evolution.md``.

Sandboxing
----------
Every call runs with ``--permission-mode dontAsk`` — the CLI denies anything not
already allowed, which for a role that only has to think is the whole tool
surface. Callers that need file access pass ``allowed_tools`` explicitly, and
the write-capable roles are pointed at an isolated candidate workspace via
``cwd``, never at the live tree.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import BackendResult, BackendUnavailable, ReasoningBackend

#: Environment variables that would silently redirect a call to paid API billing
#: (or a third-party provider) instead of the operator's subscription.
BILLING_REDIRECT_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_URL",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "AWS_BEARER_TOKEN_BEDROCK",
)

#: Tools a thinking-only role gets. Nothing that writes, nothing that reaches the
#: network. Read is included because "look at the file you are reasoning about"
#: is the difference between analysis and guesswork.
READ_ONLY_TOOLS = ("Read", "Glob", "Grep")

#: Reasoning-effort levels the installed CLI accepts (``claude --effort <level>``).
#: An unknown level is refused at construction rather than passed through, because
#: the CLI rejects it at call time and the campaign would have already reserved and
#: been charged for the cycle that carried it.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


class ClaudeCLIBackend(ReasoningBackend):
    """Invoke the installed, authenticated ``claude`` CLI as a subprocess."""

    name = "claude_cli"

    def __init__(
        self,
        *,
        binary: str = "claude",
        default_model: str = "sonnet",
        default_effort: str | None = None,
        transcript_dir: str | Path | None = None,
        timeout_s: float = 900.0,
        cwd: str | Path | None = None,
        allowed_tools: tuple[str, ...] = (),
        permission_mode: str = "dontAsk",
        max_calls: int | None = None,
        ledger: Any = None,
        budget_resource: str = "llm_output_tokens",
        budget_phase: str = "ADAPTATION",
        budget_owner: str = "claude_cli",
        default_max_output_tokens: int = 8192,
    ) -> None:
        self.binary = binary
        self.default_model = default_model
        #: Reasoning effort for every call this backend makes, or ``None`` to let
        #: the CLI use its own default. It is validated here, not at call time:
        #: the campaign ledger reserves tokens *before* the call, so a level the
        #: CLI would reject costs a reserved cycle to discover.
        if default_effort is not None and default_effort not in EFFORT_LEVELS:
            raise ValueError(
                f"effort {default_effort!r} is not one of {list(EFFORT_LEVELS)}; the CLI "
                f"refuses an unknown level and the cycle carrying it would already be reserved"
            )
        self.default_effort = default_effort
        #: Where to persist the FULL text of every call: the system prompt, the user prompt,
        #: the response, and the measured usage. The repository's own default is to record a
        #: prompt's content address and never its text, which is the right default for a study
        #: whose records are published. An operator who is debugging or auditing a run needs
        #: the other thing, so it is opt-in and off unless a directory is named.
        #:
        #: These files contain whatever the campaign was told, which includes the lessons pack
        #: and the campaign's own history. Treat the directory as run output, not as a record
        #: anyone else should read.
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        if self.transcript_dir is not None:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.cwd = str(cwd) if cwd else None
        self.allowed_tools = tuple(allowed_tools)
        self.permission_mode = permission_mode
        self.max_calls = max_calls
        #: Step 9's :class:`~drug_agent.selfevolve.target.budget.BudgetLedger`,
        #: when this backend runs inside a budgeted campaign. ``max_calls`` is an
        #: in-process counter that a restart resets; the ledger is durable, is
        #: shared across processes, and accounts in the unit Appendix A §4
        #: actually declares — generated output tokens. Both are enforced.
        self.ledger = ledger
        self.budget_resource = budget_resource
        self.budget_phase = budget_phase
        self.budget_owner = budget_owner
        #: What to reserve when the caller names no cap. A reservation has to be
        #: taken before the call, and the number of tokens a model will generate
        #: is not knowable before the call, so the *cap* is what is reserved and
        #: the measured usage is what is settled.
        self.default_max_output_tokens = int(default_max_output_tokens)
        #: Calls whose ``usage.output_tokens`` the CLI did not report. Counted,
        #: never silently treated as zero-token calls.
        self.n_calls_unmetered = 0
        #: Transcripts that could not be written. Counted rather than raised, and never zero
        #: by assumption: a transcript archive with silent gaps is worse than none.
        self.n_transcripts_failed = 0
        self.n_calls = 0
        self.total_seconds = 0.0
        self.total_cost_estimate_usd = 0.0
        #: Set once a call comes back reporting the subscription limit is spent,
        #: so the controller can persist and stop instead of hammering it.
        self.quota_exhausted = False
        self.last_error: str | None = None
        #: Reservation ids this backend has already closed, so the settle in
        #: ``complete``'s ``finally`` cannot charge one twice.
        self._settled: set[str] = set()

    # -- transcripts -----------------------------------------------------
    def _record_transcript(
        self,
        *,
        role: str,
        model: str,
        effort: str | None,
        system: str,
        user: str,
        response: str,
        parsed: Any,
        usage: Mapping[str, Any] | None,
        elapsed_s: float,
        session_id: str | None,
        ok: bool,
        error: str | None,
    ) -> None:
        """One JSON per call, appended-to-never, named so the order is the call order.

        Written on a best-effort basis and never allowed to fail a call: an unwritable
        transcript directory is an operator problem, not a reason to lose a campaign cycle that
        has already been paid for. A failure is counted so silence is not mistaken for success.
        """
        if self.transcript_dir is None:
            return
        try:
            index = self.n_calls
            safe_role = re.sub(r"[^A-Za-z0-9._-]+", "-", role or "call").strip("-") or "call"
            path = self.transcript_dir / f"{index:04d}-{safe_role}.json"
            path.write_text(
                json.dumps(
                    {
                        "call_index": index,
                        "role": role,
                        "model_requested": model,
                        "effort_requested": effort,
                        "ok": ok,
                        "error": error,
                        "elapsed_s": elapsed_s,
                        "session_id": session_id,
                        "usage": dict(usage or {}),
                        "system_prompt": system,
                        "user_prompt": user,
                        "response": response,
                        "structured_output": parsed,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            self.n_transcripts_failed += 1

    # -- availability ----------------------------------------------------
    def resolved_binary(self) -> str | None:
        return shutil.which(self.binary)

    def available(self) -> tuple[bool, str]:
        path = self.resolved_binary()
        if not path:
            return False, f"{self.binary!r} is not on PATH"
        if self.quota_exhausted:
            return False, "subscription usage limit reached earlier in this run"
        if self.max_calls is not None and self.n_calls >= self.max_calls:
            return False, f"call budget exhausted ({self.n_calls}/{self.max_calls})"
        return True, ""

    def billing_environment(self) -> dict[str, str]:
        """Which billing-redirecting variables are set in *our* environment.

        Reported rather than silently handled: an operator who exported an API
        key deserves to be told that the evolver is removing it from the child,
        not to discover the removal in a diff six weeks later.
        """
        return {k: "<set>" for k in BILLING_REDIRECT_VARS if os.environ.get(k)}

    def child_env(self, *, max_output_tokens: int | None = None) -> dict[str, str]:
        """The child's environment: ours, minus anything that would bill the API.

        ``os.environ`` is never mutated. The operator's shell configuration is
        never touched and no credential file is read or written.
        """
        env = {k: v for k, v in os.environ.items() if k not in BILLING_REDIRECT_VARS}
        if max_output_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(max_output_tokens))
        # A nested Claude Code session must not inherit the parent's session
        # identity, or the two will be confused for one another in logs.
        for key in ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_BRIDGE_SESSION_ID"):
            env.pop(key, None)
        return env

    # -- the call --------------------------------------------------------
    def build_command(
        self,
        *,
        system: str,
        schema: dict[str, Any] | None,
        model: str | None,
        allowed_tools: tuple[str, ...] | None = None,
        effort: str | None = None,
    ) -> list[str]:
        cmd = [self.binary, "-p", "--output-format", "json"]
        cmd += ["--model", model or self.default_model]
        resolved_effort = effort or self.default_effort
        if resolved_effort:
            cmd += ["--effort", resolved_effort]
        cmd += ["--permission-mode", self.permission_mode]
        tools = allowed_tools if allowed_tools is not None else self.allowed_tools
        if tools:
            cmd += ["--allowedTools", ",".join(tools)]
        if system:
            cmd += ["--append-system-prompt", system]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        return cmd

    def complete(
        self,
        *,
        system: str,
        user: str,
        role: str = "",
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        cwd: str | Path | None = None,
        max_output_tokens: int | None = None,
        effort: str | None = None,
    ) -> BackendResult:
        started = time.time()
        model = model or self.default_model
        effort = effort or self.default_effort
        usable, reason = self.available()
        if not usable:
            self.last_error = reason
            return self._fail(role, model, started, f"claude CLI unavailable: {reason}")

        cmd = self.build_command(
            system=system, schema=schema, model=model, allowed_tools=allowed_tools, effort=effort
        )

        # Reserve before the subprocess starts. A cap checked after the tokens
        # were generated is a report, not a budget.
        reservation = self._reserve(max_output_tokens, role=role)
        if reservation is not None and reservation.get("already_completed"):
            self.last_error = "this request key was already settled in the budget ledger"
            return self._fail(role, model, started, self.last_error)

        self.n_calls += 1
        # Every exit from here settles the reservation, including the failures.
        # It used to sit only on the normal path, so a timeout returned before
        # it was ever reached and the hold was never released: a real ledger.json
        # from the campaign series shows state RESERVED, settled_at null and
        # consumed 0.0 for a call whose child had already generated at least
        # 48,000 output tokens across three max_tokens truncations. A leaked
        # reservation is not a free call — it is a permanently held one, and in
        # a longer campaign it drives a spurious RESOURCE_BUDGET_EXHAUSTED.
        result: BackendResult | None = None
        try:
            try:
                proc = subprocess.run(
                    cmd,
                    input=user,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s or self.timeout_s,
                    cwd=str(cwd) if cwd else self.cwd,
                    env=self.child_env(max_output_tokens=max_output_tokens),
                    check=False,
                )
            except FileNotFoundError:
                self.last_error = "binary vanished between the availability check and the call"
                result = self._fail(role, model, started, self.last_error)
            except subprocess.TimeoutExpired:
                self.last_error = f"timed out after {timeout_s or self.timeout_s}s"
                result = self._fail(role, model, started, self.last_error)
            except OSError as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                result = self._fail(role, model, started, self.last_error)
            else:
                result = self._parse(proc, role=role, model=model, started=started, effort=effort)
            # Recorded here rather than inside `_parse`, which never sees the prompts. Both the
            # failures above and the normal path land on it, because a call that errored is the
            # one an operator most wants the text of.
            self._record_transcript(
                role=role,
                model=model,
                effort=effort,
                system=system,
                user=user,
                response=getattr(result, "text", "") or "",
                parsed=getattr(result, "parsed", None),
                usage={
                    "input_tokens": getattr(result, "input_tokens", None),
                    "output_tokens": getattr(result, "output_tokens", None),
                    "cost_estimate_usd": getattr(result, "cost_estimate_usd", None),
                },
                elapsed_s=getattr(result, "elapsed_s", 0.0),
                session_id=getattr(result, "session_id", None),
                ok=bool(getattr(result, "ok", False)),
                error=getattr(result, "error", None),
            )
            return result
        finally:
            self.total_seconds += time.time() - started
            # ``result`` is None only if something above raised out of this
            # method entirely; settle whatever we have rather than nothing. The
            # normal path reaches this exactly once, so nothing double-settles.
            self._settle(
                reservation,
                result
                if result is not None
                else self._fail(role, model, started, self.last_error or "the call did not return"),
                started=started,
            )

    # -- budget ----------------------------------------------------------
    def _reserve(self, max_output_tokens: int | None, *, role: str) -> dict[str, Any] | None:
        """Hold this call's output-token cap in the campaign ledger."""
        if self.ledger is None:
            return None
        cap = int(max_output_tokens or self.default_max_output_tokens)
        return dict(
            self.ledger.reserve(
                self.budget_resource,
                float(cap),
                owner=self.budget_owner,
                phase=self.budget_phase,
                note=f"claude CLI call, role={role or 'unspecified'}",
            )
        )

    def _settle(self, reservation: dict[str, Any] | None, result: BackendResult, *, started: float) -> None:
        """Charge what the call generated — or charge the cap and say why not.

        The CLI's documented JSON result carries ``usage.output_tokens``. When it
        does, that is a measurement and it is what is charged. When it does not —
        a crash, a non-JSON reply, a quota refusal — the reservation is charged
        **in full** with ``UNAVAILABLE`` telemetry: an unmeasurable call is not a
        free call, and Appendix A §4's "unknown provider accounting is recorded
        as unavailable, never zero" is exactly this branch.
        """
        if reservation is None:
            return
        # Settling twice charges the same work twice and the ledger refuses it.
        # Now that settling happens in a ``finally``, an explicit guard is
        # cheaper than reasoning about every path that reaches it.
        reservation_id = str(reservation.get("reservation_id") or "")
        if not reservation_id or reservation_id in self._settled:
            return
        self._settled.add(reservation_id)
        tokens = result.output_tokens
        measured = isinstance(tokens, (int, float)) and not isinstance(tokens, bool)
        if not measured:
            self.n_calls_unmetered += 1
        self.ledger.settle(
            reservation["reservation_id"],
            consumed=float(tokens) if tokens is not None and measured else None,
            telemetry="MEASURED" if measured else "UNAVAILABLE",
            status="SETTLED" if result.ok else "FAILED",
            note="" if measured else "the claude CLI reported no usage.output_tokens for this call",
            extra_costs={
                "llm_input_tokens": float(result.input_tokens)
                if isinstance(result.input_tokens, (int, float))
                else None,
                "llm_calls": 1.0,
                "llm_wall_seconds": round(time.time() - started, 3),
                "llm_cost_estimate_usd": (
                    float(result.cost_estimate_usd) if result.cost_estimate_usd is not None else None
                ),
            },
        )

    def _parse(
        self,
        proc: subprocess.CompletedProcess[str],
        *,
        role: str,
        model: str,
        started: float,
        effort: str | None = None,
    ) -> BackendResult:
        elapsed = round(time.time() - started, 3)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        payload: dict[str, Any] | None = None
        if stdout:
            try:
                candidate = json.loads(stdout)
                if isinstance(candidate, dict):
                    payload = candidate
            except json.JSONDecodeError:
                payload = None

        if payload is None:
            detail = stderr or stdout or f"exit {proc.returncode} with no output"
            self.last_error = detail[:500]
            if _looks_like_quota(detail):
                self.quota_exhausted = True
            return BackendResult(
                ok=False,
                error=f"claude CLI returned no JSON (exit {proc.returncode}): {detail[:500]}",
                backend=self.name,
                model=model,
                role=role,
                elapsed_s=elapsed,
                raw={"stdout": stdout[:4000], "stderr": stderr[:2000], "returncode": proc.returncode},
            )

        usage = payload.get("usage") or {}
        result_text = str(payload.get("result") or "")
        is_error = bool(payload.get("is_error")) or proc.returncode != 0
        api_error = payload.get("api_error_status")
        if is_error and _looks_like_quota(result_text + " " + str(api_error)):
            self.quota_exhausted = True
        if is_error:
            self.last_error = result_text[:500] or f"exit {proc.returncode}"

        cost = payload.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            self.total_cost_estimate_usd += float(cost)

        return BackendResult(
            text=result_text,
            parsed=payload.get("structured_output"),
            ok=not is_error,
            error=None if not is_error else (result_text[:1000] or f"exit {proc.returncode}"),
            backend=self.name,
            model=model,
            role=role,
            elapsed_s=elapsed,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_estimate_usd=float(cost) if isinstance(cost, (int, float)) else None,
            session_id=payload.get("session_id"),
            raw={
                "returncode": proc.returncode,
                # The model the CLI actually resolved, as opposed to the alias
                # asked for. Step 10 records the identity of the model that ran
                # and must notice if it changes mid-study, so an alias like
                # "sonnet" is not an answer to "which model produced this".
                "model_requested": model,
                # Recorded beside the model because effort changes what the call
                # costs and what it produces, so a run that cannot say which
                # level it asked for cannot be compared with one that can.
                "effort_requested": effort,
                "model_resolved": _resolved_model(payload),
                "model_usage": sorted((payload.get("modelUsage") or {}).keys())
                if isinstance(payload.get("modelUsage"), dict)
                else None,
                "subtype": payload.get("subtype"),
                "num_turns": payload.get("num_turns"),
                "duration_api_ms": payload.get("duration_api_ms"),
                "api_error_status": api_error,
                "permission_denials": payload.get("permission_denials"),
                "stderr": stderr[:1000],
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "binary": self.resolved_binary(),
            "n_calls": self.n_calls,
            "n_calls_unmetered": self.n_calls_unmetered,
            # Three different facts, never collapsed: nothing was called, every
            # call reported its tokens, no call did, or some did and some did not.
            "output_token_telemetry": _telemetry_state(self.n_calls, self.n_calls_unmetered),
            "budget_ledger": None if self.ledger is None else str(self.ledger.path),
            "seconds": round(self.total_seconds, 3),
            "cost_estimate_usd": round(self.total_cost_estimate_usd, 6),
            "quota_exhausted": self.quota_exhausted,
            "last_error": self.last_error,
            "billing_vars_in_parent_env": sorted(self.billing_environment()),
            "billing_vars_removed_from_child": sorted(BILLING_REDIRECT_VARS),
        }


def _output_tokens(entry: Any) -> float:
    if not isinstance(entry, dict):
        return 0.0
    for key in ("outputTokens", "output_tokens"):
        value = entry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _resolved_model(payload: dict[str, Any]) -> str | None:
    """The concrete model identifier the CLI reports, or ``None``.

    The documented JSON result carries per-model usage keyed by the resolved
    identifier. An alias ("sonnet") is what the caller asked for; this is what
    answered. When the CLI reports neither, the answer is ``None`` — unknown,
    never the alias dressed up as a measurement.
    """
    usage = payload.get("modelUsage")
    if isinstance(usage, dict) and usage:
        # A Claude Code session can touch more than one model — a small one for
        # its own bookkeeping, the requested one for the work. Reporting the
        # alphabetically first would name the wrong one, so the model that
        # generated the most output is the one this call is attributed to, and
        # `model_usage` keeps the full list next to it.
        return max(sorted(usage), key=lambda name: _output_tokens(usage[name]))
    for key in ("model", "modelId", "model_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _telemetry_state(n_calls: int, n_unmetered: int) -> str:
    if n_calls == 0:
        return "NO_CALLS"
    if n_unmetered == 0:
        return "MEASURED"
    if n_unmetered == n_calls:
        return "UNAVAILABLE"
    return "PARTIAL"


_QUOTA_MARKERS = (
    "usage limit",
    "rate_limit",
    "rate limit",
    "quota",
    "429",
    "billing_error",
    "credit balance",
    "upgrade to",
)


def _looks_like_quota(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _QUOTA_MARKERS)


def require_claude(binary: str = "claude") -> str:
    path = shutil.which(binary)
    if not path:
        raise BackendUnavailable(f"{binary!r} is not on PATH")
    return path


__all__ = ["BILLING_REDIRECT_VARS", "READ_ONLY_TOOLS", "ClaudeCLIBackend", "require_claude"]
