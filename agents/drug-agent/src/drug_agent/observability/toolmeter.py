"""Tool metering — how each tool was used, and how well it performed.

The platform runs the same handful of tools thousands of times per program (RDKit
standardizes every candidate; the activity model predicts for every candidate).
Writing one journal line per call would bury the signal, so metering *aggregates*
by ``(tool_id, operation)`` and keeps the tail that actually matters: failures,
and any quality metrics a tool reports about its own output.

Nothing here is mandatory. When no ledger is bound to the current context every
entry point is a no-op with a single ``ContextVar`` read, so instrumented tools
behave exactly as they did before — this is an observability seam, not a
dependency.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any


@dataclass
class ToolUsage:
    """Aggregated usage of one ``(tool_id, operation)`` pair."""

    tool_id: str
    operation: str
    backend: str = ""
    n_calls: int = 0
    n_errors: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float | None = None
    #: Per-metric running aggregates the tool reported about its own output,
    #: e.g. generation validity, or the fraction of in-domain predictions.
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    #: First few distinct failures, kept verbatim — the tail is what you debug.
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    #: Guards every counter above. The Engine fans candidate batches across a
    #: thread pool, so one entry really is written from several threads at once.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def observe(self, elapsed_ms: float, *, error: str | None = None) -> None:
        with self._lock:
            self.n_calls += 1
            self.total_ms += elapsed_ms
            self.max_ms = max(self.max_ms, elapsed_ms)
            self.min_ms = elapsed_ms if self.min_ms is None else min(self.min_ms, elapsed_ms)
            if error is not None:
                self.n_errors += 1
                if len(self.errors) < 5 and error not in self.errors:
                    self.errors.append(error)

    def observe_metric(self, key: str, value: float) -> None:
        with self._lock:
            agg = self.metrics.setdefault(key, {"n": 0.0, "sum": 0.0, "min": value, "max": value})
            agg["n"] += 1
            agg["sum"] += value
            agg["min"] = min(agg["min"], value)
            agg["max"] = max(agg["max"], value)

    def to_dict(self) -> dict[str, Any]:
        metrics = {
            key: {
                "n": int(agg["n"]),
                "mean": round(agg["sum"] / agg["n"], 6) if agg["n"] else None,
                "min": round(agg["min"], 6),
                "max": round(agg["max"], 6),
            }
            for key, agg in sorted(self.metrics.items())
        }
        return {
            "tool_id": self.tool_id,
            "operation": self.operation,
            "backend": self.backend or None,
            "n_calls": self.n_calls,
            "n_errors": self.n_errors,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.total_ms / self.n_calls, 3) if self.n_calls else None,
            "min_ms": round(self.min_ms, 3) if self.min_ms is not None else None,
            "max_ms": round(self.max_ms, 3),
            "metrics": metrics,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


class ToolLedger:
    """Collects :class:`ToolUsage` for one unit of work (one pipeline run).

    Thread-safe, because it has to be: the Engine runs every task handler in a
    worker thread and fans candidate batches out across a pool, so one ledger is
    written from several threads at once.
    """

    def __init__(self, *, label: str = "") -> None:
        self.label = label
        self._usage: dict[tuple[str, str], ToolUsage] = {}
        self._lock = threading.Lock()

    def usage(self, tool_id: str, operation: str) -> ToolUsage:
        key = (tool_id, operation)
        with self._lock:
            if key not in self._usage:
                self._usage[key] = ToolUsage(tool_id=tool_id, operation=operation)
            return self._usage[key]

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            entries = sorted(self._usage.values(), key=lambda u: (u.tool_id, u.operation))
        return [u.to_dict() for u in entries]

    def totals(self) -> dict[str, Any]:
        calls = sum(u.n_calls for u in self._usage.values())
        errors = sum(u.n_errors for u in self._usage.values())
        ms = sum(u.total_ms for u in self._usage.values())
        return {
            "n_tools": len({u.tool_id for u in self._usage.values()}),
            "n_operations": len(self._usage),
            "n_calls": calls,
            "n_errors": errors,
            "total_ms": round(ms, 3),
        }

    def __bool__(self) -> bool:
        return bool(self._usage)


_LEDGER: ContextVar[ToolLedger | None] = ContextVar("drug_agent_tool_ledger", default=None)


def active_ledger() -> ToolLedger | None:
    """The ledger bound to the current context, if any."""
    return _LEDGER.get()


@contextmanager
def metering(ledger: ToolLedger) -> Iterator[ToolLedger]:
    """Bind ``ledger`` for the duration of the block (and any code it calls)."""
    token = _LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _LEDGER.reset(token)


@contextmanager
def measure(
    tool_id: str,
    operation: str,
    *,
    backend: str = "",
    metrics: dict[str, float] | None = None,
) -> Iterator[dict[str, float]]:
    """Time one tool call and fold it into the active ledger.

    Yields a dict the caller may fill with quality metrics discovered *during*
    the call (the caller usually cannot know them up front). Both that dict and
    the ``metrics`` argument are aggregated.
    """
    ledger = _LEDGER.get()
    observed: dict[str, float] = {}
    if ledger is None:  # unmetered — stay out of the way entirely
        yield observed
        return
    usage = ledger.usage(tool_id, operation)
    if backend and not usage.backend:
        usage.backend = backend
    started = time.perf_counter()
    error: str | None = None
    try:
        yield observed
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        usage.observe((time.perf_counter() - started) * 1000.0, error=error)
        for key, value in {**(metrics or {}), **observed}.items():
            try:
                usage.observe_metric(key, float(value))
            except (TypeError, ValueError):
                continue


def carry_metering(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` so a worker thread meters into the *submitting* thread's ledger.

    A new thread starts with an empty context, so a ``ContextVar`` set on the
    caller is invisible inside a ``ThreadPoolExecutor``. Every task handler in
    this platform runs in such a thread, which means without this the ledger
    bound around a pipeline run would record nothing at all — silently, since an
    empty ledger looks exactly like a run that used no tools.

    This rebinds one variable rather than replaying a whole
    ``contextvars.copy_context()``, and that is deliberate: a captured ``Context``
    cannot be entered twice, so a single wrapper mapped across a pool would raise
    ``RuntimeError: cannot enter context`` the moment two workers overlapped.
    Setting the variable per call is reentrant and safe under concurrency, at the
    cost of propagating only what this module owns.

    The ledger *object* is shared, so aggregates still land in one place;
    :class:`ToolLedger` is locked for that reason.
    """
    ledger = _LEDGER.get()
    if ledger is None:
        return fn  # nothing to carry — stay out of the way entirely

    @wraps(fn)
    def runner(*args: Any, **kwargs: Any) -> Any:
        token = _LEDGER.set(ledger)
        try:
            return fn(*args, **kwargs)
        finally:
            _LEDGER.reset(token)

    return runner


def note(tool_id: str, operation: str, message: str) -> None:
    """Attach a free-text note to a tool's ledger entry (deduplicated)."""
    ledger = _LEDGER.get()
    if ledger is None:
        return
    usage = ledger.usage(tool_id, operation)
    if message not in usage.notes and len(usage.notes) < 10:
        usage.notes.append(message)


def instrument(tool_id: str, operation: str) -> Any:
    """Decorator: meter every call to an adapter method.

    The backend label is read from ``self.backend`` when present, so a MOCKED
    adapter is recorded as mocked without the call site having to say so — which
    matters, because "this ran, but on the surrogate" is exactly the kind of
    thing a journal exists to remember.

    Use this for methods with nothing interesting to report beyond "it ran and
    how long it took". Where the call knows something about the *quality* of its
    own output, prefer an explicit ``with measure(...) as m`` block and put the
    numbers in ``m``.
    """

    def decorate(fn: Any) -> Any:
        @wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            with measure(tool_id, operation, backend=_backend_of(self)):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorate


def _backend_of(adapter: Any) -> str:
    """The backend name an adapter is running, for the ledger.

    Adapters that pick a backend expose it directly; the always-real ones only
    name it in their capability profile. Reading both matters because "this ran,
    but on the mock" is exactly what a journal exists to remember.
    """
    backend = getattr(adapter, "backend", None)
    if backend:
        return str(backend)
    profile = getattr(adapter, "profile", None)
    return str(getattr(profile, "backend", "") or "")
