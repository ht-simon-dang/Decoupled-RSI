"""RunContext — the shared service bundle threaded through engine and agents.

Agents never import global singletons; everything they need (settings, stores,
telemetry, LLM provider, memory, cancellation) arrives via this context. That
keeps agents testable and the data-flow explicit.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

from ..contracts.run import Run, TaskEvent
from ..observability.telemetry import Telemetry
from ..observability.toolmeter import carry_metering
from ..settings.config import Settings
from ..storage.artifacts import ArtifactStore
from ..storage.db import StateStore
from ..util import utcnow_iso

if TYPE_CHECKING:
    from ..memory.store import MemoryStore
    from ..tools.llm_provider_tool import LLMProvider
    from .registry import CapabilityRegistry

T = TypeVar("T")
R = TypeVar("R")


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise RunCancelledError("run cancelled")


class RunCancelledError(RuntimeError):
    pass


class RunContext:
    def __init__(
        self,
        *,
        settings: Settings,
        store: StateStore,
        artifacts: ArtifactStore,
        registry: CapabilityRegistry,
        provider: LLMProvider,
        memory: MemoryStore,
        run: Run,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.artifacts = artifacts
        self.registry = registry
        self.provider = provider
        self.memory = memory
        self.run = run
        self.telemetry = telemetry or Telemetry(run.id, mode=settings.telemetry)
        self.cancel_token = CancellationToken()
        self.correlation_id = run.id
        # Intra-process cache for non-serializable state (fitted models, datasets).
        # Never authoritative: rebuilt deterministically from persisted inputs on
        # resume, so a fresh process can continue a run without it.
        self.blackboard: dict[str, Any] = {}

    # --- events ------------------------------------------------------------
    def emit_event(
        self,
        event: str,
        *,
        task_id: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        **detail: Any,
    ) -> None:
        ev = TaskEvent(
            ts=utcnow_iso(),
            run_id=self.run.id,
            task_id=task_id,
            event=event,
            from_state=from_state,
            to_state=to_state,
            detail=detail,
            correlation_id=self.correlation_id,
        )
        self.store.append_event(ev)

    # --- persistence helpers ----------------------------------------------
    def save_object(self, otype: str, obj: Any, *, program_id: str | None = None) -> None:
        self.store.put_object(otype, obj, run_id=self.run.id, program_id=program_id or self.run.program_id)

    def list_objects(self, otype: str) -> list[dict[str, Any]]:
        return self.store.list_objects(otype, run_id=self.run.id)

    # --- bounded candidate-batch fan-out ----------------------------------
    def map_candidates(self, fn: Callable[[T], R], items: list[T], *, label: str = "batch") -> list[R]:
        """Apply ``fn`` across a candidate batch with bounded concurrency.

        Deterministic runs execute sequentially in input order (stable IDs);
        otherwise a bounded thread pool fans the batch out. This is the
        candidate-batch fan-out with a concurrency cap.
        """
        self.cancel_token.raise_if_cancelled()
        if self.settings.deterministic or self.settings.max_concurrency <= 1 or len(items) <= 1:
            return [fn(x) for x in items]
        workers = min(self.settings.max_concurrency, len(items))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=label) as ex:
            return list(ex.map(carry_metering(fn), items))
