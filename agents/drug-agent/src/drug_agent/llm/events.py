"""Typed event stream that powers the live WebUI.

The master agent narrates *everything* it does as a sequence of typed events —
phase transitions, streamed thinking tokens, the plan, the transcribed chemical
requirements, each dispatch, the specialists' results, the critique, and the
decision to accept / refine / replan. The web layer subscribes to a run's queue
and forwards these as Server-Sent Events. Late joiners get the backlog replayed.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

# Event types (the WebUI switches on these).
RUN_STARTED = "run_started"
PHASE_STARTED = "phase_started"
THINKING = "thinking"  # streamed reasoning-channel delta
CONTENT = "content"  # streamed answer-channel delta (usually hidden)
PLAN_READY = "plan_ready"
REQUIREMENTS_READY = "requirements_ready"
CYCLE_STARTED = "cycle_started"
DISPATCH = "dispatch"
CHEMICAL_RESULT = "chemical_result"
EVALUATION_RESULT = "evaluation_result"
CRITIQUE_READY = "critique_ready"
DECISION = "decision"
REQUIREMENTS_ADJUSTED = "requirements_adjusted"
METRICS = "metrics"  # per-cycle scoreboard point (ablation chart)
LOG = "log"
RUN_SUCCEEDED = "run_succeeded"
RUN_FAILED = "run_failed"
DONE = "done"  # stream sentinel

#: Token-level deltas. These exist to make the UI feel live and arrive in the
#: thousands per program — one journal line each would be an order of magnitude
#: more than everything else combined, and pure duplication: the harness already
#: records the complete thinking and content of every exchange in ``chat.jsonl``.
STREAMING_DELTAS: frozenset[str] = frozenset({THINKING, CONTENT})


@dataclass
class Event:
    seq: int
    ts: float
    type: str
    phase: str = ""
    cycle: int = 0
    title: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "type": self.type,
            "phase": self.phase,
            "cycle": self.cycle,
            "title": self.title,
            "data": self.data,
        }


class RunSession:
    """Holds one master-agent run: an async fan-out queue + a replayable log.

    ``emit`` is called from the agent (async). Each subscriber gets its own queue
    seeded with the backlog so a browser that connects mid-run sees the whole
    story. ``finished`` unblocks subscribers once the run ends.
    """

    def __init__(self, run_id: str, *, on_event: Callable[[Event], None] | None = None) -> None:
        self.run_id = run_id
        self.log: list[Event] = []
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._seq = 0
        self.finished = False
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        #: Synchronous side-channel for durable recording (the run journal).
        #: Deliberately not a subscriber queue: the journal must see every event
        #: as it happens, not when some consumer gets round to draining a queue.
        self.on_event = on_event

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def emit(
        self,
        type: str,
        *,
        phase: str = "",
        cycle: int = 0,
        title: str = "",
        **data: Any,
    ) -> Event:
        ev = Event(
            seq=self._next_seq(),
            ts=time.time(),
            type=type,
            phase=phase,
            cycle=cycle,
            title=title,
            data=data,
        )
        self.log.append(ev)
        if self.on_event is not None:
            # Recording must never break a run: losing the log of a run is a far
            # better failure than losing the run.
            with contextlib.suppress(Exception):
                self.on_event(ev)
        for q in list(self._subscribers):
            q.put_nowait(ev)
        # Yield control so streamed deltas actually reach subscribers promptly.
        await asyncio.sleep(0)
        return ev

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue()
        for ev in self.log:  # replay backlog
            q.put_nowait(ev)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(q)

    async def stream(self) -> AsyncIterator[Event]:
        q = self.subscribe()
        try:
            while True:
                ev = await q.get()
                yield ev
                if ev.type == DONE:
                    break
        finally:
            self.unsubscribe(q)

    async def close(self, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        self.finished = True
        self.result = result
        self.error = error
        await self.emit(DONE, title="stream closed")


class SessionRegistry:
    """Process-wide registry of active/finished run sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, RunSession] = {}

    def create(self, run_id: str, *, on_event: Callable[[Event], None] | None = None) -> RunSession:
        session = RunSession(run_id, on_event=on_event)
        self._sessions[run_id] = session
        return session

    def get(self, run_id: str) -> RunSession | None:
        return self._sessions.get(run_id)

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())


REGISTRY = SessionRegistry()
