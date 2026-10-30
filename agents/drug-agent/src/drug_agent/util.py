"""Cross-cutting primitives: deterministic clock, IDs, hashing, canonical JSON.

These are intentionally dependency-light so every other module can import them
without creating cycles. Determinism is a first-class concern: in deterministic
mode (used by tests and ``--deterministic`` runs) the clock is frozen and IDs
are drawn from a seeded counter so that run artifacts are byte-reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Deterministic clock
# ---------------------------------------------------------------------------


class Clock:
    """A clock that can run in wall-clock or frozen/stepping deterministic mode.

    A single process-wide instance is used via :func:`get_clock`. Deterministic
    mode is enabled by :func:`enable_determinism`, which freezes the epoch and
    advances a fixed step on every ``now()`` call so ordering is preserved while
    values stay reproducible across runs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._deterministic = False
        self._epoch = datetime(2025, 1, 1, tzinfo=UTC)
        self._step = timedelta(seconds=1)
        self._ticks = 0

    def enable_deterministic(self, epoch: datetime | None = None) -> None:
        with self._lock:
            # Idempotent: only reset the tick counter on the first enable so that
            # multiple runs in one process keep advancing (distinct timestamps).
            if not self._deterministic:
                self._ticks = 0
            self._deterministic = True
            if epoch is not None:
                self._epoch = epoch if epoch.tzinfo else epoch.replace(tzinfo=UTC)

    def disable_deterministic(self) -> None:
        with self._lock:
            self._deterministic = False

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    def now(self) -> datetime:
        if not self._deterministic:
            return datetime.now(UTC)
        with self._lock:
            value = self._epoch + self._step * self._ticks
            self._ticks += 1
            return value

    def now_iso(self) -> str:
        return self.now().isoformat()


_CLOCK = Clock()


def get_clock() -> Clock:
    return _CLOCK


def utcnow() -> datetime:
    return _CLOCK.now()


def utcnow_iso() -> str:
    return _CLOCK.now_iso()


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


#: Environment variable that pins the deterministic ID scope, so a run can be
#: reproduced byte-for-byte from another process without reintroducing the
#: collision this scope exists to prevent.
ID_SCOPE_ENV = "DRUGAGENT_ID_SCOPE"

_SCOPE_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyz")


def _normalize_scope(scope: str) -> str:
    """Accept a caller-pinned scope only if it can survive being an ID segment."""
    token = str(scope).strip().lower()
    if not token or len(token) > 32 or not set(token) <= _SCOPE_CHARS:
        raise ValueError(f"id scope must be 1-32 characters from [0-9a-z]; got {scope!r}")
    return token


def _mint_scope() -> str:
    """A token no other process will draw.

    ``uuid4`` is the whole guarantee here: the previous design drew nothing at
    all, so every process in deterministic mode restarted its counters at zero
    and minted the same ``run_00000000`` as every other.
    """
    import uuid

    return uuid.uuid4().hex[:10]


class _IdFactory:
    """Prefixed identifier factory.

    Non-deterministic mode uses random 128-bit hex.

    Deterministic mode uses a per-prefix monotonic counter *within a scope*.
    The scope is minted once per process (or pinned by the caller, or by
    :data:`ID_SCOPE_ENV`) and prefixes every deterministic ID, which is what
    makes two runs in one workspace two runs rather than one overwriting the
    other. Determinism is unchanged in the sense that matters — an identical
    sequence of requests under an identical scope yields an identical sequence
    of IDs — but reproducing a previous run's identifiers is now something a
    caller asks for explicitly instead of something that happens by accident.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._deterministic = False
        self._counters: dict[str, int] = {}
        self._scope = ""

    def set_deterministic(self, value: bool, *, scope: str | None = None) -> None:
        with self._lock:
            if not value:
                self._deterministic = False
                self._scope = ""
                return
            requested = scope if scope is not None else os.environ.get(ID_SCOPE_ENV) or None
            token = _normalize_scope(requested) if requested is not None else None
            if not self._deterministic:
                # Idempotent enable: counters are cleared only when first
                # switching on, so successive runs in one process keep drawing
                # distinct IDs.
                self._counters.clear()
                self._scope = token or _mint_scope()
            elif token is not None and token != self._scope:
                # An explicit re-pin (a replay) restarts the sequence it names.
                self._counters.clear()
                self._scope = token
            self._deterministic = True

    @property
    def scope(self) -> str:
        return self._scope if self._deterministic else ""

    def new(self, prefix: str) -> str:
        if not self._deterministic:
            import uuid

            return f"{prefix}_{uuid.uuid4().hex}"
        with self._lock:
            n = self._counters.get(prefix, 0)
            self._counters[prefix] = n + 1
            return f"{prefix}_{self._scope}{n:06x}"


_IDS = _IdFactory()


def new_id(prefix: str) -> str:
    """Return a fresh prefixed identifier (e.g. ``run_1a2b...``)."""
    return _IDS.new(prefix)


def current_id_scope() -> str:
    """The active deterministic ID scope, or ``""`` outside deterministic mode.

    Record it beside a run to be able to replay that run's identifiers with
    ``enable_determinism(scope=...)``.
    """
    return _IDS.scope


def enable_determinism(epoch: datetime | None = None, *, scope: str | None = None) -> None:
    """Enable deterministic clock + IDs process-wide (idempotent).

    ``scope`` pins the ID scope (see :class:`_IdFactory`); omitted, it is taken
    from :data:`ID_SCOPE_ENV` and otherwise minted once per process.
    """
    _CLOCK.enable_deterministic(epoch)
    _IDS.set_deterministic(True, scope=scope)


def disable_determinism() -> None:
    _CLOCK.disable_deterministic()
    _IDS.set_deterministic(False)


# ---------------------------------------------------------------------------
# Canonical JSON + content hashing
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding: sorted keys, no insignificant whitespace.

    Used for content hashing and cache keys. ``default=str`` makes it tolerant
    of datetimes/enums/paths so hashing never raises on rich objects.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def content_hash(obj: Any, *, algo: str = "sha256") -> str:
    """Return ``<algo>:<hexdigest>`` for canonical JSON of ``obj``."""
    data = canonical_json(obj).encode("utf-8")
    h = hashlib.new(algo)
    h.update(data)
    return f"{algo}:{h.hexdigest()}"


def hash_bytes(data: bytes, *, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(data)
    return f"{algo}:{h.hexdigest()}"


def stable_hash_int(value: str, *, modulo: int | None = None) -> int:
    """Deterministic non-cryptographic-but-stable int from a string.

    Unlike Python's salted ``hash()``, this is stable across processes, which
    matters for reproducible pseudo-random behaviour keyed on molecule IDs.
    """
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], "big")
    return n % modulo if modulo else n


def merge_dedup(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in mappings:
        out.update(m)
    return out
