"""Atomic resource reservations and the campaign's actual-cost ledger.

Step 9 of the per-target evolution program. Appendix A §4 declares the campaign
caps — six experiment proposals, 12 aggregate GPU-hours, 24 hours wall, three
concurrent CPU jobs, one large GPU job, 100,000 generated LLM output tokens if
measurable. Until this module existed those were numbers in a YAML file that
nothing read. This is the thing that reads them.

Four properties are what make "no autonomous action can exceed its declared
budget unnoticed" a checkable claim rather than a promise:

**Reserve before you spend, atomically, across processes.** A reservation is one
``BEGIN IMMEDIATE`` transaction against a SQLite ledger with
``PRAGMA busy_timeout=30000`` — the same shape Step 2 used to make run identity
unique under eight concurrent claimants, reused here rather than reinvented as a
lock file. Two processes that ask for the last GPU slot at the same instant get
one admission and one refusal, and the refusal happens *before* anything is
launched.

**``requested``, ``reserved``, ``consumed`` and ``released`` are four different
numbers.** A docking batch that reserves 24 calls, burns 19 and then dies
releases 5. It does not refund the 19. A failed trial, a retry and a partial
result are all spend; Appendix A §4 says to charge them and this is where they
are charged.

**Unknown is not zero.** ``None`` in a cost record means the telemetry was
unavailable. It is stored as SQL ``NULL``, serialises as JSON ``null``, and is
reported next to an explicit ``UNAVAILABLE`` state so it can never be summed
into a total as if the resource had been free. A *measured* zero — Step 8's
``llm_output_tokens: 0``, observed by recording which modules each arm process
loaded — is a different record and stays different forever. A reservation
settled with unknown consumption is charged **in full**, because the alternative
is that an unmeasurable job is a free job.

**Wall time is enforced whether or not any provider tells you anything.** A
subprocess cap is a local guarantee: :func:`run_accounted` gives the child its
own process session, cancels the whole group on the cap (SIGTERM → grace →
SIGKILL, via Step 4's :func:`~drug_agent.selfevolve.target.worker._kill_group`),
charges the partial spend and writes a resumable checkpoint.

The public convenience surface Appendix A §7.2 declares is one function,
:func:`reserve_budget`. The ledger underneath it is deliberately larger: a
campaign needs settlement, release, restart identity, phase separation, cache
accounting and GPU telemetry, and none of those fit in four positional
arguments.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import resource as _resource
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import ContractError, ErrorCategory

Record = dict[str, Any]

#: Bumped when the ledger's meaning changes. Stored in the database so a ledger
#: file says which rules it was written under.
CONTRACT_VERSION = "1.0.0"

#: A reservation's lifecycle. ``RESERVED`` holds capacity; ``SETTLED`` has a
#: final consumption; ``RELEASED`` gave the whole reservation back unspent.
RESERVATION_STATES: tuple[str, ...] = ("RESERVED", "SETTLED", "RELEASED")

#: How a reservation ended. Every one of these is charged.
OUTCOMES: tuple[str, ...] = ("SETTLED", "FAILED", "CANCELLED", "RELEASED")

#: Telemetry states. ``MEASURED`` means a number was observed — zero included.
#: ``UNAVAILABLE`` means nothing was observed and the amount is null.
TELEMETRY_STATES: tuple[str, ...] = ("MEASURED", "UNAVAILABLE")

#: Appendix A §4 keeps adaptation spend separate from final-evaluation spend,
#: and gives baseline tuning and teacher generation their own budgets.
#: ``UNSPECIFIED`` is the honest default: it is not a fifth budget, it is a row
#: whose owner did not say, and it is visible as such in the report.
PHASES: tuple[str, ...] = (
    "ADAPTATION",
    "EVALUATION",
    "BASELINE_TUNING",
    "TEACHER_GENERATION",
    "UNSPECIFIED",
)

#: How much to trust a row. ``measured`` was observed by this ledger;
#: ``provisional`` was migrated from an earlier step's manual accounting and is
#: never silently promoted.
EVIDENCE_CLASSES: tuple[str, ...] = ("measured", "provisional")

#: Seconds between GPU samples while a job runs. One sample is always taken at
#: launch, so even a sub-second job has evidence rather than an absence.
GPU_POLL_SECONDS = 0.2

_AUTO = object()


class BudgetExceeded(ContractError):
    """A reservation would have taken a resource past its declared cap.

    A :class:`~drug_agent.selfevolve.target.contracts.ContractError`, and
    therefore a ``ValueError``, per Appendix A §7.2. Its ``category`` is
    ``INVALID_VALUE`` — the amount is out of range — and its ``field`` is the
    resource, so a caller can branch on which cap it hit without parsing text.
    """


def _fail(
    category: ErrorCategory,
    message: str,
    *,
    field: str | None = None,
    cls: type[ContractError] = ContractError,
) -> Any:
    raise cls(category, message, contract="BudgetLedger", field=field)


def _amount(value: Any, *, field: str, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        _fail(
            ErrorCategory.COST_UNRECORDED, f"{field} is null; an unknown amount is not an amount", field=field
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        _fail(ErrorCategory.INVALID_TYPE, f"{field}={value!r} is not a number", field=field)
    if not math.isfinite(numeric):
        _fail(ErrorCategory.NON_FINITE_VALUE, f"{field}={value!r} is not finite", field=field)
    if numeric < 0:
        _fail(ErrorCategory.INVALID_VALUE, f"{field}={value!r} is negative", field=field)
    return numeric


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(ErrorCategory.EMPTY_IDENTIFIER, f"{field}={value!r} is blank", field=field)
    return str(value)


def _canonical(payload: Any) -> str:
    """Key-order-independent JSON. The same facts always hash the same way."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS caps (
    resource     TEXT PRIMARY KEY,
    limit_amount REAL NOT NULL,
    unit         TEXT NOT NULL DEFAULT '',
    note         TEXT NOT NULL DEFAULT '',
    declared_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id       TEXT PRIMARY KEY,
    request_key          TEXT UNIQUE,
    resource             TEXT NOT NULL,
    phase                TEXT NOT NULL,
    owner                TEXT NOT NULL DEFAULT '',
    requested            REAL NOT NULL,
    reserved             REAL NOT NULL,
    consumed             REAL NOT NULL DEFAULT 0.0,
    released             REAL NOT NULL DEFAULT 0.0,
    state                TEXT NOT NULL,
    outcome              TEXT NOT NULL DEFAULT '',
    telemetry            TEXT,
    consumption_measured INTEGER,
    evidence_class       TEXT NOT NULL DEFAULT 'measured',
    note                 TEXT NOT NULL DEFAULT '',
    created_at           REAL NOT NULL,
    settled_at           REAL
);
CREATE INDEX IF NOT EXISTS reservations_resource ON reservations(resource);
CREATE TABLE IF NOT EXISTS costs (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    reservation_id TEXT,
    resource       TEXT NOT NULL,
    amount         REAL,
    telemetry      TEXT NOT NULL,
    phase          TEXT NOT NULL,
    owner          TEXT NOT NULL DEFAULT '',
    evidence_class TEXT NOT NULL DEFAULT 'measured',
    kind           TEXT NOT NULL DEFAULT 'SETTLEMENT',
    note           TEXT NOT NULL DEFAULT '',
    recorded_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS costs_resource ON costs(resource);
CREATE TABLE IF NOT EXISTS cache_events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key   TEXT NOT NULL,
    hit         INTEGER NOT NULL,
    owner       TEXT NOT NULL DEFAULT '',
    resource    TEXT NOT NULL DEFAULT '',
    recorded_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS interventions (
    seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    event  TEXT NOT NULL,
    detail TEXT NOT NULL,
    at     REAL NOT NULL
);
"""


class BudgetLedger:
    """The campaign's durable account of what it may spend and what it spent.

    One SQLite file. Safe to open from several processes at once: every mutation
    is a single ``BEGIN IMMEDIATE`` transaction, so the check and the write
    cannot be separated by another process's reservation.

    The file is *the* record. Restarting a campaign reopens it and the caps,
    the outstanding reservations and the spend are all still there — which is
    what stops a restart from being a way to start the budget over.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        campaign_id: str = "",
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                ("contract_version", CONTRACT_VERSION),
            )
            if campaign_id:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("campaign_id", campaign_id)
                )

    # -- connection -------------------------------------------------------
    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            # busy_timeout first, so every statement after it waits for a
            # concurrent writer instead of failing with SQLITE_BUSY.
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            with contextlib.suppress(sqlite3.DatabaseError):
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One serialisable write. ``BEGIN IMMEDIATE`` takes the write lock up
        front, so the read that decides and the write that commits cannot be
        interleaved with another process's decision."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # -- caps -------------------------------------------------------------
    def declare(self, resource: str, limit: float, *, unit: str = "", note: str = "") -> Record:
        """Register an enforceable cap for *resource*.

        A cap may be re-declared with the same value, and may be *reduced* as
        long as the reduction still covers what has already been committed —
        tightening a budget is always safe. It may never be enlarged: Appendix
        A §4 allows the pilot to change caps before the protocol freeze, but
        "pass a bigger number next time" is precisely the escape hatch that
        would make every other check here decorative.
        """
        resource = _text(resource, field="resource")
        amount = _amount(limit, field="limit")
        assert amount is not None
        with self._write() as conn:
            return self._declare_locked(conn, resource, amount, unit=unit, note=note)

    def _declare_locked(
        self, conn: sqlite3.Connection, resource: str, limit: float, *, unit: str = "", note: str = ""
    ) -> Record:
        row = conn.execute("SELECT * FROM caps WHERE resource = ?", (resource,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO caps(resource, limit_amount, unit, note, declared_at) VALUES(?,?,?,?,?)",
                (resource, limit, unit, note, time.time()),
            )
            return {"resource": resource, "limit": limit, "unit": unit, "changed": False}
        current = float(row["limit_amount"])
        if limit > current:
            _fail(
                ErrorCategory.INVALID_VALUE,
                f"the cap for {resource!r} was declared as {current!r} and cannot be enlarged to "
                f"{limit!r}; a cap that grows on demand is not a cap. Declare a new campaign "
                f"ledger if the protocol legitimately changed.",
                field=resource,
            )
        if limit < current:
            committed = self._usage_locked(conn, resource)["committed"]
            if limit < committed:
                _fail(
                    ErrorCategory.INVALID_VALUE,
                    f"the cap for {resource!r} cannot be reduced to {limit!r}: {committed!r} is "
                    f"already committed",
                    field=resource,
                )
            conn.execute("UPDATE caps SET limit_amount = ? WHERE resource = ?", (limit, resource))
            self._intervene_locked(
                conn, "cap_reduced", {"resource": resource, "from": current, "to": limit, "note": note}
            )
            return {"resource": resource, "limit": limit, "unit": row["unit"], "changed": True}
        return {"resource": resource, "limit": current, "unit": row["unit"], "changed": False}

    def caps(self) -> dict[str, Record]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM caps ORDER BY resource").fetchall()
        return {
            str(r["resource"]): {
                "resource": str(r["resource"]),
                "limit": float(r["limit_amount"]),
                "unit": str(r["unit"]),
                "note": str(r["note"]),
            }
            for r in rows
        }

    # -- reservations -----------------------------------------------------
    def reserve(
        self,
        resource: str,
        amount: float,
        limit: float | None = None,
        *,
        owner: str = "",
        request_key: str | None = None,
        phase: str = "UNSPECIFIED",
        note: str = "",
    ) -> Record:
        """Hold *amount* of *resource* before the work that spends it starts.

        ``limit`` declares the cap on first use; afterwards it must agree with
        what was declared (see :meth:`declare`). Omit it once the cap exists.

        ``request_key`` is the caller's identity for this unit of work. Passing
        one makes the reservation idempotent: a resumed campaign that replays
        the same key gets the original reservation back with ``resumed: True``
        and — if it had already been settled — ``already_completed: True``, so
        the work is neither charged nor executed a second time.

        Raises:
            BudgetExceeded: the reservation would pass the declared cap.
            ContractError: no cap is declared, the cap disagrees with a previous
                declaration, or the amount is negative, null or not finite.
        """
        resource = _text(resource, field="resource")
        want = _amount(amount, field="amount")
        assert want is not None
        if phase not in PHASES:
            _fail(ErrorCategory.INVALID_ENUM, f"phase {phase!r} is not one of {list(PHASES)}", field="phase")

        with self._write() as conn:
            if request_key is not None:
                existing = conn.execute(
                    "SELECT * FROM reservations WHERE request_key = ?", (request_key,)
                ).fetchone()
                if existing is not None:
                    return _reservation_record(existing, resumed=True)

            if limit is None:
                cap = conn.execute("SELECT * FROM caps WHERE resource = ?", (resource,)).fetchone()
                if cap is None:
                    _fail(
                        ErrorCategory.BUDGET_UNDECLARED,
                        f"no cap is declared for {resource!r}; nothing can be scheduled against it",
                        field=resource,
                    )
                declared = float(cap["limit_amount"])
            else:
                declared = float(_amount(limit, field="limit") or 0.0)
                self._declare_locked(conn, resource, declared)

            usage = self._usage_locked(conn, resource)
            available = declared - usage["committed"]
            if want > available + 1e-12:
                _fail(
                    ErrorCategory.INVALID_VALUE,
                    f"reserving {want!r} of {resource!r} would exceed the declared cap of "
                    f"{declared!r}: {usage['committed']!r} is already committed and only "
                    f"{max(available, 0.0)!r} remains",
                    field=resource,
                    cls=BudgetExceeded,
                )
            reservation_id = f"res_{uuid.uuid4().hex[:16]}"
            now = time.time()
            conn.execute(
                "INSERT INTO reservations(reservation_id, request_key, resource, phase, owner, "
                "requested, reserved, consumed, released, state, outcome, telemetry, "
                "consumption_measured, evidence_class, note, created_at, settled_at) "
                "VALUES(?,?,?,?,?,?,?,0.0,0.0,'RESERVED','',NULL,NULL,'measured',?,?,NULL)",
                (reservation_id, request_key, resource, phase, owner, want, want, note, now),
            )
            row = conn.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            return _reservation_record(row, resumed=False)

    def settle(
        self,
        reservation_id: str,
        *,
        consumed: float | None = None,
        telemetry: Any = _AUTO,
        status: str = "SETTLED",
        note: str = "",
        extra_costs: Mapping[str, float | None] | None = None,
    ) -> Record:
        """Close a reservation with what it actually spent.

        ``consumed`` is the measured amount. Anything the reservation held and
        did not spend is released — and *only* that: a call that burned 19 of
        24 keeps the 19.

        ``consumed=None`` with ``telemetry="UNAVAILABLE"`` is the honest record
        for a resource nothing could measure. The whole reservation is charged
        (an unmeasurable job is not a free job) and the cost row keeps a null
        amount, so the total never claims a number it does not have.

        ``consumed=None`` with no telemetry state at all is refused as
        ``COST_UNRECORDED``: a failed evaluation has a missing scientific value
        and a *recorded* cost.
        """
        if status not in OUTCOMES:
            _fail(
                ErrorCategory.INVALID_ENUM,
                f"status {status!r} is not one of {list(OUTCOMES)}",
                field="status",
            )

        if telemetry is _AUTO:
            telemetry_state = "MEASURED" if consumed is not None else None
        else:
            telemetry_state = telemetry
        if telemetry_state is not None and telemetry_state not in TELEMETRY_STATES:
            _fail(
                ErrorCategory.INVALID_ENUM,
                f"telemetry {telemetry_state!r} is not one of {list(TELEMETRY_STATES)}",
                field="telemetry",
            )
        if consumed is None and telemetry_state != "UNAVAILABLE":
            _fail(
                ErrorCategory.COST_UNRECORDED,
                f"reservation {reservation_id} was settled with neither a measured consumption "
                f"nor an explicit UNAVAILABLE telemetry state; consumption that is not recorded "
                f"makes the campaign's ledger lie",
                field="consumed",
            )
        if consumed is not None and telemetry_state == "UNAVAILABLE":
            _fail(
                ErrorCategory.INVALID_VALUE,
                f"reservation {reservation_id} reports consumption {consumed!r} and telemetry "
                f"UNAVAILABLE at the same time; one of them is wrong",
                field="telemetry",
            )
        measured = _amount(consumed, field="consumed", allow_none=True)

        with self._write() as conn:
            row = self._require_reservation(conn, reservation_id)
            if str(row["state"]) != "RESERVED":
                _fail(
                    ErrorCategory.INCONSISTENT_STATUS,
                    f"reservation {reservation_id} is already {row['state']}; settling it twice "
                    f"would charge the same work twice",
                    field="reservation_id",
                )
            held = float(row["reserved"])
            if measured is None:
                charged, released, known = held, 0.0, False
            else:
                charged, released, known = measured, max(held - measured, 0.0), True
                if measured > held:
                    self._intervene_locked(
                        conn,
                        "reservation_overrun",
                        {
                            "reservation_id": reservation_id,
                            "resource": str(row["resource"]),
                            "reserved": held,
                            "consumed": measured,
                        },
                    )
            conn.execute(
                "UPDATE reservations SET consumed=?, released=?, state=?, outcome=?, telemetry=?, "
                "consumption_measured=?, note=?, settled_at=? WHERE reservation_id=?",
                (
                    charged,
                    released,
                    "SETTLED",
                    status,
                    telemetry_state,
                    1 if known else 0,
                    note or str(row["note"]),
                    time.time(),
                    reservation_id,
                ),
            )
            self._cost_locked(
                conn,
                resource=str(row["resource"]),
                amount=measured,
                telemetry=str(telemetry_state),
                phase=str(row["phase"]),
                owner=str(row["owner"]),
                evidence_class=str(row["evidence_class"]),
                reservation_id=reservation_id,
                kind="SETTLEMENT",
                note=note,
            )
            for extra_resource, extra_amount in (extra_costs or {}).items():
                self._cost_locked(
                    conn,
                    resource=_text(extra_resource, field="resource"),
                    amount=_amount(extra_amount, field=extra_resource, allow_none=True),
                    telemetry="MEASURED" if extra_amount is not None else "UNAVAILABLE",
                    phase=str(row["phase"]),
                    owner=str(row["owner"]),
                    evidence_class=str(row["evidence_class"]),
                    reservation_id=reservation_id,
                    kind="OBSERVATION",
                    note=note,
                )
            updated = conn.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            return _reservation_record(updated, resumed=False)

    def release(self, reservation_id: str, *, note: str = "") -> Record:
        """Give back a whole reservation that spent nothing.

        This is the concurrency-slot path: holding ``max_concurrent_gpu_jobs``
        costs no capacity, it only excludes other holders. Anything that *did*
        spend goes through :meth:`settle` instead.
        """
        with self._write() as conn:
            row = self._require_reservation(conn, reservation_id)
            if str(row["state"]) != "RESERVED":
                _fail(
                    ErrorCategory.INCONSISTENT_STATUS,
                    f"reservation {reservation_id} is already {row['state']}",
                    field="reservation_id",
                )
            conn.execute(
                "UPDATE reservations SET released=?, state='RELEASED', outcome='RELEASED', "
                "telemetry='MEASURED', consumption_measured=1, note=?, settled_at=? "
                "WHERE reservation_id=?",
                (float(row["reserved"]), note or str(row["note"]), time.time(), reservation_id),
            )
            updated = conn.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            return _reservation_record(updated, resumed=False)

    def _require_reservation(self, conn: sqlite3.Connection, reservation_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
        ).fetchone()
        if row is None:
            _fail(
                ErrorCategory.INVALID_REFERENCE,
                f"no reservation {reservation_id!r} in {self.path}",
                field="reservation_id",
            )
        return row

    # -- usage ------------------------------------------------------------
    def usage(self, resource: str) -> Record:
        """What *resource* has committed, and what is still available."""
        with self._connect() as conn:
            return self._usage_locked(conn, _text(resource, field="resource"))

    def _usage_locked(self, conn: sqlite3.Connection, resource: str) -> Record:
        rows = conn.execute("SELECT * FROM reservations WHERE resource = ?", (resource,)).fetchall()
        cap_row = conn.execute("SELECT * FROM caps WHERE resource = ?", (resource,)).fetchone()
        requested = sum(float(r["requested"]) for r in rows)
        reserved = sum(float(r["reserved"]) for r in rows)
        consumed = sum(float(r["consumed"]) for r in rows)
        released = sum(float(r["released"]) for r in rows)
        outstanding = sum(float(r["reserved"]) for r in rows if str(r["state"]) == "RESERVED")
        committed = consumed + outstanding
        limit = float(cap_row["limit_amount"]) if cap_row is not None else None
        return {
            "resource": resource,
            "limit": limit,
            "requested": round(requested, 9),
            "reserved": round(reserved, 9),
            "consumed": round(consumed, 9),
            "released": round(released, 9),
            "outstanding": round(outstanding, 9),
            "committed": round(committed, 9),
            "available": None if limit is None else round(limit - committed, 9),
            "reservations": len(rows),
            "unknown_consumption": sum(1 for r in rows if r["consumption_measured"] == 0),
        }

    # -- costs ------------------------------------------------------------
    def record_cost(
        self,
        resource: str,
        amount: float | None,
        *,
        telemetry: Any = _AUTO,
        phase: str = "UNSPECIFIED",
        owner: str = "",
        kind: str = "OBSERVATION",
        reservation_id: str | None = None,
        evidence_class: str = "measured",
        note: str = "",
    ) -> Record:
        """Record an observed cost that no reservation holds capacity for.

        Observation fields — peak memory, CPU seconds, output tokens — are not
        resource caps. They are recorded so the campaign can be compared, and
        ``None`` here is an unavailable measurement, never zero.
        """
        if telemetry is _AUTO:
            telemetry = "MEASURED" if amount is not None else "UNAVAILABLE"
        if telemetry not in TELEMETRY_STATES:
            _fail(ErrorCategory.INVALID_ENUM, f"telemetry {telemetry!r} is unknown", field="telemetry")
        if evidence_class not in EVIDENCE_CLASSES:
            _fail(
                ErrorCategory.INVALID_ENUM,
                f"evidence_class {evidence_class!r} is not one of {list(EVIDENCE_CLASSES)}",
                field="evidence_class",
            )
        value = _amount(amount, field=resource, allow_none=True)
        with self._write() as conn:
            return self._cost_locked(
                conn,
                resource=_text(resource, field="resource"),
                amount=value,
                telemetry=str(telemetry),
                phase=phase,
                owner=owner,
                evidence_class=evidence_class,
                reservation_id=reservation_id,
                kind=kind,
                note=note,
            )

    def _cost_locked(
        self,
        conn: sqlite3.Connection,
        *,
        resource: str,
        amount: float | None,
        telemetry: str,
        phase: str,
        owner: str,
        evidence_class: str,
        reservation_id: str | None,
        kind: str,
        note: str,
    ) -> Record:
        conn.execute(
            "INSERT INTO costs(reservation_id, resource, amount, telemetry, phase, owner, "
            "evidence_class, kind, note, recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                reservation_id,
                resource,
                amount,
                telemetry,
                phase,
                owner,
                evidence_class,
                kind,
                note,
                time.time(),
            ),
        )
        return {
            "resource": resource,
            "amount": amount,
            "telemetry": telemetry,
            "phase": phase,
            "owner": owner,
            "evidence_class": evidence_class,
            "kind": kind,
        }

    # -- cache accounting -------------------------------------------------
    def cache_event(self, key: str, *, hit: bool, owner: str = "", resource: str = "") -> Record:
        """Record one cache lookup.

        Raw calls, unique misses and hits are three different numbers, and
        :meth:`cache_report` keeps them apart per owner as well as overall. A
        warm arm that answered 200 lookups from cache and a cold arm that
        computed 200 have the same raw-call count and very different spend, and
        an efficiency comparison that could not see the difference would be
        measuring the cache rather than the method.
        """
        key = _text(key, field="cache_key")
        with self._write() as conn:
            conn.execute(
                "INSERT INTO cache_events(cache_key, hit, owner, resource, recorded_at) VALUES(?,?,?,?,?)",
                (key, 1 if hit else 0, owner, resource, time.time()),
            )
        return {"cache_key": key, "hit": bool(hit), "owner": owner, "resource": resource}

    def cache_report(self) -> Record:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cache_events ORDER BY seq").fetchall()
        overall = _CacheTally()
        by_owner: dict[str, _CacheTally] = {}
        for row in rows:
            overall.add(str(row["cache_key"]), bool(row["hit"]))
            by_owner.setdefault(str(row["owner"]), _CacheTally()).add(str(row["cache_key"]), bool(row["hit"]))
        return {
            **overall.as_record(),
            "unique_keys": len(overall.keys),
            "by_owner": {owner: tally.as_record() for owner, tally in sorted(by_owner.items())},
        }

    # -- interventions ----------------------------------------------------
    def intervene(self, event: str, detail: Mapping[str, Any] | None = None) -> Record:
        with self._write() as conn:
            return self._intervene_locked(conn, event, detail)

    def _intervene_locked(
        self, conn: sqlite3.Connection, event: str, detail: Mapping[str, Any] | None = None
    ) -> Record:
        payload = dict(detail or {})
        conn.execute(
            "INSERT INTO interventions(event, detail, at) VALUES(?,?,?)",
            (event, _canonical(payload), time.time()),
        )
        return {"event": event, "detail": payload}

    def interventions(self) -> list[Record]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM interventions ORDER BY seq").fetchall()
        return [
            {"event": str(r["event"]), "detail": json.loads(str(r["detail"])), "at": float(r["at"])}
            for r in rows
        ]

    # -- GPU concurrency slot ---------------------------------------------
    @contextlib.contextmanager
    def gpu_slot(
        self, *, slots: float = 1.0, limit: float | None = None, owner: str = "", resource: str = "gpu_slots"
    ) -> Iterator[Record]:
        """Hold one of the declared concurrent GPU jobs for the block's duration.

        Acquired atomically, so two processes cannot both believe they have the
        one large GPU job Appendix A §4 allows, and released on the way out —
        including when the body raises, because a crashed trainer that kept the
        slot would deadlock every later job.
        """
        row = self.reserve(
            resource, slots, limit, owner=owner, phase="UNSPECIFIED", note="gpu concurrency slot"
        )
        try:
            yield row
        finally:
            with contextlib.suppress(ContractError):
                self.release(row["reservation_id"], note="gpu concurrency slot returned")

    # -- migration --------------------------------------------------------
    def migrate_provisional(
        self,
        ledger: Mapping[str, Any],
        *,
        source_path: str,
        owner_field: str = "baseline_id",
    ) -> Record:
        """Import an earlier step's manually kept ledger, *as provisional*.

        Step 8 was told to run its bounded probes against a manual ledger,
        because this module did not exist yet. Those rows are real measurements
        of real runs, but they were not taken by this ledger and they are not
        promoted: every row lands with ``evidence_class = "provisional"`` and
        stays separable from anything measured here.

        The GPU column arrives exactly as Step 8 recorded it — null amount,
        ``UNAVAILABLE`` telemetry — rather than being back-filled with a zero
        that nobody observed.
        """
        kind = str(ledger.get("ledger_kind") or "PROVISIONAL_MANUAL")
        arms = list(ledger.get("arms") or [])
        if not arms:
            _fail(ErrorCategory.MISSING_FIELD, "the provisional ledger declares no arms", field="arms")

        totals: dict[str, float | None] = {}
        telemetry_totals: dict[str, str] = {}
        now = time.time()
        with self._write() as conn:
            for index, arm in enumerate(arms):
                owner = str(arm.get(owner_field) or f"arm-{index}")
                wall = _amount(arm.get("wall_seconds"), field="wall_seconds", allow_none=True)
                reservation_id = f"res_prov_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    "INSERT INTO reservations(reservation_id, request_key, resource, phase, owner, "
                    "requested, reserved, consumed, released, state, outcome, telemetry, "
                    "consumption_measured, evidence_class, note, created_at, settled_at) "
                    "VALUES(?,?,?,?,?,?,?,?,0.0,'SETTLED',?,?,?,'provisional',?,?,?)",
                    (
                        reservation_id,
                        f"provisional:{source_path}:{owner}",
                        "wall_seconds",
                        "BASELINE_TUNING",
                        owner,
                        wall or 0.0,
                        wall or 0.0,
                        wall or 0.0,
                        "SETTLED" if str(arm.get("status")) == "SUCCEEDED" else "FAILED",
                        "MEASURED" if wall is not None else "UNAVAILABLE",
                        1 if wall is not None else 0,
                        f"migrated from {source_path}",
                        now,
                        now,
                    ),
                )
                for field in (
                    "wall_seconds",
                    "cpu_user_seconds",
                    "cpu_system_seconds",
                    "max_rss_kb",
                    "gpu_seconds",
                    "retries",
                    "failures",
                    "tuning_configurations",
                ):
                    if field not in arm:
                        continue
                    value = arm[field]
                    amount = None if value is None else _amount(value, field=field, allow_none=True)
                    state = str(arm.get("gpu_telemetry") or "UNAVAILABLE") if field == "gpu_seconds" else None
                    telemetry = state or ("MEASURED" if amount is not None else "UNAVAILABLE")
                    if telemetry not in TELEMETRY_STATES:
                        telemetry = "UNAVAILABLE"
                    self._cost_locked(
                        conn,
                        resource=field,
                        amount=amount,
                        telemetry=telemetry,
                        phase="BASELINE_TUNING",
                        owner=owner,
                        evidence_class="provisional",
                        reservation_id=reservation_id,
                        kind="MIGRATED",
                        note=f"migrated from {source_path}",
                    )
                    _accumulate(totals, telemetry_totals, field, amount, telemetry)
            for field, value in dict(ledger.get("totals") or {}).items():
                if field in {"gpu_telemetry", "llm_note"}:
                    continue
                if field in totals:
                    continue
                amount = None if value is None else _amount(value, field=field, allow_none=True)
                telemetry = "MEASURED" if amount is not None else "UNAVAILABLE"
                if field == "gpu_seconds":
                    telemetry = str(ledger.get("totals", {}).get("gpu_telemetry") or "UNAVAILABLE")
                self._cost_locked(
                    conn,
                    resource=field,
                    amount=amount,
                    telemetry=telemetry,
                    phase="BASELINE_TUNING",
                    owner="TOTAL",
                    evidence_class="provisional",
                    reservation_id=None,
                    kind="MIGRATED_TOTAL",
                    note=f"migrated from {source_path}",
                )
                _accumulate(totals, telemetry_totals, field, amount, telemetry)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)",
                ("provisional_source", _canonical({"source_path": source_path, "ledger_kind": kind})),
            )
            self._intervene_locked(
                conn,
                "provisional_ledger_migrated",
                {"source_path": source_path, "ledger_kind": kind, "arms": len(arms)},
            )
        out_totals: dict[str, Any] = {k: totals[k] for k in sorted(totals)}
        for field, state in telemetry_totals.items():
            if state == "UNAVAILABLE":
                out_totals[field] = None
        out_totals["gpu_telemetry"] = telemetry_totals.get("gpu_seconds", "UNAVAILABLE")
        return {
            "ledger_kind": kind,
            "source_path": source_path,
            "arms": len(arms),
            "totals": out_totals,
            "evidence_class": "provisional",
            "note": (
                "migrated with its provisional status preserved; these rows were measured by "
                "Step 8's per-process accounting, not by this ledger, and are never promoted"
            ),
        }

    # -- reporting --------------------------------------------------------
    def report(self) -> Record:
        """The whole ledger as JSON-ready data. ``null`` means unknown."""
        with self._connect() as conn:
            reservations = conn.execute("SELECT * FROM reservations ORDER BY created_at, rowid").fetchall()
            costs = conn.execute("SELECT * FROM costs ORDER BY seq").fetchall()
            meta = {str(r["key"]): str(r["value"]) for r in conn.execute("SELECT * FROM meta").fetchall()}
            caps = conn.execute("SELECT * FROM caps ORDER BY resource").fetchall()

        cost: dict[str, Record] = {}
        for row in costs:
            entry = cost.setdefault(
                str(row["resource"]),
                {
                    "measured": None,
                    "telemetry": "MEASURED",
                    "entries": 0,
                    "measured_entries": 0,
                    "unknown_entries": 0,
                },
            )
            entry["entries"] = int(entry["entries"]) + 1
            if row["amount"] is None:
                entry["unknown_entries"] = int(entry["unknown_entries"]) + 1
            else:
                entry["measured_entries"] = int(entry["measured_entries"]) + 1
                entry["measured"] = round(float(entry["measured"] or 0.0) + float(row["amount"]), 9)
        for entry in cost.values():
            if entry["unknown_entries"] and entry["measured_entries"]:
                entry["telemetry"] = "PARTIAL"
            elif entry["unknown_entries"]:
                entry["telemetry"] = "UNAVAILABLE"
            else:
                entry["telemetry"] = "MEASURED"

        attempts: dict[str, Record] = {}
        phases: dict[str, dict[str, float]] = {}
        evidence: dict[str, int] = {}
        for row in reservations:
            key = str(row["resource"])
            stat = attempts.setdefault(key, {"total": 0, "failed": 0, "cancelled": 0, "succeeded": 0})
            stat["total"] = int(stat["total"]) + 1
            outcome = str(row["outcome"])
            if outcome == "FAILED":
                stat["failed"] = int(stat["failed"]) + 1
            elif outcome == "CANCELLED":
                stat["cancelled"] = int(stat["cancelled"]) + 1
            elif outcome == "SETTLED":
                stat["succeeded"] = int(stat["succeeded"]) + 1
            charged = float(row["consumed"]) + (
                float(row["reserved"]) if str(row["state"]) == "RESERVED" else 0.0
            )
            if charged:
                bucket = phases.setdefault(str(row["phase"]), {})
                bucket[key] = round(bucket.get(key, 0.0) + charged, 9)
            evidence[str(row["evidence_class"])] = evidence.get(str(row["evidence_class"]), 0) + 1

        resources = sorted({str(r["resource"]) for r in reservations} | {str(c["resource"]) for c in caps})
        return {
            "contract_version": meta.get("contract_version", CONTRACT_VERSION),
            "campaign_id": meta.get("campaign_id", self.campaign_id),
            "ledger_path": str(self.path),
            "caps": self.caps(),
            "usage": {name: self.usage(name) for name in resources},
            "cost": cost,
            "attempts": attempts,
            "phases": phases,
            "evidence_classes": evidence,
            "cache": self.cache_report(),
            "reservations": [_reservation_record(row, resumed=False) for row in reservations],
            "interventions": self.interventions(),
            "provenance": json.loads(meta["provisional_source"]) if "provisional_source" in meta else {},
            "unknown_is_not_zero": (
                "A null amount is telemetry that was never observed. A 0.0 is a number that was. "
                "They are stored, summed and serialised differently on purpose."
            ),
        }

    def write_report(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.report(), indent=2, sort_keys=True), encoding="utf-8")
        return out


class _CacheTally:
    def __init__(self) -> None:
        self.raw = 0
        self.hits = 0
        self.misses = 0
        self.keys: set[str] = set()
        self.missed_keys: set[str] = set()

    def add(self, key: str, hit: bool) -> None:
        self.raw += 1
        self.keys.add(key)
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self.missed_keys.add(key)

    def as_record(self) -> Record:
        return {
            "raw_calls": self.raw,
            "hits": self.hits,
            "misses": self.misses,
            "unique_misses": len(self.missed_keys),
        }


def _accumulate(
    totals: dict[str, float | None],
    telemetry: dict[str, str],
    field: str,
    amount: float | None,
    state: str,
) -> None:
    if amount is None:
        telemetry[field] = "UNAVAILABLE"
        totals.setdefault(field, None)
        return
    telemetry.setdefault(field, state)
    current = totals.get(field)
    totals[field] = round((current or 0.0) + amount, 6)


def _reservation_record(row: sqlite3.Row, *, resumed: bool) -> Record:
    measured = row["consumption_measured"]
    return {
        "reservation_id": str(row["reservation_id"]),
        "request_key": row["request_key"],
        "resource": str(row["resource"]),
        "phase": str(row["phase"]),
        "owner": str(row["owner"]),
        "requested": float(row["requested"]),
        "reserved": float(row["reserved"]),
        "consumed": float(row["consumed"]),
        "released": float(row["released"]),
        "state": str(row["state"]),
        "outcome": str(row["outcome"]),
        "telemetry": row["telemetry"],
        "consumption_measured": None if measured is None else bool(measured),
        "evidence_class": str(row["evidence_class"]),
        "note": str(row["note"]),
        "created_at": float(row["created_at"]),
        "settled_at": None if row["settled_at"] is None else float(row["settled_at"]),
        "resumed": bool(resumed),
        "already_completed": str(row["state"]) != "RESERVED",
    }


# ---------------------------------------------------------------------------
# The public convenience surface (Appendix A section 7.2)
# ---------------------------------------------------------------------------
def reserve_budget(ledger: str | Path, resource: str, amount: float, limit: float) -> Record:
    """Reserve *amount* of *resource* in the ledger at *ledger*, under *limit*.

    The thin adapter Appendix A §7.2 declares. It opens (or creates) the ledger,
    declares the cap on first use, and takes the reservation in one atomic
    transaction — so the answer to "may I spend this?" and the act of spending
    it cannot be separated by another process.

    >>> import tempfile, pathlib
    >>> db = pathlib.Path(tempfile.mkdtemp()) / "budget.sqlite"
    >>> reserve_budget(db, "oracle_calls", 8, 10)["reserved"]
    8.0
    >>> reserve_budget(db, "oracle_calls", 3, 10)
    Traceback (most recent call last):
    ...
    drug_agent.selfevolve.target.budget.BudgetExceeded: [INVALID_VALUE] ...

    Raises:
        BudgetExceeded: a ``ValueError`` whose ``category`` is ``INVALID_VALUE``
            and whose ``field`` is the resource, when the reservation would pass
            the cap. Nothing is reserved.
        ContractError: ``BUDGET_UNDECLARED`` when ``limit`` is null — an unknown
            cap is not a cap — ``INVALID_VALUE`` for a negative amount or an
            attempt to enlarge a cap that was already declared, and
            ``NON_FINITE_VALUE`` for NaN or infinity.
    """
    if limit is None:
        _fail(
            ErrorCategory.BUDGET_UNDECLARED,
            f"no cap was supplied for {resource!r}; an unknown cap is not a cap",
            field=str(resource),
        )
    return BudgetLedger(ledger).reserve(resource, amount, limit)


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------
def cache_key(
    *,
    kind: str,
    molecular_state: Mapping[str, Any],
    receptor: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    parameters: Mapping[str, Any],
    checkpoint: Any = None,
) -> str:
    """The content address of one expensive computation.

    Four things decide whether a cached answer is *the same answer*, and all
    four are in the key: the standardized molecular state, the receptor
    preparation, the evaluator/model weights, and the parameters. The scorer
    seed lives in ``parameters`` and is required, because it is load-bearing —
    Step 7 measured seed 7 moving a docking score by 0.149 against seed 42, so a
    key that ignored it would serve one seed's answer to another seed's
    question. ``checkpoint`` distinguishes a resumed fit from a fresh one.

    Key order does not matter; the canonical form is sorted.

    Raises:
        ContractError: ``MISSING_FIELD`` when any component is empty, when the
            evaluator does not name an ``evaluator_hash``, or when the
            parameters do not name a ``seed``.
    """
    kind = _text(kind, field="kind")
    components = {
        "molecular_state": molecular_state,
        "receptor": receptor,
        "evaluator": evaluator,
        "parameters": parameters,
    }
    for name, value in components.items():
        if not isinstance(value, Mapping) or not value:
            _fail(
                ErrorCategory.MISSING_FIELD,
                f"cache component {name!r} is empty; a key that ignores it would serve one "
                f"computation's answer to a different computation's question",
                field=name,
            )
    if "evaluator_hash" not in evaluator:
        _fail(
            ErrorCategory.MISSING_FIELD,
            "the cache key's evaluator component does not name an evaluator_hash",
            field="evaluator",
        )
    if "seed" not in parameters:
        _fail(
            ErrorCategory.MISSING_FIELD,
            "the cache key's parameters do not name a seed; the frozen scorer's seed is "
            "load-bearing and belongs in the key",
            field="parameters",
        )
    payload = {
        "contract_version": CONTRACT_VERSION,
        "kind": kind,
        "checkpoint": checkpoint,
        **{name: dict(value) for name, value in components.items()},
    }
    return "sha256:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


# ---------------------------------------------------------------------------
# GPU telemetry
# ---------------------------------------------------------------------------
def gpu_telemetry() -> Record:
    """What this host can actually tell us about its GPUs, right now.

    Returns ``state: "MEASURED"`` with a device list, or ``state:
    "UNAVAILABLE"`` with a reason. Never a zero standing in for an absence:
    "there is no GPU telemetry here" and "the job used no GPU" are different
    facts and a campaign that confused them would report free training runs.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {
            "state": "UNAVAILABLE",
            "reason": "nvidia-smi is not on PATH",
            "devices": None,
            "source": None,
        }
    try:
        proc = subprocess.run(
            [binary, "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "state": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
            "devices": None,
            "source": binary,
        }
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "state": "UNAVAILABLE",
            "reason": f"nvidia-smi exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}",
            "devices": None,
            "source": binary,
        }
    devices: list[Record] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        with contextlib.suppress(ValueError):
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                }
            )
    if not devices:
        return {
            "state": "UNAVAILABLE",
            "reason": "nvidia-smi returned no parsable device row",
            "devices": None,
            "source": binary,
        }
    return {"state": "MEASURED", "reason": "", "devices": devices, "source": binary}


#: How far up a process tree :func:`_is_descendant` will walk before giving up.
#: Deep enough for conda-run -> driver -> DataLoader worker, shallow enough that a
#: corrupted ``/proc`` cannot spin.
_ANCESTRY_MAX_DEPTH = 32


def _parent_pid(pid: int) -> int | None:
    """*pid*'s parent, read from ``/proc``. ``None`` when it cannot be read."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _is_descendant(pid: int, ancestor: int) -> bool:
    """Whether *pid* is *ancestor* or was forked, at any depth, beneath it."""
    seen = pid
    for _ in range(_ANCESTRY_MAX_DEPTH):
        if seen == ancestor:
            return True
        if seen <= 1:
            return False
        parent = _parent_pid(seen)
        if parent is None:
            return False
        seen = parent
    return False


class _GpuSampler:
    """Polls ``nvidia-smi`` for compute processes belonging to one job.

    Read-only. It queries what is already running; it never touches clocks,
    power limits, persistence mode or any other machine setting.

    "Belonging to" is a process **group** by default, which is what
    :func:`run_accounted` needs: it gives its child its own session and the CUDA
    workers that child forks stay in it. That test is wrong for a job whose work
    happens in a deliberately *detached* subprocess, and
    :class:`~drug_agent.tools._ext.backend.ExternalBackend` launches every external
    driver with ``start_new_session=True`` — so a driver holding 3 GB of GPU memory
    has its own pgid, matches nothing, and the job reports 0.0 GPU-seconds. A
    silent zero is the one answer this module exists to prevent, so *descendant_of*
    adds the second test: a compute process that is a descendant of that pid, at
    any depth and across sessions, is this job's.
    """

    def __init__(
        self,
        pgid: int,
        *,
        interval: float = GPU_POLL_SECONDS,
        descendant_of: int | None = None,
    ) -> None:
        self.pgid = pgid
        self.descendant_of = descendant_of
        self.interval = interval
        self.samples = 0
        self.matched = 0
        self.peak_mib: int | None = None
        self.state = "MEASURED"
        self.reason = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._binary = shutil.which("nvidia-smi")
        if self._binary is None:
            self.state = "UNAVAILABLE"
            self.reason = "nvidia-smi is not on PATH"

    def start(self) -> None:
        if self.state != "MEASURED":
            return
        self.sample()
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def sample(self) -> None:
        if self._binary is None:
            return
        try:
            proc = subprocess.run(
                [self._binary, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.state = "UNAVAILABLE"
            self.reason = f"{type(exc).__name__}: {exc}"
            return
        if proc.returncode != 0:
            self.state = "UNAVAILABLE"
            self.reason = f"nvidia-smi exited {proc.returncode}"
            return
        self.samples += 1
        used_here = 0
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid, mib = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if not self._is_ours(pid):
                continue
            used_here += mib
        if used_here:
            self.matched += 1
            self.peak_mib = max(self.peak_mib or 0, used_here)

    def _is_ours(self, pid: int) -> bool:
        try:
            if os.getpgid(pid) == self.pgid:
                return True
        except (ProcessLookupError, PermissionError):
            return False
        return self.descendant_of is not None and _is_descendant(pid, self.descendant_of)

    def stop(self) -> Record:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.state != "MEASURED":
            return {
                "telemetry": "UNAVAILABLE",
                "gpu_seconds": None,
                "max_gpu_memory_mib": None,
                "samples": self.samples,
                "reason": self.reason,
            }
        return {
            "telemetry": "MEASURED",
            "gpu_seconds": round(self.matched * self.interval, 3),
            "max_gpu_memory_mib": self.peak_mib if self.peak_mib is not None else 0,
            "samples": self.samples,
            "reason": (
                "sampled nvidia-smi compute processes in this job's process group"
                + (" or beneath its pid" if self.descendant_of is not None else "")
                + "; 0.0 means the job held no CUDA context in any sample"
            ),
        }


@contextlib.contextmanager
def gpu_accounting(*, descendant_of: int | None = None, pgid: int | None = None) -> Iterator[Record]:
    """Measure the GPU seconds a block of *in-process* work consumes.

    :func:`run_accounted` covers work that is a subprocess this module launched.
    This covers the other shape: a probe that calls an adapter, which launches its
    own detached driver, which is where the CUDA context actually lives. The
    yielded record is empty until the block exits and is then filled in place with
    ``telemetry`` (``MEASURED`` / ``UNAVAILABLE``), ``gpu_seconds``,
    ``max_gpu_memory_mib``, ``samples`` and ``reason`` — the same five fields the
    ledger already knows how to read, so a caller cannot invent a sixth meaning.

    ``telemetry == "UNAVAILABLE"`` and ``gpu_seconds == 0.0`` are different facts
    and both are reported as themselves. A zero here means every poll saw no CUDA
    context belonging to this job, which is a measurement.
    """
    sampler = _GpuSampler(pgid if pgid is not None else os.getpgrp(), descendant_of=descendant_of)
    record: Record = {}
    sampler.start()
    try:
        yield record
    finally:
        record.update(sampler.stop())
        record["poll_interval_seconds"] = sampler.interval


# ---------------------------------------------------------------------------
# Accounted subprocess execution
# ---------------------------------------------------------------------------
def _session() -> Any:
    def _apply() -> None:  # pragma: no cover - runs in the forked child
        os.setsid()

    return _apply


def run_accounted(
    argv: Sequence[str],
    *,
    ledger: BudgetLedger,
    resource: str = "wall_seconds",
    wall_seconds: float,
    owner: str = "",
    phase: str = "UNSPECIFIED",
    request_key: str | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_dir: str | Path | None = None,
    gpu: bool = False,
    gpu_slots: float = 0.0,
    gpu_slot_limit: float | None = None,
    grace_seconds: float = 5.0,
) -> Record:
    """Run one subprocess inside its declared wall-time budget, and charge it.

    The reservation is taken **before** ``Popen``, so a job that does not fit
    the remaining campaign cap never starts. The child gets its own process
    session; on the cap the whole group is cancelled through Step 4's
    :func:`~drug_agent.selfevolve.target.worker._kill_group` (SIGTERM, grace,
    SIGKILL), which is the mechanism verified to leave no orphan — not a second
    one written here.

    What is charged is what was *used*, cancelled runs included. A resumable
    checkpoint is written next to the logs, naming the reservation, so a
    restarted campaign can reconcile against it instead of re-charging.

    Raises:
        BudgetExceeded: before anything is launched.
    """
    from .worker import _kill_group  # local: keeps the import graph acyclic

    reservation = ledger.reserve(
        resource,
        wall_seconds,
        owner=owner,
        phase=phase,
        request_key=request_key,
        note=" ".join(map(str, argv))[:200],
    )
    if reservation["already_completed"]:
        return {
            "status": "SKIPPED_ALREADY_COMPLETED",
            "reservation_id": reservation["reservation_id"],
            "resumed": True,
            "note": "this request key was already settled; re-running it would charge and execute it twice",
            "cost": {"wall_seconds": reservation["consumed"]},
            "timed_out": False,
            "exit_status": None,
        }

    slot_ctx: Any = contextlib.nullcontext()
    if gpu_slots:
        slot_ctx = ledger.gpu_slot(slots=gpu_slots, limit=gpu_slot_limit, owner=owner)

    logs = Path(log_dir) if log_dir is not None else None
    if logs is not None:
        logs.mkdir(parents=True, exist_ok=True)
    stdout_path = (logs / "stdout.log") if logs else None
    stderr_path = (logs / "stderr.log") if logs else None

    before = _resource.getrusage(_resource.RUSAGE_CHILDREN)
    started = time.time()
    timed_out = False
    kill_report: Record = {}
    sampler: _GpuSampler | None = None
    exit_status: int | None = None

    try:
        with slot_ctx:
            handles = [
                stdout_path.open("wb") if stdout_path else None,
                stderr_path.open("wb") if stderr_path else None,
            ]
            try:
                proc = subprocess.Popen(
                    [str(token) for token in argv],
                    cwd=str(cwd) if cwd else None,
                    env=dict(env) if env is not None else None,
                    stdout=handles[0] if handles[0] is not None else subprocess.DEVNULL,
                    stderr=handles[1] if handles[1] is not None else subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=_session(),
                )
            finally:
                for handle in handles:
                    if handle is not None:
                        handle.close()
            if gpu:
                with contextlib.suppress(ProcessLookupError, OSError):
                    sampler = _GpuSampler(os.getpgid(proc.pid))
                    sampler.start()
            try:
                exit_status = proc.wait(timeout=wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_report = _kill_group(proc, grace_s=grace_seconds)
                proc.poll()
                exit_status = proc.returncode if proc.returncode is not None else 124
    finally:
        elapsed = round(time.time() - started, 3)
        gpu_record = sampler.stop() if sampler is not None else _no_gpu_record(gpu)
        after = _resource.getrusage(_resource.RUSAGE_CHILDREN)

    status = "CANCELLED" if timed_out else ("SUCCEEDED" if exit_status == 0 else "FAILED")
    outcome = "CANCELLED" if timed_out else ("SETTLED" if exit_status == 0 else "FAILED")
    cost: Record = {
        "wall_seconds": elapsed,
        "cpu_user_seconds": round(after.ru_utime - before.ru_utime, 3),
        "cpu_system_seconds": round(after.ru_stime - before.ru_stime, 3),
        "max_rss_kb": int(after.ru_maxrss),
        "gpu": gpu_record,
    }
    ledger.settle(
        reservation["reservation_id"],
        consumed=elapsed,
        telemetry="MEASURED",
        status=outcome,
        extra_costs={
            "cpu_user_seconds": cost["cpu_user_seconds"],
            "cpu_system_seconds": cost["cpu_system_seconds"],
            "gpu_seconds": gpu_record["gpu_seconds"],
        },
    )
    record: Record = {
        "status": status,
        "exit_status": exit_status,
        "timed_out": timed_out,
        "argv": [str(token) for token in argv],
        "owner": owner,
        "phase": phase,
        "reservation_id": reservation["reservation_id"],
        "reserved_wall_seconds": float(reservation["reserved"]),
        "cost": cost,
        "kill_report": kill_report,
        "stdout_path": str(stdout_path) if stdout_path else None,
        "stderr_path": str(stderr_path) if stderr_path else None,
        "resumed": False,
    }
    record["checkpoint_path"] = str(_write_checkpoint(logs, record)) if logs else None
    return record


def _no_gpu_record(requested: bool) -> Record:
    if not requested:
        return {
            "telemetry": "UNAVAILABLE",
            "gpu_seconds": None,
            "max_gpu_memory_mib": None,
            "samples": 0,
            "reason": "GPU sampling was not requested for this job",
        }
    return {
        "telemetry": "UNAVAILABLE",
        "gpu_seconds": None,
        "max_gpu_memory_mib": None,
        "samples": 0,
        "reason": "the child exited before a process group could be sampled",
    }


def _write_checkpoint(log_dir: Path, record: Mapping[str, Any]) -> Path:
    path = log_dir / f"checkpoint-{record['reservation_id']}.json"
    path.write_text(json.dumps(dict(record), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


__all__ = [
    "CONTRACT_VERSION",
    "EVIDENCE_CLASSES",
    "GPU_POLL_SECONDS",
    "OUTCOMES",
    "PHASES",
    "RESERVATION_STATES",
    "TELEMETRY_STATES",
    "BudgetExceeded",
    "BudgetLedger",
    "cache_key",
    "gpu_accounting",
    "gpu_telemetry",
    "reserve_budget",
    "run_accounted",
]


if __name__ == "__main__":  # pragma: no cover - operator convenience
    if len(sys.argv) >= 2:
        print(json.dumps(BudgetLedger(sys.argv[1]).report(), indent=2, sort_keys=True))
