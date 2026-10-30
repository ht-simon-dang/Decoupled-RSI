"""Relational state store (SQLite by default, any SQLAlchemy URL in production).

Holds durable control-plane state (runs, tasks, event log), a generic
content/object table used by the evidence and memory planes, a task-result
cache, and knowledge tables (entities, relationships). All rows store their
canonical object as JSON so the schema stays stable while contracts evolve.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    event,
    func,
    insert,
    inspect,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError

from ..util import utcnow_iso


class IdentityConflictError(RuntimeError):
    """A write would have taken over an identifier another run already owns.

    Two runs in one workspace used to share identifiers — deterministic IDs
    restarted at zero in every process — and every write was an unconditional
    upsert, so the second run silently replaced the first run's run row, task
    rows and child objects. Refusing the write is what makes "two runs" mean two
    runs. Nothing is ever rekeyed to resolve this: an existing record keeps the
    identifier it was published under.
    """


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    """Per-connection SQLite settings, applied before anything else runs.

    ``busy_timeout`` first, so the statements after it wait for a concurrent
    writer instead of failing. WAL is applied only when it is not already set:
    switching journal mode needs a moment with no other connection, another
    process may be doing exactly that, and one application is enough for the
    whole database. Falling back to the default journal is slower under
    concurrency, never incorrect.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        current = str((cursor.execute("PRAGMA journal_mode").fetchone() or [""])[0]).lower()
        if current != "wal":
            with contextlib.suppress(Exception):
                cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


_metadata = MetaData()

runs = Table(
    "runs",
    _metadata,
    Column("id", String, primary_key=True),
    Column("brief_id", String, index=True),
    Column("program_id", String, index=True, nullable=True),
    Column("state", String, index=True),
    Column("data", JSON),
    Column("updated_at", String),
)

tasks = Table(
    "tasks",
    _metadata,
    Column("id", String, primary_key=True),
    Column("run_id", String, index=True),
    Column("name", String, index=True),
    Column("state", String, index=True),
    Column("cache_key", String, index=True, nullable=True),
    Column("data", JSON),
    Column("updated_at", String),
)

events = Table(
    "events",
    _metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, index=True),
    Column("task_id", String, index=True, nullable=True),
    Column("ts", String),
    Column("event", String),
    Column("data", JSON),
)

#: Generic object rows.
#:
#: The key is ``(type, id, run_scope)``, where ``run_scope`` is the owning run's
#: ID or ``""`` for an object that belongs to no single run (a brief, a program).
#: It used to be ``(type, id)`` alone, which is why a second run in one
#: workspace destroyed the first run's candidates, scorecards and portfolio: the
#: objects are read back per run — every caller in the repository filters by
#: ``run_id`` — but only one row per identifier could exist, so the second
#: writer replaced the first's. Two runs now hold two rows and nothing is
#: rekeyed: each row keeps the identifier it was written under.
#:
#: The physical table is ``objects_v2`` because a primary key cannot be widened
#: in place in SQLite. :meth:`StateStore._migrate_objects` copies a legacy
#: ``objects`` table across on first open and leaves it untouched afterwards.
OBJECTS_TABLE = "objects_v2"
LEGACY_OBJECTS_TABLE = "objects"

objects = Table(
    OBJECTS_TABLE,
    _metadata,
    Column("type", String, primary_key=True),
    Column("id", String, primary_key=True),
    Column("run_scope", String, primary_key=True, default=""),
    Column("run_id", String, index=True, nullable=True),
    Column("program_id", String, index=True, nullable=True),
    Column("data", JSON),
    Column("updated_at", String),
)

cache = Table(
    "cache",
    _metadata,
    Column("cache_key", String, primary_key=True),
    Column("run_id", String, index=True, nullable=True),
    Column("data", JSON),
    Column("created_at", String),
)

entities = Table(
    "entities",
    _metadata,
    Column("id", String, primary_key=True),
    Column("program_id", String, index=True, nullable=True),
    Column("type", String, index=True),
    Column("name", String, index=True),
    Column("data", JSON),
)

relationships = Table(
    "relationships",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("program_id", String, index=True, nullable=True),
    Column("subject_id", String, index=True),
    Column("predicate", String, index=True),
    Column("object_id", String, index=True),
    Column("data", JSON),
)


class StateStore:
    """Thread-safe facade over the relational store."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        if db_url.startswith("sqlite:///"):
            from pathlib import Path

            Path(db_url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
        is_sqlite = db_url.startswith("sqlite")
        # ``timeout`` is SQLite's busy timeout. Without it a second *process*
        # opening the same workspace raised "database is locked" before it had
        # written anything — two runs in one workspace could not even start
        # concurrently, which is the same defect as overwriting, one step
        # earlier. Measured: two pipeline processes against one home, one of
        # them dead on ``PRAGMA journal_mode=WAL``.
        connect_args = {"check_same_thread": False, "timeout": 30.0} if is_sqlite else {}
        self.engine: Engine = create_engine(db_url, future=True, connect_args=connect_args)
        self._lock = threading.RLock()
        #: Which SQLite journal mode this store actually got. WAL is what lets
        #: readers and a writer coexist; it is a database-level setting, so a
        #: concurrent process may be applying it at the same moment.
        self.journal_mode = ""
        if is_sqlite:
            event.listen(self.engine, "connect", _sqlite_pragmas)
            with self.engine.begin() as conn:
                self.journal_mode = str(conn.exec_driver_sql("PRAGMA journal_mode").scalar() or "")
        self._create_schema()
        #: Rows carried over from a pre-run-scope workspace, for the record.
        self.migrated_objects = self._migrate_objects()

    # --- runs --------------------------------------------------------------
    @staticmethod
    def _run_values(run: Any) -> dict[str, Any]:
        return {
            "id": run.id,
            "brief_id": run.brief_id,
            "program_id": run.program_id,
            "state": str(run.state),
            "data": run.model_dump(mode="json"),
            "updated_at": utcnow_iso(),
        }

    def insert_run(self, run: Any) -> None:
        """Claim a new run ID, or refuse.

        This is the uniqueness constraint, enforced by the database rather than
        by a read-then-write in application code: two processes racing for the
        same identifier both reach the INSERT, and SQLite's primary key admits
        exactly one of them. :meth:`save_run` then updates the run that won.
        """
        values = self._run_values(run)
        with self._lock, self.engine.begin() as conn:
            try:
                conn.execute(insert(runs).values(**values))
            except IntegrityError as exc:
                raise IdentityConflictError(
                    f"run id {run.id!r} is already recorded in {self.db_url}; "
                    "refusing to overwrite an existing run's history"
                ) from exc

    def save_run(self, run: Any) -> None:
        self._upsert(runs, {"id": run.id}, self._run_values(run))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._get_data(runs, runs.c.id == run_id)

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(runs.c.data).order_by(runs.c.updated_at.desc()).limit(limit))
            return [self._as_dict(r[0]) for r in rows]

    # --- tasks -------------------------------------------------------------
    def save_task(self, task: Any) -> None:
        d = task.model_dump(mode="json")
        self._upsert(
            tasks,
            {"id": task.id},
            {
                "id": task.id,
                "run_id": task.run_id,
                "name": task.name,
                "state": str(task.state),
                "cache_key": task.cache_key,
                "data": d,
                "updated_at": utcnow_iso(),
            },
            owner="run_id",
        )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._get_data(tasks, tasks.c.id == task_id)

    def list_tasks(self, run_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(tasks.c.data).where(tasks.c.run_id == run_id))
            data = [self._as_dict(r[0]) for r in rows]
        return sorted(data, key=lambda t: t.get("created_at", ""))

    # --- events ------------------------------------------------------------
    def append_event(self, event: Any) -> None:
        d = event.model_dump(mode="json") if hasattr(event, "model_dump") else event
        with self._lock, self.engine.begin() as conn:
            conn.execute(
                insert(events).values(
                    run_id=d.get("run_id"),
                    task_id=d.get("task_id"),
                    ts=d.get("ts", utcnow_iso()),
                    event=d.get("event", ""),
                    data=d,
                )
            )

    def list_events(self, run_id: str, task_id: str | None = None) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            q = select(events.c.data).where(events.c.run_id == run_id)
            if task_id:
                q = q.where(events.c.task_id == task_id)
            rows = conn.execute(q.order_by(events.c.seq))
            return [self._as_dict(r[0]) for r in rows]

    # --- generic objects ---------------------------------------------------
    def put_object(
        self, otype: str, obj: Any, *, run_id: str | None = None, program_id: str | None = None
    ) -> None:
        oid = obj.id if hasattr(obj, "id") else obj["id"]
        d = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
        self._upsert(
            objects,
            {"type": otype, "id": oid, "run_scope": run_id or ""},
            {
                "type": otype,
                "id": oid,
                "run_scope": run_id or "",
                "run_id": run_id,
                "program_id": program_id,
                "data": d,
                "updated_at": utcnow_iso(),
            },
        )

    def get_object(self, otype: str, oid: str, *, run_id: str | None = None) -> dict[str, Any] | None:
        """One object by type and ID.

        ``run_id`` selects that run's copy of a per-run object; without it the
        run-free row is preferred and any run's row is the fallback, which is
        what every pre-existing caller means (briefs, programs, reports and
        control records are either run-free or already carry the run in their
        identifier).
        """
        base = (objects.c.type == otype) & (objects.c.id == oid)
        if run_id is not None:
            found = self._get_data(objects, base & (objects.c.run_scope == run_id))
            if found is not None:
                return found
        with self.engine.begin() as conn:
            row = conn.execute(
                select(objects.c.data).where(base).order_by(objects.c.run_scope).limit(1)
            ).first()
            return self._as_dict(row[0]) if row else None

    def list_objects(
        self, otype: str, *, run_id: str | None = None, program_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            q = select(objects.c.data).where(objects.c.type == otype)
            if run_id is not None:
                q = q.where(objects.c.run_id == run_id)
            if program_id is not None:
                q = q.where(objects.c.program_id == program_id)
            rows = conn.execute(q)
            return [self._as_dict(r[0]) for r in rows]

    # --- cache -------------------------------------------------------------
    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        return self._get_data(cache, cache.c.cache_key == cache_key)

    def cache_put(self, cache_key: str, data: dict[str, Any], run_id: str | None = None) -> None:
        self._upsert(
            cache,
            {"cache_key": cache_key},
            {
                "cache_key": cache_key,
                "run_id": run_id,
                "data": data,
                "created_at": utcnow_iso(),
            },
        )

    # --- memory: entities & relationships ---------------------------------
    def put_entity(
        self, entity_id: str, etype: str, name: str, data: dict[str, Any], program_id: str | None = None
    ) -> None:
        self._upsert(
            entities,
            {"id": entity_id},
            {
                "id": entity_id,
                "program_id": program_id,
                "type": etype,
                "name": name,
                "data": data,
            },
        )

    def query_entities(
        self, *, program_id: str | None = None, etype: str | None = None, name_contains: str | None = None
    ) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            q = select(entities.c.data)
            if program_id is not None:
                q = q.where(entities.c.program_id == program_id)
            if etype is not None:
                q = q.where(entities.c.type == etype)
            if name_contains:
                q = q.where(entities.c.name.like(f"%{name_contains}%"))
            rows = conn.execute(q)
            return [self._as_dict(r[0]) for r in rows]

    def add_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        data: dict[str, Any] | None = None,
        program_id: str | None = None,
    ) -> None:
        with self._lock, self.engine.begin() as conn:
            conn.execute(
                insert(relationships).values(
                    program_id=program_id,
                    subject_id=subject_id,
                    predicate=predicate,
                    object_id=object_id,
                    data=data or {},
                )
            )

    def query_relationships(
        self, *, subject_id: str | None = None, predicate: str | None = None, program_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            q = select(
                relationships.c.subject_id,
                relationships.c.predicate,
                relationships.c.object_id,
                relationships.c.data,
            )
            if subject_id is not None:
                q = q.where(relationships.c.subject_id == subject_id)
            if predicate is not None:
                q = q.where(relationships.c.predicate == predicate)
            if program_id is not None:
                q = q.where(relationships.c.program_id == program_id)
            rows = conn.execute(q)
            return [
                {"subject_id": s, "predicate": p, "object_id": o, "data": self._as_dict(d)}
                for s, p, o, d in rows
            ]

    # --- internals ---------------------------------------------------------
    @staticmethod
    def _as_dict(data: Any) -> dict[str, Any]:
        if isinstance(data, str):
            return json.loads(data)
        return data or {}

    def _get_data(self, table: Table, where: Any) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            row = conn.execute(select(table.c.data).where(where)).first()
            return self._as_dict(row[0]) if row else None

    def _create_schema(self, attempts: int = 5) -> None:
        """Create the missing tables, tolerating another process doing the same.

        ``create_all`` inspects and then creates, and those two steps are not
        one transaction: two processes opening the same fresh workspace both
        decide a table is missing, and the loser dies on "table runs already
        exists" before it has run anything. Measured with two pipeline
        processes against one home. Re-inspecting is the whole fix — the
        winner's table is the table both of them asked for.
        """
        for attempt in range(attempts):
            try:
                _metadata.create_all(self.engine, checkfirst=True)
                return
            except OperationalError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def _migrate_objects(self) -> int:
        """Carry a legacy ``objects`` table into the run-scoped one, once.

        Copy, never move: the legacy table is left exactly as it was, and every
        row keeps its own ``type``/``id``. A row that had no ``run_id`` becomes
        a run-free row; a row that had one becomes that run's row. Rows already
        present are skipped, so the migration is safe to re-enter and safe to
        run while another process is opening the same workspace.
        """
        inspector = inspect(self.engine)
        if LEGACY_OBJECTS_TABLE not in inspector.get_table_names():
            return 0
        legacy_columns = {c["name"] for c in inspector.get_columns(LEGACY_OBJECTS_TABLE)}
        if "run_scope" in legacy_columns:  # not the old shape
            return 0
        with self._lock, self.engine.begin() as conn:
            before = conn.execute(select(func.count()).select_from(objects)).scalar() or 0
            conn.exec_driver_sql(
                f"INSERT OR IGNORE INTO {OBJECTS_TABLE} "
                "(type, id, run_scope, run_id, program_id, data, updated_at) "
                "SELECT type, id, COALESCE(run_id, ''), run_id, program_id, data, updated_at "
                f"FROM {LEGACY_OBJECTS_TABLE}"
            )
            after = conn.execute(select(func.count()).select_from(objects)).scalar() or 0
        return int(after - before)

    def _upsert(
        self, table: Table, key: dict[str, Any], values: dict[str, Any], *, owner: str | None = None
    ) -> None:
        """Insert or update one row, refusing to rebind it to a different owner.

        ``owner`` names the column that says which run a row belongs to. An
        update that would move an existing row to another run is not an update,
        it is one run's history being overwritten by another's, and it raises
        :class:`IdentityConflictError`. A row whose owner is unset on either
        side is shared on purpose — a brief is stored once and read by every run
        of it — so only two *different* non-null owners conflict.
        """
        with self._lock, self.engine.begin() as conn:
            cond = None
            for k, v in key.items():
                c = table.c[k] == v
                cond = c if cond is None else (cond & c)
            assert cond is not None
            columns = [table.c[next(iter(key))]] + ([table.c[owner]] if owner else [])
            exists = conn.execute(select(*columns).where(cond)).first()
            if exists:
                if owner:
                    held, incoming = exists[1], values.get(owner)
                    if held is not None and incoming is not None and held != incoming:
                        raise IdentityConflictError(
                            f"{table.name} row {key} belongs to {owner}={held!r}; "
                            f"refusing to rebind it to {owner}={incoming!r}"
                        )
                conn.execute(update(table).where(cond).values(**values))
            else:
                conn.execute(insert(table).values(**values))

    def clear(self) -> None:
        with self._lock, self.engine.begin() as conn:
            for t in (events, tasks, runs, objects, cache, entities, relationships):
                conn.execute(delete(t))
