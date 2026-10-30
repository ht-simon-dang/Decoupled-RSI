"""Private checkpoints and append-only evidence, with durable tool budgets."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

from .config import atomic_json, data_root, digest, encode
from .contracts import BudgetExceeded

TERMINAL = {"complete", "blocked", "failed", "budget_exhausted", "cancelled", "interrupted"}


class Store:
    def __init__(self, root=None):
        self.root = Path(root or data_root() / "runs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "runs.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY, state TEXT NOT NULL, brief TEXT NOT NULL,
            payload TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs(id), time REAL, kind TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS calls(
            id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs(id), tool TEXT, request_hash TEXT,
            status TEXT, result TEXT, started REAL, elapsed REAL);
        CREATE TABLE IF NOT EXISTS observations(
            run_id TEXT REFERENCES runs(id), id TEXT, data TEXT, created REAL, PRIMARY KEY(run_id,id));
        CREATE TABLE IF NOT EXISTS reviews(
            id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs(id), candidate_id TEXT,
            decision TEXT, note TEXT, created REAL);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def create(self, brief, payload=None):
        run_id = uuid.uuid4().hex
        now = time.time()
        with self.db:
            self.db.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                (run_id, "queued", encode(brief), encode(payload or {}), now, now),
            )
        self.event(run_id, "queued", {})
        return run_id

    def get(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise KeyError("Unknown run")
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError("Unknown run")
        result = dict(row)
        result["brief"] = json.loads(result["brief"])
        result["payload"] = json.loads(result["payload"])
        return result

    def update(self, run_id, state, payload, *, force=False):
        with self.db:
            query = "UPDATE runs SET state=?,payload=?,updated=? WHERE id=?"
            if not force:
                query += " AND state!='cancelled'"
            self.db.execute(query, (state, encode(payload), time.time(), run_id))

    def set_brief(self, run_id, brief):
        with self.db:
            self.db.execute(
                "UPDATE runs SET brief=?,updated=? WHERE id=?", (encode(brief), time.time(), run_id)
            )

    def event(self, run_id, kind, data):
        with self.db:
            self.db.execute(
                "INSERT INTO events(run_id,time,kind,data) VALUES(?,?,?,?)",
                (run_id, time.time(), kind, encode(data)),
            )

    def events(self, run_id):
        return [
            {**dict(r), "data": json.loads(r["data"])}
            for r in self.db.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
        ]

    def list_runs(self, limit=100):
        rows = self.db.execute(
            "SELECT id,state,created,updated,brief FROM runs ORDER BY created DESC LIMIT ?", (limit,)
        )
        return [
            {k: r[k] for k in ("id", "state", "created", "updated")}
            | {"name": json.loads(r["brief"]).get("name", "Qwen planning")}
            for r in rows
        ]

    def begin_call(self, run_id, tool, request_hash, limit):
        try:
            self.db.execute("BEGIN IMMEDIATE")
            count = self.db.execute("SELECT COUNT(*) FROM calls WHERE run_id=?", (run_id,)).fetchone()[0]
            if count >= limit:
                raise BudgetExceeded("Tool-call budget exhausted; resume cannot reset it")
            row = self.db.execute(
                "INSERT INTO calls(run_id,tool,request_hash,status,started) VALUES(?,?,?,?,?)",
                (run_id, tool, request_hash, "running", time.time()),
            )
            self.db.commit()
            return row.lastrowid
        except BaseException:
            self.db.rollback()
            raise

    def finish_call(self, call_id, result, elapsed):
        with self.db:
            self.db.execute(
                "UPDATE calls SET status=?,result=?,elapsed=? WHERE id=?",
                (result["status"], encode(result), elapsed, call_id),
            )

    def cached_call(self, run_id, tool, request_hash):
        row = self.db.execute(
            "SELECT result FROM calls WHERE run_id=? AND tool=? AND request_hash=? AND status='ok' ORDER BY id DESC LIMIT 1",
            (run_id, tool, request_hash),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def cost(self, run_id):
        row = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(elapsed),0) FROM calls WHERE run_id=?", (run_id,)
        ).fetchone()
        return {"tool_calls": row[0], "tool_seconds": row[1]}

    def artifact(self, value):
        sha = digest(value)
        path = self.root / "artifacts" / (sha + ".json")
        if not path.exists():
            atomic_json(path, value)
        return sha

    def observation_rows(self, run_id):
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT data FROM observations WHERE run_id=? ORDER BY created,id", (run_id,)
            )
        ]

    def reviews(self, run_id):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT candidate_id,decision,note,created FROM reviews WHERE run_id=? ORDER BY id", (run_id,)
            )
        ]
