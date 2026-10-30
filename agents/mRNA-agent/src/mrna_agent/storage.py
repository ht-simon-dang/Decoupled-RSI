"""Append-only evidence, resumable checkpoints, content addressed artifacts, disk quota."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .contracts import canonical, identity

MAX_BYTES = 50_000_000_000  # Conservative decimal 50 GB, including environments and downloads.


def data_root():
    return Path(os.environ.get("MRNA_DATA_DIR", str(Path.home() / ".cache/decoupled-rsi/mrna"))).expanduser().resolve()


def used_bytes(root=None):
    total = 0
    for directory, _, names in os.walk(root or data_root(), followlinks=False):
        for name in names:
            p = Path(directory) / name
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except FileNotFoundError:
                pass
    return total


@contextlib.contextmanager
def quota(reserve=0):
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".quota.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if used_bytes(root) + reserve > MAX_BYTES:
            raise OSError("mRNA data directory 50 GB quota would be exceeded")
        yield


class Store:
    def __init__(self, root=None):
        self.root = Path(root or data_root() / "runs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        (self.root / "locks").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "journal.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,created REAL,brief TEXT,state TEXT,payload TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,run_id TEXT REFERENCES runs(id),time REAL,kind TEXT,payload TEXT);
        CREATE TABLE IF NOT EXISTS constructs(id TEXT PRIMARY KEY,sequence_id TEXT,body TEXT);
        CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY,run_id TEXT REFERENCES runs(id),construct_id TEXT REFERENCES constructs(id),context_id TEXT,body TEXT);
        CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,construct_id TEXT REFERENCES constructs(id),context_id TEXT,body TEXT,created REAL);
        CREATE TABLE IF NOT EXISTS evolution(id TEXT PRIMARY KEY,body TEXT,created REAL);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextlib.contextmanager
    def exclusive(self, run_id):
        self.get(run_id)
        with (self.root / "locks" / (run_id + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Run already has an active executor") from exc
            yield

    def create(self, brief, instruction=None):
        run_id = uuid.uuid4().hex
        with quota(2_000_000), self.db:
            self.db.execute("INSERT INTO runs VALUES(?,?,?,?,?)", (run_id, time.time(), canonical(brief), "created", canonical({"instruction": instruction} if instruction else {})))
        self.event(run_id, "created", {"brief_sha256": identity(brief)})
        return run_id

    def get(self, run_id):
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown run ID")
        return {**dict(row), "brief": json.loads(row["brief"]), "payload": json.loads(row["payload"])}

    def list_runs(self, limit=100):
        return [{"id": r["id"], "created": r["created"], "state": r["state"],
                 "name": json.loads(r["brief"]).get("name", "Natural language task"),
                 "count": len(json.loads(r["payload"]).get("candidates", []))}
                for r in self.db.execute("SELECT * FROM runs ORDER BY created DESC LIMIT ?", (limit,))]

    def event(self, run_id, kind, body):
        with quota(len(canonical(body).encode()) + 65536), self.db:
            self.db.execute("INSERT INTO events(run_id,time,kind,payload) VALUES(?,?,?,?)", (run_id, time.time(), kind, canonical(body)))

    def events(self, run_id):
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in self.db.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))]

    def update(self, run_id, state, payload):
        with quota(len(canonical(payload).encode()) * 3 + 131072), self.db:
            # External pause/cancellation wins over a late worker checkpoint.
            current = self.get(run_id)
            if current["state"] in {"paused", "cancelled"} and state not in {"queued", "paused", "cancelled"}:
                raise InterruptedError(current["state"])
            self.db.execute("UPDATE runs SET state=?,payload=? WHERE id=?", (state, canonical(payload), run_id))
        self.event(run_id, "state", {"state": state})

    def set_brief(self, run_id, brief):
        with self.db:
            self.db.execute("UPDATE runs SET brief=? WHERE id=?", (canonical(brief), run_id))
        self.event(run_id, "target.locked", {"brief": brief, "sha256": identity(brief)})

    def artifact(self, value):
        key = identity(value)
        path = self.root / "artifacts" / (key + ".json")
        raw = canonical(value).encode()
        with quota(len(raw)):
            if not path.exists():
                tmp = path.with_suffix(".partial")
                tmp.write_bytes(raw)
                tmp.replace(path)
        return key

    def add_construct(self, construct):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO constructs VALUES(?,?,?)", (construct.id, construct.sequence_id, canonical(construct.card())))

    def add_evaluation(self, run_id, construct_id, context, body):
        key = identity({"run_id": run_id, "construct_id": construct_id, "context": context, "body": body})
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO evaluations VALUES(?,?,?,?,?)", (key, run_id, construct_id, identity(context), canonical(body)))
        return key

    def construct(self, key):
        row = self.db.execute("SELECT body FROM constructs WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Unknown construct identity")
        return json.loads(row[0])
