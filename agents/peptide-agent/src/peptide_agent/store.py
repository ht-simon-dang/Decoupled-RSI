from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, root: str | Path = ".peptide-agent"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "runs.sqlite3", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, brief TEXT NOT NULL,
                skill TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run_id TEXT,
                time TEXT, kind TEXT, payload TEXT);
        """)

    def create(self, brief, skill):
        run_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                            (run_id, encode(brief), skill, "created", "{}"))
        return run_id

    def event(self, run_id, kind, payload):
        with self.db:
            self.db.execute("INSERT INTO events (run_id,time,kind,payload) VALUES (?,?,?,?)",
                            (run_id, datetime.now(timezone.utc).isoformat(), kind, encode(payload)))

    def artifact(self, value):
        data = encode(value).encode()
        digest = hashlib.sha256(data).hexdigest()
        path = self.artifacts / f"{digest}.json"
        if not path.exists():
            path.write_bytes(data)
        return digest

    def update(self, run_id, state, payload):
        with self.db:
            self.db.execute("UPDATE runs SET state=?, payload=? WHERE id=?",
                            (state, encode(payload), run_id))
        self.event(run_id, "state", {"state": state})

    def get(self, run_id):
        row = self.db.execute("SELECT brief,skill,state,payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown run id")
        return dict(id=run_id, brief=json.loads(row[0]), skill=row[1], state=row[2], payload=json.loads(row[3]))

    def events(self, run_id):
        return [{"id": i, "time": t, "kind": k, "payload": json.loads(p)} for i, t, k, p in self.db.execute(
            "SELECT id,time,kind,payload FROM events WHERE run_id=? ORDER BY id", (run_id,))]

    def list_runs(self, limit=100):
        result = []
        for id_, brief, skill, state, payload in self.db.execute(
                "SELECT id,brief,skill,state,payload FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)):
            b, p = json.loads(brief), json.loads(payload)
            result.append({"id": id_, "name": b.get("name", p.get("operation", "Peptide task")),
                           "source": b.get("source", "tool"), "state": state, "skill": skill,
                           "count": len(p.get("sequences", []))})
        return result

    def set_brief(self, run_id, brief, skill):
        with self.db:
            self.db.execute("UPDATE runs SET brief=?,skill=? WHERE id=?", (encode(brief), skill, run_id))
