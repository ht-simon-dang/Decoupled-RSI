"""SQLite journal and content addressed evidence; resumes retain consumed budgets."""
from __future__ import annotations

import json
import sqlite3
import uuid
import fcntl
from datetime import datetime, timezone
from pathlib import Path

from .config import atomic_bytes, check_space, data_root, digest, encode

TERMINAL = {"complete", "failed", "blocked", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root=None):
        self.root = Path(root or data_root() / "runs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        check_space(1_000_000)
        self.db = sqlite3.connect(self.root / "runs.sqlite3", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, created TEXT NOT NULL,
            brief TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
            time TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
            time TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
            tool TEXT NOT NULL, request_sha TEXT NOT NULL, result_sha TEXT, status TEXT NOT NULL,
            seconds REAL NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS event_run ON events(run_id,id);
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def create(self, brief, payload=None):
        run_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (run_id, now(), encode(brief).decode(), "created", encode(payload or {}).decode()))
        self.event(run_id, "created", {"brief_sha256": self.artifact(brief)})
        return run_id

    def get(self, run_id):
        row = self.db.execute("SELECT created,brief,state,payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown run id")
        return {"id": run_id, "created": row[0], "brief": json.loads(row[1]), "state": row[2], "payload": json.loads(row[3])}

    def update(self, run_id, state, payload):
        check_space(len(encode(payload)) * 3)
        with self.db:
            current = self.get(run_id)
            if current["state"] == "cancelled" and state != "queued":
                raise InterruptedError("Run cancelled")
            self.db.execute("UPDATE runs SET state=?,payload=? WHERE id=?", (state, encode(payload).decode(), run_id))
            self.db.execute("INSERT INTO events(run_id,time,kind,payload) VALUES(?,?,?,?)", (run_id, now(), "stage", encode({"stage": state}).decode()))

    def set_brief(self, run_id, brief):
        with self.db:
            self.db.execute("UPDATE runs SET brief=? WHERE id=?", (encode(brief).decode(), run_id))

    def event(self, run_id, kind, payload):
        check_space(len(encode(payload)) * 2)
        with self.db:
            self.db.execute("INSERT INTO events(run_id,time,kind,payload) VALUES(?,?,?,?)", (run_id, now(), kind, encode(payload).decode()))

    def events(self, run_id):
        return [{"id": i, "time": t, "kind": k, "payload": json.loads(p)} for i,t,k,p in self.db.execute("SELECT id,time,kind,payload FROM events WHERE run_id=? ORDER BY id", (run_id,))]

    def artifact(self, value):
        sha = digest(value)
        path = self.root / "artifacts" / f"{sha}.json"
        if not path.exists():
            atomic_bytes(path, encode(value))
        return sha

    def list_runs(self, limit=100, offset=0):
        return [{"id": i, "created": t, "name": json.loads(b).get("name", "Planning…"), "state": s,
                 "count": len(json.loads(p).get("candidates", []))} for i,t,b,s,p in self.db.execute("SELECT * FROM runs ORDER BY created DESC LIMIT ? OFFSET ?", (limit, offset))]

    def recover(self):
        lock_root=self.root/'locks';lock_root.mkdir(exist_ok=True)
        for (run_id,) in self.db.execute("SELECT id FROM runs WHERE state NOT IN ('complete','failed','blocked','cancelled')").fetchall():
            with (lock_root/f'{run_id}.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                run = self.get(run_id)
                self.update(run_id, "blocked", {**run["payload"], "error": "Interrupted process; resume from saved checkpoints"})

    def begin_call(self, run_id, tool, args, limit):
        sha = self.artifact(args)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            n = self.db.execute("SELECT count(*) FROM calls WHERE run_id=?", (run_id,)).fetchone()[0]
            if n >= limit:
                raise RuntimeError("Persistent tool call budget exhausted")
            cur = self.db.execute("INSERT INTO calls(run_id,tool,request_sha,status) VALUES(?,?,?,?)", (run_id, tool, sha, "running"))
        self.event(run_id, "tool.start", {"tool": tool, "input_sha256": sha, "call": cur.lastrowid})
        return cur.lastrowid

    def finish_call(self, run_id, call_id, result, seconds):
        sha = self.artifact(result)
        with self.db:
            self.db.execute("UPDATE calls SET result_sha=?,status=?,seconds=? WHERE id=?", (sha, result["status"], seconds, call_id))
        self.event(run_id, "tool.finish", {"tool": result["tool"], "status": result["status"], "output_sha256": sha, "seconds": seconds, "error": result.get("error")})

    def cost(self, run_id):
        row = self.db.execute("SELECT count(*),coalesce(sum(seconds),0) FROM calls WHERE run_id=?", (run_id,)).fetchone()
        return {"tool_calls": row[0], "tool_seconds": row[1]}

    def observations(self, run_id):
        return [{"id": i, "time": t, **json.loads(p)} for i,t,p in self.db.execute("SELECT id,time,payload FROM observations WHERE run_id=? ORDER BY rowid", (run_id,))]

    def add_observation(self, run_id, value):
        return self.add_observations(run_id,[value])[0]

    def add_observations(self,run_id,values):
        check_space(len(encode(values))*3)
        results=[]
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            for value in values:
                # Source identifies the original measurement, not the importing database.
                identity={k:value[k] for k in ('candidate_id','product_id','endpoint','conditions','source','batch','replicate','substrate_id','environment_id')}
                sha=digest({'run_id':run_id,'measurement':identity})
                old=self.db.execute('SELECT payload FROM observations WHERE id=?',(sha,)).fetchone()
                if old:
                    if json.loads(old[0])!=value:raise ValueError('Conflicting duplicate measurement; retain provenance and submit an explicit new version/replicate')
                    results.append({'id':sha,'duplicate':True});continue
                self.db.execute('INSERT INTO observations VALUES(?,?,?,?)',(sha,run_id,now(),encode(value).decode()))
                self.db.execute('INSERT INTO events(run_id,time,kind,payload) VALUES(?,?,?,?)',(run_id,now(),'experiment.received',encode({'observation_sha256':sha,'candidate_id':value['candidate_id']}).decode()))
                results.append({'id':sha,'duplicate':False})
        return results
