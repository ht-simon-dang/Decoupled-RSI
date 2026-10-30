"""Append-only experiments retain context, censoring and technical failures."""

from __future__ import annotations

import time
from collections import defaultdict

from .config import digest, encode
from .contracts import Brief, Candidate, Observation
from .delivery import allowed_candidate, public_run
from .evaluation import context_id


def ingest(store, run_id, values):
    if not 1 <= len(values) <= 128:
        raise ValueError("Supply 1..128 observations")
    run = store.get(run_id)
    brief = Brief.model_validate(run["brief"])
    candidates = {c["id"]: Candidate.model_validate(c) for c in run["payload"].get("candidates", [])}
    observations = [Observation.model_validate(value) for value in values]
    if len({o.id for o in observations}) != len(observations):
        raise ValueError("Observation ids must be unique")
    for o in observations:
        if o.candidate_id not in candidates:
            raise ValueError("Observation refers to an unknown candidate identity")
        if o.context_id not in {context_id(candidates[o.candidate_id], s) for s in brief.states}:
            raise ValueError("Measurement context differs from this format/antigen state")
        existing = store.db.execute(
            "SELECT data FROM observations WHERE run_id=? AND id=?", (run_id, o.id)
        ).fetchone()
        if existing and existing[0] != encode(o.model_dump()):
            raise ValueError("Observation ids are immutable; use a new id for corrections")
    with store.db:
        for o in observations:
            store.db.execute(
                "INSERT OR IGNORE INTO observations VALUES(?,?,?,?)",
                (run_id, o.id, encode(o.model_dump()), time.time()),
            )
    store.event(run_id, "feedback_imported", {"count": len(observations)})
    return analyze(store, run_id)


def analyze(store, run_id):
    run = store.get(run_id)
    observations = store.observation_rows(run_id)
    groups = defaultdict(list)
    for o in observations:
        key = digest(
            {k: o[k] for k in ("context_id", "endpoint", "unit", "conditions", "batch", "glycoform")}
        )
        groups[key].append(o)
    summaries, next_steps = [], set()
    for key, rows in groups.items():
        measured = [r for r in rows if r["status"] == "measured"]
        resolved = [r for r in measured if r["outcome"] != "undetermined"]
        summaries.append(
            {
                "group": key,
                "endpoint": rows[0]["endpoint"],
                "context_id": rows[0]["context_id"],
                "conditions": rows[0]["conditions"],
                "unit": rows[0]["unit"],
                "attempts": len(rows),
                "measured": len(measured),
                "resolved": len(resolved),
                "positive": sum(r["outcome"] == "positive" for r in resolved),
                "negative": sum(r["outcome"] == "negative" for r in resolved),
                "censored": sum(r["status"] == "censored" for r in rows),
                "technical_failures": sum(r["status"] == "technical_failure" for r in rows),
                "denominator_note": "Counts are assay records/replicates in this context, not independent candidate families",
            }
        )
        for row in rows:
            if row["status"] == "technical_failure":
                next_steps.add("修复测量或样品问题；技术失败不作为分子阴性标签。")
            elif row["status"] == "censored":
                next_steps.add("保留检测限与删失方向；扩大有效测量范围后再更新属性模型。")
            elif row["outcome"] == "negative":
                next_steps.add(
                    {
                        "binding": "检查抗原状态、表位与配对，比较新的结合几何。",
                        "mechanism": "分别核对结合与功能，优先调整表位或格式。",
                        "developability": "在保留结合和功能约束下进行局部优化。",
                        "expression": "先排查表达、装配和格式，再决定是否重设计。",
                    }.get(row["endpoint"], "复核该终点的证据与适用条件。")
                )
    public = public_run(run)
    allowed = {c["id"] for c in public["candidates"]}
    return {
        "groups": summaries,
        "next_steps": sorted(next_steps),
        "observations": [o for o in observations if o["candidate_id"] in allowed],
        "withheld_observation_count": sum(o["candidate_id"] not in allowed for o in observations),
        "learning": "Condition-matched data retained; no pretrained project-specific property model is claimed",
    }


def review(store, run_id, candidate_id, decision, note):
    if (
        decision not in {"accept", "reject", "request_changes"}
        or not isinstance(note, str)
        or not 1 <= len(note) <= 2000
    ):
        raise ValueError("Supply an explicit human decision and a note")
    allowed_candidate(store.get(run_id), candidate_id)
    with store.db:
        store.db.execute(
            "INSERT INTO reviews(run_id,candidate_id,decision,note,created) VALUES(?,?,?,?,?)",
            (run_id, candidate_id, decision, note, time.time()),
        )
    store.event(run_id, "human_review", {"candidate_id": candidate_id, "decision": decision})
    return {
        "candidate_id": candidate_id,
        "decision": decision,
        "note": note,
        "meaning": "Human research decision; not experimental or clinical validation",
    }
