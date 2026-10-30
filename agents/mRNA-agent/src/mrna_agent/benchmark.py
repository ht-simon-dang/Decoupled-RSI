"""Sealed computational audit and decoupled workflow promotion, never biological claims."""
from __future__ import annotations
import json
import time
from difflib import SequenceMatcher
from .contracts import Brief, Budget, Context, Objective, identity, canonical
from .generation import make_baseline
from .workflow import Engine

# Protein-disjoint development, fixed audit and final inspection tasks. Tiny peptide payloads
# exercise the software contract; they are not evidence of cross-protein efficacy.
TASKS = (
    ("dev-short", "development", "MNDTEAI"),
    ("audit-distinct", "audit", "MALWMRLLPLLALLALWGPDPAAA"),
    ("final-distinct", "final", "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATY"),
)
PROTOCOL = {"version": "computational-audit-v1", "tasks": TASKS, "candidates": 6, "shortlist": 3,
            "seed": 2718, "tool_calls": 16, "seconds": 900, "primary": "valid_fraction", "secondary": ["diversity", "unique_count"],
            "promotion": "No task regresses valid fraction; equal budgets and tasks; >=0.01 mean diversity gain or >=1 extra unique candidate. No biological-efficacy claim."}
PROTOCOL_HASH = identity(PROTOCOL)


def summary(run):
    p = run["payload"]
    candidates = p.get("candidates", [])
    valid = [e for e in p.get("evaluations", []) if e["status"] != "rejected"]
    sequences = [c["sequence"] for c in candidates]
    distances = [1-SequenceMatcher(None,a,b,autojunk=False).ratio() for i,a in enumerate(sequences) for b in sequences[i+1:]]
    return {"run_id": run["id"], "state": run["state"], "valid_fraction": len(valid)/len(candidates) if candidates else 0,
            "unique_count": len(set(sequences)), "diversity": sum(distances)/len(distances) if distances else 0,
            "seconds": p.get("elapsed_seconds"), "tool_calls": p.get("tool_calls"), "experimental_successes": 0}


def benchmark(store, versions=("v1", "v2"), splits=("development", "audit", "final"), progress=None):
    result = {"protocol": PROTOCOL, "protocol_hash": PROTOCOL_HASH, "rows": [], "scope": "Software/computational audit only; no experimental biological validation"}
    for task, split, protein in TASKS:
        if split not in splits: continue
        for version in versions:
            brief = Brief(name=f"{task}-{version}", protein=protein, context=Context(target_cell="HEK293"),
                          budget=Budget(candidates=6,shortlist=3,tool_calls=16,seconds=900),
                          objectives=[Objective(metric="start_accessibility",stage="computational")], evaluators=["vienna"], generators=["gemorna","lineardesign"],
                          seed=2718, workflow_version=version)
            run = Engine(store).run(brief, planner=False)
            row = {"task": task, "split": split, "version": version, "budget": brief.budget.model_dump(), "protein_sha256": identity(protein), **summary(run)}
            result["rows"].append(row)
            if progress: progress(row)
    result["sha256"] = store.artifact(result)
    return result


def decision(result):
    if result.get("protocol_hash") != PROTOCOL_HASH or identity(result.get("protocol")) != PROTOCOL_HASH:
        raise ValueError("Sealed benchmark protocol was changed")
    expected = {(t, v) for t, _, _ in TASKS for v in ("v1","v2")}
    rows = result["rows"]
    if len(rows)!=len(expected) or {(r["task"],r["version"]) for r in rows}!=expected:
        raise ValueError("Need complete development, audit and untouched final comparison")
    regressions, observed_regressions = [], []
    pairs = []
    for task, split, protein in TASKS:
        a=next(r for r in rows if r["task"]==task and r["version"]=="v1")
        b=next(r for r in rows if r["task"]==task and r["version"]=="v2")
        budget=Budget(candidates=6,shortlist=3,tool_calls=16,seconds=900).model_dump()
        if a["budget"] != budget or b["budget"] != budget or a["protein_sha256"] != identity(protein) or b["protein_sha256"] != identity(protein) or a["split"]!=split or b["split"]!=split:
            raise ValueError("Task or budget changed during evolution comparison")
        if a["state"]!="complete" or b["state"]!="complete" or b["valid_fraction"] < a["valid_fraction"] or b["valid_fraction"] < 1:
            regressions.append(task)
        pairs.append((a,b))
        for metric in ("valid_fraction","diversity","unique_count"):
            if b[metric] < a[metric]:observed_regressions.append({"task":task,"metric":metric,"delta":b[metric]-a[metric]})
    diversity_gain=sum(b["diversity"]-a["diversity"] for a,b in pairs)/len(pairs)
    count_gain=sum(b["unique_count"]-a["unique_count"] for a,b in pairs)/len(pairs)
    promote=not regressions and (diversity_gain>=0.01 or count_gain>=1)
    return {"promote": promote, "regressions": regressions, "observed_regressions":observed_regressions, "mean_diversity_gain": diversity_gain, "mean_unique_gain": count_gain,
            "reason": "Fixed-budget computational gate passed" if promote else "No demonstrated gain or regression; keep v1",
            "experimental_gain": "not_evaluated", "protocol_hash": PROTOCOL_HASH}


def evolve(store, progress=None):
    # One fixed final audit per candidate/protocol. Re-running returns its original result.
    key = identity({"candidate": "v2", "baseline": "v1", "protocol": PROTOCOL_HASH})
    old = store.db.execute("SELECT body FROM evolution WHERE id=?", (key,)).fetchone()
    if old: return json.loads(old[0])
    report = benchmark(store, progress=progress)
    verdict = decision(report)
    result = {"id": key, "problem": "Candidate selection may concentrate in one mechanism", "change": "v2 sorts single-region hypotheses before factorial combinations", "baseline": "v1", "candidate": "v2", "benchmark": report, "decision": verdict, "created": time.time()}
    with store.db:
        store.db.execute("INSERT INTO evolution VALUES(?,?,?)", (key,canonical(result),time.time()))
        if verdict["promote"]:
            store.db.execute("INSERT OR REPLACE INTO settings VALUES('active_workflow','v2')")
    return result


def rollback(store):
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO settings VALUES('active_workflow','v1')")
    return {"active_workflow": "v1", "history_preserved": True}


def history(store):
    active = store.db.execute("SELECT value FROM settings WHERE key='active_workflow'").fetchone()
    return {"active_workflow": active[0] if active else "v1", "protocol": PROTOCOL, "protocol_hash": PROTOCOL_HASH,
            "cycles": [json.loads(r[0]) for r in store.db.execute("SELECT body FROM evolution ORDER BY created DESC")]}
