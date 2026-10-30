#!/usr/bin/env python
"""Offline deterministic demo: run the sEH brief end-to-end through the gate.

Exercises the full workflow, pauses at the human-approval gate, approves it,
resumes to completion, and prints where the report and bundle landed. No
network, no credentials, CPU-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

from drug_agent.service import build_engine, load_brief
from drug_agent.settings.config import load_settings

BRIEF = sys.argv[1] if len(sys.argv) > 1 else "fixtures/briefs/target_seh.yaml"


def main() -> int:
    settings = load_settings("local", deterministic=True).resolved()
    engine = build_engine(settings)
    engine.history_source = "demo"  # the running history records this run like any other
    brief = load_brief(BRIEF)
    run = engine.create_run(brief, deterministic=True)
    print(f"[demo] created run {run.id} ({len(run.task_ids)} tasks)")

    final = engine.run(run.id)
    print(f"[demo] paused at gate: state={final.state}")
    if str(final.state) != "AWAITING_APPROVAL":
        print("[demo] WARNING: expected approval pause")

    engine.approve_gate(run.id, "portfolio_review", by="demo")
    final = engine.resume(run.id)
    print(f"[demo] final state: {final.state}")

    bundle = Path(settings.runs_dir) / run.id
    if final.state.value != "SUCCEEDED":
        print("[demo] FAILED: run did not succeed")
        return 1
    portfolio = engine.store.list_objects("portfolio", run_id=run.id)[0]
    print(
        f"[demo] shortlist: {len(portfolio['shortlist'])} candidates, "
        f"pareto={portfolio['n_pareto']}, diversity={portfolio['diversity']}"
    )
    print(f"[demo] report:  {bundle / 'report.md'}")
    print(f"[demo] html:    {bundle / 'report.html'}")
    print(f"[demo] bundle:  {bundle}")
    print(f"[demo] manifest:{bundle / 'manifest.json'}")
    print("[demo] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
