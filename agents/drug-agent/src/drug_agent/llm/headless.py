#!/usr/bin/env python
"""Headless master-agent runner — drive a full program from the terminal.

Prints the event stream as it happens, which is what you want when comparing the
adaptive and naive refinement strategies without a browser::

    python -m drug_agent.llm.headless adaptive 3 "Find potent, drug-like sEH inhibitors…"
    python -m drug_agent.llm.headless naive    3 "…same instruction…"
    python -m drug_agent.llm.headless adaptive 3 "…"                  # master agent picks the source
    python -m drug_agent.llm.headless adaptive 3 "…" library+pfm 60   # pin de-novo generation

Compare the two scoreboards: naive stays flat (the backend is deterministic),
adaptive climbs as the master agent re-prompts generation. Real PFM sampling is
stochastic, so that flat naive baseline only holds for the library and surrogate
backends.
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from ..agents.llm.master_agent import MasterAgent
from .config import load_config
from .events import RunSession

DEFAULT_INSTRUCTION = (
    "Find selective, potent, drug-like sEH (EPHX2) inhibitors with novel scaffolds and no structural alerts."
)


async def run(
    instruction: str,
    *,
    strategy: str = "adaptive",
    cycles: int = 3,
    generator: str | None = None,
    n_generated: int | None = None,
    run_id: str | None = None,
) -> int:
    """Drive one program, printing progress. Returns a process exit code."""
    cfg = load_config()
    # A fresh id per run: journals are keyed by it, and a fixed id would have
    # each terminal run quietly overwrite the record of the last one.
    run_id = run_id or "mrun_" + uuid.uuid4().hex[:12]
    session = RunSession(run_id)
    agent = MasterAgent(cfg)
    journal = agent.open_journal(
        session,
        instruction,
        source="cli",
        model=cfg.model,
        strategy=strategy,
        generator=generator or cfg.generator,
        # Left unset when the caller said nothing: the master agent picks it, and
        # transcription writes the real value back onto the journal.
        n_generated=n_generated,
        max_cycles=cycles,
    )
    task = asyncio.create_task(
        agent.run(
            session,
            instruction,
            strategy=strategy,
            max_cycles=cycles,
            generator=generator,
            n_generated=n_generated,
            source="cli",
            journal=journal,
        )
    )
    print(f"=== request {run_id}")
    failed = False
    think: dict[str, int] = {}
    async for event in session.stream():
        kind, data = event.type, event.data
        if kind in ("thinking", "content"):
            key = f"{event.phase}#{event.cycle}"
            think[key] = think.get(key, 0) + len(data.get("delta", ""))
            continue
        if kind == "run_started":
            print(f"=== source={data.get('generator')} (n_generated={data.get('n_generated')})")
        elif kind == "phase_started":
            print(f"\n=== {event.phase} (cycle {event.cycle}): {event.title}")
        elif kind == "plan_ready":
            print("  PLAN:", [f"{c['metric']} {c['target']}" for c in data["criteria"]])
            if data.get("unscorable"):
                print("  DROPPED (unmeasurable):", [c["metric"] for c in data["unscorable"]])
        elif kind == "requirements_ready":
            print("  REQ:", data["summary"])
        elif kind == "cycle_started":
            print(f"\n----- CYCLE {event.cycle} ({data.get('strategy')}) -----")
        elif kind == "chemical_result":
            print("  CHEM:", data["summary"])
        elif kind == "evaluation_result":
            report = data["report"]
            print(f"  EVAL: {report['n_criteria_met']}/{report['n_criteria_total']} met")
            for crit in report["criteria"]:
                mark = "MET " if crit["met"] else "MISS"
                print(f"       [{mark}] {crit['key']}={crit['value']} target={crit['target']}")
        elif kind == "critique_ready":
            crit = data["critique"]
            print(f"  CRITIQUE: {crit['overall_verdict']} — {(crit['diagnosis'] or '')[:160]}")
        elif kind == "decision":
            print(f"  DECISION: {data['verdict']} (final={data['is_final']})")
        elif kind == "requirements_adjusted":
            print(f"  ADJUSTED [{data.get('strategy')}]: {len(data.get('changes', []))} changes")
            for change in data.get("changes", []):
                print(f"       • {change['parameter']}: {change['old_value']} -> {change['new_value']}")
        elif kind == "log":
            print(f"  · {event.title}: {str(data.get('detail', ''))[:200]}")
        elif kind == "run_succeeded":
            print("\n### SCOREBOARD:")
            for point in data["result"]["scoreboard"]:
                print("   ", point)
        elif kind == "run_failed":
            failed = True
            print("\n### FAILED:", data.get("error"))
        elif kind == "done":
            break
    await task
    print("\nthinking chars/phase:", think)
    if journal is not None:
        print(f"journal: {journal.dir}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    strategy = argv[0] if len(argv) > 0 else "adaptive"
    cycles = int(argv[1]) if len(argv) > 1 else 3
    instruction = argv[2] if len(argv) > 2 else DEFAULT_INSTRUCTION
    generator = argv[3] if len(argv) > 3 else None
    n_generated = int(argv[4]) if len(argv) > 4 else None
    return asyncio.run(
        run(
            instruction,
            strategy=strategy,
            cycles=cycles,
            generator=generator,
            n_generated=n_generated,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
