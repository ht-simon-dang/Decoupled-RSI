#!/usr/bin/env python
"""Run a multi-stage campaign round to completion, unattended, inside one envelope.

``run_campaign_series.py`` runs ONE stage at a bounded concurrency. A round is
several stages in a declared order, because some campaigns build the data or the
evidence that later ones consume. Chaining them by hand over four days is how a
stage gets launched against a dependency that never finished.

WHAT THIS DRIVER DECIDES, AND WHAT IT REFUSES TO DECIDE
------------------------------------------------------
It makes **no scientific decision**, deliberately and for the same reason
``run_campaign_series.py`` does not: a stop-gate that the automation evaluates is a
gate the thing being gated can satisfy. So this driver never reads a development
metric, never compares a campaign against its incumbent, never skips a stage
because an earlier result looked weak, and never retries a campaign. A campaign
that failed has produced a result.

It enforces exactly two **mechanical** preconditions, both of which are about
whether a stage *can* run rather than whether it *should*:

``TERMINAL``
    Every campaign in a prior stage that this stage declares a dependency on must
    have written a ``campaign.json``. That file is where the controller records the
    terminal state, so its absence means the campaign is still running or died
    without recording anything. Launching a dependent stage against a dependency
    that has not finished is not a gate judgement -- the input simply is not there.

``ENVELOPE``
    The round's declared ``gpu_seconds`` envelope is re-checked from the campaigns'
    own budget ledgers between stages. The per-campaign cap is enforced by Step 9's
    ledger; nothing enforces the SUM, and on a single-device host the sum is the
    whole plan. A round that has already spent its envelope stops with
    ``ENVELOPE_EXHAUSTED`` rather than starting a stage it cannot pay for.

Everything it observes goes into ``<output-root>/round-<name>.json``, rewritten
after every stage so an operator reading it mid-round sees the truth so far.

Stage plan format::

    {
      "round": "neg-v1",
      "model": "claude-opus-5-5",
      "effort": "max",
      "gpu_envelope_seconds": 360000,
      "stages": [
        {
          "name": "stage-A-foundation",
          "max_concurrent": 2,
          "max_cycles": 3,
          "depends_on": [],
          "campaigns": ["neg_a_rejected_corpus", "neg_a_pairing"]
        }
      ]
    }

``campaigns`` are record stems under ``--record-dir``. ``depends_on`` names earlier
stages by name.

Usage::

    PYTHONPATH=src conda run --no-capture-output -n drug-agent python \\
        scripts/research/run_round.py --plan configs/research/zak-c9r/rounds/neg-v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # pragma: no cover - operator convenience
    sys.path.insert(0, str(REPO / "src"))

SERIES = REPO / "scripts" / "research" / "run_campaign_series.py"
DEFAULT_RECORD_DIR = REPO / "configs" / "research" / "zak-c9r" / "campaigns"
DEFAULT_OUTPUT_ROOT = REPO / ".external" / "research" / "zak-c9r" / "derived" / "campaigns"


#: Stage keys a single campaign may override. Everything else is a property of the stage,
#: and a campaign that could override `max_cycles` per arm would make "one stage" meaningless
#: as a unit of comparison -- so that one is here deliberately and the rest are not.
#: Seconds between reaping the rolling pool. Long on purpose: a campaign cycle here is tens of
#: minutes, so a tighter poll buys nothing and only adds wakeups to a run that lasts days.
POLL_SECONDS = 20.0

PER_CAMPAIGN_OVERRIDES = frozenset(
    {"allow", "max_generated", "proposer", "per_cycle_output_tokens", "worker_effort"}
)


def _effective_flags(stage: dict[str, Any], over: dict[str, Any], effort: str | None) -> dict[str, Any]:
    """The flags one campaign actually runs under: its own override, else the stage's.

    Module level rather than a closure so the stage it reads is the one passed in. A closure
    over the stage loop variable would resolve late, which is the kind of bug that produces a
    round where every campaign ran under the last stage's menu.
    """
    return {
        "proposer": over.get("proposer", stage.get("proposer")),
        "allow": tuple(over.get("allow", stage.get("allow")) or ()) or None,
        "max_generated": over.get("max_generated", stage.get("max_generated")),
        "per_cycle_output_tokens": over.get("per_cycle_output_tokens", stage.get("per_cycle_output_tokens")),
        "worker_effort": over.get("worker_effort", stage.get("worker_effort") or effort),
    }


def _dir_name(campaign_id: str) -> str:
    """``camp_zak_c9r_neg_a_foo`` -> ``neg-a-foo``, matching the series driver."""
    return campaign_id.removeprefix("camp_zak_c9r_").replace("_", "-")


def _campaign_id(record: Path) -> str:
    return str(json.loads(record.read_text(encoding="utf-8"))["campaign_id"])


def _terminal_state(output_root: Path, campaign_id: str) -> str | None:
    """The state the controller recorded, or ``None`` if it recorded nothing yet.

    Read from ``campaign.json`` under the campaign's own output directory, because
    that is the only place the terminal state is written -- the config record stays
    ``RUNNING`` for the life of the project by design.
    """
    path = output_root / _dir_name(campaign_id) / "campaign.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("state") or "") or None
    except (json.JSONDecodeError, OSError):
        return None


def _gpu_spent(output_root: Path, campaign_id: str) -> float:
    """GPU-seconds this campaign's ledger has CHARGED, or 0.0 if it has no ledger.

    Read from ``reservations.consumed`` where ``resource = 'gpu_seconds'``. Two things
    about that column decide whether this number can bound a round, and both were
    checked against the ledgers on this host rather than assumed:

    *It is a charge, not a measurement.* ``costs.amount`` for ``gpu_seconds`` is NULL in
    every campaign ledger here, because GPU telemetry is UNAVAILABLE -- the same finding
    the program records as B-P2.4, that GPU figures written before commit cdc04346 are
    lower bounds. With no telemetry the ledger settles the reservation at its full
    reserved amount: every campaign checked shows ``consumed == reserved`` exactly
    (7,200/7,200, 21,600/21,600, 18,200/18,200). So this over-charges rather than
    under-charges, which is the only safe direction for a cap.

    *A campaign with no ledger has charged nothing measurable*, reported as 0.0 rather
    than as unknown. An envelope built from a floor stops late rather than early, so
    the round's own per-campaign caps remain the binding constraint and this is the
    backstop across them.
    """
    ledger = output_root / _dir_name(campaign_id) / "budget.sqlite"
    if not ledger.exists():
        return 0.0
    try:
        # A SHORT busy timeout, because the pool polls ledgers that live campaigns are writing.
        # At sqlite's 5-second default, one poll over ten in-flight campaigns costs most of a
        # minute of blocking, which is how a dry run over a running round came to look like a
        # hang. A locked ledger reads as 0.0; under-reporting a spend is the safe direction for
        # a cap and the next poll corrects it.
        con = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=0.5)
    except sqlite3.Error:
        return 0.0
    try:
        row = con.execute("SELECT SUM(consumed) FROM reservations WHERE resource = 'gpu_seconds'").fetchone()
    except sqlite3.Error:
        return 0.0
    finally:
        con.close()
    return float(row[0]) if row and row[0] is not None else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, type=Path, help="the stage plan JSON")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve every record, print the command each stage would run, launch nothing",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="campaigns in flight at once, as a ROLLING pool rather than a batch: a replacement "
        "launches the moment one finishes. The plan's own `max_concurrent` wins if both are "
        "given. Justified by measurement rather than optimism -- the wall-to-GPU ratio here is "
        "10.27, so one campaign leaves the device idle about nine tenths of the time, and vina's "
        "search depends on its own --cpu argument rather than on OS scheduling, so campaigns "
        "running together return the same scores they would alone",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        help="resume: skip stages before this one. The dependency check still applies, so a "
        "resume cannot skip past an unfinished dependency",
    )
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    round_name = str(plan["round"])
    stages: list[dict[str, Any]] = list(plan["stages"])
    envelope = plan.get("gpu_envelope_seconds")
    model = plan.get("model")
    effort = plan.get("effort")
    lessons_file = plan.get("lessons_file")
    # One subdirectory per campaign under this root, holding the full text of every model call
    # that campaign made. Separate from the campaign's own output tree on purpose: that tree is
    # the study record and is addressed by hashes, while this is an operator's archive of what
    # was said, and the two should not be confused for one another.
    transcript_root = plan.get("transcript_root")
    max_proposer_retries = plan.get("max_proposer_retries")
    # Two different clocks, and confusing them is what killed the first two campaigns of
    # this round. `claude_timeout_s` bounds the WORKER's code-writing session; at maximum
    # reasoning effort the default 900 s is not enough and the dispatch dies at 901 s with
    # INFRASTRUCTURE_ERROR. `timeout_s` bounds the EXPERIMENT the worker's code then runs,
    # which for a multi-replicate training arm is far longer than either.
    timeout_s = plan.get("timeout_s")
    claude_timeout_s = plan.get("claude_timeout_s")

    # --- resolve every record BEFORE launching anything ---------------------
    # A stage that dies on a missing record on day three has already spent the
    # earlier stages. Every path in the plan is resolved up front.
    problems: list[str] = []
    resolved: dict[str, list[tuple[str, Path]]] = {}
    names = [str(s["name"]) for s in stages]
    if len(set(names)) != len(names):
        problems.append(f"stage names are not unique: {names}")
    for stage in stages:
        pairs: list[tuple[str, Path, dict[str, Any]]] = []
        for entry in stage["campaigns"]:
            # A campaign may be named as a bare stem, or as an object carrying its own
            # narrowing. Both forms exist because a stage's flags are applied by
            # `run_campaign_series.py` to every campaign it launches, so two campaigns that
            # need different menus or different generation ceilings could not otherwise share
            # a stage -- which forced dependency-independent arms apart for no reason.
            if isinstance(entry, str):
                stem, over = entry, {}
            elif isinstance(entry, dict) and entry.get("stem"):
                stem, over = str(entry["stem"]), {k: v for k, v in entry.items() if k != "stem"}
            else:
                problems.append(
                    f"stage {stage['name']}: campaign entry {entry!r} is neither a record stem "
                    f"nor an object with a 'stem' key"
                )
                continue
            unknown = set(over) - PER_CAMPAIGN_OVERRIDES
            if unknown:
                problems.append(
                    f"stage {stage['name']}: campaign {stem} declares {sorted(unknown)}, which "
                    f"this driver does not forward; the overridable keys are "
                    f"{sorted(PER_CAMPAIGN_OVERRIDES)}"
                )
                continue
            record = args.record_dir / f"{stem}.json"
            if not record.exists():
                problems.append(f"stage {stage['name']}: no record at {record}")
                continue
            pairs.append((_campaign_id(record), record, over))
        resolved[str(stage["name"])] = pairs
        for dep in stage.get("depends_on") or []:
            if dep not in names:
                problems.append(f"stage {stage['name']}: depends_on names unknown stage {dep!r}")
            elif names.index(dep) >= names.index(str(stage["name"])):
                problems.append(
                    f"stage {stage['name']}: depends_on {dep!r} which is not an EARLIER stage; "
                    f"the plan's order is the execution order and a later dependency can never "
                    f"be satisfied"
                )
    if args.from_stage is not None and args.from_stage not in names:
        problems.append(f"--from-stage {args.from_stage!r} is not a stage in this plan")
    if lessons_file is not None and not Path(lessons_file).exists():
        # Checked here rather than per stage: a lessons file that goes missing on day
        # three would teach the first half of the round and not the second, and nothing
        # in the outcomes would say which campaigns saw it.
        problems.append(f"lessons_file {lessons_file} does not exist")
    if problems:
        print(f"REFUSED: {len(problems)} problem(s) in the plan; nothing was launched", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    total_campaigns = sum(len(v) for v in resolved.values())
    print(f"round {round_name}: {len(stages)} stages, {total_campaigns} campaigns")
    if envelope:
        print(f"  gpu envelope {float(envelope):,.0f} seconds ({float(envelope) / 3600:.1f} GPU-hours)")
    if model or effort:
        print(f"  proposer model {model or '<cli default>'} at effort {effort or '<cli default>'}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    record_path = args.output_root / f"round-{round_name}.json"
    observed: list[dict[str, Any]] = []

    def _write() -> None:
        record_path.write_text(
            json.dumps(
                {
                    "round": round_name,
                    "plan": str(args.plan),
                    "model": model,
                    "effort": effort,
                    "lessons_file": lessons_file,
                    "transcript_root": transcript_root,
                    "max_proposer_retries": max_proposer_retries,
                    "timeout_s": timeout_s,
                    "claude_timeout_s": claude_timeout_s,
                    "gpu_envelope_seconds": envelope,
                    "stages": observed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    started_round = time.time()

    # --- one rolling pool across the WHOLE round -----------------------------
    # The stage list is a DEPENDENCY order, not an execution schedule. Running it
    # stage-by-stage leaves the machine mostly idle: the measured wall-to-GPU ratio on this
    # pipeline is 10.27, so a single campaign holds the device about a tenth of its wall
    # clock and the rest is CPU docking. The operator authorised a rolling pool of N, and
    # rolling rather than batched is the point -- a batch of N waits for its slowest member
    # before any replacement starts, which reintroduces the idleness it was meant to remove.
    #
    # A campaign becomes ELIGIBLE when every campaign of every stage its own stage depends on
    # has written a terminal state. That is the same mechanical precondition the stage loop
    # enforced, applied per campaign instead of per stage, so the dependency semantics do not
    # change -- only how much of the machine is busy while they are satisfied.
    #
    # Concurrency here is score-neutral and that is why it is allowed at all: vina's search
    # depends on its own --cpu argument rather than on OS scheduling, so two campaigns docking
    # at once return the same numbers they would alone. Nothing pins --cpu, deliberately:
    # pinning it would change the frozen scorer's output relative to every prior run.
    pool_size = int(plan.get("max_concurrent") or args.max_concurrent or 1)

    queue: list[dict[str, Any]] = []
    stage_of: dict[str, str] = {}
    for stage in stages:
        name = str(stage["name"])
        if args.from_stage is not None and names.index(name) < names.index(args.from_stage):
            observed.append({"stage": name, "outcome": "SKIPPED_BY_OPERATOR"})
            continue
        for cid, record, over in resolved[name]:
            stage_of[cid] = name
            queue.append({"campaign_id": cid, "record": record, "stage": name, "over": over})

    deps_of = {str(s["name"]): list(s.get("depends_on") or []) for s in stages}
    members_of = {str(s["name"]): [cid for cid, _, _ in resolved[str(s["name"])]] for s in stages}

    def _eligible(entry: dict[str, Any]) -> bool:
        # A dry run launches nothing, so nothing ever reaches a terminal state and a dependency
        # on a later stage can never clear. Without this the pool finds no eligible campaign,
        # falls out of its launch loop, and spins the outer loop at full CPU forever -- which is
        # exactly how this looked the first time: two minutes of user time and six lines of
        # output. Resolving everything as eligible is what makes --dry-run print the whole plan.
        if args.dry_run:
            return True
        for dep in deps_of.get(entry["stage"], []):
            for cid in members_of.get(dep, []):
                if _terminal_state(args.output_root, cid) is None:
                    return False
        return True

    # A campaign that already carries a terminal state is DONE, not restarted. That is what
    # makes a resume a resume: re-running it would re-label a finished history.
    already = [e for e in queue if _terminal_state(args.output_root, e["campaign_id"]) is not None]
    for e in already:
        print(f"already terminal, not relaunched: {e['campaign_id']}")
    queue = [e for e in queue if _terminal_state(args.output_root, e["campaign_id"]) is None]

    print(f"\nrolling pool of {pool_size}; {len(queue)} campaigns to run, {len(already)} already terminal")

    running: list[dict[str, Any]] = []
    launched = 0
    stalled_reported = False
    while queue or running:
        # reap
        still: list[dict[str, Any]] = []
        for job in running:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["log"].close()
            state = _terminal_state(args.output_root, job["campaign_id"])
            elapsed = round(time.time() - job["started"], 1)
            print(f"finished {job['campaign_id']}: exit {rc} after {elapsed}s, state {state}")
            observed.append(
                {
                    "campaign_id": job["campaign_id"],
                    "stage": job["stage"],
                    "outcome": "RAN",
                    "exit_status": rc,
                    "wall_seconds": elapsed,
                    "terminal_state": state,
                    "gpu_seconds_settled": _gpu_spent(args.output_root, job["campaign_id"]),
                }
            )
            _write()
        running = still

        # the round-level envelope, re-checked before each launch rather than per stage
        spent = sum(
            _gpu_spent(args.output_root, cid) for pairs_ in resolved.values() for cid, _, _over in pairs_
        )
        if envelope and spent >= float(envelope):
            print(
                f"STOPPED: the round has settled {spent:,.0f} of its {float(envelope):,.0f} "
                f"gpu_second envelope; {len(queue)} campaigns not launched",
                file=sys.stderr,
            )
            observed.append(
                {
                    "outcome": "ENVELOPE_EXHAUSTED",
                    "gpu_seconds_settled": spent,
                    "not_launched": [e["campaign_id"] for e in queue],
                }
            )
            queue = []

        # launch every eligible campaign the pool has room for
        while queue and len(running) < pool_size:
            nxt = next((e for e in queue if _eligible(e)), None)
            if nxt is None:
                break
            queue.remove(nxt)
            flags = _effective_flags(
                next(s for s in stages if str(s["name"]) == nxt["stage"]), nxt["over"], effort
            )
            stage = next(s for s in stages if str(s["name"]) == nxt["stage"])
            out = args.output_root / _dir_name(nxt["campaign_id"])
            # Created only for a real launch. A dry run that leaves directories behind is not a
            # dry run, and an empty campaign directory later reads as RUNNING_OR_INCOMPLETE to
            # the report, which is a lie about a campaign that never started.
            if not args.dry_run:
                out.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable,
                str(REPO / "scripts" / "research" / "run_campaign.py"),
                "--campaign",
                str(nxt["record"]),
                "--output",
                str(out),
                "--max-cycles",
                str(int(stage["max_cycles"])),
                # The registered bar, pinned here exactly as run_campaign_series.py pins it:
                # a schedule must not be able to move the number a campaign has to beat.
                "--min-improvement",
                "4.0",
            ]
            if flags["proposer"]:
                argv += ["--proposer", str(flags["proposer"])]
            if flags["allow"]:
                argv += ["--allow", *[str(a) for a in flags["allow"]]]
            if flags["per_cycle_output_tokens"] is not None:
                argv += ["--per-cycle-output-tokens", str(flags["per_cycle_output_tokens"])]
            if flags["max_generated"] is not None:
                argv += ["--max-generated", str(int(flags["max_generated"]))]
            if model:
                argv += ["--model", str(model)]
            if effort:
                argv += ["--effort", str(effort)]
            if flags["worker_effort"]:
                argv += ["--worker-effort", str(flags["worker_effort"])]
            if timeout_s is not None:
                argv += ["--timeout-s", str(float(timeout_s))]
            if claude_timeout_s is not None:
                argv += ["--claude-timeout-s", str(float(claude_timeout_s))]
            if max_proposer_retries is not None:
                argv += ["--max-proposer-retries", str(int(max_proposer_retries))]
            if transcript_root:
                argv += [
                    "--transcript-dir",
                    str(Path(transcript_root) / _dir_name(nxt["campaign_id"])),
                ]
            if lessons_file:
                argv += ["--lessons-file", str(lessons_file)]

            if args.dry_run:
                print(f"  would launch [{nxt['stage']}] {nxt['campaign_id']}")
                print("    " + " ".join(argv))
                observed.append(
                    {
                        "campaign_id": nxt["campaign_id"],
                        "stage": nxt["stage"],
                        "outcome": "DRY_RUN",
                        "argv": argv,
                    }
                )
                continue

            log = (out / "run.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=REPO,
                env={**os.environ, "PYTHONPATH": str(REPO / "src")},
            )
            launched += 1
            running.append(
                {
                    "campaign_id": nxt["campaign_id"],
                    "stage": nxt["stage"],
                    "proc": proc,
                    "log": log,
                    "started": time.time(),
                }
            )
            print(f"launched [{nxt['stage']}] {nxt['campaign_id']} ({len(running)}/{pool_size} in flight)")

        if args.dry_run:
            # One pass prints the whole plan, so there is nothing to come back for. An
            # unconditional break also means a dry run can never spin, whatever the
            # eligibility rule does.
            break

        # Nothing running and nothing eligible means every remaining campaign is waiting on a
        # dependency that will never finish. Reported rather than spun on.
        if not running and queue:
            if not stalled_reported:
                blocked = sorted({e["stage"] for e in queue})
                print(
                    f"STOPPED: {len(queue)} campaigns remain and none is eligible; their stages "
                    f"{blocked} depend on campaigns that recorded no terminal state",
                    file=sys.stderr,
                )
                observed.append(
                    {
                        "outcome": "DEPENDENCY_UNFINISHED",
                        "not_launched": [e["campaign_id"] for e in queue],
                        "blocked_stages": blocked,
                    }
                )
                stalled_reported = True
                _write()
            break
        if running:
            time.sleep(POLL_SECONDS)

    _write()
    print(f"\nround {round_name} finished after {round(time.time() - started_round, 1)}s -> {record_path}")
    print(f"  launched {launched}, already terminal {len(already)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
