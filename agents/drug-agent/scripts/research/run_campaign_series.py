#!/usr/bin/env python
"""Run one stage of the self-evolution campaign series, at a bounded concurrency.

A thin, deliberately dumb driver over ``run_campaign.py``. It launches each
campaign as its own subprocess, keeps at most ``--max-concurrent`` of them alive,
and records what happened. It makes no scientific decision of any kind:

* it does **not** retry a campaign. A campaign that fails has produced a result;
  re-running it until it succeeds is how a series stops being a sample.
* it does **not** stop early on a good result, and does not skip a campaign
  because an earlier one failed. The stage's campaign list is fixed before the
  stage starts and every campaign in it runs.
* it does **not** evaluate the stage gate. That is an operator decision made on
  the recorded outcomes, deliberately outside this script, so that a gate cannot
  be silently satisfied by the thing it is gating.

Usage::

    PYTHONPATH=src conda run --no-capture-output -n drug-agent python \\
        scripts/research/run_campaign_series.py \\
        --stage stage-1 --max-concurrent 2 --max-cycles 2 \\
        --campaigns configs/research/zak-c9r/campaigns/selfevo_s1_[ab].json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # pragma: no cover - operator convenience
    sys.path.insert(0, str(REPO / "src"))
OUTPUT_ROOT = REPO / ".external" / "research" / "zak-c9r" / "derived" / "campaigns"

from drug_agent.selfevolve.backends.claude_cli_backend import EFFORT_LEVELS  # noqa: E402


def _dir_name(campaign_id: str) -> str:
    """``camp_zak_c9r_selfevo_s1_a`` -> ``selfevo-s1-a``."""
    return campaign_id.removeprefix("camp_zak_c9r_").replace("_", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns", nargs="+", required=True, type=Path)
    parser.add_argument("--stage", required=True, help="name recorded in the series ledger")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, required=True)
    parser.add_argument("--proposer", choices=("claude", "random"), default="claude")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model", default=None)
    # Effort is forwarded, unlike --worker-model, because it is not a fallback: a
    # stage whose campaigns reasoned at different effort levels is not one stage,
    # and the level is not recoverable from the outcome afterwards.
    parser.add_argument(
        "--effort",
        default=None,
        choices=EFFORT_LEVELS,
        help="reasoning effort for every PROPOSER in the stage",
    )
    parser.add_argument(
        "--worker-effort",
        default=None,
        choices=EFFORT_LEVELS,
        help="reasoning effort for every experiment WORKER in the stage (default: --effort)",
    )
    # Forwarded, and forwarded to EVERY campaign in the stage without exception: the
    # notes a proposer is shown are part of the arm, so a stage where treatment saw
    # the lessons and control did not is not one stage.
    # Forwarded because a proposer outage is not evenly distributed across arms: the
    # random-menu control has no model in the loop and cannot suffer it, so leaving the
    # bound at its default removes treatment campaigns only.
    parser.add_argument("--max-proposer-retries", type=int, default=None)
    parser.add_argument(
        "--lessons-file",
        default=None,
        type=Path,
        help="prior-round conclusions handed to every proposer in the stage",
    )
    parser.add_argument(
        "--allow",
        nargs="+",
        default=None,
        help="narrow the action menu for every campaign in the stage. It can only ever remove "
        "an intervention, never add one, and the SAME narrowing is applied to treatment and "
        "control alike -- a control with a different menu is not a control",
    )
    parser.add_argument(
        "--max-generated",
        type=int,
        default=None,
        help="ENFORCED ceiling on the molecules an experiment may GENERATE, applied to every "
        "campaign in the stage. Like --allow it narrows one way and never widens, and the SAME "
        "ceiling reaches treatment and control alike. It is not the submission budget -- that "
        "stays at the registered 150 -- and it is not a cost control: the measured cost tracks "
        "molecules SCORED, not molecules generated. It is a control on selection pressure, and "
        "it binds change_sampling only; the worker-authored actions are restricted with --allow",
    )
    parser.add_argument("--timeout-s", type=float, default=None, help="per-dispatch wall budget")
    parser.add_argument(
        "--claude-timeout-s",
        type=float,
        default=None,
        help="per-CLI-call timeout for the EXPERIMENT WORKER. It is NOT the proposer's timeout "
        "-- run_campaign.py routes it into dispatch_kwargs['claude_timeout_s'] -- and an "
        "operator who raises this believing they have given the proposer more wall time has "
        "not. Use --proposer-timeout-s for that",
    )
    parser.add_argument(
        "--proposer-timeout-s",
        type=float,
        default=None,
        help="per-CLI-call timeout for the PROPOSER. Previously unreachable from a series run: "
        "run_campaign.py accepted it, this driver never forwarded it, and every stage of the "
        "26-campaign series therefore ran the proposer at its 600 s default. One campaign died "
        "on exactly that timeout with zero StructuredOutput attempts recorded",
    )
    parser.add_argument("--worker-max-output-tokens", type=int, default=None)
    # An INTERFACE CAP: the per-call output ceiling the child passes to
    # CLAUDE_CODE_MAX_OUTPUT_TOKENS. It is not a scientific threshold and not
    # the campaign's llm_output_tokens budget, which is declared on the ledger
    # and is unchanged by this flag. It is defaulted up because the measurement
    # is unambiguous: every max_tokens truncation in the series happened in a
    # stage running the proposer at 16000 and stages at 32000 had none, and a
    # truncated response is lost entirely while the campaign is still charged.
    # Pass the flag explicitly to run a stage at any other ceiling.
    parser.add_argument("--proposer-max-output-tokens", type=int, default=32000)
    parser.add_argument(
        "--per-cycle-output-tokens",
        type=float,
        default=None,
        help="the token headroom a cycle must have BEFORE it spends a reasoning call. Leaving "
        "this unset disarms the llm_output_tokens arm of the controller's own "
        "_resource_shortfall() guard, which is how a campaign walks into an unpayable "
        "reservation and raises BudgetExceeded as a crash instead of stopping cleanly",
    )
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    # The stage NAME is recorded verbatim in the ledger's contents; only the FILENAME is
    # made path-safe. A stage called "round/stage#1" would otherwise resolve to a
    # `series-round/` subdirectory that nothing creates, and the run dies on the ledger
    # write AFTER its campaigns have already been launched -- which is how a crash here
    # orphans real work rather than preventing it.
    safe_stage = re.sub(r"[^A-Za-z0-9._-]+", "-", str(args.stage)).strip("-") or "stage"
    ledger_path = args.output_root / f"series-{safe_stage}.json"

    queue = list(args.campaigns)
    running: list[dict] = []
    done: list[dict] = []
    started_at = time.time()

    def _launch(path: Path) -> dict:
        campaign = json.loads(path.read_text(encoding="utf-8"))
        out = args.output_root / _dir_name(str(campaign["campaign_id"]))
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "run.log").open("w", encoding="utf-8")
        argv = [
            sys.executable,
            str(REPO / "scripts" / "research" / "run_campaign.py"),
            "--campaign",
            str(path),
            "--output",
            str(out),
            "--max-cycles",
            str(args.max_cycles),
            "--min-improvement",
            "4.0",
            "--proposer",
            args.proposer,
        ]
        if args.allow:
            argv += ["--allow", *args.allow]
        if args.per_cycle_output_tokens is not None:
            argv += ["--per-cycle-output-tokens", str(args.per_cycle_output_tokens)]
        # Every flag this stage narrows with has to appear HERE as well as in the
        # parser above. A flag added to run_campaign.py and not enumerated in this
        # vector is unreachable from a series run: the child is launched with
        # exactly these arguments and nothing inherits.
        #
        # run_campaign.py flags this driver deliberately does NOT forward, so the
        # next reader does not have to diff two parsers to find out (audited
        # against run_campaign.py's parser):
        #   --ledger                      child default <output>/budget.sqlite is
        #                                 what a series wants; one ledger path
        #                                 shared by concurrent campaigns would be
        #                                 an accounting collision
        #   --stop-on-first-improvement   a stage runs its declared cycles; stopping
        #                                 early on a good result is a scientific
        #                                 decision this driver does not make
        #   --max-infrastructure-retries  child default 1
        #   --worker-model                child falls back to --model
        # (--effort and --worker-effort ARE forwarded, below, beside --model.)
        #   --no-probe                    a stage probes; a stage that used the
        #                                 committed capability record for some
        #                                 campaigns and not others is not one stage
        #   --budget-note                 per-run operator justification, not a
        #                                 stage-wide setting
        # --min-improvement is pinned to 4.0 below rather than exposed: it is the
        # registered bar and a stage must not be able to move it.
        for flag, value in (
            ("--max-generated", args.max_generated),
            ("--timeout-s", args.timeout_s),
            ("--claude-timeout-s", args.claude_timeout_s),
            ("--proposer-timeout-s", args.proposer_timeout_s),
            ("--worker-max-output-tokens", args.worker_max_output_tokens),
            ("--proposer-max-output-tokens", args.proposer_max_output_tokens),
        ):
            if value is not None:
                argv += [flag, str(value)]
        if args.model:
            argv += ["--model", args.model]
        if args.effort:
            argv += ["--effort", args.effort]
        if args.worker_effort:
            argv += ["--worker-effort", args.worker_effort]
        if args.max_proposer_retries is not None:
            argv += ["--max-proposer-retries", str(args.max_proposer_retries)]
        if args.lessons_file:
            argv += ["--lessons-file", str(args.lessons_file)]
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env)
        print(f"launched {campaign['campaign_id']} (seed {campaign['seed']}) -> {out}", flush=True)
        return {
            "campaign_id": str(campaign["campaign_id"]),
            "seed": campaign["seed"],
            "record": str(path),
            "output": str(out),
            "log": str(out / "run.log"),
            "pid": proc.pid,
            "proc": proc,
            "started_at": time.time(),
        }

    def _write_ledger() -> None:
        ledger_path.write_text(
            json.dumps(
                {
                    "stage": args.stage,
                    "proposer": args.proposer,
                    "max_cycles": args.max_cycles,
                    "max_concurrent": args.max_concurrent,
                    "min_improvement": 4.0,
                    "allowed_actions": list(args.allow) if args.allow else "ALL",
                    "max_generated": args.max_generated,
                    "per_cycle_output_tokens": args.per_cycle_output_tokens,
                    "timeout_s": args.timeout_s,
                    # The worker's per-call timeout and the proposer's are two
                    # different numbers and are recorded as two different keys.
                    # Neither proposer number was in the provenance record
                    # before, which is why a stage that ran the proposer at
                    # 16000/600s and a stage that ran it at 32000 were
                    # indistinguishable after the fact.
                    "claude_timeout_s": args.claude_timeout_s,
                    "proposer_timeout_s": args.proposer_timeout_s,
                    "proposer_max_output_tokens": args.proposer_max_output_tokens,
                    "worker_max_output_tokens": args.worker_max_output_tokens,
                    "elapsed_seconds": round(time.time() - started_at, 1),
                    "n_queued": len(queue),
                    "n_running": len(running),
                    "n_done": len(done),
                    "campaigns": [{k: v for k, v in e.items() if k != "proc"} for e in done + running],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _terminate(signum, _frame):  # pragma: no cover - operator interrupt
        print(f"\nsignal {signum}: terminating {len(running)} running campaign(s)", flush=True)
        for entry in running:
            entry["proc"].terminate()
        _write_ledger()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    while queue or running:
        while queue and len(running) < args.max_concurrent:
            running.append(_launch(queue.pop(0)))
            _write_ledger()
        time.sleep(10)
        for entry in list(running):
            code = entry["proc"].poll()
            if code is None:
                continue
            running.remove(entry)
            entry["exit_code"] = int(code)
            entry["finished_at"] = time.time()
            entry["wall_seconds"] = round(entry["finished_at"] - entry["started_at"], 1)
            entry.pop("proc")
            done.append(entry)
            print(f"finished {entry['campaign_id']}: exit {code} after {entry['wall_seconds']}s", flush=True)
            _write_ledger()

    _write_ledger()
    print(f"\nstage {args.stage}: {len(done)} campaign(s), ledger {ledger_path}")
    for entry in done:
        print(f"  {entry['campaign_id']:34s} exit={entry['exit_code']} {entry['wall_seconds']}s")
    # A nonzero campaign exit is a scientific outcome, not a driver failure.
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
