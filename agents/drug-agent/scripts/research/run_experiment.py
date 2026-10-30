#!/usr/bin/env python
"""Dispatch one structured experiment plan, under the campaign's declared caps.

Step 10 of the per-target evolution program. This is the operator-facing entry
point to :func:`drug_agent.selfevolve.target.worker.dispatch_plan`: it reads a
plan, opens the campaign's budget ledger with the caps declared in
``configs/research/zak-c9r.yaml``, dispatches the plan, and writes the whole
dispatch envelope — validation, capability report, reservations, execution
detail, artifact provenance and the independent development evaluation — next
to the run's output.

Nothing here decides whether an experiment succeeded. The worker's own
``result.json`` is testimony; the number that counts is computed by the frozen
Step 7 evaluator from the files the run wrote, in the dispatcher's process,
after the run has exited.

Exit status is the ``ExperimentResult``'s own ``exit_code``, which comes from
``drug_agent.cli.main.run_exit_code`` and from no second copy of that table. A
capability that does not exist here and a budget that refuses to launch are both
*results* and both exit non-zero with a structured record, never silently.

Usage::

    conda run --no-capture-output -n drug-agent python \\
        scripts/research/run_experiment.py \\
        --plan configs/research/zak-c9r/experiments/E01.json \\
        --output .external/research/zak-c9r/derived/experiments/E01

``--allow`` narrows the campaign's action menu for this dispatch; it can only
ever remove interventions, never add one the dispatcher does not implement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # pragma: no cover - operator convenience
    sys.path.insert(0, str(REPO / "src"))

from drug_agent.selfevolve.target.budget import BudgetLedger  # noqa: E402
from drug_agent.selfevolve.target.controller import campaign_caps  # noqa: E402
from drug_agent.selfevolve.target.worker import SUPPORTED_ACTIONS, dispatch_plan  # noqa: E402

CONFIG = REPO / "configs" / "research" / "zak-c9r.yaml"


def _campaign_caps() -> dict[str, float]:
    """The campaign's declared caps, as ledger resources.

    The mapping lives in ``target.controller.CAP_FACTORS`` — Step 11 moved it
    there so the campaign loop and a single hand-dispatched experiment enforce
    the *same* table. Two copies is how a campaign ends up enforcing a different
    cap from the experiment running inside it.
    """
    import yaml

    budget = yaml.safe_load(CONFIG.read_text(encoding="utf-8")).get("budget") or {}
    return campaign_caps(budget)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ledger", default=None, help="budget ledger path (default: <output>/budget.sqlite)")
    parser.add_argument(
        "--allow",
        nargs="*",
        default=None,
        help=f"narrow the action menu for this dispatch (default: all of {list(SUPPORTED_ACTIONS)})",
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--claude-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--model", default=None, help="passed to the CLI's --model; the resolved id is recorded"
    )
    parser.add_argument("--request-key", default=None, help="makes the dispatch idempotent across restarts")
    parser.add_argument("--no-probe", action="store_true", help="use the committed capability record")
    parser.add_argument("--no-ledger", action="store_true", help="run without campaign budget accounting")
    args = parser.parse_args(argv)

    plan: dict[str, Any] = json.loads(args.plan.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ledger = None
    if not args.no_ledger:
        ledger = BudgetLedger(Path(args.ledger) if args.ledger else output / "budget.sqlite")
        for resource, limit in sorted(_campaign_caps().items()):
            ledger.declare(resource, limit, note="declared in configs/research/zak-c9r.yaml")

    envelope = dispatch_plan(
        plan,
        allowed_actions=list(args.allow) if args.allow else list(SUPPORTED_ACTIONS),
        output_dir=output,
        log_dir=output / "logs",
        ledger=ledger,
        probe=not args.no_probe,
        timeout_s=args.timeout_s,
        claude_timeout_s=args.claude_timeout_s,
        claude_model=args.model,
        max_output_tokens=args.max_output_tokens,
        request_key=args.request_key,
    )
    result = envelope["result"]
    evaluation = envelope["independent_evaluation"]
    print(f"experiment {result['experiment_id']}: {result['status']} ({result['failure_category']})")
    if envelope.get("reason"):
        print(f"  reason: {envelope['reason']}")
    print(f"  capability: {envelope['support']['support']} ({envelope['support']['evidence_class']})")
    print(f"  artifacts: {result['artifacts'] or 'none'}")
    print(f"  independent {evaluation['metric']}: {evaluation['status']} value={evaluation.get('value')}")
    print(f"  cost: {result['actual_cost']}")
    if ledger is not None:
        ledger.write_report(output / "ledger.json")
        print(f"  ledger report: {output / 'ledger.json'}")
    envelope_path = output / "logs" / f"dispatch-{result['experiment_id']}.json"
    print(f"  envelope: {envelope_path}")
    return int(result["exit_code"])


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
