#!/usr/bin/env python
"""Manifest-driven baseline, capability and PMO execution for the ZAK study.

Step 8 of the per-target evolution program. Three subcommands, all of which run
without an LLM:

``baseline --manifest PATH --output DIR``
    Execute one baseline manifest, a directory of them, or the committed index.
    Each arm is validated, its required capability verbs are probed for real,
    and it is then run as a bounded subprocess: its own process session, a
    wall-clock timeout and a ``RLIMIT_CPU`` ceiling, so a runaway arm cannot
    outlive its budget or leave a GPU held. Results are validated against the
    output schema the manifest promised and collected into ``study.json``
    alongside a provisional cost ledger with the fields Step 9's budget service
    will need.

``capabilities --output DIR``
    Probe every registered ``(capability, verb)`` pair separately and write
    ``capabilities.json``. ``sample``, ``fit``, ``resume``, ``predict`` and
    ``export`` are different questions with different answers; a working sampler
    does not establish a working trainer.

``pilot --manifest PATH --output DIR``
    Assess development-pilot feasibility before the protocol is frozen. Each
    campaign named by the manifest is either **run** (when ``--run-new`` is
    given, it has no terminal record, and the declared budget can actually pay
    for it) or **reused** from the terminal record it already wrote. The command
    then computes throughput, the between-run spread of the primary metric,
    class/scaffold coverage, the failure histogram, the measured cost and a
    power assessment, and writes ``pilot.json``. Reuse is the default because a
    campaign that already ran to a terminal state is measured evidence, and
    re-running it would charge the operator twice for the same number.

``freeze --protocol PATH --pilot PATH --output PATH``
    Validate a study protocol, re-measure the hashes it locks against, serialize
    it canonically, hash it, and write ``protocol.lock.json``. It refuses a lock
    that disagrees with this checkout's evaluator, and it refuses to overwrite a
    lock under which a test result has already been revealed.

``confirm --protocol PATH --output DIR``
    Execute the locked confirmation study — or refuse, which is what it does
    today. It re-measures the four locked content addresses and the baseline
    arms and refuses a mismatch before anything irreversible starts; it refuses
    a run described as sealed while the lock says the configuration is not; it
    refuses a lock under which a held-out result has already been revealed; and
    it refuses to read the held-out partition at all unless every condition of
    the readiness gate is met. The refusal is the default and execution is the
    explicit, argued exception, because ``test_access_count`` moves from 0 to 1
    exactly once and closes the lock when it does.

``ablate --protocol PATH --manifests DIR --output DIR``
    Register Step 16's ablation comparisons *before* their own final
    evaluation, machine-diff every arm manifest against its declared base, and
    report which arms ran and which are ``NOT_RUN`` with their reason, cost and
    unblock conditions. A manifest that moves a configuration pointer no
    declared factor owns refuses the whole run: an arm that ablates more than it
    declares attributes nothing, and reporting it beside the clean arms would
    launder it. The command reads no partition, charges no budget and — without
    ``--allow-regeneration`` — will not let an arm that needs new computation
    report as executed.

``pmo --manifest PATH --output DIR``
    Run one official-protocol PMO (``mol_opt``) task. PMO is Appendix A §3.4's
    *secondary* optimizer benchmark and is never a prerequisite for accepting a
    ZAK-specific predictor or receptor artifact. The manifest records every
    deviation from upstream, and an unavailable harness is reported as an
    unavailable competitor with an unblock, never as a score.

Exit status is 0 when every requested arm produced a validated result, and
non-zero when any arm failed or was unavailable — an unavailable competitor is a
named limitation, and the command says so loudly rather than exiting 0 with a
hole in the table.

Usage::

    conda run --no-capture-output -n drug-agent python \\
        scripts/research/run_target_study.py baseline \\
        --manifest configs/research/zak-c9r/baselines \\
        --output .external/research/zak-c9r/derived/baselines/run-01
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # pragma: no cover - operator convenience
    sys.path.insert(0, str(REPO / "src"))

from drug_agent.selfevolve.target.baselines import (  # noqa: E402
    PROBE_MATRIX,
    baseline_manifest_paths,
    generator_substitution_check,
    load_manifest,
    probe_capability,
    run_baseline_manifest,
    sha256_text,
    validate_baseline_manifest,
)
from drug_agent.selfevolve.target.budget import (  # noqa: E402
    BudgetExceeded,
    BudgetLedger,
    run_accounted,
)

DEFAULT_MANIFEST_DIR = REPO / "configs" / "research" / "zak-c9r" / "baselines"
CONFIG_DIR = REPO / "configs" / "research" / "zak-c9r"


class _ExplicitValue(argparse.Action):
    """Store a value and record that the caller really passed it.

    A default and a value that happens to equal the default are different facts
    when the default is a RETIRED comparator bar: applying it to a run directory
    that never asked for it is a substitution, and asking for it by name is a
    decision. `sweep-report` branches on `<dest>_explicit`.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def _json_file(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _atomic_json(path: Path, payload: Any) -> Path:
    """Write a record whole or not at all.

    A multi-hour run is read while it is still running, and a half-written
    checkpoint read as a complete one is worse than no checkpoint at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
    return path


def _resolve_manifests(path: Path) -> list[Path]:
    if path.is_dir():
        return baseline_manifest_paths(path)
    if path.name == "index.json":
        index = json.loads(path.read_text())
        return [path.parent / name for name in index["manifests"]]
    return [path]


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def cmd_baseline(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = _resolve_manifests(Path(args.manifest))
    if not paths:
        raise SystemExit(f"no baseline manifests found at {args.manifest}")

    contamination = generator_substitution_check()
    if contamination["contaminated"]:
        # The study driver itself must be clean: an arm inherits this process's
        # environment, and a driver running under the suite's generator
        # substitution would produce fixture molecules under a backend's name.
        print(json.dumps({"status": "CONTAMINATED", **contamination}, indent=2), file=sys.stderr)
        return 3

    ledger = _open_ledger(args, output)
    started = time.time()
    arms: list[dict[str, Any]] = []
    for path in paths:
        manifest = validate_baseline_manifest(load_manifest(path))
        baseline_id = str(manifest["baseline_id"])
        if args.only and baseline_id not in args.only:
            continue
        arm_dir = output / baseline_id
        print(f"[{time.strftime('%H:%M:%S')}] running {baseline_id}", flush=True)
        try:
            record = run_baseline_manifest(
                manifest,
                arm_dir,
                wall_seconds=float(args.wall_seconds),
                cpu_seconds=int(args.cpu_seconds),
                check_capabilities=not args.skip_capability_probes,
                ledger=ledger,
                # One reservation per arm per study directory, so a resumed study
                # neither re-charges nor re-runs an arm that already settled.
                request_key=None if ledger is None else f"{output.name}:{baseline_id}",
            )
        except BudgetExceeded as exc:
            # A refused arm is a budget outcome with a name, never a silent skip.
            print(f"[{time.strftime('%H:%M:%S')}] {baseline_id}: BUDGET_EXCEEDED {exc}", flush=True)
            arms.append(
                {
                    "baseline_id": baseline_id,
                    "arm": manifest["arm"],
                    "track": manifest["track"],
                    "status": "BUDGET_EXCEEDED",
                    "primary_metric": None,
                    "primary_value": None,
                    "reason": str(exc),
                    "cost": {},
                }
            )
            continue
        arms.append(_summarize(record, manifest))
        print(
            f"[{time.strftime('%H:%M:%S')}] {baseline_id}: {arms[-1]['status']} "
            f"{arms[-1]['primary_metric']}={arms[-1]['primary_value']}",
            flush=True,
        )

    study = _study_record(arms, started=started, output=output, ledger=ledger)
    (output / "study.json").write_text(json.dumps(study, indent=2, sort_keys=True))
    if ledger is not None:
        path = ledger.write_report(output / "ledger.json")
        print(f"[{time.strftime('%H:%M:%S')}] budget ledger report: {path}", flush=True)
    print(json.dumps({k: v for k, v in study.items() if k != "arms"}, indent=2))
    failed = [a for a in arms if a["status"] != "SUCCEEDED"]
    return 0 if not failed else 1


def _open_ledger(args: argparse.Namespace, output: Path) -> Any:
    """The campaign's budget ledger for this study, unless it was switched off.

    Every arm reserves before it launches and settles what it used, so the
    declared caps in ``configs/research/zak-c9r.yaml`` are enforced by the code
    that runs the arms rather than by a number in a file nothing reads.
    """
    if getattr(args, "no_ledger", False):
        return None
    from drug_agent.selfevolve.target.budget import BudgetLedger

    path = Path(args.ledger) if getattr(args, "ledger", None) else output / "budget.sqlite"
    ledger = BudgetLedger(path, campaign_id=str(getattr(args, "campaign_id", "") or "zak-c9r-baselines"))
    caps = _declared_caps(args)
    for resource, (limit, unit) in caps.items():
        ledger.declare(resource, limit, unit=unit, note="declared in configs/research/zak-c9r.yaml")
    print(
        f"[{time.strftime('%H:%M:%S')}] budget ledger {path} with caps "
        f"{ {k: v[0] for k, v in caps.items()} }",
        flush=True,
    )
    return ledger


def _declared_caps(args: argparse.Namespace) -> dict[str, tuple[float, str]]:
    """Appendix A §4's campaign caps, read from the target configuration.

    The wall cap is the campaign's, not one arm's: 24 hours across everything.
    ``--wall-hours`` narrows it for a bounded probe; it can never widen it,
    because :meth:`BudgetLedger.declare` refuses an enlargement.
    """
    budget: dict[str, Any] = {}
    config = REPO / "configs" / "research" / "zak-c9r.yaml"
    if config.is_file():
        import yaml

        loaded = yaml.safe_load(config.read_text()) or {}
        budget = dict(loaded.get("budget") or {})
    wall_hours = float(getattr(args, "wall_hours", 0.0) or 0.0) or float(budget.get("wall_hours", 24.0))
    return {
        "wall_seconds": (wall_hours * 3600.0, "seconds"),
        "gpu_seconds": (float(budget.get("gpu_hours", 12.0)) * 3600.0, "seconds"),
        "llm_output_tokens": (float(budget.get("llm_output_tokens", 100_000)), "tokens"),
    }


def _gpu_totals(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Measured GPU seconds, or an explicit unknown. Never a stand-in zero.

    An arm whose sampler ran contributes its measurement — zero included, which
    is a real observation that the arm held no CUDA context. An arm whose
    telemetry was unavailable makes the *total* unknown, because a sum that
    silently drops an unmeasured term is a smaller number presented as a fact.
    """
    seconds = 0.0
    measured = 0
    unknown = 0
    peak: int | None = None
    for arm in arms:
        cost = arm.get("cost") or {}
        if not cost:
            continue
        if cost.get("gpu_telemetry") == "MEASURED" and cost.get("gpu_seconds") is not None:
            measured += 1
            seconds += float(cost["gpu_seconds"])
            if cost.get("max_gpu_memory_mib") is not None:
                peak = max(peak or 0, int(cost["max_gpu_memory_mib"]))
        else:
            unknown += 1
    if unknown and not measured:
        state = "UNAVAILABLE"
    elif unknown:
        state = "PARTIAL"
    elif measured:
        state = "MEASURED"
    else:
        state = "UNAVAILABLE"
    return {
        "gpu_seconds": None if state in {"UNAVAILABLE", "PARTIAL"} else round(seconds, 3),
        "gpu_telemetry": state,
        "gpu_seconds_measured_subtotal": round(seconds, 3) if measured else None,
        "gpu_arms_measured": measured,
        "gpu_arms_unknown": unknown,
        "max_gpu_memory_mib": peak,
    }


def _summarize(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """One row of the study table: everything a later comparison needs, no bulk."""
    result = record.get("result") or {}
    return {
        "baseline_id": record["baseline_id"],
        "arm": record["arm"],
        "track": record["track"],
        "description": record["description"],
        "status": record["status"],
        "exit_status": record.get("exit_status"),
        "primary_metric": result.get("primary_metric"),
        "primary_value": result.get("primary_value"),
        "coverage": result.get("coverage"),
        "n_pool": result.get("n_pool"),
        "n_scored": result.get("n_scored"),
        "n_failed": result.get("n_failed"),
        "pool_hash": result.get("pool_hash"),
        "train_pool_hash": result.get("train_pool_hash"),
        "n_train": result.get("n_train"),
        "qualification_rule": result.get("qualification_rule"),
        "n_submitted": result.get("n_submitted"),
        "n_evaluated": result.get("n_evaluated"),
        "n_qualifying_compounds": result.get("n_qualifying_compounds"),
        "funnel": result.get("funnel"),
        "secondary_retrieval_evaluation": result.get("secondary_retrieval_evaluation"),
        "model": result.get("model"),
        "generator": result.get("generator"),
        "tuning": _tuning_summary(result.get("tuning")),
        "boundary": result.get("boundary"),
        "common_exclusion": result.get("common_exclusion"),
        "manifest_hash": record["manifest_hash"],
        "manifest_path": f"configs/research/zak-c9r/baselines/{record['baseline_id']}.json",
        "result_path": record.get("result_path"),
        "capability_probes": [
            {k: v for k, v in p.items() if k != "source_ids"} for p in record.get("capability_probes", [])
        ],
        "reason": record.get("reason"),
        "unblock": record.get("unblock"),
        "deviations": list(manifest.get("deviations") or []),
        "cost": record.get("cost"),
    }


def _tuning_summary(tuning: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tuning:
        return None
    return {
        k: v
        for k, v in tuning.items()
        # `trials` is the per-configuration detail; it stays in the arm's own
        # result file rather than bloating the committed study table.
        if k != "trials"
    } | {"n_trials_recorded": len(tuning.get("trials") or [])}


def _study_record(
    arms: list[dict[str, Any]], *, started: float, output: Path, ledger: Any = None
) -> dict[str, Any]:
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluator_hash

    cost_ledger = {
        "ledger_kind": "MEASURED_INTEGRATED" if ledger is not None else "PROVISIONAL_MANUAL",
        "ledger_path": None if ledger is None else str(ledger.path),
        "note": (
            "Step 9's integrated budget ledger reserved each arm's wall budget before it "
            "launched and settled what it used. Unknown telemetry is recorded as unknown, "
            "never as zero."
            if ledger is not None
            else "Rows measured per arm with local process accounting; no integrated ledger "
            "was attached to this run."
        ),
        "arms": [
            {
                "baseline_id": a["baseline_id"],
                "status": a["status"],
                **(a.get("cost") or {}),
                "tuning_configurations": (a.get("tuning") or {}).get("n_configurations_evaluated"),
                "tuning_failures": (a.get("tuning") or {}).get("n_configurations_failed"),
            }
            for a in arms
        ],
        "totals": {
            "wall_seconds": round(sum((a.get("cost") or {}).get("wall_seconds") or 0.0 for a in arms), 3),
            "cpu_user_seconds": round(
                sum((a.get("cost") or {}).get("cpu_user_seconds") or 0.0 for a in arms), 3
            ),
            **_gpu_totals(arms),
            "llm_output_tokens": 0,
            "llm_note": "no arm consults an LLM; each arm records the LLM modules it imported",
        },
    }
    if ledger is not None:
        report = ledger.report()
        cost_ledger["usage"] = report["usage"]
        cost_ledger["caps"] = report["caps"]
        cost_ledger["cost"] = report["cost"]
        cost_ledger["evidence_classes"] = report["evidence_classes"]
    return {
        "contract_version": "1.0.0",
        "record_kind": "target_baseline_study",
        "target_id": "CHEMBL3886",
        "target_alias": "zak/c9r",
        "primary_track": "A",
        "primary_metric": "mae_pactivity",
        "evaluated_partition": "development",
        "held_out_partition_note": (
            "The test partition is not read, fitted on or scored by any arm in this step."
        ),
        "evaluator_hash": evaluator_hash(),
        "started_at": started,
        "finished_at": time.time(),
        "wall_seconds": round(time.time() - started, 3),
        "output_dir": str(output),
        "arms": arms,
        "ledger": cost_ledger,
    }


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------
def cmd_capabilities(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    probes = [probe_capability(cap, verb, workdir=output) for cap, verb in PROBE_MATRIX]
    record = {
        "contract_version": "1.0.0",
        "record_kind": "target_capability_probes",
        "note": (
            "Each verb is probed separately. AVAILABLE_EXECUTED means a real computation was "
            "observed on this host; UNAVAILABLE carries a reason and an unblock; NOT_TESTED "
            "means no evidence was gathered either way."
        ),
        "contamination_check": generator_substitution_check(),
        "probes": probes,
        "wall_seconds": round(time.time() - started, 3),
    }
    (output / "capabilities.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    for probe in probes:
        print(f"{probe['capability_id']:34s} {probe['verb']:8s} {probe['status']}")
    return 0


# ---------------------------------------------------------------------------
# pin — the compact, committed reference record
# ---------------------------------------------------------------------------
#: Keys that carry per-compound detail or bulk. They stay in the run directory
#: under `.external/` and never enter the committed record: a comparison needs
#: the metric, the denominators and the hashes, not 34 absolute errors.
_BULK_KEYS = frozenset({"endpoint", "trials", "rows", "absolute_errors", "failures"})


def cmd_pin(args: argparse.Namespace) -> int:
    """Collect a study, the capability probes and the PMO task into one record."""
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluator_hash

    study = json.loads(Path(args.study).read_text())
    capabilities = json.loads(Path(args.capabilities).read_text()) if args.capabilities else None
    pmo = json.loads(Path(args.pmo).read_text()) if args.pmo else None
    index = json.loads((DEFAULT_MANIFEST_DIR / "index.json").read_text())

    arms = [_pin_arm(a) for a in study["arms"]]
    for entry in index.get("unavailable_arms", []):
        arms.append(
            {
                **entry,
                "primary_metric": None,
                "primary_value": None,
                "exit_status": None,
                "evidence_class": "inspected",
            }
        )

    record = {
        "contract_version": "1.0.0",
        "record_kind": "target_baseline_reference_results",
        "target_id": study["target_id"],
        "target_alias": study["target_alias"],
        "primary_track": study["primary_track"],
        "primary_metric": study["primary_metric"],
        "evaluated_partition": study["evaluated_partition"],
        "held_out_partition_note": study["held_out_partition_note"],
        "evaluator_hash": evaluator_hash(),
        "arms": arms,
        "capability_probes": [
            {k: v for k, v in p.items() if k != "source_ids"} for p in (capabilities or {}).get("probes", [])
        ],
        "contamination_check": (capabilities or {}).get("contamination_check"),
        "pmo": _pin_pmo(pmo) if pmo else None,
        "ledger": study["ledger"],
        "run_output_dir": study["output_dir"],
        "notes": [
            "Every number here was produced on the development partition. The held-out "
            "partition was not read, fitted on or scored by any arm.",
            "Per-compound predictions, absolute errors, docked poses and tuning trial "
            "records stay in the run directory under .external/ and are not committed.",
            "An arm with status UNAVAILABLE has primary_value null and a reason. An absent "
            "competitor is a named limitation, never a score.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} with {len(arms)} arms")
    return 0


def _pin_arm(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in arm.items()
        if k not in _BULK_KEYS and not (isinstance(v, (dict, list)) and _too_big(v))
    }


def _too_big(value: Any) -> bool:
    return len(json.dumps(value)) > 20000


def _pin_pmo(pmo: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in pmo.items() if not k.endswith("_tail") and k != "argv"}


# ---------------------------------------------------------------------------
# pilot — feasibility, variance and power, before anything is frozen
# ---------------------------------------------------------------------------
def cmd_pilot(args: argparse.Namespace) -> int:
    """Run or reuse the development pilot campaigns and assess feasibility.

    Appendix A §5 allows up to three development pilot campaigns and excludes
    their artifacts and results from confirmation. This command never opens a
    held-out label: every number it reports comes from a campaign's own terminal
    record on the development partition.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluator_hash
    from drug_agent.selfevolve.target.protocol import assess_power, pilot_assessment

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    entries = [e for e in manifest["campaigns"] if e.get("in_scope", True)]
    started = time.time()

    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    launched = 0
    for entry in entries:
        run_dir = REPO / entry["output"] if not Path(entry["output"]).is_absolute() else Path(entry["output"])
        record_path = run_dir / "campaign.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if record.get("state") in _TERMINAL_CAMPAIGN_STATES:
                records.append(record)
                provenance.append(
                    {
                        "campaign_id": entry["campaign_id"],
                        "provenance": "REUSED_EXECUTED",
                        "record_path": str(record_path),
                        "record_sha256": sha256_text(record_path.read_text()),
                        "state": record.get("state"),
                        "evidence_class": "measured",
                        "note": entry.get("note", ""),
                    }
                )
                continue
        if not args.run_new:
            provenance.append(
                {
                    "campaign_id": entry["campaign_id"],
                    "provenance": "NOT_RUN",
                    "record_path": str(record_path),
                    "evidence_class": "not tested",
                    "reason": "no terminal record on disk and --run-new was not given",
                }
            )
            continue
        if launched >= int(args.max_new_campaigns):
            provenance.append(
                {
                    "campaign_id": entry["campaign_id"],
                    "provenance": "NOT_RUN",
                    "evidence_class": "not tested",
                    "reason": f"the pilot cap of {args.max_new_campaigns} new campaigns is already spent",
                }
            )
            continue
        outcome = _launch_pilot_campaign(entry, run_dir, args)
        provenance.append(outcome)
        if outcome["provenance"] == "EXECUTED_NOW":
            launched += 1
            records.append(json.loads(record_path.read_text()))

    external = list(manifest.get("external_observations") or [])
    comparator = float(manifest["comparator_value"])
    campaign_values = [v for record in records for v in pilot_assessment([record])["development_values"]]
    same_plan = [float(o["value"]) for o in external]
    pooled = [*same_plan, *campaign_values]

    assessment = pilot_assessment(
        records,
        pilot_id=manifest["pilot_id"],
        target_id=manifest["target_id"],
        track=manifest["track"],
        primary_metric=manifest["primary_metric"],
        evaluated_partition="development",
        held_out_partition_note=(
            "No arm, campaign or computation in this record read, fitted on or scored the test "
            "partition. Every value is a development-partition measurement."
        ),
        evaluator_hash=evaluator_hash(),
        manifest_path=str(Path(args.manifest)),
        manifest_sha256=sha256_text(Path(args.manifest).read_text()),
        campaign_provenance=provenance,
        deviations=list(manifest.get("deviations") or []),
        reference_points=manifest.get("reference_points"),
        signal_controls=manifest.get("signal_controls"),
        external_observations=external,
    )
    assessment["variance"] = {
        "comparator_id": manifest["comparator_id"],
        "comparator_value": comparator,
        "estimators": {
            "same_plan_worker_variance": assess_power(
                observations=same_plan,
                comparator_value=comparator,
                locked_n=int(args.locked_n),
                min_power=float(args.min_power),
                evidence_class="inspected",
                note=(
                    "the identical plan re-dispatched; the worker wrote a different implementation "
                    "each time. This isolates worker-implementation variance with the proposal held "
                    "fixed, so it is a LOWER BOUND on campaign-to-campaign variance"
                ),
            )
            if len(same_plan) >= 2
            else None,
            "campaign_process_variance": assess_power(
                observations=pooled,
                comparator_value=comparator,
                locked_n=int(args.locked_n),
                min_power=float(args.min_power),
                evidence_class="measured+inspected",
                note=(
                    "the four same-plan dispatches pooled with the one self-originated campaign "
                    "nominee. It contains only one distinct proposal, so it too is a LOWER BOUND on "
                    "campaign-to-campaign variance and the power it implies is an UPPER BOUND"
                ),
            )
            if len(pooled) >= 2
            else None,
        },
        "locked_estimator": "campaign_process_variance",
        "locked_estimator_reason": (
            "it is the estimator whose replication unit is closest to the confirmation replicate — "
            "a campaign — and it is the more conservative of the two"
        ),
    }
    assessment["coverage"] = _pilot_coverage()
    assessment["baseline_quality"] = manifest.get("reference_points")
    assessment["wall_seconds"] = round(time.time() - started, 3)
    assessment["new_campaigns_launched"] = launched

    (output / "pilot.json").write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    summary = {
        "n_campaigns": assessment["n_campaigns"],
        "throughput": assessment["throughput"],
        "cost": assessment["cost"],
        "new_campaigns_launched": launched,
        "power": {
            name: None
            if block is None
            else {
                k: block[k]
                for k in (
                    "n_observations",
                    "process_mean",
                    "effect_under_test",
                    "process_sd_point",
                    "process_sd_interval",
                    "power_at_locked_n_point",
                    "power_at_locked_n_worst_case",
                    "n_for_min_power_worst_case",
                )
            }
            for name, block in assessment["variance"]["estimators"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {output / 'pilot.json'}")
    # A pilot that gathered no usable campaign is not a successful assessment.
    return 0 if records else 1


#: A campaign in one of these states has finished and will not change. Anything
#: else is still running or was interrupted, and reusing it would report a
#: partial run as a measurement.
_TERMINAL_CAMPAIGN_STATES = frozenset({"NOMINATED", "COMPLETED", "NO_IMPROVEMENT", "FAILED", "BLOCKED"})


def _launch_pilot_campaign(entry: dict[str, Any], run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Launch one pilot campaign, or refuse it with the arithmetic that refused it.

    A campaign costs the operator real reasoning tokens. Before anything is
    launched the declared per-campaign token allowance is compared against the
    remaining authorisation; a campaign that cannot be paid for is refused by
    name and by number rather than started and abandoned halfway.
    """
    needed = float(args.per_campaign_output_tokens)
    remaining = float(args.authorised_output_tokens) - float(args.spent_output_tokens)
    if needed > remaining:
        return {
            "campaign_id": entry["campaign_id"],
            "provenance": "REFUSED_BUDGET",
            "evidence_class": "measured",
            "reason": (
                f"a campaign needs {needed:,.0f} output tokens and {remaining:,.0f} remain of the "
                f"{float(args.authorised_output_tokens):,.0f} authorised "
                f"({float(args.spent_output_tokens):,.0f} already measured as spent). "
                f"The campaign is not started rather than started and abandoned"
            ),
            "needed_output_tokens": needed,
            "remaining_output_tokens": remaining,
        }
    argv = [
        sys.executable,
        str(REPO / "scripts" / "research" / "run_campaign.py"),
        "--campaign",
        str(REPO / entry["config"]),
        "--output",
        str(run_dir),
        "--max-cycles",
        str(int(args.max_cycles)),
        "--per-cycle-output-tokens",
        str(float(args.per_cycle_output_tokens)),
    ]
    ledger = BudgetLedger(run_dir / "budget.sqlite", campaign_id=entry["campaign_id"])
    ledger.declare(
        "wall_seconds",
        float(args.campaign_wall_seconds),
        unit="seconds",
        note="declared by the pilot manifest",
    )
    run = run_accounted(
        argv,
        ledger=ledger,
        resource="wall_seconds",
        wall_seconds=float(args.campaign_wall_seconds),
        owner=f"PILOT/{entry['campaign_id']}",
        phase="PILOT",
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        log_dir=run_dir,
    )
    return {
        "campaign_id": entry["campaign_id"],
        "provenance": "EXECUTED_NOW" if run["exit_status"] == 0 else "EXECUTION_FAILED",
        "evidence_class": "executed",
        "exit_status": run["exit_status"],
        "wall_seconds": run["cost"]["wall_seconds"],
        "checkpoint_path": run["checkpoint_path"],
    }


def _pilot_coverage() -> dict[str, Any]:
    """Class and scaffold coverage of the pools the pilot scored, from the split card.

    Counts and identifiers only. No measured value of any compound appears here,
    and nothing from the held-out partition is read.
    """
    config_path = REPO / "configs" / "research" / "zak-c9r.yaml"
    if not config_path.is_file():
        return {"status": "UNAVAILABLE", "reason": f"{config_path} is absent"}
    import yaml

    config = yaml.safe_load(config_path.read_text()) or {}
    split = config.get("split") or {}
    track_a = (config.get("benchmark") or {}).get("track_a") or {}
    baselines = ((config.get("baselines") or {}).get("track_a")) or {}
    counts = split.get("counts") or {}
    return {
        "development_pool_hash": baselines.get("pool_hash"),
        "n_development_pool": baselines.get("n_pool"),
        "n_train": baselines.get("n_train"),
        "split_counts": counts,
        "similarity_threshold": split.get("similarity_threshold"),
        "cross_partition_max_tanimoto": split.get("cross_partition_max_tanimoto"),
        "coverage_on_every_scored_arm": 1.0,
        "coverage_evidence": (
            "every Track A arm and every scored pilot cycle covered 34/34 of the development pool; "
            "the coverage non-regression tolerance is "
            f"{track_a.get('coverage_tolerance')}"
        ),
        "ranking_branch": track_a.get("ranking_branch"),
        "class_balance_note": (
            "the ranking branch is INSUFFICIENT_TEST_DATA: the development partition holds 0 "
            "positives under the frozen label policy, so no class-balance statistic exists to "
            "report and EF10 cannot be a primary endpoint"
        ),
    }


# ---------------------------------------------------------------------------
# freeze — the canonical, hashed protocol lock
# ---------------------------------------------------------------------------
def cmd_freeze(args: argparse.Namespace) -> int:
    """Validate a protocol against this checkout and write the hashed lock."""
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluator_hash
    from drug_agent.selfevolve.target.protocol import freeze_protocol, load_lock, protocol_hash

    protocol = json.loads(Path(args.protocol).read_text())
    measured = evaluator_hash()
    if protocol.get("evaluator_hash") != measured:
        print(
            json.dumps(
                {
                    "status": "STALE_EVALUATOR",
                    "locked": protocol.get("evaluator_hash"),
                    "measured": measured,
                    "unblock": (
                        "the protocol commits to an evaluator address this checkout does not "
                        "produce. Re-freeze the benchmark records deliberately, then re-freeze "
                        "the protocol; do not edit the address"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    out = Path(args.output)
    if out.is_file() and not args.force:
        existing = load_lock(out)
        exposure = existing["protocol"].get("evaluation_exposure") or {}
        if exposure.get("test_access_count", 0) > 0 or exposure.get("revealed_at") is not None:
            print(
                json.dumps(
                    {
                        "status": "LOCK_CLOSED",
                        "existing_protocol_hash": existing["protocol_hash"],
                        "reason": (
                            "a test result has been revealed under the existing lock. A new protocol "
                            "version needs a new evaluation-exposure declaration and cannot be "
                            "presented as the original untouched confirmation"
                        ),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 3
        if existing["protocol_hash"] == protocol_hash(protocol):
            print(json.dumps({"status": "UNCHANGED", "protocol_hash": existing["protocol_hash"]}, indent=2))
            return 0

    pilot_ref = None
    if args.pilot:
        pilot_path = Path(args.pilot)
        pilot_ref = {
            "path": str(pilot_path),
            "sha256": sha256_text(pilot_path.read_text()),
            "record_kind": json.loads(pilot_path.read_text()).get("record_kind"),
        }

    path = freeze_protocol(
        protocol,
        out,
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        frozen_by="scripts/research/run_target_study.py freeze",
        source_protocol_path=str(Path(args.protocol)),
        source_protocol_sha256=sha256_text(Path(args.protocol).read_text()),
        pilot_evidence=pilot_ref,
        measured_evaluator_hash=measured,
        note=(
            "The protocol below completely determines the confirmation study and its inference. "
            "It was frozen with test_access_count == 0. Any later version must declare what it "
            "supersedes and carry a new evaluation-exposure declaration."
        ),
    )
    lock = load_lock(path)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "lock_path": str(path),
                "protocol_hash": lock["protocol_hash"],
                "primary_track": lock["protocol"].get("primary_track"),
                "primary_metric": lock["protocol"]["primary_metric"],
                "feasibility": lock["protocol"]["feasibility"]["classification"],
                "max_final_nominations": lock["protocol"]["max_final_nominations"],
                "evaluator_hash": lock["protocol"]["evaluator_hash"],
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------
#: Exit codes. A refusal never shares a code with a success, so a caller that
#: only checks `!= 0` still cannot mistake a NOT_RUN for a confirmation.
CONFIRM_EXECUTED = 0
CONFIRM_HASH_MISMATCH = 2
CONFIRM_LOCK_CLOSED = 3
CONFIRM_UNSEALED_CLAIM = 4
CONFIRM_NOT_RUN = 5
CONFIRM_BOUNDARY_VIOLATION = 6
CONFIRM_PROTOCOL_NOT_AUTHENTIC = 7


def _confirm_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the protocol against its REGISTERED address, or refuse.

    Accepts either the protocol source or a frozen lock, and neither is believed
    on its own account. `load_lock` re-derives a lock's hash and compares it with
    the hash stored *in the same file*, which proves the file agrees with itself
    and nothing more: a lock with fields flipped and re-stamped with this
    project's own `protocol_hash()` passes that check exactly as the real one
    does, and every downstream consumer — the lock-closed check, the locked side
    of the preflight, the ledger's denominator and its one-nomination cap, all
    the gate conditions, the intention-to-run denominator and the recorded cost —
    reads out of whatever it was handed.

    So the comparison is against `confirmation.REGISTERED_PROTOCOL_HASH`, which
    is a constant in the library rather than anything this command was passed.

    Raises:
        ValueError: a `ContractError` when the document is not the registered
            protocol. The caller turns that into exit
            `CONFIRM_PROTOCOL_NOT_AUTHENTIC`.
    """
    from drug_agent.selfevolve.target.confirmation import authenticate_protocol

    protocol, authentication = authenticate_protocol(_json_file(path), source=str(path))
    return protocol, authentication


def _measured_baseline_hashes(manifests: Path) -> dict[str, str]:
    """Re-derive each baseline arm's content address from its manifest bytes.

    Deliberately *not* read back out of the pinned results record: a comparator
    address copied from the same file that reports the comparison proves nothing
    about the manifest that would actually run.
    """
    measured: dict[str, str] = {}
    for path in _resolve_manifests(manifests):
        record = validate_baseline_manifest(load_manifest(path))
        measured[str(record["baseline_id"])] = sha256_text(json.dumps(record, sort_keys=True))
    return measured


def cmd_confirm(args: argparse.Namespace) -> int:
    """Execute the locked confirmation — or refuse, which is today's outcome.

    The order below is the contract.

    1. A document declaring a spent partition is refused as `LOCK_CLOSED`,
       before it is authenticated, because that refusal holds either way.
    2. The protocol is AUTHENTICATED against the registered content address.
       Nothing below this line reads a document that did not hash to it — and
       everything below this line reads out of it: the locked side of the
       preflight, the ledger's denominator and its one-nomination cap, all the
       gate conditions, the intention-to-run denominator and the recorded cost.
    3. Preflight refuses a study that is not the locked one before anything
       expensive or irreversible starts, using only addresses this command
       MEASURED: no address may be typed on the command line.
    4. The nomination ledger is audited for content changes, not just for how
       many entries it carries.
    5. The readiness gate decides whether the held-out partition may be read at
       all. No flag can open it alone: every condition is derived from the
       authenticated lock and the ledger, the first of them re-derives the
       protocol address independently, and `--release-held-out` is the last
       rather than a bypass of the rest.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluator_hash
    from drug_agent.selfevolve.target.confirmation import (
        NominationBoundary,
        confirmation_readiness,
        intention_to_run_table,
        not_run_record,
        preflight,
    )
    from drug_agent.selfevolve.target.contracts import ContractError

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    document = _json_file(Path(args.protocol))

    # Read BEFORE authentication, and used only to REFUSE. A document declaring a
    # spent partition is refused whether or not it is the registered one, and
    # reporting that as an authentication failure would bury the more serious
    # fact. Nothing here can let a run proceed: the authentication below is the
    # only way past this point, and it is not reached by any document that
    # hashes to anything but the registered address.
    body = document if isinstance(document, Mapping) else {}
    carried = body.get("protocol")
    if isinstance(carried, Mapping):
        body = carried
    declared_exposure = dict(body.get("evaluation_exposure") or {})
    if declared_exposure.get("test_access_count", 0) > 0 or declared_exposure.get("revealed_at") is not None:
        print(
            json.dumps(
                {
                    "status": "LOCK_CLOSED",
                    "test_access_count": declared_exposure.get("test_access_count"),
                    "revealed_at": declared_exposure.get("revealed_at"),
                    "reason": (
                        "a held-out result has already been revealed under this lock. Any further "
                        "study needs a superseding version with its own exposure declaration and "
                        "may never be presented as the original confirmation"
                    ),
                    "note": (
                        "read from the document as supplied, before it was authenticated. This "
                        "check can only refuse; nothing proceeds on it"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return CONFIRM_LOCK_CLOSED

    try:
        protocol, authentication = _confirm_protocol(Path(args.protocol))
    except ContractError as error:
        print(
            json.dumps(
                {
                    "status": "PROTOCOL_NOT_AUTHENTIC",
                    "category": error.category.value,
                    "field": error.field,
                    "protocol_source": str(Path(args.protocol)),
                    "detail": str(error),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return CONFIRM_PROTOCOL_NOT_AUTHENTIC
    lock_hash = str(authentication["measured_protocol_hash"])
    exposure = dict(protocol.get("evaluation_exposure") or {})

    # Only the addresses this command can genuinely RE-DERIVE go in here. The
    # evaluation kernel is hashed from the bytes on disk; the baseline arms are
    # hashed from their manifests. `target_hash`, `data_hash` and `split_hash`
    # cannot be re-derived without rebuilding the split, which would read the
    # held-out family assignment — so they are deliberately absent, and
    # `preflight` reports them as not independently measured rather than
    # comparing the protocol with itself and calling that agreement.
    measured: dict[str, Any] = {"evaluator_hash": evaluator_hash()}
    # `if args.baselines:` treated an EMPTY value as "no baselines", which silently
    # skipped the moved-comparator refusal and left `checked: false` nested in the
    # record as its only trace. An empty value is now a refusal: a safety check
    # that can be switched off by passing nothing is not a safety check.
    if args.baselines is None or not str(args.baselines).strip():
        print(
            json.dumps(
                {
                    "status": "REFUSED_EMPTY_VALUE",
                    "field": "--baselines",
                    "reason": (
                        "--baselines was given with no value. An empty value does not disable the "
                        "comparator check, it is refused: a confirmation is a COMPARISON against "
                        "the locked baseline arms, and skipping their re-measurement would leave "
                        "`baseline_hashes.checked: false` as the only trace that the "
                        "moved-comparator refusal never ran"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return CONFIRM_HASH_MISMATCH
    measured["baseline_hashes"] = _measured_baseline_hashes(Path(args.baselines))
    # Every caller-typed address is refused, not only `evaluator_hash`. A value
    # typed on the command line is not a measurement, and preflight compares
    # `measured` against the protocol's own fields — so accepting one reinstates
    # the circular self-comparison that was removed as deviation 7: the caller
    # types the protocol's value, the preflight finds it agrees, and the record
    # reports an address as verified that nothing measured.
    for override in args.assert_hash or []:
        field = override.partition("=")[0].strip()
        print(
            json.dumps(
                {
                    "status": "REFUSED_ASSERTION",
                    "field": field,
                    "value_ignored": True,
                    "reason": (
                        "an address typed on the command line is a CLAIM, not a measurement, and "
                        "preflight compares what it is given against the protocol's own field. "
                        "Accepting one would make the preflight compare the protocol with a value "
                        "the caller copied out of it and call the result agreement. evaluator_hash "
                        "is measured from this checkout's bytes; target_hash, data_hash and "
                        "split_hash cannot be re-derived here without rebuilding the split, which "
                        "reads the held-out family assignment, so they are reported as "
                        "NOT independently measured rather than asserted"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return CONFIRM_HASH_MISMATCH

    try:
        checked = preflight(protocol, measured, described_as_sealed=args.sealed)
    except ContractError as error:
        print(
            json.dumps(
                {"status": error.category.value, "field": error.field, "detail": str(error)}, indent=2
            ),
            file=sys.stderr,
        )
        if error.category.value == "UNSUPPORTED_CLAIM":
            return CONFIRM_UNSEALED_CLAIM
        return CONFIRM_HASH_MISMATCH

    if checked.get("unchecked_safety_checks"):
        print(
            json.dumps(
                {
                    "status": "UNCHECKED_SAFETY_CHECK",
                    "unchecked": checked["unchecked_safety_checks"],
                    "reason": (
                        "the lock registers comparator arms that nothing re-measured in this run. "
                        "A confirmation is a comparison against those arms, so a check that did "
                        "not run is refused rather than recorded as a nested false"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return CONFIRM_HASH_MISMATCH

    ledger_path = Path(args.nominations) if args.nominations else output / "nominations.json"
    if ledger_path.is_file():
        # The design — the denominator and the one-nomination cap — comes from the
        # LOCK; only the entries come from the file. A ledger that sets its own cap
        # is recorded as a violation rather than believed.
        boundary = NominationBoundary.resume_under_protocol(
            protocol, _json_file(ledger_path), path=ledger_path
        )
    else:
        boundary = NominationBoundary.from_protocol(protocol, path=ledger_path)

    # The ledger on disk is what a later reader believes, and a file can be
    # hand-edited after the fact. Re-check the boundary's invariants over what
    # was actually loaded before any of it is allowed to influence a decision.
    violations = boundary.audit()
    if violations:
        print(
            json.dumps(
                {
                    "status": "BOUNDARY_VIOLATION",
                    "nomination_ledger": str(ledger_path),
                    "violations": violations,
                    "reason": (
                        "the nomination ledger breaks the release boundary it is supposed to "
                        "record. Nothing is scored against a ledger that cannot be trusted to "
                        "say what was nominated before any test feedback existed"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return CONFIRM_BOUNDARY_VIOLATION

    readiness = confirmation_readiness(
        protocol,
        track=args.track,
        boundary=boundary,
        authorisation=_json_file(Path(args.authorisation)) if args.authorisation else None,
        operator_release_requested=args.release_held_out,
    )
    # CLASS 3. The table refuses a held-out value while the boundary is not
    # RELEASED, so it needs the ledger's state — and it now DERIVES that from the
    # boundary object rather than being told a word.
    #
    # This command used to pass `boundary_state=boundary.state`, where
    # `boundary.state` came from a `released` block in the nomination ledger, a
    # file nothing authenticates. A hand-written released block therefore flipped
    # the same switch through the CLI: it made `decisions_for_worker()` hand
    # held-out-carrying entries back to the worker and let this table read them
    # into measured deltas. `resume_under_protocol` now derives a release from the
    # entries beside it instead of believing the claim, `audit()` reports an
    # uncorroborated one, and the audit above refuses the run before this line.
    #
    # Both calls below REFUSE rather than return on a ledger that came off disk
    # carrying something they will not put in a record — a number under a name
    # nobody forbade, or evidence that a release happened. A refusal must not
    # arrive as a traceback: it is the same BOUNDARY_VIOLATION the audit reports,
    # and no record is written.
    try:
        table = intention_to_run_table(protocol, boundary.entries, boundary=boundary)

        record = not_run_record(
            protocol,
            readiness,
            protocol_hash=lock_hash,
            protocol_source=str(Path(args.protocol)),
            protocol_authentication=authentication,
            preflight=checked,
            intention_to_run=table,
            nomination_ledger=str(ledger_path),
            nomination_ledger_digest=boundary.content_digest(),
            recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            recorded_by="scripts/research/run_target_study.py confirm",
        )
    except ContractError as error:
        print(
            json.dumps(
                {
                    "status": "BOUNDARY_VIOLATION",
                    "category": error.category.value,
                    "field": error.field,
                    "paths": list(error.missing or []),
                    "nomination_ledger": str(ledger_path),
                    "detail": str(error),
                    "reason": (
                        "the branch-status record asserts that the held-out partition was NOT "
                        "READ, so it is refused rather than written over anything that says "
                        "otherwise. No record was written. The paths are named; no value is"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return CONFIRM_BOUNDARY_VIOLATION
    if readiness["decision"] == "READY":
        record["status"] = "READY_NOT_EXECUTED"
        record["why_not_failed"] = (
            "every gate condition is met. Execution of the held-out evaluation is deliberately "
            "not wired into this command: the run that spends test_access_count is launched by "
            "the registered independent evaluation harness under its own operator approval"
        )
    _atomic_json(output / "confirmation.json", record)
    _atomic_json(output / "readiness.json", readiness)
    _atomic_json(output / "intention-to-run.json", table)
    if args.pin:
        # The committed record is the compact one: the branch's status, why, what
        # it would cost and the conditions that would let it run. The run
        # directory keeps the rest.
        pinned = {
            key: value
            for key, value in record.items()
            if key not in {"preflight", "intention_to_run", "nomination_ledger", "protocol_source"}
        }
        # Carried into the committed record: which address the protocol was
        # authenticated against is the fact every other number here depends on.
        pinned["protocol_authentication"] = {
            key: value for key, value in authentication.items() if key != "source"
        }
        pinned["preflight"] = {
            "status": checked["status"],
            "hashes_agreed": checked["hashes_agreed"],
            # Carried into the committed record deliberately: a pinned preflight
            # that showed only what agreed would read as a stronger check than
            # the one that ran.
            "not_independently_measured": checked["not_independently_measured"],
            "not_independently_measured_note": checked["not_independently_measured_note"],
            "baseline_hashes": checked["baseline_hashes"],
            "sealed": checked["sealed"],
        }
        pinned["intention_to_run"] = {
            "treatment": table["treatment"],
            "denominator": table["denominator"],
            "locked_campaigns": table["locked_campaigns"],
            "n_nominated": table["n_nominated"],
            "never": table["never"],
        }
        _atomic_json(Path(args.pin), pinned)

    summary = {
        "status": record["status"],
        "implementation_status": record["implementation_status"],
        "protocol_version": protocol.get("protocol_version"),
        "protocol_hash": lock_hash,
        "protocol_authenticated_against": authentication["registered_protocol_hash"],
        "track": args.track,
        "decision": readiness["decision"],
        "blocking_conditions": readiness["blocking_conditions"],
        "test_access_count": exposure.get("test_access_count"),
        "held_out_partition": "NOT READ",
        "denominator": table["denominator"],
        "n_nominated": table["n_nominated"],
        "output": str(output),
    }
    if readiness["decision"] == "READY":
        print(json.dumps(summary, indent=2))
        return CONFIRM_EXECUTED
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return CONFIRM_NOT_RUN


# ---------------------------------------------------------------------------
# ablate
# ---------------------------------------------------------------------------
ABLATE_REPORT_WRITTEN = 0
ABLATE_LOCK_CLOSED = 3
ABLATE_UNDECLARED_FACTOR = 4
ABLATE_NO_ARM_EXECUTED = 5
ABLATE_MANIFEST_INVALID = 6
ABLATE_PROTOCOL_NOT_AUTHENTIC = 7

DEFAULT_ABLATION_DIR = CONFIG_DIR / "ablations"
DEFAULT_CAMPAIGN_DIR = REPO / ".external" / "research" / "zak-c9r" / "derived" / "campaigns"

#: The four predeclared arms, in the order the lock declares them. Read from the
#: protocol at run time; this is only the fallback ordering for a summary line.
_ABLATION_ARM_ORDER = (
    "ABL-1-no-history",
    "ABL-2-shared-menu-random",
    "ABL-3-docking-feature-only",
    "ABL-4-acquisition-enabled",
)


def _ablation_bases(directory: Path) -> dict[str, Any]:
    """Every base configuration in *directory*, keyed by file stem and by base_id."""
    bases: dict[str, Any] = {}
    for path in sorted(directory.glob("base-*.json")):
        record = _json_file(path)
        bases[path.stem] = record
        bases[path.name] = record
        bases[str(record.get("base_id"))] = record
    return bases


def _real_histories(directory: Path) -> list[dict[str, Any]]:
    """Every terminal campaign history on disk, as ``{campaign_id, records}``.

    These are the only real campaign histories this programme owns. The mechanism
    diff runs against them rather than against an invented one, because what
    "suppress the history" moves depends on what the history contains.
    """
    out: list[dict[str, Any]] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*/history.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out.append({"campaign_id": path.parent.name, "n_records": len(records), "records": records})
    return out


def cmd_ablate(args: argparse.Namespace) -> int:
    """Register the ablation comparisons, diff the manifests, and report what ran.

    The order below is the contract.

    1. A document declaring a spent partition is refused before anything else,
       exactly as `confirm` refuses it, because this command's own evidence
       would be worthless under a closed lock.
    2. The protocol is AUTHENTICATED against the registered content address.
       Every arm's readiness gate, the matched-resource accounting and the closed
       arm set are read out of it afterwards.
    3. `test_access_count` is re-verified across the whole tree by the project's
       own scanner, and the value it observes is what the report records as
       evidence that no test result was viewed. The claim is measured, not typed.
    4. Every manifest is machine-diffed against its declared base. A manifest
       that moves a pointer no declared factor owns, or declares a factor that
       moves nothing, refuses the whole run at `ABLATE_UNDECLARED_FACTOR` — an
       unclean ablation attributes nothing and reporting it beside clean ones
       would launder it.
    5. The comparisons are registered, each with its honest registration status.
       A comparison whose evidence already exists cannot be registered
       CONFIRMATORY; `ComparisonRegistry` refuses that rather than relying on
       this command to pass the right flag.
    6. Only then is any evidence read, and only from paths the caller named.
    """
    from drug_agent.selfevolve.target.ablation import (
        REGISTERED_COMPARISONS,
        ComparisonRegistry,
        ablation_report,
        arm_not_run_record,
        arm_readiness,
        distillation_attribution,
        history_derived_fields_are_still_these,
        history_suppression_diff,
        load_manifest,
        manifest_paths,
        matched_resource_accounting,
        revert_main_factor_attribution,
        sweep_compute_attribution,
        verify_declared_factors_only,
    )
    from drug_agent.selfevolve.target.contracts import ContractError
    from drug_agent.selfevolve.target.generation_sweep import verify_test_access_count

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    document = _json_file(Path(args.protocol))

    body = document if isinstance(document, Mapping) else {}
    carried = body.get("protocol")
    if isinstance(carried, Mapping):
        body = carried
    exposure = dict(body.get("evaluation_exposure") or {})
    if exposure.get("test_access_count", 0) > 0 or exposure.get("revealed_at") is not None:
        print(
            json.dumps(
                {
                    "status": "LOCK_CLOSED",
                    "test_access_count": exposure.get("test_access_count"),
                    "reason": (
                        "a held-out result has already been revealed under this lock. Every "
                        "ablation evaluated afterwards is explanatory, and this command will not "
                        "write a record that could be read as confirmation"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return ABLATE_LOCK_CLOSED

    try:
        protocol, authentication = _confirm_protocol(Path(args.protocol))
    except ContractError as error:
        print(
            json.dumps(
                {
                    "status": "PROTOCOL_NOT_AUTHENTIC",
                    "category": error.category.value,
                    "field": error.field,
                    "detail": str(error),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return ABLATE_PROTOCOL_NOT_AUTHENTIC

    access = verify_test_access_count()
    if not access["all_zero"]:
        print(
            json.dumps(
                {
                    "status": "LOCK_CLOSED",
                    "n_records_declaring_test_access_count": access["n_records"],
                    "values": access["values"],
                    "reason": "a record in the tree declares a non-zero test_access_count",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return ABLATE_LOCK_CLOSED
    test_results_viewed = False

    manifest_dir = Path(args.manifests)
    bases = _ablation_bases(manifest_dir)
    diffs: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in manifest_paths(manifest_dir):
        try:
            manifest = load_manifest(path)
        except ContractError as error:
            print(
                json.dumps(
                    {
                        "status": "MANIFEST_INVALID",
                        "manifest": str(path),
                        "category": error.category.value,
                        "field": error.field,
                        "detail": str(error),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return ABLATE_MANIFEST_INVALID
        base = bases.get(str(manifest["base_manifest"]))
        if base is None:
            print(
                json.dumps(
                    {
                        "status": "MANIFEST_INVALID",
                        "manifest": str(path),
                        "detail": f"base_manifest {manifest['base_manifest']!r} is not in {manifest_dir}",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return ABLATE_MANIFEST_INVALID
        try:
            diff = verify_declared_factors_only(base, manifest)
        except ContractError as error:
            print(
                json.dumps(
                    {
                        "status": "UNDECLARED_FACTOR",
                        "arm_id": manifest["arm_id"],
                        "category": error.category.value,
                        "field": error.field,
                        "detail": str(error),
                        "note": (
                            "an arm that moves more than it declares attributes nothing. The run is "
                            "refused rather than reported beside the clean arms"
                        ),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return ABLATE_UNDECLARED_FACTOR
        manifests.append(manifest)
        diffs.append(diff)

    registry = ComparisonRegistry(protocol=protocol)
    for comparison in REGISTERED_COMPARISONS:
        registry.register(
            comparison,
            evidence_exists=bool(comparison["evidence_exists"]),
            test_results_viewed=test_results_viewed,
        )
    registry_record = registry.record()

    resources = matched_resource_accounting(protocol)

    readiness: list[dict[str, Any]] = []
    not_run: list[dict[str, Any]] = []
    for arm in (protocol.get("ablation") or {}).get("arms") or []:
        gate = arm_readiness(
            protocol,
            str(arm["arm_id"]),
            regeneration_permitted=bool(args.allow_regeneration),
        )
        readiness.append(gate)
        if gate["decision"] != "EXECUTED":
            not_run.append(arm_not_run_record(protocol, gate))

    mechanism: dict[str, Any] | None = None
    histories = _real_histories(Path(args.histories))
    if histories:
        per_campaign = [
            {
                "campaign_id": entry["campaign_id"],
                "n_records": entry["n_records"],
                **{
                    k: v
                    for k, v in history_suppression_diff(entry["records"]).items()
                    if k not in ("contract_version", "record_kind", "generated_by")
                },
            }
            for entry in histories
        ]
        pooled = history_suppression_diff([r for entry in histories for r in entry["records"]])
        mechanism = {
            **pooled,
            "per_campaign": per_campaign,
            "drift_check": history_derived_fields_are_still_these(),
            "campaigns": [entry["campaign_id"] for entry in histories],
        }

    executed: list[dict[str, Any]] = []
    if args.distill_ablation:
        repeats = _json_file(Path(args.distill_repeats)) if args.distill_repeats else None
        executed.append(distillation_attribution(_json_file(Path(args.distill_ablation)), repeats))
    if args.campaign:
        campaign = _json_file(Path(args.campaign))
        selected = (campaign.get("selection") or {}).get("selected") or {}
        incumbent = dict(campaign.get("incumbent") or {})
        reference = next(
            (
                {"id": c["id"], "value": c.get("development_value"), "role": c.get("role")}
                for c in protocol.get("comparators") or []
                if c.get("role") == "REFERENCE_BAR_ALWAYS_REPORTED"
            ),
            None,
        )
        executed.append(
            revert_main_factor_attribution(
                {
                    "artifact_id": selected.get("artifact_id"),
                    "campaign_id": campaign.get("campaign_id"),
                    "action": selected.get("action"),
                    "metric": campaign.get("development_metric"),
                    "value": selected.get("development_metric_value"),
                    "evaluated_partition": selected.get("evaluated_partition"),
                },
                {
                    "artifact_id": incumbent.get("artifact_id"),
                    "baseline_id": incumbent.get("baseline_id"),
                    "metric": incumbent.get("metric"),
                    "value": incumbent.get("value"),
                    "evaluated_partition": incumbent.get("evaluated_partition"),
                    "pool_hash": incumbent.get("pool_hash"),
                    "n_pool": incumbent.get("n_pool"),
                },
                reference=reference,
            )
        )
    if args.sweep_report:
        executed.append(sweep_compute_attribution(_json_file(Path(args.sweep_report))))

    report = ablation_report(
        protocol,
        registry=registry_record,
        resources=resources,
        readiness=readiness,
        not_run=not_run,
        executed=executed,
        diffs=diffs,
        mechanism=mechanism,
        test_access={
            "n_records_declaring_test_access_count": access["n_records"],
            "values": access["values"],
            "all_zero": access["all_zero"],
            "verified_at": access["verified_at"],
            "scanner": "drug_agent.selfevolve.target.generation_sweep.verify_test_access_count",
        },
        protocol_hash=authentication["measured_protocol_hash"],
        protocol_authenticated_against=authentication["registered_protocol_hash"],
        manifest_dir=str(manifest_dir),
        n_manifests=len(manifests),
        regeneration_permitted=bool(args.allow_regeneration),
    )

    _atomic_json(output / "ablation.json", report)
    _atomic_json(output / "registry.json", registry_record)
    _atomic_json(output / "diffs.json", {"n_diffs": len(diffs), "diffs": diffs})
    if args.pin:
        _atomic_json(Path(args.pin), _pinned_ablation_record(report))

    summary = {
        "status": "REPORT_WRITTEN",
        "protocol_version": protocol.get("protocol_version"),
        "protocol_hash": authentication["measured_protocol_hash"],
        "test_access_count": access["values"],
        "held_out_partition": "NOT READ",
        "n_comparisons_registered": registry_record["n_comparisons"],
        "n_registered_before_evaluation": registry_record["n_registered_before_evaluation"],
        "n_manifests_diffed": len(diffs),
        "n_manifests_matched": sum(1 for d in diffs if d["matched"]),
        "n_arms_executed": sum(1 for g in readiness if g["decision"] == "EXECUTED"),
        "n_arms_not_run": len(not_run),
        "arms_not_run": [n["arm_id"] for n in not_run],
        "n_controlled_comparisons_executed": len(executed),
        "factors_separated": report["attribution"]["factors_separated_on_development_evidence"],
        "factors_with_no_executed_comparison": report["attribution"][
            "factors_with_no_controlled_comparison_executed"
        ],
        "unique_mechanism_claim_supported": report["attribution"]["unique_mechanism_claim"]["supported"],
        "n_unmatched_factors": report["n_unmatched_factors"],
        "output": str(output),
    }
    if args.require_executed_arm and not summary["n_arms_executed"]:
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return ABLATE_NO_ARM_EXECUTED
    print(json.dumps(summary, indent=2))
    return ABLATE_REPORT_WRITTEN


def _pinned_ablation_record(report: Mapping[str, Any]) -> dict[str, Any]:
    """The committed status record: everything but the bulk of the diffs.

    Pinned so a later reader can see which comparisons exist, which arms ran and
    on what conditions the rest could, without needing the gitignored runtime
    directory. Per-pointer diff bodies stay out; their counts and verdicts stay in.
    """
    diffs = [
        {
            "variant_id": d.get("variant_id"),
            "base_id": d.get("base_id"),
            "base_address": d.get("base_address"),
            "variant_address": d.get("variant_address"),
            "n_changed": d.get("n_changed"),
            "changed_pointers": [c["pointer"] for c in d.get("changed") or []],
            "changes_by_factor": d.get("changes_by_factor"),
            "n_undeclared_changes": d.get("n_undeclared_changes"),
            "declared_but_unchanged": d.get("declared_but_unchanged"),
            "matched": d.get("matched"),
        }
        for d in report.get("configuration_diffs") or []
    ]
    mechanism = dict(report.get("mechanism_diff") or {})
    # The attribution table embeds each executed comparison's whole record, and
    # `executed` below carries the same objects. Keep one copy: the table keeps
    # the pointer and the verdict, `executed` keeps the evidence.
    attribution = json.loads(json.dumps(report.get("attribution") or {}))
    for row in attribution.get("factors") or []:
        for evidence in row.get("evidence") or []:
            result = evidence.get("result")
            evidence["result"] = (
                None
                if not result
                else {
                    "record_kind": result.get("record_kind"),
                    "status": result.get("status"),
                    "decision": result.get("decision"),
                    "matched": result.get("matched"),
                    "see": "executed[] in this record, keyed by comparison_id",
                }
            )
    return {
        "contract_version": "1.0.0",
        "record_kind": "ablation_status",
        "protocol_version": report.get("protocol_version"),
        "protocol_hash": report.get("protocol_hash"),
        "generated_at": report.get("generated_at"),
        "generated_by": "scripts/research/run_target_study.py ablate --pin",
        "test_results_viewed": report.get("test_results_viewed"),
        "test_results_viewed_evidence": report.get("test_results_viewed_evidence"),
        "why": report.get("why"),
        "registry": report.get("registry"),
        "matched_resources": report.get("matched_resources"),
        "configuration_diffs": diffs,
        "mechanism_diff": {k: v for k, v in mechanism.items() if k not in ("history_derived_fields",)},
        "not_run": report.get("not_run"),
        "executed": report.get("executed"),
        "unmatched_factors": report.get("unmatched_factors"),
        "attribution": attribution,
    }


# ---------------------------------------------------------------------------
# pmo
# ---------------------------------------------------------------------------
def cmd_pmo(args: argparse.Namespace) -> int:
    # Absolute: the upstream harness runs with its own checkout as the working
    # directory, so a relative output path would be created inside .external/.
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    required = ("contract_version", "harness", "interpreter", "method", "oracle", "deviations")
    missing = [f for f in required if f not in manifest]
    if missing:
        raise SystemExit(f"the PMO manifest is missing {missing}")

    harness = Path(manifest["harness"])
    if not harness.is_absolute():
        harness = REPO / harness
    interpreter = Path(manifest["interpreter"])
    if not interpreter.is_absolute():
        interpreter = REPO / interpreter

    record: dict[str, Any] = {
        "contract_version": "1.0.0",
        "record_kind": "pmo_smoke_task",
        "secondary_benchmark_note": (
            "Appendix A §3.4: PMO is a SECONDARY optimizer benchmark. It is never a "
            "prerequisite for accepting a ZAK-specific predictor or receptor artifact, and "
            "nothing here is a leaderboard result."
        ),
        "harness": str(harness),
        "interpreter": str(interpreter),
        "method": manifest["method"],
        "oracle": manifest["oracle"],
        "max_oracle_calls": manifest.get("max_oracle_calls"),
        "n_runs": manifest.get("n_runs", 1),
        "seed": manifest.get("seed", 0),
        "deviations": list(manifest["deviations"]),
        "manifest_hash": sha256_text(json.dumps(manifest, sort_keys=True)),
    }

    if not (harness / "run.py").is_file() or not interpreter.is_file():
        record.update(
            {
                "status": "UNAVAILABLE",
                "primary_value": None,
                "reason": (
                    f"the official mol_opt harness is not usable here: "
                    f"{'run.py missing under ' + str(harness) if not (harness / 'run.py').is_file() else ''}"
                    f"{' interpreter ' + str(interpreter) + ' missing' if not interpreter.is_file() else ''}"
                ),
                "unblock": str(manifest.get("unblock") or "clone wenhao-gao/mol_opt and install PyTDC"),
            }
        )
        (output / "pmo.json").write_text(json.dumps(record, indent=2, sort_keys=True))
        print(json.dumps(record, indent=2))
        return 1

    argv = [
        str(interpreter),
        "run.py",
        "--method",
        str(manifest["method"]),
        "--oracles",
        str(manifest["oracle"]),
        "--max_oracle_calls",
        str(manifest.get("max_oracle_calls", 500)),
        "--n_runs",
        str(manifest.get("n_runs", 1)),
        "--seed",
        str(manifest.get("seed", 0)),
        "--wandb",
        "disabled",
        "--output_dir",
        str(output),
    ]
    # Oracle calls are the unit PMO's own sample-efficiency claim is denominated
    # in (Appendix A §3.4), so the declared call budget is reserved in the ledger
    # *before* the harness starts and settled with the calls the run actually
    # made. The subprocess itself goes through `run_accounted`, which gives it its
    # own process session, enforces the wall cap with process-group cancellation
    # and writes a resumable checkpoint.
    ledger = _open_ledger(args, output)
    # Wall-time enforcement is not optional: `--no-ledger` switches off *campaign*
    # accounting, not the cap on this subprocess. A local ledger carrying only
    # this run's wall cap keeps the cancellation path in place either way.
    wall_ledger = ledger
    if wall_ledger is None:
        wall_ledger = BudgetLedger(output / "budget.sqlite", campaign_id="wall-cap-only")
        wall_ledger.declare(
            "wall_seconds", float(args.wall_seconds), unit="seconds", note="this run's cap only"
        )
    declared_calls = float(manifest.get("max_oracle_calls", 500)) * float(manifest.get("n_runs", 1))
    oracle_reservation = None
    if ledger is not None:
        ledger.declare(
            "oracle_calls",
            declared_calls,
            unit="calls",
            note=f"declared by {Path(args.manifest).name}",
        )
        oracle_reservation = ledger.reserve(
            "oracle_calls",
            declared_calls,
            owner=f"PMO/{manifest['method']}/{manifest['oracle']}",
            phase="EVALUATION",
            request_key=f"pmo:{output.name}:{record['manifest_hash']}",
        )
        if oracle_reservation["already_completed"]:
            raise SystemExit(
                f"this PMO task already settled in the ledger at {ledger.path}; re-running it "
                f"into {output} would charge and execute the same oracle calls twice"
            )

    run = run_accounted(
        argv,
        ledger=wall_ledger,
        resource="wall_seconds",
        wall_seconds=float(args.wall_seconds),
        owner=f"PMO/{manifest['method']}/{manifest['oracle']}",
        phase="EVALUATION",
        cwd=str(harness),
        env={**os.environ, "WANDB_MODE": "disabled"},
        log_dir=output,
        gpu=True,
    )
    stdout = Path(run["stdout_path"]).read_text(errors="replace")
    stderr = Path(run["stderr_path"]).read_text(errors="replace")
    (output / "pmo.stdout.log").write_text(stdout)
    (output / "pmo.stderr.log").write_text(stderr)

    metrics: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and "auc_top10" in stripped:
            try:
                metrics = json.loads(stripped.replace("'", '"'))
            except ValueError:
                metrics = None
            if metrics:
                break

    if oracle_reservation is not None and ledger is not None:
        # `n_oracle` is what the harness reports it spent. When it reports
        # nothing -- a crash, a cancellation -- the calls it already made are
        # unknown, so the whole reservation is charged rather than refunded.
        spent = (metrics or {}).get("n_oracle")
        measured = isinstance(spent, (int, float)) and not isinstance(spent, bool)
        ledger.settle(
            oracle_reservation["reservation_id"],
            consumed=float(spent) if measured else None,
            telemetry="MEASURED" if measured else "UNAVAILABLE",
            status="SETTLED" if run["status"] == "SUCCEEDED" and metrics else "FAILED",
            note="" if measured else "the PMO harness reported no n_oracle for this run",
        )

    record.update(
        {
            "argv": argv,
            "exit_status": run["exit_status"],
            "wall_seconds": run["cost"]["wall_seconds"],
            "timed_out": run["timed_out"],
            "status": "SUCCEEDED" if run["exit_status"] == 0 and metrics else "FAILED",
            "primary_metric": "auc_top10",
            "primary_value": (metrics or {}).get("auc_top10"),
            "metrics": metrics,
            "cost": run["cost"],
            "budget": None
            if ledger is None
            else {
                "ledger_path": str(ledger.path),
                "wall_reservation_id": run["reservation_id"],
                "oracle_reservation_id": oracle_reservation["reservation_id"] if oracle_reservation else None,
                "declared_oracle_calls": declared_calls,
                "usage": {name: ledger.usage(name) for name in ("oracle_calls", "wall_seconds")},
            },
            "checkpoint_path": run["checkpoint_path"],
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    )
    if ledger is not None:
        ledger.write_report(output / "ledger.json")
    (output / "pmo.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in record.items() if not k.endswith("_tail")}, indent=2))
    return 0 if record["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# generators — measured availability and throughput for the §3.1 arms
# ---------------------------------------------------------------------------
def cmd_generators(args: argparse.Namespace) -> int:
    """Measure every in-scope generator arm and write the durable record.

    Runs each arm's registered ``sample`` probe — the same tiny computation the
    capability matrix uses — inside wall/CPU/RSS/GPU accounting, and writes
    ``generators.json`` plus a manifest of digests. Nothing generated here is
    scored: the scoring rule is not frozen, and Appendix A §3.3 requires it to be
    registered before candidate search.

    Exit status is 0 whenever the survey completed. An arm that could not run is a
    named limitation inside the record, not a failure of the survey: a backend
    failure is a missing scientific outcome with a consumed cost, never a zero.
    """
    from drug_agent.selfevolve.target.generator_survey import SURVEY_ARMS, run_survey, summary_rows

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    arms = list(args.only) if args.only else list(SURVEY_ARMS)
    unknown = [a for a in arms if a not in SURVEY_ARMS]
    if unknown:
        print(f"not in scope for this survey: {', '.join(unknown)}", file=sys.stderr)
        return 2

    contamination = generator_substitution_check()
    if contamination.get("contaminated"):
        print(f"refusing to survey a contaminated lane: {contamination}", file=sys.stderr)
        return 3

    record = run_survey(arms, workdir=output)
    record["contamination_check"] = contamination
    full = output / "generators.json"
    full.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    rows = summary_rows(record)
    (output / "generators-summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))

    if args.pin:
        # The committed half: the summary table, the digests, the cap arithmetic
        # and a manifest that names the run directory by hash. The per-arm record
        # with its warnings and funnels stays under `.external/`, which is
        # gitignored, because a comparison needs the numbers and the hashes, not
        # the raw output.
        pin = Path(args.pin)
        pin.parent.mkdir(parents=True, exist_ok=True)
        pinned = {
            "contract_version": record["contract_version"],
            "record_kind": "generator_availability_and_throughput_summary",
            "note": record["note"],
            "predecessor_commit": args.predecessor or None,
            "host": record["host"],
            "run": {
                "path": str(full),
                "sha256": sha256_text(full.read_text()),
                "record_hash": record["record_hash"],
                "arms_surveyed": arms,
            },
            "contamination_check": contamination,
            "summary": rows,
            "evidence": {
                arm["capability_id"]: {
                    "evidence": arm["evidence"],
                    "weights": arm["weights"],
                    "upstream_ref": arm["upstream_ref"],
                    "checkout": arm["checkout"],
                    "device": arm["device"],
                    "gpu_kernel": arm["gpu_kernel"],
                    "gpu_kernel_torch": arm["gpu_kernel_torch"],
                    "ref_ligand_sha256": arm["ref_ligand_sha256"],
                    "pocket_center": arm["pocket_center"],
                }
                for arm in record["arms"]
            },
            "admitted_to_sweep": record["admitted_to_sweep"],
            "executed_without_usable_yield": record["executed_without_usable_yield"],
            "did_not_execute": record["did_not_execute"],
            "status_token_note": record["status_token_note"],
            "named_limitations": record["named_limitations"],
            "operator_decisions": record["operator_decisions"],
            "cap_arithmetic": record["cap_arithmetic"],
            "operator_cap_arithmetic": record["operator_cap_arithmetic"],
            "totals": record["totals"],
        }
        pin.write_text(json.dumps(pinned, indent=2, sort_keys=True, default=str) + "\n")

    width = max(len(r["arm"]) for r in rows)
    for row in rows:
        gpu = row["gpu_seconds"]
        gpu_s = f"{gpu:8.1f}" if isinstance(gpu, (int, float)) else f"{row['gpu_telemetry']!s:>8s}"
        rate = row["molecules_per_gpu_second"]
        rate_s = f"{rate:9.4f}" if isinstance(rate, (int, float)) else "        -"
        print(
            f"{row['arm']:<{width}s}  {row['status']:<18s} {row['unique'] or 0!s:>3s} uniq "
            f"{row['wall_seconds']:9.1f}s wall {gpu_s}s gpu {rate_s} mol/gpu-s"
        )
    for key in ("cap_arithmetic", "operator_cap_arithmetic"):
        block = record[key]
        print(
            f"\n{block['label']}: {block['unique_cap']} unique x {block['seeds']} seed(s) "
            f"= {block['unique_molecules_per_arm_total']} unique/arm across "
            f"{block['n_admitted']} admitted arms -> "
            f"{block['total_gpu_seconds_all_admitted_all_seeds']} GPU-s, "
            f"{block['total_hours_all_admitted_all_seeds']} h wall"
        )
    print(json.dumps(record["cap_arithmetic"], indent=2, default=str))
    print(json.dumps(record["operator_cap_arithmetic"], indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# sweep — Stage D's nine-arm generation sweep
# ---------------------------------------------------------------------------
#: The molecules the endpoint-rate probe is timed on. Real ZAK reference
#: chemistry would be a novelty-gate member and a poor proxy for generated
#: output; these are small, ordinary, obviously-not-ZAK structures whose only
#: job is to make the panel do one real forward pass so the estimate rests on a
#: measurement rather than on a guess.
_RATE_PROBE_SMILES = (
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "COc1cc2c(cc1OC)CCN(C)C2",
    "O=C(Nc1ccccc1)c1ccccc1O",
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "Clc1ccccc1C1=NCC(=O)Nc2ccccc21",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "OC(=O)c1ccccc1Nc1c(Cl)cccc1Cl",
    "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "CC1=C(C(=O)Nc2ccccc2)S(=O)(=O)c2ccccc2N1C",
    "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1",
)


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run, score and report Stage D's nine-arm generation sweep.

    The order matters and is the order the work order sets. Nothing is spent
    until the frozen inputs have been verified, the held-out access count has
    been checked, the cost has been estimated from measured rates and the
    estimate has been reserved. A launch that does not fit the remaining
    authorisation is **refused by number** and exits non-zero rather than being
    started and abandoned.

    Exit status: 0 when the sweep ran (a qualifying yield of zero is a completed
    sweep, not a failure), 3 on a contaminated lane, 4 on a budget refusal, 5
    when a frozen input does not verify.
    """
    from drug_agent.selfevolve.target import generation_sweep as sweep_mod

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    contamination = generator_substitution_check()
    if contamination.get("contaminated"):
        print(f"refusing to sweep a contaminated lane: {contamination}", file=sys.stderr)
        return 3

    try:
        qualification = sweep_mod.load_qualification()
        lock = sweep_mod.load_lock(Path(args.lock) if args.lock else None)
    except SystemExit as exc:
        print(f"frozen input did not verify: {exc}", file=sys.stderr)
        return 5

    access = sweep_mod.verify_test_access_count()
    if not access["all_zero"]:
        print(
            f"test_access_count is not 0 everywhere ({access['values']}); this lock is closed",
            file=sys.stderr,
        )
        return 5
    print(
        f"[{time.strftime('%H:%M:%S')}] test_access_count verified 0 across "
        f"{access['n_records']} declaring records",
        flush=True,
    )

    arms = sweep_mod.sweep_arms(qualification)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {a["capability_id"] for a in arms}
        if unknown:
            print(f"not admitted to this sweep: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        arms = [a for a in arms if a["capability_id"] in wanted]
    seeds = [int(s) for s in (args.seeds or sweep_mod.sweep_seeds(lock))]
    caps = sweep_mod.stage_d_caps(qualification)

    # How many usable molecules step 15A measured per molecule *requested*, used
    # only to size how many a cell asks for. It changes no cap and no threshold,
    # and the loop tops up from measurement if a call under-delivers.
    #
    # The denominator is `requested`, not `raw`. Some backends over-deliver -
    # PrexSyn returned 8 raw samples for a request of 4 - so a raw-attempt
    # denominator would make this ask for twice what it needs and pay for it.
    # What this loop controls is the request, so the request is what the rate is
    # stated per.
    hints = {
        str(row["arm"]): (
            float(row["unique"]) / float(row["requested"])
            if row.get("requested") and row.get("unique") is not None
            else None
        )
        for row in (_json_file(CONFIG_DIR / "generator-availability.json").get("summary") or [])
    }

    # If this run directory already holds a scored arm, its own docking rate is
    # better evidence for the rest of this run than the lock's teacher rate. A
    # resumed sweep therefore re-estimates against what it measured, and tops up
    # its reservation before spending the difference.
    dock_rate = sweep_mod.measured_docking_rate(output)
    if dock_rate is not None:
        print(
            f"[{time.strftime('%H:%M:%S')}] measured docking rate from this run: "
            f"{dock_rate['seconds_per_molecule']}s/molecule over {dock_rate['n_molecules']} "
            f"molecules in {', '.join(dock_rate['arms'])} (the lock's teacher rate is "
            f"{sweep_mod.MEASURED_DOCK_SECONDS_PER_MOLECULE:.4f}); GPU occupancy during docking "
            f"{dock_rate['gpu_seconds_per_molecule']}s/molecule "
            f"({dock_rate['gpu_telemetry']}) - occupancy, not GPU work: Vina is CPU-only",
            flush=True,
        )

    # Prefer this run's own measured endpoint rate over the probe. It is a better
    # estimator, and it means a resumed sweep does not load the ADMET-AI panel -
    # and so does not open a CUDA context - before it starts docking.
    rate = sweep_mod.measured_endpoint_rate(output)
    if rate is None:
        rate = sweep_mod.measure_endpoint_rate(_RATE_PROBE_SMILES[: int(args.rate_probe_molecules)])
        rate["source"] = "pre-launch probe; nothing has been scored in this run directory yet"
    print(
        f"[{time.strftime('%H:%M:%S')}] endpoint panel rate: "
        f"{rate['seconds_per_molecule']}s/molecule over {rate['n_probe_molecules']} molecules, "
        f"gpu={rate['gpu_seconds']} ({rate['gpu_telemetry']}) - {rate.get('source', '')}",
        flush=True,
    )
    estimate = sweep_mod.estimate_sweep(
        n_arms=len(arms),
        n_seeds=len(seeds),
        unique_cap=int(caps["unique_molecules_per_arm_per_seed"]),
        endpoint_seconds_per_molecule=rate["seconds_per_molecule"],
        endpoint_gpu_seconds_per_molecule=rate["gpu_seconds_per_molecule"],
        endpoint_rate_evidence=(
            f"measured on this host: {rate['n_probe_molecules']} molecules through the same "
            f"safety.assess + synth.assess path the sweep uses, {rate['wall_seconds']}s wall"
        ),
        posebusters_seconds_per_molecule=(
            None
            if args.no_posebusters
            else float((dock_rate or {}).get("posebusters_seconds_per_molecule") or args.posebusters_rate)
        ),
        docking_seconds_per_molecule=(
            None if dock_rate is None else float(dock_rate["seconds_per_molecule"])
        ),
        docking_gpu_seconds_per_molecule=(
            None if dock_rate is None else dock_rate.get("gpu_seconds_per_molecule")
        ),
        docking_rate_evidence=(
            None
            if dock_rate is None
            else (
                f"measured by this run over {dock_rate['n_molecules']} generated molecules in "
                f"{', '.join(dock_rate['arms'])}: {dock_rate['docking_wall_seconds']}s of docking. "
                + str(dock_rate["why_it_replaces_the_locked_rate"])
            )
        ),
    )
    estimate["endpoint_rate_evidence_record"] = rate
    estimate["docking_rate_probe"] = dock_rate
    _atomic_json(output / "estimate.json", estimate)
    print(
        f"[{time.strftime('%H:%M:%S')}] estimate: generation "
        f"{estimate['generation']['wall_seconds']}s wall / {estimate['generation']['gpu_seconds']}s GPU; "
        f"scoring {estimate['scoring']['wall_seconds']}s wall; total "
        f"{estimate['totals']['wall_seconds']}s ({estimate['totals']['wall_hours']}h) and "
        f"{estimate['totals']['gpu_seconds']}s GPU",
        flush=True,
    )

    ledger = None
    reservation: dict[str, Any] = {"status": "NO_LEDGER", "reason": "--no-ledger was given"}
    if not args.no_ledger:
        from drug_agent.selfevolve.target.budget import BudgetLedger

        ledger_path = Path(args.ledger) if args.ledger else output / "budget.sqlite"
        ledger = BudgetLedger(ledger_path, campaign_id=str(args.campaign_id))
        caps_declared = sweep_mod.declare_caps(
            ledger,
            wall_limit=float(args.wall_hours) * 3600.0,
            gpu_limit=float(args.gpu_hours) * 3600.0,
            dock_calls=float(args.dock_calls),
        )
        reservation = sweep_mod.refuse_or_reserve(
            ledger,
            estimate=estimate,
            owner="step-15D/sweep",
            request_key=f"{output.name}:stage-d-sweep",
        )
        reservation["declared_caps"] = caps_declared
        _atomic_json(output / "reservation.json", reservation)
        if reservation["status"] == "REFUSED_BUDGET":
            print(f"REFUSED_BUDGET: {reservation['reason']}", file=sys.stderr)
            ledger.write_report(output / "ledger.json")
            return 4
        held = reservation.get("total_held") or {
            r: v["reserved"] for r, v in reservation["reservations"].items()
        }
        print(
            f"[{time.strftime('%H:%M:%S')}] reserved "
            + ", ".join(f"{r}={v:,.1f}" for r, v in sorted(held.items()))
            + (
                f" (including top-ups on {', '.join(sorted(reservation['top_ups']))})"
                if reservation.get("top_ups")
                else ""
            ),
            flush=True,
        )

    if args.estimate_only:
        if ledger is not None:
            # Every hold this call took, parents and top-ups alike. Releasing only
            # the parents would leave an estimate-only run silently holding budget.
            taken = list((reservation.get("reservations") or {}).values())
            taken.extend(r for records in (reservation.get("top_ups") or {}).values() for r in records)
            for record in taken:
                with contextlib.suppress(Exception):
                    ledger.release(record["reservation_id"], note="--estimate-only: nothing was run")
            ledger.write_report(output / "ledger.json")
        print(json.dumps({"estimate": estimate, "reservation": reservation}, indent=2, default=str))
        return 0

    manifest = _json_file(DEFAULT_MANIFEST_DIR / "B-B0-pfm-frozen.json")
    record = sweep_mod.run_sweep(
        output=output,
        arms=arms,
        seeds=seeds,
        caps=caps,
        qualification=qualification,
        scorer={"preparation": manifest["preparation"], "scorer": manifest["preparation"]["scorer"]},
        ledger=ledger,
        posebusters=not args.no_posebusters,
        resume=not args.no_resume,
        yield_hints=hints,
        call_timeout_seconds=float(args.call_timeout_seconds),
        cell_wall_seconds=float(args.cell_wall_seconds),
    )
    record["contamination_check"] = contamination
    record["test_access_count_verified"] = {k: v for k, v in access.items() if k != "records_declaring_it"}
    record["predecessor_commit"] = args.predecessor
    _atomic_json(output / "sweep.json", record)

    ledger_report = None
    if ledger is not None:
        measured = sweep_mod.measured_totals(record["arms"])
        _settle_sweep(ledger, reservation, measured, record)
        ledger_report = ledger.report()
        ledger.write_report(output / "ledger.json")

    report = sweep_mod.build_report(
        record,
        estimate=estimate,
        reservation=reservation,
        ledger_report=ledger_report,
        comparator=args.comparator,
    )
    report["contamination_check"] = contamination
    report["test_access_count_verified"] = record["test_access_count_verified"]
    _atomic_json(output / "report.json", report)

    if args.pin:
        pin = Path(args.pin)
        pin.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(pin, _pinned_sweep_record(report, record, output, args.predecessor))
        print(f"[{time.strftime('%H:%M:%S')}] pinned {pin}", flush=True)

    _print_sweep_summary(report)
    return 0


#: `sweep-report` exit codes. 3 is new: the rebuild would have written a report
#: at a different comparator bar from the one the run directory was measured at.
SWEEP_REPORT_NO_SWEEP = 2
SWEEP_REPORT_BAR_CONFLICT = 3


def _bar_a_report_was_written_at(report: Mapping[str, Any], caps: Mapping[str, Any]) -> str | None:
    """Which comparator bar an existing report.json was scored against.

    Read from the report's own `comparator_bar.name` when it has one. A report
    written before that block existed carries only a number, so the per-arm mean
    is matched back against the registry — 1.5625 at the locked caps is the
    retired Stage D null and nothing else. A mean matching no registered bar
    comes back as `UNREGISTERED_LAMBDA`, which is never equal to a bar name and
    therefore never silently overwritten.
    """
    from drug_agent.selfevolve.target import generation_sweep as sweep_mod

    named = (report.get("comparator_bar") or {}).get("name")
    if named:
        return str(named)
    lam = (report.get("primary") or {}).get("baseline_lambda_per_arm")
    if lam is None:
        return None
    for candidate in sorted(sweep_mod.COMPARATOR_BARS):
        implied = sweep_mod.comparator_lambda(sweep_mod.comparator_bar(candidate), caps)
        if abs(implied - float(lam)) <= 1e-12:
            return candidate
    return "UNREGISTERED_LAMBDA"


def cmd_sweep_report(args: argparse.Namespace) -> int:
    """Rebuild Stage D's report from the artifacts already on disk. No GPU.

    Everything this reads was written by the sweep: the per-arm ``scoring.json``
    carries every molecule's ``pose_molblock``, and ``sweep.json`` carries every
    count. **Nothing is generated and nothing is re-docked.** PoseBusters is pure
    CPU over RDKit, so the recomputation this performs costs 0.0 GPU-seconds
    against a sweep that has already consumed 3,893.2 of them.

    It exists because a summary can be wrong about a measurement that was right.
    The sweep's ``pose_quality`` block reported 0 poses passing for all nine arms,
    and that figure was an artifact of counting four boolean columns as checks
    when they are not: three cofactor/water columns that are vacuously false
    against a receptor with no cofactors and no waters, and one inverted-polarity
    detail column. The poses were fine; the arithmetic over them was not.

    **The comparator bar.** `--comparator` defaults to the RETIRED Stage D null,
    because this command exists to rebuild the report those counts were
    delivered against. That default is only defensible for a run directory that
    really was measured at that bar, and it used to be applied to any directory
    at all — and written straight over the `report.json` that `sweep` had
    produced at the registered bar, whose only surviving trace would have been a
    number. So before anything is recomputed:

    * the bar the existing report was written at is read back from it (by name,
      or from its per-arm mean for a report predating the name), and
    * a rebuild at a different bar refuses rather than overwrites; it can be
      redirected with `--report-out`, or asked for the bar the directory was
      actually measured at, and
    * the retired default is not applied to a directory with no report to
      inherit it from: there the bar has to be named.

    Exit status: 0 when the report was rebuilt, 2 when the run directory does not
    hold a completed sweep, 3 when the rebuild would change the bar a report was
    written at.
    """
    from drug_agent.selfevolve.target import generation_sweep as sweep_mod

    output = Path(args.output)
    sweep_path = output / "sweep.json"
    if not sweep_path.is_file():
        print(f"no sweep.json in {output}; nothing to rebuild", file=sys.stderr)
        return SWEEP_REPORT_NO_SWEEP
    record = _json_file(sweep_path)

    report_path = Path(args.report_out) if args.report_out else output / "report.json"
    if not args.arms_only:
        from drug_agent.selfevolve.target.generation_sweep import (
            REGISTERED_COMPARATOR,
            STAGE_D_REGISTERED_NULL,
        )

        caps = dict(record.get("caps") or {})
        existing = _json_file(report_path) if report_path.is_file() else None
        written_at = _bar_a_report_was_written_at(existing, caps) if existing is not None else None
        explicit = bool(getattr(args, "comparator_explicit", False))
        if written_at is not None and written_at != args.comparator:
            print(
                json.dumps(
                    {
                        "status": "COMPARATOR_BAR_CONFLICT",
                        "report": str(report_path),
                        "written_at": written_at,
                        "would_be_rebuilt_at": args.comparator,
                        "reason": (
                            "this report was written at one comparator bar and the rebuild is "
                            "scored at another. Overwriting it would replace a result measured "
                            "against one null with a result measured against a different one, in "
                            "a file of the same name"
                        ),
                        "either": f"--comparator {written_at} to rebuild at the bar it was measured at",
                        "or": "--report-out PATH to write the rebuild somewhere else",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return SWEEP_REPORT_BAR_CONFLICT
        if written_at is None and not explicit and args.comparator == STAGE_D_REGISTERED_NULL:
            print(
                json.dumps(
                    {
                        "status": "RETIRED_BAR_NOT_INHERITED",
                        "report": str(report_path),
                        "default_comparator": args.comparator,
                        "registered_comparator": REGISTERED_COMPARATOR,
                        "reason": (
                            "the default bar for this command is the RETIRED Stage D null, which "
                            "is only defensible when rebuilding a report that was measured "
                            "against it. There is no report here to inherit that bar from, so "
                            "applying it would score this run directory against a retired null "
                            "nobody asked for. Name the bar"
                        ),
                        "either": f"--comparator {REGISTERED_COMPARATOR} for the registered bar",
                        "or": f"--comparator {STAGE_D_REGISTERED_NULL} to say the retired null is meant",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return SWEEP_REPORT_BAR_CONFLICT

    for arm in record.get("arms") or ():
        name = str(arm["arm"])
        if args.only and name not in set(args.only):
            continue
        arm_dir = output / "cells" / name
        scored_path = arm_dir / "scoring.json"
        if not scored_path.is_file():
            print(f"  {name}: no scoring.json, left as it was", flush=True)
            continue
        scored = _json_file(scored_path)
        rows = list(scored.get("rows") or ())
        if not rows:
            continue
        pose_quality = scored.get("pose_quality") or {}
        if "intramolecular_validity" in pose_quality and args.reuse:
            print(f"[{time.strftime('%H:%M:%S')}] {name}: already decomposed, reused", flush=True)
        else:
            receptor = arm_dir / "scoring" / "docking" / "receptor.pdb"
            print(
                f"[{time.strftime('%H:%M:%S')}] {name}: recomputing pose quality over {len(rows)} "
                f"stored molblocks (CPU only, 0.0 GPU-seconds)",
                flush=True,
            )
            started = time.time()
            pose_quality = sweep_mod.recompute_pose_quality(
                rows,
                receptor_pdb=receptor if receptor.is_file() else None,
                workdir=arm_dir / "scoring",
                stored=scored.get("pose_quality"),
                verbose=bool(args.verbose),
                jobs=int(args.jobs),
                mol_mode=not args.no_mol_mode,
            )
            for row in rows:
                row["posebusters"] = (pose_quality.get("reports") or {}).get(str(row["molecule_id"]))
            scored["pose_quality"] = pose_quality
            scored["rows"] = rows
            _atomic_json(scored_path, scored)
            print(f"[{time.strftime('%H:%M:%S')}]   {name}: {time.time() - started:.0f}s CPU", flush=True)

        arm["pose_quality"] = {k: v for k, v in pose_quality.items() if k != "reports"}
        arm["distributions"] = sweep_mod.distributions(rows)
        arm_path = arm_dir / "arm.json"
        if arm_path.is_file():
            stored_arm = _json_file(arm_path)
            stored_arm["pose_quality"] = arm["pose_quality"]
            stored_arm["distributions"] = arm["distributions"]
            _atomic_json(arm_path, stored_arm)
        intra = pose_quality.get("intramolecular_validity") or {}
        print(
            f"[{time.strftime('%H:%M:%S')}]   {name}: intramolecular "
            f"{intra.get('n_passing_every_check')}/{intra.get('n_measured')} "
            f"({intra.get('pass_rate')}), reproduced "
            f"{(pose_quality.get('reproduces_the_recorded_run') or {}).get('identical')}/"
            f"{(pose_quality.get('reproduces_the_recorded_run') or {}).get('compared')} "
            f"of the recorded dock-mode verdicts",
            flush=True,
        )
    if args.arms_only:
        print("--arms-only: per-arm records updated; sweep.json and report.json left alone")
        return 0

    record["pose_quality_recomputed"] = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "from": "the pose_molblock stored on every scoring row",
        "gpu_seconds": 0.0,
        "gpu_telemetry": "MEASURED",
        "why": (
            "the sweep's own pose_quality block counted four boolean columns as checks that are "
            "not PoseBusters tests against this receptor, which made n_passing_all_checks zero by "
            "construction for every arm. No molecule was regenerated and no pose was re-docked"
        ),
    }
    _atomic_json(sweep_path, record)

    def maybe(name: str) -> Any:
        path = output / name
        return _json_file(path) if path.is_file() else None

    report = sweep_mod.build_report(
        record,
        estimate=maybe("estimate.json"),
        reservation=maybe("reservation.json"),
        ledger_report=maybe("ledger.json"),
        comparator=args.comparator,
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] comparator bar: {report['comparator_bar']['name']} "
        f"({report['comparator_bar']['status']}), "
        f"lambda={report['comparator_bar']['baseline_lambda_per_arm']}",
        flush=True,
    )
    for carried in ("contamination_check", "test_access_count_verified", "predecessor_commit"):
        if carried in record:
            report[carried] = record[carried]
    report["pose_quality_recomputed"] = record["pose_quality_recomputed"]
    _atomic_json(report_path, report)

    if args.pin:
        pin = Path(args.pin)
        _atomic_json(pin, _pinned_sweep_record(report, record, output, args.predecessor))
        print(f"[{time.strftime('%H:%M:%S')}] pinned {pin}", flush=True)

    _print_sweep_summary(report)
    _print_pose_quality_summary(report)
    return 0


def _print_pose_quality_summary(report: Mapping[str, Any]) -> None:
    print("")
    print("POSE QUALITY  decomposed; PoseBusters gates nothing")
    print("  intramolecular validity (mode `mol`, the generated-3D-quality number)")
    for arm, block in sorted((report.get("pose_quality") or {}).items()):
        intra = block.get("intramolecular_validity") or {}
        if not intra:
            print(f"    {arm:<14} {block.get('status')}")
            continue
        print(
            f"    {arm:<14} {intra.get('n_passing_every_check')}/{intra.get('n_measured')} poses "
            f"clean ({intra.get('pass_rate')}), {block.get('n_no_pose')} molecules returned no pose"
        )
    first = next(iter((report.get("pose_quality") or {}).values()), {})
    vacuous = sorted((first.get("protein_context") or {}).get("vacuous_checks") or {})
    non_tests = sorted(
        (first.get("protein_context") or {}).get("columns_that_are_not_posebusters_tests") or {}
    )
    census = (first.get("protein_context") or {}).get("receptor_census") or {}
    print(f"  receptor entity census: {census.get('entities')}")
    print(f"  vacuous against this receptor ({len(vacuous)}): {', '.join(vacuous)}")
    print(f"  counted but not PoseBusters tests ({len(non_tests)}): {', '.join(non_tests)}")


def _settle_sweep(
    ledger: Any,
    reservation: Mapping[str, Any],
    measured: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Close the sweep's reservations with what it actually spent.

    A resource may be held by a parent reservation plus one or more top-ups taken
    when the estimate was corrected upwards mid-run. The measured consumption is
    applied down that chain in order, so each hold is settled at the part of the
    spend it actually covered and an unused top-up settles at a measured zero
    rather than being charged. The last hold in the chain absorbs any excess, so
    an overrun is recorded as an overrun rather than quietly disappearing.

    ``consumed=None`` with ``telemetry="UNAVAILABLE"`` is the honest record for a
    resource nothing could measure: the whole reservation is charged, because an
    unmeasurable job is not a free job, and the cost row keeps a null amount so
    the total never claims a number it does not have.
    """
    held = dict(reservation.get("reservations") or {})
    top_ups = dict(reservation.get("top_ups") or {})

    def chain(resource: str) -> list[Mapping[str, Any]]:
        parent = held.get(resource)
        return [] if parent is None else [parent, *list(top_ups.get(resource) or ())]

    def settle_chain(resource: str, consumed: float, note: str) -> None:
        # Only a hold that is still RESERVED can absorb a settlement; a released or
        # already-settled row would swallow the consumption and report nothing.
        links = [link for link in chain(resource) if link.get("state") == "RESERVED"]
        remaining = float(consumed)
        for index, link in enumerate(links):
            reserved = float(link.get("reserved") or 0.0)
            take = remaining if index == len(links) - 1 else min(remaining, reserved)
            with contextlib.suppress(Exception):
                ledger.settle(link["reservation_id"], consumed=max(0.0, take), note=note)
            remaining = max(0.0, remaining - take)

    settle_chain(
        "wall_seconds",
        float(measured["total_wall_seconds"]),
        "measured wall across generation and scoring",
    )
    settle_chain(
        "dock_calls",
        float(measured["n_molecules_scored"]),
        "one docking call per unique molecule submitted",
    )

    gpu_total = measured.get("total_gpu_seconds")
    if gpu_total is None:
        for link in (link for link in chain("gpu_seconds") if link.get("state") == "RESERVED"):
            with contextlib.suppress(Exception):
                ledger.settle(
                    link["reservation_id"],
                    consumed=None,
                    telemetry="UNAVAILABLE",
                    note=(
                        "at least one cell's GPU telemetry was unavailable, so the sweep's GPU "
                        "total is unknown rather than zero"
                    ),
                )
    else:
        settle_chain("gpu_seconds", float(gpu_total), "measured GPU seconds")

    for arm in record.get("arms") or ():
        for outcome in arm.get("backend_outcomes") or ():
            with contextlib.suppress(Exception):
                ledger.intervene(
                    "BACKEND_OUTCOME",
                    {
                        "arm": arm["arm"],
                        "seed": outcome["seed"],
                        "status": outcome["status"],
                        "consumed_wall_seconds": outcome.get("consumed_wall_seconds"),
                        "consumed_gpu_seconds": outcome.get("consumed_gpu_seconds"),
                        "gpu_telemetry": outcome.get("gpu_telemetry"),
                    },
                )
        if arm.get("scoring_failed"):
            with contextlib.suppress(Exception):
                ledger.intervene(
                    "SCORING_FAILED",
                    {
                        "arm": arm["arm"],
                        "reason": arm.get("scoring_failure_reason"),
                        "n_molecules_not_scored": arm.get("n_molecules_not_scored"),
                    },
                )


def _pinned_sweep_record(
    report: Mapping[str, Any],
    sweep: Mapping[str, Any],
    output: Path,
    predecessor: str | None,
) -> dict[str, Any]:
    """The committed half: the tables and the hashes, not the raw output.

    Raw poses, per-molecule rows and driver logs stay under ``.external/``, which
    is gitignored. What is committed is what a comparison needs — the counts, the
    funnels, the distributions, the correction and the digests that name the run.
    """
    payload = json.dumps(sweep, sort_keys=True, default=str).encode()
    return {
        "contract_version": report["contract_version"],
        "record_kind": "stage_d_generation_sweep_summary",
        "predecessor_commit": predecessor,
        "run_directory": str(output),
        "run_hash": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "classification": report["classification"],
        "classification_reason": report["classification_reason"],
        "protocol_version": report["protocol_version"],
        "protocol_hash": report["protocol_hash"],
        "qualification_hash": report["qualification_hash"],
        "derived_qualification_hash": report["derived_qualification_hash"],
        "evaluator_hash": report["evaluator_hash"],
        "caps": report["caps"],
        "seeds": report["seeds"],
        "test_access_count_verified": report["test_access_count_verified"],
        "primary": report["primary"],
        "declared_secondary_rate": report["declared_secondary_rate"],
        "secondary_pooled": report["secondary_pooled"],
        "raw_qualifying_compounds": report["raw_qualifying_compounds"],
        "generation_funnel": report["generation_funnel"],
        "unconstrained_distributions": report["unconstrained_distributions"],
        "novelty_and_nearest_neighbour": report["novelty_and_nearest_neighbour"],
        "pose_quality": report["pose_quality"],
        "pose_quality_recomputed": report.get("pose_quality_recomputed"),
        "backend_outcomes": report["backend_outcomes"],
        "cost": report["cost"],
        "limits": report["limits"],
    }


def _print_sweep_summary(report: Mapping[str, Any]) -> None:
    primary = report["primary"]
    print("")
    print(f"PRIMARY  qualifying_scaffolds, stratified by conditioning class ({report['classification']})")
    print(f"         known-active pass rate through this same bar: {primary['known_active_pass_rate']}")
    for stratum, rows in primary["strata"].items():
        print(f"  [{stratum}]")
        for row in rows:
            print(
                f"    {row['arm']:<14} {row['qualifying_scaffolds']:>3} scaffolds  "
                f"{row['n_qualifying_compounds']:>3} compounds  "
                f"{row['n_submitted']:>4} submitted  per-seed {row['per_seed_qualifying_scaffolds']}  "
                f"p={row['p_value']:.4f} holm_adj={row['holm_adjusted_p']:.4f} "
                f"{'REJECTED_NULL' if row['rejected'] else 'not rejected'}"
            )
    holm = primary["holm"]
    print(
        f"  Holm across {holm['family_size']} arms at FWER {holm['family_wise_alpha']}: "
        f"{primary['n_arms_rejecting_the_null_under_holm']} rejected"
    )
    measured = report["cost"]["measured"]
    print(
        f"COST     measured {measured['total_wall_seconds']}s wall, "
        f"{measured['total_gpu_seconds']}s GPU ({measured['total_gpu_telemetry']}), "
        f"{measured['n_molecules_scored']} molecules scored"
    )


#: `report` exit codes. A package is written whenever one can be written at all
#: — a package that refuses to describe a broken reference cannot report one —
#: so the non-zero codes are for the conditions under which writing it would be
#: a lie rather than for the conditions under which it is incomplete.
REPORT_WRITTEN = 0
REPORT_STUDY_ABSENT = 2
REPORT_LOCK_CLOSED = 3
REPORT_DANGLING_REFERENCE = 4
REPORT_LABEL_LEAK = 5

#: `reproduce` exit codes.
REPRODUCE_OK = 0
REPRODUCE_NO_BUNDLE = 2
REPRODUCE_DISAGREEMENT = 4


DEFAULT_STUDY_DIR = REPO / ".external" / "research" / "zak-c9r" / "derived"


def cmd_report(args: argparse.Namespace) -> int:
    """Assemble the research evidence package.

    The order below is the contract.

    1. `test_access_count` is re-verified across the whole tree by the project's
       own scanner, *before* anything is written. A package assembled under a
       lock that has already spent the held-out partition would describe a
       different study from the one its records claim, and this command refuses
       rather than stamping a date on it.
    2. The study directory is read. Every record that is absent is recorded as
       absent; none of them stops the package, because a package that cannot
       describe a missing record cannot report one.
    3. Every table is built from a declared denominator and checked against it.
       `NO_IMPROVEMENT`, `BLOCKED`, `FAILED` and `NOT_RUN` rows are in the
       tables; an absent metric is `MISSING` and never 0.
    4. Every reference the package emits is resolved. A dangling one is reported
       *in the package* and then exits 4, so the evidence of the break travels
       with the break.
    5. The written package is scanned for per-compound identities. A leak exits
       5 — after the scan has named what leaked, so the operator can see it.
    """
    from drug_agent.selfevolve.target import reporting
    from drug_agent.selfevolve.target.generation_sweep import verify_test_access_count

    study = Path(args.study)
    if not study.is_dir():
        print(
            json.dumps(
                {
                    "status": "STUDY_ABSENT",
                    "study": str(study),
                    "reason": (
                        "the study directory holding the derived records is not there. Nothing "
                        "was written: a package built from no records would report no results "
                        "and look identical to a study that produced none"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return REPORT_STUDY_ABSENT

    access = verify_test_access_count()
    if not access["all_zero"]:
        print(
            json.dumps(
                {
                    "status": "LOCK_CLOSED",
                    "n_records_declaring_test_access_count": access["n_records"],
                    "values": access["values"],
                    "reason": (
                        "a record in this tree declares a non-zero test_access_count. The package "
                        "this command would write describes an EXPLORATORY study whose held-out "
                        "partition was never read, and that is no longer what the tree says"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return REPORT_LOCK_CLOSED

    package = reporting.build_package(study, Path(args.output))

    pinned: str | None = None
    if args.pin:
        pin_path = Path(args.pin)
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        pin_path.write_text(json.dumps(reporting.compact_record(package), indent=2, sort_keys=True) + "\n")
        pinned = str(pin_path)

    summary = {
        "status": "PACKAGE_WRITTEN",
        "output": package["output_directory"],
        "classification": package["classification"],
        "test_access_count": package["held_out_partition"]["test_access_count"],
        "n_tables": len(package["tables"]),
        "n_figures": sum(1 for f in package["figures"] if f.get("status") == "WRITTEN"),
        "n_raw_exports": len(package["raw"]),
        "references_checked": package["reference_validation"]["n_references_checked"],
        "dangling_references": package["reference_validation"]["n_dangling"],
        "recomputation": {
            name: check.get("reproduces")
            for name, check in package["independent_recomputation"]["crosschecks"].items()
        },
        "compound_identities_in_the_package": package["label_safety"]["compound_identities_in_the_package"],
        "manifest_sha256": package["manifest_sha256"],
        "pinned": pinned,
    }
    print(json.dumps(summary, indent=2))

    if not package["label_safety"]["clean"]:
        print(
            "REFUSED: the package contains per-compound identities; see label_safety in the "
            "manifest. Counts and statistics only.",
            file=sys.stderr,
        )
        return REPORT_LABEL_LEAK
    if package["reference_validation"]["n_dangling"]:
        print(
            json.dumps(package["reference_validation"]["dangling"], indent=2),
            file=sys.stderr,
        )
        return REPORT_DANGLING_REFERENCE
    return REPORT_WRITTEN


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Recompute the package's numbers, or rerun its declared scientific subset.

    ``--mode metrics`` is the default and reruns **no** costly scientific tool:
    it re-reads the recorded raw results, rebuilds every table, independently
    recounts the qualifying-scaffold yield, re-adds the cost totals and
    recomputes the Track A mean absolute errors, then compares all of it with
    what the package recorded. ``--mode scientific`` reruns the subset declared
    in ``reporting.SCIENTIFIC_SUBSET`` and reports every member it cannot run as
    ``NOT_RUN`` with its reason, its measured cost and its unblock.
    """
    from drug_agent.selfevolve.target import reporting
    from drug_agent.selfevolve.target.contracts import ContractError

    bundle = Path(args.bundle)
    try:
        if args.mode == "scientific":
            result = reporting.reproduce_scientific(
                bundle,
                study=args.study,
                allow_regeneration=bool(args.allow_regeneration),
            )
        else:
            result = reporting.reproduce_metrics(bundle, study=args.study)
    except ContractError as error:
        print(
            json.dumps(
                {"status": "NO_BUNDLE", "category": error.category.value, "detail": str(error)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return REPRODUCE_NO_BUNDLE

    output = Path(args.output) if args.output else bundle / f"reproduction-{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.mode == "scientific":
        print(
            json.dumps(
                {
                    "status": "SCIENTIFIC_SUBSET_RERUN",
                    "declared_subset_size": result["declared_subset_size"],
                    "n_rerun": result["n_rerun"],
                    "n_not_run": result["n_not_run"],
                    "all_rerun_members_agree": result["all_rerun_members_agree"],
                    "not_run": [
                        {"id": entry["id"], "because": entry.get("still_not_run_because")}
                        for entry in result["results"]
                        if entry.get("status") == "NOT_RUN"
                    ],
                    "written": str(output),
                },
                indent=2,
            )
        )
        return REPRODUCE_OK if result["all_rerun_members_agree"] else REPRODUCE_DISAGREEMENT

    shipped = result["shipped_files"]
    print(
        json.dumps(
            {
                "status": "METRICS_RECOMPUTED",
                "reran_any_costly_scientific_tool": result["reran_any_costly_scientific_tool"],
                "n_tables": result["n_tables"],
                "n_tables_agreeing": result["n_tables_agreeing"],
                "all_tables_reproduce": result["all_tables_reproduce"],
                # The block below is a DETERMINISM check: the same functions over
                # the same files in a second process, not a reimplementation.
                "recomputation_class": result["independent_recomputation"]["class"],
                "crosschecks": {
                    name: check.get("reproduces")
                    for name, check in result["independent_recomputation"]["crosschecks"].items()
                },
                "all_crosschecks_reproduce": result["all_crosschecks_reproduce"],
                # This command used to exit 0 over a bundle whose published CSVs
                # had been falsified and whose declared figure and raw export had
                # been deleted: it validated tables.json only and never opened a
                # shipped file or checked a declared address.
                "declared_files_verified": f"{shipped['n_verified']}/{shipped['n_declared']}",
                "all_declared_files_verify": shipped["all_declared_files_verify"],
                "declared_file_failures": [
                    {"kind": f["kind"], "path": f["path"], "status": f["status"]}
                    for f in shipped["failures"]
                ],
                "what_this_mode_did_NOT_check": result["what_this_mode_did_NOT_check"],
                "unreachable_sources": result["unreachable_sources"],
                "written": str(output),
            },
            indent=2,
        )
    )
    if not (
        result["all_tables_reproduce"]
        and result["all_crosschecks_reproduce"]
        and result["all_declared_files_verify"]
    ):
        return REPRODUCE_DISAGREEMENT
    return REPRODUCE_OK


def main(argv: list[str] | None = None) -> int:
    # Imported here rather than typed as literals: the CLI's comparator choices
    # and its default ARE the registry, so a bar added or retired in
    # `generation_sweep` reaches the command line without anyone remembering to
    # copy a string. Defect 1 of the step-15E gate review was exactly a bar that
    # existed in the record and not in the code.
    from drug_agent.selfevolve.target.generation_sweep import (
        COMPARATOR_BARS,
        REGISTERED_COMPARATOR,
        STAGE_D_REGISTERED_NULL,
    )

    _COMPARATOR_CHOICES = sorted(COMPARATOR_BARS)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    baseline = sub.add_parser("baseline", help="run baseline manifests")
    baseline.add_argument("--manifest", default=str(DEFAULT_MANIFEST_DIR))
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--only", nargs="*", default=None, help="run only these baseline ids")
    baseline.add_argument("--wall-seconds", type=float, default=3600.0)
    baseline.add_argument("--cpu-seconds", type=int, default=14400)
    baseline.add_argument("--skip-capability-probes", action="store_true")
    baseline.add_argument(
        "--ledger", default=None, help="budget ledger path (default: <output>/budget.sqlite)"
    )
    baseline.add_argument("--campaign-id", default="zak-c9r-baselines")
    baseline.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="narrow the campaign wall cap for a bounded probe; it can never widen it",
    )
    baseline.add_argument(
        "--no-ledger",
        action="store_true",
        help="run without budget accounting (Step 8 behaviour; recorded as PROVISIONAL_MANUAL)",
    )
    baseline.set_defaults(func=cmd_baseline)

    generators = sub.add_parser(
        "generators", help="measure availability and throughput for the in-scope generator arms"
    )
    generators.add_argument("--output", required=True)
    generators.add_argument("--only", nargs="*", default=None, help="survey only these capability ids")
    generators.add_argument("--pin", default=None, help="also write the committed compact summary here")
    generators.add_argument("--predecessor", default=None, help="commit the survey was measured at")
    generators.set_defaults(func=cmd_generators)

    sweep = sub.add_parser("sweep", help="run Stage D's nine-arm generation sweep")
    sweep.add_argument("--output", required=True)
    sweep.add_argument("--only", nargs="*", default=None, help="sweep only these capability ids")
    sweep.add_argument("--seeds", nargs="*", type=int, default=None, help="override the lock's seeds")
    sweep.add_argument("--lock", default=None, help="protocol lock to verify against")
    sweep.add_argument("--estimate-only", action="store_true", help="estimate and reserve, run nothing")
    sweep.add_argument("--ledger", default=None)
    sweep.add_argument("--campaign-id", default="zak-c9r-stage-d")
    sweep.add_argument("--no-ledger", action="store_true")
    sweep.add_argument("--wall-hours", type=float, default=24.0)
    sweep.add_argument("--gpu-hours", type=float, default=12.0)
    sweep.add_argument("--dock-calls", type=float, default=2000.0)
    sweep.add_argument("--no-posebusters", action="store_true")
    sweep.add_argument(
        "--posebusters-rate",
        type=float,
        default=0.6,
        help="seconds per pose, for the pre-launch estimate only",
    )
    sweep.add_argument("--no-resume", action="store_true", help="ignore existing per-cell checkpoints")
    sweep.add_argument("--rate-probe-molecules", type=int, default=12)
    sweep.add_argument("--call-timeout-seconds", type=float, default=2400.0)
    sweep.add_argument("--cell-wall-seconds", type=float, default=5400.0)
    sweep.add_argument("--pin", default=None, help="also write the committed compact summary here")
    sweep.add_argument("--predecessor", default=None, help="commit the sweep was measured at")
    sweep.add_argument(
        "--comparator",
        default=REGISTERED_COMPARATOR,
        choices=_COMPARATOR_CHOICES,
        help=(
            "the registered comparator bar the arms are scored against. Default is the bar step "
            "15E registered (4 qualifying scaffolds in 300 molecules under seven gates); the "
            "retired Stage D null is reachable only by naming it"
        ),
    )
    sweep.set_defaults(func=cmd_sweep)

    sweep_report = sub.add_parser(
        "sweep-report",
        help="rebuild Stage D's report from artifacts already on disk (CPU only, no GPU, no re-run)",
    )
    sweep_report.add_argument("--output", required=True)
    sweep_report.add_argument("--only", nargs="*", default=None, help="rebuild only these arms")
    sweep_report.add_argument("--verbose", action="store_true")
    sweep_report.add_argument(
        "--no-mol-mode",
        action="store_true",
        help=(
            "take the intramolecular battery from the dock-mode invocation instead of running "
            "`mol` as well. Identical modules and identical parameters, and far cheaper: mol.yml "
            "declares energy_ratio twice and a failed conformer ensemble is not cached"
        ),
    )
    sweep_report.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker processes for the pose recomputation; changes no verdict, only wall clock",
    )
    sweep_report.add_argument(
        "--reuse", action="store_true", help="keep a per-arm decomposition that is already on disk"
    )
    sweep_report.add_argument(
        "--arms-only",
        action="store_true",
        help="update the per-arm records only; leave sweep.json and report.json alone",
    )
    sweep_report.add_argument("--pin", default=None, help="also write the committed compact summary here")
    sweep_report.add_argument("--predecessor", default=None)
    sweep_report.add_argument(
        "--report-out",
        default=None,
        help=(
            "write the rebuilt report here instead of <output>/report.json. The way to rebuild at "
            "a bar the existing report was not written at without overwriting it"
        ),
    )
    sweep_report.add_argument(
        "--comparator",
        action=_ExplicitValue,
        default=STAGE_D_REGISTERED_NULL,
        choices=_COMPARATOR_CHOICES,
        help=(
            "the registered comparator bar to rebuild against. This command exists to rebuild "
            "STAGE D's report, which was measured against the null registered before any arm ran, "
            "so its default is that retired null DELIBERATELY - rebuilding a delivered result "
            "against a bar registered afterwards would be a post-hoc re-derivation. Pass "
            "`--comparator step-15E-superseding` to score the same counts against the current bar. "
            "The retired default is applied only to a run directory whose own report was already "
            "written at it; anywhere else the bar must be named, and a rebuild never overwrites a "
            "report written at a different bar"
        ),
    )
    sweep_report.set_defaults(func=cmd_sweep_report, comparator_explicit=False)

    capabilities = sub.add_parser("capabilities", help="probe each capability verb separately")
    capabilities.add_argument("--output", required=True)
    capabilities.set_defaults(func=cmd_capabilities)

    pin_cmd = sub.add_parser("pin", help="write the committed compact reference record")
    pin_cmd.add_argument("--study", required=True)
    pin_cmd.add_argument("--capabilities", default=None)
    pin_cmd.add_argument("--pmo", default=None)
    pin_cmd.add_argument(
        "--output", default=str(REPO / "configs" / "research" / "zak-c9r" / "baseline-results.json")
    )
    pin_cmd.set_defaults(func=cmd_pin)

    pilot = sub.add_parser("pilot", help="assess development-pilot feasibility, variance and power")
    pilot.add_argument("--manifest", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument(
        "--run-new",
        action="store_true",
        help="launch any manifest campaign with no terminal record (spends real reasoning tokens)",
    )
    pilot.add_argument("--max-new-campaigns", type=int, default=3, help="Appendix A section 5's pilot cap")
    pilot.add_argument("--max-cycles", type=int, default=3)
    pilot.add_argument("--per-cycle-output-tokens", type=float, default=20899.0)
    pilot.add_argument("--per-campaign-output-tokens", type=float, default=71526.0)
    pilot.add_argument("--campaign-wall-seconds", type=float, default=86400.0)
    pilot.add_argument(
        "--authorised-output-tokens",
        type=float,
        default=175000.0,
        help="the development cap plus any operator-authorised increment",
    )
    pilot.add_argument(
        "--spent-output-tokens",
        type=float,
        default=181319.0,
        help="measured consumption to date; a campaign is refused rather than abandoned mid-run",
    )
    pilot.add_argument("--locked-n", type=int, default=5, help="confirmation campaigns the lock declares")
    pilot.add_argument("--min-power", type=float, default=0.8)
    pilot.set_defaults(func=cmd_pilot)

    freeze = sub.add_parser("freeze", help="validate, canonicalise and hash the protocol lock")
    freeze.add_argument("--protocol", required=True)
    freeze.add_argument("--pilot", default=None)
    freeze.add_argument(
        "--output",
        default=str(REPO / ".external" / "research" / "zak-c9r" / "protocol.lock.json"),
    )
    freeze.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing unexposed lock; it can never overwrite an exposed one",
    )
    freeze.set_defaults(func=cmd_freeze)

    confirm = sub.add_parser(
        "confirm",
        help="execute the locked confirmation, or refuse and record the branch as NOT_RUN",
    )
    confirm.add_argument(
        "--protocol",
        required=True,
        help=(
            "the frozen lock or the protocol source. Either way it is AUTHENTICATED against the "
            "registered content address before any of it is trusted: a document that agrees with "
            "its own stored hash is self-consistent, not authentic"
        ),
    )
    confirm.add_argument("--output", required=True)
    confirm.add_argument("--track", default="A", help="the registered track being confirmed")
    confirm.add_argument(
        "--nominations",
        default=None,
        help="nomination ledger path (default: <output>/nominations.json); resumed, never duplicated",
    )
    confirm.add_argument(
        "--baselines",
        default=str(DEFAULT_MANIFEST_DIR),
        help=(
            "baseline manifests the locked comparator addresses are RE-MEASURED from. The "
            "address is re-derived from the manifest bytes, not read back out of a pinned record. "
            "An empty value is REFUSED rather than treated as `no baselines`: a safety check that "
            "can be switched off by passing nothing is not a safety check"
        ),
    )
    confirm.add_argument(
        "--authorisation",
        default=None,
        help="operator authorisation record covering the locked design's token shortfall",
    )
    confirm.add_argument(
        "--assert-hash",
        action="append",
        default=None,
        metavar="FIELD=ADDRESS",
        help=(
            "REFUSED for every field. An address typed here is a claim, not a measurement, and "
            "preflight would compare it against the protocol's own field - the circular "
            "self-comparison removed as deviation 7. The flag is kept so the refusal is "
            "discoverable rather than the argument being silently ignored"
        ),
    )
    confirm.add_argument(
        "--sealed",
        action="store_true",
        help=(
            "describe this run as sealed. Refused while the lock's evaluation_exposure.sealed is "
            "false: a run on this host is not sealed and may not be called one"
        ),
    )
    confirm.add_argument(
        "--release-held-out",
        action="store_true",
        help=(
            "explicitly ask to spend the held-out partition. NECESSARY and nowhere near "
            "sufficient: every other gate condition is derived from the lock and the nomination "
            "ledger, so this flag can never open the gate on its own"
        ),
    )
    confirm.add_argument(
        "--pin",
        default=None,
        help="also write the committed compact branch-status record here",
    )
    confirm.set_defaults(func=cmd_confirm)

    ablate = sub.add_parser(
        "ablate",
        help="register the ablation comparisons, machine-diff the arm manifests and report attribution",
    )
    ablate.add_argument("--protocol", type=Path, default=CONFIG_DIR / "protocol.json")
    ablate.add_argument("--manifests", type=Path, default=DEFAULT_ABLATION_DIR)
    ablate.add_argument("--output", type=Path, required=True)
    ablate.add_argument(
        "--histories",
        type=Path,
        default=DEFAULT_CAMPAIGN_DIR,
        help="campaign directory whose history.jsonl files the mechanism diff runs against",
    )
    ablate.add_argument("--distill-ablation", type=Path, help="Step 12 ablation.json")
    ablate.add_argument("--distill-repeats", type=Path, help="Step 12 ablation-repeats/repeated.json")
    ablate.add_argument("--campaign", type=Path, help="a terminal campaign.json whose recipe is reverted")
    ablate.add_argument("--sweep-report", type=Path, help="the Stage D sweep report.json")
    ablate.add_argument("--pin", type=Path, help="write the committed compact status record here")
    ablate.add_argument(
        "--allow-regeneration",
        action="store_true",
        help=(
            "the operator permits this run to generate, dock or fit. Absent, every arm that needs "
            "new computation reports EXECUTION_WITHOUT_REGENERATION unmet and stays NOT_RUN"
        ),
    )
    ablate.add_argument(
        "--require-executed-arm",
        action="store_true",
        help="exit ABLATE_NO_ARM_EXECUTED when no predeclared arm cleared its gate",
    )
    ablate.set_defaults(func=cmd_ablate)

    report = sub.add_parser(
        "report",
        help="assemble the research evidence package from the recorded study",
    )
    report.add_argument(
        "--study",
        default=str(DEFAULT_STUDY_DIR),
        help="the derived-record root the package is built from",
    )
    report.add_argument("--output", required=True, help="where to write the package")
    report.add_argument(
        "--pin",
        default=None,
        help="also write the committed compact index (addresses, claims, blockers, verdicts)",
    )
    report.set_defaults(func=cmd_report)

    reproduce = sub.add_parser(
        "reproduce",
        help="recompute a package's numbers, or rerun its declared scientific subset",
    )
    reproduce.add_argument("--bundle", required=True, help="a package written by `report`")
    reproduce.add_argument(
        "--mode",
        choices=("metrics", "scientific"),
        default="metrics",
        help=(
            "metrics (default) recomputes tables and statistics from recorded raw results and "
            "reruns no costly scientific tool; scientific reruns the declared subset"
        ),
    )
    reproduce.add_argument(
        "--study",
        default=None,
        help="override the study directory the package recorded (a relocated custodian store)",
    )
    reproduce.add_argument("--output", default=None, help="where to write the reproduction record")
    reproduce.add_argument(
        "--allow-regeneration",
        action="store_true",
        help=(
            "the operator permits this run to generate, dock or fit. Absent, every scientific "
            "subset member needing new computation reports NOT_RUN; present, they still do, on "
            "their other conditions"
        ),
    )
    reproduce.set_defaults(func=cmd_reproduce)

    pmo = sub.add_parser("pmo", help="run one official-protocol PMO task")
    pmo.add_argument("--manifest", required=True)
    pmo.add_argument("--output", required=True)
    pmo.add_argument("--wall-seconds", type=float, default=3600.0)
    pmo.add_argument("--ledger", default=None, help="budget ledger path (default: <output>/budget.sqlite)")
    pmo.add_argument("--campaign-id", default="zak-c9r-pmo")
    pmo.add_argument("--wall-hours", type=float, default=0.0)
    pmo.add_argument("--no-ledger", action="store_true", help="run without budget accounting")
    pmo.set_defaults(func=cmd_pmo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
