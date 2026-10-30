"""What every generator on this host costs, measured, before a sweep is sized.

Operator plan v1.1 §3.1 asks for an eleven-arm generation comparison. Before any
GPU is committed to one, somebody has to be able to answer three questions with
numbers rather than with a catalogue entry:

1. **which arms execute here at all**, with real evidence rather than an import;
2. **what one molecule costs** on each of them, in GPU-seconds and wall-seconds;
3. **whether a single cap is reachable for every arm**, because Appendix A §3.3
   requires identical caps and forbids reporting a timeout as a fixed-budget
   yield of zero.

This module answers them by running each arm's registered ``sample`` probe —
:func:`~drug_agent.selfevolve.target.baselines.probe_capability`, the same tiny
computation the capability matrix uses, not a second sampling path — inside full
cost accounting, and by writing down what came out.

**It scores nothing.** No quality metric, no ranking, no qualification count. The
scoring rule is not frozen and Appendix A §3.3 requires it to be registered before
candidate search, so a quality number produced here would be the forbidden
ordering. What is recorded is a funnel (requested, raw, valid, unique), a cost, a
conditioning label and a set of digests. A reader can tell from the record alone
that arm A saw the pocket and arm B never did, which is the thing a later
comparison table most easily hides.

**GPU seconds are measured or explicitly absent.** The sampler polls
``nvidia-smi`` for compute processes beneath this process, because every external
generator's driver runs in a *detached* session and a process-group test would
score a busy 3 GB driver as 0.0 GPU-seconds. ``telemetry: "UNAVAILABLE"`` and
``gpu_seconds: 0.0`` mean different things and are reported as themselves.

**A failure is an outcome.** An arm that cannot run is recorded with its measured
reason, its unblock and the cost it consumed getting there. It is never a zero and
never a gap in the table.
"""

from __future__ import annotations

import json
import os
import platform
import resource as _resource
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .baselines import (
    GENERATOR_PROBES,
    Record,
    probe_capability,
    sha256_text,
)
from .budget import gpu_accounting, gpu_telemetry

#: The eleven arms operator plan v1.1 §3.1 puts in scope: the two Step 8 already
#: measured, plus the nine that had no probe. The other ten generators in the
#: catalogue keep their recorded reasons and are deliberately not touched here.
SURVEY_ARMS: tuple[str, ...] = (
    "molecule.generate.pfm",
    "molecule.generate.reinvent4",
    *tuple(str(entry["id"]) for entry in GENERATOR_PROBES),
)

#: Appendix A §3.3's ceiling for the first generation comparison, per arm per
#: seed. A ceiling, not a target: this module reports what it would cost and
#: leaves the choice to the operator.
APPENDIX_A_RAW_ATTEMPT_CAP = 2000
APPENDIX_A_UNIQUE_CAP = 200

#: The cap the operator chose for Stage D, recorded here so the arithmetic that
#: justifies it and the arithmetic that will be spent against it are in the same
#: record. **Fewer molecules, more seeds**: the frozen protocol makes the
#: deployment seed the replication unit and says in terms that the molecule is
#: not, so seeds buy inference and molecules do not. The Appendix A ceiling
#: (200 unique, one seed) is *not* deleted — it is what makes this choice legible
#: as a choice rather than a limit.
OPERATOR_UNIQUE_CAP = 50
OPERATOR_SEEDS = 3

#: Decisions taken by the operator on this survey's evidence, recorded verbatim
#: so a later step reads the decision rather than re-deriving it — and, in the
#: first case, so nobody re-opens a question that was closed on purpose.
OPERATOR_DECISIONS: tuple[Record, ...] = (
    {
        "id": "heavy_atom_window",
        "decision": "KEEP THE 10-70 HEAVY-ATOM WINDOW FROZEN",
        "taken_by": "operator",
        "reason": (
            "The window was declared before candidate search. Lowering it after seeing that one "
            "arm produces fragment-scale output is the retrofit Appendix A §3.3 forbids, and "
            "6-9 heavy atoms is fragment scale, not drug scale."
        ),
        "consequence": (
            "Pocket2Mol stands as a measured yield of zero from a working backend, with its "
            "135.8 consumed GPU-seconds on the record and status EXECUTED_NO_YIELD rather than "
            "UNAVAILABLE."
        ),
    },
    {
        "id": "sweep_cap",
        "decision": "STAGE D SWEEP CAP: 50 unique molecules per arm per seed, 3 seeds",
        "taken_by": "operator",
        "reason": (
            "The frozen protocol makes the deployment seed the replication unit and states "
            "explicitly that the molecule is not, so seeds buy inference and molecules do not. "
            "Fewer molecules over more seeds is the better spend of the same budget."
        ),
        "consequence": (
            "cap_arithmetic is reported twice: at Appendix A §3.3's 200-unique ceiling (one "
            "seed) and at the operator's 50-unique cap over 3 seeds. The ceiling arithmetic is "
            "retained because it is what justifies the choice."
        ),
    },
)


def _adapter_facts(capability_id: str) -> Record:
    """Conditioning, required inputs and synthesis-awareness, read live.

    Read from the adapter rather than from a list here, because §1.6 of the work
    order is a statement about what the arms *are*: three of the eleven never see
    the target. A table that compared an unconditioned prior against a
    pocket-conditioned diffusion model without saying so would not be a
    measurement, and hand-maintaining the labels is how that happens.
    """
    from drug_agent.settings.config import Settings
    from drug_agent.tools.registry import generator_adapters

    adapters = generator_adapters(Settings())
    adapter = adapters.get(capability_id)
    if adapter is None:
        return {"conditioning": None, "required_inputs": [], "synthesis_aware": None}
    return {
        "conditioning": getattr(adapter, "conditioning", None),
        "required_inputs": list(getattr(adapter, "required_inputs", ()) or ()),
        "synthesis_aware": bool(getattr(adapter, "synthesis_aware", False)),
        "interpreter": getattr(getattr(adapter, "ext", None), "python", None),
        "device": getattr(getattr(adapter, "ext", None), "device", None),
        "checkout": str(getattr(getattr(adapter, "ext", None), "root", "") or "") or None,
        "upstream_ref": getattr(getattr(adapter, "spec", None), "upstream_ref", None),
    }


def _rates(raw: int, unique: int, wall: float, gpu: float | None) -> Record:
    """Throughput, with **both** denominators Appendix A §3.3's caps are written in.

    §3.3 caps two different things per arm per seed — 2,000 *raw generation
    attempts* and 200 *unique molecules submitted to the expensive pipeline* — and
    they are not the same number here: PocketFlow's probe produced four raw
    attempts and one usable molecule, so a cost stated per unique molecule is four
    times the cost stated per attempt. Reporting one rate would size the sweep
    against whichever cap happened to be chosen, so both are carried, each with the
    count it was divided by.

    Any rate whose denominator was not measured is ``None``. Dividing by a GPU
    second count that telemetry never produced would be a fabrication, and a zero
    would be worse: it would read as free.
    """
    return {
        "raw_attempts": raw,
        "unique_molecules": unique,
        "wall_seconds": round(wall, 3),
        "gpu_seconds": gpu,
        "molecules_per_wall_second": round(unique / wall, 6) if wall > 0 else None,
        "molecules_per_gpu_second": (round(unique / gpu, 6) if gpu else None),
        "raw_attempts_per_wall_second": round(raw / wall, 6) if wall > 0 else None,
        "raw_attempts_per_gpu_second": (round(raw / gpu, 6) if gpu else None),
        "wall_seconds_per_molecule": round(wall / unique, 3) if unique else None,
        "gpu_seconds_per_molecule": (round(gpu / unique, 3) if gpu and unique else None),
        "wall_seconds_per_raw_attempt": round(wall / raw, 3) if raw else None,
        "gpu_seconds_per_raw_attempt": (round(gpu / raw, 3) if gpu and raw else None),
        "yield_unique_per_raw_attempt": round(unique / raw, 4) if raw else None,
    }


def survey_arm(capability_id: str, *, workdir: str | Path | None = None) -> Record:
    """Run one arm's ``sample`` probe under wall/CPU/RSS/GPU accounting."""
    facts = _adapter_facts(capability_id)
    before = _resource.getrusage(_resource.RUSAGE_CHILDREN)
    self_before = _resource.getrusage(_resource.RUSAGE_SELF)
    started = time.time()
    with gpu_accounting(descendant_of=os.getpid()) as gpu:
        probe = probe_capability(capability_id, "sample", workdir=workdir)
    wall = time.time() - started
    after = _resource.getrusage(_resource.RUSAGE_CHILDREN)
    self_after = _resource.getrusage(_resource.RUSAGE_SELF)

    source_ids = dict(probe.get("source_ids") or {})
    funnel = dict(source_ids.get("funnel") or {})
    # ``n_returned`` is what the expensive pipeline would actually receive, which
    # is what a cap has to be sized against. ``n_distinct`` would have counted
    # Pocket2Mol's four sub-threshold structures as yield.
    unique = int(funnel.get("n_returned") or 0)
    raw = int(funnel.get("n_raw") or 0)
    gpu_seconds = gpu.get("gpu_seconds") if gpu.get("telemetry") == "MEASURED" else None

    return {
        "capability_id": capability_id,
        "verb": "sample",
        "status": probe["status"],
        "evidence": probe["evidence"],
        "reason": probe.get("reason"),
        "unblock": probe.get("unblock"),
        "conditioning": source_ids.get("conditioning") or facts.get("conditioning"),
        "required_inputs": source_ids.get("required_inputs") or facts.get("required_inputs"),
        "synthesis_aware": (
            source_ids.get("synthesis_aware")
            if source_ids.get("synthesis_aware") is not None
            else facts.get("synthesis_aware")
        ),
        "interpreter": source_ids.get("interpreter") or facts.get("interpreter"),
        "device": source_ids.get("device") or facts.get("device"),
        "gpu_kernel": source_ids.get("gpu_kernel"),
        "gpu_kernel_torch": source_ids.get("gpu_kernel_torch"),
        "checkout": source_ids.get("checkout") or facts.get("checkout"),
        "upstream_ref": source_ids.get("upstream_ref") or facts.get("upstream_ref"),
        "weights": source_ids.get("weights") or {},
        "pocket": source_ids.get("pocket"),
        "pocket_sha256": source_ids.get("pocket_sha256"),
        "pocket_center": source_ids.get("pocket_center"),
        "ref_ligand_sdf": source_ids.get("ref_ligand_sdf"),
        "ref_ligand_sha256": source_ids.get("ref_ligand_sha256"),
        "funnel": funnel,
        "cost": {
            "wall_seconds": round(wall, 3),
            "cpu_user_seconds": round(
                (after.ru_utime - before.ru_utime) + (self_after.ru_utime - self_before.ru_utime), 3
            ),
            "cpu_system_seconds": round(
                (after.ru_stime - before.ru_stime) + (self_after.ru_stime - self_before.ru_stime), 3
            ),
            "max_rss_kb": int(max(after.ru_maxrss, self_after.ru_maxrss)),
            "gpu_seconds": gpu_seconds,
            "gpu_telemetry": gpu.get("telemetry", "UNAVAILABLE"),
            "gpu_samples": gpu.get("samples"),
            "gpu_poll_interval_seconds": gpu.get("poll_interval_seconds"),
            "max_gpu_memory_mib": gpu.get("max_gpu_memory_mib"),
            "gpu_note": gpu.get("reason"),
        },
        "throughput": _rates(raw, unique, wall, gpu_seconds),
        "probe_seconds": probe.get("seconds"),
    }


def _scaled(value: float | None, seeds: int) -> float | None:
    """*value* over *seeds* seeds, or ``None`` when there was no value to scale."""
    return round(value * seeds, 1) if value is not None else None


def _binding_gpu(
    binding_on: str | None,
    gpu_attempt: float | None,
    gpu_unique: float | None,
    raw_cap: int,
    unique_cap: int,
) -> float | None:
    """GPU seconds at whichever cap stops the arm first, per seed."""
    if binding_on == "raw_attempt_cap":
        return round(gpu_attempt * raw_cap, 1) if gpu_attempt else None
    if binding_on == "unique_cap":
        return round(gpu_unique * unique_cap, 1) if gpu_unique else None
    return None


def cap_arithmetic(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_cap: int = APPENDIX_A_RAW_ATTEMPT_CAP,
    unique_cap: int = APPENDIX_A_UNIQUE_CAP,
    seeds: int = 1,
    label: str = "appendix_a_ceiling",
) -> Record:
    """What Appendix A §3.3's two ceilings would cost each admitted arm.

    §3.3 caps an arm at 2,000 raw attempts **and** 200 unique molecules, whichever
    it reaches first, so both are costed and the *binding* one — the cheaper of the
    two, because that is where the arm stops — is what the budget must cover.

    Deliberately linear in the measured per-attempt cost and deliberately labelled
    as a **lower bound on the rate**, i.e. an *upper* bound on the cost: every probe
    here pays one model load for a handful of molecules, so the marginal cost of
    attempt 2,000 is below the measured average. Extrapolating from a cold-start
    measurement can only overstate the total, which is the safe direction for a
    budget and the unsafe direction for a promise. It is stated rather than
    corrected because correcting it would need a second measurement per arm, which
    this step was not authorised to spend.

    The number that decides the cap is the slowest admitted arm, so that is returned
    separately. Choosing the cap belongs to the dispatcher and the operator; this
    returns the arithmetic they need.
    """
    admitted = [r for r in rows if r["status"] == "AVAILABLE_EXECUTED"]
    per_arm: list[Record] = []
    for row in admitted:
        rate = row["throughput"]
        wall_attempt = rate.get("wall_seconds_per_raw_attempt")
        gpu_attempt = rate.get("gpu_seconds_per_raw_attempt")
        wall_unique = rate.get("wall_seconds_per_molecule")
        gpu_unique = rate.get("gpu_seconds_per_molecule")
        wall_at_raw = round(wall_attempt * raw_cap, 1) if wall_attempt else None
        wall_at_unique = round(wall_unique * unique_cap, 1) if wall_unique else None
        candidates = [v for v in (wall_at_raw, wall_at_unique) if v is not None]
        binding_wall = min(candidates) if candidates else None
        binding_on = None
        if binding_wall is not None:
            binding_on = "raw_attempt_cap" if binding_wall == wall_at_raw else "unique_cap"
        per_arm.append(
            {
                "capability_id": row["capability_id"],
                "measured_raw_attempts": rate.get("raw_attempts"),
                "measured_unique_molecules": rate.get("unique_molecules"),
                "measured_wall_seconds": rate.get("wall_seconds"),
                "measured_gpu_seconds": rate.get("gpu_seconds"),
                "yield_unique_per_raw_attempt": rate.get("yield_unique_per_raw_attempt"),
                "wall_seconds_per_raw_attempt": wall_attempt,
                "gpu_seconds_per_raw_attempt": gpu_attempt,
                "wall_seconds_per_unique_molecule": wall_unique,
                "gpu_seconds_per_unique_molecule": gpu_unique,
                "wall_seconds_at_raw_cap": wall_at_raw,
                "gpu_seconds_at_raw_cap": round(gpu_attempt * raw_cap, 1) if gpu_attempt else None,
                "wall_seconds_at_unique_cap": wall_at_unique,
                "gpu_seconds_at_unique_cap": round(gpu_unique * unique_cap, 1) if gpu_unique else None,
                "binding_cap": binding_on,
                "binding_wall_seconds": binding_wall,
                "binding_hours": round(binding_wall / 3600.0, 2) if binding_wall else None,
                "binding_wall_seconds_all_seeds": (
                    round(binding_wall * seeds, 1) if binding_wall is not None else None
                ),
                "binding_gpu_seconds": _binding_gpu(binding_on, gpu_attempt, gpu_unique, raw_cap, unique_cap),
                "binding_gpu_seconds_all_seeds": _scaled(
                    _binding_gpu(binding_on, gpu_attempt, gpu_unique, raw_cap, unique_cap), seeds
                ),
            }
        )
    costed = [a for a in per_arm if a["binding_wall_seconds"] is not None]
    slowest = max(costed, key=lambda a: a["binding_wall_seconds"]) if costed else None
    total_wall = round(sum(a["binding_wall_seconds"] for a in costed), 1) if costed else None
    gpu_at_binding = [
        (a["gpu_seconds_at_unique_cap"] if a["binding_cap"] == "unique_cap" else a["gpu_seconds_at_raw_cap"])
        for a in costed
    ]
    measured_gpu = [g for g in gpu_at_binding if g is not None]
    total_gpu = round(sum(measured_gpu), 1) if measured_gpu else None

    # The arms the identical cap cannot be applied to, stated as arithmetic rather
    # than as prose. An arm whose measured yield is zero never reaches the unique
    # cap: it runs to the raw-attempt ceiling and stops with nothing, and reporting
    # that as a fixed-budget qualifying yield is exactly what Appendix A §3.3
    # forbids. What it would cost to find that out is the number that decides
    # whether it is worth running at all.
    unreachable: list[Record] = []
    for row in rows:
        rate = row["throughput"]
        yielded = rate.get("unique_molecules") or 0
        attempted = rate.get("raw_attempts") or 0
        if yielded or not attempted:
            continue
        wall_attempt = rate.get("wall_seconds_per_raw_attempt")
        gpu_attempt = rate.get("gpu_seconds_per_raw_attempt")
        unreachable.append(
            {
                "capability_id": row["capability_id"],
                "status": row["status"],
                "measured_raw_attempts": attempted,
                "measured_unique_molecules": 0,
                "wall_seconds_per_raw_attempt": wall_attempt,
                "gpu_seconds_per_raw_attempt": gpu_attempt,
                "wall_seconds_to_exhaust_raw_cap": round(wall_attempt * raw_cap, 1) if wall_attempt else None,
                "gpu_seconds_to_exhaust_raw_cap": round(gpu_attempt * raw_cap, 1) if gpu_attempt else None,
                "unique_molecules_at_that_point": 0,
                "why": (
                    f"measured yield is 0 of {attempted} attempts, so this arm can never reach the "
                    f"{unique_cap}-unique cap; it would run to the {raw_cap}-attempt ceiling and stop "
                    f"with nothing. Appendix A §3.3: that is a backend outcome with a consumed cost, "
                    f"not a fixed-budget qualifying yield of zero."
                ),
            }
        )
    return {
        "label": label,
        "raw_attempt_cap": raw_cap,
        "unique_cap": unique_cap,
        "seeds": seeds,
        "unique_molecules_per_arm_total": unique_cap * seeds,
        "note": (
            "Appendix A §3.3 permits 2,000 raw attempts AND 200 unique molecules per arm per "
            "seed, whichever is reached first. That is a ceiling, not a target. Every figure "
            "below is a linear extrapolation from a cold-start probe that paid one model load "
            "for a handful of attempts, so it OVERSTATES the per-attempt cost at scale and is a "
            "conservative budget, not a promise."
        ),
        "per_arm": sorted(per_arm, key=lambda a: a["capability_id"]),
        "n_admitted": len(admitted),
        "n_costed": len(costed),
        "slowest_admitted_arm": slowest,
        "total_wall_seconds_all_admitted": total_wall,
        "total_hours_all_admitted": round(total_wall / 3600.0, 2) if total_wall else None,
        "total_gpu_seconds_all_admitted": total_gpu,
        "total_wall_seconds_all_admitted_all_seeds": _scaled(total_wall, seeds),
        "total_hours_all_admitted_all_seeds": (round(total_wall * seeds / 3600.0, 2) if total_wall else None),
        "total_gpu_seconds_all_admitted_all_seeds": _scaled(total_gpu, seeds),
        "total_gpu_seconds_note": (
            "summed over the admitted arms that produced a measured GPU-second count, at each "
            "arm's own binding cap. An arm measured at 0.0 GPU-seconds ran on CPU and contributes "
            "nothing here; an arm whose telemetry was UNAVAILABLE is excluded and named rather "
            "than counted as free."
        ),
        "arms_excluded_from_the_gpu_total": [
            a["capability_id"] for a in costed if a["gpu_seconds_per_raw_attempt"] is None
        ],
        "arms_that_cannot_reach_the_unique_cap": unreachable,
        "identical_cap_rule": (
            "All arms receive identical caps (Appendix A §3.3). An identical cap that one arm "
            "cannot finish becomes a timeout, and a timeout reported as a fixed-budget yield is "
            "exactly what §3.3 forbids. The cap therefore has to fit the slowest admitted arm, "
            "and an arm that cannot reach it has to be named rather than truncated."
        ),
    }


def run_survey(arms: Iterable[str] = SURVEY_ARMS, *, workdir: str | Path | None = None) -> Record:
    """Measure every arm and assemble the record, in one pass, in a fixed order."""
    started = time.time()
    rows = [survey_arm(arm, workdir=workdir) for arm in arms]
    admitted = [r["capability_id"] for r in rows if r["status"] == "AVAILABLE_EXECUTED"]
    # Two different scientific facts, two different lists, and the status token on
    # each row says which. `EXECUTED_NO_YIELD` carries its funnel because the
    # funnel is the finding: the backend ran, and what it produced did not survive
    # the pipeline's own screening.
    limitations = [
        {
            "capability_id": r["capability_id"],
            "status": r["status"],
            "backend_executed": r["status"] == "EXECUTED_NO_YIELD",
            "reason": r["reason"],
            "unblock": r["unblock"],
            "consumed_wall_seconds": r["cost"]["wall_seconds"],
            "consumed_gpu_seconds": r["cost"]["gpu_seconds"],
            "consumed_gpu_telemetry": r["cost"]["gpu_telemetry"],
            "funnel": r["funnel"],
        }
        for r in rows
        if r["status"] != "AVAILABLE_EXECUTED"
    ]
    record: Record = {
        "contract_version": "1.0.0",
        "record_kind": "generator_availability_and_throughput",
        "note": (
            "Measured availability and cost for the generator arms in operator plan v1.1 §3.1. "
            "No molecule generated here is scored, ranked or counted towards a qualification: "
            "the scoring rule is not frozen and Appendix A §3.3 requires it to be registered "
            "before candidate search."
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": gpu_telemetry(),
        },
        "arms": rows,
        "admitted_to_sweep": admitted,
        "executed_without_usable_yield": [
            r["capability_id"] for r in rows if r["status"] == "EXECUTED_NO_YIELD"
        ],
        "did_not_execute": [r["capability_id"] for r in rows if r["status"] == "UNAVAILABLE"],
        "status_token_note": (
            "EXECUTED_NO_YIELD and UNAVAILABLE are both refused by every gate and are not the "
            "same finding. EXECUTED_NO_YIELD means the backend ran and consumed the cost recorded "
            "beside it; UNAVAILABLE means it never ran. Appendix A §3.3 separates 'log backend "
            "failures as missing scientific outcomes with consumed cost' from 'fixed-budget "
            "qualifying yield may be zero after failures', and these tokens are that separation."
        ),
        "named_limitations": limitations,
        "operator_decisions": list(OPERATOR_DECISIONS),
        "cap_arithmetic": cap_arithmetic(rows),
        "operator_cap_arithmetic": cap_arithmetic(
            rows,
            unique_cap=OPERATOR_UNIQUE_CAP,
            seeds=OPERATOR_SEEDS,
            label="operator_stage_d_cap",
        ),
        "totals": {
            "n_arms": len(rows),
            "n_available_executed": len(admitted),
            "n_unusable": len(limitations),
            "n_executed_without_usable_yield": sum(1 for r in rows if r["status"] == "EXECUTED_NO_YIELD"),
            "n_did_not_execute": sum(1 for r in rows if r["status"] == "UNAVAILABLE"),
            "wall_seconds": round(time.time() - started, 3),
            "gpu_seconds_measured": round(
                sum(r["cost"]["gpu_seconds"] or 0.0 for r in rows if r["cost"]["gpu_seconds"] is not None),
                3,
            ),
            "gpu_telemetry_states": sorted({r["cost"]["gpu_telemetry"] for r in rows}),
        },
    }
    record["record_hash"] = sha256_text(json.dumps(record, sort_keys=True, default=str))
    return record


def summary_rows(record: Mapping[str, Any]) -> list[Record]:
    """The committed summary table's rows: one per arm, no free text."""
    out: list[Record] = []
    for arm in record.get("arms", []):
        rate = arm.get("throughput") or {}
        funnel = arm.get("funnel") or {}
        out.append(
            {
                "arm": arm["capability_id"],
                "status": arm["status"],
                "conditioning": arm.get("conditioning"),
                "synthesis_aware": arm.get("synthesis_aware"),
                "interpreter": arm.get("interpreter"),
                "requested": funnel.get("n_requested"),
                "raw": funnel.get("n_raw"),
                "valid": funnel.get("n_valid"),
                "invalid": funnel.get("n_invalid"),
                "duplicate": funnel.get("n_duplicate"),
                "distinct": funnel.get("n_distinct"),
                "screened_out": funnel.get("n_screened_out"),
                "unique": funnel.get("n_returned"),
                "heavy_atom_window": [funnel.get("min_heavy_atoms"), funnel.get("max_heavy_atoms")],
                "wall_seconds": (arm.get("cost") or {}).get("wall_seconds"),
                "gpu_seconds": (arm.get("cost") or {}).get("gpu_seconds"),
                "gpu_telemetry": (arm.get("cost") or {}).get("gpu_telemetry"),
                "molecules_per_gpu_second": rate.get("molecules_per_gpu_second"),
                "molecules_per_wall_second": rate.get("molecules_per_wall_second"),
                "pocket_sha256": arm.get("pocket_sha256"),
                "reason": arm.get("reason"),
                "unblock": arm.get("unblock"),
            }
        )
    return out


__all__ = [
    "APPENDIX_A_RAW_ATTEMPT_CAP",
    "APPENDIX_A_UNIQUE_CAP",
    "OPERATOR_DECISIONS",
    "OPERATOR_SEEDS",
    "OPERATOR_UNIQUE_CAP",
    "SURVEY_ARMS",
    "cap_arithmetic",
    "run_survey",
    "summary_rows",
    "survey_arm",
]
