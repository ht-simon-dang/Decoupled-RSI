#!/usr/bin/env python
"""Generation at the ZAK C9R target: today's submission policy, and a cheap pre-filter.

Named for the arm it was written to reproduce, but generator-agnostic: ``--generator``
selects any registered de-novo sampler (``pocketxmol``, ``synflownet``) and the
adapter is resolved through the live registry. The conditioning is READ from the
adapter -- ``baselines._generation_inputs`` builds the pocket payload for a
pocket-conditioned sampler and returns nothing for an unconditional one -- so no
branch in this file compares a generator by name. The pre-filter is identical for
every arm: a filter fitted per arm would make the arms incomparable, which is the
whole reason for measuring them.

WHAT THIS MEASURES
------------------
The nine-arm zak-c9r table records ``pocketxmol`` at 47 qualifying scaffolds per
150 submitted. That 47 is a property of the *submission policy*, not of the
model. The sweep's own records say so:

* it spent 180 raw attempts against a lock-permitted 2000 per seed, and 0.75% of
  its GPU allocation;
* it then submitted the first 50 unique molecules per seed **unfiltered**;
* it carries 150 distinct Bemis-Murcko scaffolds in 150 molecules, so scaffold
  diversity is already saturated and *acceptance* is the only remaining lever;
* 93/150 of those molecules pass ``properties`` and 123/150 pass
  ``synthetic_accessibility`` -- two gates that need no docking at all.

So the question this module asks is narrow and answerable: if the arm spends the
budget it already had, and spends the *submission* budget on molecules that have
already cleared the gates that cost nothing, how many scaffolds are accepted?

It answers it with two runs of the same code under two named selection rules,
and nothing else changes between them.

THE PRE-FILTER, AND WHY ITS BOUNDARY IS THE WHOLE POINT
-------------------------------------------------------
``--selection cheap_gate_prefilter`` admits a molecule to the submission only if
it passes three gates:

* ``properties``               -- descriptors.mw, descriptors.logp, n_structural_alerts
* ``synthetic_accessibility``  -- RDKit Contrib sascorer, rounded to 3dp
* ``safety``                   -- the ADMET-AI hERG head, rounded to 6dp

Every one of those is computable from a SMILES string. **None** of them reads a
docked pose, a Vina score, or any other output of the expensive measurement.
That boundary is not a performance choice, it is the scientific content of the
experiment: filtering a submission on ``structure_score`` would be selecting the
submission on the very measurement the primary endpoint is computed from, which
is the defect this programme caught in an earlier round. A pre-filter that
touches a docking output does not produce a smaller-but-honest funnel, it
produces a number that means nothing. :func:`cheap_gate_table` therefore refuses
to run if the frozen qualification ever grows an endpoint gate this module has
not explicitly classified as cheap.

The thresholds are READ from ``configs/research/zak-c9r/qualification.json`` and
are never written, defaulted, or inlined here. The pass/fail arithmetic is the
frozen evaluator's own ``_property_passes`` and ``at_least_as_favourable``,
imported rather than re-implemented, so a molecule that this module calls a
cheap-gate survivor is one the evaluator will agree passed those same gates.

WHAT IT WRITES
--------------
``molecules.json``
    the scored rows from :func:`generation_sweep.score_molecules` -- the exact
    shape ``actions.change_sampling`` writes, which is the shape the frozen
    evaluator reads.
``funnel.json``
    generated / valid / unique / submitted, the selection rule by name, the
    per-gate pre-filter counts, and the pocket and checkpoint digests.
``cheap_gates.json``
    the per-molecule cheap-gate table over the whole candidate pool.
``evaluation.json``
    the seven-gate result from the frozen ``evaluate_portfolio``, so the run can
    be read without a campaign. This is a measurement artefact. It is NOT a
    campaign record and this module registers nothing.

Scoring is never re-implemented here: there is exactly one definition of
"qualifying" in this programme and it lives outside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

Record = dict[str, Any]

#: The arms this module can run, as the live registry names them. The adapter is
#: resolved through the registry, never constructed by name here, and the call is
#: the one signature ``GenerativeToolAdapter`` gives every generator -- so adding
#: an arm is a line in this tuple, not a branch in the sampling code.
GENERATORS = ("pocketxmol", "synflownet", "reinvent4")
DEFAULT_GENERATOR = "pocketxmol"


def capability_id_for(generator: str) -> str:
    """The registry id for an arm name. The registry, not a constructor, owns the adapter."""
    return f"molecule.generate.{generator}"


#: The digest the sweep's records carry for the ZAK pocket. Checked against the
#: file the adapter's OWN declared inputs resolve to, and only when the adapter
#: declares it needs a pocket at all: a pocket-conditioned run against a
#: different pocket is a different experiment, and an unconditional generator has
#: no pocket to check.
POCKET_SHA256 = "bf77fb48e746eac37e2bdbfb7e83285d0df203eb8da807f20af61aba47754032"

#: The sweep's sampling configuration, unchanged.
BATCH_SIZE = 16
CALL_TIMEOUT_SECONDS = 2400.0

#: The two named selection rules. ``generation_order`` is today's policy and the
#: default, so running this module with no ``--selection`` reproduces rather than
#: intervenes.
SELECTION_GENERATION_ORDER = "generation_order"
SELECTION_CHEAP_PREFILTER = "cheap_gate_prefilter"

#: The gates a molecule can be judged on from its SMILES alone. The property gate
#: is a list of limits in the qualification; the other two are endpoint gates.
#: ``structure_score`` is deliberately absent and :func:`cheap_gate_table`
#: enforces that.
CHEAP_ENDPOINT_GATES = ("safety", "synthetic_accessibility")
EXPENSIVE_GATES = ("structure_score", "geometry")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> str:
    """Write JSON and return its sha256, the way ``actions._write`` does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    path.write_text(text, encoding="utf-8")
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# the cheap gates
# ---------------------------------------------------------------------------
def cheap_gate_table(
    molecules: list[tuple[str, str]],
    *,
    qualification: dict[str, Any],
) -> Record:
    """Judge every molecule on the gates that need no docking.

    Returns the per-molecule table, the per-gate counts, and the measured cost.

    The thresholds come from *qualification* and the comparisons come from the
    frozen evaluator. Nothing here reads a pose or a score.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import (
        _property_passes,
        at_least_as_favourable,
    )
    from drug_agent.selfevolve.target.baseline_arms import (
        _descriptors,
        _generation_quality_endpoints,
    )

    limits = list(qualification.get("property_limits") or [])
    if not limits:
        raise SystemExit("the frozen qualification declares no property_limits; refusing to guess")

    declared = dict(qualification.get("endpoint_thresholds") or {})
    applied = {n: r for n, r in sorted(declared.items()) if r.get("status") == "CALIBRATED"}

    # A gate this module has never classified is a gate it cannot promise is
    # cheap. Refusing here is the only thing that keeps the pre-filter honest if
    # the frozen qualification grows a gate later: the alternative is silently
    # ignoring it (a pre-filter that no longer matches the evaluator) or silently
    # applying it (a pre-filter that might read a docking output).
    unknown = sorted(set(applied) - set(CHEAP_ENDPOINT_GATES))
    if unknown:
        raise SystemExit(
            f"the frozen qualification applies endpoint gate(s) {unknown} that this module has "
            f"not classified as docking-free. A pre-filter may only read quantities computable "
            f"from a SMILES; classify them explicitly or do not pre-filter on them"
        )
    for gate in EXPENSIVE_GATES:
        if gate in applied:
            raise SystemExit(f"{gate} reached the cheap pre-filter; that is the defect, not a pass")

    started = time.time()
    _say(f"cheap gates: descriptors + hERG + SA over {len(molecules)} candidates")
    endpoints = _generation_quality_endpoints(molecules)
    endpoint_wall = time.time() - started

    rows: list[Record] = []
    counts: Counter[str] = Counter()
    for molecule_id, smiles in molecules:
        properties = _descriptors(smiles)
        breaches = [limit for limit in limits if not _property_passes(properties, limit)]
        gates: Record = {"properties": {"passes": not breaches, "breaches": breaches}}

        panel = endpoints.get(molecule_id) or {}
        for name, record in applied.items():
            metric = str(record["metric"])
            bar = float(record["threshold"])
            precision = record.get("precision")
            entry = panel.get(metric)
            observed = None
            if isinstance(entry, dict) and entry.get("status") == "SUCCEEDED":
                observed = entry.get("value")
            if observed is None:
                # Same convention as the evaluator: an endpoint nobody could
                # compute fails its gate. It never becomes a favourable number.
                gates[name] = {"passes": False, "reason": "NOT_EVALUATED", "value": None, "threshold": bar}
                continue
            value = round(float(observed), int(precision)) if precision is not None else float(observed)
            ok = at_least_as_favourable(value, bar, metric)
            gates[name] = {
                "passes": ok,
                "reason": None if ok else "LESS_FAVOURABLE_THAN_THRESHOLD",
                "value": value,
                "threshold": bar,
                "metric": metric,
            }

        passing = [n for n, g in gates.items() if g["passes"]]
        failing = sorted(n for n, g in gates.items() if not g["passes"])
        for name in gates:
            counts[f"n_{name}_pass"] += int(gates[name]["passes"])
            counts[f"n_{name}_fail"] += int(not gates[name]["passes"])
        survives = not failing
        counts["n_cheap_survivors"] += int(survives)
        counts["n_candidates"] += 1
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "properties": properties,
                "endpoints": panel,
                "gates": gates,
                "failed_cheap_gates": failing,
                "passes_cheap_gates": survives,
                "n_cheap_gates_passed": len(passing),
            }
        )

    return {
        "rows": rows,
        "counts": dict(sorted(counts.items())),
        "gates_applied": ["properties", *sorted(applied)],
        "gates_never_read": list(EXPENSIVE_GATES),
        "rule": (
            "a candidate survives the pre-filter when it passes properties AND every calibrated "
            "endpoint gate that is computable from its SMILES. No docking output is read: "
            "structure_score and geometry are decided later, by the frozen evaluator, over the "
            "submission this filter chose"
        ),
        "thresholds": {
            "property_limits": limits,
            **{
                n: {"metric": r["metric"], "threshold": r["threshold"], "precision": r.get("precision")}
                for n, r in applied.items()
            },
        },
        "cost": {
            "wall_seconds": round(time.time() - started, 3),
            "endpoint_wall_seconds": round(endpoint_wall, 3),
            "n_candidates": len(molecules),
        },
    }


def endpoint_distributions(rows: list[Record], *, qualification: dict[str, Any]) -> Record:
    """Where the submitted molecules SIT on each endpoint gate, not just whether they passed.

    A pre-filter that includes an endpoint gate makes the pass count 150 by
    construction, and a pass count of 150 is then uninformative: it says nothing
    about whether the arm cleared the bar comfortably or scraped it. The quantity
    that still carries information is the headroom -- the distance from the
    observed value to the frozen threshold -- so it is reported as a
    distribution, with an explicit count of molecules sitting within 0.05 of the
    bar. An arm whose median sits 0.04 from the limit is one re-draw away from
    failing it and should not be read as having passed the gate robustly.
    """
    declared = dict(qualification.get("endpoint_thresholds") or {})
    applied = {n: r for n, r in sorted(declared.items()) if r.get("status") == "CALIBRATED"}

    def _quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return round(ordered[0], 6)
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return round(ordered[low] * (1 - weight) + ordered[high] * weight, 6)

    out: Record = {}
    for name, record in applied.items():
        metric = str(record["metric"])
        bar = float(record["threshold"])
        precision = record.get("precision")
        values: list[float] = []
        n_missing = 0
        for row in rows:
            entry = (row.get("endpoints") or {}).get(metric)
            usable = (
                isinstance(entry, dict)
                and entry.get("status") == "SUCCEEDED"
                and entry.get("value") is not None
            )
            if not usable:
                n_missing += 1
                continue
            value = float(entry["value"])
            values.append(round(value, int(precision)) if precision is not None else value)
        # Headroom under a LOWER_IS_BETTER endpoint is bar - value; the sign is
        # taken from the frozen orientation rather than assumed.
        lower_is_better = str(record.get("orientation") or "LOWER_IS_BETTER") == "LOWER_IS_BETTER"
        headroom = [(bar - v) if lower_is_better else (v - bar) for v in values]
        out[name] = {
            "metric": metric,
            "threshold": bar,
            "orientation": record.get("orientation"),
            "n_measured": len(values),
            "n_not_evaluated": n_missing,
            "min": round(min(values), 6) if values else None,
            "p25": _quantile(values, 0.25),
            "median": _quantile(values, 0.50),
            "p75": _quantile(values, 0.75),
            "max": round(max(values), 6) if values else None,
            "n_passing": sum(1 for h in headroom if h >= 0),
            "median_headroom": _quantile(headroom, 0.50),
            "min_headroom": round(min(headroom), 6) if headroom else None,
            "n_within_0.05_of_threshold": sum(1 for h in headroom if 0 <= h <= 0.05),
            "n_over_threshold": sum(1 for h in headroom if h < 0),
            "note": (
                "headroom is the distance from the frozen bar in the favourable direction; a "
                "negative headroom is a molecule over the limit. After a pre-filter that includes "
                "this gate, n_passing is 150 by construction and the headroom is the real signal"
            ),
        }
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def generate_pool(
    *,
    generator: str,
    n_generate: int,
    seed: int,
    chunk_size: int,
    workdir: Path,
) -> Record:
    """Draw ``n_generate`` RAW samples from one registered generator.

    The configuration is the sweep's, item for item: the registry's own adapter,
    the pocket payload the adapter itself declares it needs, ``batch_size=16``,
    the frozen 10-70 heavy-atom window, and the sweep's continuation-seed rule
    for calls after the first. The only thing this adds is chunking: an
    ``n_generate`` of 2000 in a single driver call is one subprocess holding an
    hour of GPU with no checkpoint, and a crash at minute 58 would lose all of it.

    THE CONDITIONING IS READ, NOT ASSUMED. The payload comes from
    ``baselines._generation_inputs``, which reads the adapter's own
    ``conditioning`` and ``required_inputs`` -- so a pocket-conditioned sampler
    gets the ZAK pocket and an unconditional one gets ``pocket=None``, and
    neither is decided by a name comparison here. An adapter that declares an
    input this host cannot supply is a refusal naming the input, never a run fed
    something else.
    """
    from drug_agent.selfevolve.target.baselines import (
        PROBE_MAX_HEAVY_ATOMS,
        PROBE_MIN_HEAVY_ATOMS,
        _generation_inputs,
        _loaded_weights,
        require_uncontaminated,
    )
    from drug_agent.selfevolve.target.budget import gpu_accounting
    from drug_agent.selfevolve.target.generation_sweep import (
        _generator_adapter,
        continuation_seed,
        generation_funnel,
    )

    capability_id = capability_id_for(generator)

    # The suite's generator substitution turns every backend into a stub. A
    # reproduction that ran against the stub would report a number about nothing.
    contamination = require_uncontaminated(f"{capability_id} pocketxmol_zak_c9r")

    adapter = _generator_adapter(capability_id)
    if not adapter.available():
        raise SystemExit(
            f"{capability_id} is UNAVAILABLE on this host: "
            f"{getattr(getattr(adapter, 'profile', None), 'unavailable_reason', None)}"
        )

    payload, missing, provenance = _generation_inputs(adapter)
    if missing:
        raise SystemExit(f"{capability_id}: this host has no {'; no '.join(missing)}")

    # Verify the conditioning input when there is one. A pocket-conditioned run
    # against a pocket that is not the sweep's is a different experiment, not a
    # noisy reproduction of this arm; an unconditional generator has no pocket to
    # verify and says so rather than silently skipping a check.
    pocket_path = payload.get("pdb_path")
    pocket_sha: str | None = None
    if pocket_path:
        pocket_sha = _sha256_path(Path(str(pocket_path)))
        if pocket_sha != POCKET_SHA256:
            raise SystemExit(
                f"the pocket at {pocket_path} digests to sha256:{pocket_sha}, and the sweep's "
                f"records name sha256:{POCKET_SHA256}. A different pocket is a different experiment"
            )
        _say(f"pocket verified: sha256:{pocket_sha}")
    else:
        _say(
            f"{capability_id} declares conditioning={getattr(adapter, 'conditioning', None)!r} and "
            f"required_inputs={tuple(getattr(adapter, 'required_inputs', ()) or ())}: pocket=None"
        )

    weights: Record = {}
    try:
        weights = _loaded_weights(adapter)
    except Exception as exc:  # a missing digest is recorded, never invented
        weights = {"error": f"{type(exc).__name__}: {exc}"}

    collected: dict[str, Record] = {}
    calls: list[Record] = []
    funnels: list[Record] = []
    provenances: list[Record | None] = []
    n_raw_total = 0
    n_valid_total = 0
    n_duplicate_across_calls = 0
    call_index = 0
    gpu_seconds = 0.0
    gpu_measured = True
    started = time.time()

    while n_raw_total < n_generate:
        request = min(chunk_size, n_generate - n_raw_total)
        sub_seed = continuation_seed(seed, call_index)
        call_started = time.time()
        gpu: Record = {}
        _say(
            f"generate call {call_index}: request={request} seed={sub_seed} "
            f"({n_raw_total}/{n_generate} raw so far, {len(collected)} unique)"
        )
        with gpu_accounting(descendant_of=os.getpid()) as sampled:
            gpu = sampled
            batch = adapter.generate(
                n_molecules=int(request),
                pocket=payload or None,
                seed=int(sub_seed),
                batch_size=int(min(BATCH_SIZE, request)),
                min_heavy_atoms=PROBE_MIN_HEAVY_ATOMS,
                max_heavy_atoms=PROBE_MAX_HEAVY_ATOMS,
                max_runtime_seconds=CALL_TIMEOUT_SECONDS,
            )
        if gpu.get("telemetry") == "MEASURED":
            gpu_seconds += float(gpu.get("gpu_seconds") or 0.0)
        else:
            gpu_measured = False

        # THE ADAPTER'S OWN STATEMENT of which weights it read. Distinct from
        # ``_loaded_weights`` above, which digests the files the adapter's SPEC
        # declares: that says what should have been loaded, this says what was.
        # ``config['checkpoint']`` is not a substitute -- it is an echo of the
        # argument and agrees with itself whether or not the load took. No
        # checkpoint is ever passed by this module, so an adapter that supports
        # an override (reinvent4 does) is sampling its parent prior, and this is
        # the field that proves it rather than asserting it.
        call_provenance = dict(dict(batch.config or {}).get("sample_provenance") or {})
        provenances.append(call_provenance or None)

        returned = list(batch.molecules)
        funnel = generation_funnel(batch, len(returned))
        funnels.append(funnel)
        n_raw_total += int(batch.n_raw)
        n_valid_total += int(funnel.get("n_valid") or 0)

        new = 0
        for molecule in returned:
            canonical = str(getattr(molecule, "smiles", "") or "")
            if not canonical:
                continue
            if canonical in collected:
                n_duplicate_across_calls += 1
                continue
            index = len(collected)
            collected[canonical] = {
                # The sweep's own id scheme, so a row from this module and a row
                # from the sweep are traceable to the same arm and seed.
                "molecule_id": f"{generator}_s{seed}_{index:04d}",
                "smiles": canonical,
                "n_heavy_atoms": getattr(molecule, "n_heavy_atoms", None),
                "sdf_block": getattr(molecule, "sdf_block", None),
                "qc_flags": list(getattr(molecule, "qc_flags", ()) or ()),
                "call_index": call_index,
                "call_seed": sub_seed,
                "sample_index": getattr(molecule, "sample_index", None),
            }
            new += 1

        calls.append(
            {
                "call_index": call_index,
                "seed": sub_seed,
                "requested": int(request),
                "n_raw": int(batch.n_raw),
                "n_valid": int(funnel.get("n_valid") or 0),
                "n_returned": len(returned),
                "n_new": new,
                "wall_seconds": round(time.time() - call_started, 3),
                "gpu_seconds": gpu.get("gpu_seconds"),
                "gpu_telemetry": gpu.get("telemetry", "UNAVAILABLE"),
                "errors": list(batch.errors),
                "warnings": list(getattr(batch, "warnings", ()) or ()),
                "sample_provenance": call_provenance or None,
            }
        )
        _say(
            f"  call {call_index} -> {batch.n_raw} raw / {funnel.get('n_valid')} valid / "
            f"{new} new unique in {calls[-1]['wall_seconds']}s "
            f"gpu={calls[-1]['gpu_seconds']} ({calls[-1]['gpu_telemetry']})"
        )
        call_index += 1
        if int(batch.n_raw) <= 0:
            # A call that drew nothing will draw nothing again; looping would
            # burn the wall clock producing an empty record.
            break
        # Checkpoint as we go: a pool this expensive should survive a crash.
        _write(
            workdir / "generation.json",
            {
                "seed": seed,
                "n_generate_requested": n_generate,
                "n_raw": n_raw_total,
                "n_unique": len(collected),
                "calls": calls,
                "molecules": list(collected.values()),
            },
        )

    pool = list(collected.values())
    # One statement for the whole pool, but only if every call agreed. A pool
    # assembled from calls that loaded DIFFERENT weights is not one arm's sample,
    # and collapsing it to the first call's provenance would hide that.
    distinct_provenance = {json.dumps(p, sort_keys=True) for p in provenances}
    sample_provenance: Any
    if len(distinct_provenance) == 1:
        sample_provenance = json.loads(next(iter(distinct_provenance)))
    else:
        sample_provenance = {
            "status": "INCONSISTENT_ACROSS_CALLS",
            "per_call": provenances,
            "note": (
                "the calls that built this pool did not all report the same loaded weights; "
                "the pool is not a single arm's sample and the count is not attributable"
            ),
        }
    return {
        "sample_provenance": sample_provenance,
        "checkpoint_override_passed": None,
        "checkpoint_override_note": (
            "this module never passes `checkpoint=`, so an adapter that supports an override "
            "sampled its parent prior. `sample_provenance` above is the adapter's own record of "
            "what it loaded; `weights` below is the digest of what its spec declares it loads"
        ),
        "molecules": pool,
        "calls": calls,
        "call_funnels": funnels,
        "n_generated": n_raw_total,
        "n_valid": n_valid_total,
        "n_unique": len(pool),
        "n_invalid": n_raw_total - n_valid_total,
        "n_duplicate_across_calls": n_duplicate_across_calls,
        "pocket": str(pocket_path) if pocket_path else None,
        "pocket_sha256": f"sha256:{pocket_sha}" if pocket_sha else None,
        "pocket_center": payload.get("center"),
        "pocket_payload_keys": sorted(payload),
        "generation_inputs_provenance": provenance,
        "batch_size": BATCH_SIZE,
        "heavy_atom_window": [PROBE_MIN_HEAVY_ATOMS, PROBE_MAX_HEAVY_ATOMS],
        "chunk_size": chunk_size,
        "continuation_seed_rule": (
            "call 0 uses the declared seed; call k>0 uses seed*1000+k -- generation_sweep's own "
            "rule, so a chunked draw is the same sequence of cells the sweep would have drawn"
        ),
        "capability_id": capability_id,
        "generator": generator,
        "task": getattr(getattr(adapter, "task", None), "name", None),
        "conditioning": getattr(adapter, "conditioning", None),
        "synthesis_aware": bool(getattr(adapter, "synthesis_aware", False)),
        "device": getattr(getattr(adapter, "ext", None), "device", None),
        "checkout": str(getattr(getattr(adapter, "ext", None), "root", "") or "") or None,
        "upstream_ref": getattr(getattr(adapter, "spec", None), "upstream_ref", None),
        "weights": weights,
        "contamination_check": contamination,
        "cost": {
            "wall_seconds": round(time.time() - started, 3),
            "gpu_seconds": round(gpu_seconds, 3) if gpu_measured else None,
            "gpu_telemetry": "MEASURED" if gpu_measured else "UNAVAILABLE",
        },
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run(
    *,
    generator: str,
    output_dir: Path,
    n_submit: int,
    n_generate: int,
    seed: int,
    selection: str,
    chunk_size: int,
    posebusters: bool,
    evaluate: bool,
) -> Record:
    from drug_agent.selfevolve.target.baseline_arms import _repo_root
    from drug_agent.selfevolve.target.budget import gpu_accounting
    from drug_agent.selfevolve.target.generation_sweep import score_molecules

    repo_root = _repo_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()

    qualification_path = repo_root / "configs" / "research" / "zak-c9r" / "qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    calibration_path = repo_root / "configs" / "research" / "zak-c9r" / "calibration.json"
    scorer = dict(json.loads(calibration_path.read_text(encoding="utf-8"))["scorer"])

    # --- generate ---------------------------------------------------------
    generated = generate_pool(
        generator=generator,
        n_generate=n_generate,
        seed=seed,
        chunk_size=chunk_size,
        workdir=output_dir,
    )
    pool = list(generated["molecules"])
    _say(
        f"pool: {generated['n_generated']} generated / {generated['n_valid']} valid / "
        f"{generated['n_unique']} unique"
    )
    if not pool:
        raise SystemExit("the sampler returned no usable molecule; there is nothing to score")

    # --- the cheap gates, measured over the WHOLE pool either way ---------
    # In generation_order mode these counts are observational: they are recorded
    # and not consulted. Computing them in both arms is what makes the two runs
    # comparable gate by gate rather than only at the bottom line.
    cheap = cheap_gate_table([(m["molecule_id"], m["smiles"]) for m in pool], qualification=qualification)
    _write(output_dir / "cheap_gates.json", cheap)
    survivors = {r["molecule_id"] for r in cheap["rows"] if r["passes_cheap_gates"]}
    _say(
        f"cheap gates: {len(survivors)}/{len(pool)} survivors "
        f"({cheap['counts'].get('n_properties_pass')} properties, "
        f"{cheap['counts'].get('n_synthetic_accessibility_pass')} SA, "
        f"{cheap['counts'].get('n_safety_pass')} safety)"
    )

    # --- selection --------------------------------------------------------
    if selection == SELECTION_GENERATION_ORDER:
        eligible = pool
        selection_note = (
            "the first n_submit DISTINCT molecules in the order the sampler emitted them, keeping "
            "the first occurrence of each duplicate. This is actions.change_sampling's rule and "
            "today's submission policy: no filter, no quality ranking, no selection pressure"
        )
    elif selection == SELECTION_CHEAP_PREFILTER:
        eligible = [m for m in pool if m["molecule_id"] in survivors]
        selection_note = (
            "the first n_submit DISTINCT molecules, in the order the sampler emitted them, that "
            "pass properties AND synthetic_accessibility AND safety -- the three gates computable "
            "without docking. Generation order is preserved WITHIN the survivors, so the only "
            "difference from generation_order is the removal of molecules already known to fail a "
            "gate. Nothing is ranked and no docking output is read"
        )
    else:  # pragma: no cover - argparse constrains this
        raise SystemExit(f"unknown selection rule {selection!r}")

    selected = eligible[:n_submit]
    # Emission order is actions.change_sampling's: canonical-SMILES ascending,
    # which fixes the gen_NNNN ids but not the membership.
    submitted_smiles = sorted(m["smiles"] for m in selected)
    submitted = [{"molecule_id": f"gen_{i:04d}", "smiles": s} for i, s in enumerate(submitted_smiles)]
    _say(f"selection={selection}: {len(eligible)} eligible -> {len(submitted)} submitted")

    # --- score: the one definition of "scored" ---------------------------
    _say(f"scoring {len(submitted)} molecules through score_molecules (Vina + endpoint panel)")
    score_started = time.time()
    with gpu_accounting(descendant_of=os.getpid()) as score_gpu:
        scored = score_molecules(
            [(m["molecule_id"], m["smiles"]) for m in submitted],
            scorer=scorer,
            workdir=output_dir / "scoring",
            posebusters=posebusters,
        )
    score_wall = time.time() - score_started
    rows = list(scored["rows"])
    scoring_cost = dict(scored.get("cost") or {})
    molecules_sha256 = _write(output_dir / "molecules.json", rows)
    _say(f"scored {len(rows)} rows in {score_wall:.1f}s -> molecules.json {molecules_sha256}")

    # --- the funnel -------------------------------------------------------
    funnel: Record = {
        "record_kind": "worker_experiment_funnel",
        "experiment_module": "drug_agent.selfevolve.target.experiments.pocketxmol_zak_c9r",
        "arm": generator,
        "generator": generator,
        "capability_id": capability_id_for(generator),
        "conditioning": generated.get("conditioning"),
        "seed": seed,
        "n_generated": generated["n_generated"],
        "n_generate_requested": n_generate,
        "n_valid": generated["n_valid"],
        "n_invalid": generated["n_invalid"],
        "n_unique": generated["n_unique"],
        "n_duplicate_across_calls": generated["n_duplicate_across_calls"],
        "n_cheap_gate_survivors": len(survivors),
        "n_eligible_for_submission": len(eligible),
        "n_submission_budget": n_submit,
        "n_submitted": len(submitted),
        "selection_rule": selection,
        "selection_rule_note": selection_note,
        "submission_emission_order": "canonical_smiles_ascending",
        "prefilter_applied": selection == SELECTION_CHEAP_PREFILTER,
        "prefilter_counts": cheap["counts"],
        "prefilter_gates_applied": cheap["gates_applied"],
        "prefilter_gates_never_read": cheap["gates_never_read"],
        "prefilter_rule": cheap["rule"],
        "prefilter_thresholds": cheap["thresholds"],
        "prefilter_threshold_source": str(qualification_path),
        "prefilter_cost": cheap["cost"],
        "generation": {k: v for k, v in generated.items() if k not in {"molecules", "call_funnels"}},
        "pocket": generated.get("pocket"),
        "pocket_sha256": generated.get("pocket_sha256"),
        "checkpoint_sha256": {
            name: record.get("sha256")
            for name, record in (generated["weights"] or {}).items()
            if isinstance(record, dict)
        },
        "weights": generated["weights"],
        "sample_provenance": generated.get("sample_provenance"),
        "checkpoint_override_passed": None,
        "checkpoint_override_note": generated.get("checkpoint_override_note"),
        "scorer": {
            "engine": scorer.get("engine"),
            "metric": scorer.get("metric"),
            "receptor_sha256": scorer.get("receptor_sha256"),
            "exhaustiveness": scorer.get("exhaustiveness"),
            "n_poses": scorer.get("n_poses"),
            "seed": scorer.get("seed"),
            "source": str(calibration_path),
        },
        "qualification_hash": qualification.get("qualification_hash"),
        "molecules_sha256": molecules_sha256,
        "submitted_endpoint_distributions": endpoint_distributions(rows, qualification=qualification),
        "scoring": {
            "status": "SUCCEEDED",
            "cost": scoring_cost,
            "wall_seconds": round(score_wall, 3),
            "gpu_seconds": score_gpu.get("gpu_seconds"),
            "gpu_telemetry": score_gpu.get("telemetry", "UNAVAILABLE"),
            "pose_quality_status": (scored.get("pose_quality") or {}).get("status"),
        },
        "cost": {
            "generation_wall_seconds": generated["cost"]["wall_seconds"],
            "generation_gpu_seconds": generated["cost"]["gpu_seconds"],
            "generation_gpu_telemetry": generated["cost"]["gpu_telemetry"],
            "scoring_wall_seconds": round(score_wall, 3),
            "scoring_gpu_seconds": score_gpu.get("gpu_seconds"),
            "scoring_gpu_telemetry": score_gpu.get("telemetry", "UNAVAILABLE"),
            "run_wall_seconds": round(time.time() - run_started, 3),
        },
        "note": (
            "generated, valid, unique, cheap-gate-survivor and submitted are five numbers and stay "
            "five numbers. The submission budget is fixed; the pre-filter changes WHICH molecules "
            "the budget is spent on and nothing else"
        ),
        "not_a_campaign_record": (
            "this module registers nothing, creates no campaign record and launches no series. It "
            "writes molecules.json in the scored-row shape the frozen evaluator reads, and stops"
        ),
    }

    # --- evaluate, so the run is readable without a campaign --------------
    if evaluate:
        funnel["evaluation"] = _evaluate(rows, output_dir=output_dir, qualification=qualification)
    _write(output_dir / "funnel.json", funnel)
    return funnel


def _evaluate(rows: list[Record], *, output_dir: Path, qualification: dict[str, Any]) -> Record:
    """Run the frozen seven-gate evaluator over the scored rows.

    This is the same call ``_score_arm`` and ``_portfolio_result`` make, with the
    same derived qualification: the frozen record plus Step 8's training-pool
    extension of the novelty exclusion set. Re-deriving it here rather than
    hard-coding a count is what keeps this number comparable to the sweep's.
    """
    from drug_agent.selfevolve.evaluation.target_benchmark import evaluate_portfolio, evaluator_hash
    from drug_agent.selfevolve.target.baseline_arms import (
        CONFIG_DIR,
        _common_exclusion_set,
        prepare_track_a,
    )

    # The same manifest ``run_target_study`` hands the sweep, for the same
    # reason: ``prepare_track_a`` needs the frozen stratum id, and the training
    # pool it returns is what extends the novelty exclusion set.
    manifest = json.loads((CONFIG_DIR / "baselines" / "B-B0-pfm-frozen.json").read_text(encoding="utf-8"))
    prepared = prepare_track_a(dict(manifest["preparation"]))
    common = _common_exclusion_set(qualification, prepared)
    derived = common["qualification"]

    evaluated = evaluate_portfolio(rows, qualification=derived)

    # Which gate refused which molecule, and how often a molecule was refused by
    # exactly one gate. The single-gate counts are the whole reason the
    # pre-filter is worth measuring: a molecule refused ONLY by properties is a
    # molecule the submission policy threw away for free.
    only: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    for row in evaluated["rows"]:
        failed = list(row.get("failed_gates") or ())
        if not failed:
            continue
        if len(failed) == 1:
            only[failed[0]] += 1
        else:
            pairs["+".join(sorted(failed))] += 1

    scaffolds = sorted({str(r["scaffold_id"]) for r in evaluated["rows"] if r.get("scaffold_id")})
    summary = {
        "qualifying_scaffolds": int(evaluated["qualifying_scaffolds"]),
        "n_qualifying_compounds": int(evaluated["n_qualifying_compounds"]),
        "n_evaluated": int(evaluated["n_evaluated"]),
        "n_distinct_scaffolds_submitted": len(scaffolds),
        "scaffolds_per_molecule_submitted": (
            round(len(scaffolds) / int(evaluated["n_evaluated"]), 6) if evaluated["n_evaluated"] else None
        ),
        "funnel": dict(evaluated["funnel"]),
        "failed_exactly_one_gate": dict(sorted(only.items())),
        "failed_multiple_gates": dict(sorted(pairs.items())),
        "applied_endpoint_gates": list(evaluated["applied_endpoint_gates"]),
        "unapplied_endpoint_gates": list(evaluated["unapplied_endpoint_gates"]),
        "evaluator_hash": evaluated["evaluator_hash"],
        "qualification_hash": evaluated["qualification_hash"],
        "derived_qualification_hash": derived.get("qualification_hash"),
        "common_exclusion": {k: v for k, v in common.items() if k != "qualification"},
    }
    _write(
        output_dir / "evaluation.json",
        {
            "record_kind": "worker_experiment_evaluation",
            "not_a_campaign_record": (
                "produced by a worker experiment module so the run can be read without a campaign; "
                "nothing here is registered and no nomination follows from it"
            ),
            "evaluator_hash": evaluator_hash(),
            **summary,
            "rows": evaluated["rows"],
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pocketxmol_zak_c9r",
        description=(
            "Sample a registered generator at the ZAK C9R target, choose a submission under a "
            "NAMED rule, score it with generation_sweep.score_molecules, and write "
            "molecules.json + funnel.json. Registers nothing."
        ),
    )
    parser.add_argument(
        "--generator",
        choices=list(GENERATORS),
        default=DEFAULT_GENERATOR,
        help="which registered generator to sample; the pre-filter is identical for all of them",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-submit", type=int, default=150, help="the fixed submission budget")
    parser.add_argument(
        "--n-generate",
        type=int,
        default=None,
        help="RAW samples to draw (default: --n-submit, i.e. today's no-oversampling policy)",
    )
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument(
        "--selection",
        choices=[SELECTION_GENERATION_ORDER, SELECTION_CHEAP_PREFILTER],
        default=SELECTION_GENERATION_ORDER,
        help="how the submission budget is spent; defaults to today's policy",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="raw samples per driver call; checkpointing granularity only",
    )
    parser.add_argument("--no-posebusters", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args(argv)

    n_generate = args.n_generate if args.n_generate is not None else args.n_submit
    if n_generate < args.n_submit:
        raise SystemExit(
            f"--n-generate {n_generate} is below --n-submit {args.n_submit}: the submission budget "
            f"is the denominator the primary metric is an absolute COUNT at"
        )

    funnel = run(
        generator=str(args.generator),
        output_dir=args.output_dir,
        n_submit=int(args.n_submit),
        n_generate=int(n_generate),
        seed=int(args.seed),
        selection=str(args.selection),
        chunk_size=int(args.chunk_size),
        posebusters=not args.no_posebusters,
        evaluate=not args.no_evaluate,
    )
    print(json.dumps({k: v for k, v in funnel.items() if k != "generation"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
