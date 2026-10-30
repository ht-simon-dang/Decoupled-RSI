"""Oversample-and-select: how a fixed submission budget is spent.

THE PROBLEM THIS SOLVES
-----------------------
The registered endpoint is an ABSOLUTE COUNT at a fixed budget: ``qualifying_scaffolds``
per 150 submitted molecules. Two numbers are therefore in play and they are not the
same number:

    n_generated  -- how many candidates the sampler was asked for.  UNBOUNDED.
    n_submitted  -- how many of them are handed to the frozen evaluator.  FIXED at 150.

Today's policy spends the budget on the first 150 distinct molecules the sampler
emitted. That is a *default*, not a design: it submits whatever arrived first,
including molecules already knowable to fail a gate. Measured on this target, 42 of
pocketxmol's 150 submitted molecules failed ONLY the properties gate -- 28% of the
budget spent on candidates that could have been rejected in microseconds, before a
single pose was computed.

This module makes the spending rule explicit, named, and recorded, and offers one
alternative to the default.

THE MECHANISM, IN FOUR STEPS
----------------------------
1. GENERATE a pool of ``n_generate`` raw samples where ``n_generate >= n_submit``.
   De-duplicate, keeping the FIRST occurrence in emission order.
2. JUDGE every molecule in the pool on the gates computable from a SMILES string
   alone -- properties, synthetic accessibility, hERG. This costs ~20 s for 2,000
   molecules. It reads no pose and no docking score.
3. SELECT the first ``n_submit`` molecules in GENERATION ORDER from the eligible set:
   the whole pool under ``generation_order``, the survivors under
   ``cheap_gate_prefilter``. Generation order is preserved *within* the survivors,
   so the only difference between the two rules is the removal of molecules already
   known to fail.
4. SCORE the submission through the single frozen scoring path. The expensive gates
   -- ``structure_score`` and ``geometry`` -- are decided there, by the evaluator,
   over the submission this module chose.

WHY THIS IS A SUBMISSION POLICY AND NOT A MODEL IMPROVEMENT
----------------------------------------------------------
No weight changes. No decoding parameter changes. The molecules are the same draws
from the same generator; only *which* of them fill the budget differs. Measured on
pocketxmol across three independent seeds, the count roughly doubles: 54/58/59 under
``generation_order`` and 107/113/113 under ``cheap_gate_prefilter``.

THE LINE THIS MODULE WILL NOT CROSS
-----------------------------------
A pre-filter may read ONLY quantities computable from a SMILES. It may never read a
docking output. Selecting on ``structure_score`` would mean running the expensive
measurement on candidates and submitting the ones that passed -- which is not a
prediction, it is a lookup of the answer, and it is the exact defect this programme
caught in an earlier round where a campaign submitted the 21 scaffolds it had already
measured as passing.

:func:`cheap_gate_table` enforces the line three ways, and all three are refusals
rather than warnings:

* ``structure_score`` or ``geometry`` reaching it is a hard error.
* A calibrated endpoint gate this module has not explicitly classified as
  docking-free is a hard error -- so if the frozen qualification grows a gate later,
  this fails loudly instead of silently ignoring it (a filter that no longer matches
  the evaluator) or silently applying it (a filter that might read a pose).
* Thresholds are read from the frozen qualification and the comparisons are the
  frozen evaluator's own predicates, imported rather than restated. There is exactly
  one definition of "passes a gate" in this programme and this module does not add a
  second.

THE PREDICTION THAT IS ALLOWED WHERE THE LOOKUP IS NOT
------------------------------------------------------
Reading a docking output of the molecule being selected is forbidden. PREDICTING one
is not, and ``surrogate_rank`` is that prediction: cheap gates first, then the
survivors ranked by a model fitted on the docking outputs of OTHER runs, then the
top N. The molecules it ranks have never been docked, and
:mod:`drug_agent.selfevolve.target.surrogate_prefilter` holds four refusals that keep
it that way -- a SMILES-only featuriser, a pool carrying any docking output refused,
a molecule in the model's own training set refused, and a run the model was fitted on
refused. The rule name and the model's hash go into the funnel beside the counts, so
a reader can tell which model produced a number; a model supplied to a rule that does
not rank is refused rather than recorded and ignored.

USING IT IN A COMPARISON
------------------------
The rule must be identical across arms or the comparison is meaningless, and it must
be recorded by name in the funnel so a reader can tell which rule produced a number.
Measured: the pre-filter is worth about +54 scaffolds on pocketxmol, +59 on
synflownet, and **-1 on reinvent4**, whose bottleneck is the docking gate at 93%
failure. It is not a general-purpose improvement; it is conditional on the generator
having candidates worth rescuing.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

Record = dict[str, Any]

#: The named selection rules. ``generation_order`` is today's policy and the default
#: everywhere, so a caller that does not choose reproduces current behaviour exactly.
SELECTION_GENERATION_ORDER = "generation_order"
SELECTION_CHEAP_PREFILTER = "cheap_gate_prefilter"
#: Cheap gates first, then the survivors ranked by a surrogate fitted on the docking
#: outputs of OTHER runs. It COMPOSES with the pre-filter rather than replacing it,
#: and the surrogate reads only quantities computable from a SMILES --- see
#: :mod:`drug_agent.selfevolve.target.surrogate_prefilter`, which holds the refusals
#: that keep it a prediction rather than a lookup.
SELECTION_SURROGATE_RANK = "surrogate_rank"
#: Cheap gates first, then ONE molecule per distinct Bemis-Murcko scaffold, taken in generation
#: order. It composes with the pre-filter exactly as ``surrogate_rank`` does, and it RANKS NOTHING:
#: within the survivors the order is still the order the sampler emitted them, and the only change is
#: that a molecule whose scaffold has already been submitted is skipped.
#:
#: It exists because ``qualifying_scaffolds`` counts SCAFFOLDS. Submitting two molecules on one
#: scaffold spends two of the 150 budget slots on at most one count, so a rule that spends each slot
#: on a distinct scaffold raises the ceiling of the metric without touching what the gates measure.
#: camp_zak_c9r_pfm_v2_dpo_pose discovered that at cycle 4 (77 against the prefilter's 69) and its
#: cycles 4 to 8 are all measured under it, so reproducing any of those numbers outside a campaign
#: needs the rule to live here rather than inside a workspace that gets torn down.
SELECTION_SCAFFOLD_DIVERSE = "scaffold_diverse"
SELECTION_RULES = (
    SELECTION_GENERATION_ORDER,
    SELECTION_CHEAP_PREFILTER,
    SELECTION_SURROGATE_RANK,
    SELECTION_SCAFFOLD_DIVERSE,
)

#: Calibrated endpoint gates this module has classified as computable from a SMILES.
#: Adding a name here is a claim that the quantity needs no pose; it is checked
#: against the frozen qualification at run time and an unclassified gate is refused.
CHEAP_ENDPOINT_GATES = ("safety", "synthetic_accessibility")

#: Gates that require a pose. These may never reach the pre-filter.
EXPENSIVE_GATES = ("structure_score", "geometry")

SELECTION_NOTES: dict[str, str] = {
    SELECTION_GENERATION_ORDER: (
        "the first n_submit DISTINCT molecules in the order the sampler emitted them, keeping "
        "the first occurrence of each duplicate. This is today's submission policy: no filter, "
        "no quality ranking, no selection pressure"
    ),
    SELECTION_CHEAP_PREFILTER: (
        "the first n_submit DISTINCT molecules, in the order the sampler emitted them, that pass "
        "every gate computable from a SMILES. Generation order is preserved WITHIN the survivors, "
        "so the only difference from generation_order is the removal of molecules already known to "
        "fail a gate. Nothing is ranked and no docking output is read"
    ),
    SELECTION_SCAFFOLD_DIVERSE: (
        "three stages, nothing ranked. Stage 1 walks the DISTINCT molecules that pass every gate "
        "computable from a SMILES, in the order the sampler emitted them, and keeps the FIRST "
        "molecule of each distinct Bemis-Murcko scaffold until the budget holds that many distinct "
        "scaffolds. Stage 2 falls back to the remaining survivors in generation order, and only if "
        "they carry fewer scaffolds than the budget. Stage 3 tops up to exactly the budget with "
        "remaining distinct molecules that FAILED a cheap gate. No docking output, no learned "
        "surrogate and no gate beyond the frozen cheap ones enters the choice. Stage 3 is the one "
        "to watch: a submission it fills is not comparable with one it did not, because a top-up "
        "molecule is already known to fail a gate -- report n_topped_up beside every count"
    ),
    SELECTION_SURROGATE_RANK: (
        "the n_submit molecules that pass every gate computable from a SMILES and then rank "
        "highest under a surrogate fitted on the docking outputs of OTHER runs. The surrogate "
        "reads only the SMILES of the molecules it ranks, and none of them has been docked, so "
        "this is a PREDICTION of structure_score and geometry and not a lookup of them. Ties "
        "break on generation order, so a constant surrogate reduces this rule exactly to "
        "cheap_gate_prefilter"
    ),
}


def cheap_gate_table(molecules: list[tuple[str, str]], *, qualification: Record) -> Record:
    """Judge every candidate on the gates that need no docking.

    *molecules* is a list of ``(molecule_id, smiles)``. Returns the per-molecule
    table, the per-gate counts, the thresholds applied, and the measured cost.

    Thresholds come from *qualification*; the comparisons are the frozen evaluator's
    own predicates. Nothing here reads a pose or a docking score, and the guards
    below refuse rather than warn if that could stop being true.
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
        raise ValueError("the frozen qualification declares no property_limits; refusing to guess")

    declared = dict(qualification.get("endpoint_thresholds") or {})
    applied = {name: rec for name, rec in sorted(declared.items()) if rec.get("status") == "CALIBRATED"}

    # The expensive-gate check runs FIRST. Both guards would fire on a docking gate,
    # but only this one names what actually went wrong; letting the generic
    # "unclassified" message win would bury the serious case in the ordinary one.
    for gate in EXPENSIVE_GATES:
        if gate in applied:
            raise ValueError(
                f"{gate} reached the cheap pre-filter; that is the defect, not a pass. Selecting on "
                f"a gate that needs a pose means submitting the molecules an expensive measurement "
                f"already passed, which is a lookup and not a prediction"
            )

    unknown = sorted(set(applied) - set(CHEAP_ENDPOINT_GATES))
    if unknown:
        raise ValueError(
            f"the frozen qualification applies endpoint gate(s) {unknown} that this module has not "
            f"classified as docking-free. A pre-filter may only read quantities computable from a "
            f"SMILES; classify them explicitly or do not pre-filter on them"
        )

    started = time.time()
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
                # The evaluator's own convention: an endpoint nobody could compute
                # FAILS its gate. It never becomes a favourable number by default.
                gates[name] = {
                    "passes": False,
                    "reason": "NOT_EVALUATED",
                    "value": None,
                    "threshold": bar,
                    "metric": metric,
                }
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

        failing = sorted(name for name, gate in gates.items() if not gate["passes"])
        for name, gate in gates.items():
            counts[f"n_{name}_pass"] += int(gate["passes"])
            counts[f"n_{name}_fail"] += int(not gate["passes"])
        counts["n_cheap_survivors"] += int(not failing)
        counts["n_candidates"] += 1
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "properties": properties,
                "endpoints": panel,
                "gates": gates,
                "failed_cheap_gates": failing,
                "passes_cheap_gates": not failing,
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
                name: {
                    "metric": rec["metric"],
                    "threshold": rec["threshold"],
                    "precision": rec.get("precision"),
                }
                for name, rec in applied.items()
            },
        },
        "cost": {
            "wall_seconds": round(time.time() - started, 3),
            "endpoint_wall_seconds": round(endpoint_wall, 3),
            "n_candidates": len(molecules),
        },
    }


def select_submission(
    pool: list[Record],
    *,
    n_submit: int,
    selection: str = SELECTION_GENERATION_ORDER,
    cheap: Record | None = None,
    surrogate: Any | None = None,
) -> Record:
    """Choose which ``n_submit`` of *pool* spend the submission budget.

    *pool* is a list of records carrying at least ``molecule_id`` and ``smiles``, in
    the order the sampler emitted them, already de-duplicated first-occurrence-kept.
    *cheap* is the output of :func:`cheap_gate_table` over that same pool; it is
    REQUIRED when *selection* is ``cheap_gate_prefilter`` or ``surrogate_rank`` and
    optional otherwise (where it is recorded but not consulted, so that both arms
    carry the same telemetry and can be compared gate by gate rather than only at
    the bottom line).

    *surrogate* is REQUIRED by ``surrogate_rank`` and refused by every other rule.
    It must expose ``score_pool(records) -> list[float]``, higher-is-better, and a
    ``model_hash``; the hash is recorded in the funnel exactly as the rule name is,
    so a reader can tell which model produced a number. The records it is handed
    carry ``molecule_id`` and ``smiles`` and nothing else: a pool member carrying a
    docking output is refused here and again inside the surrogate, because ranking
    a molecule an expensive measurement already judged is the lookup this module
    exists to prevent.

    Returns the chosen molecules and the funnel that explains the choice. The five
    counts -- generated, valid, unique, cheap-survivor, submitted -- stay five
    separate numbers; the submission budget is fixed and this changes only WHICH
    molecules it is spent on.
    """
    if selection not in SELECTION_RULES:
        raise ValueError(f"unknown selection rule {selection!r}; expected one of {list(SELECTION_RULES)}")
    if n_submit < 1:
        raise ValueError(f"n_submit must be positive, got {n_submit}")
    if surrogate is not None and selection != SELECTION_SURROGATE_RANK:
        raise ValueError(
            f"a surrogate was supplied to {selection!r}, which does not rank. A model that is "
            f"present but unused is worse than absent: the funnel would name it and the "
            f"submission would not reflect it"
        )

    survivors: set[str] | None = None
    if cheap is not None:
        survivors = {row["molecule_id"] for row in cheap["rows"] if row["passes_cheap_gates"]}

    if selection in (SELECTION_CHEAP_PREFILTER, SELECTION_SURROGATE_RANK, SELECTION_SCAFFOLD_DIVERSE):
        if survivors is None:
            raise ValueError(
                f"{selection} needs the cheap_gate_table for this pool; refusing to select "
                f"on gates it has not measured"
            )
        eligible = [m for m in pool if m["molecule_id"] in survivors]
    else:
        eligible = list(pool)

    surrogate_card: Record | None = None
    if selection == SELECTION_SURROGATE_RANK:
        if surrogate is None:
            raise ValueError("surrogate_rank needs a surrogate; refusing to rank on a model it does not have")
        from drug_agent.selfevolve.target.surrogate_prefilter import assert_record_is_docking_free

        for member in pool:
            assert_record_is_docking_free(member, what=f"pool member {member.get('molecule_id')!r}")
        scores = surrogate.score_pool(
            [{"molecule_id": m["molecule_id"], "smiles": m["smiles"]} for m in eligible]
        )
        if len(scores) != len(eligible):
            raise ValueError(
                f"the surrogate returned {len(scores)} scores for {len(eligible)} eligible molecules"
            )
        # Descending by score, ties broken by position among the survivors -- which
        # IS generation order, because `eligible` preserves it. A constant surrogate
        # therefore reproduces cheap_gate_prefilter exactly.
        ranked = sorted(range(len(eligible)), key=lambda i: (-scores[i], i))
        eligible = [eligible[i] for i in ranked]
        surrogate_card = surrogate.card() if hasattr(surrogate, "card") else {}

    scaffold_card: Record | None = None
    if selection == SELECTION_SCAFFOLD_DIVERSE:
        selected, scaffold_card = _scaffold_diverse_pick(pool, eligible, n_submit=n_submit)
    else:
        selected = eligible[:n_submit]

    # Emission order is canonical-SMILES ascending, matching change_sampling: it
    # fixes the gen_NNNN ids and the docking order, and it does NOT touch which
    # molecules were chosen. Membership is decided above, in generation order.
    submitted = [
        {"molecule_id": f"gen_{index:04d}", "smiles": smiles}
        for index, smiles in enumerate(sorted(m["smiles"] for m in selected))
    ]

    funnel: Record = {
        "selection_rule": selection,
        "selection_rule_note": SELECTION_NOTES[selection],
        "prefilter_applied": selection
        in (SELECTION_CHEAP_PREFILTER, SELECTION_SURROGATE_RANK, SELECTION_SCAFFOLD_DIVERSE),
        "surrogate_applied": selection == SELECTION_SURROGATE_RANK,
        "surrogate_model_hash": getattr(surrogate, "model_hash", None) if surrogate is not None else None,
        "surrogate_card": surrogate_card,
        **({"scaffold_diverse": scaffold_card} if scaffold_card else {}),
        **({"n_topped_up": scaffold_card["n_stage3_topped_up"]} if scaffold_card else {}),
        "submission_emission_order": "canonical_smiles_ascending",
        "n_pool": len(pool),
        "n_cheap_gate_survivors": len(survivors) if survivors is not None else None,
        "n_eligible_for_submission": len(eligible),
        "n_submission_budget": n_submit,
        "n_submitted": len(submitted),
        "budget_filled": len(submitted) == n_submit,
        "oversample_factor": round(len(pool) / n_submit, 3) if n_submit else None,
        "note": (
            "generated, valid, unique, cheap-gate-survivor and submitted are five numbers and stay "
            "five numbers. The submission budget is fixed; the pre-filter changes WHICH molecules "
            "the budget is spent on and nothing else"
        ),
    }
    if selection == SELECTION_GENERATION_ORDER and survivors is not None:
        funnel["cheap_gate_survivors_observational"] = (
            "the cheap gates were measured over this pool and RECORDED but NOT consulted; this arm "
            "submitted in generation order. The counts are here so the two arms are comparable gate "
            "by gate"
        )
    return {"submitted": submitted, "selected": selected, "eligible": eligible, "funnel": funnel}


def _murcko(smiles: str) -> str | None:
    """The Bemis-Murcko scaffold SMILES, or ``None`` when RDKit cannot produce one.

    An acyclic molecule yields the empty string, which is a real answer and not a failure: the frozen
    evaluator counts it under its own per-molecule key rather than pooling every acyclic molecule into
    one scaffold. So the empty string is returned as-is and the caller keys on the molecule instead.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:  # a scaffold RDKit refuses is reported as absent, never guessed
        return None


def _scaffold_diverse_pick(
    pool: list[Record], eligible: list[Record], *, n_submit: int
) -> tuple[list[Record], Record]:
    """One molecule per distinct Bemis-Murcko scaffold, then two declared fallbacks.

    The stages are kept SEPARATE in the returned card because they are not the same measurement. A
    budget filled entirely from stage 1 is 150 slots spent on 150 distinct scaffolds; a budget that
    reached stage 3 spent some of them on molecules already known to fail a gate, which lowers the
    count by construction. camp_zak_c9r_pfm_v2_dpo_pose cycle 3 filled 127 of 150 that way and scored
    44 against the same arm's 77 at zero top-ups, so the distinction is worth more than the total.
    """
    chosen: list[Record] = []
    seen_scaffolds: set[str] = set()
    used: set[str] = set()

    # Stage 1: first of each distinct scaffold, in generation order.
    for member in eligible:
        if len(chosen) >= n_submit:
            break
        scaffold = _murcko(member.get("smiles", ""))
        if scaffold is None:
            continue
        key = scaffold or f"acyclic:{member['molecule_id']}"
        if key in seen_scaffolds:
            continue
        seen_scaffolds.add(key)
        chosen.append(member)
        used.add(member["molecule_id"])
    n_stage1 = len(chosen)

    # Stage 2: the remaining survivors, in generation order, only if stage 1 fell short.
    for member in eligible:
        if len(chosen) >= n_submit:
            break
        if member["molecule_id"] in used:
            continue
        chosen.append(member)
        used.add(member["molecule_id"])
    n_stage2 = len(chosen) - n_stage1

    # Stage 3: molecules that FAILED a cheap gate, which is the only stage that lowers the count.
    for member in pool:
        if len(chosen) >= n_submit:
            break
        if member["molecule_id"] in used:
            continue
        chosen.append(member)
        used.add(member["molecule_id"])
    n_stage3 = len(chosen) - n_stage1 - n_stage2

    card: Record = {
        "n_stage1_first_of_its_scaffold": n_stage1,
        "n_stage2_survivor_repeat_scaffold": n_stage2,
        "n_stage3_topped_up": n_stage3,
        "n_distinct_scaffolds_among_survivors": len(seen_scaffolds),
        "budget_filled_from_stage1_alone": n_stage1 >= n_submit,
        "why_the_stages_are_reported_apart": (
            "a slot filled at stage 3 holds a molecule already known to fail a cheap gate, so it "
            "cannot contribute a qualifying scaffold. A count with top-ups and a count without are "
            "different measurements and must not be pooled"
        ),
    }
    return chosen, card


def plan_oversample(n_submit: int, *, expected_survival: float, headroom: float = 1.5) -> int:
    """How many raw samples to ask for so the budget can be filled from survivors.

    *expected_survival* is the fraction of candidates expected to clear the cheap
    gates -- measured at 0.48-0.50 for pocketxmol on this target, 0.54 for
    synflownet, 0.50-0.62 for reinvent4. *headroom* covers the variance of a single
    draw; 1.5 is deliberately generous because generation is cheap relative to
    docking and an under-filled budget is an absolute-count loss that cannot be
    recovered afterwards.

    Returns at least *n_submit*: oversampling may never reduce the pool below the
    budget, because a short submission is not comparable to a full one on a metric
    that counts.
    """
    if not 0.0 < expected_survival <= 1.0:
        raise ValueError(f"expected_survival must be in (0, 1], got {expected_survival}")
    if headroom < 1.0:
        raise ValueError(f"headroom must be at least 1.0, got {headroom}")
    return max(n_submit, round(n_submit / expected_survival * headroom))
