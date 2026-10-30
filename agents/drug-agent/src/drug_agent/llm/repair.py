"""Deterministic repairs of model output, gated by the runtime policy.

The master agent already contains one repair of this kind: a success criterion
naming a metric nothing can compute is excluded from scoring and reported,
because leaving it in makes ACCEPT unreachable and the run burns every cycle
chasing a number that will never exist. The two functions here are the same idea
applied to two more failures that the request journal shows actually happening.

Both are **off by default**. Each changes what a plan or an adjustment means, and
that is a change the evaluation has to earn rather than something to switch on
because it sounds right.

``repair_criteria_for_goal``
    A criterion is unscorable when the metric cannot be computed *for this
    program*, and whether it can depends on the goal kind — which is only known
    after transcription. ``max_pIC50`` on a phenotype-rescue program needs
    bioactivity for a protein target that the program does not have, so it reads
    as missed every cycle. Observed: the one phenotype request in the journal
    planned both ``max_pIC50 >= 7.0`` and ``median_pIC50 >= 6.0`` and scored
    1 / 5, with the evaluation agent noting "No pIC50 predictions available for
    this run (target/objective mismatch)".

``merge_adjustment``
    A refinement is supposed to move generation levers. Observed: one adjustment
    reported four targeted changes (seeds, notes, ``n_generated``,
    ``avoid_scaffolds``) while the object it emitted silently dropped **all
    three objectives and all five hard constraints** — the schema gives both
    fields a default of ``[]``, so omission is deletion. The next cycle then
    scored ``hard_constraint_pass_rate == 1.0`` because there were no
    constraints left to fail, and lost ``qed_median`` and ``sa_median``
    entirely because there were no objectives left to score. This carries
    forward anything the model did not say it was changing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .schemas import ChemicalRequirements, ParameterChange, SuccessCriterion

#: Metrics that require a protein-target bioactivity model, and so can never be
#: computed for a phenotype-rescue program on this platform.
TARGET_ONLY_METRICS = frozenset({"max_pIC50", "median_pIC50", "in_domain_fraction"})

#: The mirror: only a phenotype program produces a restoration signature.
PHENOTYPE_ONLY_METRICS = frozenset({"restoration_median"})


def inapplicable_metrics(kind: str) -> frozenset[str]:
    """Metrics that cannot apply to a program of this ``kind``."""
    return PHENOTYPE_ONLY_METRICS if str(kind) == "target_based" else TARGET_ONLY_METRICS


def repair_criteria_for_goal(
    criteria: list[SuccessCriterion], kind: str
) -> tuple[list[SuccessCriterion], list[SuccessCriterion]]:
    """Split criteria into ``(applicable, inapplicable_for_this_goal_kind)``.

    Nothing is invented and nothing is silently deleted: the caller reports the
    second list exactly as it already reports metrics the platform cannot
    compute at all. If *every* criterion turns out inapplicable the input is
    returned unchanged, because a plan with no criteria is worse than a plan
    with bad ones — it cannot be judged at all.
    """
    blocked = inapplicable_metrics(kind)
    keep = [c for c in criteria if c.metric not in blocked]
    dropped = [c for c in criteria if c.metric in blocked]
    if not keep:
        return list(criteria), []
    return keep, dropped


# ---------------------------------------------------------------------------
# Refinement merge
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z0-9_.]+")


def _mentioned_tokens(changes: list[ParameterChange]) -> set[str]:
    """The field paths the model named in the ``parameter`` slot of its changes.

    ``parameter`` is the structured half of a ``ParameterChange``: the refine
    prompt asks for "which requirement you changed (e.g. seed_smiles,
    descriptors.mw constraint, pIC50 target_value)". That — and only that — is a
    declaration of intent to touch a field.

    The other three slots are deliberately excluded, because reading them turned
    this guard into the thing it exists to prevent:

    * ``reason`` is free prose *about* the change. One observed adjustment gave
      the reason "…improving the chance of reaching pIC50 >= 7.0…" while changing
      seeds; the token ``pic50`` fell out of that sentence, the merge read it as
      "the model said it was changing pIC50", and the pIC50 objective was dropped
      on the model's behalf — the silent deletion this function is here to stop.
    * ``old_value`` / ``new_value`` hold values, not paths. A change reporting
      ``parameter="objectives"`` with ``old_value="pIC50, qed, sa_score"`` would
      otherwise name every objective at once and authorise deleting all of them.

    Excluding them fails *closed*: a field the model genuinely meant to drop but
    only mentioned in prose is carried forward instead, and the caller's notes
    say so out loud. Restoring one entry too many is visible and reversible;
    deleting the brief's primary objective is neither.
    """
    tokens: set[str] = set()
    for change in changes:
        for tok in _TOKEN.findall(str(change.parameter).lower()):
            tokens.add(tok)
            if "." in tok:
                tokens.update(tok.split("."))
    return tokens


def _names_field(tokens: set[str], name: str) -> bool:
    """Whether one of the named parameter paths actually refers to ``name``.

    Matching is on the whole path or its leaf: ``descriptors.mw`` names the
    ``descriptors.mw`` constraint, and a bare ``mw`` names it too. A shared
    *prefix* does not — the previous version matched any dotted component longer
    than two characters, so a change to ``descriptors.logp`` also claimed
    ``descriptors.mw`` and every other descriptor constraint in the brief.
    """
    lowered = name.lower()
    return lowered in tokens or lowered.rsplit(".", 1)[-1] in tokens


def merge_adjustment(
    old: ChemicalRequirements,
    new: ChemicalRequirements,
    changes: list[ParameterChange],
    *,
    max_drop_fraction: float = 1.0,
) -> tuple[ChemicalRequirements, list[str]]:
    """Carry forward requirement entries the model did not say it was changing.

    Returns the merged requirements and a list of human-readable notes naming
    every entry that was restored, so the restoration is visible in the event
    stream and the journal rather than being another silent rewrite.

    ``max_drop_fraction`` is a second, blunter guard: after carry-forward, if the
    adjustment still removes more than this fraction of the original objectives
    plus constraints, the remainder is restored too and the note says so. It
    exists because "the model listed a change for every field it deleted" is a
    thing a model can do while still destroying the program.
    """
    notes: list[str] = []
    tokens = _mentioned_tokens(changes)

    kept_obj = {o.name for o in new.objectives}
    restored_obj = [o for o in old.objectives if o.name not in kept_obj and not _names_field(tokens, o.name)]

    kept_con = {c.property for c in new.property_constraints}
    restored_con = [
        c
        for c in old.property_constraints
        if c.property not in kept_con and not _names_field(tokens, c.property)
    ]

    if restored_obj:
        new.objectives = list(new.objectives) + restored_obj
        notes.append(
            "restored objectives the adjustment dropped without saying so: "
            + ", ".join(o.name for o in restored_obj)
        )
    if restored_con:
        new.property_constraints = list(new.property_constraints) + restored_con
        notes.append(
            "restored hard constraints the adjustment dropped without saying so: "
            + ", ".join(f"{c.property}{c.op}{c.value}" for c in restored_con)
        )

    total_old = len(old.objectives) + len(old.property_constraints)
    if total_old and max_drop_fraction < 1.0:
        now_obj = {o.name for o in new.objectives}
        now_con = {c.property for c in new.property_constraints}
        still_dropped_obj = [o for o in old.objectives if o.name not in now_obj]
        still_dropped_con = [c for c in old.property_constraints if c.property not in now_con]
        dropped = len(still_dropped_obj) + len(still_dropped_con)
        if dropped / total_old > max_drop_fraction:
            new.objectives = list(new.objectives) + still_dropped_obj
            new.property_constraints = list(new.property_constraints) + still_dropped_con
            notes.append(
                f"adjustment removed {dropped}/{total_old} requirement entries, over the "
                f"{max_drop_fraction:.0%} cap — the remainder was restored"
            )

    # An adjustment that leaves no objective at all makes the brief meaningless;
    # the bridge would silently substitute a bare pIC50 objective.
    if old.objectives and not new.objectives:
        new.objectives = list(old.objectives)
        notes.append("adjustment left no objectives at all — the original set was restored")

    return new, notes


__all__ = [
    "PHENOTYPE_ONLY_METRICS",
    "TARGET_ONLY_METRICS",
    "inapplicable_metrics",
    "merge_adjustment",
    "repair_criteria_for_goal",
]
