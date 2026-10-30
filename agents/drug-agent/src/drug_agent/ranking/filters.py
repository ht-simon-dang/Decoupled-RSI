"""Hard-constraint evaluation, applied *before* Pareto ranking (§10 step 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from ..contracts.brief import Constraint

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class ConstraintOutcome:
    name: str
    passed: bool
    detail: str


def _get(props: dict[str, Any], key: str) -> Any:
    # Support dotted lookups like 'descriptors.mw'.
    cur: Any = props
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def evaluate_constraint(constraint: Constraint, props: dict[str, Any]) -> ConstraintOutcome:
    val = _get(props, constraint.property)
    op = constraint.op
    if val is None:
        return ConstraintOutcome(constraint.name, False, f"missing property {constraint.property!r}")
    try:
        if op in _OPS:
            ok = bool(_OPS[op](val, constraint.value))
        elif op == "in":
            choices = constraint.value if isinstance(constraint.value, list) else []
            ok = val in choices
        elif op == "not_in":
            choices = constraint.value if isinstance(constraint.value, list) else []
            ok = val not in choices
        elif op == "is_true":
            ok = bool(val) is True
        elif op == "is_false":
            ok = bool(val) is False
        else:
            return ConstraintOutcome(constraint.name, False, f"unknown op {op!r}")
    except TypeError as exc:
        return ConstraintOutcome(constraint.name, False, f"type error: {exc}")
    detail = f"{constraint.property}={val} {op} {constraint.value} -> {'pass' if ok else 'fail'}"
    return ConstraintOutcome(constraint.name, ok, detail)


class ConstraintEvaluation(NamedTuple):
    """One candidate's hard-filter verdict, with the vacuous case made visible.

    Unpacks as ``(passed, outcomes)`` — which is how every caller reads it — but
    carries ``evaluated`` beside the verdict, because "no rule rejected this
    candidate" and "no rule was applied to this candidate" are different facts
    and ADR-0011 forbids conflating them.

    ``passed`` stays ``True`` for an empty constraint list: the filter is a gate,
    and a gate with no rules must not reject anyone — a brief may legitimately
    declare no hard constraints. What it must not do is let a *reporter*
    republish that vacuous ``True`` as a measurement. A run whose constraints
    were lost in a refinement was observed publishing
    ``hard_constraint_pass_rate = 1.0`` with an empty failure map, i.e. a perfect
    score on zero checks. Anything computing a rate has to ask ``evaluated``
    (or ``n_evaluated``) first and report absence when it is false.
    """

    passed: bool
    outcomes: list[ConstraintOutcome]

    @property
    def evaluated(self) -> bool:
        """Whether any constraint was actually checked."""
        return bool(self.outcomes)

    @property
    def n_evaluated(self) -> int:
        """How many constraints were checked — the denominator of any pass rate."""
        return len(self.outcomes)

    @property
    def failures(self) -> list[ConstraintOutcome]:
        """The constraints this candidate was measured against and failed."""
        return [o for o in self.outcomes if not o.passed]


def evaluate_constraints(constraints: list[Constraint], props: dict[str, Any]) -> ConstraintEvaluation:
    """Apply every hard constraint to one candidate's properties.

    ``all([])`` is ``True``, so the verdict alone cannot tell a candidate that
    survived five checks from one that faced none; the returned
    :class:`ConstraintEvaluation` carries both.
    """
    outcomes = [evaluate_constraint(c, props) for c in constraints]
    return ConstraintEvaluation(passed=all(o.passed for o in outcomes), outcomes=outcomes)
