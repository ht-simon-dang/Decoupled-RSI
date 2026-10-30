"""Evaluation agent — the independent judge.

It consumes the CHEMICAL agent's real, tool-computed results and scores them
against the master agent's success criteria. Every number here is derived from
the backend scorecards/portfolio (RDKit + the activity model + the ranking
plane); the evaluation agent computes, it never predicts. The master LLM then
*interprets* this report to decide accept / refine / replan.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...llm.bridge import BackendRun
from ...llm.schemas import SuccessCriterion
from . import ReasoningAgent

# Endpoint display metadata.
_ENDPOINT_UNITS = {"pIC50": "", "sa_score": "", "qed": "", "hERG": "", "restoration_score": ""}


def _endpoint_value(scorecard: dict[str, Any], name: str) -> float | None:
    for e in scorecard.get("endpoints", []):
        if e.get("endpoint") == name and not e.get("missing"):
            v = e.get("raw_value")
            return float(v) if v is not None else None
    return None


def _endpoint_in_domain(scorecard: dict[str, Any], name: str) -> bool | None:
    for e in scorecard.get("endpoints", []):
        if e.get("endpoint") == name:
            return e.get("applicability") == "IN_DOMAIN"
    return None


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def _max(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(max(xs), 4) if xs else None


class MetricResult(BaseModel):
    """One computed metric, or one scored success criterion.

    ``available`` is the flag that keeps "measured" apart from "missing": a
    metric with nothing to measure reports ``available=False, value=None``, never
    a substitute ``0.0``. Zero is a result — it says the thing was measured and
    came out at the bottom of its range — and a run that reports it for an
    absent measurement makes a failure indistinguishable from a gap. The
    capability envelope already reports absence this way (``MetricRange.available``);
    this is the same rule on the evaluation side.
    """

    model_config = ConfigDict(protected_namespaces=())
    key: str
    label: str
    value: float | None = None
    unit: str = ""
    target: str | None = None
    #: ``True`` when a criterion was measured and met, ``False`` when it was
    #: measured and missed, ``None`` when it could not be scored at all.
    met: bool | None = None
    #: ``False`` when no measurement exists for this metric in this run.
    available: bool = True


class TopCandidate(BaseModel):
    candidate_id: str
    smiles: str
    scaffold: str | None = None
    pIC50: float | None = None
    qed: float | None = None
    sa_score: float | None = None
    hERG: float | None = None
    restoration_score: float | None = None
    utility: float | None = None
    pareto_rank: int | None = None
    decision: str | None = None
    in_domain: bool | None = None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    run_id: str
    state: str
    #: How many scorecards exist — i.e. candidates that reached scoring. When the
    #: pipeline stops before ranking this is 0 even though structures were built,
    #: which read as "the generator produced nothing" and sent the critic hunting
    #: for a generation bug. ``n_pool_candidates`` is the honest denominator.
    n_candidates: int = 0
    #: Candidates the pipeline actually built, scored or not.
    n_pool_candidates: int = 0
    #: Why the pipeline stopped, when it did not finish. Authoritative: it is the
    #: engine's own account of the failing task, not an inference from the nulls.
    failure_reason: str | None = None
    n_passed_filters: int = 0
    n_pareto: int = 0
    n_clusters: int = 0
    shortlist_size: int = 0
    requested_shortlist: int = 0
    diversity: float | None = None
    hypervolume: float | None = None
    decision_counts: dict[str, int] = Field(default_factory=dict)
    constraint_failure_counts: dict[str, int] = Field(default_factory=dict)
    metrics: list[MetricResult] = Field(default_factory=list)
    criteria: list[MetricResult] = Field(default_factory=list)
    top_candidates: list[TopCandidate] = Field(default_factory=list)
    headline_utility: float | None = None
    n_criteria_met: int = 0
    n_criteria_total: int = 0
    #: How many of ``n_criteria_total`` could actually be scored. ``n_criteria_met``
    #: is a fraction of *this*, not of the total: the rest were never measured, and
    #: reading them as misses invents evidence the run does not have.
    n_criteria_scored: int = 0
    #: The metric key of every criterion that could not be scored, in plan order.
    unmeasured_criteria: list[str] = Field(default_factory=list)
    all_criteria_met: bool = False
    notes: list[str] = Field(default_factory=list)

    def compact_json(self) -> str:
        """A trimmed view for the critique/refine prompts (authoritative numbers)."""
        return self.model_dump_json(
            indent=2,
            include={
                "state",
                "failure_reason",
                "n_candidates",
                "n_pool_candidates",
                "n_passed_filters",
                "n_pareto",
                "n_clusters",
                "shortlist_size",
                "requested_shortlist",
                "diversity",
                "decision_counts",
                "constraint_failure_counts",
                "n_criteria_met",
                "n_criteria_scored",
                "unmeasured_criteria",
                "criteria",
                "top_candidates",
                "notes",
            },
        )


def _parse_target(target: str) -> tuple[str, float] | None:
    m = re.search(r"(<=|>=|==|!=|<|>)?\s*([-+]?\d*\.?\d+)", target or "")
    if not m:
        return None
    op = m.group(1) or ">="
    try:
        return op, float(m.group(2))
    except ValueError:
        return None


def _compare(value: float, op: str, threshold: float) -> bool:
    return {
        "<": value < threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
        "==": abs(value - threshold) < 1e-9,
        "!=": abs(value - threshold) >= 1e-9,
    }.get(op, value >= threshold)


def _arm_warnings(run: Any) -> list[str]:
    """One note per distinct thing an agent said produced nothing, deduped.

    A broken batch arm writes the same sentence for every candidate; the model
    needs the fact once, not ninety times.
    """
    seen: dict[str, str] = {}
    for record in getattr(run, "agent_warnings", None) or []:
        message = str((record or {}).get("message") or "").strip()
        if not message:
            continue
        task = str((record or {}).get("task") or "a pipeline step")
        seen.setdefault(message, f"{task}: {message}")
    return list(seen.values())[:6]


class EvaluationAgent(ReasoningAgent):
    """The independent judge.

    Scores the pipeline's output against the master agent's success criteria
    using only numbers the tool plane produced. Separated from
    :class:`~drug_agent.agents.llm.chemical_agent.ChemicalAgent` so that
    nothing both produces a number and grades it. It computes; it does not
    predict, and it does not call a model.
    """

    agent_id = "agent.llm.evaluation"
    llm_backed = False
    actions = {"evaluate": "Score a completed run against the plan's success criteria."}

    def evaluate(
        self, run: BackendRun, criteria: list[SuccessCriterion], requested_shortlist: int
    ) -> EvaluationReport:
        """Independently score a backend run against the plan's success criteria."""
        portfolio = run.portfolio or {}
        scorecards = run.scorecards
        by_id = {sc.get("candidate_id"): sc for sc in scorecards}

        shortlist_members = portfolio.get("shortlist", [])
        shortlist_ids = [m.get("candidate_id") for m in shortlist_members]
        shortlist_scs = [by_id[i] for i in shortlist_ids if i in by_id]

        # Decision + constraint diagnostics over the whole pool.
        decision_counts: dict[str, int] = {}
        constraint_failures: dict[str, int] = {}
        n_passed = 0
        for sc in scorecards:
            d = str(sc.get("decision"))
            decision_counts[d] = decision_counts.get(d, 0) + 1
            if sc.get("hard_constraints_passed"):
                n_passed += 1
            for cname, ok in (sc.get("hard_constraint_outcomes") or {}).items():
                if not ok:
                    constraint_failures[cname] = constraint_failures.get(cname, 0) + 1

        def _short_vals(name: str) -> list[float]:
            return [v for sc in shortlist_scs if (v := _endpoint_value(sc, name)) is not None]

        def _pool_vals(name: str) -> list[float]:
            return [v for sc in scorecards if (v := _endpoint_value(sc, name)) is not None]

        n_all = len(scorecards)
        # A hard-constraint pass rate needs a denominator of *checks*, not of
        # candidates. Every scorecard records one outcome per declared constraint,
        # so a brief that arrived with none — or whose constraints a refinement
        # quietly dropped — yields zero checks across the whole pool, and
        # `n_passed / n_all` then publishes a flawless 1.0 measured on nothing.
        n_constraint_checks = sum(len(sc.get("hard_constraint_outcomes") or {}) for sc in scorecards)
        n_clusters_raw = portfolio.get("n_clusters")
        short_indomain: list[bool] = [
            flag for sc in shortlist_scs if (flag := _endpoint_in_domain(sc, "pIC50")) is not None
        ]

        # --- metric registry: key -> (label, value, unit) --------------------
        registry: dict[str, tuple[str, float | None, str]] = {
            "max_pIC50": ("Best predicted pIC50 (shortlist)", _max(_short_vals("pIC50")), ""),
            "median_pIC50": ("Median predicted pIC50 (shortlist)", _median(_short_vals("pIC50")), ""),
            "qed_median": ("Median QED (shortlist)", _median(_short_vals("qed")), ""),
            "sa_median": ("Median SA score (shortlist)", _median(_short_vals("sa_score")), ""),
            "herg_median": ("Median hERG proxy (shortlist)", _median(_short_vals("hERG")), ""),
            "restoration_median": (
                "Median restoration score (shortlist)",
                _median(_short_vals("restoration_score")),
                "",
            ),
            "hard_constraint_pass_rate": (
                "Hard-constraint pass rate (pool)",
                round(n_passed / n_all, 4) if n_all and n_constraint_checks else None,
                "",
            ),
            "mean_pairwise_distance": (
                "Shortlist diversity (1 - mean Tanimoto)",
                portfolio.get("diversity"),
                "",
            ),
            "n_scaffold_clusters": (
                "Distinct scaffold clusters",
                float(n_clusters_raw) if n_clusters_raw is not None else None,
                "",
            ),
            "shortlist_filled": (
                "Shortlist filled (fraction of requested)",
                round(len(shortlist_ids) / requested_shortlist, 4) if requested_shortlist else None,
                "",
            ),
            "in_domain_fraction": (
                "In-applicability-domain fraction (shortlist)",
                round(sum(short_indomain) / len(short_indomain), 4) if short_indomain else None,
                "",
            ),
        }

        # `None` above is deliberate everywhere a substitute `0.0` used to sit.
        # An empty shortlist did not score 0% in-domain and a pipeline that never
        # ranked did not produce 0 clusters; there is simply no measurement, and
        # `available=False` is how the report says so.
        metrics = [
            MetricResult(key=k, label=lbl, value=val, unit=unit, available=val is not None)
            for k, (lbl, val, unit) in registry.items()
        ]

        # --- score each success criterion -----------------------------------
        # Three ways a criterion fails to produce a verdict, and they are not the
        # same fact: the platform has no such metric, the metric exists but this
        # run produced no value for it, or the target string cannot be parsed into
        # a comparison. None of them is a miss. The old code marked all three
        # `met=None` and then counted, listed and reported *none* of them — a
        # criterion that was never measured simply disappeared from the report
        # while still holding ACCEPT out of reach, which is the exact conflation
        # ADR-0011 forbids.
        scored: list[MetricResult] = []
        unknown_metrics: list[str] = []
        unmeasured: list[str] = []
        unparseable: list[str] = []
        unscored_keys: list[str] = []
        n_met = 0
        n_scored = 0
        for crit in criteria:
            key = crit.metric.strip()
            entry = registry.get(key)
            if entry is None:
                # The master agent filters these out before dispatch; one arriving
                # here means the vocabulary drifted, and a silent `met=None` would
                # make the run unacceptable for a reason nobody could see.
                unknown_metrics.append(key)
            label = entry[0] if entry else key
            value = entry[1] if entry else None
            unit = entry[2] if entry else ""
            met: bool | None = None
            parsed = _parse_target(crit.target)
            if value is None:
                if entry is not None:
                    unmeasured.append(key)
            elif parsed is None:
                unparseable.append(f"{key} (target {crit.target!r})")
            else:
                met = _compare(value, parsed[0], parsed[1])
                n_scored += 1
                n_met += 1 if met else 0
            if met is None:
                unscored_keys.append(key)
            scored.append(
                MetricResult(
                    key=key,
                    label=label,
                    value=value,
                    unit=unit,
                    target=crit.target,
                    met=met,
                    available=value is not None,
                )
            )

        # --- top candidates (from shortlist) --------------------------------
        scaffold_by_id = {c.get("id"): c.get("scaffold") for c in run.candidates}
        top: list[TopCandidate] = []
        for m in shortlist_members:
            cid = m.get("candidate_id")
            sc = by_id.get(cid, {})
            top.append(
                TopCandidate(
                    candidate_id=cid or "",
                    smiles=m.get("canonical_smiles", ""),
                    scaffold=scaffold_by_id.get(cid),
                    pIC50=_endpoint_value(sc, "pIC50"),
                    qed=_endpoint_value(sc, "qed"),
                    sa_score=_endpoint_value(sc, "sa_score"),
                    hERG=_endpoint_value(sc, "hERG"),
                    restoration_score=_endpoint_value(sc, "restoration_score"),
                    utility=m.get("utility"),
                    pareto_rank=m.get("pareto_rank"),
                    decision=str(sc.get("decision")) if sc.get("decision") else None,
                    in_domain=_endpoint_in_domain(sc, "pIC50"),
                )
            )

        notes: list[str] = []
        if unknown_metrics:
            notes.append(
                "These success criteria name metrics this platform cannot compute, so they can never "
                f"be marked met: {', '.join(sorted(set(unknown_metrics)))}."
            )
        if unmeasured:
            notes.append(
                f"{len(unmeasured)} of {len(criteria)} success criteria were NOT MEASURED this run — "
                "the metric exists but nothing produced a value for it, so they are unscored, not "
                f"missed. Metrics with no value: {', '.join(sorted(set(unmeasured)))}."
            )
        if unparseable:
            notes.append(
                "These success criteria carry a target no comparison could be read out of, so they "
                f"were left unscored rather than counted as missed: {', '.join(sorted(set(unparseable)))}."
            )
        if n_all and not n_constraint_checks:
            notes.append(
                f"No hard constraint was evaluated for any of the {n_all} candidates, so the "
                "hard-constraint pass rate is UNAVAILABLE rather than 1.0 — every candidate cleared a "
                "filter that checked nothing. If the brief declared hard constraints, they were lost "
                "before the filter stage."
            )
        if run.dropped_seeds:
            notes.append(
                f"RDKit rejected {len(run.dropped_seeds)} invalid proposed seed SMILES: {run.dropped_seeds}"
            )
        if not portfolio:
            if run.error:
                notes.append(
                    f"The pipeline did not reach ranking, so nothing was scored. The engine's "
                    f"reason: {run.error} This is the cause to address — the empty metrics are "
                    f"its consequence, not a separate problem."
                )
            else:
                notes.append("No portfolio was produced (pipeline did not reach ranking).")
        if shortlist_scs and not _short_vals("pIC50"):
            pool_ic = _pool_vals("pIC50")
            if pool_ic:
                notes.append(
                    "Shortlist has no in-applicability-domain pIC50 — the top candidates sit outside the "
                    "activity model's domain, so potency is unconfirmed (relevant pool values exist)."
                )
            else:
                notes.append("No pIC50 predictions available for this run (target/objective mismatch).")
        if constraint_failures:
            worst = max(constraint_failures.items(), key=lambda kv: kv[1])
            notes.append(f"Most-violated hard constraint: {worst[0]} ({worst[1]}/{n_all} candidates failed).")
        # What the pipeline's own agents said produced nothing. These are the
        # CAUSE behind an unmeasured criterion, and until they crossed the bridge
        # the critique saw only the symptom: an arm the master agent had switched
        # on could fail in full view of the pipeline and in silence to the model.
        for warning in _arm_warnings(run):
            notes.append(warning)

        utilities = [m.get("utility") for m in shortlist_members if m.get("utility") is not None]

        return EvaluationReport(
            run_id=run.run_id,
            state=run.state,
            n_candidates=n_all,
            n_pool_candidates=len(run.candidates),
            failure_reason=run.error,
            n_passed_filters=n_passed,
            n_pareto=int(portfolio.get("n_pareto", 0)),
            n_clusters=int(portfolio.get("n_clusters", 0)),
            shortlist_size=len(shortlist_ids),
            requested_shortlist=requested_shortlist,
            diversity=portfolio.get("diversity"),
            hypervolume=portfolio.get("hypervolume"),
            decision_counts=decision_counts,
            constraint_failure_counts=constraint_failures,
            metrics=metrics,
            criteria=scored,
            top_candidates=top,
            headline_utility=round(max(utilities), 4) if utilities else None,
            n_criteria_met=n_met,
            n_criteria_total=len(criteria),
            n_criteria_scored=n_scored,
            unmeasured_criteria=unscored_keys,
            # Unchanged on purpose: an unmeasured criterion still blocks ACCEPT.
            # Absence is not evidence of success any more than it is of failure.
            all_criteria_met=bool(criteria) and n_met == len(criteria),
            notes=notes,
        )
