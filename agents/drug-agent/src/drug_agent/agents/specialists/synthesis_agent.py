"""Synthetic accessibility, and a route when a real engine can search for one.

Two tools, and only the first is unconditional. ``synth.assess`` is RDKit's own
Ertl-Schuffenhauer SA score, which is what the ranking plane consumes as
``sa_score``. It used to be a home-made complexity proxy wearing that backend
name; :mod:`drug_agent.tools.synthesis_tool` records what the swap moved and by
how much.

``synth.route.aizynthfinder`` is the retrosynthesis ADR-0011 removed: the route
that used to be here was hash-derived and was deleted, leaving
``SynthesisPlan.routes`` empty "until a real retrosynthesis engine exists". One
does, so ``route_backend`` wires it in and the plan carries real routes to
purchasable precursors. It stays off by default because a tree search per
candidate is minutes of work the pipeline has never budgeted for, and because a
run that did not ask for routes must cost what it always cost.

Wired and absent, it is skipped with its reason recorded — never substituted, and
never allowed to turn "this deployment cannot search routes" into an empty route
list that reads like "this molecule has none".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...contracts.base import Quantity
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import ApplicabilityDomain, TrustLabel
from ...contracts.predictions import Prediction
from ...tools.retrosynthesis.aizynthfinder_tool import AiZynthFinderAdapter
from ...tools.synthesis_tool import SynthesisAdapter
from . import Assessment, CandidateAssessmentAgent, SkipBatch, unavailable_note

if TYPE_CHECKING:
    from ...contracts.agent import AgentRequest
    from ...contracts.synthesis import SynthesisPlan
    from ...orchestration.context import RunContext

#: Names ``route_backend`` accepts to switch retrosynthetic search on.
ROUTE_BACKENDS = frozenset({"auto", "aizynthfinder", AiZynthFinderAdapter.capability_id})

#: How many candidates in one batch get a real route search.
#:
#: A search is bounded at 120 s per molecule and this step runs on every candidate
#: that passed the hard filters — ninety of them in a routine cycle, which is up
#: to three hours for one arm of one cycle. An arm that expensive is one no run
#: can afford to switch on, which is a different way of being unreachable.
#:
#: So it is capped, in batch order, and `batch_warnings` says how many were left
#: out. Batch order is not a ranking and this cap is not a sample of the best
#: candidates: routes are searched here, before anything is ranked, so there is no
#: "best" to take yet. What the cap buys is an affordable, honest answer about
#: part of the batch instead of an unaffordable one about all of it.
DEFAULT_ROUTE_BUDGET = 12


class SynthesisAgent(CandidateAssessmentAgent):
    agent_id = "agent.synthesis"
    actions = {"assess_synthesis": "RDKit SA score (Ertl-Schuffenhauer); retrosynthetic route if wired."}
    batch_label = "synth"
    count_key = "n_assessed"

    def __init__(
        self,
        backend: str = "rdkit-sa",
        *,
        settings: Any = None,
        route_backend: str = "none",
        route_budget: int = DEFAULT_ROUTE_BUDGET,
    ) -> None:
        self.tool = SynthesisAdapter(backend=backend)
        wanted = (route_backend or "none").strip().lower()
        if wanted not in ROUTE_BACKENDS and wanted not in {"none", ""}:
            raise ValueError(
                f"unknown route backend {route_backend!r}; this agent drives: none, aizynthfinder"
            )
        self.router = AiZynthFinderAdapter(settings=settings) if wanted in ROUTE_BACKENDS else None
        #: Reasons the route arm produced nothing this batch. Collected rather
        #: than raised: see `_add_routes`.
        self._route_failures: list[str] = []
        self.route_budget = max(0, int(route_budget))
        #: How many candidates this batch wanted a route for but did not get one
        #: because the budget ran out. Reported, never silently dropped.
        self._routes_skipped = 0
        self._routes_searched = 0

    def describe_capability(self) -> CapabilityProfile:
        p = self.tool.profile
        p.capability_id = self.agent_id
        p.title = "Synthesizability & route optimization"
        if self.router is not None:
            p.backend = f"{p.backend},route={self.router.backend}"
        return p

    def prepare(self, request: AgentRequest, ctx: RunContext) -> Any:
        """Refuse the whole batch when there is no SA score to compute.

        ``synth.assess`` is UNAVAILABLE only when RDKit's SA_Score contribution
        cannot load, which is a broken install rather than a missing optional
        extra — but the honest outcome is still the perturbation agent's: skip the
        batch with the reason, so a program that ranks on ``sa_score`` loses the
        objective visibly instead of ranking on a stand-in.
        """
        # Per batch, not per process: a stale failure from a previous batch would
        # report a loss this one did not suffer.
        self._route_failures = []
        self._routes_skipped = 0
        self._routes_searched = 0
        if not self.tool.available():
            reason = self.tool.profile.unavailable_reason or "no synthetic-accessibility score"
            raise SkipBatch(
                warning=f"synth.assess produced NOTHING: {reason}",
                explanation=(
                    "Synthetic accessibility unavailable: RDKit's SA_Score contribution did "
                    "not load, so no sa_score predictions were produced."
                ),
            )
        return None

    def assess_candidate(self, candidate: dict, prep: Any, ctx: RunContext) -> Assessment:
        plan = self.tool.assess(candidate["id"], candidate["canonical_smiles"])
        self._add_routes(plan, candidate)
        assessment = Assessment(objects=[("synthesis", plan)])
        if plan.sa_score is not None:
            assessment.predictions.append(
                Prediction(
                    candidate_id=candidate["id"],
                    molecule_inchikey=candidate.get("inchikey"),
                    endpoint="sa_score",
                    kind="property",
                    quantity=Quantity(value=plan.sa_score, unit="sa", trust_label=TrustLabel.PREDICTED),
                    applicability=ApplicabilityDomain.IN_DOMAIN,
                    model_id="synth.sa.ertl",
                    # 2.0.0 with a renamed model: the endpoint kept its name while
                    # the number behind it changed backend, so the model id and
                    # version are the only place a stored Prediction can say that
                    # it is not comparable with one written before this change.
                    model_version="2.0.0",
                    trust_label=TrustLabel.PREDICTED,
                    limitation=plan.limitation,
                )
            )
        return assessment

    def _add_routes(self, plan: SynthesisPlan, candidate: dict) -> None:
        """Search real routes and merge them onto the SA plan, when one can be searched.

        The two tools answer different questions about the same molecule — how
        complex it looks, and how it might actually be made — so the search fills
        in the half of :class:`SynthesisPlan` the SA score leaves empty rather than
        replacing the object. ``building_block_availability`` comes across too: it
        is the searcher's own count of precursors in stock, not an estimate.
        """
        if self.router is None or not self.router.available():
            return
        if self._route_failures:
            # The engine is broken for this batch, not for this molecule: a
            # missing model file fails identically ninety times. Stop after the
            # first, and let `batch_warnings` report the one reason.
            return
        if self._routes_searched >= self.route_budget:
            self._routes_skipped += 1
            return
        self._routes_searched += 1
        # An OPTIONAL arm must never be able to fail the run. Measured: the master
        # agent switched route search on because the goal asked about makeability,
        # AiZynthFinder's model file was missing, the adapter raised, and the whole
        # cycle ended having scored nothing — 39 generated structures, 90
        # candidates, one broken side-arm, zero results. The SA score is what this
        # step must deliver; a route is what it may add. Losing the route costs the
        # route, and the reason is written onto the plan so the loss is visible.
        try:
            found = self.router.plan(candidate_id=candidate["id"], smiles=candidate["canonical_smiles"])
        except Exception as exc:
            reason = f"{self.router.capability_id} produced NOTHING: {type(exc).__name__}: {exc}"
            plan.risk_factors = list(dict.fromkeys([*plan.risk_factors, reason[:300]]))
            self._route_failures.append(reason)
            return
        plan.routes = list(found.routes)
        plan.best_route_id = found.best_route_id
        plan.building_block_availability = found.building_block_availability
        plan.risk_factors = list(dict.fromkeys([*plan.risk_factors, *found.risk_factors]))

    def batch_metrics(self, prep: Any, assessments: list[Assessment]) -> dict[str, float]:
        """How much of the batch a route search actually solved.

        Reported only when routes were searched for: a zero solved-fraction from a
        deployment that never searched would read as "nothing is makeable".

        The denominator is what was SEARCHED, not what was assessed. With a route
        budget those differ — twelve searches over ninety candidates — and
        dividing by ninety would cap the fraction at 0.13 however well the search
        did, which is the same lie the paragraph above refuses.
        """
        if self.router is None or not self.router.available() or not assessments:
            return {}
        if not self._routes_searched:
            return {}
        plans = [obj for a in assessments for store_type, obj in a.objects if store_type == "synthesis"]
        solved = sum(1 for plan in plans if any(route.feasible for route in plan.routes))
        searched = float(self._routes_searched)
        return {
            "n_routes_searched": searched,
            "n_candidates_assessed": float(len(plans)),
            "route_solved_fraction": round(solved / searched, 4),
        }

    def batch_warnings(self, prep: Any) -> list[str]:
        if self.router is not None and not self.router.available():
            return [unavailable_note(self.router)]
        # A route arm that was asked for, was reported available, and then failed
        # anyway. One line naming the first reason and the count, because the same
        # broken engine fails identically for every candidate in the batch.
        out: list[str] = []
        if self._route_failures:
            n = len(self._route_failures)
            out.append(f"route search produced NOTHING for {n} candidate(s): {self._route_failures[0]}")
        if self._routes_skipped:
            out.append(
                f"route search covered {self._routes_searched} of "
                f"{self._routes_searched + self._routes_skipped} candidates: the per-batch budget "
                f"is {self.route_budget}, and the remainder carry an SA score with no route"
            )
        return out

    def explain_batch(self, n: int, prep: Any) -> str:
        if self.router is not None and self.router.available():
            return f"Assessed synthesizability for {n} candidates, with routes from {self.router.backend}."
        return f"Assessed synthesizability for {n} candidates."
