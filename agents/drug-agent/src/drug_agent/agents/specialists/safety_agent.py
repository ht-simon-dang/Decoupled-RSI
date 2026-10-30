"""Toxicity & safety prediction — structural alerts + pretrained tox endpoints.

The one specialist that emits several predictions per candidate, one per toxicity
endpoint. Its :meth:`validate_result` is deliberately independent of its own
execute path: probabilities are re-read from the store and range-checked there,
so a bug in scoring cannot also silence the check on scoring.

Two things about the batch are this agent's own. :meth:`prepare` scores the whole
candidate list through ADMET-AI in a single pass, because that model's cost is
dominated by fixed per-call setup and a per-candidate loop would turn seconds into
minutes against a 120 s task budget. And when ADMET-AI is absent the batch still
runs — structural alerts are real without it — carrying a warning that names what
produced nothing, the same treatment the structure agent gives docking.

Absence of signal is not a safety claim, and every profile carries that
disclaimer — the platform evaluator enforces it. An endpoint that was never
evaluated is not absence of signal either, which is why
:mod:`drug_agent.tools.safety_tool` reports it as a flag rather than a blank.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...contracts.agent import ValidationIssue
from ...contracts.base import Quantity
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import TrustLabel
from ...contracts.predictions import Prediction
from ...tools.safety_tool import SafetyAdapter
from . import Assessment, CandidateAssessmentAgent, unavailable_note

if TYPE_CHECKING:
    from ...contracts.agent import AgentRequest
    from ...orchestration.context import RunContext


class SafetyAgent(CandidateAssessmentAgent):
    agent_id = "agent.safety"
    actions = {"assess_safety": "RDKit structural alerts + pretrained ADMET-AI tox endpoints."}
    batch_label = "safety"
    count_key = "n_assessed"

    def __init__(self, *, settings: Any = None) -> None:
        self.tool = SafetyAdapter(settings=settings)

    def describe_capability(self) -> CapabilityProfile:
        p = self.tool.profile
        p.capability_id = self.agent_id
        p.title = "Toxicity & safety prediction"
        return p

    def prepare(self, request: AgentRequest, ctx: RunContext) -> Any:
        """Score the whole batch through the endpoint models once.

        Returns how many molecules the panel produced a profile for, so
        :meth:`batch_metrics` can report coverage rather than leaving a reader to
        assume it was total. Never raises :class:`SkipBatch`: unlike the
        perturbation agent, this one still has a real computation to do without
        its optional backend.
        """
        smiles = [c["canonical_smiles"] for c in self.candidates(ctx)]
        return self.tool.prefetch(smiles)

    def assess_candidate(self, candidate: dict, prep: Any, ctx: RunContext) -> Assessment:
        prof = self.tool.assess(candidate["id"], candidate["canonical_smiles"])
        assessment = Assessment(objects=[("safety", prof)])
        for ep in prof.endpoints:
            if ep.score is None:
                continue
            assessment.predictions.append(
                Prediction(
                    candidate_id=candidate["id"],
                    molecule_inchikey=candidate.get("inchikey"),
                    endpoint=ep.endpoint,
                    kind="safety",
                    quantity=Quantity(value=ep.score, unit="probability", trust_label=TrustLabel.PREDICTED),
                    applicability=ep.applicability,
                    calibrated=ep.calibrated,
                    model_id=ep.model_id,
                    model_version=ep.model_version,
                    trust_label=TrustLabel.PREDICTED,
                    limitation=ep.limitation,
                )
            )
        return assessment

    def batch_metrics(self, prep: Any, assessments: list[Assessment]) -> dict[str, float]:
        """How much of the batch the endpoint models actually scored.

        Reported only when the panel is wired: a zero from a deployment that never
        ran it would read as "nothing could be scored" rather than "nothing was
        asked", and the warning below is what says which happened.
        """
        if not self.tool.panel.available():
            return {}
        return {"n_tox_panel_scored": float(prep or 0)}

    def batch_warnings(self, prep: Any) -> list[str]:
        if not self.tool.panel.available():
            return [unavailable_note(self.tool.panel)]
        return []

    def explain_batch(self, n: int, prep: Any) -> str:
        return f"Safety-assessed {n} candidates via {self.tool.backend} (no-signal is NOT a safety claim)."

    def validate_result(self, request, result, ctx) -> list[ValidationIssue]:
        # Independent check: safety probabilities must be within [0,1].
        issues = []
        for p in ctx.store.list_objects("prediction", run_id=ctx.run.id):
            if p.get("kind") == "safety":
                v = (p.get("quantity") or {}).get("value")
                if v is not None and not (0.0 <= v <= 1.0):
                    issues.append(
                        ValidationIssue(code="prob_range", message=f"safety prob out of range: {v}")
                    )
        return issues
