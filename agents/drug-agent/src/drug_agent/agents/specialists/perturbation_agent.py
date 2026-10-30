"""Single-cell & organoid perturbation — unavailable, and loudly so.

The disease signature this agent used to score against is real. The score was
not: the candidate's effect on each gene was a hash of the gene name and the
SMILES (see :mod:`drug_agent.tools.singlecell_tool`). Under make-it-or-fail that
computation is gone, so this agent produces nothing and says why.

This is the most consequential of the removals. ``restoration_score`` is the
**primary objective of the phenotype brief**, so a phenotype program now fails
to produce a ranked portfolio instead of ranking on a hash. That is the intended
outcome: the program was never measuring phenotype rescue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...contracts.agent import AgentRequest, ValidationIssue
from ...contracts.base import Quantity
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import TrustLabel
from ...contracts.predictions import Prediction
from ...tools.singlecell_tool import SingleCellAdapter
from . import Assessment, CandidateAssessmentAgent, SkipBatch

if TYPE_CHECKING:
    from ...orchestration.context import RunContext


class PerturbationAgent(CandidateAssessmentAgent):
    agent_id = "agent.perturbation"
    actions = {"assess_perturbation": "Single-cell signature-reversal restoration score."}
    batch_label = "perturb"
    count_key = "n_scored"

    def __init__(self) -> None:
        self.tool = SingleCellAdapter()

    def describe_capability(self) -> CapabilityProfile:
        p = self.tool.profile
        p.capability_id = self.agent_id
        p.title = "Single-cell & organoid perturbation"
        return p

    def validate_result(self, request, result, ctx) -> list[ValidationIssue]:
        # Nothing may claim a restoration score while the tool is unavailable.
        if self.tool.available():
            return []
        leaked = [
            p
            for p in ctx.store.list_objects("prediction", run_id=ctx.run.id)
            if p.get("endpoint") == "restoration_score"
        ]
        if leaked:
            return [
                ValidationIssue(
                    code="unavailable_capability_emitted_output",
                    severity="error",
                    message=(
                        f"{len(leaked)} restoration_score predictions exist with no model to produce them"
                    ),
                )
            ]
        return []

    def prepare(self, request: AgentRequest, ctx: RunContext) -> Any:
        # Refuse before touching the batch: there is no perturbation-response
        # model, so no candidate can be scored. The dataset would load and the
        # signature would compute — that part is real — but scoring a molecule
        # against it is the step that has no implementation.
        if not self.tool.available():
            reason = self.tool.profile.unavailable_reason or "no perturbation-response model"
            raise SkipBatch(
                warning=f"perturbation scoring produced NOTHING: {reason}",
                explanation=(
                    "Perturbation assessment unavailable: no model relates a molecule to a "
                    "transcriptional response, so no restoration scores were produced."
                ),
            )
        fixture_path = request.inputs.get("sc_fixture", "fixtures/data/sc_perturbation.npz")
        ds = self.tool.load_fixture(fixture_path)
        return ds, self.tool.disease_signature(ds)

    def assess_candidate(self, candidate: dict, prep: Any, ctx: RunContext) -> Assessment:
        ds, signature = prep
        sig = self.tool.score_candidate(candidate["id"], candidate["canonical_smiles"], ds, signature)
        assessment = Assessment(objects=[("perturbation", sig)])
        if sig.restoration_score is not None:
            assessment.predictions.append(
                Prediction(
                    candidate_id=candidate["id"],
                    molecule_inchikey=candidate.get("inchikey"),
                    endpoint="restoration_score",
                    kind="property",
                    quantity=Quantity(
                        value=sig.restoration_score,
                        unit="cosine_reversal",
                        trust_label=TrustLabel.PREDICTED,
                    ),
                    applicability=sig.context_applicability,
                    model_id=sig.method,
                    model_version=sig.method_version,
                    trust_label=TrustLabel.PREDICTED,
                    limitation=sig.limitation,
                )
            )
        return assessment

    def batch_metrics(self, prep: Any, assessments: list[Assessment]) -> dict[str, float]:
        _ds, signature = prep
        return {"n_signature_genes": float(len(signature))}

    def explain_batch(self, n: int, prep: Any) -> str:
        ds, signature = prep
        return f"Scored {n} candidates against a {len(signature)}-gene disease signature ({ds.cell_context})."
