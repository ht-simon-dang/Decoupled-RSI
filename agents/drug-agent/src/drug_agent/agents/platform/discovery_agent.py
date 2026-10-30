"""Scientific discovery — hypotheses, and the experiments that would test them.

Two actions on one agent because they are two halves of the same loop: what the
run currently believes, and the cheapest observation that would move that belief.
Everything produced here is labelled HYPOTHESIZED or PROPOSED, never OBSERVED.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...contracts.agent import AgentRequest, AgentResult
from ...contracts.base import EvidenceRef
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism, TrustLabel
from ...contracts.experiment import ExperimentProposal
from . import PlatformAgent

if TYPE_CHECKING:
    from ...orchestration.context import RunContext


class ScientificDiscoveryAgent(PlatformAgent):
    agent_id = "agent.discovery"
    actions = {
        "generate_hypotheses": "Propose testable hypotheses from evidence.",
        "propose_experiments": "Propose next experiments to reduce uncertainty.",
    }

    def describe_capability(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.agent_id,
            version="1.0.0",
            title="Scientific discovery",
            description="Hypothesis generation and experiment proposal.",
            input_schema_ref="EvidenceItem",
            output_schema_ref="ExperimentProposal",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE,
            backend="rules+fake_llm",
            validation_checks=["hypotheses labelled HYPOTHESIZED", "proposals labelled PROPOSED"],
        )

    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        if request.action == "generate_hypotheses":
            return self._hypotheses(request, ctx)
        return self._experiments(request, ctx)

    def _hypotheses(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        evidence = ctx.memory.list_evidence(run_id=ctx.run.id)
        target = request.inputs.get("target_name", "the target")
        hyps = [
            f"Lipophilicity within a moderate window improves potency against {target} "
            "(testable via matched-pair analysis).",
            "Aryl-substituted analogues of active seeds retain activity while modulating ADMET.",
        ]
        for i, h in enumerate(hyps):
            ctx.store.put_object(
                "hypothesis",
                {
                    "id": f"hyp_{i}",
                    "statement": h,
                    "trust_label": str(TrustLabel.HYPOTHESIZED),
                    "evidence_count": len(evidence),
                },
                run_id=ctx.run.id,
            )
        return self._result(
            request,
            payload={"n_hypotheses": len(hyps)},
            metrics={"n_hypotheses": float(len(hyps))},
            explanation=f"Generated {len(hyps)} HYPOTHESIZED statements from {len(evidence)} evidence items.",
        )

    def _experiments(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        portfolios = ctx.store.list_objects("portfolio", run_id=ctx.run.id)
        shortlist = portfolios[0]["shortlist"] if portfolios else []
        n = 0
        for m in shortlist[:6]:
            cid = m["candidate_id"]
            proposal = ExperimentProposal(
                run_id=ctx.run.id,
                candidate_id=cid,
                title=f"Confirm primary activity for {cid}",
                hypothesis="Predicted activity reflects true binding affinity.",
                assay_type="biochemical_IC50",
                rationale="Highest-priority uncertainty is the predicted potency; a direct "
                "biochemical IC50 provides an OBSERVED anchor.",
                expected_information_gain="Reduces activity-model uncertainty and applicability risk.",
                priority=0.8,
                reduces_uncertainty_on=["pIC50"],
                evidence=[
                    EvidenceRef(evidence_id=e["id"]) for e in ctx.memory.list_evidence(run_id=ctx.run.id)[:2]
                ],
            )
            ctx.store.put_object("experiment", proposal, run_id=ctx.run.id)
            n += 1
        return self._result(
            request,
            payload={"n_proposals": n},
            metrics={"n_proposals": float(n)},
            explanation=f"Proposed {n} PROPOSED next-experiments for the shortlist.",
        )
