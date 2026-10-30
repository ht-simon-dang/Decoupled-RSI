"""Memory & knowledge — entity/relationship indexing with program isolation.

Turns a run's candidates and target into knowledge-graph entities and edges, so
what one run learned is addressable later without re-reading its object store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...contracts.agent import AgentRequest, AgentResult
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism
from . import PlatformAgent

if TYPE_CHECKING:
    from ...orchestration.context import RunContext


class MemoryKnowledgeAgent(PlatformAgent):
    agent_id = "agent.memory"
    actions = {"index_knowledge": "Build knowledge-graph relationships for the run."}

    def describe_capability(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.agent_id,
            version="1.0.0",
            title="Memory & knowledge",
            description="Entity/relationship indexing with program isolation.",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE,
            backend="relational",
        )

    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        target_id = request.inputs.get("target_id", "target")
        target_name = request.inputs.get("target_name", "target")
        ctx.memory.upsert_entity(target_id, "target", target_name, {"run_id": ctx.run.id})
        n_rel = 0
        for cand in ctx.store.list_objects("candidate", run_id=ctx.run.id):
            ctx.memory.upsert_entity(
                cand["id"],
                "candidate",
                cand.get("inchikey") or cand["id"],
                {"smiles": cand.get("canonical_smiles")},
            )
            ctx.memory.relate(cand["id"], "evaluated_against", target_id)
            n_rel += 1
        return self._result(
            request,
            payload={"n_relationships": n_rel},
            metrics={"n_relationships": float(n_rel)},
            explanation=f"Indexed {n_rel} candidate→target relationships.",
        )
