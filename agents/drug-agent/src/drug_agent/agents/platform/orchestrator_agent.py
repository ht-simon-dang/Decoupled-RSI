"""Orchestrator — the control plane's companion inside the agent pool.

Reports on the run rather than driving it: the Engine owns execution, and this
agent exists so "how far has this got?" is answerable through the same
request/result contract as everything else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...contracts.agent import AgentRequest, AgentResult
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CapabilityStatus, CostClass, Determinism
from . import PlatformAgent

if TYPE_CHECKING:
    from ...orchestration.context import RunContext


class OrchestratorAgent(PlatformAgent):
    agent_id = "agent.orchestrator"
    actions = {"monitor": "Summarize run plan and progress."}

    def describe_capability(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.agent_id,
            version="1.0.0",
            title="Orchestrator / planning & monitoring",
            description="Planning and run monitoring (control-plane companion).",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.TRIVIAL,
            status=CapabilityStatus.AVAILABLE,
            backend="builtin",
        )

    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        tasks = ctx.store.list_tasks(ctx.run.id)
        by_state: dict[str, int] = {}
        for t in tasks:
            by_state[t["state"]] = by_state.get(t["state"], 0) + 1
        return self._result(
            request, payload={"task_states": by_state}, explanation=f"Run has {len(tasks)} tasks: {by_state}."
        )
