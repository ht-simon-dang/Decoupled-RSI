"""The common ScientificAgent interface and the agent taxonomy.

Every platform/specialist agent implements the same six-method contract so the
orchestrator can treat them uniformly and the evaluator can validate their
output independently of their producer.

:class:`AgentKind` names the three planes an agent can belong to. It lives here,
beside the interface, so an agent class declares its own plane and nothing has to
keep a second list of which id belongs where.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..contracts.agent import AgentPlan, AgentRequest, AgentResult, PlanStep, ValidationIssue
from ..contracts.capability import CapabilityProfile
from ..contracts.enums import StrEnum

if TYPE_CHECKING:
    from ..orchestration.context import RunContext


class AgentKind(StrEnum):
    """Which plane an agent belongs to.

    ``PLATFORM``   Cross-cutting roles that shape a run: literature intelligence,
                   hypothesis generation, orchestration, memory, evaluation.
    ``SPECIALIST`` Scientific roles that own one domain: molecules, structure,
                   synthesis, safety, perturbation. These do the pipeline work.
    ``REASONING``  LLM-backed roles. Only these ever call a language model.
    """

    PLATFORM = "PLATFORM"
    SPECIALIST = "SPECIALIST"
    REASONING = "REASONING"


class AgentError(RuntimeError):
    """Raised for unrecoverable agent failures (mapped to FAILED_FINAL)."""


class RetryableAgentError(RuntimeError):
    """Raised for transient failures (mapped to FAILED_RETRYABLE)."""


class ScientificAgent(ABC):
    """Base class for all agents."""

    agent_id: str = "agent.base"
    version: str = "1.0.0"
    #: The plane this agent belongs to. Set on the per-plane base classes.
    kind: AgentKind = AgentKind.SPECIALIST
    #: Actions this agent accepts, mapped to a human description.
    actions: dict[str, str] = {}

    @abstractmethod
    def describe_capability(self) -> CapabilityProfile:
        """Publish this agent's capability profile to the registry."""

    def validate_request(self, request: AgentRequest) -> list[ValidationIssue]:
        """Return validation issues for a request (empty = valid)."""
        issues: list[ValidationIssue] = []
        if request.action not in self.actions:
            issues.append(
                ValidationIssue(
                    code="unknown_action",
                    message=f"{self.agent_id} does not support action {request.action!r}",
                )
            )
        return issues

    def plan(self, request: AgentRequest, ctx: RunContext) -> AgentPlan:
        """Produce an inspectable plan. Default: single step from the action."""
        return AgentPlan(
            request_id=request.id,
            agent_id=self.agent_id,
            rationale=f"{self.agent_id}:{request.action}",
            steps=[
                PlanStep(
                    order=1,
                    description=self.actions.get(request.action, request.action),
                    capability_id=self.agent_id,
                )
            ],
        )

    @abstractmethod
    def execute(self, request: AgentRequest, ctx: RunContext) -> AgentResult:
        """Perform the action and return a typed, schema-valid result."""

    def validate_result(
        self, request: AgentRequest, result: AgentResult, ctx: RunContext
    ) -> list[ValidationIssue]:
        """Scientific/consistency validation of a result (empty = valid).

        Overridden by agents to enforce their specific invariants. This runs in
        the VALIDATING task state, independent of ``execute``.
        """
        return []

    def explain(self, result: AgentResult) -> str:
        """Human-readable explanation derived from the structured result."""
        return result.explanation or f"{self.agent_id} completed {result.action}."

    # --- helpers -----------------------------------------------------------
    def _result(self, request: AgentRequest, **kw) -> AgentResult:
        return AgentResult(request_id=request.id, agent_id=self.agent_id, action=request.action, **kw)
