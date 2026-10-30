"""Agent interaction contracts: request, plan, result."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import EvidenceRef, IdentifiedModel, SchemaModel


class AgentRequest(IdentifiedModel):
    """A typed request handed to a :class:`ScientificAgent`."""

    id_prefix: str = "areq"
    run_id: str = ""
    task_id: str | None = None
    agent_id: str = Field(description="Target agent capability id.")
    action: str = Field(description="Named action within the agent.")
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    deterministic: bool = True
    seed: int = 1729
    correlation_id: str | None = None


class PlanStep(SchemaModel):
    order: int
    description: str
    capability_id: str | None = None
    expected_output: str | None = None


class AgentPlan(IdentifiedModel):
    """A structured, inspectable plan an agent produces before executing."""

    id_prefix: str = "aplan"
    request_id: str = ""
    agent_id: str = ""
    rationale: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    est_cost_class: str | None = None


class ValidationIssue(SchemaModel):
    code: str
    message: str
    severity: str = Field(default="error", description="'error' | 'warning' | 'info'.")
    field: str | None = None


class AgentResult(IdentifiedModel):
    """The typed result an agent returns. Payload is schema-validated per action."""

    id_prefix: str = "ares"
    request_id: str = ""
    agent_id: str = ""
    action: str = ""
    status: str = Field(default="ok", description="'ok' | 'partial' | 'error'.")

    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    explanation: str = ""

    tokens_used: int = 0
    seconds_used: float = 0.0
    external_calls: int = 0

    @property
    def ok(self) -> bool:
        return self.status != "error" and not any(i.severity == "error" for i in self.validation_issues)
