"""Run, Task and RunManifest contracts — the control-plane state."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import IdentifiedModel, SchemaModel
from .enums import ApprovalStatus, RunState, TaskState


class RetryPolicy(SchemaModel):
    max_attempts: int = Field(default=1, ge=1, le=20)
    backoff_seconds: float = Field(default=0.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = False


class Task(IdentifiedModel):
    """A single node in the run's task graph."""

    id_prefix: str = "task"
    run_id: str = ""
    name: str = Field(description="Task type, e.g. 'apply_hard_filters'.")
    capability_id: str | None = Field(default=None, description="Capability this task invokes.")
    state: TaskState = TaskState.PENDING
    depends_on: list[str] = Field(default_factory=list, description="Task IDs that must finish.")
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    attempt: int = 0
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float | None = None

    approval_gate: str | None = Field(default=None, description="Gate name if this awaits approval.")

    cache_key: str | None = None
    cache_hit: bool = False

    result_ref: str | None = Field(default=None, description="Artifact ref of AgentResult.")
    error: str | None = None
    error_type: str | None = None

    started_at: str | None = None
    finished_at: str | None = None
    superseded_by: str | None = None


class ApprovalGate(SchemaModel):
    name: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    task_id: str | None = None
    decided_by: str | None = None
    decided_at: str | None = None
    reason: str | None = None


class Budget(SchemaModel):
    tokens_limit: int | None = None
    tokens_spent: int = 0
    seconds_limit: float | None = None
    seconds_spent: float = 0.0
    external_calls_limit: int | None = None
    external_calls_spent: int = 0

    def remaining_tokens(self) -> float:
        return float("inf") if self.tokens_limit is None else max(0, self.tokens_limit - self.tokens_spent)


class Run(IdentifiedModel):
    """A single execution of a brief through the workflow."""

    id_prefix: str = "run"
    brief_id: str = ""
    program_id: str | None = None
    profile: str = "local"
    deterministic: bool = False
    state: RunState = RunState.CREATED

    task_ids: list[str] = Field(default_factory=list)
    gates: list[ApprovalGate] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)

    seed: int = 1729
    config_hash: str | None = None
    brief_hash: str | None = None

    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    tags: list[str] = Field(default_factory=list)


class TaskEvent(SchemaModel):
    """An append-only event in the run's task-event log."""

    ts: str
    run_id: str
    task_id: str | None = None
    event: str
    from_state: str | None = None
    to_state: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None


class RunManifest(IdentifiedModel):
    """Reproducibility manifest emitted with every run bundle."""

    id_prefix: str = "manifest"
    run_id: str = ""
    brief_id: str = ""
    profile: str = "local"
    deterministic: bool = False

    repo_revision: str | None = None
    repo_dirty: bool | None = None
    # Hash of the conda environment specification (environment.lock.yml when a
    # solved lock has been exported, else environment.yml).
    lock_hash: str | None = None
    # Only set when an operator runs the platform inside a container image of
    # their own. The project ships no image and no compose stack; the supported
    # environment is the conda environment in environment.yml.
    container_digest: str | None = None

    os: str | None = None
    platform: str | None = None
    python_version: str | None = None
    cpu: str | None = None
    gpu: str | None = None

    seeds: dict[str, int] = Field(default_factory=dict)
    config_hash: str | None = None
    data_snapshots: dict[str, str] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    env_vars_present: list[str] = Field(default_factory=list, description="Names only; no values.")

    started_at: str | None = None
    finished_at: str | None = None
    task_states: dict[str, str] = Field(default_factory=dict)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    capability_matrix: dict[str, str] = Field(default_factory=dict)
