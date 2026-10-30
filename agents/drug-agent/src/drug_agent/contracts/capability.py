"""Capability profile contract — how the registry describes an adapter."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import SchemaModel
from .enums import CapabilityStatus, CostClass, Determinism


class CapabilityProfile(SchemaModel):
    """Self-description an adapter publishes to the capability registry."""

    capability_id: str = Field(description="Stable dotted id, e.g. 'chem.standardize'.")
    version: str = "1.0.0"
    title: str = ""
    description: str = ""

    input_schema_ref: str | None = Field(default=None, description="Name of the input contract.")
    output_schema_ref: str | None = Field(default=None, description="Name of the output contract.")

    determinism: Determinism = Determinism.DETERMINISTIC
    cost_class: CostClass = CostClass.CHEAP
    status: CapabilityStatus = CapabilityStatus.AVAILABLE

    runtime_requirements: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {'gpu': False, 'binary': 'vina', 'extra': 'sim'}.",
    )
    timeout_seconds: float | None = 60.0
    max_retries: int = 0

    license: str | None = None
    data_restrictions: str | None = None
    known_failure_modes: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)

    provides_mock: bool = False
    backend: str = Field(default="builtin", description="Selected backend name.")
    unavailable_reason: str | None = None
