"""Base model machinery shared by all versioned contracts.

Design decisions (see ADR-0002):

* ``extra="forbid"`` — unknown *critical* fields are rejected. Extensions must
  live under the explicit ``ext`` namespace, never as free-floating keys.
* Every contract carries ``schema_version``, a stable ``id`` and creation
  timestamp, sourced from the deterministic clock/ID factory.
* ``Provenance`` and ``EvidenceRef`` give every object a traceable lineage.
* ``Quantity`` couples a value with its unit, qualifier and trust label so a
  number can never be interpreted without its context.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..util import new_id, utcnow_iso
from .enums import Qualifier, TrustLabel

SCHEMA_VERSION = "1.0.0"


class SchemaModel(BaseModel):
    """Strict, versioned base for every contract object."""

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        ser_json_timedelta="iso8601",
        # We deliberately use fields like ``model_id`` / ``model_version``.
        protected_namespaces=(),
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Contract schema version.")
    ext: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit namespace for non-critical extension fields.",
    )

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


class IdentifiedModel(SchemaModel):
    """A :class:`SchemaModel` with a stable ID and creation timestamp."""

    id: str = Field(default="", description="Stable prefixed identifier.")
    created_at: str = Field(default_factory=utcnow_iso, description="ISO-8601 UTC creation time.")

    id_prefix: str = "obj"

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            # Assign after construction so subclasses can override id_prefix.
            object.__setattr__(self, "id", new_id(self.id_prefix))


class Provenance(SchemaModel):
    """Where a value/object came from and how to reproduce it."""

    source: str = Field(description="Human/system origin, e.g. 'rdkit', 'chembl_fixture'.")
    source_version: str | None = Field(default=None, description="Version of the source.")
    method: str | None = Field(default=None, description="Capability/algorithm that produced it.")
    method_version: str | None = None
    snapshot_id: str | None = Field(default=None, description="Data snapshot identifier.")
    content_hash: str | None = Field(default=None, description="Content hash of the source data.")
    trust_label: TrustLabel = TrustLabel.CALCULATED
    notes: str | None = None


class EvidenceRef(SchemaModel):
    """A lightweight pointer to a full EvidenceItem (see contracts.evidence)."""

    evidence_id: str
    trust_label: TrustLabel = TrustLabel.CURATED
    relation: str | None = Field(default=None, description="How it relates, e.g. 'supports'.")


class Quantity(SchemaModel):
    """A scientific value with unit, qualifier, uncertainty and trust label.

    ``original_value`` / ``original_unit`` preserve the source representation so
    unit normalization is never lossy.
    """

    value: float | None = Field(default=None, description="Normalized numeric value.")
    unit: str | None = Field(default=None, description="Normalized unit (UCUM-ish token).")
    qualifier: Qualifier = Qualifier.EXACT
    value_high: float | None = Field(default=None, description="Upper bound for ranges.")
    std: float | None = Field(default=None, description="Standard deviation / 1-sigma uncertainty.")
    ci_low: float | None = None
    ci_high: float | None = None
    original_value: str | None = Field(default=None, description="Verbatim source value.")
    original_unit: str | None = None
    trust_label: TrustLabel = TrustLabel.CALCULATED

    @property
    def is_missing(self) -> bool:
        return self.value is None and self.value_high is None
