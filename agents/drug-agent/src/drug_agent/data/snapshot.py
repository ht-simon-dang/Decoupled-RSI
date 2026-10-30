"""Versioned, content-hashed data snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..util import content_hash, utcnow_iso


@dataclass
class DataSnapshot:
    name: str
    source: str
    kind: str
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)
    license: str = "unknown"

    @property
    def content_hash(self) -> str:
        return content_hash({"source": self.source, "kind": self.kind, "records": self.records})

    @property
    def snapshot_id(self) -> str:
        return f"{self.name}@{self.content_hash[:19]}"

    def data_card(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "kind": self.kind,
            "n_records": len(self.records),
            "license": self.license,
            "content_hash": self.content_hash,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
