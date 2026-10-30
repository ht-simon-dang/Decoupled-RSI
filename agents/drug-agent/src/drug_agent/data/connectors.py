"""Data connectors behind a common interface.

Every connector is fixture-backed by default and performs NO network I/O. A
network-capable connector must call ``self._check_network(url)`` (deny-by-default
egress) and implement caching + rate limiting; none are enabled in the default
build. Connectors emit versioned, content-hashed :class:`DataSnapshot` objects.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..security.egress import EgressPolicy
from .snapshot import DataSnapshot

FIXTURE_ROOT = Path("fixtures/data")


class Connector(ABC):
    name: str = "base"
    kind: str = "generic"
    license: str = "synthetic_demo"
    network_capable: bool = False

    def __init__(self, fixture_root: Path | str = FIXTURE_ROOT, egress: EgressPolicy | None = None) -> None:
        self.fixture_root = Path(fixture_root)
        self.egress = egress or EgressPolicy.from_env()

    def _check_network(self, url: str) -> None:
        self.egress.check(url)  # raises unless explicitly allowed

    @abstractmethod
    def load(self, **kwargs: Any) -> DataSnapshot: ...


class LocalFilesConnector(Connector):
    name = "local_files"
    kind = "files"

    def load(self, path: str | None = None, **kwargs: Any) -> DataSnapshot:
        p = Path(path) if path else self.fixture_root
        records = [{"path": str(fp), "size": fp.stat().st_size} for fp in sorted(p.glob("*")) if fp.is_file()]
        return DataSnapshot(
            name="local_files", source=str(p), kind=self.kind, records=records, license=self.license
        )


class ChemblLikeConnector(Connector):
    name = "chembl"
    kind = "bioactivity"

    def load(self, target_id: str | None = None, **kwargs: Any) -> DataSnapshot:
        path = self.fixture_root / "seh_bioactivity.csv"
        records: list[dict[str, Any]] = []
        with path.open() as f:
            for row in csv.DictReader(f):
                if target_id and row.get("target_id") != target_id:
                    continue
                records.append(row)
        return DataSnapshot(
            name="chembl_bioactivity",
            source=f"fixture:{path.name}",
            kind=self.kind,
            records=records,
            license=self.license,
            metadata={"note": "synthetic activity labels; demo only"},
        )


class PubChemLikeConnector(Connector):
    name = "pubchem"
    kind = "compounds"

    def load(self, **kwargs: Any) -> DataSnapshot:
        path = self.fixture_root / "compound_library.csv"
        with path.open() as f:
            records = list(csv.DictReader(f))
        return DataSnapshot(
            name="pubchem_compounds",
            source=f"fixture:{path.name}",
            kind=self.kind,
            records=records,
            license=self.license,
        )


class PdbLikeConnector(Connector):
    name = "pdb"
    kind = "structures"

    def load(self, **kwargs: Any) -> DataSnapshot:
        path = self.fixture_root / "structures.json"
        data = json.loads(path.read_text())
        return DataSnapshot(
            name="pdb_structures",
            source=f"fixture:{path.name}",
            kind=self.kind,
            records=data.get("structures", []),
            metadata={k: v for k, v in data.items() if k != "structures"},
            license=self.license,
        )


class OpenTargetsLikeConnector(Connector):
    name = "opentargets"
    kind = "target_disease"

    def load(self, **kwargs: Any) -> DataSnapshot:
        path = self.fixture_root / "targets.json"
        data = json.loads(path.read_text())
        return DataSnapshot(
            name="target_disease",
            source=f"fixture:{path.name}",
            kind=self.kind,
            records=data.get("associations", []),
            metadata={k: v for k, v in data.items() if k != "associations"},
            license=self.license,
        )


class LiteratureConnector(Connector):
    name = "literature"
    kind = "literature"

    def load(self, **kwargs: Any) -> DataSnapshot:
        path = self.fixture_root / "literature.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return DataSnapshot(
            name="literature",
            source=f"fixture:{path.name}",
            kind=self.kind,
            records=records,
            license=self.license,
        )


class SingleCellConnector(Connector):
    name = "singlecell"
    kind = "single_cell"

    def load(self, **kwargs: Any) -> DataSnapshot:
        path = self.fixture_root / "sc_perturbation.npz"
        return DataSnapshot(
            name="sc_perturbation",
            source=f"fixture:{path.name}",
            kind=self.kind,
            records=[{"path": str(path), "format": "npz"}],
            metadata={"cell_context": "vascular_endothelium_organoid"},
            license=self.license,
        )


def default_connectors(
    fixture_root: Path | str = FIXTURE_ROOT, egress: EgressPolicy | None = None
) -> dict[str, Connector]:
    classes = [
        LocalFilesConnector,
        ChemblLikeConnector,
        PubChemLikeConnector,
        PdbLikeConnector,
        OpenTargetsLikeConnector,
        LiteratureConnector,
        SingleCellConnector,
    ]
    return {c.name: c(fixture_root, egress) for c in classes}  # type: ignore[abstract]
