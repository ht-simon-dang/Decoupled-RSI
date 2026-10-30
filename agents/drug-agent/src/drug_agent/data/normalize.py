"""Assay unit normalization and molecule de-duplication.

Unit conversion is tested (see tests/test_units.py). Incompatible endpoints are
never merged; only concentration endpoints (IC50/Ki/EC50/Kd) are converted to
pIC50-style -log10(M) values, preserving the original value/unit/qualifier.
"""

from __future__ import annotations

import math
from typing import Any

from ..contracts.base import Provenance, Quantity
from ..contracts.chem import AssayRecord
from ..contracts.enums import Qualifier, TrustLabel

# Multipliers to convert a concentration unit to molar (M).
_CONC_TO_M: dict[str, float] = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}
_CONCENTRATION_ENDPOINTS = {"IC50", "EC50", "Ki", "Kd", "IC90", "AC50"}
_QUALIFIER_MAP = {
    "=": Qualifier.EXACT,
    "<": Qualifier.LESS_THAN,
    ">": Qualifier.GREATER_THAN,
    "<=": Qualifier.LESS_EQUAL,
    ">=": Qualifier.GREATER_EQUAL,
    "~": Qualifier.APPROX,
}


def to_molar(value: float, unit: str) -> float:
    if unit not in _CONC_TO_M:
        raise ValueError(f"unknown concentration unit: {unit!r}")
    return value * _CONC_TO_M[unit]


def to_pic50(value: float, unit: str) -> float:
    """Convert a concentration to a -log10(M) potency value (pIC50-style)."""
    molar = to_molar(value, unit)
    if molar <= 0:
        raise ValueError("concentration must be positive")
    return round(-math.log10(molar), 4)


def normalize_assay_value(endpoint: str, value: float, unit: str) -> tuple[float, str]:
    """Return a normalized (value, unit). Concentration endpoints -> pIC50."""
    if endpoint in _CONCENTRATION_ENDPOINTS and unit in _CONC_TO_M:
        return to_pic50(value, unit), "pIC50"
    return value, unit


def build_assay_record(row: dict[str, Any], *, source: str, snapshot_id: str) -> AssayRecord:
    endpoint = str(row.get("endpoint", "IC50"))
    original_value = str(row["value"])  # verbatim source representation
    raw_value = float(row["value"])
    raw_unit = str(row.get("unit", "nM"))
    relation = str(row.get("relation", "="))
    norm_value, norm_unit = normalize_assay_value(endpoint, raw_value, raw_unit)
    measurement = Quantity(
        value=norm_value,
        unit=norm_unit,
        qualifier=_QUALIFIER_MAP.get(relation, Qualifier.EXACT),
        original_value=original_value,
        original_unit=raw_unit,
        trust_label=TrustLabel.OBSERVED,
    )
    return AssayRecord(
        molecule_smiles=row.get("smiles"),
        target_id=row.get("target_id"),
        target_name=row.get("target_name"),
        endpoint=endpoint,
        measurement=measurement,
        assay_type=row.get("assay_type"),
        assay_description=row.get("assay_description"),
        relation=_QUALIFIER_MAP.get(relation, Qualifier.EXACT),
        year=int(row["year"]) if row.get("year") else None,
        series_id=row.get("series_id"),
        provenance=Provenance(
            source=source, snapshot_id=snapshot_id, trust_label=TrustLabel.OBSERVED, method="assay_fixture"
        ),
        trust_label=TrustLabel.OBSERVED,
    )


def dedup_by_inchikey(molecules: list[Any]) -> tuple[list[Any], list[str]]:
    """De-duplicate molecules by InChIKey, keeping first. Returns (unique, dup_ids)."""
    seen: dict[str, Any] = {}
    dups: list[str] = []
    for m in molecules:
        key = getattr(m, "inchikey", None) or getattr(m, "canonical_smiles", None)
        if key is None:
            seen[getattr(m, "id", None) or str(id(m))] = m
            continue
        if key in seen:
            dups.append(getattr(m, "id", ""))
        else:
            seen[key] = m
    return list(seen.values()), dups
