"""Ingestion, assay strata, provenance and normalization for one target.

This module turns raw source records into standardized measurement rows without
ever losing the record they came from. Four properties are load-bearing, and the
rest of the module exists to keep them true.

**A standardized row is a view of a source record, never a replacement.** Every
row carries the original value, unit and relation verbatim, a verbatim copy of
the source record under ``raw``, and a ``provenance`` pointer naming the
snapshot file, the record's index inside it, the snapshot's SHA-256, the URL, the
fetch time and the licence. :func:`reconstruct_source` follows that pointer back
to disk and refuses the row if what it finds disagrees. "Traceable" is therefore
a thing that can fail, not a claim in a document.

**A censored measurement stays censored, in both directions.** ``> 1000 nM`` is
not ``1000 nM``. Converted to a negative log concentration the inequality
*reverses*: the compound's pActivity is at most 6, an upper bound. Collapsing
that into an exact 6.0 would silently invent 435 exact ZAK measurements out of
inequalities, which is why :func:`normalize_measurement` reports ``exact``,
``censored``, ``p_relation`` and ``p_bound`` rather than a bare number.

**Strata are structural, not editorial.** A stratum id is built from the target,
the endpoint, the assay type and the BAO format, so Kd, Ki, a biochemical IC50
and a cell-based IC50 land in different strata because their identifiers differ
— not because someone remembered to keep them apart. Nothing in this module
merges two strata, and :func:`deduplicate_records` merges only across *sources*.

**Identity is decided by structure.** The 6JUU ligand ``C9R`` is compound *6r*.
The paper's headline compound *6p* is a different molecule with a different
formula, and it is the one carrying the reported Kd = 8.0 nM.
:func:`resolve_reference_compound` matches on InChIKey and returns an honest
``NOT_FOUND`` rather than the nearest neighbour, so a name can never carry a
label onto the wrong structure. :func:`pocket_matches_target` applies the same
rule to the receptor: a pocket file is checked residue-by-residue against the
target manifest's author-numbered residue map, which is what a soluble epoxide
hydrolase pocket labelled "ZAK" fails.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import CONTRACT_VERSION, ContractError, ErrorCategory

Record = dict[str, Any]

# ---------------------------------------------------------------------------
# Units and relations
# ---------------------------------------------------------------------------
#: The explicit concentration factor table required by the work order. Molar is
#: the only internal unit; nothing else is ever stored as a "value".
UNIT_FACTORS_M: dict[str, float] = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
}

#: MICRO SIGN (U+00B5) and GREEK SMALL LETTER MU (U+03BC) are visually identical
#: and both occur in deposited assay data. They are folded to ASCII ``u`` before
#: lookup so ``uM``/``µM``/``μM`` cannot become three different units, while the
#: symbol the source actually wrote is kept on the row.
_MICRO_SIGNS = ("µ", "μ")

_UNIT_BY_FOLDED: dict[str, str] = {u.lower(): u for u in UNIT_FACTORS_M}

#: Every relation a standardized row may carry. Anything else is refused rather
#: than guessed at, because a guessed relation silently changes what a row means.
RELATIONS: tuple[str, ...] = ("=", "<", ">", "<=", ">=", "~")

_RELATION_BY_RAW: dict[str, str] = {
    "=": "=",
    "==": "=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    "=<": "<=",
    "≤": "<=",  # ≤
    ">=": ">=",
    "=>": ">=",
    "≥": ">=",  # ≥
    "~": "~",
    "≈": "~",  # ≈
}

#: Negating a logarithm reverses the inequality. ``~`` is an approximation, not a
#: bound, so it has no direction to reverse.
REVERSED_RELATION: dict[str, str] = {
    "=": "=",
    "<": ">",
    ">": "<",
    "<=": ">=",
    ">=": "<=",
    "~": "~",
}

#: Relations that assert a bound rather than a value.
CENSORING_RELATIONS = frozenset({"<", ">", "<=", ">="})

#: Endpoints whose *value* is a concentration, and so the only ones a
#: concentration unit may be applied to.
#:
#: This list is not decoration. ChEMBL carries six ZAK rows whose
#: ``standard_type`` is ``% inhibition`` and whose ``standard_units`` is ``uM``:
#: the number is a percentage and the unit belongs to the concentration the
#: assay was *run at*. Converting 44.9 there yields "44.9 uM potency", a
#: fabricated IC50-like value with a real provenance trail attached. The unit
#: table alone cannot catch that, because the unit is valid; only the endpoint
#: can. Anything outside this set is excluded with its endpoint named.
CONCENTRATION_ENDPOINTS = frozenset(
    {
        "IC50", "IC90", "IC95", "EC50", "EC90", "ED50", "AC50", "GI50", "XC50",
        "CC50", "TD50", "LD50", "MIC", "Ki", "Kd", "Kb", "Km", "Potency",
    }
)  # fmt: skip


def canonical_unit(unit: str) -> str:
    """Fold a source's spelling of a concentration unit onto the factor table.

    Raises:
        ContractError: ``INVALID_ENUM`` if the unit is not a concentration unit
            this module can convert. A mass-per-volume unit such as
            ``ug.mL-1`` is refused rather than converted with an assumed
            molecular weight.
    """
    folded = str(unit).strip()
    for sign in _MICRO_SIGNS:
        folded = folded.replace(sign, "u")
    resolved = _UNIT_BY_FOLDED.get(folded.lower())
    if resolved is None:
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"{unit!r} is not a convertible concentration unit; known units are {sorted(UNIT_FACTORS_M)}",
            contract="Measurement",
            field="unit",
        )
    return resolved


def canonical_relation(relation: str) -> str:
    """Fold a source's spelling of a relation onto :data:`RELATIONS`.

    Raises:
        ContractError: ``INVALID_ENUM`` for anything unrecognised.
    """
    raw = str(relation).strip()
    resolved = _RELATION_BY_RAW.get(raw)
    if resolved is None:
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"{relation!r} is not a known measurement relation; known relations are {list(RELATIONS)}",
            contract="Measurement",
            field="relation",
        )
    return resolved


def normalize_measurement(value: float, unit: str, relation: str) -> Record:
    """Standardize one concentration measurement, preserving the original.

    Returns a record carrying, in order: what the source said
    (``source_value``/``source_unit``/``source_relation``), the canonical form
    (``unit``/``relation``), the molar value (``value_M``), whether the
    measurement is exact or a bound (``exact``/``censored``), and the negative
    log concentration with its *reversed* inequality
    (``p_value``/``p_relation``/``p_bound``).

    >>> row = normalize_measurement(1000.0, "nM", ">")
    >>> row["value_M"], row["relation"], row["exact"]
    (1e-06, '>', False)
    >>> row["p_value"], row["p_relation"], row["p_bound"]
    (6.0, '<', 'upper')

    Raises:
        ContractError: (a ``ValueError``) ``INVALID_ENUM`` for an unknown unit or
            relation, ``INVALID_VALUE`` for a nonpositive or non-numeric
            concentration, ``NON_FINITE_VALUE`` for NaN or infinity.
    """
    canon_unit = canonical_unit(unit)
    canon_relation = canonical_relation(relation)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"{value!r} is not a number",
            contract="Measurement",
            field="value",
        ) from exc
    if not math.isfinite(numeric):
        raise ContractError(
            ErrorCategory.NON_FINITE_VALUE,
            f"{value!r} is not finite and is not a measurement",
            contract="Measurement",
            field="value",
        )
    if numeric <= 0:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"concentration must be positive, got {numeric!r}; a nonpositive "
            "concentration has no negative log and is not a measurement",
            contract="Measurement",
            field="value",
        )

    value_m = numeric * UNIT_FACTORS_M[canon_unit]
    p_relation = REVERSED_RELATION[canon_relation]
    if p_relation in ("<", "<="):
        bound = "upper"
    elif p_relation in (">", ">="):
        bound = "lower"
    else:
        bound = "none"
    return {
        "source_value": numeric,
        "source_unit": unit,
        "source_relation": relation,
        "unit": canon_unit,
        "relation": canon_relation,
        "value_M": value_m,
        "exact": canon_relation == "=",
        "censored": canon_relation in CENSORING_RELATIONS,
        "p_value": round(-math.log10(value_m), 6),
        "p_relation": p_relation,
        "p_bound": bound,
    }


# ---------------------------------------------------------------------------
# Molecular identity
# ---------------------------------------------------------------------------
def _rdkit() -> Any:
    try:
        import rdkit
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - rdkit is a hard dependency here
        raise ContractError(
            ErrorCategory.MISSING_COMPONENT,
            "rdkit is required to standardize molecular identity",
            contract="Molecule",
            field="smiles",
        ) from exc
    RDLogger.DisableLog("rdApp.*")
    return rdkit, Chem, rdMolDescriptors, rdMolStandardize, MurckoScaffold


def _standardizer_id() -> str:
    import rdkit

    return f"rdkit-{rdkit.__version__}/cleanup+fragment-parent+uncharge/keep-stereo/v1"


#: The exact, versioned standardization recipe every molecule here passes
#: through. It is stored on every molecule record and in the data card so a
#: later worker can tell whether two datasets were normalized the same way.
STANDARDIZER_ID = _standardizer_id()


def standardize_molecule(smiles: str) -> Record:
    """Standardize one structure, keeping the input and the stereochemistry.

    The recipe is RDKit ``Cleanup`` (sanitize, normalize functional groups,
    disconnect metals), ``FragmentParent`` (drop counter-ions and solvates) and
    ``Uncharger`` (neutralize where chemically sensible). Stereochemistry is
    never removed: an enantiomer pair keeps two different InChIKeys and shares
    one ``connectivity_key`` (the InChIKey's first block), which is how the
    dataset can group a stereoisomer family without flattening it.

    Raises:
        ContractError: ``INVALID_VALUE`` if the structure cannot be parsed.
    """
    _rk, Chem, rdMolDescriptors, rdMolStandardize, MurckoScaffold = _rdkit()
    import rdkit

    text = "" if smiles is None else str(smiles).strip()
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"{smiles!r} is not a parseable SMILES structure",
            contract="Molecule",
            field="smiles",
        )

    cleaned = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(cleaned)
    neutral = rdMolStandardize.Uncharger().uncharge(parent)
    Chem.AssignStereochemistry(neutral, cleanIt=True, force=True)

    standard_smiles = Chem.MolToSmiles(neutral)  # isomeric by default
    inchi = Chem.MolToInchi(neutral)
    inchikey = Chem.InchiToInchiKey(inchi) if inchi else ""
    if not inchikey:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"{smiles!r} standardized but produced no InChIKey, so it has no stable identity",
            contract="Molecule",
            field="smiles",
        )
    centres = Chem.FindMolChiralCenters(neutral, includeUnassigned=True, useLegacyImplementation=False)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(neutral)
    scaffold = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""

    return {
        "input_smiles": smiles,
        "standard_smiles": standard_smiles,
        "inchi": inchi,
        "inchikey": inchikey,
        "connectivity_key": inchikey.split("-")[0],
        "formula": rdMolDescriptors.CalcMolFormula(neutral),
        "scaffold_smiles": scaffold,
        "n_stereocentres_specified": sum(1 for _, tag in centres if tag != "?"),
        "n_stereocentres_unspecified": sum(1 for _, tag in centres if tag == "?"),
        "n_heavy_atoms": neutral.GetNumHeavyAtoms(),
        "rdkit_version": rdkit.__version__,
        "standardizer_id": STANDARDIZER_ID,
    }


#: Appendix A §3.2's family edge rules, minus the curated medicinal-chemistry
#: series (no curated series list exists for this target), recorded so the
#: omission is visible wherever a family count is reported.
FAMILY_EDGE_RULES = {
    "shared_nonempty_bemis_murcko_scaffold": True,
    "morgan_r2_2048_tanimoto_at_least": 0.80,
    "curated_medicinal_chemistry_series": "UNAVAILABLE: no curated series list for this target",
    "empty_scaffolds_are_not_one_family": True,
}


def family_components(molecules: Sequence[Mapping[str, Any]], *, threshold: float = 0.80) -> Record:
    """Connected components over Appendix A §3.2's structural-family edges.

    Two molecules are joined by a shared **nonempty** Bemis-Murcko scaffold or a
    Morgan radius-2 / 2,048-bit Tanimoto of at least ``threshold``. An empty
    scaffold joins nothing: acyclic molecules are not all one family, and
    treating them as one is how an entire split leaks.

    This is a *count*, not the benchmark split. Step 6 owns partitioning.
    """
    _rk, Chem, _desc, _std, _scaf = _rdkit()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    scaffolds: list[str] = []
    for mol in molecules:
        rdmol = Chem.MolFromSmiles(mol["standard_smiles"])
        fps.append(gen.GetFingerprint(rdmol) if rdmol is not None else None)
        scaffolds.append(str(mol.get("scaffold_smiles") or ""))

    n = len(molecules)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_scaffold: dict[str, list[int]] = {}
    for i, scaffold in enumerate(scaffolds):
        if scaffold:
            by_scaffold.setdefault(scaffold, []).append(i)
    n_scaffold_edges = 0
    for members in by_scaffold.values():
        for j in members[1:]:
            union(members[0], j)
            n_scaffold_edges += 1

    n_similarity_edges = 0
    for i in range(n):
        if fps[i] is None:
            continue
        for j in range(i + 1, n):
            if fps[j] is None or find(i) == find(j):
                continue
            if DataStructs.TanimotoSimilarity(fps[i], fps[j]) >= threshold:
                union(i, j)
                n_similarity_edges += 1

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    components = [sorted(v) for _, v in sorted(groups.items())]
    return {
        "components": components,
        "n_components": len(components),
        "n_empty_scaffold": sum(1 for s in scaffolds if not s),
        "n_scaffold_edges": n_scaffold_edges,
        "n_similarity_edges": n_similarity_edges,
        "largest_component": max((len(c) for c in components), default=0),
        "edge_rules": dict(FAMILY_EDGE_RULES),
    }


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------
#: Used where a source states no assay format. It is a stratum of its own, never
#: folded into a ChEMBL stratum: an unknown assay format is not a matching one.
UNSPECIFIED = "UNSPECIFIED"


def stratum_id(record: Mapping[str, Any]) -> str:
    """The assay-compatible stratum a row belongs to.

    ``<target>|<endpoint>|<assay_type>|<bao_format>``. Because the identifier is
    *built from* the fields that make two measurements incomparable, two rows can
    only share a stratum by genuinely agreeing on all four. There is no code path
    that merges strata.
    """
    return "|".join(
        str(record.get(key) or UNSPECIFIED) for key in ("target_id", "endpoint", "assay_type", "bao_format")
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _provenance(
    *,
    source: str,
    snapshot_sha256: str,
    snapshot_file: str,
    record_index: int,
    fetched_at: str,
    source_url: str,
    license: str,
    primary_key: str,
) -> Record:
    return {
        "source": source,
        "snapshot_file": snapshot_file,
        "snapshot_sha256": snapshot_sha256,
        "record_index": int(record_index),
        "primary_key": primary_key,
        "fetched_at": fetched_at,
        "source_url": source_url,
        "license": license,
    }


def _document_key(*, pmid: Any = None, doi: Any = None, fallback: Any = None) -> str:
    """The study a measurement came from, preferring globally stable ids.

    A PubMed id is the same study in every database, which is what makes
    cross-source mirror detection possible at all; a database-local document id
    is not, so it is only ever the fallback.
    """
    if pmid not in (None, "", 0):
        return f"pmid:{pmid}"
    if doi not in (None, ""):
        return f"doi:{str(doi).lower()}"
    if fallback not in (None, ""):
        return f"local:{fallback}"
    return "unknown"


def reconstruct_source(row: Mapping[str, Any], snapshot_root: Path | str) -> Record:
    """Return the original source record a standardized row was built from.

    This is what makes "reversible lineage" checkable: it opens the snapshot
    file the row names, reads the record at the index the row names, and
    compares it with the copy the row carries. Either both agree — in which case
    the original record is returned — or the row is refused.

    Raises:
        ContractError: ``UNVERIFIABLE_REFERENCE`` when the snapshot file or the
            index is not there, ``INVALID_VALUE`` when the snapshot and the row's
            own copy disagree.
    """
    provenance = row.get("provenance") or {}
    path = Path(snapshot_root) / str(provenance.get("snapshot_file", ""))
    if not path.is_file():
        raise ContractError(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            f"snapshot file {path} named by this row is not present",
            contract="Measurement",
            field="provenance.snapshot_file",
            missing=[str(path)],
        )
    records = json.loads(path.read_text())
    index = int(provenance.get("record_index", -1))
    if not isinstance(records, list) or not 0 <= index < len(records):
        extent = len(records) if isinstance(records, list) else "not a list"
        raise ContractError(
            ErrorCategory.UNVERIFIABLE_REFERENCE,
            f"record index {index} is outside {path} ({extent})",
            contract="Measurement",
            field="provenance.record_index",
        )
    original = records[index]
    if original != row.get("raw"):
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"the copy carried by this row disagrees with {path}[{index}]; "
            "the row has been edited since ingestion and cannot be traced",
            contract="Measurement",
            field="raw",
        )
    return original


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------
def _row(
    *,
    provenance: Record,
    raw: Mapping[str, Any],
    molecule: Record,
    measurement: Record,
    endpoint: str,
    target_id: str,
    assay_type: str,
    bao_format: str,
    bao_label: str | None,
    assay_id: str | None,
    assay_description: str | None,
    document_key: str,
    document_id: str | None,
    organism: str | None,
    molecule_id: str | None,
    extra: Mapping[str, Any] | None = None,
) -> Record:
    row: Record = {
        "endpoint": endpoint,
        "target_id": target_id,
        "assay_type": assay_type,
        "bao_format": bao_format,
        "bao_label": bao_label,
        "assay_id": assay_id,
        "assay_description": assay_description,
        "document_key": document_key,
        "document_id": document_id,
        "organism": organism,
        "molecule_id": molecule_id,
        "molecule": molecule,
        "inchikey": molecule["inchikey"],
        "connectivity_key": molecule["connectivity_key"],
        **measurement,
        "provenance": provenance,
        "raw": json.loads(json.dumps(raw)),  # a detached verbatim copy
        "contract_version": CONTRACT_VERSION,
    }
    row["stratum_id"] = stratum_id(row)
    if extra:
        row.update(extra)
    return row


def normalize_chembl_activity(
    raw: Mapping[str, Any],
    *,
    source: str = "chembl",
    snapshot_sha256: str,
    snapshot_file: str,
    record_index: int,
    fetched_at: str,
    source_url: str,
    license: str,
    document_index: Mapping[str, Mapping[str, Any]] | None = None,
) -> Record:
    """Standardize one ChEMBL ``activity`` record.

    ``standard_*`` fields are used because they are ChEMBL's own harmonized
    view; the untouched ``value``/``units``/``relation`` pair survives inside
    ``raw`` either way. A row without a usable structure, value, unit or
    relation is *refused* here and recorded as an exclusion by
    :func:`ingest_source_rows` — never quietly defaulted.

    ``document_index`` maps ``document_chembl_id`` to that document's record, and
    it is what makes cross-source mirror detection possible at all: a ChEMBL
    activity names a ChEMBL document, BindingDB names a PubMed id, and without
    the index the two spellings of the same study never meet, so every mirror is
    silently counted as an independent experiment.

    Raises:
        ContractError: with the field that made the row unusable.
    """
    smiles = raw.get("canonical_smiles")
    if not smiles:
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            "activity row carries no structure",
            contract="ChemblActivity",
            field="canonical_smiles",
        )
    endpoint = str(raw.get("standard_type") or "")
    if endpoint not in CONCENTRATION_ENDPOINTS:
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"{endpoint!r} is not an endpoint whose value is a concentration; "
            "its units describe something other than the reported number",
            contract="ChemblActivity",
            field="standard_type",
        )
    value = raw.get("standard_value")
    if value in (None, ""):
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            "activity row carries no standard_value; a text-only or qualitative "
            "record is not a quantitative measurement",
            contract="ChemblActivity",
            field="standard_value",
        )
    measurement = normalize_measurement(
        float(str(value)), raw.get("standard_units") or "", raw.get("standard_relation") or ""
    )
    molecule = standardize_molecule(smiles)
    activity_id = raw.get("activity_id")
    return _row(
        provenance=_provenance(
            source=source,
            snapshot_sha256=snapshot_sha256,
            snapshot_file=snapshot_file,
            record_index=record_index,
            fetched_at=fetched_at,
            source_url=source_url,
            license=license,
            primary_key=f"activity_id={activity_id}",
        ),
        raw=raw,
        molecule=molecule,
        measurement=measurement,
        endpoint=endpoint,
        target_id=str(raw.get("target_chembl_id") or UNSPECIFIED),
        assay_type=str(raw.get("assay_type") or UNSPECIFIED),
        bao_format=str(raw.get("bao_format") or UNSPECIFIED),
        bao_label=raw.get("bao_label"),
        assay_id=raw.get("assay_chembl_id"),
        assay_description=raw.get("assay_description"),
        document_key=_document_key(
            pmid=(document_index or {}).get(str(raw.get("document_chembl_id")), {}).get("pubmed_id"),
            doi=(document_index or {}).get(str(raw.get("document_chembl_id")), {}).get("doi"),
            fallback=raw.get("document_chembl_id"),
        ),
        document_id=raw.get("document_chembl_id"),
        organism=raw.get("target_organism"),
        molecule_id=raw.get("molecule_chembl_id"),
        extra={
            "pchembl_value": raw.get("pchembl_value"),
            "data_validity_comment": raw.get("data_validity_comment"),
            "potential_duplicate": raw.get("potential_duplicate"),
            "src_id": raw.get("src_id"),
            "document_year": raw.get("document_year"),
        },
    )


#: BindingDB's REST affinities give a value with the relation glued to the front
#: and no unit at all; the service documents its affinities as nanomolar.
_BINDINGDB_UNIT = "nM"


def normalize_bindingdb_affinity(
    raw: Mapping[str, Any],
    *,
    source: str = "bindingdb",
    snapshot_sha256: str,
    snapshot_file: str,
    record_index: int,
    fetched_at: str,
    source_url: str,
    license: str,
    target_id: str = "CHEMBL3886",
    organism: str | None = "Homo sapiens",
) -> Record:
    """Standardize one BindingDB ``getLigandsByUniprots`` affinity record.

    BindingDB states no assay type or BAO format, so the row is stratified as
    ``UNSPECIFIED`` on both. That keeps it out of every ChEMBL stratum unless
    :func:`deduplicate_records` identifies it as a mirror of a ChEMBL row, which
    is the only way it can acquire assay metadata it did not publish.
    """
    smiles = raw.get("smile") or raw.get("smiles")
    if not smiles:
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            "affinity row carries no structure",
            contract="BindingDbAffinity",
            field="smile",
        )
    text = str(raw.get("affinity") or "").strip()
    if not text:
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            "affinity row carries no value",
            contract="BindingDbAffinity",
            field="affinity",
        )
    relation = "="
    for prefix in (">=", "<=", ">", "<", "~"):
        if text.startswith(prefix):
            relation, text = prefix, text[len(prefix) :].strip()
            break
    try:
        value = float(text)
    except ValueError as exc:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"affinity {raw.get('affinity')!r} is not a number after its relation",
            contract="BindingDbAffinity",
            field="affinity",
        ) from exc
    measurement = normalize_measurement(value, _BINDINGDB_UNIT, relation)
    molecule = standardize_molecule(smiles)
    monomer = raw.get("monomerid")
    return _row(
        provenance=_provenance(
            source=source,
            snapshot_sha256=snapshot_sha256,
            snapshot_file=snapshot_file,
            record_index=record_index,
            fetched_at=fetched_at,
            source_url=source_url,
            license=license,
            primary_key=f"monomerid={monomer}&affinity_type={raw.get('affinity_type')}&pmid={raw.get('pmid')}",
        ),
        raw=raw,
        molecule=molecule,
        measurement=measurement,
        endpoint=str(raw.get("affinity_type") or UNSPECIFIED),
        target_id=target_id,
        assay_type=UNSPECIFIED,
        bao_format=UNSPECIFIED,
        bao_label=None,
        assay_id=None,
        assay_description=raw.get("query"),
        document_key=_document_key(pmid=raw.get("pmid"), doi=raw.get("doi")),
        document_id=raw.get("pmid") or raw.get("doi"),
        organism=organism,
        molecule_id=f"BDBM{monomer}" if monomer else None,
        extra={"bindingdb_monomer_id": monomer, "unit_assumed": True},
    )


def ingest_source_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    normalizer: Any,
    *,
    exclusion_registry: Mapping[str, Any] | None = None,
    **provenance: Any,
) -> Record:
    """Normalize a whole snapshot file, recording every row it could not use.

    Returns ``{"rows": [...], "exclusions": [...]}``. An exclusion carries the
    record index, the source's own primary key, the stable error category and
    the offending field, so "coverage" is a subtraction a reader can audit
    rather than a number to take on trust.

    When ``exclusion_registry`` is a Step 6 protected registry, every row that
    normalized successfully is also screened against it before it is admitted,
    and a held-out compound is recorded as a ``TEST_LEAKAGE`` exclusion instead
    of reaching ``rows``. This is the acquisition boundary: matching is on
    identity — the compound, its source aliases, its stereoisomer family and its
    structural family — so a held-out compound arriving under another database's
    identifier, in another endpoint or another assay format, is still refused.

    The parameter defaults to ``None`` and screening is then skipped entirely,
    so an ingestion that declares no registry behaves exactly as before.
    """
    from .splits import screen_records

    rows: list[Record] = []
    exclusions: list[Record] = []
    for index, raw in enumerate(raw_rows):
        try:
            row = normalizer(raw, record_index=index, **provenance)
        except ContractError as exc:
            exclusions.append(
                {
                    "record_index": index,
                    "source": provenance.get("source", getattr(normalizer, "__name__", "?")),
                    "category": exc.category.value,
                    "field": exc.field,
                    "reason": str(exc),
                    "endpoint": raw.get("standard_type") or raw.get("affinity_type"),
                    "units": raw.get("standard_units"),
                }
            )
            continue
        if exclusion_registry is not None:
            aliases = [str(row.get("molecule_id") or "")]
            aliases += [str(entry.get("molecule_id") or "") for entry in row.get("sources", ()) or ()]
            verdict = screen_records(
                [
                    {
                        "molecule_id": row["inchikey"],
                        "aliases": sorted({a for a in aliases if a}),
                        "connectivity_key": row.get("connectivity_key"),
                        "source": row["provenance"]["source"],
                        "endpoint": row["endpoint"],
                    }
                ],
                exclusion_registry,
            )
            if verdict["quarantined"]:
                hit = verdict["quarantined"][0]
                exclusions.append(
                    {
                        "record_index": index,
                        "source": row["provenance"]["source"],
                        "category": hit["category"],
                        "field": hit["matched_on"],
                        "matched_on": hit["matched_on"],
                        "matched_value": hit["matched_value"],
                        "inchikey": row["inchikey"],
                        "molecule_id": row.get("molecule_id"),
                        "stratum_id": row["stratum_id"],
                        "reason": hit["reason"],
                        "endpoint": row["endpoint"],
                        "units": row.get("unit"),
                    }
                )
                continue
        rows.append(row)
    return {"rows": rows, "exclusions": exclusions}


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------
#: Which source's metadata wins when the same experiment appears in two
#: databases. ChEMBL publishes the assay record and the BAO format; BindingDB
#: republishes the number. Lower rank wins, deterministically.
SOURCE_RANK = {"chembl": 0, "bindingdb": 1}


def _mirror_key(row: Mapping[str, Any]) -> tuple:
    """What makes two records the *same experiment* in two different databases.

    Study, structure, endpoint, relation and value. Deliberately **not** the
    assay id: a mirror does not carry the other database's assay id, which is
    the whole reason a mirror is hard to spot.
    """
    return (
        row["document_key"],
        row["inchikey"],
        row["endpoint"],
        row["relation"],
        float(f"{row['value_M']:.6g}"),
    )


def deduplicate_records(rows: Sequence[Mapping[str, Any]]) -> Record:
    """Collapse cross-source mirrors; keep genuine within-source repeats.

    A **mirror** is one experiment published by two databases. It becomes one
    row whose ``sources`` list names both, so the experiment is counted once
    while both provenance trails survive.

    A **repeat** is two records from the same database — typically the same
    number recorded under two assay descriptions, as happens for compound 6p's
    Kd = 8.0 nM. They stay two rows. Averaging them would fabricate a precision
    nobody measured, and collapsing them on compound-plus-endpoint would erase
    the assay distinction the strata exist to protect. They are counted in
    ``n_suspected_replicates`` instead, for Step 6 to partition together.

    Returns ``{"rows": [...], "report": {...}}``.
    """
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_mirror_key(row), []).append(row)

    out: list[Record] = []
    n_mirrors = 0
    n_absorbed = 0
    n_suspected_replicates = 0
    n_alias_ambiguous = 0

    def _source_entry(row: Mapping[str, Any]) -> Record:
        return {
            "source": row["provenance"]["source"],
            "primary_key": row["provenance"]["primary_key"],
            "snapshot_file": row["provenance"]["snapshot_file"],
            "snapshot_sha256": row["provenance"]["snapshot_sha256"],
            "source_url": row["provenance"]["source_url"],
            "fetched_at": row["provenance"]["fetched_at"],
            "license": row["provenance"]["license"],
            "molecule_id": row.get("molecule_id"),
            "assay_id": row.get("assay_id"),
            "endpoint": row["endpoint"],
            "stratum_id": row["stratum_id"],
        }

    for key in sorted(groups, key=lambda k: tuple(str(part) for part in k)):
        members = sorted(
            groups[key],
            key=lambda r: (
                SOURCE_RANK.get(r["provenance"]["source"], 99),
                str(r.get("assay_id") or ""),
                r["provenance"]["primary_key"],
            ),
        )
        # Partition by source rank, not by value equality: two rows from the same
        # database are two experiments even when every field matches, and a `not
        # in` test on dictionaries would silently fold them into one.
        best_rank = SOURCE_RANK.get(members[0]["provenance"]["source"], 99)
        primary = [r for r in members if SOURCE_RANK.get(r["provenance"]["source"], 99) == best_rank]
        aliases = [r for r in members if SOURCE_RANK.get(r["provenance"]["source"], 99) != best_rank]
        if len(primary) > 1:
            n_suspected_replicates += 1
            if aliases:
                n_alias_ambiguous += 1
        for position, row in enumerate(primary):
            merged = dict(row)
            sources = [_source_entry(row)]
            if position == 0 and aliases:
                sources.extend(_source_entry(a) for a in aliases)
                n_absorbed += len(aliases)
                n_mirrors += 1
            merged["sources"] = sources
            merged["n_sources"] = len(sources)
            merged["mirror_key"] = "|".join(str(p) for p in key)
            merged["is_suspected_replicate"] = len(primary) > 1
            out.append(merged)

    per_source_in: dict[str, int] = {}
    for row in rows:
        source = row["provenance"]["source"]
        per_source_in[source] = per_source_in.get(source, 0) + 1
    per_source_out: dict[str, int] = {}
    for row in out:
        source = row["provenance"]["source"]
        per_source_out[source] = per_source_out.get(source, 0) + 1

    return {
        "rows": out,
        "report": {
            "n_rows_in": len(rows),
            "n_rows_out": len(out),
            "n_cross_source_mirrors": n_mirrors,
            "n_alias_rows_absorbed": n_absorbed,
            "n_suspected_replicates": n_suspected_replicates,
            "n_alias_ambiguous": n_alias_ambiguous,
            "per_source_rows_in": per_source_in,
            "per_source_rows_kept": per_source_out,
            "mirror_key_fields": ["document_key", "inchikey", "endpoint", "relation", "value_M"],
            "source_rank": dict(SOURCE_RANK),
        },
    }


# ---------------------------------------------------------------------------
# Data card
# ---------------------------------------------------------------------------
#: Two exact measurements of the same molecule in the same stratum that differ by
#: more than this many log units are reported as a conflict rather than averaged.
LABEL_CONFLICT_LOG_UNITS = 1.0


def build_data_card(
    deduplicated: Mapping[str, Any],
    *,
    sources: Sequence[Mapping[str, Any]] = (),
    exclusions: Sequence[Mapping[str, Any]] = (),
    target_manifest: Mapping[str, Any] | None = None,
) -> Record:
    """Summarize a standardized dataset the way the next worker has to read it.

    Everything reported here is a count over the rows that survived, except
    ``exclusions``, which is what did not. Per-stratum counts report rows,
    independent molecules and — separately — how many of those molecules have an
    **exact** measurement, because that last number, not the row count, is what
    limits continuous regression.
    """
    rows = list(deduplicated.get("rows", []))
    report = dict(deduplicated.get("report", {}))

    strata: dict[str, Record] = {}
    for row in rows:
        entry = strata.setdefault(
            row["stratum_id"],
            {
                "stratum_id": row["stratum_id"],
                "endpoint": row["endpoint"],
                "assay_type": row["assay_type"],
                "bao_format": row["bao_format"],
                "bao_label": row.get("bao_label"),
                "target_id": row["target_id"],
                "n_rows": 0,
                "_molecules": set(),
                "_exact_molecules": set(),
                "_documents": set(),
                "_assays": set(),
                "n_exact": 0,
                "n_censored": 0,
                "_p_values": [],
            },
        )
        entry["n_rows"] += 1
        entry["_molecules"].add(row["inchikey"])
        entry["_documents"].add(row["document_key"])
        if row.get("assay_id"):
            entry["_assays"].add(row["assay_id"])
        if row["exact"]:
            entry["n_exact"] += 1
            entry["_exact_molecules"].add(row["inchikey"])
            entry["_p_values"].append(row["p_value"])
        if row["censored"]:
            entry["n_censored"] += 1

    stratum_rows: list[Record] = []
    for entry in strata.values():
        p_values = entry.pop("_p_values")
        entry["n_molecules"] = len(entry.pop("_molecules"))
        entry["n_exact_molecules"] = len(entry.pop("_exact_molecules"))
        entry["n_documents"] = len(entry.pop("_documents"))
        entry["n_assays"] = len(entry.pop("_assays"))
        entry["p_value_min"] = round(min(p_values), 4) if p_values else None
        entry["p_value_max"] = round(max(p_values), 4) if p_values else None
        stratum_rows.append(entry)
    stratum_rows.sort(key=lambda s: (-s["n_molecules"], s["stratum_id"]))

    # Label conflicts: the same molecule measured twice in the same stratum, far apart.
    by_molecule_stratum: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["exact"]:
            by_molecule_stratum.setdefault((row["inchikey"], row["stratum_id"]), []).append(row)
    conflicts: list[Record] = []
    for (inchikey, stratum), members in sorted(by_molecule_stratum.items()):
        if len(members) < 2:
            continue
        spread = max(m["p_value"] for m in members) - min(m["p_value"] for m in members)
        if spread > LABEL_CONFLICT_LOG_UNITS:
            conflicts.append(
                {
                    "inchikey": inchikey,
                    "stratum_id": stratum,
                    "n_measurements": len(members),
                    "spread_log10": round(spread, 4),
                    "p_values": sorted(round(m["p_value"], 4) for m in members),
                    "assay_ids": sorted({str(m.get("assay_id")) for m in members}),
                    "molecule_ids": sorted({str(m.get("molecule_id")) for m in members}),
                }
            )

    unique = {row["inchikey"]: row["molecule"] for row in rows}
    families = family_components(list(unique.values())) if unique else {"n_components": 0}

    source_membership: dict[str, set[str]] = {}
    for row in rows:
        for entry in row.get("sources", [{"source": row["provenance"]["source"]}]):
            source_membership.setdefault(entry["source"], set()).add(row["mirror_key"])
    overlap: Record = {name: len(keys) for name, keys in sorted(source_membership.items())}
    names = sorted(source_membership)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap[f"{a}&{b}"] = len(source_membership[a] & source_membership[b])
    overlap["n_experiments"] = len({row["mirror_key"] for row in rows})

    missing = {
        "no_assay_id": sum(1 for r in rows if not r.get("assay_id")),
        "no_bao_format": sum(1 for r in rows if r["bao_format"] == UNSPECIFIED),
        "no_assay_type": sum(1 for r in rows if r["assay_type"] == UNSPECIFIED),
        "no_document": sum(1 for r in rows if r["document_key"] == "unknown"),
        "no_molecule_id": sum(1 for r in rows if not r.get("molecule_id")),
        "unit_assumed": sum(1 for r in rows if r.get("unit_assumed")),
        "flagged_potential_duplicate": sum(1 for r in rows if r.get("potential_duplicate")),
        "data_validity_comment": sum(1 for r in rows if r.get("data_validity_comment")),
    }

    exclusion_summary: dict[str, int] = {}
    exclusion_detail: dict[str, int] = {}
    for item in exclusions:
        key = f"{item.get('source', '?')}:{item.get('category', '?')}:{item.get('field', '-')}"
        exclusion_summary[key] = exclusion_summary.get(key, 0) + 1
        detail = f"{key}|endpoint={item.get('endpoint')}|units={item.get('units')}"
        exclusion_detail[detail] = exclusion_detail.get(detail, 0) + 1

    card: Record = {
        "card_kind": "target_measurement_data_card",
        "contract_version": CONTRACT_VERSION,
        # Deliberately no timestamp. A wall-clock stamp inside the document would
        # change its content hash on every rebuild, so two builds of the same
        # snapshot would be unequal artifacts describing identical data. The build
        # time is recorded in artifact_index.json, where it belongs.
        "determinism": (
            "byte-reproducible from the same source snapshot, standardizer and "
            "contract version; the build time lives in artifact_index.json"
        ),
        "standardizer_id": STANDARDIZER_ID,
        "rdkit_version": STANDARDIZER_ID.split("/")[0].removeprefix("rdkit-"),
        "target_manifest_hash": (target_manifest or {}).get("manifest_hash"),
        "target_alias": (target_manifest or {}).get("alias"),
        "sources": [dict(s) for s in sources],
        "source_overlap": overlap,
        "deduplication": report,
        "n_rows": len(rows),
        "n_independent_molecules": len(unique),
        "n_families": families.get("n_components", 0),
        "families": families,
        "strata": stratum_rows,
        "n_strata": len(stratum_rows),
        "label_conflicts": conflicts,
        "n_label_conflicts": len(conflicts),
        "missing_metadata": missing,
        "exclusions": [dict(e) for e in exclusions],
        "exclusion_summary": exclusion_summary,
        "exclusion_detail": dict(sorted(exclusion_detail.items(), key=lambda kv: (-kv[1], kv[0]))),
        "n_excluded": len(exclusions),
    }
    return card


# ---------------------------------------------------------------------------
# Reference compound identity
# ---------------------------------------------------------------------------
def resolve_reference_compound(smiles: str, rows: Sequence[Mapping[str, Any]]) -> Record:
    """Find the measurements belonging to *this structure*, by InChIKey.

    Nothing here matches on a compound name, a paper label or a document. The
    6JUU ligand C9R and the paper's compound 6p appear in the same publication
    and differ by one substituent; resolving by anything other than structure is
    exactly how 6p's reported Kd ends up attached to 6r.

    A structure with no measurement returns ``status = "NOT_FOUND"`` and an empty
    list — never the nearest neighbour.
    """
    query = standardize_molecule(smiles)
    hits = [r for r in rows if r["inchikey"] == query["inchikey"]]
    molecule_ids = sorted({str(r["molecule_id"]) for r in hits if r.get("molecule_id")})
    return {
        "status": "RESOLVED" if hits else "NOT_FOUND",
        "query_smiles": smiles,
        "inchikey": query["inchikey"],
        "connectivity_key": query["connectivity_key"],
        "formula": query["formula"],
        # One id when the structure resolves cleanly; the first of several, with
        # the full list beside it, when a source registered the same structure
        # more than once. Never a nearest neighbour, and never a silent pick.
        "molecule_id": molecule_ids[0] if molecule_ids else None,
        "molecule_ids": molecule_ids,
        "n_measurements": len(hits),
        "measurements": [
            {
                "endpoint": r["endpoint"],
                "stratum_id": r["stratum_id"],
                "source_value": r["source_value"],
                "source_unit": r["source_unit"],
                "relation": r["relation"],
                "value_M": r["value_M"],
                "p_value": r["p_value"],
                "exact": r["exact"],
                "assay_id": r.get("assay_id"),
                "assay_description": r.get("assay_description"),
                "document_key": r["document_key"],
                "provenance": r["provenance"],
            }
            for r in hits
        ],
    }


# ---------------------------------------------------------------------------
# Structural identity: PDB parsing, sequence checks, pocket binding
# ---------------------------------------------------------------------------
#: Three-letter to one-letter, for the residues a prepared pocket may contain.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}  # fmt: skip


def parse_pdb_residues(text: str) -> dict[str, dict[str, str]]:
    """``{chain: {author residue label: one-letter code}}`` from ATOM records.

    Author numbering is deliberate: a prepared pocket is a *subset* of residues
    with gaps, so it has no sequence of its own to align. What it does keep is
    the numbering of the structure it was cut from, which is exactly the key
    that lets a pocket be checked against a target.
    """
    residues: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[17:20].strip().upper()
        one = THREE_TO_ONE.get(name)
        if one is None:
            continue
        chain = line[21].strip() or "_"
        label = line[22:27].strip()  # residue number plus insertion code
        residues.setdefault(chain, {})[label] = one
    return residues


def _identity(a: str, b: str) -> tuple[int, int, list[Record]]:
    matches = 0
    mismatches: list[Record] = []
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x == y:
            matches += 1
        elif len(mismatches) < 50:
            mismatches.append({"offset": i, "query": x, "reference": y})
    return matches, min(len(a), len(b)), mismatches


def sequence_identity(
    query: str, reference: str, *, aligned_regions: Sequence[Mapping[str, Any]] | None = None
) -> Record:
    """Compare a structure's sequence with its reference sequence.

    When the deposition supplies a SIFTS alignment it is used — and then checked,
    because a recorded alignment is a claim like any other. An independent
    best-ungapped-offset scan runs regardless and is reported next to it, so a
    disagreement between the two is visible rather than hidden behind whichever
    one was consulted first.
    """
    best: Record = {"offset": None, "identity": 0.0, "n_aligned": 0}
    for offset in range(-len(query), len(reference)):
        q_start = max(0, -offset)
        r_start = max(0, offset)
        span = min(len(query) - q_start, len(reference) - r_start)
        if span < 20:
            continue
        matches, n, mm = _identity(query[q_start : q_start + span], reference[r_start : r_start + span])
        identity = matches / n if n else 0.0
        if identity > best["identity"]:
            best = {
                "offset": offset,
                "identity": round(identity, 6),
                "n_aligned": n,
                "n_matches": matches,
                # Kept so a disagreement with the deposited alignment can be read
                # rather than guessed at: an expression tag on the construct
                # shows up here as a handful of terminal mismatches, which is a
                # different fact from a wrong protein.
                "mismatches": mm[:20],
            }

    result: Record = {
        "method": "independent_best_ungapped_offset",
        "independent_best_offset": best,
        "identity": best["identity"],
        "n_aligned": best["n_aligned"],
        "n_mismatches": best["n_aligned"] - best.get("n_matches", 0),
        "mismatches": [],
    }
    if aligned_regions:
        matches = total = 0
        mismatches: list[Record] = []
        for region in aligned_regions:
            q_start = int(region["entity_beg_seq_id"]) - 1
            r_start = int(region["ref_beg_seq_id"]) - 1
            span = int(region["length"])
            m, n, mm = _identity(query[q_start : q_start + span], reference[r_start : r_start + span])
            matches += m
            total += n
            mismatches.extend(mm)
        result.update(
            {
                "method": "deposited_sifts_alignment",
                "identity": round(matches / total, 6) if total else 0.0,
                "n_aligned": total,
                "n_mismatches": total - matches,
                "mismatches": mismatches[:50],
                "aligned_regions": [dict(r) for r in aligned_regions],
                "agrees_with_independent_scan": abs((matches / total if total else 0.0) - best["identity"])
                < 0.01,
            }
        )
    return result


#: A prepared pocket is a subset of a structure's residues. At or above this
#: fraction of author-numbered residues agreeing with the target's residue map,
#: the pocket came from that structure; below it, it did not.
POCKET_IDENTITY_THRESHOLD = 0.90


def pocket_matches_target(
    pocket_path: Path | str,
    manifest: Mapping[str, Any],
    *,
    threshold: float = POCKET_IDENTITY_THRESHOLD,
) -> Record:
    """Decide whether a prepared pocket file really belongs to this target.

    This is the check whose absence let the platform generate molecules against
    a soluble epoxide hydrolase pocket, score them, and report ``ACCEPT`` for
    ZAK. Every residue in the pocket is looked up in the manifest's
    author-numbered residue map for the target's chain; a pocket cut from the
    target structure agrees on essentially all of them, and a pocket from
    another protein does not.

    An unreadable or residue-free pocket is ``match = False`` with a reason. An
    absent check is never a pass.
    """
    path = Path(pocket_path)
    structure = manifest.get("structure", {}) or {}
    residue_map: dict[str, dict[str, str]] = structure.get("residue_map", {}) or {}
    verdict: Record = {
        "match": False,
        "reason": None,
        "identity": 0.0,
        "n_residues_checked": 0,
        "n_matched": 0,
        "n_unmapped": 0,
        "pocket_path": str(path),
        "pocket_sha256": None,
        "target_alias": manifest.get("alias"),
        "target_manifest_hash": manifest.get("manifest_hash"),
        "expected_pdb_id": structure.get("pdb_id"),
        "threshold": threshold,
        "mismatch_examples": [],
    }
    try:
        data = path.read_bytes()
    except OSError as exc:
        verdict["reason"] = "POCKET_UNREADABLE"
        verdict["error"] = repr(exc)
        return verdict
    verdict["pocket_sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()

    pocket = parse_pdb_residues(data.decode("utf-8", errors="replace"))
    if not pocket:
        verdict["reason"] = "NO_RESIDUES"
        return verdict
    if not residue_map:
        verdict["reason"] = "MANIFEST_HAS_NO_RESIDUE_MAP"
        return verdict

    checked = matched = unmapped = 0
    examples: list[Record] = []
    for chain, residues in sorted(pocket.items()):
        # A prepared pocket may be renumbered onto a single chain; fall back to
        # the manifest's own chain when the label does not exist there, rather
        # than skipping the comparison and reporting nothing.
        reference = residue_map.get(chain) or (
            next(iter(residue_map.values())) if len(residue_map) == 1 else {}
        )
        for label, one in sorted(residues.items()):
            checked += 1
            expected = reference.get(label)
            if expected is None:
                unmapped += 1
                if len(examples) < 20:
                    examples.append({"chain": chain, "residue": label, "found": one, "expected": None})
            elif expected == one:
                matched += 1
            elif len(examples) < 20:
                examples.append({"chain": chain, "residue": label, "found": one, "expected": expected})

    verdict.update(
        {
            "n_residues_checked": checked,
            "n_matched": matched,
            "n_unmapped": unmapped,
            "identity": round(matched / checked, 6) if checked else 0.0,
            "mismatch_examples": examples,
            "pocket_chains": sorted(pocket),
        }
    )
    if checked == 0:
        verdict["reason"] = "NO_RESIDUES"
    elif verdict["identity"] >= threshold:
        verdict["match"] = True
        verdict["reason"] = "MATCH"
    else:
        verdict["reason"] = "SEQUENCE_MISMATCH"
    return verdict
