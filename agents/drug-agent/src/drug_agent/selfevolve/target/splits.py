"""Frozen family-level partitions and the contamination exclusions that follow.

Appendix A §3.2 asks for a deterministic 60/20/20 train/development/test
allocation *at the connected structural-family level*. This module is the whole
of that: the family graph, the allocation, the preflight feasibility guard, the
protected exclusion registry the ingestion boundary screens against, and the
immutable manifest that lets any of it be replayed.

Four decisions in here are worth reading before using it.

**A family is a connected component, and the edges are the specification's.**
Two compounds are joined by a shared **nonempty** Bemis-Murcko scaffold, a
shared curated medicinal-chemistry series, a shared connectivity key (the
stereoisomer family), or a Morgan radius-2 / 2,048-bit Tanimoto of at least
:data:`SIMILARITY_THRESHOLD`. An empty scaffold joins nothing — acyclic
compounds are not one family, and pretending they are is how a whole split
leaks while every check still passes. The cutoff is a module constant precisely
so that lowering it to rescue an infeasible split is a visible edit rather than
a quiet parameter change.

**A compound can have more than one recorded structure.** Four InChIKeys in the
real ZAK snapshot carry two different canonical SMILES, because two sources
wrote different tautomers of the same compound. Grouping therefore considers
*every* representation a compound was seen with, and takes the maximum
similarity over representation pairs. Without that, the family graph depends on
which row happened to be ingested first — a reproducibility bug that only shows
up as an unexplained change in the component count.

**Allocation is whole-group, deterministic, and weighted before it is run.**
Families are ordered by weight and by a seeded stable hash, then each is placed
in whichever partition currently has the largest deficit against its target.
The weight dimensions are declared by the caller *before* any candidate search;
they exist so a split can respect label feasibility — Appendix A's "respecting
label feasibility where possible without separating a family" — not so it can be
re-run until it looks better. Nothing here observes a model's performance, and
the returned record is equal to itself on a second call.

**Infeasibility is an output, not an exception to route around.** A single
connected component larger than the smallest partition's target makes 60/20/20
unreachable. That is reported in ``infeasible_reasons`` and by
:func:`preflight_feasibility` as ``INSUFFICIENT_TEST_DATA``. It is never fixed
by cutting the component, shrinking the test share, or reducing the cutoff.

Identifiers are the split; measured values are the answer. Nothing this module
writes — registry, manifest or report — contains a measured label.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...util import content_hash, stable_hash_int
from .contracts import CONTRACT_VERSION, ContractError, ErrorCategory
from .data import _rdkit

Record = dict[str, Any]

# ---------------------------------------------------------------------------
# The specification's constants, written out so a change to one is a diff
# ---------------------------------------------------------------------------
#: Partition names, in allocation-preference order. The order is also the
#: deterministic tie-break when two partitions have an identical deficit.
PARTITIONS: tuple[str, str, str] = ("train", "development", "test")

#: Appendix A §3.2's allocation target.
DEFAULT_FRACTIONS: dict[str, float] = {"train": 0.6, "development": 0.2, "test": 0.2}

#: Appendix A §3.2's similarity edge. Never lower this to make a split feasible.
SIMILARITY_THRESHOLD = 0.80
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

#: The kinds of edge that can put two compounds in one family.
FAMILY_EDGE_KINDS: tuple[str, ...] = (
    "SAME_CONNECTIVITY",
    "SHARED_SCAFFOLD",
    "CURATED_SERIES",
    "SIMILARITY",
)

#: Appendix A §3.2's preflight guard. Minimum software feasibility, never a
#: guarantee of statistical power.
PREFLIGHT_GUARD: dict[str, int] = {
    "min_test_compounds": 30,
    "min_test_families": 5,
    "min_positives": 10,
    "min_negatives": 10,
}

#: Appendix A §3.2's provisional binary label policy, in molar.
LABEL_POLICY: dict[str, float] = {
    "positive_at_most_M": 100e-9,
    "negative_at_least_M": 1000e-9,
}

#: What the protected registry does and does not provide. Quoted into handoffs.
EXCLUSION_BOUNDARY_NOTE = (
    "The protected registry is a process control at the ingestion boundary, not an "
    "access-control boundary. It guarantees that a record matching a held-out compound, "
    "family, stereoisomer or source alias is refused by screen_records() on every "
    "endpoint and every source. It does NOT guarantee that a worker running as the same "
    "OS user could not read the registry directly, and it says nothing about what a "
    "public model already learned from the public databases these compounds come from. "
    "Record public and pretraining exposure as known or unknown, never as secret."
)

_CONTRACT = "SplitManifest"


# ---------------------------------------------------------------------------
# Structural families
# ---------------------------------------------------------------------------
def _representations(compound: Mapping[str, Any]) -> list[str]:
    """Every distinct SMILES this compound was recorded with, sorted."""
    variants = compound.get("smiles_variants")
    if variants is None:
        single = compound.get("smiles")
        variants = [single] if single else []
    if isinstance(variants, str):
        variants = [variants]
    return sorted({str(v) for v in variants if v})


class _Union:
    """Union-find whose result does not depend on the order of the unions."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        parent = self._parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(self, i: int, j: int) -> bool:
        a, b = self.find(i), self.find(j)
        if a == b:
            return False
        self._parent[max(a, b)] = min(a, b)
        return True


def structural_families(
    compounds: Sequence[Mapping[str, Any]],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Record:
    """Connected components over Appendix A §3.2's structural-family edges.

    Each entry needs a ``compound_key`` and at least one structure, given as
    ``smiles`` or as ``smiles_variants`` when the compound was recorded with
    more than one canonical form. ``connectivity_key`` (the InChIKey's first
    block) and ``series_id`` add the stereoisomer and curated-series edges.

    A compound with no parseable structure is still a compound: it becomes a
    singleton family and is counted in ``n_unparseable``, never dropped.

    Returns ``family_of`` (compound key → family id), the families themselves,
    the per-kind edge counts and the cutoff actually used. Family ids are
    assigned after sorting the components, so the result does not depend on the
    order the compounds arrived in.

    Raises:
        ContractError: ``EMPTY_IDENTIFIER`` for a blank ``compound_key``,
            ``INVALID_VALUE`` for a duplicate one, ``MISSING_FIELD`` when a
            compound carries no structure at all.
    """
    _rk, Chem, _desc, _std, MurckoScaffold = _rdkit()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    keys: list[str] = []
    for entry in compounds:
        key = str(entry.get("compound_key") or "").strip()
        if not key:
            raise ContractError(
                ErrorCategory.EMPTY_IDENTIFIER,
                "a compound with no key cannot be grouped, excluded or replayed",
                contract=_CONTRACT,
                field="compound_key",
            )
        keys.append(key)
    if len(set(keys)) != len(keys):
        duplicated = sorted({k for k in keys if keys.count(k) > 1})
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"compound keys must be unique; repeated: {duplicated}",
            contract=_CONTRACT,
            field="compound_key",
            missing=duplicated,
        )

    order = sorted(range(len(keys)), key=lambda i: keys[i])
    entries = [compounds[i] for i in order]
    keys = [keys[i] for i in order]
    n = len(keys)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    scaffold_sets: list[set[str]] = []
    fingerprints: list[list[Any]] = []
    n_unparseable = 0
    n_empty_scaffold = 0
    for entry, key in zip(entries, keys, strict=True):
        smiles = _representations(entry)
        if not smiles:
            raise ContractError(
                ErrorCategory.MISSING_FIELD,
                f"compound {key!r} carries no structure, so it cannot be grouped",
                contract=_CONTRACT,
                field="smiles",
            )
        scaffolds: set[str] = set()
        fps: list[Any] = []
        for text in smiles:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                continue
            fps.append(generator.GetFingerprint(mol))
            core = MurckoScaffold.GetScaffoldForMol(mol)
            if core is not None:
                canonical = Chem.MolToSmiles(core)
                if canonical:
                    scaffolds.add(canonical)
        if not fps:
            n_unparseable += 1
        if not scaffolds:
            n_empty_scaffold += 1
        scaffold_sets.append(scaffolds)
        fingerprints.append(fps)

    union = _Union(n)
    edges = dict.fromkeys(FAMILY_EDGE_KINDS, 0)

    # 1. Stereoisomer family: one connectivity, several InChIKeys.
    by_connectivity: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        value = str(entry.get("connectivity_key") or "").strip()
        if value:
            by_connectivity.setdefault(value, []).append(i)
    for members in by_connectivity.values():
        for j in members[1:]:
            if union.union(members[0], j):
                edges["SAME_CONNECTIVITY"] += 1

    # 2. Curated medicinal-chemistry series, when the dataset declares one.
    by_series: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        value = str(entry.get("series_id") or "").strip()
        if value:
            by_series.setdefault(value, []).append(i)
    for members in by_series.values():
        for j in members[1:]:
            if union.union(members[0], j):
                edges["CURATED_SERIES"] += 1

    # 3. Shared *nonempty* Bemis-Murcko scaffold. An empty one joins nothing.
    by_scaffold: dict[str, list[int]] = {}
    for i, scaffolds in enumerate(scaffold_sets):
        for scaffold in sorted(scaffolds):
            by_scaffold.setdefault(scaffold, []).append(i)
    for members in sorted(by_scaffold.values()):
        for j in members[1:]:
            if union.union(members[0], j):
                edges["SHARED_SCAFFOLD"] += 1

    # 4. Morgan r2/2048 Tanimoto at or above the cutoff, over every pair of
    #    recorded representations, so a tautomer twin cannot hide an edge.
    for i in range(n):
        if not fingerprints[i]:
            continue
        for j in range(i + 1, n):
            if not fingerprints[j] or union.find(i) == union.find(j):
                continue
            best = max(DataStructs.TanimotoSimilarity(a, b) for a in fingerprints[i] for b in fingerprints[j])
            if best >= threshold and union.union(i, j):
                edges["SIMILARITY"] += 1

    grouped: dict[int, list[str]] = {}
    for i in range(n):
        grouped.setdefault(union.find(i), []).append(keys[i])
    components = sorted(sorted(component) for component in grouped.values())

    width = max(4, len(str(max(len(components) - 1, 0))))
    families: list[Record] = []
    family_of: dict[str, str] = {}
    for index, component in enumerate(components):
        family_id = f"fam_{index:0{width}d}"
        families.append({"family_id": family_id, "n_molecules": len(component), "molecule_ids": component})
        for member in component:
            family_of[member] = family_id

    return {
        "contract_version": CONTRACT_VERSION,
        "family_of": family_of,
        "families": families,
        "n_families": len(families),
        "n_molecules": n,
        "largest_family": max((f["n_molecules"] for f in families), default=0),
        "n_empty_scaffold": n_empty_scaffold,
        "n_unparseable": n_unparseable,
        "edges": edges,
        "threshold": threshold,
        "fingerprint": {"kind": "morgan", "radius": MORGAN_RADIUS, "bits": MORGAN_BITS},
        "edge_rules": {
            "shared_nonempty_bemis_murcko_scaffold": True,
            "shared_connectivity_key_stereoisomer_family": True,
            "curated_medicinal_chemistry_series": (
                True if by_series else "UNAVAILABLE: no curated series declared in the input"
            ),
            "morgan_r2_2048_tanimoto_at_least": threshold,
            "empty_scaffolds_are_not_one_family": True,
        },
    }


def _partition_rank(name: str) -> tuple[int, str]:
    """Order partition names by :data:`PARTITIONS`, not alphabetically."""
    return (PARTITIONS.index(name) if name in PARTITIONS else len(PARTITIONS), name)


def cross_partition_max_similarity(
    compounds: Sequence[Mapping[str, Any]],
    membership: Mapping[str, str],
) -> Record:
    """Largest Tanimoto similarity between any two compounds on opposite sides.

    Appendix A §3.2 asks for this to be reported. By construction it must sit
    below :data:`SIMILARITY_THRESHOLD` — a pair at or above the cutoff is one
    family and one family does not cross — so a value at or above the cutoff is
    evidence that the split was not built from the family graph it claims.
    """
    _rk, Chem, _desc, _std, _scaf = _rdkit()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    keys: list[str] = []
    fingerprints: list[list[Any]] = []
    for entry in sorted(compounds, key=lambda e: str(e.get("compound_key"))):
        key = str(entry.get("compound_key") or "")
        if key not in membership:
            continue
        fps = []
        for text in _representations(entry):
            mol = Chem.MolFromSmiles(text)
            if mol is not None:
                fps.append(generator.GetFingerprint(mol))
        if fps:
            keys.append(key)
            fingerprints.append(fps)

    by_pair: dict[str, float] = {}
    worst = 0.0
    worst_pair: list[str] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            side_i, side_j = membership[keys[i]], membership[keys[j]]
            if side_i == side_j:
                continue
            value = max(
                DataStructs.TanimotoSimilarity(a, b) for a in fingerprints[i] for b in fingerprints[j]
            )
            label = "|".join(sorted((side_i, side_j), key=_partition_rank))
            if value > by_pair.get(label, -1.0):
                by_pair[label] = value
            if value > worst:
                worst, worst_pair = value, sorted((keys[i], keys[j]))
    return {
        "max_similarity": round(worst, 6),
        "max_similarity_pair": worst_pair,
        "by_pair": {k: round(v, 6) for k, v in sorted(by_pair.items())},
        "threshold": SIMILARITY_THRESHOLD,
        "n_compared": len(keys),
    }


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def _molecule_view(
    rows: Sequence[Mapping[str, Any]], weight_keys: Sequence[str]
) -> tuple[list[str], dict[str, Record]]:
    """Collapse rows to one record per molecule, refusing any disagreement."""
    view: dict[str, Record] = {}
    order: list[str] = []
    consistent = ("family_id", "connectivity_key", "series_id", *weight_keys)
    for row in rows:
        raw = row.get("molecule_id")
        molecule_id = str(raw).strip() if raw is not None else ""
        if not molecule_id:
            raise ContractError(
                ErrorCategory.EMPTY_IDENTIFIER,
                "a row with no molecule_id cannot be partitioned or excluded later",
                contract=_CONTRACT,
                field="molecule_id",
            )
        entry = view.get(molecule_id)
        if entry is None:
            entry = {"molecule_id": molecule_id, "smiles_variants": set(), "aliases": set()}
            view[molecule_id] = entry
            order.append(molecule_id)
        for key in consistent:
            if key not in row:
                continue
            if key in entry and entry[key] != row[key]:
                raise ContractError(
                    ErrorCategory.INVALID_VALUE,
                    f"molecule {molecule_id!r} carries two different {key} values "
                    f"({entry[key]!r} and {row[key]!r}); a contradictory grouping is a leak",
                    contract=_CONTRACT,
                    field=key,
                )
            entry[key] = row[key]
        for text in _representations(row):
            entry["smiles_variants"].add(text)
        for alias in row.get("aliases", ()) or ():
            if alias:
                entry["aliases"].add(str(alias))
        entry["aliases"].add(molecule_id)
    for entry in view.values():
        entry["smiles_variants"] = sorted(entry["smiles_variants"])
        entry["aliases"] = sorted(entry["aliases"])
    return sorted(order), view


def _resolve_families(
    molecule_ids: Sequence[str], view: Mapping[str, Record], threshold: float
) -> tuple[dict[str, str], Record | None]:
    declared = [mid for mid in molecule_ids if str(view[mid].get("family_id") or "").strip()]
    if declared and len(declared) != len(molecule_ids):
        missing = sorted(set(molecule_ids) - set(declared))
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            "family_id must be declared for every molecule or for none; a half-declared "
            f"grouping invents singleton families for the rest ({len(missing)} of "
            f"{len(molecule_ids)} undeclared)",
            contract=_CONTRACT,
            field="family_id",
            missing=missing[:20],
        )
    if declared:
        return {mid: str(view[mid]["family_id"]) for mid in molecule_ids}, None

    without_structure = [mid for mid in molecule_ids if not view[mid]["smiles_variants"]]
    if without_structure:
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            "a molecule needs either a family_id or a structure to be grouped; "
            f"{len(without_structure)} of {len(molecule_ids)} have neither",
            contract=_CONTRACT,
            field="family_id",
            missing=sorted(without_structure)[:20],
        )
    computed = structural_families(
        [
            {
                "compound_key": mid,
                "smiles_variants": view[mid]["smiles_variants"],
                "connectivity_key": view[mid].get("connectivity_key"),
                "series_id": view[mid].get("series_id"),
            }
            for mid in molecule_ids
        ],
        threshold=threshold,
    )
    return dict(computed["family_of"]), computed


def partition_families(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    fractions: Mapping[str, float] | None = None,
    weight_keys: Sequence[str] = (),
    threshold: float = SIMILARITY_THRESHOLD,
) -> Record:
    """Allocate whole structural families to train/development/test.

    ``rows`` are molecule records. Each needs a ``molecule_id``; grouping comes
    from an already-assigned ``family_id`` when the dataset has one, and
    otherwise from :func:`structural_families` over ``smiles`` /
    ``smiles_variants`` (plus ``connectivity_key`` and ``series_id``). Several
    rows may describe one molecule — that is what a compound measured in five
    assays looks like — but they must agree on every grouping and weight field.

    ``weight_keys`` names the numeric dimensions the allocation balances, in
    priority order, with the molecule count always last. Declaring
    ``("n_exact",)`` makes the split land near 60/20/20 *on the molecules that
    can actually be regressed* rather than on the raw compound count. Declare it
    before any candidate search; re-running with a different weighting after
    seeing a result is exactly the reshuffle this module exists to prevent.

    The allocation is greedy and deterministic: families are ordered by weight
    descending, then by a seeded stable hash, then by family id, and each goes to
    whichever partition has the largest remaining deficit. A family is never
    divided, so exact fractions are not generally reachable; ``infeasible_reasons``
    reports when a single component is large enough to make 60/20/20 unreachable.

    Returns a JSON-serializable record with ``membership``, ``family_of``,
    ``families``, per-partition ``counts``, ``achieved_fractions``,
    ``infeasible_reasons`` and a ``split_hash``. It carries no timestamp and no
    measured value, so it is equal to itself on a second call.

    Raises:
        ContractError: ``EMPTY_IDENTIFIER`` for a blank molecule id,
            ``MISSING_FIELD`` when a molecule has neither family nor structure or
            when only some rows declare a family, ``INVALID_VALUE`` for a
            contradictory grouping, an unknown weight, or fractions that are not
            a positive distribution over :data:`PARTITIONS`.
    """
    shares = dict(DEFAULT_FRACTIONS if fractions is None else fractions)
    if set(shares) != set(PARTITIONS):
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"fractions must cover exactly {list(PARTITIONS)}, got {sorted(shares)}",
            contract=_CONTRACT,
            field="fractions",
        )
    if any(v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            f"fractions must be non-negative and sum to 1.0, got {shares} summing to "
            f"{sum(shares.values())!r}",
            contract=_CONTRACT,
            field="fractions",
        )

    weight_keys = tuple(str(k) for k in weight_keys)
    if "n_molecules" in weight_keys:
        raise ContractError(
            ErrorCategory.INVALID_VALUE,
            "n_molecules is always the final weight dimension and cannot be declared twice",
            contract=_CONTRACT,
            field="weight_keys",
        )
    molecule_ids, view = _molecule_view(rows, weight_keys)
    if not molecule_ids:
        raise ContractError(
            ErrorCategory.MISSING_FIELD,
            "no rows to partition",
            contract=_CONTRACT,
            field="rows",
        )
    family_of, computed = _resolve_families(molecule_ids, view, threshold)

    dimensions = (*weight_keys, "n_molecules")
    members: dict[str, list[str]] = {}
    for molecule_id in molecule_ids:
        members.setdefault(family_of[molecule_id], []).append(molecule_id)
    weights: dict[str, dict[str, float]] = {}
    for family_id, group in members.items():
        totals = dict.fromkeys(dimensions, 0.0)
        totals["n_molecules"] = float(len(group))
        for key in weight_keys:
            for molecule_id in group:
                raw = view[molecule_id].get(key, 0)
                try:
                    totals[key] += float(raw or 0)
                except (TypeError, ValueError) as exc:
                    raise ContractError(
                        ErrorCategory.INVALID_TYPE,
                        f"weight {key!r} on molecule {molecule_id!r} is {raw!r}, not a number",
                        contract=_CONTRACT,
                        field=key,
                    ) from exc
        weights[family_id] = totals

    grand = {d: sum(w[d] for w in weights.values()) for d in dimensions}
    targets = {p: {d: shares[p] * grand[d] for d in dimensions} for p in PARTITIONS}

    order = sorted(
        members,
        key=lambda f: (
            tuple(-weights[f][d] for d in dimensions),
            stable_hash_int(f"{seed}:{f}"),
            f,
        ),
    )
    current = {p: dict.fromkeys(dimensions, 0.0) for p in PARTITIONS}
    family_partition: dict[str, str] = {}
    for family_id in order:
        chosen = max(
            PARTITIONS,
            key=lambda p: (
                tuple(targets[p][d] - current[p][d] for d in dimensions),
                -PARTITIONS.index(p),
            ),
        )
        family_partition[family_id] = chosen
        for d in dimensions:
            current[chosen][d] += weights[family_id][d]

    membership = {mid: family_partition[family_of[mid]] for mid in molecule_ids}
    counts: dict[str, Record] = {}
    for partition in PARTITIONS:
        group = [f for f in sorted(members) if family_partition[f] == partition]
        counts[partition] = {
            "n_families": len(group),
            "n_molecules": int(sum(weights[f]["n_molecules"] for f in group)),
            "weights": {d: round(current[partition][d], 6) for d in dimensions},
        }
    achieved = {
        partition: {d: round(current[partition][d] / grand[d], 6) if grand[d] else 0.0 for d in dimensions}
        for partition in PARTITIONS
    }

    largest_id = max(sorted(members), key=lambda f: (weights[f]["n_molecules"], f))
    largest = {
        "family_id": largest_id,
        "n_molecules": int(weights[largest_id]["n_molecules"]),
        "partition": family_partition[largest_id],
        "share_of_dataset": round(weights[largest_id]["n_molecules"] / len(molecule_ids), 6),
    }
    smallest_target = min(shares[p] for p in PARTITIONS)
    reasons: list[str] = []
    if weights[largest_id]["n_molecules"] > smallest_target * len(molecule_ids):
        reasons.append("LARGEST_FAMILY_EXCEEDS_SMALLEST_TARGET")
    if len(members) < len(PARTITIONS):
        reasons.append("FEWER_FAMILIES_THAN_PARTITIONS")

    identity = {
        "contract_version": CONTRACT_VERSION,
        "seed": int(seed),
        "fractions": {p: shares[p] for p in PARTITIONS},
        "weight_dimensions": list(dimensions),
        "similarity_threshold": threshold,
        "membership": membership,
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "seed": int(seed),
        "fractions": {p: shares[p] for p in PARTITIONS},
        "weight_dimensions": list(dimensions),
        "similarity_threshold": threshold,
        "family_source": "DECLARED" if computed is None else "COMPUTED",
        "family_graph": computed,
        "membership": membership,
        "family_of": family_of,
        "family_partition": family_partition,
        "families": [
            {
                "family_id": family_id,
                "partition": family_partition[family_id],
                "n_molecules": int(weights[family_id]["n_molecules"]),
                "weights": {d: round(weights[family_id][d], 6) for d in dimensions},
                "molecule_ids": sorted(members[family_id]),
            }
            for family_id in sorted(members)
        ],
        "n_molecules": len(molecule_ids),
        "n_families": len(members),
        "counts": counts,
        "achieved_fractions": achieved,
        "largest_family": largest,
        "infeasible_reasons": reasons,
        "allocation_rule": (
            "whole connected families, ordered by weight descending then by "
            "stable_hash_int(f'{seed}:{family_id}') then by family id, each placed in the "
            "partition with the largest remaining deficit; ties by PARTITIONS order"
        ),
        "split_hash": content_hash(identity),
    }


# ---------------------------------------------------------------------------
# Labels and the preflight feasibility guard
# ---------------------------------------------------------------------------
def label_class(
    value_M: float | None,
    relation: str,
    *,
    policy: Mapping[str, float] = LABEL_POLICY,
) -> str:
    """Map one measurement to ``positive`` / ``negative`` / ``intermediate`` /
    ``inconclusive`` / ``unmeasured`` under the provisional policy.

    A censored record is mapped only when its inequality *conclusively*
    establishes the class: ``> 1000 nM`` is inactive, ``> 500 nM`` could be
    600 nM or 60 uM and settles nothing. A molecule that was never measured is
    ``unmeasured`` — Appendix A §3.2 forbids turning it into a negative.
    """
    if value_M is None:
        return "unmeasured"
    positive_at_most = float(policy["positive_at_most_M"])
    negative_at_least = float(policy["negative_at_least_M"])
    value = float(value_M)
    text = str(relation or "=").strip()
    if text in ("=", "~", "=="):
        if value <= positive_at_most:
            return "positive"
        if value >= negative_at_least:
            return "negative"
        return "intermediate"
    if text in (">", ">="):
        return "negative" if value >= negative_at_least else "inconclusive"
    if text in ("<", "<="):
        return "positive" if value <= positive_at_most else "inconclusive"
    return "inconclusive"


def preflight_feasibility(
    rows: Sequence[Mapping[str, Any]],
    partition: Mapping[str, Any],
    *,
    mode: str,
    guard: Mapping[str, int] = PREFLIGHT_GUARD,
    partition_name: str = "test",
    label_key: str = "label_class",
    exact_key: str | None = None,
) -> Record:
    """Appendix A §3.2's guard, run on the held-out partition.

    ``mode`` is ``"regression"`` or ``"ranking"``. Both require at least
    ``min_test_compounds`` unique compounds in at least ``min_test_families``
    family groups; ranking additionally requires ``min_positives`` and
    ``min_negatives``. ``exact_key`` names a per-row flag when only exactly
    measured compounds count toward the compound total, which is what continuous
    regression on a mostly censored stratum actually has.

    ``partition_name`` is ``"test"`` for the guard itself. Pointing it at
    ``"development"`` is how a primary mode is chosen without consulting held-out
    data; the returned ``partition`` field says which one was measured, and the
    guard's own key names keep Appendix A's ``min_test_*`` wording whichever
    partition was read.

    Returns ``FEASIBLE`` or ``INSUFFICIENT_TEST_DATA`` with every measured count
    and the names of the checks that failed. It never proposes a smaller test
    share, a lower similarity cutoff, or a merged stratum.

    Raises:
        ContractError: ``INVALID_ENUM`` for an unrecognized ``mode``.
    """
    if mode not in ("regression", "ranking"):
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"mode must be 'regression' or 'ranking', got {mode!r}",
            contract=_CONTRACT,
            field="mode",
        )
    membership = dict(partition["membership"])
    family_of = dict(partition["family_of"])

    compounds: set[str] = set()
    families: set[str] = set()
    classes: dict[str, int] = {}
    for row in rows:
        molecule_id = str(row.get("molecule_id") or "")
        if membership.get(molecule_id) != partition_name:
            continue
        if exact_key is not None and not row.get(exact_key):
            continue
        compounds.add(molecule_id)
        families.add(str(family_of.get(molecule_id, molecule_id)))
        label = row.get(label_key)
        if label:
            classes[str(label)] = classes.get(str(label), 0) + 1

    n_compounds = len(compounds)
    n_families = len(families)
    n_positive = classes.get("positive", 0)
    n_negative = classes.get("negative", 0)

    failed: list[str] = []
    if n_compounds < int(guard["min_test_compounds"]):
        failed.append("min_test_compounds")
    if n_families < int(guard["min_test_families"]):
        failed.append("min_test_families")
    if mode == "ranking":
        if n_positive < int(guard["min_positives"]):
            failed.append("min_positives")
        if n_negative < int(guard["min_negatives"]):
            failed.append("min_negatives")

    return {
        "contract_version": CONTRACT_VERSION,
        "mode": mode,
        "partition": partition_name,
        "status": "FEASIBLE" if not failed else "INSUFFICIENT_TEST_DATA",
        "guard": dict(guard),
        "n_compounds": n_compounds,
        "n_families": n_families,
        "class_counts": dict(sorted(classes.items())),
        "n_positives": n_positive,
        "n_negatives": n_negative,
        "failed_checks": failed,
        "exact_only": exact_key is not None,
        "note": (
            "Minimum software feasibility, not statistical power. A failure is reported as "
            "INSUFFICIENT_TEST_DATA and answered with grouped development cross-validation, "
            "never by shrinking the test share or lowering the family cutoff."
        ),
    }


# ---------------------------------------------------------------------------
# The protected exclusion registry
# ---------------------------------------------------------------------------
#: The order match keys are tried in. The first hit is the reported reason.
MATCH_KEYS: tuple[str, ...] = ("molecule_id", "source_alias", "connectivity_key", "family_id")


def build_exclusion_registry(
    rows: Sequence[Mapping[str, Any]],
    partition: Mapping[str, Any],
    *,
    measurement_rows: Sequence[Mapping[str, Any]] = (),
    protected: Sequence[str] = ("test",),
) -> Record:
    """Everything the ingestion boundary must refuse, as identifiers only.

    A held-out compound is protected under four identities at once: its own id,
    every source alias it was ever published under (ChEMBL and BindingDB name the
    same experiment differently — the real snapshot has 207 such mirrors), its
    connectivity key so a stereoisomer cannot walk in, and its family id so a
    never-before-seen analogue in a protected component cannot either. The match
    is on identity alone, so it holds across every endpoint and assay format.

    ``measurement_rows`` is optional and adds aliases observed in the measurement
    records — the ``sources[].molecule_id`` a mirror carries.

    The registry contains no measured value of any kind. Identifiers are the
    split; values are the answer.
    """
    membership = dict(partition["membership"])
    family_of = dict(partition["family_of"])
    requested = {str(p) for p in protected}
    protected = [p for p in PARTITIONS if p in requested]
    if not protected:
        # An empty protected set is a silent no-op wearing a registry's name:
        # every later screen would admit everything while the call site still
        # reads as an exclusion. Refuse it here rather than at the leak.
        raise ContractError(
            ErrorCategory.INVALID_ENUM,
            f"protected must name at least one of {list(PARTITIONS)}, got {sorted(requested)}; "
            "a registry that protects nothing would admit every held-out compound",
            contract=_CONTRACT,
            field="protected",
        )

    molecule_ids: set[str] = set()
    families: set[str] = set()
    connectivity: set[str] = set()
    aliases: set[str] = set()
    for molecule_id, side in membership.items():
        if side not in protected:
            continue
        molecule_ids.add(molecule_id)
        families.add(str(family_of.get(molecule_id, molecule_id)))

    def _collect(record: Mapping[str, Any], molecule_id: str) -> None:
        value = str(record.get("connectivity_key") or "").strip()
        if value:
            connectivity.add(value)
        for alias in record.get("aliases", ()) or ():
            if alias and str(alias) != molecule_id:
                aliases.add(str(alias))

    for row in rows:
        molecule_id = str(row.get("molecule_id") or "")
        if molecule_id in molecule_ids:
            _collect(row, molecule_id)
    for row in measurement_rows:
        molecule_id = str(row.get("inchikey") or row.get("molecule_id") or "")
        if molecule_id not in molecule_ids:
            continue
        value = str(row.get("connectivity_key") or "").strip()
        if value:
            connectivity.add(value)
        own = str(row.get("molecule_id") or "")
        if own and own != molecule_id:
            aliases.add(own)
        for source in row.get("sources", ()) or ():
            alias = str(source.get("molecule_id") or "")
            if alias and alias != molecule_id:
                aliases.add(alias)

    aliases -= molecule_ids
    return {
        "contract_version": CONTRACT_VERSION,
        "registry_kind": "protected_exclusion_registry",
        "split_hash": partition["split_hash"],
        "seed": partition["seed"],
        "protected_partitions": protected,
        "molecule_ids": sorted(molecule_ids),
        "family_ids": sorted(families),
        "connectivity_keys": sorted(connectivity),
        "source_aliases": sorted(aliases),
        "n_protected_molecules": len(molecule_ids),
        "n_protected_families": len(families),
        "n_source_aliases": len(aliases),
        "match_keys": list(MATCH_KEYS),
        "boundary_note": EXCLUSION_BOUNDARY_NOTE,
    }


def screen_records(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> Record:
    """Admit or quarantine incoming records against the protected registry.

    Every record must carry at least one identifier. Matching is by identity, so
    a record arriving under a foreign database's id, in a different endpoint and
    a different assay format, is still refused — which is the only way the
    exclusion means anything, because that is the shape a real acquisition
    arrives in.

    Returns ``{"admitted": [...], "quarantined": [...]}``; a quarantine entry
    carries :class:`ErrorCategory.TEST_LEAKAGE`, which key matched and the value
    that matched it. Nothing is dropped silently.

    Raises:
        ContractError: ``MISSING_FIELD`` for a record with no identifier at all,
            which cannot be checked and therefore cannot be admitted.
    """
    protected_ids = set(registry.get("molecule_ids", ()))
    protected_aliases = set(registry.get("source_aliases", ()))
    protected_connectivity = set(registry.get("connectivity_keys", ()))
    protected_families = set(registry.get("family_ids", ()))

    admitted: list[Record] = []
    quarantined: list[Record] = []
    for index, row in enumerate(rows):
        molecule_id = str(row.get("molecule_id") or "").strip()
        connectivity = str(row.get("connectivity_key") or "").strip()
        family_id = str(row.get("family_id") or "").strip()
        extra = [str(a) for a in (row.get("aliases") or ()) if a]
        if not (molecule_id or connectivity or family_id or extra):
            raise ContractError(
                ErrorCategory.MISSING_FIELD,
                f"record {index} carries no identifier, so it cannot be screened against "
                "the protected registry and cannot be admitted",
                contract=_CONTRACT,
                field="molecule_id",
            )
        candidates: list[tuple[str, str]] = []
        if molecule_id and molecule_id in protected_ids:
            candidates.append(("molecule_id", molecule_id))
        for alias in [molecule_id, *extra]:
            if alias and alias in protected_aliases:
                candidates.append(("source_alias", alias))
                break
        for alias in extra:
            if alias and alias in protected_ids:
                candidates.append(("molecule_id", alias))
                break
        if connectivity and connectivity in protected_connectivity:
            candidates.append(("connectivity_key", connectivity))
        if family_id and family_id in protected_families:
            candidates.append(("family_id", family_id))
        if not candidates:
            admitted.append(dict(row))
            continue
        matched_on, matched_value = min(candidates, key=lambda c: MATCH_KEYS.index(c[0]))
        quarantined.append(
            {
                "record_index": index,
                "category": ErrorCategory.TEST_LEAKAGE.value,
                "matched_on": matched_on,
                "matched_value": matched_value,
                "molecule_id": molecule_id or None,
                "source": row.get("source"),
                "endpoint": row.get("endpoint"),
                "reason": (
                    f"{matched_on}={matched_value!r} is in the protected registry for split "
                    f"{registry.get('split_hash')}; held-out identity is matched across every "
                    "endpoint and every source"
                ),
            }
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "split_hash": registry.get("split_hash"),
        "admitted": admitted,
        "quarantined": quarantined,
        "n_admitted": len(admitted),
        "n_quarantined": len(quarantined),
        "n_screened": len(rows),
        "match_keys": list(MATCH_KEYS),
    }


# ---------------------------------------------------------------------------
# The immutable manifest
# ---------------------------------------------------------------------------
def build_split_manifest(
    *,
    partition: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    target_manifest_hash: str,
    data_hash: str | None = None,
    stratum_id: str | None = None,
    primary_mode: str | None = None,
    similarity_report: Mapping[str, Any] | None = None,
    exclusion_registry: Mapping[str, Any] | None = None,
    composition: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> Record:
    """The frozen description of one split, hashable and replayable.

    It carries counts, hashes, the edge rules and the cutoff that were actually
    used, the feasibility verdict, and — deliberately — **no measured label** and
    no timestamp. A wall-clock stamp would make two builds of the same split
    unequal artifacts describing identical data.

    Membership itself is reachable through ``split_hash``: the manifest commits
    to it without publishing the held-out identities alongside the public
    summary.
    """
    manifest: Record = {
        "contract_version": CONTRACT_VERSION,
        "manifest_kind": "target_split_manifest",
        "target_manifest_hash": target_manifest_hash,
        "data_hash": data_hash,
        "stratum_id": stratum_id,
        "primary_mode": primary_mode,
        "seed": partition["seed"],
        "fractions": dict(partition["fractions"]),
        "weight_dimensions": list(partition["weight_dimensions"]),
        "similarity_threshold": partition["similarity_threshold"],
        "family_source": partition["family_source"],
        "edge_rules": (partition.get("family_graph") or {}).get("edge_rules"),
        "family_edges": (partition.get("family_graph") or {}).get("edges"),
        "n_molecules": partition["n_molecules"],
        "n_families": partition["n_families"],
        "counts": dict(partition["counts"]),
        "achieved_fractions": dict(partition["achieved_fractions"]),
        "largest_family": dict(partition["largest_family"]),
        "infeasible_reasons": list(partition["infeasible_reasons"]),
        "allocation_rule": partition["allocation_rule"],
        "split_hash": partition["split_hash"],
        "feasibility": dict(feasibility),
        "cross_partition_similarity": dict(similarity_report) if similarity_report else None,
        "composition": dict(composition) if composition else None,
        "exclusion": (
            {
                "n_protected_molecules": exclusion_registry["n_protected_molecules"],
                "n_protected_families": exclusion_registry["n_protected_families"],
                "n_source_aliases": exclusion_registry["n_source_aliases"],
                "match_keys": list(exclusion_registry["match_keys"]),
                "boundary_note": exclusion_registry["boundary_note"],
            }
            if exclusion_registry
            else None
        ),
        "notes": list(notes),
        "determinism": (
            "byte-reproducible from the same rows, seed, fractions, weight dimensions and "
            "cutoff; carries no timestamp so two builds of one split are one artifact"
        ),
    }
    manifest["manifest_hash"] = content_hash(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Access: written restrictively, then actually measured
# ---------------------------------------------------------------------------
def write_protected_registry(directory: Path | str, registry: Mapping[str, Any]) -> Record:
    """Write the registry under owner-only permissions and report what was set.

    The modes are real and are read back from the filesystem. They are also not
    a security boundary when the reader shares the owner's uid — see
    :func:`probe_protected_access`, which measures rather than assumes.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    path = root / "protected_exclusion_registry.json"
    path.write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    dir_stat, file_stat = root.stat(), path.stat()
    return {
        "path": str(path),
        "dir_mode": format(dir_stat.st_mode & 0o777, "04o"),
        "file_mode": format(file_stat.st_mode & 0o777, "04o"),
        "owner_uid": file_stat.st_uid,
        "bytes": file_stat.st_size,
        "content_hash": content_hash(registry),
        "boundary_note": EXCLUSION_BOUNDARY_NOTE,
    }


def probe_protected_access(path: Path | str, *, timeout_s: float = 60.0) -> Record:
    """Try to read the protected file from a real subprocess, and report what happened.

    This deliberately does not assert that the file is unreachable. It runs a
    separate process and records whether that process could open it, alongside
    the uid on each side. On a host where the worker runs as the file's owner the
    honest result is ``readable: True`` and ``boundary: "NONE_SAME_UID"``; a run
    on such a host may not be called sealed.

    ``boundary`` has three values, not two. A probe that failed before it could
    answer reports ``UNDETERMINED``: "the child said nothing" is not the same
    fact as "the child was refused", and collapsing them would manufacture
    evidence of a boundary out of a crash.
    """
    target = Path(path)
    file_stat = target.stat()
    script = (
        "import json,os,sys\n"
        "p=sys.argv[1]\n"
        "try:\n"
        "    n=len(json.loads(open(p).read()).get('molecule_ids',[]))\n"
        "    print(json.dumps({'readable':True,'uid':os.getuid(),'n':n}))\n"
        "except OSError as e:\n"
        "    print(json.dumps({'readable':False,'uid':os.getuid(),'error':type(e).__name__}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    try:
        observed = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        observed = {}
    reported = observed.get("readable")
    readable = reported is True
    probe_uid = int(observed.get("uid", os.getuid()))

    # Three outcomes, not two. A probe that crashed before it could answer has
    # told us nothing, and reporting that as OS_PERMISSIONS would invent exactly
    # the boundary this function exists to check for.
    if readable:
        boundary = "NONE_SAME_UID"
        note = (
            "The probe runs as the same OS user as the owner, so file permissions are not an "
            "access-control boundary here; the protected registry is a process control at the "
            "ingestion boundary. A run on this host is not sealed."
        )
    elif reported is False:
        boundary = "OS_PERMISSIONS"
        note = "The probe could not open the file with its own runtime identity."
    else:
        boundary = "UNDETERMINED"
        note = (
            "The probe process did not report a result, so nothing was measured. This is not "
            "evidence of a boundary; treat access as unverified until a probe reports."
        )
    return {
        "path": str(target),
        "probe_uid": probe_uid,
        "file_owner_uid": file_stat.st_uid,
        "file_mode": format(file_stat.st_mode & 0o777, "04o"),
        "readable": readable,
        "reported": reported,
        "exit_code": proc.returncode,
        "boundary": boundary,
        "evidence": "executed: a separate process attempted to open the file",
        "note": note,
        "stderr": proc.stderr.strip()[:400],
    }


__all__ = [
    "DEFAULT_FRACTIONS",
    "EXCLUSION_BOUNDARY_NOTE",
    "FAMILY_EDGE_KINDS",
    "LABEL_POLICY",
    "MATCH_KEYS",
    "MORGAN_BITS",
    "MORGAN_RADIUS",
    "PARTITIONS",
    "PREFLIGHT_GUARD",
    "SIMILARITY_THRESHOLD",
    "build_exclusion_registry",
    "build_split_manifest",
    "cross_partition_max_similarity",
    "label_class",
    "partition_families",
    "preflight_feasibility",
    "probe_protected_access",
    "screen_records",
    "structural_families",
    "write_protected_registry",
]
