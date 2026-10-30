"""Experimental structures — the coordinates somebody actually measured.

Every other adapter in this package *predicts* where atoms are. This one reads
where they were **observed**: a deposited X-ray, cryo-EM or NMR model, parsed out
of a local cache of RCSB PDB entries. That makes it the cheapest tool here and the
one whose answer outranks the rest — a 1.95 Å co-crystal is not a competitor to a
docking pose, it is the thing a docking pose is trying to reproduce.

**What it computes.** From a PDB id, a UniProt accession or a gene symbol it
returns an :class:`ExperimentalStructure` read out of the file itself:

    ``method``            the deposited experiment, from ``EXPDTA`` (PDB) or
                          ``_exptl.method`` (mmCIF) — never inferred
    ``resolution_a``      Å, from ``REMARK   2 RESOLUTION.`` or
                          ``_refine.ls_d_res_high``; ``None`` for methods that
                          report none, which is not the same as 0.0
    ``r_work`` / ``r_free``  refinement residuals when the file carries them
    ``chains``            one entry per chain actually present in the
                          coordinates, with its residue and atom counts, its
                          ``SEQRES`` length, and its one-letter sequence
    ``het_groups``        every non-polymer group, with the atoms present in the
                          file, its ``HETNAM`` name and ``FORMUL`` formula, and
                          the centroid of its heavy atoms
    ``ligands``           those het groups minus water and minus a named,
                          documented set of crystallisation additives
    ``uniprot_accessions``   from ``DBREF`` / ``_struct_ref``
    ``path``              the file on disk, which is what a docking or pocket
                          step actually needs

The centroid is arithmetic over the coordinates in the file, so it is a real
measurement of where that ligand sits and can be handed straight to a docking box
or to ``scripts/prepare_pocket.py``.

**Setup reaches the network; running never does.** ``scripts/fetch_experimental_
structures.py`` is the only piece that talks to RCSB — ``https://files.rcsb.org/
download/<ID>.pdb`` for coordinates, ``https://search.rcsb.org/rcsbsearch/v2/
query`` to turn a UniProt accession into entry ids — and it goes through
:class:`~drug_agent.security.egress.EgressPolicy` like everything else, so it
refuses unless an operator has deliberately opened egress. This adapter opens no
socket at all: it globs a directory and parses bytes. A cache that is absent or
empty is UNAVAILABLE, naming the directory and the command that fills it.

**Where provenance comes from, and where it does not.** The fetch script writes
``rcsb_fetch_index.json`` beside the coordinates — source URL, HTTP status, byte
count, SHA-256 and a UTC fetch timestamp per entry, plus the query that selected
them. That record is *provenance only*. Not one scientific value in this module is
read from it: resolution, method, ligands and sequences all come from parsing the
coordinate file, every time, so a stale or hand-edited index can misattribute a
file but can never invent a number. When a cached file has no fetch record — the
two sEH entries in ``.external/pdb`` predate the script — the structure says so
with a ``no_fetch_record`` qc flag and reports its own SHA-256 instead of claiming
a download it cannot evidence. When the record exists but the hash has moved, that
is flagged too rather than quietly preferred.

**What it deliberately does not do.** It does not prepare anything. The
coordinates are the deposited ones: waters present, alternate locations
unresolved, no hydrogens added, no protonation states assigned, occupancies
untouched. ``ProteinStructure.prepared`` is ``False`` and
``preparation_assumptions`` says exactly that, because a receptor that has been
silently "cleaned" is a different molecule from the one that was measured. It does
not dock, score, or rank by anything but the numbers the depositor reported. And
it does not fall back to a prediction when the cache has no entry for a target:
the answer to "is there an experimental structure of this" is allowed to be no.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...contracts.base import Provenance
from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.structure import ProteinStructure
from ...data.snapshot import DataSnapshot
from ...observability.toolmeter import measure
from .._ext.adapter import NO_BACKEND
from .._ext.spec import ExternalSpec
from . import StructureError, StructureToolAdapter

#: This adapter's contract with the PDB file formats, not a version of the PDB.
EXPERIMENTAL_ADAPTER_VERSION = "1.0.0"

#: What the fetch script writes beside the coordinates. Provenance only — see the
#: module docstring on why no scientific value is ever read out of it.
FETCH_INDEX_NAME = "rcsb_fetch_index.json"

#: Coordinate files the cache is scanned for, gzip included because that is what
#: ``files.rcsb.org`` serves by default for large entries.
STRUCTURE_SUFFIXES: tuple[str, ...] = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".cif.gz")

#: Groups that are solvent or crystallisation aids rather than a bound ligand.
#: This is a *presentation* choice, not a filter: ``het_groups`` always carries
#: every group the file contains and ``ligands`` is the same list minus these, so
#: a caller that disagrees with the set can ignore it. Waters are separate.
# A 43-code table reads as a grid, not as one name per line.
# fmt: off
CRYSTALLISATION_ADDITIVES: frozenset[str] = frozenset(
    {
        "HOH", "DOD", "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE",
        "NI", "CD", "CU", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "P6G", "DMS",
        "MPD", "TRS", "EPE", "ACT", "ACY", "FMT", "IMD", "BME", "MES", "IOD",
        "BR", "NO3", "CIT", "TLA", "MLI", "AZI", "NH4", "UNX", "UNL",
    }
)
# fmt: on

#: Water, kept separate from the additive set because "does this entry contain a
#: ligand" and "how well ordered is the solvent" are different questions.
WATER_CODES: frozenset[str] = frozenset({"HOH", "DOD", "WAT"})

#: Three-letter to one-letter, for the sequence read out of ``SEQRES``. ``MSE`` is
#: included because selenomethionine is a phasing substitution for methionine and
#: every downstream consumer wants the parent residue; everything else unknown
#: becomes ``X`` rather than being dropped, so the sequence keeps its numbering.
# fmt: off
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}
# fmt: on

#: Nucleotide residue names, used only to label a chain's ``kind``.
_NUCLEIC = frozenset({"DA", "DC", "DG", "DT", "DI", "A", "C", "G", "U", "I", "N"})

SPEC = ExternalSpec(
    name="RCSB experimental structure cache",
    env_prefix="EXPERIMENTAL",
    checkout_relpath=".external/pdb",
    # Deliberately no sentinel file: a sentinel would name one PDB id, and this
    # cache is whatever an operator fetched. Emptiness is checked in __init__
    # instead, which is the failure a bare `is_dir()` would miss.
    selection_modules=("Bio",),
    # PDB records are parsed here with the standard library, the way
    # scripts/prepare_pocket.py already does; biopython is on the run path only
    # for mmCIF, through Bio.PDB.MMCIF2Dict. It is a declared dependency of the
    # main environment, so declaring it here costs nothing and stops an mmCIF
    # entry failing mid-call on a machine where the cache looked fine.
    runtime_modules=("Bio",),
    conda_envs=("drug-agent",),
    upstream_url="https://www.rcsb.org",
    upstream_ref="wwPDB file formats: PDB v3.30, PDBx/mmCIF",
    license="RCSB PDB entries are released under CC0 1.0 Universal (public domain)",
    install_hint="make experimental-structures",
    needs_gpu=False,
)


class ExperimentalStructureError(StructureError):
    """The cache was populated, a file was selected, and it could not be read."""


@dataclass(frozen=True)
class HetGroup:
    """One non-polymer group, as the coordinate file records it."""

    comp_id: str
    chain: str
    res_seq: str
    n_atoms: int
    n_heavy_atoms: int
    centroid: tuple[float, float, float] | None = None
    name: str | None = None
    formula: str | None = None

    @property
    def is_water(self) -> bool:
        return self.comp_id.upper() in WATER_CODES

    @property
    def is_additive(self) -> bool:
        """Solvent, ion or cryoprotectant by the documented set above."""
        return self.comp_id.upper() in CRYSTALLISATION_ADDITIVES

    def label(self) -> str:
        return f"{self.comp_id}/{self.chain}{self.res_seq}"


@dataclass(frozen=True)
class ChainInfo:
    """One chain present in the coordinates, counted rather than assumed."""

    chain_id: str
    kind: str
    n_residues: int
    n_atoms: int
    #: Length declared by ``SEQRES`` / ``_entity_poly``, which includes residues
    #: that were never resolved. Differs from ``n_residues`` on purpose.
    seqres_length: int | None = None
    sequence: str = ""

    @property
    def n_unresolved(self) -> int | None:
        """Declared residues with no coordinates — disorder, not a parse failure."""
        if self.seqres_length is None:
            return None
        return max(0, self.seqres_length - self.n_residues)


@dataclass
class ExperimentalStructure:
    """One deposited entry: what was measured, what is in it, and where it lives."""

    pdb_id: str = ""
    path: str = ""
    file_format: str = "pdb"
    title: str | None = None
    classification: str | None = None
    deposited_on: str | None = None
    method: str | None = None
    resolution_a: float | None = None
    r_work: float | None = None
    r_free: float | None = None
    chains: list[ChainInfo] = field(default_factory=list)
    het_groups: list[HetGroup] = field(default_factory=list)
    uniprot_accessions: list[str] = field(default_factory=list)
    gene_names: list[str] = field(default_factory=list)
    molecule_names: list[str] = field(default_factory=list)
    organisms: list[str] = field(default_factory=list)
    n_atoms: int = 0
    n_models: int = 1
    sha256: str = ""
    n_bytes: int = 0
    #: The fetch record for this file, if the setup script left one. Provenance
    #: only; never a source of a reported measurement.
    fetch_record: dict[str, Any] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)

    # --- derived views -----------------------------------------------------
    @property
    def waters(self) -> list[HetGroup]:
        return [g for g in self.het_groups if g.is_water]

    @property
    def ligands(self) -> list[HetGroup]:
        """Non-water, non-additive groups, largest first.

        Size orders them because a co-crystallised inhibitor is essentially always
        the biggest non-polymer group in an entry — but the ordering is a
        convenience, not a claim, and every group stays visible in
        ``het_groups``.
        """
        return sorted(
            (g for g in self.het_groups if not g.is_water and not g.is_additive),
            key=lambda g: (g.n_heavy_atoms, g.n_atoms),
            reverse=True,
        )

    @property
    def primary_ligand(self) -> HetGroup | None:
        ligands = self.ligands
        return ligands[0] if ligands else None

    @property
    def has_ligand(self) -> bool:
        return bool(self.ligands)

    @property
    def sequences(self) -> dict[str, str]:
        """One-letter sequence per polypeptide chain, from ``SEQRES``.

        This is the input a folding model wants, taken from the deposited
        construct rather than from a database — so a complex predicted from it is
        a prediction *of this entry*, comparable with it.
        """
        return {c.chain_id: c.sequence for c in self.chains if c.kind == "polypeptide" and c.sequence}

    @property
    def source(self) -> str:
        """Where this file came from, as far as it can be evidenced."""
        url = self.fetch_record.get("url")
        return str(url) if url else f"local:{self.path}"

    @property
    def snapshot_id(self) -> str:
        """``<pdb id>@<first 19 hex of the file's SHA-256>``.

        The same shape as :class:`~drug_agent.data.snapshot.DataSnapshot`'s, and
        computed the only way that is honest for a file on disk: over the bytes
        that were actually parsed.
        """
        return f"{self.pdb_id or 'entry'}@{self.sha256[:19]}"

    def provenance(self) -> Provenance:
        """Lineage for this structure, including whether a fetch record backs it."""
        notes = "no fetch record in the cache; hash computed from the local file"
        if self.fetch_record:
            notes = (
                f"fetched {self.fetch_record.get('fetched_at', 'at an unrecorded time')}"
                f" from {self.fetch_record.get('url', 'an unrecorded url')}"
            )
        return Provenance(
            source=self.source,
            source_version=self.fetch_record.get("fetched_at"),
            method="structure.experimental",
            method_version=EXPERIMENTAL_ADAPTER_VERSION,
            snapshot_id=self.snapshot_id,
            content_hash=self.sha256,
            # Deposited coordinates were measured in the physical world. This is
            # the only adapter in this package entitled to say OBSERVED.
            trust_label=TrustLabel.OBSERVED,
            notes=notes,
        )

    def to_protein_structure(self) -> ProteinStructure:
        """The platform's receptor contract, filled only from what was parsed."""
        polymer = [c for c in self.chains if c.kind == "polypeptide"]
        sequence = "".join(c.sequence for c in polymer)
        return ProteinStructure(
            accession=self.uniprot_accessions[0] if self.uniprot_accessions else None,
            name=self.molecule_names[0] if self.molecule_names else self.title,
            sequence_checksum=hashlib.sha256(sequence.encode()).hexdigest() if sequence else None,
            construct_label=",".join(c.chain_id for c in polymer) or None,
            coverage=self._coverage(polymer),
            pdb_id=self.pdb_id or None,
            method=self.method,
            resolution=self.resolution_a,
            quality_notes=self._quality_notes(),
            # The deposited file, untouched. Anything else would be a claim that
            # a preparation step ran, and none has.
            prepared=False,
            preparation_assumptions=[
                "deposited coordinates as released; no preparation has been applied",
                "waters and crystallisation additives are still present",
                "alternate locations unresolved; occupancies untouched",
                "no hydrogens added and no protonation states assigned",
            ],
            structure_ref=self.path,
            provenance=self.provenance(),
        )

    @staticmethod
    def _coverage(polymer: list[ChainInfo]) -> str | None:
        """Resolved-versus-declared residues, per chain, as plain text."""
        parts = [
            f"{c.chain_id}:{c.n_residues}/{c.seqres_length}" for c in polymer if c.seqres_length is not None
        ]
        return "; ".join(parts) or None

    def _quality_notes(self) -> str | None:
        bits: list[str] = []
        if self.resolution_a is not None:
            bits.append(f"resolution {self.resolution_a:.2f} A")
        if self.r_work is not None:
            bits.append(f"R_work {self.r_work:.3f}")
        if self.r_free is not None:
            bits.append(f"R_free {self.r_free:.3f}")
        if self.waters:
            bits.append(f"{len(self.waters)} ordered waters")
        ligands = self.ligands
        if ligands:
            bits.append("ligands " + ", ".join(g.label() for g in ligands))
        return "; ".join(bits) or None

    def metrics_for_ledger(self) -> dict[str, float]:
        """The numbers worth carrying across runs, as floats.

        Resolution is the one number that decides whether this structure should
        outrank a prediction at all, and the ligand count is how "we resolved a
        target to an apo structure" stops being invisible.
        """
        out: dict[str, float] = {
            "experimental_n_chains": float(len(self.chains)),
            "experimental_n_ligands": float(len(self.ligands)),
            "experimental_n_atoms": float(self.n_atoms),
        }
        if self.resolution_a is not None and math.isfinite(self.resolution_a):
            out["experimental_resolution_a"] = float(self.resolution_a)
        if self.r_free is not None and math.isfinite(self.r_free):
            out["experimental_r_free"] = float(self.r_free)
        return out


class ExperimentalStructureAdapter(StructureToolAdapter):
    """Resolve a target or a PDB id to a real, measured structure in the local cache."""

    capability_id = "structure.experimental"
    spec = SPEC
    engine = "rcsb-pdb-cache"
    version = EXPERIMENTAL_ADAPTER_VERSION
    produces_nothing = "resolves no experimental structures"
    trust_label = TrustLabel.OBSERVED

    def __init__(
        self,
        *,
        settings: Any = None,
        timeout_s: float | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        # Parsing two 400 kB text files is milliseconds; the ceiling exists only so
        # a pathological cache cannot wedge a task.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 60.0)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else self.ext.root
        self.entry_paths = _scan(self.cache_dir) if self.cache_dir else []
        if self.cache_dir is not None and not self.entry_paths:
            # `_is_real_checkout` is satisfied by any directory, so an empty cache
            # would otherwise report AVAILABLE and then resolve nothing. Appending
            # here keeps available(), the card and the raised exception on one
            # answer, the way Boltz2Adapter does for its checkpoints.
            self._missing = [*self._missing, self._empty_cache_blocker()]
            self.backend = NO_BACKEND
        self._parsed: dict[Path, ExperimentalStructure] = {}
        self._fetch_index: dict[str, dict[str, Any]] | None = None

    # --- configuration -----------------------------------------------------
    def _empty_cache_blocker(self) -> str:
        return (
            f"the experimental-structure cache {self.cache_dir} contains no "
            f"{'/'.join(STRUCTURE_SUFFIXES)} file — populate it with "
            "`python scripts/fetch_experimental_structures.py 3I28 5AI0`, which downloads from "
            "https://files.rcsb.org/download/<ID>.pdb once and is the only step that "
            "touches the network"
        )

    @property
    def unavailable_reason(self) -> str | None:
        """The whole reason, including the emptiness the backend's probe cannot see."""
        if not self._missing:
            return None
        hint = f" Install it with `{self.spec.install_hint}`." if self.spec.install_hint else ""
        return (
            f"{self.spec.name} is not usable here, so this capability is UNAVAILABLE and "
            f"{self.produces_nothing} — there is no surrogate backend.{hint} "
            f"Reason: {'; '.join(self._missing)}."
        )

    # --- profile -----------------------------------------------------------
    @property
    def profile(self) -> CapabilityProfile:
        requirements = dict(self.ext.runtime_requirements())
        requirements["cache_dir"] = str(self.cache_dir) if self.cache_dir else None
        requirements["n_entries"] = len(self.entry_paths)
        requirements["network_at_runtime"] = False
        return self.build_profile(
            title="Experimental structures from a local RCSB cache",
            description=(
                "Resolves a PDB id, UniProt accession or gene symbol to a deposited "
                "experimental structure held in a local cache, and reports what the file "
                "itself says: method, resolution, refinement residuals, chains and their "
                "sequences, and every non-polymer group with the centroid of its heavy "
                "atoms. Coordinates are the deposited ones — nothing is prepared, cleaned "
                "or predicted. Populating the cache is a separate, network-using step."
            ),
            output_schema_ref="ProteinStructure",
            runtime_requirements=requirements,
            # The same file parses to the same numbers on every machine.
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            known_failure_modes=[
                "the cache holds no entry for the requested target; there is no fallback",
                "an apo entry resolves with zero ligands, which is an answer, not an error",
                "resolution is absent for NMR and for some cryo-EM entries",
                "a cached file with no fetch record cannot evidence where it came from",
                "obsoleted or superseded entries are served from the cache until refetched",
                "mmCIF parsing needs biopython; a PDB-only cache does not",
            ],
            validation_checks=[
                "method and resolution are parsed from the file, never defaulted",
                "resolution is None rather than 0.0 when the entry reports none",
                "every het group's centroid is computed from its own heavy atoms",
                "a file whose hash differs from its fetch record is flagged, not preferred",
            ],
        )

    # --- operations --------------------------------------------------------
    def resolve(self, pdb_id: str) -> ExperimentalStructure:
        """The cached entry with this PDB id, parsed."""
        self.require_available()
        with measure(self.capability_id, "resolve", backend=self.backend) as metered:
            wanted = pdb_id.strip().upper()
            for path in self.entry_paths:
                if _stem_id(path) == wanted:
                    structure = self._parse(path)
                    for key, value in structure.metrics_for_ledger().items():
                        metered[key] = value
                    return structure
            known = ", ".join(sorted(_stem_id(p) for p in self.entry_paths)) or "nothing"
            raise ExperimentalStructureError(
                f"no experimental structure for {wanted!r} in {self.cache_dir} (cache holds {known}); "
                f"fetch it with `python scripts/fetch_experimental_structures.py {wanted}`"
            )

    def find(
        self,
        *,
        target: str | None = None,
        with_ligand: bool = False,
        method: str | None = None,
        max_resolution_a: float | None = None,
    ) -> list[ExperimentalStructure]:
        """Every cached entry matching *target*, best first.

        *target* is matched against the PDB id, the ``DBREF`` UniProt accessions,
        the ``SOURCE`` gene names and the ``COMPND`` molecule names — all read from
        the files, so the cache needs no separate mapping table to keep in step.
        Ordering is by resolution ascending, entries with a ligand first, because
        those are the two things that decide whether a structure is worth docking
        into; unresolved resolutions sort last rather than being dropped.
        """
        self.require_available()
        with measure(self.capability_id, "find", backend=self.backend) as metered:
            hits = [s for s in self._all() if _matches(s, target)]
            if with_ligand:
                hits = [s for s in hits if s.has_ligand]
            if method:
                needle = method.strip().lower()
                hits = [s for s in hits if s.method and needle in s.method.lower()]
            if max_resolution_a is not None:
                hits = [s for s in hits if s.resolution_a is not None and s.resolution_a <= max_resolution_a]
            hits.sort(key=_ranking_key)
            metered["experimental_n_matches"] = float(len(hits))
            metered["experimental_n_cached"] = float(len(self.entry_paths))
            return hits

    def best_for_target(self, target: str, *, with_ligand: bool = False) -> ExperimentalStructure:
        """The single best cached structure for *target*, or a refusal naming what is there."""
        hits = self.find(target=target, with_ligand=with_ligand)
        if not hits:
            known = ", ".join(sorted(_stem_id(p) for p in self.entry_paths)) or "nothing"
            qualifier = " with a bound ligand" if with_ligand else ""
            raise ExperimentalStructureError(
                f"no cached experimental structure{qualifier} matches {target!r} "
                f"(cache {self.cache_dir} holds {known}); fetch one with "
                f"`python scripts/fetch_experimental_structures.py --uniprot <accession>`"
            )
        return hits[0]

    def receptor(self, pdb_id: str) -> ProteinStructure:
        """The platform's receptor contract for one entry, ready for the blackboard."""
        return self.resolve(pdb_id).to_protein_structure()

    def catalog(self) -> DataSnapshot:
        """The whole cache as a content-hashed snapshot with a data card.

        One record per entry, carrying what was parsed *and* whether a fetch record
        backs it. The card is what a report cites when it says a run used measured
        coordinates, so it has to be able to say "and this one arrived without
        provenance" out loud.
        """
        self.require_available()
        with measure(self.capability_id, "catalog", backend=self.backend) as metered:
            records = [
                {
                    "pdb_id": s.pdb_id,
                    "path": s.path,
                    "file_format": s.file_format,
                    "method": s.method,
                    "resolution_a": s.resolution_a,
                    "r_work": s.r_work,
                    "r_free": s.r_free,
                    "chains": [c.chain_id for c in s.chains],
                    "ligands": [g.comp_id for g in s.ligands],
                    "uniprot_accessions": s.uniprot_accessions,
                    "gene_names": s.gene_names,
                    "sha256": s.sha256,
                    "n_bytes": s.n_bytes,
                    "source": s.source,
                    "fetched_at": s.fetch_record.get("fetched_at"),
                    "qc_flags": s.qc_flags,
                }
                for s in self._all()
            ]
            metered["experimental_n_cached"] = float(len(records))
            n_with_record = sum(1 for r in records if r["fetched_at"])
            return DataSnapshot(
                name="rcsb_experimental_structures",
                source=str(self.cache_dir),
                kind="experimental_structures",
                records=records,
                license=self.spec.license,
                metadata={
                    "fetch_index": FETCH_INDEX_NAME,
                    "n_with_fetch_record": n_with_record,
                    "n_without_fetch_record": len(records) - n_with_record,
                    "download_endpoint": "https://files.rcsb.org/download/<ID>.pdb",
                    "search_endpoint": "https://search.rcsb.org/rcsbsearch/v2/query",
                    "network_at_runtime": False,
                },
            )

    # --- real backend ------------------------------------------------------
    def _all(self) -> list[ExperimentalStructure]:
        return [self._parse(path) for path in self.entry_paths]

    def _parse(self, path: Path) -> ExperimentalStructure:
        """Parse one cached file, memoised for the life of this adapter."""
        cached = self._parsed.get(path)
        if cached is not None:
            return cached
        try:
            raw = _read_bytes(path)
        except OSError as exc:
            raise ExperimentalStructureError(f"cannot read {path}: {exc}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        try:
            if _is_mmcif(path):
                structure = _parse_mmcif(raw.decode("utf-8", "replace"), path)
            else:
                structure = _parse_pdb(raw.decode("utf-8", "replace"), path)
        except ExperimentalStructureError:
            raise
        except (ValueError, KeyError, IndexError) as exc:
            raise ExperimentalStructureError(f"{path} is not a readable structure file: {exc}") from exc
        structure.sha256 = digest
        structure.n_bytes = len(raw)
        structure.path = str(path)
        if not structure.pdb_id:
            structure.pdb_id = _stem_id(path)
            structure.qc_flags.append("pdb_id_taken_from_filename")
        self._attach_fetch_record(structure)
        self._parsed[path] = structure
        return structure

    def _attach_fetch_record(self, structure: ExperimentalStructure) -> None:
        """Bind the fetch record for this entry, and say so when it disagrees."""
        index = self._load_fetch_index()
        record = index.get(structure.pdb_id.upper())
        if not record:
            structure.qc_flags.append("no_fetch_record")
            return
        recorded = str(record.get("sha256") or "")
        if recorded and recorded != structure.sha256:
            structure.qc_flags.append("sha256_differs_from_fetch_record")
        structure.fetch_record = dict(record)

    def _load_fetch_index(self) -> dict[str, dict[str, Any]]:
        """Read ``rcsb_fetch_index.json`` once, tolerating its absence."""
        if self._fetch_index is not None:
            return self._fetch_index
        index: dict[str, dict[str, Any]] = {}
        path = (self.cache_dir / FETCH_INDEX_NAME) if self.cache_dir else None
        if path is not None and path.is_file():
            try:
                card = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                card = {}
            records = card.get("records") if isinstance(card, dict) else None
            for entry in records or []:
                if isinstance(entry, dict) and entry.get("pdb_id"):
                    index[str(entry["pdb_id"]).upper()] = entry
        self._fetch_index = index
        return index


# --- file discovery -------------------------------------------------------
def _scan(cache_dir: Path | None) -> list[Path]:
    """Every coordinate file in the cache, in a stable order."""
    if cache_dir is None or not cache_dir.is_dir():
        return []
    found = [
        path
        for path in sorted(cache_dir.iterdir())
        if path.is_file() and any(path.name.lower().endswith(sfx) for sfx in STRUCTURE_SUFFIXES)
    ]
    return found


def _stem_id(path: Path) -> str:
    """The PDB id a filename claims, e.g. ``3I28.pdb.gz`` -> ``3I28``."""
    name = path.name
    for suffix in sorted(STRUCTURE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.removeprefix("pdb").upper()


def _read_bytes(path: Path) -> bytes:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def _is_mmcif(path: Path) -> bool:
    return ".cif" in path.name.lower() or path.name.lower().endswith(".mmcif")


def _matches(structure: ExperimentalStructure, target: str | None) -> bool:
    """Whether *target* names this entry, by any identifier the file carries."""
    if not target:
        return True
    needle = target.strip().upper()
    if not needle:
        return True
    if needle == structure.pdb_id.upper():
        return True
    if any(needle == acc.upper() for acc in structure.uniprot_accessions):
        return True
    if any(needle == gene.upper() for gene in structure.gene_names):
        return True
    return any(needle in name.upper() for name in structure.molecule_names)


def _ranking_key(structure: ExperimentalStructure) -> tuple[int, float, str]:
    """Ligand-bearing first, then resolution ascending, then the id for stability."""
    resolution = structure.resolution_a
    return (
        0 if structure.has_ligand else 1,
        resolution if resolution is not None else float("inf"),
        structure.pdb_id,
    )


# --- PDB format -----------------------------------------------------------
def _parse_pdb(text: str, path: Path) -> ExperimentalStructure:
    """Read a legacy-format PDB entry by its fixed columns.

    Column offsets follow the wwPDB v3.30 format specification and match the
    parser ``scripts/prepare_pocket.py`` already uses, so a pocket cut from a file
    and this tool's description of the same file cannot disagree about which atom
    is which.
    """
    structure = ExperimentalStructure(file_format="pdb")
    title_parts: list[str] = []
    expdta_parts: list[str] = []
    seqres: dict[str, list[str]] = {}
    het_atom_counts: dict[str, int] = {}
    hetnam: dict[str, list[str]] = {}
    formul: dict[str, list[str]] = {}
    compnd_parts: list[str] = []
    source_parts: list[str] = []
    atoms: list[dict[str, Any]] = []
    model_count = 0
    in_first_model = True

    for line in text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            model_count += 1
            in_first_model = model_count <= 1
            continue
        if record == "ENDMDL":
            in_first_model = False
            continue
        if record == "HEADER":
            structure.classification = line[10:50].strip() or None
            structure.deposited_on = line[50:59].strip() or None
            structure.pdb_id = line[62:66].strip().upper()
        elif record == "TITLE":
            title_parts.append(line[10:80].rstrip())
        elif record == "EXPDTA":
            expdta_parts.append(line[10:79].strip())
        elif record == "COMPND":
            compnd_parts.append(line[10:80].rstrip())
        elif record == "SOURCE":
            source_parts.append(line[10:80].rstrip())
        elif record == "REMARK":
            _read_pdb_remark(line, structure)
        elif record == "DBREF" and line[26:32].strip().upper() == "UNP":
            accession = line[33:41].strip().upper()
            if accession and accession not in structure.uniprot_accessions:
                structure.uniprot_accessions.append(accession)
        elif record == "SEQRES":
            seqres.setdefault(line[11:12].strip() or "_", []).extend(line[19:70].split())
        elif record == "HET":
            key = f"{line[7:10].strip()}|{line[12:13].strip()}|{line[13:18].strip()}"
            het_atom_counts[key] = _int_or(line[20:25], 0)
        elif record == "HETNAM":
            hetnam.setdefault(line[11:14].strip(), []).append(line[15:70].strip())
        elif record == "FORMUL":
            formul.setdefault(line[12:15].strip(), []).append(line[19:70].strip())
        elif record in ("ATOM", "HETATM") and in_first_model:
            parsed = _parse_pdb_atom(line)
            if parsed is not None:
                atoms.append(parsed)

    structure.title = _join_continuation(title_parts) or None
    structure.method = _join_continuation(expdta_parts).rstrip(";") or None
    structure.molecule_names = _pdb_spec_values(compnd_parts, "MOLECULE")
    structure.gene_names = [
        gene.strip().upper()
        for entry in _pdb_spec_values(source_parts, "GENE")
        for gene in entry.split(",")
        if gene.strip()
    ]
    structure.organisms = _pdb_spec_values(source_parts, "ORGANISM_SCIENTIFIC")
    structure.n_models = max(model_count, 1)
    if not atoms:
        raise ExperimentalStructureError(f"{path} carries no ATOM/HETATM records")
    structure.n_atoms = len(atoms)
    structure.chains = _chains_from_atoms(atoms, seqres)
    structure.het_groups = _het_groups_from_atoms(
        atoms,
        declared=het_atom_counts,
        names={k: " ".join(v).strip() for k, v in hetnam.items()},
        formulas={k: " ".join(v).strip() for k, v in formul.items()},
    )
    if model_count > 1:
        structure.qc_flags.append(f"first_of_{model_count}_models_parsed")
    return structure


def _read_pdb_remark(line: str, structure: ExperimentalStructure) -> None:
    """Pull resolution and refinement residuals out of the REMARK blocks."""
    number = line[7:10].strip()
    body = line[11:].strip()
    if number == "2" and body.upper().startswith("RESOLUTION."):
        # "RESOLUTION.    1.95 ANGSTROMS." — and "NOT APPLICABLE." for methods
        # that report none, which must stay None rather than becoming 0.0.
        token = body[len("RESOLUTION.") :].strip().split()
        if token:
            structure.resolution_a = _float_or_none(token[0])
    elif number == "3":
        upper = body.upper()
        if upper.startswith("R VALUE") and "WORKING SET" in upper and ":" in body:
            structure.r_work = _float_or_none(body.rsplit(":", 1)[1])
        elif upper.startswith("FREE R VALUE") and ":" in body and "TEST" not in upper:
            structure.r_free = _float_or_none(body.rsplit(":", 1)[1])


def _parse_pdb_atom(line: str) -> dict[str, Any] | None:
    """One fixed-column ATOM/HETATM record, or ``None`` if the columns do not parse."""
    try:
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        return None
    element = line[76:78].strip().upper() or line[12:16].strip()[:1].upper()
    return {
        "tag": line[:6].strip(),
        "res_name": line[17:20].strip(),
        "chain": line[21:22].strip() or "_",
        "res_seq": line[22:27].strip(),
        "xyz": (x, y, z),
        "element": element,
        "altloc": line[16:17].strip(),
    }


def _pdb_spec_values(lines: list[str], token: str) -> list[str]:
    """Values of one token in a PDB specification list (``COMPND`` / ``SOURCE``)."""
    joined = " ".join(part.strip() for part in lines)
    values: list[str] = []
    for chunk in joined.split(";"):
        key, _, value = chunk.partition(":")
        if key.strip().upper() == token and value.strip():
            values.append(value.strip())
    return values


def _join_continuation(parts: list[str]) -> str:
    """Fold a multi-line PDB text field into one string."""
    return " ".join(part.strip() for part in parts if part.strip()).strip()


# --- mmCIF format ---------------------------------------------------------
def _parse_mmcif(text: str, path: Path) -> ExperimentalStructure:
    """Read a PDBx/mmCIF entry through biopython's ``MMCIF2Dict``.

    The category names below are the mmCIF equivalents of the PDB records above —
    ``_exptl.method`` for ``EXPDTA``, ``_refine.ls_d_res_high`` for ``REMARK 2`` —
    so the same entry described in either format reports the same numbers.
    """
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as exc:  # pragma: no cover - guarded by runtime_modules
        raise ExperimentalStructureError(
            f"{path} is mmCIF and biopython is not importable here: {exc}; "
            "install it with `conda env update -f environment.yml`"
        ) from exc

    import io

    data = MMCIF2Dict(io.StringIO(text))
    structure = ExperimentalStructure(file_format="mmcif")
    structure.pdb_id = _first(data, "_entry.id").upper()
    structure.title = _first(data, "_struct.title") or None
    structure.classification = _first(data, "_struct_keywords.pdbx_keywords") or None
    structure.deposited_on = _first(data, "_pdbx_database_status.recvd_initial_deposition_date") or None
    structure.method = "; ".join(_values(data, "_exptl.method")) or None
    structure.resolution_a = _float_or_none(_first(data, "_refine.ls_d_res_high"))
    structure.r_work = _float_or_none(_first(data, "_refine.ls_R_factor_R_work"))
    structure.r_free = _float_or_none(_first(data, "_refine.ls_R_factor_R_free"))
    # Polymer entities only: `_entity.pdbx_description` also names every ligand and
    # cryoprotectant in the entry, and a target search must not match on "GLYCEROL".
    structure.molecule_names = [
        description
        for entity_type, description in zip(
            _values(data, "_entity.type"), _values(data, "_entity.pdbx_description"), strict=False
        )
        if description and entity_type.lower() == "polymer"
    ]
    structure.organisms = [v for v in _values(data, "_entity_src_gen.pdbx_gene_src_scientific_name") if v]
    structure.gene_names = [
        gene.strip().upper()
        for entry in _values(data, "_entity_src_gen.pdbx_gene_src_gene")
        for gene in entry.split(",")
        if gene.strip() and gene.strip() != "?"
    ]
    # `strict=False` throughout: an absent mmCIF category reads back as an empty
    # list rather than raising, so a file that simply does not declare `_struct_ref`
    # must yield no accessions, not a ValueError.
    for db, accession in zip(
        _values(data, "_struct_ref.db_name"),
        _values(data, "_struct_ref.pdbx_db_accession"),
        strict=False,
    ):
        if db.upper() == "UNP" and accession and accession.upper() not in structure.uniprot_accessions:
            structure.uniprot_accessions.append(accession.upper())

    names = dict(zip(_values(data, "_chem_comp.id"), _values(data, "_chem_comp.name"), strict=False))
    formulas = dict(zip(_values(data, "_chem_comp.id"), _values(data, "_chem_comp.formula"), strict=False))
    # `_entity_poly` declares the construct, one row per entity, with the chains it
    # covers in `pdbx_strand_id`. The canonical one-letter code is the mmCIF
    # equivalent of SEQRES, so both formats report the deposited construct rather
    # than only the residues that happened to be ordered.
    declared: dict[str, str] = {}
    for strands, sequence in zip(
        _values(data, "_entity_poly.pdbx_strand_id"),
        _values(data, "_entity_poly.pdbx_seq_one_letter_code_can"),
        strict=False,
    ):
        cleaned = "".join(ch for ch in sequence if not ch.isspace())
        for strand in str(strands).split(","):
            if strand.strip():
                declared[strand.strip()] = cleaned

    atoms, model_count = _mmcif_atoms(data)
    if not atoms:
        raise ExperimentalStructureError(f"{path} carries no _atom_site records")
    structure.n_atoms = len(atoms)
    structure.n_models = max(model_count, 1)
    structure.chains = [
        ChainInfo(
            chain_id=c.chain_id,
            kind=c.kind,
            n_residues=c.n_residues,
            n_atoms=c.n_atoms,
            seqres_length=len(declared[c.chain_id]) if c.chain_id in declared else None,
            sequence=declared.get(c.chain_id, c.sequence) if c.kind == "polypeptide" else "",
        )
        for c in _chains_from_atoms(atoms, {})
    ]
    structure.het_groups = _het_groups_from_atoms(atoms, declared={}, names=names, formulas=formulas)
    if model_count > 1:
        structure.qc_flags.append(f"first_of_{model_count}_models_parsed")
    return structure


def _mmcif_atoms(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """The first model's ``_atom_site`` rows, in the shape the PDB parser produces."""
    groups = _values(data, "_atom_site.group_PDB")
    if not groups:
        return [], 0
    comps = _values(data, "_atom_site.label_comp_id")
    chains = _values(data, "_atom_site.auth_asym_id") or _values(data, "_atom_site.label_asym_id")
    seqs = _values(data, "_atom_site.auth_seq_id") or _values(data, "_atom_site.label_seq_id")
    xs = _values(data, "_atom_site.Cartn_x")
    ys = _values(data, "_atom_site.Cartn_y")
    zs = _values(data, "_atom_site.Cartn_z")
    elements = _values(data, "_atom_site.type_symbol")
    models = _values(data, "_atom_site.pdbx_PDB_model_num")
    altlocs = _values(data, "_atom_site.label_alt_id")
    first_model = models[0] if models else ""
    model_count = len(dict.fromkeys(models)) if models else 1

    atoms: list[dict[str, Any]] = []
    for i, group in enumerate(groups):
        if models and models[i] != first_model:
            continue
        x, y, z = _float_or_none(xs[i]), _float_or_none(ys[i]), _float_or_none(zs[i])
        if x is None or y is None or z is None:
            continue
        atoms.append(
            {
                "tag": group.strip().upper(),
                "res_name": comps[i].strip().upper() if i < len(comps) else "",
                "chain": (chains[i].strip() if i < len(chains) else "") or "_",
                "res_seq": seqs[i].strip() if i < len(seqs) else "",
                "xyz": (x, y, z),
                "element": (elements[i].strip().upper() if i < len(elements) else ""),
                "altloc": (altlocs[i].strip() if i < len(altlocs) else "").replace(".", ""),
            }
        )
    return atoms, model_count


def _first(data: dict[str, Any], key: str) -> str:
    values = _values(data, key)
    return values[0] if values else ""


def _values(data: dict[str, Any], key: str) -> list[str]:
    """One mmCIF item as a list of strings, with ``?`` and ``.`` read as absent."""
    raw = data.get(key)
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    return ["" if str(v).strip() in ("?", ".") else str(v).strip() for v in items]


# --- shared coordinate analysis -------------------------------------------
def _chains_from_atoms(atoms: list[dict[str, Any]], seqres: dict[str, list[str]]) -> list[ChainInfo]:
    """Per-chain counts and sequence, from the coordinates that are really there."""
    order: list[str] = []
    residues: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for atom in atoms:
        chain = atom["chain"]
        if chain not in residues:
            residues[chain] = {}
            counts[chain] = 0
            order.append(chain)
        counts[chain] += 1
        if atom["tag"] == "ATOM" or atom["res_name"] in _THREE_TO_ONE:
            residues[chain].setdefault(atom["res_seq"], atom["res_name"])

    chains: list[ChainInfo] = []
    for chain in order:
        names = list(residues[chain].values())
        kind = _chain_kind(names)
        declared = seqres.get(chain)
        sequence = "".join(_THREE_TO_ONE.get(n, "X") for n in (declared or names))
        chains.append(
            ChainInfo(
                chain_id=chain,
                kind=kind,
                n_residues=len(names),
                n_atoms=counts[chain],
                seqres_length=len(declared) if declared else None,
                sequence=sequence if kind == "polypeptide" else "",
            )
        )
    return chains


def _chain_kind(residue_names: list[str]) -> str:
    """Label a chain from what its residues are, with no assumption from its id."""
    if not residue_names:
        return "other"
    amino = sum(1 for n in residue_names if n in _THREE_TO_ONE)
    nucleic = sum(1 for n in residue_names if n in _NUCLEIC)
    if amino >= max(1, len(residue_names) // 2):
        return "polypeptide"
    if nucleic >= max(1, len(residue_names) // 2):
        return "nucleic"
    return "other"


def _het_groups_from_atoms(
    atoms: list[dict[str, Any]],
    *,
    declared: dict[str, int],
    names: dict[str, str],
    formulas: dict[str, str],
) -> list[HetGroup]:
    """Group HETATM records into non-polymer entities and locate each one.

    The centroid is over *heavy* atoms because hydrogens are absent from most
    crystal structures, so including them where they exist would make the same
    ligand land in two different places depending on the entry.
    """
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for atom in atoms:
        if atom["tag"] != "HETATM":
            continue
        key = (atom["res_name"], atom["chain"], atom["res_seq"])
        buckets.setdefault(key, []).append(atom)

    groups: list[HetGroup] = []
    for comp_id, chain, res_seq in sorted(buckets):
        members = buckets[(comp_id, chain, res_seq)]
        heavy = [a for a in members if a["element"] not in ("H", "D")]
        centroid: tuple[float, float, float] | None = None
        if heavy:
            centroid = (
                sum(a["xyz"][0] for a in heavy) / len(heavy),
                sum(a["xyz"][1] for a in heavy) / len(heavy),
                sum(a["xyz"][2] for a in heavy) / len(heavy),
            )
        groups.append(
            HetGroup(
                comp_id=comp_id,
                chain=chain,
                res_seq=res_seq,
                n_atoms=declared.get(f"{comp_id}|{chain}|{res_seq}", len(members)),
                n_heavy_atoms=len(heavy),
                centroid=centroid,
                name=names.get(comp_id) or None,
                formula=formulas.get(comp_id) or None,
            )
        )
    return groups


def _float_or_none(text: str | None) -> float | None:
    """A float, or ``None`` for the many ways a PDB file says "not applicable"."""
    if text is None:
        return None
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int_or(text: str, default: int) -> int:
    try:
        return int(text.strip())
    except (TypeError, ValueError):
        return default


__all__ = [
    "CRYSTALLISATION_ADDITIVES",
    "EXPERIMENTAL_ADAPTER_VERSION",
    "FETCH_INDEX_NAME",
    "SPEC",
    "STRUCTURE_SUFFIXES",
    "ChainInfo",
    "ExperimentalStructure",
    "ExperimentalStructureAdapter",
    "ExperimentalStructureError",
    "HetGroup",
]
