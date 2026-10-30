"""Pocket detection: find the site, then hand a generator something it can use.

Until now this platform knew exactly one binding site — the hand-cut
``.external/pockets/seh_3i28_pocket10.pdb``, made by running
``scripts/prepare_pocket.py`` around the crystallographic ligand ``34N`` of sEH.
That file is why every pocket-conditioned generator in
:mod:`drug_agent.tools.generation` can run at all on this repository's own target,
and its existence is also the ceiling: a structure with no co-crystallised ligand
has no pocket, so it has no ``PocketSpec.pdb_path``, so PocketXMol, Pocket2Mol,
TargetDiff and PocketFlow all refuse it. Detection is what removes that ceiling.

**The output contract is the point of this package.** A detector that returns three
floats has not finished the job: :class:`~drug_agent.contracts.brief.PocketSpec` is
a *path* to a prepared pocket PDB, because a generator samples inside the box those
atoms imply and a whole-protein file gives it a meaningless one. So every adapter
here ends by cutting the site out of the source structure with
:func:`~drug_agent.tools.pockets._pdb.cut_pocket` — the same rule, radius and file
shape ``scripts/prepare_pocket.py`` uses — and returns a
:class:`DetectedPocket` whose :meth:`DetectedPocket.to_pocket_spec` is
immediately assignable to ``GenerationSpec.pocket``. The full path from a bare PDB
to a sampled molecule is then::

    detection = FpocketAdapter().detect(pdb, out_dir=Path("pockets/"))
    brief.generation.pocket = detection.pockets[0].to_pocket_spec()

The second consumer wants a box rather than a file — Vina and GNINA take
``--center_x/y/z`` and ``--size_x/y/z`` — and :meth:`DetectedPocket.box_size` is
that, derived from the site's own extent where the detector reported one.

**Scores from different detectors are not the same quantity, and are never
merged.** fpocket's ``Score`` is a rank-order statistic over alpha-sphere geometry;
its ``Druggability Score`` is a separate logistic model trained on a druggable-site
benchmark; P2Rank's ``score`` is a sum of per-SAS-point ligandability and its
``probability`` is that score pushed through a fitted transformer. Each rides on
:attr:`DetectedPocket.score` with its own :attr:`DetectedPocket.score_kind`, and
the engine's own numbers are carried verbatim in
:attr:`DetectedPocket.properties` rather than folded into anything. Comparing
``53.21`` from P2Rank with ``1.048`` from fpocket would be comparing two different
axes; the honest cross-check is geometric — do the two detectors put a pocket in
the same place — and that is what :func:`centroid_distance` is for.

**A detected pocket is a prediction, and the file says so.** The prepared PDB
carries ``REMARK`` lines naming the detector, its version, the rank and the score,
because once the file is on disk nothing else can distinguish "predicted by a
geometry heuristic" from "cut around a ligand somebody crystallised".
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...contracts.brief import PocketSpec
from ...data.snapshot import DataSnapshot
from .._ext.adapter import ExternalToolAdapter
from ._pdb import (
    Atom,
    PocketCut,
    Point,
    bounding_size,
    centroid,
    cut_pocket,
    distance,
    parse_atoms,
    write_pocket_pdb,
)

#: Filename the setup scripts write beside a fetched artifact, and the only place
#: a runtime path is allowed to learn where that artifact came from. Reading it is
#: a file read; there is no network call anywhere on the detection path.
PROVENANCE_FILENAME = "drugagent_provenance.json"

#: Default radius of the prepared pocket cut, in Angstrom. Ten is not a guess: it
#: is the CrossDocked ``pocket10`` convention the structure-conditioned generators
#: were trained on, it is ``scripts/prepare_pocket.py``'s default, and it is what
#: the reference ``seh_3i28_pocket10.pdb`` was cut at — so a detected pocket and
#: the known one are the same kind of object and can be differenced.
DEFAULT_POCKET_RADIUS = 10.0

#: How many ranked pockets get a prepared PDB written by default. Writing every
#: pocket of a 29-pocket fpocket run costs a megabyte of files nobody reads; the
#: top few are what a generator or a docking box would ever be pointed at.
DEFAULT_N_PREPARED = 3


class PocketError(RuntimeError):
    """The selected detector ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`: the backend was
    installed, it started, and it did not return a usable pocket. An input the
    detector cannot read, an empty result, an unparseable output table.
    """


@dataclass
class DetectedPocket:
    """One site a detector found, with a prepared pocket file when one was written.

    ``score`` is whatever the engine ranks by and ``score_kind`` names which
    quantity that is — they travel together because a bare float from a pocket
    detector is uninterpretable and, three artifacts downstream, indistinguishable
    from any other float.
    """

    rank: int
    score: float | None
    score_kind: str
    center: Point
    engine: str
    engine_version: str
    #: Points the site is made of — alpha-sphere centres, a predicted centroid.
    #: The pocket cut is taken around these, not around ``center`` alone.
    site_points: list[Point] = field(default_factory=list)
    #: The engine's own name for this site — ``pocket1`` from P2Rank, ``pocket.1``
    #: from a PRANK rescoring of fpocket. Kept verbatim so a result can be lined up
    #: with the run that produced it; ``rank`` is this package's ordering, not the
    #: engine's, and the two need not agree.
    label: str | None = None
    probability: float | None = None
    druggability: float | None = None
    volume_a3: float | None = None
    #: ``chain_resseq`` labels of the residues in the prepared cut. When no file was
    #: written and the engine reported a residue list of its own — P2Rank's
    #: ``residue_ids`` — that list is kept here instead, so a caller who asked for no
    #: files still learns which residues were implicated. The two are different
    #: selections, and ``pdb_path`` is what says which one this is.
    residues: tuple[str, ...] = ()
    n_residues: int = 0
    n_atoms: int = 0
    radius_a: float = DEFAULT_POCKET_RADIUS
    #: Path of the prepared pocket PDB, or ``None`` when the caller asked for no
    #: files. A pocket without one cannot condition a generator — see
    #: :meth:`to_pocket_spec`.
    pdb_path: str | None = None
    #: The engine's own numbers, verbatim and unconverted.
    properties: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def box_size(self, *, pad: float = 4.0) -> tuple[float, float, float]:
        """Docking-box dimensions in Angstrom, for an engine that wants a box.

        The other consumer of a detected pocket is a docking engine, and Vina and
        GNINA take ``--size_x/y/z`` rather than a file. A detector that reported an
        *extent* — fpocket's alpha spheres — gives a box that follows the cavity's
        shape; one that reported only a centre has nothing to derive an anisotropic
        box from, so the fallback is the cube implied by the pocket radius. Saying
        which of the two happened is the caller's business and is visible in
        ``site_points``.
        """
        if len(self.site_points) > 1:
            return bounding_size(self.site_points, pad=pad)
        side = 2.0 * self.radius_a
        return (side, side, side)

    def to_pocket_spec(
        self,
        *,
        receptor_id: str | None = None,
        pdb_id: str | None = None,
        radius: float | None = None,
    ) -> PocketSpec:
        """This pocket as the contract the generators consume.

        ``notes`` records the detector, the rank and the score, because
        ``PocketSpec`` is copied into a brief and then into a run manifest, and by
        that point this object is gone. A spec whose ``pdb_path`` is ``None`` — the
        caller detected without writing files — is still useful to a docking engine
        that only needs a box, and is refused by the pocket-conditioned generators,
        which is the correct outcome rather than a silent one.
        """
        score = "unscored" if self.score is None else f"{self.score:g}"
        return PocketSpec(
            receptor_id=receptor_id,
            pdb_id=pdb_id,
            pdb_path=self.pdb_path,
            center=[round(c, 3) for c in self.center],
            radius=float(self.radius_a if radius is None else radius),
            notes=(
                f"detected by {self.engine} {self.engine_version}, rank {self.rank}, "
                f"{self.score_kind}={score}; predicted site, not a ligand-derived pocket"
            ),
        )


@dataclass
class PocketDetection:
    """Everything one detection run produced, ranked best first."""

    structure_path: str
    engine: str
    engine_version: str
    pockets: list[DetectedPocket] = field(default_factory=list)
    runtime_seconds: float = 0.0
    #: Snapshot id of the model artifact the prediction used, when the backend has
    #: one. ``None`` for a purely geometric method, which has no model to pin.
    snapshot_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def top(self) -> DetectedPocket | None:
        return self.pockets[0] if self.pockets else None

    def metrics(self) -> dict[str, float]:
        """Quality numbers for the tool ledger. Facts about this run only."""
        out: dict[str, float] = {
            "n_pockets_detected": float(len(self.pockets)),
            "n_pockets_prepared": float(sum(1 for p in self.pockets if p.pdb_path)),
            "pocket_detect_runtime_seconds": round(self.runtime_seconds, 3),
        }
        best = self.top
        if best is not None:
            out["top_pocket_n_atoms"] = float(best.n_atoms)
            out["top_pocket_n_residues"] = float(best.n_residues)
            if best.score is not None:
                out["top_pocket_score"] = float(best.score)
            if best.probability is not None:
                out["top_pocket_probability"] = float(best.probability)
            if best.druggability is not None:
                out["top_pocket_druggability"] = float(best.druggability)
            if best.volume_a3 is not None:
                out["top_pocket_volume_a3"] = float(best.volume_a3)
        return out


def centroid_distance(pocket: DetectedPocket, reference_pdb: str | Path) -> float:
    """Angstrom between a detected pocket's centre and a known pocket's centroid.

    The only honest way to compare two detectors, and the only honest way to check
    one against a hand-prepared site: their scores are on different axes, but they
    are both claims about a *place*. Used against
    ``.external/pockets/seh_3i28_pocket10.pdb`` this answers "did it find the
    catalytic site" with a number instead of an assertion.
    """
    path = Path(reference_pdb)
    if not path.is_file():
        raise PocketError(f"reference pocket {path} does not exist")
    points = [atom.xyz for atom in parse_atoms(path.read_text()) if not atom.is_hydrogen]
    if not points:
        raise PocketError(f"reference pocket {path} contains no heavy atoms")
    return distance(pocket.center, centroid(points))


class PocketToolAdapter(ExternalToolAdapter):
    """Shared input handling, pocket cutting and provenance for the detectors.

    Three backends, one job: turn a structure into ranked sites and write the top
    ones out as prepared pocket PDBs. What differs between them is how the site is
    found — alpha-sphere geometry, a random forest over solvent-accessible surface
    points, an equivariant GNN — and everything after that is identical, so it lives
    here and is written once.
    """

    produces_nothing = "detects no pockets and prepares no pocket file"
    #: What :attr:`DetectedPocket.score` is for this engine. Never a bare "score".
    score_kind: str = "score"

    @contextlib.contextmanager
    def scratch(self, prefix: str) -> Iterator[Path]:
        """A temporary directory that is always removed, even on an abort."""
        workdir = Path(tempfile.mkdtemp(prefix=f"drugagent_{prefix}_"))
        try:
            yield workdir
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # --- input --------------------------------------------------------------
    def read_structure(self, structure_pdb: str | Path) -> tuple[Path, list[Atom]]:
        """Resolve and parse the input structure once, for the whole call.

        Both the detector subprocess and the pocket cut need this file, and parsing
        it twice would let them disagree about which model of an NMR ensemble the
        pocket came from.
        """
        path = Path(structure_pdb)
        if not path.is_file():
            raise PocketError(f"structure file {path} does not exist")
        atoms = parse_atoms(path.read_text())
        if not atoms:
            raise PocketError(f"{path} contains no ATOM/HETATM records — is it a PDB file?")
        return path, atoms

    def stage_structure(self, source: Path, workdir: Path) -> Path:
        """Copy the input into *workdir*, because fpocket writes beside its input.

        Measured, not assumed: ``fpocket -f /abs/path/5AI0.pdb`` run from an empty
        directory created ``5AI0_out/`` next to ``5AI0.pdb``, in ``.external/pdb/``,
        and nothing in the working directory. A detector must never leave output in
        the repository's structure store, so every backend here runs on a copy.
        """
        staged = workdir / source.name
        shutil.copyfile(source, staged)
        return staged

    # --- output -------------------------------------------------------------
    def prepare_pocket_file(
        self,
        atoms: list[Atom],
        *,
        site_points: list[Point],
        radius: float,
        out_dir: Path,
        stem: str,
        provenance: dict[str, Any],
    ) -> tuple[Path, PocketCut]:
        """Cut the site out of the structure and write it where a generator can read it.

        This is the step that turns a detection into something usable. The cut rule
        is :func:`~drug_agent.tools.pockets._pdb.cut_pocket`'s, i.e.
        ``scripts/prepare_pocket.py``'s, so the file is interchangeable with a
        hand-prepared one everywhere downstream.
        """
        cut = cut_pocket(atoms, site_points=site_points, radius=radius)
        if cut.n_atoms == 0:
            raise PocketError(
                f"the cut around {stem} selected no standard-residue atoms within "
                f"{radius:g} A — the detected centre is not on the protein"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}.pdb"
        path.write_text(write_pocket_pdb(cut, provenance=provenance))
        return path, cut

    # --- provenance ---------------------------------------------------------
    def artifact_snapshot(self) -> DataSnapshot | None:
        """The fetched artifact this backend runs on, as a content-hashed snapshot.

        Rule for every backend in this package: setup may use the network, the run
        may not. What makes the cached artifact *real data* rather than a stand-in
        is that its origin is recorded — the URL it came from, the release it is,
        the digest of the bytes, and when they were fetched. The setup script writes
        those facts beside the artifact; this reads them back and lets
        :meth:`~drug_agent.data.snapshot.DataSnapshot.data_card` hash them into a
        stable ``snapshot_id`` that a run can cite.

        ``None`` when the backend has no fetched artifact at all — fpocket computes
        geometry and has no model to pin — or when the setup script that would have
        written the record has not run.
        """
        root = self.ext.root
        if root is None:
            return None
        record = root / PROVENANCE_FILENAME
        if not record.is_file():
            return None
        try:
            payload = json.loads(record.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        files = payload.get("files")
        return DataSnapshot(
            name=str(payload.get("name") or self.spec.name.lower()),
            source=str(payload.get("source") or self.spec.upstream_url),
            kind=str(payload.get("kind") or "model_release"),
            records=list(files) if isinstance(files, list) else [],
            metadata={k: v for k, v in payload.items() if k not in ("files", "name", "source", "kind")},
            license=str(payload.get("license") or self.spec.license),
        )

    def snapshot_id(self) -> str | None:
        snapshot = self.artifact_snapshot()
        return snapshot.snapshot_id if snapshot is not None else None


__all__ = [
    "DEFAULT_N_PREPARED",
    "DEFAULT_POCKET_RADIUS",
    "DetectedPocket",
    "PROVENANCE_FILENAME",
    "PocketDetection",
    "PocketError",
    "PocketToolAdapter",
    "centroid_distance",
]
