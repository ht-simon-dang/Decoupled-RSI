"""Structure-based tools: put a ligand in a pocket, or judge one that is already there.

Five upstream projects live here and they divide cleanly in two.

**Producers** — GNINA, DiffDock-L, FlowDock and Boltz-2 take a receptor and a
ligand and return coordinates: a docked pose, a predicted complex. Their output is
a :class:`~drug_agent.contracts.structure.Pose`, and the thing they disagree about
is what the score on it *means*. A GNINA CNN affinity, a DiffDock confidence and a
Boltz-2 affinity head are three different numbers on three different scales, so
each adapter records its own ``score_unit`` and its own ``engine`` and none of them
is silently comparable with another. A docking score is not an experimental
affinity, and this package never lets one become one.

**A judge** — PoseBusters takes a pose from any of them and asks whether it is
physically possible at all: bond lengths, angles, ring flatness, internal clashes,
and whether the ligand is even inside the pocket it claims. It produces no
coordinates and no score. It is here rather than in a validation package because a
pose and its verdict travel together, and because the one thing every producer
needs is somebody who does not believe it.

:class:`StructureToolAdapter` owns what all five share: marshalling a ligand and a
receptor into a scratch directory the way an external process expects to find them,
and turning the SDF that comes back into ranked :class:`Pose` objects with the real
engine name stamped on them. ``Pose.engine`` defaults to the literal string
``"mock"`` in the contract, so an adapter that forgets to set it ships an object
that claims to be fake — the base class sets it from ``self.engine``, and a test
asserts it is never ``"mock"``.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from ...contracts.enums import TrustLabel
from ...contracts.structure import Pose
from .._ext.adapter import ExternalToolAdapter

RDLogger.DisableLog("rdApp.*")


class StructureError(RuntimeError):
    """The selected structure backend ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`: the engine was
    installed, it started, and it did not return a usable pose.
    """


class StructureToolAdapter(ExternalToolAdapter):
    """Shared input marshalling and pose parsing for the structure backends."""

    #: What the score on a returned pose actually is. Overridden per engine —
    #: these are not the same quantity and must never be compared across tools.
    score_unit: str = "kcal/mol"
    #: Lower score is better (docking energies). False for confidence-like scores.
    lower_is_better: bool = True
    #: How much a pose from this engine is worth epistemically.
    #:
    #: ``CALCULATED`` is for an empirical scoring function — GNINA's
    #: ``minimizedAffinity`` is a Vina-lineage energy, a deterministic computation
    #: over coordinates. ``PREDICTED`` is for anything a network produced:
    #: DiffDock's confidence classifier, FlowDock's plDDT head, Boltz-2's affinity
    #: head. Those are not the same kind of number, and this is the one field a
    #: reader consults to tell them apart, so the shared parser must not decide it
    #: for every engine. It defaults to ``CALCULATED`` because the docking case is
    #: the physics case; every ML-scored engine overrides it.
    trust_label: TrustLabel = TrustLabel.CALCULATED

    @contextlib.contextmanager
    def scratch(self, prefix: str) -> Iterator[Path]:
        """A temporary directory that is always removed, even on a CUDA abort."""
        workdir = Path(tempfile.mkdtemp(prefix=f"drugagent_{prefix}_"))
        try:
            yield workdir
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def write_ligand(self, smiles: str, path: Path, *, seed: int = 0xF00D) -> None:
        """Embed *smiles* into a 3D SDF the external tool can read.

        Docking engines want a starting conformer, not a graph. Embedding fails for
        a real fraction of generated chemistry — strained cages, allenes, valence
        errors that survived parsing — and that is a per-molecule failure, not a
        tool failure, so it raises :class:`StructureError` for the caller to record
        against the candidate rather than against the engine.
        """
        from rdkit.Chem import AllChem

        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            raise StructureError(f"unparseable SMILES: {smiles!r}")
        rdmol = Chem.AddHs(rdmol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(rdmol, params) != 0:
            raise StructureError(f"could not embed a 3D conformer for {smiles!r}")
        AllChem.MMFFOptimizeMolecule(rdmol)
        Chem.MolToMolFile(rdmol, str(path))

    def poses_from_sdf(
        self,
        path: Path,
        *,
        candidate_id: str,
        receptor_id: str | None,
        score_prop: str,
        engine_version: str,
        extra_props: tuple[str, ...] = (),
    ) -> list[Pose]:
        """Read an engine's output SDF into ranked :class:`Pose` objects.

        *score_prop* is the SDF property the engine writes its score into. A
        molecule with no such property still becomes a pose, with ``score=None``
        and a ``qc_flag`` — the coordinates are real even when the score is
        missing, and dropping the pose would hide that the engine produced one.
        """
        if not path.is_file():
            return []
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        poses: list[Pose] = []
        for idx, mol in enumerate(supplier):
            if mol is None:
                continue
            flags: list[str] = []
            score: float | None = None
            if mol.HasProp(score_prop):
                try:
                    score = float(mol.GetProp(score_prop))
                except ValueError:
                    flags.append(f"unparseable_{score_prop}")
            else:
                flags.append(f"missing_{score_prop}")
            block = Chem.MolToMolBlock(mol, kekulize=False)
            poses.append(
                Pose(
                    candidate_id=candidate_id,
                    receptor_id=receptor_id,
                    rank=idx,
                    score=score,
                    score_unit=self.score_unit,
                    pose_ref=block,
                    engine=self.engine,
                    engine_version=engine_version,
                    trust_label=self.trust_label,
                    qc_flags=flags + self._extra_flags(mol, extra_props),
                )
            )
        return self.rank(poses)

    @staticmethod
    def _extra_flags(mol: Any, props: tuple[str, ...]) -> list[str]:
        """Carry named engine properties through as ``key=value`` flags."""
        out: list[str] = []
        for prop in props:
            if mol.HasProp(prop):
                out.append(f"{prop}={mol.GetProp(prop)}")
        return out

    def rank(self, poses: list[Pose]) -> list[Pose]:
        """Sort by score in this engine's direction and renumber ``rank`` from 0.

        Poses with no score keep their engine order and sort last: an unscored pose
        is not a bad pose, but it cannot be claimed to be a good one.
        """
        scored = [p for p in poses if p.score is not None]
        unscored = [p for p in poses if p.score is None]
        scored.sort(key=lambda p: p.score or 0.0, reverse=not self.lower_is_better)
        ordered = scored + unscored
        for rank, pose in enumerate(ordered):
            pose.rank = rank
        return ordered


__all__ = ["StructureError", "StructureToolAdapter"]
