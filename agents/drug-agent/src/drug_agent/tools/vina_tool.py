"""Structure-based docking with AutoDock Vina — the real search, out of process.

**This capability had no backend at all until repair R1.** What used to be here was
a deterministic pseudo-docking: an affinity built from molecular weight, logP and
hydrogen-bond counts plus a per-structure hash jitter, wrapped in a ``Pose`` with
``engine="mock"`` and an interaction fingerprint whose contact residues were bits
of ``hash(smiles + rank)``. Every number in it was a function of the ligand's own
descriptors; none of it knew the receptor existed. It was removed rather than kept
behind a flag, and this module then reported ``UNAVAILABLE`` and raised, for three
stated reasons. Two of them were about the deployment and one was about this file:

  1. *No receptor was ever resolved.* Fixed upstream of here. The structure agent
     assembles a :class:`~drug_agent.agents.specialists.structure_agent.Receptor`
     from the brief's own :class:`~drug_agent.contracts.brief.PocketSpec`, and
     Step 5 built a real ZAK target manifest with a prepared, hashed pocket at
     ``.external/research/zak-c9r/derived/``. There is a receptor to dock into.
  2. *Vina pins libboost 1.86 while the pinned rdkit needs 1.90,* so it cannot
     share this interpreter. It never needed to. ``vina`` is an executable, and
     this adapter runs it as a subprocess out of its own conda environment. The
     binary resolves its Boost through an ``RPATH`` of ``$ORIGIN/../lib``, so the
     two dependency sets never meet; nothing here imports the ``vina`` Python
     package.
  3. *The batch task budget.* Still real, still a configuration question, and
     still not this file's: ``default_task_timeout_s`` is 120 s for the whole
     candidate batch, and one flexible ligand at default exhaustiveness is ~11 s
     on this host. A caller that means to dock a full batch raises that budget;
     this adapter's own budget is ``DRUGAGENT_VINA_TIMEOUT_SECONDS``.

**Measured here on 2026-09-12, and this is what makes the capability AVAILABLE
rather than merely installed.** ``vina --version`` reports ``AutoDock Vina
f458505-mod`` (package ``vina 1.2.7`` in conda environment ``drug-agent-vina``).
``scripts/research/dock_zak_reference_ligand.py`` redocks the ZAK crystal ligand
C9R into PDB 6JUU — receptor = the ATOM records of
``.external/research/zak-c9r/rcsb/6JUU.pdb`` (2,331 atoms, no waters and no
ligand), box centred on the deposited C9R at (10.708, 3.405, 10.455) sized
15.0 × 12.3 × 24.2 Å, ``--seed 42``, ``--exhaustiveness 8``, ``--num_modes 9``.
It took **10.5 s** and recovered the pose: top mode **0.55 Å symmetry-corrected
in-place RMSD** from the crystal ligand, affinity **−14.385 kcal/mol**. In place
means no superposition — the engine was given the receptor and a box, never the
answer. That is a self-docking sanity check on the install, not a benchmark, and
it is not a calibration of anything.

**Where each half of the work runs, and why it is split.**

``ligand and receptor preparation``
    In *this* interpreter, with RDKit, Meeko and Open Babel — the packages this
    environment already pins. A ligand SMILES is embedded to a 3D conformer by
    the shared :class:`~drug_agent.tools.structure.StructureToolAdapter` and then
    typed and written as PDBQT by Meeko, which is what keeps bond orders
    recoverable on the way back. A receptor PDB is converted with ``obabel -xr``.
``the search``
    In the ``vina`` executable, as a subprocess, through the package's single
    approved binary path (:func:`drug_agent.security.sandbox.run_sandboxed`).
``reading the poses back``
    In this interpreter again: Meeko's :class:`~meeko.PDBQTMolecule` and
    :class:`~meeko.RDKitMolCreate` rebuild an RDKit molecule from the docked
    PDBQT *using the ligand's own topology*, so the returned pose is a real
    molecule and not a stripped coordinate list.

**What the score is.** ``REMARK VINA RESULT`` on each output model, in kcal/mol,
lower better — the Vina empirical scoring function, not a learned affinity. It is
on the same axis as GNINA's ``minimizedAffinity`` by ancestry and is still not
interchangeable with it: different terms, different weights, different ligand
preparation. Nothing here converts it into a potency, and
:class:`~drug_agent.contracts.enums.TrustLabel` stays ``CALCULATED`` because an
empirical function over coordinates is a computation, not a prediction.

**No weights, no checkout.** Vina is a C++ program with a closed-form scoring
function; there is no checkpoint to fetch and no source tree on the run path. A
missing binary is the only missing artifact there can be, and when it is missing
this capability is ``UNAVAILABLE``, :meth:`DockingAdapter.dock` raises
:class:`~drug_agent.tools.base.ToolUnavailable`, and the platform produces no
poses. There is still no surrogate.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import CostClass, Determinism
from ..contracts.structure import Pose
from ..observability.toolmeter import measure
from ..security.sandbox import run_sandboxed
from ._ext.spec import ExternalSpec
from .structure import StructureError, StructureToolAdapter

#: This adapter's contract with Vina, not Vina's own version.
VINA_ADAPTER_VERSION = "2.0.0"

#: The executable the whole capability rests on.
VINA_BINARY = "vina"

#: The environment Vina is installed into. It gets its own for the reason the
#: module docstring gives: its Boost pin is incompatible with the pinned rdkit,
#: and the platform's verified environments are not perturbed to accommodate it.
VINA_ENV = "drug-agent-vina"

#: Where an operator can name the binary when it is neither on PATH nor in the
#: environment above — a system package, a source build.
VINA_BINARY_ENV_VAR = "DRUGAGENT_VINA_BINARY"

INSTALL_HINT = f"conda create -y -n {VINA_ENV} -c conda-forge vina"

#: Scoring functions the binary accepts, from ``--scoring`` in its own ``--help``.
#: ``ad4`` is excluded on purpose: it needs precomputed affinity maps rather than
#: a receptor PDBQT, which is a different operation with different inputs.
SCORING_FUNCTIONS = frozenset({"vina", "vinardo"})

#: The per-model score line Vina writes into its output PDBQT. Anchored at the
#: line start so a REMARK inside a ligand's own header cannot be mistaken for it.
_VINA_RESULT = re.compile(
    r"^REMARK VINA RESULT:\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
    re.MULTILINE,
)

#: The property name the score is carried under on a parsed pose, chosen to read
#: the same way GNINA's ``minimizedAffinity`` does in a report.
SCORE_PROP = "vina_affinity"

#: Per-mode quantities Vina reports beside the affinity, carried onto the pose as
#: ``key=value`` qc flags rather than folded into the score. Both are RMSDs to the
#: *best* mode of the same run — they say how diverse the output is, and nothing
#: about agreement with any experimental pose.
CARRIED_PROPS: tuple[str, ...] = ("rmsd_lb", "rmsd_ub")

#: How many lines of a failed run's output go into the raised error.
_ERROR_TAIL_LINES = 20

#: Packages this interpreter needs to marshal a ligand and a receptor into PDBQT
#: and to read the docked poses back. They are probed rather than assumed: without
#: them the binary would still run and this adapter still could not use it.
_LOCAL_PREP_MODULES: tuple[str, ...] = ("meeko", "rdkit")

#: The Open Babel executable used to convert a receptor PDB into a PDBQT. Meeko's
#: polymer path needs residue chemistry templates that a *cut-out pocket* does not
#: satisfy, and a pocket cut is exactly what this platform docks into.
OBABEL_BINARY = "obabel"


def _conda_roots() -> list[Path]:
    """Conda installations to look in, from the same variables the probe uses.

    Mirrors :func:`drug_agent.tools.pockets.fpocket_tool._conda_roots`, which
    solves the identical problem for the identical reason: the process that
    launched the platform is not the environment this backend was installed into.
    """
    roots: list[Path] = []
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        active = Path(prefix)
        roots.append(active.parent.parent if active.parent.name == "envs" else active)
    conda_exe = os.environ.get("CONDA_EXE")  # <root>/bin/conda
    if conda_exe:
        roots.append(Path(conda_exe).parent.parent)
    return roots


def _configured_binary(settings: Any) -> str | None:
    """An explicitly-named vina, following the platform's configuration precedence.

    The same order :meth:`drug_agent.tools._ext.backend.ExternalBackend._resolve`
    owns for every other configurable value, restated here for the same reason
    ``fpocket_tool`` restates it: the binary must be resolved *before* the backend
    exists, because it is what the backend is constructed to probe for.
    """
    table = getattr(settings, "external_tools", None) if settings is not None else None
    if isinstance(table, dict):
        entry = table.get("vina")
        if isinstance(entry, dict) and entry.get("binary"):
            return str(entry["binary"])
    flat = getattr(settings, "vina_binary", None) if settings is not None else None
    if flat:
        return str(flat)
    return os.environ.get(VINA_BINARY_ENV_VAR) or None


def find_vina(settings: Any = None) -> str | None:
    """The vina executable to use, or ``None`` if there is not one.

    Configuration first, then ``PATH``, then ``envs/drug-agent-vina/bin`` under
    every conda installation this process can see. The last of those is the one
    that matters here: Vina is deliberately *not* on this interpreter's PATH,
    because putting it there would mean solving its Boost pin against the pinned
    rdkit, which is the dependency fight the subprocess exists to avoid.
    """
    configured = _configured_binary(settings)
    if configured:
        found = shutil.which(configured)
        if found:
            return found
    on_path = shutil.which(VINA_BINARY)
    if on_path:
        return on_path
    for root in _conda_roots():
        cand = root / "envs" / VINA_ENV / "bin" / VINA_BINARY
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def find_obabel(settings: Any = None) -> str | None:
    """The Open Babel executable used for receptor conversion, or ``None``.

    Looked for on PATH only. Unlike vina it is a dependency of *this* environment
    — ``openbabel-wheel`` ships it beside the interpreter — so a machine where it
    is absent has an incomplete primary environment rather than a missing external
    tool, and the reason says so.
    """
    table = getattr(settings, "external_tools", None) if settings is not None else None
    if isinstance(table, dict):
        entry = table.get("vina")
        if isinstance(entry, dict) and entry.get("obabel"):
            return shutil.which(str(entry["obabel"]))
    return shutil.which(OBABEL_BINARY)


SPEC = ExternalSpec(
    name="AutoDock Vina",
    env_prefix="VINA",
    # No checkout and no weights: Vina is a packaged binary with a closed-form
    # scoring function. Nothing in its source tree is on a code path here.
    checkout_relpath="",
    sentinel_files=(),
    # No driver script and no interpreter: the search is the executable. Nothing
    # in this package imports the `vina` Python module, which is what keeps its
    # libboost 1.86 pin away from the rdkit this environment needs.
    driver_name=None,
    selection_modules=(),
    runtime_modules=(),
    weight_relpaths=(),
    conda_envs=(VINA_ENV,),
    binary=VINA_BINARY,
    upstream_url="https://github.com/ccsb-scripps/AutoDock-Vina",
    upstream_ref="v1.2.7",
    license="AutoDock Vina Apache-2.0",
    install_hint=INSTALL_HINT,
    needs_gpu=False,
)


class DockingAdapter(StructureToolAdapter):
    """Monte Carlo docking into a defined box with the Vina scoring function."""

    capability_id = "structure.dock"
    spec = SPEC
    engine = "vina"
    version = VINA_ADAPTER_VERSION
    produces_nothing = "docks nothing and produces no poses"
    #: The Vina empirical score, not a learned affinity.
    score_unit = "kcal/mol"
    lower_is_better = True

    def __init__(
        self, backend: str = "auto", *, settings: Any = None, timeout_s: float | None = None
    ) -> None:
        """*backend* is accepted for the signature this adapter has always had.

        It selects almost nothing: there is one real backend, it is AutoDock
        Vina, and either the executable is resolvable or this capability is
        UNAVAILABLE. The one value that does something is ``"none"``, which is
        how ``settings.dock_backend`` disables docking; it turns the capability
        off explicitly and says so, rather than relying on the accident that this
        adapter used to have no backend at all. Anything else is refused rather
        than ignored, so a caller asking for something that does not exist finds
        out here.
        """
        wanted = (backend or "auto").strip().lower()
        if wanted not in {"auto", "vina", "none", ""}:
            raise ValueError(f"unknown docking backend {backend!r}; this adapter drives: auto, vina, none")
        self._disabled = wanted == "none"
        # Resolve before the base class probes, and per construction rather than at
        # import: `tool_pool()` builds a fresh adapter on every call, so an install
        # that finished after startup is noticed.
        self._binary = find_vina(settings)
        if self._binary is not None:
            self.spec = dataclasses.replace(SPEC, binary=self._binary)
        self._obabel = find_obabel(settings)
        # One flexible drug-like ligand at default exhaustiveness is ~11 s here.
        # 900 s is a ceiling for a large box or a high exhaustiveness, not a budget.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 900.0)
        self._missing = [*self._missing, *self._local_blockers()]
        # Recompute after the local probe: a card that names a backend while
        # reporting UNAVAILABLE claims something it cannot do.
        self.backend = self.engine if not self._missing else "none"
        self._version: str | None = None

    def _local_blockers(self) -> list[str]:
        """Reasons *this* interpreter could not marshal inputs even with the binary.

        The search is out of process; the chemistry is not. A machine with vina and
        no Meeko can start the executable and cannot write it a ligand, and that is
        an UNAVAILABLE with a different fix, so it is reported separately from a
        missing binary rather than being discovered at dock time.
        """
        import sys

        from ._ext.probe import missing_modules

        problems: list[str] = []
        if self._disabled:
            problems.append(
                "docking is switched off by configuration (dock_backend='none'); this is a "
                "deliberate refusal, not a missing install"
            )
        if absent := missing_modules(sys.executable, _LOCAL_PREP_MODULES):
            problems.append(
                f"this interpreter cannot import {', '.join(absent)}, which is what writes "
                f"the PDBQT inputs and reads the docked poses back; `pip install meeko`"
            )
        if self._obabel is None:
            problems.append(
                f"no {OBABEL_BINARY!r} executable on PATH, which is what converts a receptor "
                f"PDB into the PDBQT vina requires; `pip install openbabel-wheel`"
            )
        return problems

    @property
    def unavailable_reason(self) -> str | None:
        """The whole reason, including the blockers that are local to this process.

        The base class reads :class:`~drug_agent.tools._ext.backend.ExternalBackend`'s
        own list, which knows about the binary and nothing else. Half of what this
        adapter needs — Meeko, RDKit, Open Babel — lives in *this* interpreter, and
        a reason that omitted them would send an operator to reinstall a binary
        that is already there. Overriding here keeps the card, the exception and
        this string the same sentence, which is what ``test_pools`` asserts.
        """
        if not self._missing:
            return None
        detail = "; ".join(self._missing)
        return (
            f"{self.spec.name} is not usable here, so this capability is UNAVAILABLE and "
            f"{self.produces_nothing} — there is no surrogate backend. "
            f"Install it with `{INSTALL_HINT}`. Reason: {detail}."
        )

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Molecular docking (AutoDock Vina)",
            description=(
                "Monte Carlo conformational search inside a defined box, scored with the "
                "Vina empirical function (kcal/mol, lower better). The search runs as a "
                "subprocess out of its own conda environment; ligand and receptor PDBQT "
                "preparation and pose read-back use RDKit, Meeko and Open Babel here."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="Pose",
            # Monte Carlo over random restarts. --seed and --cpu are both exposed
            # because reproducibility needs both: parallel chains are the other
            # source of run-to-run variation.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no search box: vina has no default and center/size are required",
                "receptor file absent, or not a format Open Babel can type as a receptor",
                "ligand SMILES that will not embed a 3D conformer (raised per molecule)",
                "the ligand starts outside the box, so vina exits with no output models",
                "a box so large the grid exhausts memory before the search starts",
                "the search exceeds the task budget and the run is terminated with no poses",
            ],
            validation_checks=[
                "score is the REMARK VINA RESULT affinity in kcal/mol, never a predicted potency",
                "rmsd_lb / rmsd_ub are carried verbatim and are RMSDs to this run's own best mode",
                "rank is re-derived from the affinity, ascending",
                "engine is 'vina' on every pose returned",
            ],
        )

    # --- operations --------------------------------------------------------
    def dock(
        self,
        candidate_id: str,
        smiles: str | None = None,
        receptor_id: str | None = None,
        n_poses: int = 9,
        *,
        receptor_pdb: str | Path | None = None,
        receptor_pdbqt: str | Path | None = None,
        ligand_sdf: str | Path | None = None,
        center: tuple[float, float, float] | None = None,
        size: tuple[float, float, float] | None = None,
        exhaustiveness: int = 8,
        seed: int = 0,
        scoring: str = "vina",
        cpu: int | None = None,
        min_rmsd: float | None = None,
        energy_range: float | None = None,
    ) -> list[Pose]:
        """Dock one ligand into a receptor and return its poses, best first.

        The first four parameters keep the positional signature this capability
        has always published, so an existing caller is unchanged. Everything the
        search actually needs — a receptor and a box — is keyword-only, because
        neither has a defensible default: docking the whole protein would be a
        different experiment reported as this one.
        """
        self.require_available()
        with measure(self.capability_id, "dock", backend=self.backend) as metered:
            poses = self._dock(
                candidate_id=candidate_id,
                smiles=smiles,
                receptor_id=receptor_id,
                n_poses=n_poses,
                receptor_pdb=receptor_pdb,
                receptor_pdbqt=receptor_pdbqt,
                ligand_sdf=ligand_sdf,
                center=center,
                size=size,
                exhaustiveness=exhaustiveness,
                seed=seed,
                scoring=scoring,
                cpu=cpu,
                min_rmsd=min_rmsd,
                energy_range=energy_range,
            )
            for key, value in self._pose_metrics(poses).items():
                metered[key] = value
        return poses

    def score_pose(
        self,
        *,
        candidate_id: str,
        receptor_pdb: str | Path | None = None,
        receptor_pdbqt: str | Path | None = None,
        pose_sdf: str | Path,
        receptor_id: str | None = None,
        scoring: str = "vina",
    ) -> Pose:
        """Score coordinates that already exist, without searching for new ones.

        Vina's ``--score_only``: no box, no Monte Carlo, no output file. The
        returned pose carries the input coordinates and the affinity Vina printed
        for them, so a pose from any producer in this package can be put on Vina's
        scale without Vina having produced it.
        """
        self.require_available()
        with measure(self.capability_id, "score_pose", backend=self.backend) as metered:
            pose = self._score_pose(
                candidate_id=candidate_id,
                receptor_pdb=receptor_pdb,
                receptor_pdbqt=receptor_pdbqt,
                pose_sdf=pose_sdf,
                receptor_id=receptor_id,
                scoring=scoring,
            )
            for key, value in self._pose_metrics([pose]).items():
                metered[key] = value
        return pose

    # --- real backend ------------------------------------------------------
    def _dock(
        self,
        *,
        candidate_id: str,
        smiles: str | None,
        receptor_id: str | None,
        n_poses: int,
        receptor_pdb: str | Path | None,
        receptor_pdbqt: str | Path | None,
        ligand_sdf: str | Path | None,
        center: tuple[float, float, float] | None,
        size: tuple[float, float, float] | None,
        exhaustiveness: int,
        seed: int,
        scoring: str,
        cpu: int | None,
        min_rmsd: float | None,
        energy_range: float | None,
    ) -> list[Pose]:
        if n_poses < 1:
            raise StructureError("vina requires num_modes >= 1")
        if exhaustiveness < 1:
            raise StructureError("vina requires exhaustiveness >= 1")
        if scoring not in SCORING_FUNCTIONS:
            raise StructureError(
                f"unknown scoring function {scoring!r}; this adapter drives "
                f"{', '.join(sorted(SCORING_FUNCTIONS))} (ad4 needs precomputed maps)"
            )
        box = self._box_args(center=center, size=size)

        with self.scratch("vina") as workdir:
            receptor = self._stage_receptor(receptor_pdb, receptor_pdbqt, workdir)
            ligand_mol, ligand = self._stage_ligand(
                smiles=smiles, ligand_sdf=ligand_sdf, workdir=workdir, seed=seed
            )
            out_pdbqt = workdir / "docked.pdbqt"
            argv = [
                "--receptor",
                str(receptor),
                "--ligand",
                str(ligand),
                *box,
                "--out",
                str(out_pdbqt),
                "--scoring",
                scoring,
                "--num_modes",
                str(int(n_poses)),
                "--exhaustiveness",
                str(int(exhaustiveness)),
                "--seed",
                str(int(seed)),
            ]
            if cpu is not None:
                if cpu < 1:
                    raise StructureError("cpu must be >= 1")
                argv += ["--cpu", str(int(cpu))]
            if min_rmsd is not None:
                argv += ["--min_rmsd", f"{float(min_rmsd):g}"]
            if energy_range is not None:
                argv += ["--energy_range", f"{float(energy_range):g}"]

            self._run(argv, workdir)
            if not out_pdbqt.is_file():
                raise StructureError(
                    f"vina exited cleanly but wrote no poses to {out_pdbqt.name}; "
                    f"check that the ligand starts inside the search box"
                )
            return self._poses_from_pdbqt(
                out_pdbqt,
                candidate_id=candidate_id,
                receptor_id=receptor_id,
                template=ligand_mol,
            )

    def _score_pose(
        self,
        *,
        candidate_id: str,
        receptor_pdb: str | Path | None,
        receptor_pdbqt: str | Path | None,
        pose_sdf: str | Path,
        receptor_id: str | None,
        scoring: str,
    ) -> Pose:
        from ..contracts.enums import TrustLabel

        if scoring not in SCORING_FUNCTIONS:
            raise StructureError(
                f"unknown scoring function {scoring!r}; this adapter drives "
                f"{', '.join(sorted(SCORING_FUNCTIONS))}"
            )
        pose_path = self._require_file(pose_sdf, "pose")
        with self.scratch("vina_score") as workdir:
            receptor = self._stage_receptor(receptor_pdb, receptor_pdbqt, workdir)
            ligand = workdir / "pose.pdbqt"
            mol = self._read_sdf(pose_path)
            ligand.write_text(self._ligand_pdbqt(mol))
            proc = self._run(
                [
                    "--receptor",
                    str(receptor),
                    "--ligand",
                    str(ligand),
                    "--scoring",
                    scoring,
                    "--score_only",
                ],
                workdir,
            )
        affinity = self._parse_score_only(proc.stdout or "")
        if affinity is None:
            raise StructureError(
                "vina --score_only printed no 'Estimated Free Energy of Binding' line:\n"
                + self._tail(proc.stdout or "")
            )
        return Pose(
            candidate_id=candidate_id,
            receptor_id=receptor_id,
            rank=0,
            score=affinity,
            score_unit=self.score_unit,
            pose_ref=pose_path.read_text(),
            engine=self.engine,
            engine_version=self._engine_version(),
            trust_label=TrustLabel.CALCULATED,
            qc_flags=["scored_only", f"scoring={scoring}"],
        )

    # --- argv ---------------------------------------------------------------
    @staticmethod
    def _box_args(
        *, center: tuple[float, float, float] | None, size: tuple[float, float, float] | None
    ) -> list[str]:
        """An explicit box. Vina has no default and no autobox for a search."""
        if center is None or size is None:
            raise StructureError(
                "vina needs a search box: pass both center=(x, y, z) and size=(x, y, z) "
                "in Angstrom. There is no default box — docking the whole protein would "
                "be a different experiment reported as this one."
            )
        cx, cy, cz = (float(v) for v in center)
        sx, sy, sz = (float(v) for v in size)
        if min(sx, sy, sz) <= 0.0:
            raise StructureError(f"search box size must be positive in every dimension, got {size!r}")
        return [
            "--center_x",
            f"{cx:g}",
            "--center_y",
            f"{cy:g}",
            "--center_z",
            f"{cz:g}",
            "--size_x",
            f"{sx:g}",
            "--size_y",
            f"{sy:g}",
            "--size_z",
            f"{sz:g}",
        ]

    # --- input marshalling ---------------------------------------------------
    def _stage_receptor(
        self, receptor_pdb: str | Path | None, receptor_pdbqt: str | Path | None, workdir: Path
    ) -> Path:
        """A receptor PDBQT in the scratch directory, converted if necessary."""
        if receptor_pdbqt is not None:
            return self._require_file(receptor_pdbqt, "receptor PDBQT")
        if receptor_pdb is None:
            raise StructureError("dock() needs either receptor_pdb or receptor_pdbqt")
        source = self._require_file(receptor_pdb, "receptor")
        if source.suffix.lower() == ".pdbqt":
            return source
        target = workdir / "receptor.pdbqt"
        self._obabel_convert(source, target)
        return target

    def _obabel_convert(self, source: Path, target: Path) -> None:
        """``obabel <pdb> -O <pdbqt> -xr`` — rigid receptor, Gasteiger charges.

        ``-xr`` is what makes it a *receptor*: without it Open Babel writes a
        flexible ligand PDBQT with a torsion tree, which vina rejects as a
        receptor. Warnings on stderr are expected and ignored — a crystallographic
        model routinely has aromatic perception Open Babel complains about — so the
        check is whether a PDBQT with ATOM records came out, not whether the
        converter was silent.
        """
        if self._obabel is None:  # pragma: no cover - require_available() has refused
            raise StructureError(f"no {OBABEL_BINARY!r} executable on PATH")
        try:
            proc = run_sandboxed(
                self._obabel,
                [str(source), "-O", str(target), "-xr"],
                cwd=target.parent,
                timeout=min(300.0, self.timeout_s),
                env=self.ext.child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise StructureError(f"obabel did not convert {source.name} within its budget") from exc
        except OSError as exc:
            raise StructureError(f"could not launch {self._obabel}: {exc}") from exc
        if not target.is_file() or "ATOM" not in target.read_text():
            detail = self._tail(proc.stderr or "") or self._tail(proc.stdout or "")
            raise StructureError(
                f"obabel produced no receptor PDBQT from {source.name} (exit {proc.returncode}):\n{detail}"
            )

    def _stage_ligand(
        self, *, smiles: str | None, ligand_sdf: str | Path | None, workdir: Path, seed: int
    ) -> tuple[Any, Path]:
        """A 3D ligand as both an RDKit molecule and a PDBQT in the scratch dir.

        The molecule is kept because it is the *template* the docked poses are read
        back against: a PDBQT has atom types and no bond orders, and rebuilding a
        molecule without the original topology would return an aromatic ring as a
        set of single bonds.
        """
        if ligand_sdf is not None:
            mol = self._read_sdf(self._require_file(ligand_sdf, "ligand"))
        elif smiles:
            sdf = workdir / "ligand.sdf"
            # Raises StructureError for chemistry that will not embed — a per-molecule
            # failure the caller records against the candidate, not against vina.
            self.write_ligand(smiles, sdf, seed=seed or 0xF00D)
            mol = self._read_sdf(sdf)
        else:
            raise StructureError("dock() needs either smiles or ligand_sdf")
        pdbqt = workdir / "ligand.pdbqt"
        pdbqt.write_text(self._ligand_pdbqt(mol))
        return mol, pdbqt

    @staticmethod
    def _read_sdf(path: Path) -> Any:
        from rdkit import Chem

        mol = Chem.MolFromMolFile(str(path), removeHs=False)
        if mol is None:
            raise StructureError(f"could not read a molecule from {path}")
        return mol

    @staticmethod
    def _ligand_pdbqt(mol: Any) -> str:
        """Type and write one 3D ligand as PDBQT with Meeko."""
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        try:
            setups = MoleculePreparation().prepare(mol)
        except Exception as exc:  # meeko raises a family of its own errors
            raise StructureError(f"meeko could not prepare the ligand: {type(exc).__name__}: {exc}") from exc
        if not setups:
            raise StructureError("meeko produced no molecule setup for the ligand")
        text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise StructureError(f"meeko could not write a ligand PDBQT: {err}")
        return str(text)

    # --- output parsing -------------------------------------------------------
    def _poses_from_pdbqt(
        self, path: Path, *, candidate_id: str, receptor_id: str | None, template: Any
    ) -> list[Pose]:
        """Read vina's output PDBQT into ranked :class:`Pose` objects.

        Meeko rebuilds the molecule from its own PDBQT annotations, so the poses
        come back with the ligand's real bond orders. The scores are read from the
        file's own ``REMARK VINA RESULT`` lines rather than from the rebuilt
        molecule, because that is where vina writes them and a conformer carries no
        properties of its own.
        """
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit import Chem

        from ..contracts.enums import TrustLabel

        results = [
            (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            for m in _VINA_RESULT.finditer(path.read_text())
        ]
        if not results:
            raise StructureError(f"vina's output {path.name} carries no REMARK VINA RESULT lines")
        try:
            pdbqt = PDBQTMolecule.from_file(str(path), skip_typing=True)
            rebuilt = RDKitMolCreate.from_pdbqt_mol(pdbqt)
        except Exception as exc:
            raise StructureError(
                f"could not rebuild molecules from vina's output: {type(exc).__name__}: {exc}"
            ) from exc
        mols = [m for m in rebuilt if m is not None]
        if not mols:
            raise StructureError("vina wrote poses that meeko could not rebuild into a molecule")
        mol = mols[0]
        n = min(mol.GetNumConformers(), len(results))
        if n == 0:
            raise StructureError("vina's output rebuilt into a molecule with no conformers")

        poses: list[Pose] = []
        for idx in range(n):
            affinity, rmsd_lb, rmsd_ub = results[idx]
            single = Chem.Mol(mol)
            single.RemoveAllConformers()
            single.AddConformer(mol.GetConformer(idx), assignId=True)
            poses.append(
                Pose(
                    candidate_id=candidate_id,
                    receptor_id=receptor_id,
                    rank=idx,
                    score=affinity,
                    score_unit=self.score_unit,
                    pose_ref=Chem.MolToMolBlock(single),
                    engine=self.engine,
                    engine_version=self._engine_version(),
                    trust_label=TrustLabel.CALCULATED,
                    qc_flags=[f"rmsd_lb={rmsd_lb:g}", f"rmsd_ub={rmsd_ub:g}"],
                )
            )
        if len(results) > n:
            # Recorded, never silently dropped: a mismatch between the score lines
            # and the conformers meeko rebuilt is a parsing fault, and pretending
            # the missing poses never existed is how it would stay invisible.
            poses[0].qc_flags = [
                *poses[0].qc_flags,
                f"unparsed_models={len(results) - n}",
            ]
        # `template` is held for the caller's topology contract; the rebuilt
        # molecule already carries bond orders from meeko's own annotations.
        del template
        return self.rank(poses)

    @staticmethod
    def _parse_score_only(text: str) -> float | None:
        """The affinity ``--score_only`` printed, or ``None`` if it printed none."""
        match = re.search(r"^Estimated Free Energy of Binding\s*:\s*(-?\d+(?:\.\d+)?)", text, re.MULTILINE)
        if match is None:
            match = re.search(r"^Affinity:\s*(-?\d+(?:\.\d+)?)", text, re.MULTILINE)
        if match is None:
            return None
        try:
            return float(match.group(1))
        except ValueError:  # pragma: no cover - the pattern already matched a number
            return None

    # --- process ------------------------------------------------------------
    def _run(self, argv: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
        """Run vina through the package's sandboxed-binary path."""
        binary = self._binary or shutil.which(VINA_BINARY)
        if binary is None:  # pragma: no cover - require_available() has already refused
            raise StructureError(f"no {VINA_BINARY!r} executable resolvable; {INSTALL_HINT}")
        try:
            proc = run_sandboxed(binary, argv, cwd=workdir, timeout=self.timeout_s, env=self.ext.child_env())
        except subprocess.TimeoutExpired as exc:
            raise StructureError(
                f"vina exceeded its {self.timeout_s:.0f}s budget and was terminated. "
                f"Lower exhaustiveness or the box size, or raise DRUGAGENT_VINA_TIMEOUT_SECONDS."
            ) from exc
        except OSError as exc:
            raise StructureError(f"could not launch {binary}: {exc}") from exc
        if proc.returncode != 0:
            detail = self._tail(proc.stderr or "") or self._tail(proc.stdout or "")
            raise StructureError(f"vina exited {proc.returncode}:\n{detail}")
        return proc

    def _engine_version(self) -> str:
        """Vina's own version string, asked for once per adapter.

        The exact wording is Vina's business, so the first non-empty line is
        recorded verbatim rather than parsed. A binary that will not answer is
        still a binary that may dock, so this degrades to ``"unknown"`` instead of
        failing the operation.
        """
        if self._version is not None:
            return self._version
        self._version = "unknown"
        binary = self._binary or shutil.which(VINA_BINARY)
        if binary is None:  # pragma: no cover - require_available() has already refused
            return self._version
        try:
            proc = run_sandboxed(binary, ["--version"], timeout=60.0, env=self.ext.child_env())
        except (subprocess.SubprocessError, OSError, ValueError):
            return self._version
        for line in (proc.stdout or "").splitlines():
            if line.strip():
                self._version = line.strip()
                break
        return self._version

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _require_file(path: str | Path, what: str) -> Path:
        resolved = Path(path)
        if not resolved.is_file():
            raise StructureError(f"{what} file {resolved} does not exist")
        return resolved

    @staticmethod
    def _pose_metrics(poses: list[Pose]) -> dict[str, float]:
        """What the ledger keeps: how many poses, and the best affinity.

        A collapse in pose count, or an affinity that stops moving, is how a wrong
        receptor, an empty box or a broken ligand writer announces itself.
        """
        out: dict[str, float] = {"vina_n_poses": float(len(poses))}
        scored = [p.score for p in poses if p.score is not None]
        if scored:
            out["vina_best_affinity_kcal_mol"] = float(min(scored))
        return out

    @staticmethod
    def _tail(text: str) -> str:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return "\n".join(lines[-_ERROR_TAIL_LINES:])


__all__ = [
    "CARRIED_PROPS",
    "INSTALL_HINT",
    "SCORE_PROP",
    "SCORING_FUNCTIONS",
    "SPEC",
    "VINA_ADAPTER_VERSION",
    "VINA_BINARY",
    "VINA_ENV",
    "DockingAdapter",
    "find_obabel",
    "find_vina",
]
