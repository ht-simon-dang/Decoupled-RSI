"""GNINA — Monte Carlo docking with a 3D convolutional network rescoring every pose.

GNINA is a fork of smina, itself a fork of AutoDock Vina. The search is Vina's
Monte Carlo; what GNINA adds is a 3D CNN over a voxelised protein-ligand grid that
rescores — and, at higher ``--cnn_scoring`` levels, refines — the poses the search
produces. It writes two different quantities onto every pose and they are not
interchangeable. The empirical Vina-lineage energy is the SDF tag
``minimizedAffinity``, in kcal/mol, lower better. The network's outputs are
``CNNscore`` (a pose-quality probability, higher better) and ``CNNaffinity`` (a
predicted pK, higher better). This adapter ranks by ``minimizedAffinity`` and
carries the CNN values through untouched, because a pK and a kcal/mol are not the
same axis and combining them would produce a number with no unit.

**This capability is AVAILABLE on this machine, and the route to it is worth
reading.** GNINA is a C++/CUDA program: it is built or downloaded, never imported,
so there is no environment to install it into and no checkpoint to fetch. The
aarch64 host this adapter was first written on could not run it at all. This host
is x86-64 and can, but not by simply putting the release asset on PATH.

  1. *The released binary is the right architecture here.* Release ``v1.3.3`` of
     ``gnina/gnina`` publishes exactly one asset, ``gnina.cuda12.8.static``
     (2,056,131,000 bytes). Its ELF header is ``7f 45 4c 46 02 01 01 03`` with
     ``e_machine`` = 62, EM_X86_64. This host reports ``x86_64`` from ``uname -m``,
     so the kernel loads it. On the aarch64 host, whose native executables carry
     ``e_machine`` = 183, the same asset produced ``cannot execute binary file:
     Exec format error`` — the kernel refusing a foreign-architecture ELF, which no
     shared library, driver or environment variable could have changed.

  2. *"static" is a misnomer, and this is the part that cost the time.* The asset
     is statically linked against its **C++** dependencies only. It still resolves
     the CUDA 12 runtime and cuDNN 9 at load time — ``libcudart.so.12``,
     ``libcublas.so.12``, ``libcublasLt.so.12``, ``libcufft.so.11``,
     ``libcusolver.so.11`` and ``libcudnn.so.9``. The ``drug-agent`` environment
     ships CUDA 13, where none of those sonames exists, so the binary dies in the
     dynamic loader with ``error while loading shared libraries`` before it reaches
     a single CUDA call. The fix is a CUDA-12 lib set beside the binary (the pip
     ``nvidia-*-cu12`` wheels unpacked into ``.external/gnina-bin/cuda12libs/``) and
     a wrapper script that prepends them to ``LD_LIBRARY_PATH`` and execs the real
     executable. ``shutil.which("gnina")`` finds that **wrapper**, which is why
     :func:`_elf_machine` tolerates a non-ELF match: a shell script on PATH is the
     supported arrangement, not a broken install.

  3. *No source build was needed, and its prerequisites are still absent.*
     ``.external/gnina`` holds the full source at ``6fe1ce2bb9c3`` (tag ``v1.3.3``),
     and its ``CMakeLists.txt`` requires CMake ≥ 3.25, ``find_package(CUDA 12.0
     REQUIRED)``, ``find_package(OpenBabel3 3.1.0...<4 REQUIRED)``,
     ``find_package(Protobuf REQUIRED)``, ``find_package(ZLIB REQUIRED)`` and
     libmolgrid. Those are recorded here because they are what a host without a
     runnable release asset — any aarch64 machine — has to satisfy instead.

  4. *The build's default CUDA architecture list would not name this GPU either.*
     For CUDA ≥ 13 ``CMakeLists.txt`` defaults ``TORCH_CUDA_ARCH_LIST`` to
     ``7.5;8.0;8.6;8.9;9.0;9.0+PTX``. ``nvidia-smi --query-gpu=name,compute_cap``
     answers ``NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 12.0``, which is
     not in that list, so a source build here would need
     ``-DTORCH_CUDA_ARCH_LIST`` and ``-DCMAKE_CUDA_ARCHITECTURES`` set explicitly.
     The downloaded asset sidesteps this: it carries its own kernels.

**Measured here on 2026-09-09.** ``gnina --version`` reports ``gnina v1.3.3
master:6fe1ce2 Built Jun 30 2026``. Redocking the sEH crystal ligand 34N into PDB
3I28 (receptor = the ATOM records only, ``--autobox_ligand`` the crystal pose,
``--autobox_add 4``, ``--seed 42``, ``--num_modes 9``, ``--exhaustiveness 8``) took
8.4 s wall and recovered the pose: **top-ranked mode at 1.20 Å symmetry-corrected
in-place RMSD from the crystal ligand, centroid 0.39 Å**, ``minimizedAffinity``
-12.35 kcal/mol, ``CNNscore`` 0.954, ``CNNaffinity`` 7.83. That is a self-docking
result on the structure the pose came from and is a sanity check on the install, not
a benchmark.

If the binary is ever removed, this capability reports ``UNAVAILABLE``, ``dock()``
and ``score_pose()`` raise :class:`~drug_agent.tools.base.ToolUnavailable`, and the
platform produces no poses at all. Under :mod:`drug_agent.tools.base` there is no
surrogate: a fabricated docking score is a float three artifacts downstream, and
nothing there could tell it from a measurement.

**What the adapter computes when the binary is there.** The argv it builds and the
outputs it reads were taken from the pinned checkout, not from memory.

    ``dock()``           ``--receptor``/``--ligand``, a search box from either
                         ``--autobox_ligand`` (plus ``--autobox_add``) or explicit
                         ``--center_x/y/z --size_x/y/z``, then ``--cnn_scoring``,
                         ``--num_modes``, ``--exhaustiveness``, ``--seed`` and
                         ``--out``, and parses the output SDF into ranked
                         :class:`~drug_agent.contracts.structure.Pose` objects.
    ``score_pose()``     ``--score_only`` over coordinates that already exist. It
                         needs no search box and, per ``output_produced =
                         !settings.score_only`` in ``gninasrc/main/main.cpp``,
                         writes no output file — the numbers are printed, and are
                         read back from the process output.

The SDF tags come from ``result_info::write`` in ``gninasrc/lib/result_info.cpp``,
which emits ``minimizedAffinity`` unconditionally and then ``minimizedRMSD``,
``CNNscore``, ``CNNaffinity``, ``CNN_VS`` (the product of the two CNN numbers) and
``CNNaffinity_variance`` when those are set:

    ``score``            ``minimizedAffinity``, kcal/mol, lower better
    ``CNNscore``         CNN pose score, dimensionless, higher better
    ``CNNaffinity``      CNN predicted pK, higher better
    ``CNN_VS``           ``CNNaffinity`` × ``CNNscore``, the virtual-screening term
    ``minimizedRMSD``    Å from the input conformer, when GNINA reports it

Everything but the score rides on the pose as ``key=value`` qc flags, which is the
domain base's existing mechanism for carrying an engine's own properties without
inventing a parallel result model.

**What it deliberately does not do.** It does not convert ``CNNaffinity`` into a
predicted potency, does not merge the CNN and empirical numbers, and does not
build a :class:`~drug_agent.contracts.prediction.Prediction` — the specialist
agents do that. It also does not adopt GNINA's own output ordering: GNINA sorts by
``CNNscore`` by default (``--pose_sort_order``, and it forces ``Energy`` when
``--cnn_scoring none``), while ``Pose.rank`` here is re-derived from
``minimizedAffinity`` by :class:`StructureToolAdapter`, so rank always means the
same thing across every docking backend in this package.

**No weights to fetch.** Unlike the generative models in this catalogue, GNINA
carries its networks inside the executable: ``gninasrc/make_model_cpp.py`` embeds
each ``.pt`` as ``_binary_lib_models_<name>_pt_start``/``_end`` linker symbols.
``gninasrc/lib/cnn_data.cpp`` names ``crossdock_default2018`` as the default
model, and ``gninasrc/lib/cnn_torch_scorer.cpp`` expands ``--cnn fast`` to
``dense`` plus ``crossdock_default2018``. So ``weight_relpaths`` is empty on
purpose: a missing binary is the only missing artifact there can be.
"""

from __future__ import annotations

import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism, TrustLabel
from ...contracts.structure import Pose
from ...observability.toolmeter import measure
from ...security.sandbox import run_sandboxed
from .._ext.spec import ExternalSpec
from . import StructureError, StructureToolAdapter

#: This adapter's contract with GNINA, not GNINA's own version.
GNINA_ADAPTER_VERSION = "1.0.0"

#: The executable the whole capability rests on.
GNINA_BINARY = "gnina"

#: The one asset on the upstream release page, and what its ELF header says.
#: Verified against https://github.com/gnina/gnina/releases/download/v1.3.3/ with a
#: 64-byte HTTP range request: magic 7f454c46, e_machine 62 (EM_X86_64).
_RELEASE_TAG = "v1.3.3"
_RELEASE_ASSET = "gnina.cuda12.8.static"
_RELEASE_ASSET_BYTES = 2_056_131_000

#: ELF ``e_machine`` values, keyed by what ``platform.machine()`` calls the host.
_ELF_MACHINE_BY_UNAME: dict[str, int] = {
    "x86_64": 62,
    "amd64": 62,
    "aarch64": 183,
    "arm64": 183,
}

#: Names for the ``e_machine`` values this probe can meet, for the error message.
_ELF_MACHINE_NAMES: dict[int, str] = {
    3: "x86 (32-bit)",
    40: "ARM (32-bit)",
    62: "x86-64",
    183: "aarch64",
    243: "riscv64",
}

#: SDF tags ``result_info::write`` attaches beside the score. Carried onto every
#: pose as ``key=value`` qc flags rather than folded into the score, because a CNN
#: pK and an empirical kcal/mol are different quantities on different scales.
CARRIED_SDF_PROPS: tuple[str, ...] = (
    "CNNscore",
    "CNNaffinity",
    "CNN_VS",
    "CNNaffinity_variance",
    "minimizedRMSD",
)

#: The SDF tag GNINA always writes, and the one this adapter ranks on.
SCORE_PROP = "minimizedAffinity"

#: Accepted ``--cnn_scoring`` levels, from ``operator>>(istream&,
#: cnn_scoring_level&)`` in ``gninasrc/lib/user_opts.cpp``.
CNN_SCORING_LEVELS = frozenset({"none", "rescore", "refinement", "metrorescore", "metrorefine", "all"})

#: Accepted ``--pose_sort_order`` tokens, from the same file.
POSE_SORT_ORDERS = frozenset({"CNNscore", "CNNaffinity", "Energy"})

#: What ``--score_only`` prints, from the ``log <<`` statements in
#: ``do_search`` (``gninasrc/main/main.cpp``). Anchored at line starts so the
#: banner and any warnings on the same stream cannot be mistaken for results.
_SCORE_ONLY_PATTERNS: dict[str, re.Pattern[str]] = {
    "minimizedAffinity": re.compile(r"^Affinity:\s*(-?\d+(?:\.\d+)?)", re.MULTILINE),
    "CNNscore": re.compile(r"^CNNscore:\s*(-?\d+(?:\.\d+)?)", re.MULTILINE),
    "CNNaffinity": re.compile(r"^CNNaffinity:\s*(-?\d+(?:\.\d+)?)", re.MULTILINE),
    "CNNvariance": re.compile(r"^CNNvariance:\s*(-?\d+(?:\.\d+)?)", re.MULTILINE),
}

#: How many lines of a failed run's output go into the raised error.
_ERROR_TAIL_LINES = 20


def _elf_machine(path: Path) -> int | None:
    """The ``e_machine`` field of *path*, or ``None`` if it is not an ELF file.

    Twenty bytes off the front of the file. This is the cheapest honest answer to
    "would the kernel load this", and it is the check the ordinary ``which`` probe
    cannot make: a binary is on PATH or it is not, but an x86-64 binary on an
    aarch64 PATH is present *and* unrunnable. Returning ``None`` for a non-ELF file
    is deliberate and load-bearing on this host: what is on PATH here is
    ``gnina-wrapper.sh``, a shell script that adds a CUDA-12 lib set before exec-ing
    the real binary, and treating a wrapper as a foreign architecture would refuse a
    working install.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(20)
    except OSError:
        return None
    if len(head) < 20 or head[:4] != b"\x7fELF":
        return None  # a wrapper script, or something that is not an executable at all
    endian = "<" if head[5] == 1 else ">"
    try:
        return int(struct.unpack_from(f"{endian}H", head, 18)[0])
    except struct.error:  # pragma: no cover - unpacking 2 bytes of a 20-byte buffer
        return None


def _platform_blockers() -> tuple[str, ...]:
    """Blockers that are true of this machine whatever else is installed.

    Evaluated once, at import, because it is a statement about the host and the
    upstream release rather than about a configuration that can change mid-process.
    It is deliberately *not* unconditional: a GNINA built from source for this
    architecture retracts the blocker, because at that point the released binary's
    architecture has stopped being the thing standing in the way.
    """
    host = platform.machine()
    host_elf = _ELF_MACHINE_BY_UNAME.get(host)
    found = shutil.which(GNINA_BINARY)

    if found is not None:
        machine = _elf_machine(Path(found))
        if machine is None or host_elf is None or machine == host_elf:
            return ()
        name = _ELF_MACHINE_NAMES.get(machine, f"ELF e_machine {machine}")
        return (
            f"{found} is a {name} executable on a {host} host, so the kernel refuses to "
            f"load it (`cannot execute binary file: Exec format error`). Build GNINA from "
            f"source for this architecture, or remove the foreign binary from PATH.",
        )

    if host_elf == _ELF_MACHINE_BY_UNAME["x86_64"]:
        # The released asset matches this host; a plain "not on PATH" is the whole story.
        # Note that downloading it is not sufficient on its own: it is "static" only in
        # its C++ dependencies and still needs the CUDA 12 runtime and cuDNN 9. That is
        # an install problem, not a platform blocker, so it belongs in install_hint.
        return ()

    return (
        f"GNINA publishes one binary asset, {_RELEASE_ASSET} ({_RELEASE_ASSET_BYTES:,} bytes) "
        f"on release {_RELEASE_TAG}, and its ELF header reports e_machine 62 (EM_X86_64). "
        f"This host is {host}, so that asset cannot be executed here at all. GNINA is a "
        f"C++/CUDA program, so the fix is a source build for this architecture — CUDA, CMake "
        f"≥ 3.25, OpenBabel 3.1.x with headers, libmolgrid, libtorch, Boost, Eigen, glog, "
        f"protobuf and jsoncpp — not an install of a package.",
    )


SPEC = ExternalSpec(
    name="GNINA",
    env_prefix="GNINA",
    checkout_relpath=".external/gnina",
    # The source of truth for the argv this adapter builds and the tags it parses.
    # A checkout missing these is not a checkout this adapter was written against.
    sentinel_files=(
        "CMakeLists.txt",
        "gninasrc/main/main.cpp",
        "gninasrc/lib/result_info.cpp",
    ),
    # No driver and no runtime modules: GNINA is an executable, not a Python
    # package, so there is no interpreter to select and nothing to import.
    driver_name=None,
    selection_modules=(),
    runtime_modules=(),
    # The CNN weights are linked into the executable as _binary_lib_models_*
    # symbols by gninasrc/make_model_cpp.py, so there is no checkpoint to fetch.
    weight_relpaths=(),
    conda_envs=(),
    binary=GNINA_BINARY,
    upstream_url="https://github.com/gnina/gnina",
    upstream_ref="6fe1ce2bb9c3",
    license="GNINA dual GPL-2.0 / Apache-2.0 (GPL required by its OpenBabel dependency)",
    install_hint=(
        "on x86-64: download gnina.cuda12.8.static from the v1.3.3 release, unpack the "
        "nvidia-*-cu12 wheels beside it (libcudart.so.12, libcublas.so.12, libcufft.so.11, "
        "libcusolver.so.11, libcudnn.so.9 — the CUDA-13 environment has none of them), and "
        "put a wrapper that prepends them to LD_LIBRARY_PATH on PATH as `gnina`. On any "
        "other architecture: build from source, see .external/gnina/README.md"
    ),
    needs_gpu=True,
    platform_blockers=_platform_blockers(),
)


class GninaAdapter(StructureToolAdapter):
    """Docking and CNN rescoring against a prepared receptor."""

    capability_id = "structure.dock.gnina"
    spec = SPEC
    engine = "gnina"
    version = GNINA_ADAPTER_VERSION
    produces_nothing = "docks nothing and produces no poses"
    #: ``minimizedAffinity`` is the empirical Vina-lineage energy, not a CNN number.
    score_unit = "kcal/mol"
    lower_is_better = True

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # Docking one flexible ligand at the default exhaustiveness is a minutes-scale
        # job, so the default budget is generous rather than the base class's.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 1800.0)
        self._version: str | None = None

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Molecular docking with CNN rescoring (GNINA)",
            description=(
                "Vina-lineage Monte Carlo docking into a defined search box, with a 3D "
                "convolutional network rescoring every pose. Reports the empirical energy "
                "(minimizedAffinity, kcal/mol) as the score and carries CNNscore and "
                "CNNaffinity through as separate, differently-scaled quantities."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="Pose",
            # Monte Carlo search over random restarts. --seed and --cpu are exposed
            # because reproducibility needs both: parallel chains are the other
            # source of run-to-run variation.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "no search box: neither autobox_ligand nor an explicit center and size",
                "receptor file absent, unreadable, or not a format OpenBabel accepts",
                "ligand SMILES that will not embed a 3D conformer (raised per molecule)",
                "the ligand starts outside the box, so every pose is clipped or rejected",
                "CUDA out of memory during CNN scoring; --cnn_scoring none or --no_gpu is the fallback",
                "the search exceeds the task budget and the run is terminated with no output SDF",
            ],
            validation_checks=[
                "score is minimizedAffinity in kcal/mol, never a CNN quantity",
                "CNN values are carried verbatim and never folded into the score",
                "rank is re-derived from minimizedAffinity, ascending",
                "engine is 'gnina' on every pose returned",
            ],
        )

    # --- operations --------------------------------------------------------
    def dock(
        self,
        *,
        candidate_id: str,
        receptor_pdb: str | Path,
        smiles: str | None = None,
        ligand_sdf: str | Path | None = None,
        receptor_id: str | None = None,
        autobox_ligand: str | Path | None = None,
        center: tuple[float, float, float] | None = None,
        size: tuple[float, float, float] | None = None,
        autobox_add: float | None = None,
        n_poses: int = 9,
        exhaustiveness: int = 8,
        seed: int = 0,
        cnn_scoring: str = "rescore",
        cnn_model: tuple[str, ...] = (),
        pose_sort_order: str = "CNNscore",
        min_rmsd_filter: float | None = None,
        cpu: int | None = None,
        use_gpu: bool = True,
        device: int = 0,
    ) -> list[Pose]:
        """Dock one ligand into *receptor_pdb* and return its poses, best first.

        The ligand is either *smiles*, embedded to a 3D conformer here, or an
        *ligand_sdf* that already has coordinates. The search box is either
        *autobox_ligand* — a reference ligand file GNINA takes the box from — or
        *center* and *size* in Angstrom. One of the two is required; GNINA has no
        default box.
        """
        self.require_available()
        with measure(self.capability_id, "dock", backend=self.backend) as metered:
            poses = self._dock(
                candidate_id=candidate_id,
                receptor_pdb=receptor_pdb,
                smiles=smiles,
                ligand_sdf=ligand_sdf,
                receptor_id=receptor_id,
                autobox_ligand=autobox_ligand,
                center=center,
                size=size,
                autobox_add=autobox_add,
                n_poses=n_poses,
                exhaustiveness=exhaustiveness,
                seed=seed,
                cnn_scoring=cnn_scoring,
                cnn_model=cnn_model,
                pose_sort_order=pose_sort_order,
                min_rmsd_filter=min_rmsd_filter,
                cpu=cpu,
                use_gpu=use_gpu,
                device=device,
            )
            for key, value in self._pose_metrics(poses).items():
                metered[key] = value
        return poses

    def score_pose(
        self,
        *,
        candidate_id: str,
        receptor_pdb: str | Path,
        pose_sdf: str | Path,
        receptor_id: str | None = None,
        cnn_scoring: str = "rescore",
        cnn_model: tuple[str, ...] = (),
        use_gpu: bool = True,
        device: int = 0,
    ) -> Pose:
        """Score coordinates that already exist, without searching for new ones.

        This is GNINA's ``--score_only``: no box, no Monte Carlo, no output file.
        The returned pose carries the input coordinates and the numbers GNINA
        printed for them, so a pose from any producer in this package can be put on
        GNINA's scale without GNINA having produced it.
        """
        self.require_available()
        with measure(self.capability_id, "score_pose", backend=self.backend) as metered:
            pose = self._score_pose(
                candidate_id=candidate_id,
                receptor_pdb=receptor_pdb,
                pose_sdf=pose_sdf,
                receptor_id=receptor_id,
                cnn_scoring=cnn_scoring,
                cnn_model=cnn_model,
                use_gpu=use_gpu,
                device=device,
            )
            for key, value in self._pose_metrics([pose]).items():
                metered[key] = value
        return pose

    # --- real backend ------------------------------------------------------
    def _dock(
        self,
        *,
        candidate_id: str,
        receptor_pdb: str | Path,
        smiles: str | None,
        ligand_sdf: str | Path | None,
        receptor_id: str | None,
        autobox_ligand: str | Path | None,
        center: tuple[float, float, float] | None,
        size: tuple[float, float, float] | None,
        autobox_add: float | None,
        n_poses: int,
        exhaustiveness: int,
        seed: int,
        cnn_scoring: str,
        cnn_model: tuple[str, ...],
        pose_sort_order: str,
        min_rmsd_filter: float | None,
        cpu: int | None,
        use_gpu: bool,
        device: int,
    ) -> list[Pose]:
        receptor = self._require_file(receptor_pdb, "receptor")
        if n_poses < 1:
            raise StructureError("gnina requires num_modes >= 1")
        if exhaustiveness < 1:
            raise StructureError("gnina requires exhaustiveness >= 1")
        if pose_sort_order not in POSE_SORT_ORDERS:
            raise StructureError(
                f"unknown pose_sort_order {pose_sort_order!r}; gnina accepts "
                f"{', '.join(sorted(POSE_SORT_ORDERS))}"
            )
        box = self._box_args(autobox_ligand=autobox_ligand, center=center, size=size, autobox_add=autobox_add)

        with self.scratch("gnina") as workdir:
            ligand = self._stage_ligand(smiles=smiles, ligand_sdf=ligand_sdf, workdir=workdir, seed=seed)
            out_sdf = workdir / "docked.sdf"
            argv = [
                "--receptor",
                str(receptor),
                "--ligand",
                str(ligand),
                *box,
                "--out",
                str(out_sdf),
                "--num_modes",
                str(int(n_poses)),
                "--exhaustiveness",
                str(int(exhaustiveness)),
                "--seed",
                str(int(seed)),
                "--pose_sort_order",
                pose_sort_order,
                *self._cnn_args(cnn_scoring=cnn_scoring, cnn_model=cnn_model),
                *self._device_args(use_gpu=use_gpu, device=device, cpu=cpu),
            ]
            if min_rmsd_filter is not None:
                argv += ["--min_rmsd_filter", f"{float(min_rmsd_filter):g}"]

            self._run(argv, workdir)
            if not out_sdf.is_file():
                raise StructureError(
                    f"gnina exited cleanly but wrote no poses to {out_sdf.name}; "
                    f"check that the ligand starts inside the search box"
                )
            return self.poses_from_sdf(
                out_sdf,
                candidate_id=candidate_id,
                receptor_id=receptor_id,
                score_prop=SCORE_PROP,
                engine_version=self._engine_version(),
                extra_props=CARRIED_SDF_PROPS,
            )

    def _score_pose(
        self,
        *,
        candidate_id: str,
        receptor_pdb: str | Path,
        pose_sdf: str | Path,
        receptor_id: str | None,
        cnn_scoring: str,
        cnn_model: tuple[str, ...],
        use_gpu: bool,
        device: int,
    ) -> Pose:
        receptor = self._require_file(receptor_pdb, "receptor")
        pose_path = self._require_file(pose_sdf, "pose")

        with self.scratch("gnina_score") as workdir:
            argv = [
                "--receptor",
                str(receptor),
                "--ligand",
                str(pose_path),
                "--score_only",
                *self._cnn_args(cnn_scoring=cnn_scoring, cnn_model=cnn_model),
                *self._device_args(use_gpu=use_gpu, device=device, cpu=None),
            ]
            proc = self._run(argv, workdir)

        values = self._parse_score_only(proc.stdout or "")
        if SCORE_PROP not in values:
            raise StructureError(
                "gnina --score_only printed no 'Affinity:' line:\n" + self._tail(proc.stdout or "")
            )
        flags = [f"{key}={values[key]:g}" for key in CARRIED_SDF_PROPS if key in values]
        if "CNNvariance" in values:
            flags.append(f"CNNvariance={values['CNNvariance']:g}")
        return Pose(
            candidate_id=candidate_id,
            receptor_id=receptor_id,
            rank=0,
            score=values[SCORE_PROP],
            score_unit=self.score_unit,
            pose_ref=pose_path.read_text(),
            engine=self.engine,
            engine_version=self._engine_version(),
            trust_label=TrustLabel.CALCULATED,
            qc_flags=["scored_only", *flags],
        )

    # --- argv ---------------------------------------------------------------
    @staticmethod
    def _box_args(
        *,
        autobox_ligand: str | Path | None,
        center: tuple[float, float, float] | None,
        size: tuple[float, float, float] | None,
        autobox_add: float | None,
    ) -> list[str]:
        """``--autobox_ligand`` or an explicit box. GNINA has no default."""
        if autobox_ligand is not None:
            path = Path(autobox_ligand)
            if not path.is_file():
                raise StructureError(f"autobox_ligand {path} does not exist")
            args = ["--autobox_ligand", str(path)]
            if autobox_add is not None:
                args += ["--autobox_add", f"{float(autobox_add):g}"]
            return args
        if center is None or size is None:
            raise StructureError(
                "gnina needs a search box: pass autobox_ligand (a reference ligand file "
                "it takes the box from) or both center=(x, y, z) and size=(x, y, z) in Angstrom"
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

    @staticmethod
    def _cnn_args(*, cnn_scoring: str, cnn_model: tuple[str, ...]) -> list[str]:
        """``--cnn_scoring`` and any explicit built-in model names."""
        if cnn_scoring not in CNN_SCORING_LEVELS:
            raise StructureError(
                f"unknown cnn_scoring {cnn_scoring!r}; gnina accepts {', '.join(sorted(CNN_SCORING_LEVELS))}"
            )
        args = ["--cnn_scoring", cnn_scoring]
        if cnn_model:
            # --cnn is multitoken: several built-in names select an ensemble.
            args += ["--cnn", *cnn_model]
        return args

    @staticmethod
    def _device_args(*, use_gpu: bool, device: int, cpu: int | None) -> list[str]:
        args: list[str] = []
        if use_gpu:
            args += ["--device", str(int(device))]
        else:
            args.append("--no_gpu")
        if cpu is not None:
            if cpu < 1:
                raise StructureError("cpu must be >= 1")
            args += ["--cpu", str(int(cpu))]
        return args

    # --- process ------------------------------------------------------------
    def _run(self, argv: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
        """Run GNINA through the package's sandboxed-binary path."""
        binary = shutil.which(GNINA_BINARY)
        if binary is None:  # pragma: no cover - require_available() has already refused
            raise StructureError(f"no {GNINA_BINARY!r} executable on PATH")
        try:
            proc = run_sandboxed(
                binary,
                argv,
                cwd=workdir,
                timeout=self.timeout_s,
                env=self.ext.child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise StructureError(
                f"gnina exceeded its {self.timeout_s:.0f}s budget and was terminated. "
                f"Lower exhaustiveness or num_modes, or raise DRUGAGENT_GNINA_TIMEOUT_SECONDS."
            ) from exc
        except OSError as exc:
            raise StructureError(f"could not launch {binary}: {exc}") from exc
        if proc.returncode != 0:
            detail = self._tail(proc.stderr or "") or self._tail(proc.stdout or "")
            raise StructureError(f"gnina exited {proc.returncode}:\n{detail}")
        return proc

    def _engine_version(self) -> str:
        """GNINA's own version string, asked for once per adapter.

        The exact wording of ``--version`` is GNINA's business, so the first
        non-empty line is recorded verbatim rather than parsed. A binary that will
        not answer is still a binary that may dock, so this degrades to
        ``"unknown"`` instead of failing the operation.
        """
        if self._version is not None:
            return self._version
        self._version = "unknown"
        binary = shutil.which(GNINA_BINARY)
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

    # --- parsing ------------------------------------------------------------
    @staticmethod
    def _parse_score_only(text: str) -> dict[str, float]:
        """The numbers ``--score_only`` printed, keyed by their SDF-tag names."""
        values: dict[str, float] = {}
        for key, pattern in _SCORE_ONLY_PATTERNS.items():
            match = pattern.search(text)
            if match is None:
                continue
            try:
                values[key] = float(match.group(1))
            except ValueError:  # pragma: no cover - the pattern already matched a number
                continue
        return values

    @staticmethod
    def _require_file(path: str | Path, what: str) -> Path:
        resolved = Path(path)
        if not resolved.is_file():
            raise StructureError(f"{what} file {resolved} does not exist")
        return resolved

    def _stage_ligand(
        self, *, smiles: str | None, ligand_sdf: str | Path | None, workdir: Path, seed: int
    ) -> Path:
        """Put a 3D ligand in the scratch directory, from SMILES or an existing file."""
        if ligand_sdf is not None:
            return self._require_file(ligand_sdf, "ligand")
        if not smiles:
            raise StructureError("dock() needs either smiles or ligand_sdf")
        ligand = workdir / "ligand.sdf"
        # Raises StructureError for chemistry that will not embed — a per-molecule
        # failure the caller records against the candidate, not against GNINA.
        self.write_ligand(smiles, ligand, seed=seed or 0xF00D)
        return ligand

    @staticmethod
    def _pose_metrics(poses: list[Pose]) -> dict[str, float]:
        """What the ledger keeps: how many poses, and the best of each quantity.

        A collapse in pose count, or an affinity that stops moving, is how a wrong
        receptor, an empty box or a broken ligand writer announces itself.
        """
        out: dict[str, float] = {"gnina_n_poses": float(len(poses))}
        scored = [p.score for p in poses if p.score is not None]
        if scored:
            out["gnina_best_affinity_kcal_mol"] = float(min(scored))
        for key in ("CNNscore", "CNNaffinity"):
            best = _best_flag_value(poses, key)
            if best is not None:
                out[f"gnina_best_{key.lower()}"] = best
        return out

    @staticmethod
    def _tail(text: str) -> str:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return "\n".join(lines[-_ERROR_TAIL_LINES:])


def _best_flag_value(poses: list[Pose], key: str) -> float | None:
    """The largest ``key=value`` carried on any pose, or ``None`` if none carry it.

    Both CNN quantities are higher-is-better, unlike the score, which is why this
    maximises rather than minimising.
    """
    prefix = f"{key}="
    values: list[float] = []
    for pose in poses:
        for flag in pose.qc_flags:
            if flag.startswith(prefix):
                try:
                    values.append(float(flag[len(prefix) :]))
                except ValueError:
                    continue
    return max(values) if values else None


__all__ = [
    "CARRIED_SDF_PROPS",
    "CNN_SCORING_LEVELS",
    "GNINA_ADAPTER_VERSION",
    "GNINA_BINARY",
    "POSE_SORT_ORDERS",
    "SCORE_PROP",
    "SPEC",
    "GninaAdapter",
]
