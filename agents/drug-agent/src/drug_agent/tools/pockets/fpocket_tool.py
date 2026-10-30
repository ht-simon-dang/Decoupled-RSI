"""fpocket — Voronoi alpha spheres, and the one pocket detector with no excuse.

fpocket (Le Guilloux, Schmidtke & Tuffery, *BMC Bioinformatics* 2009) finds cavities
by geometry alone. It computes the Voronoi tessellation of the protein's atoms,
keeps the alpha spheres whose radius falls in a band that corresponds to "small
enough to be a pocket, large enough not to be inside the protein" (3.4-6.2 A by
default), clusters the survivors by single linkage, and calls each cluster a pocket.
There is no training set, no checkpoint, no embedding and no network: the input is
coordinates and the output is a function of them.

That is why this adapter is the one in this package that is expected to be
AVAILABLE. It is a 1.8 MB C binary on conda-forge, it runs in **0.95 s** on this
repository's 4,316-atom ``.external/pdb/3I28.pdb``, and the fix for a missing
install is one command that needs no weights and no GPU::

    conda create -y -n drug-agent-fpocket -c conda-forge fpocket

**What it computes, and which numbers mean what.** ``<stem>_out/<stem>_info.txt``
carries nineteen descriptors per pocket and this adapter parses all of them into
:attr:`~drug_agent.tools.pockets.DetectedPocket.properties` verbatim. Three are
promoted to named fields because they are the ones anybody ranks on:

    ``Score``               the alpha-sphere-geometry pocket score fpocket sorts by.
                            Reported as ``score``, with ``score_kind`` saying so.
    ``Druggability Score``  a separate logistic model over hydrophobic density,
                            polarity and volume, fitted on a druggable-site
                            benchmark. 0-1, higher is more druggable. It is *not*
                            the pocket score and the two disagree in general.
    ``Volume``              a Monte-Carlo estimate, in cubic Angstrom.

**The volume is the only stochastic thing here, and it really is stochastic.**
``-v/--iterations_volume_mc`` defaults to 300 Monte-Carlo iterations and fpocket
seeds them from the clock. Measured on ``3I28.pdb``: two consecutive runs produced
byte-identical alpha-sphere coordinates, byte-identical contact atoms, and
identical values for every descriptor **except** ``Volume``, which moved from
1023.763 to 1029.159 A^3 on pocket 1 and by up to 5% on the smaller ones. The
capability is therefore labelled ``STOCHASTIC`` — one field of it is — while the
ranking, the centres and the prepared pocket files are reproducible.

**Where the output goes, which is not where you would guess.** ``fpocket -f
/abs/path/5AI0.pdb`` run from an empty directory writes ``5AI0_out/`` *next to the
input file*, not into the working directory: verified directly, and it would mean a
detection run silently depositing 29 pockets' worth of files into
``.external/pdb/``. Every run here therefore copies the structure into a scratch
directory first and deletes the whole thing afterwards.

**How a detection becomes a generator input.** The alpha-sphere centres in
``pockets/pocket<N>_vert.pqr`` are the site: they fill the cavity the way a ligand
would. :meth:`FpocketAdapter.detect` takes the cut around those points (or around
their centroid, with ``site_mode="center"``) using the same 10 A whole-residue rule
as ``scripts/prepare_pocket.py``, writes it, and returns the path on
:attr:`~drug_agent.tools.pockets.DetectedPocket.pdb_path`, which is exactly what
:attr:`~drug_agent.contracts.brief.PocketSpec.pdb_path` wants.

**Measured against this repository's own target.** On ``.external/pdb/3I28.pdb``
fpocket returns 29 pockets in 0.86 s. Pocket 1 scores 1.048 with a druggability of
0.979 over 159 alpha spheres and ~1020-1045 A^3, and its alpha-sphere centroid
(76.182, -9.069, 69.214) is **2.85 A** from the centroid of the hand-prepared
catalytic-site file ``.external/pockets/seh_3i28_pocket10.pdb``. The next-nearest of
the 29 is 15.05 A away, and the runner-up by score sits 71 A away with a
druggability of 0.001. The detector finds the sEH catalytic site, unprompted, and
ranks it first by a wide margin.

The prepared file that comes out of it is the right size for a generator. With
``site_mode="alpha_spheres"`` the rank-1 cut is 862 atoms / 103 residues, against
the hand-prepared reference's 752 / 89 — the same order, because alpha spheres fill
the cavity the way the ligand does. With ``site_mode="center"`` it is 317 / 36, the
tighter single-point cut.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ...contracts.capability import CapabilityProfile
from ...contracts.enums import CostClass, Determinism
from ...observability.toolmeter import measure
from ...security.sandbox import run_sandboxed
from .._ext.spec import ExternalSpec
from . import (
    DEFAULT_N_PREPARED,
    DEFAULT_POCKET_RADIUS,
    DetectedPocket,
    PocketDetection,
    PocketError,
    PocketToolAdapter,
)
from ._pdb import Atom, Point, centroid, parse_points

#: This adapter's contract with fpocket, not fpocket's own version.
FPOCKET_ADAPTER_VERSION = "1.0.0"

#: The executable the whole capability rests on.
FPOCKET_BINARY = "fpocket"

#: The environment ``scripts/external/fetch_pockets.py --only fpocket`` creates.
#: fpocket gets its own environment for the same reason every other backend does:
#: the nine verified environments are not to be perturbed by a new solve.
FPOCKET_ENV = "drug-agent-fpocket"

#: Where an operator can name the binary when it is neither on PATH nor in the
#: conda environment above — a system package, a source build.
FPOCKET_BINARY_ENV_VAR = "DRUGAGENT_FPOCKET_BINARY"

INSTALL_HINT = f"conda create -y -n {FPOCKET_ENV} -c conda-forge fpocket"

#: ``Key : value`` inside a ``Pocket N :`` block of ``<stem>_info.txt``. The keys are
#: free text with spaces, dots and hyphens ("Cent. of mass - Alpha Sphere max dist"),
#: so they are normalised rather than matched against a fixed list — a new fpocket
#: release adding a descriptor should surface it, not drop it.
_INFO_HEADER = re.compile(r"^Pocket\s+(\d+)\s*:")
_INFO_FIELD = re.compile(r"^\s+(?P<key>[^:]+?)\s*:\s*(?P<value>-?\d+(?:\.\d+)?)\s*$")

#: fpocket prints its version in an ANSI-bold banner when invoked with no input.
_VERSION_BANNER = re.compile(r"fpocket\s+(\d+(?:\.\d+)*)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: Descriptors promoted out of ``properties`` onto named fields, keyed by their
#: normalised ``info.txt`` name.
SCORE_KEY = "score"
DRUGGABILITY_KEY = "druggability_score"
VOLUME_KEY = "volume"

#: How the site is turned into points for the pocket cut.
SITE_MODES = ("alpha_spheres", "center")

#: How many lines of a failed run's output go into the raised error.
_ERROR_TAIL_LINES = 20


def _conda_roots() -> list[Path]:
    """Conda installations to look in, from the same variables the probe uses.

    Mirrors :func:`drug_agent.tools._ext.probe.conda_env_pythons`, which solves the
    identical problem for interpreters: the process that launched the platform is
    usually not the environment a backend was installed into.
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
    """An explicitly-named fpocket, following the platform's configuration precedence.

    :meth:`drug_agent.tools._ext.backend.ExternalBackend._resolve` owns this
    precedence for every other configurable value, and adapters are told not to read
    ``os.environ`` themselves so that it cannot differ between tools. It cannot be
    reused here for one reason: the binary has to be resolved *before* the backend
    exists, because it is what the backend is constructed to probe for. So the order
    is restated, deliberately and in one place, and it is the same order:

    1. ``settings.external_tools["fpocket"]["binary"]`` — the profile YAML;
    2. ``settings.fpocket_binary`` — a flat settings attribute;
    3. ``DRUGAGENT_FPOCKET_BINARY`` — the operator's escape hatch.

    Discovery (:func:`find_fpocket`) is what happens when none of the three answers.
    """
    table = getattr(settings, "external_tools", None) if settings is not None else None
    if isinstance(table, dict):
        entry = table.get("fpocket")
        if isinstance(entry, dict) and entry.get("binary"):
            return str(entry["binary"])
    flat = getattr(settings, "fpocket_binary", None) if settings is not None else None
    if flat:
        return str(flat)
    return os.environ.get(FPOCKET_BINARY_ENV_VAR) or None


def find_fpocket(settings: Any = None) -> str | None:
    """The fpocket executable to use, or ``None`` if there is not one.

    Configuration first (:func:`_configured_binary`), then ``PATH``, then
    ``envs/drug-agent-fpocket/bin`` under every conda installation this process can
    see. The last of those is what makes the documented install work without an
    operator having to activate an environment the platform does not run in: the
    constraint that the nine verified environments stay untouched means the binary is
    deliberately *not* on this interpreter's PATH.
    """
    configured = _configured_binary(settings)
    if configured:
        found = shutil.which(configured)
        if found:
            return found
    on_path = shutil.which(FPOCKET_BINARY)
    if on_path:
        return on_path
    for root in _conda_roots():
        cand = root / "envs" / FPOCKET_ENV / "bin" / FPOCKET_BINARY
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


SPEC = ExternalSpec(
    name="fpocket",
    env_prefix="FPOCKET",
    # No checkout: fpocket is a packaged binary, and the source tree is not on any
    # code path here. No weights either — it is geometry, so there is no artifact a
    # half-finished download could leave broken.
    checkout_relpath="",
    sentinel_files=(),
    driver_name=None,
    selection_modules=(),
    runtime_modules=(),
    weight_relpaths=(),
    conda_envs=(),
    binary=FPOCKET_BINARY,
    upstream_url="https://github.com/Discngine/fpocket",
    upstream_ref="4.2.3 (conda-forge h5fd1fdb_2)",
    license="fpocket MIT (Le Guilloux, Schmidtke & Tuffery, BMC Bioinformatics 2009, 10:168)",
    install_hint=INSTALL_HINT,
    needs_gpu=False,
)


class FpocketAdapter(PocketToolAdapter):
    """Geometry-based binding-site detection over Voronoi alpha spheres."""

    capability_id = "pocket.detect.fpocket"
    spec = SPEC
    engine = "fpocket"
    version = FPOCKET_ADAPTER_VERSION
    score_kind = "fpocket_score"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        # Resolve the binary before the base class probes for it, and resolve it per
        # construction rather than at import: `tool_pool()` builds a fresh adapter on
        # every call, so an install that finished after startup is noticed, and an
        # absolute path makes `shutil.which` in the probe succeed for a binary that
        # is deliberately not on this interpreter's PATH.
        binary = find_fpocket(settings)
        if binary is not None:
            self.spec = dataclasses.replace(SPEC, binary=binary)
        # A second on 4,000 atoms; 300 s is a ceiling for a very large complex, not
        # a budget anybody should hit.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 300.0)
        self._version: str | None = None

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Binding-site detection by Voronoi alpha spheres (fpocket)",
            description=(
                "Finds cavities from geometry alone — Voronoi tessellation, alpha-sphere "
                "filtering, single-linkage clustering — and writes each ranked site out as a "
                "prepared pocket PDB that PocketSpec.pdb_path can point at. Reports fpocket's "
                "own pocket score, its separate druggability score, and a Monte-Carlo volume."
            ),
            input_schema_ref="ProteinStructure",
            output_schema_ref="PocketSpec",
            # Only the Monte-Carlo volume varies between runs: measured on 3I28, two
            # runs agreed to the byte on alpha-sphere coordinates, contact atoms and
            # every other descriptor, and disagreed by up to 5% on Volume.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.CHEAP,
            known_failure_modes=[
                "input is not a PDB/mmCIF file fpocket's parser accepts",
                "a structure with no cavity above the alpha-sphere thresholds yields zero pockets",
                "a detected centre in a surface groove cuts a pocket too small to condition a generator",
                "Volume is a Monte-Carlo estimate and is not reproducible run to run",
                "fpocket writes beside its input, so a read-only structure directory fails the run",
            ],
            validation_checks=[
                "score is fpocket's pocket Score, never the druggability score",
                "every prepared pocket file is a whole-residue cut of the input structure",
                "pockets are ranked by score, descending, and rank is re-derived here",
                "no network access on the detection path",
            ],
        )

    # --- operations ---------------------------------------------------------
    def detect(
        self,
        structure_pdb: str | Path,
        *,
        out_dir: str | Path | None = None,
        n_prepared: int = DEFAULT_N_PREPARED,
        radius: float = DEFAULT_POCKET_RADIUS,
        site_mode: str = "alpha_spheres",
        max_pockets: int | None = None,
        min_alpha_size: float | None = None,
        max_alpha_size: float | None = None,
        min_spheres_per_pocket: int | None = None,
        clustering_distance: float | None = None,
        model_number: int | None = None,
        keep_chains: str | None = None,
        drop_chains: str | None = None,
    ) -> PocketDetection:
        """Detect pockets in *structure_pdb*, best first.

        When *out_dir* is given the top *n_prepared* pockets are written there as
        prepared pocket PDBs and each one's path lands on
        :attr:`~drug_agent.tools.pockets.DetectedPocket.pdb_path`; without it the
        geometry and scores are still returned, but nothing that needs a file — the
        pocket-conditioned generators — can consume the result.

        *site_mode* decides what the cut is taken around. ``"alpha_spheres"`` uses
        every alpha-sphere centre in the pocket, which follows a long cavity to both
        of its ends; ``"center"`` uses their centroid alone, which is exactly
        ``scripts/prepare_pocket.py --center`` and gives the tighter, more
        conventional pocket10 sphere.
        """
        self.require_available()
        if site_mode not in SITE_MODES:
            raise PocketError(f"unknown site_mode {site_mode!r}; expected one of {', '.join(SITE_MODES)}")
        with measure(self.capability_id, "detect", backend=self.backend) as metered:
            detection = self._detect(
                structure_pdb,
                out_dir=Path(out_dir) if out_dir is not None else None,
                n_prepared=n_prepared,
                radius=radius,
                site_mode=site_mode,
                max_pockets=max_pockets,
                min_alpha_size=min_alpha_size,
                max_alpha_size=max_alpha_size,
                min_spheres_per_pocket=min_spheres_per_pocket,
                clustering_distance=clustering_distance,
                model_number=model_number,
                keep_chains=keep_chains,
                drop_chains=drop_chains,
            )
            for key, value in detection.metrics().items():
                metered[key] = value
        return detection

    # --- real backend -------------------------------------------------------
    def _detect(
        self,
        structure_pdb: str | Path,
        *,
        out_dir: Path | None,
        n_prepared: int,
        radius: float,
        site_mode: str,
        max_pockets: int | None,
        min_alpha_size: float | None,
        max_alpha_size: float | None,
        min_spheres_per_pocket: int | None,
        clustering_distance: float | None,
        model_number: int | None,
        keep_chains: str | None,
        drop_chains: str | None,
    ) -> PocketDetection:
        source, atoms = self.read_structure(structure_pdb)
        started = time.perf_counter()
        warnings: list[str] = []

        with self.scratch("fpocket") as workdir:
            staged = self.stage_structure(source, workdir)
            argv = ["-f", str(staged)]
            argv += self._detection_args(
                min_alpha_size=min_alpha_size,
                max_alpha_size=max_alpha_size,
                min_spheres_per_pocket=min_spheres_per_pocket,
                clustering_distance=clustering_distance,
                model_number=model_number,
                keep_chains=keep_chains,
                drop_chains=drop_chains,
            )
            self._run(argv, workdir)

            out_root = staged.with_name(f"{staged.stem}_out")
            info_path = out_root / f"{staged.stem}_info.txt"
            if not info_path.is_file():
                raise PocketError(
                    f"fpocket exited cleanly but wrote no {info_path.name}; "
                    f"the structure may contain no protein atoms it recognises"
                )
            descriptors = self._parse_info(info_path.read_text())
            sites = {n: self._read_alpha_spheres(out_root, n) for n in descriptors}

        seconds = time.perf_counter() - started
        if not descriptors:
            warnings.append("fpocket found no pockets above its alpha-sphere thresholds")

        ordered = sorted(
            descriptors.items(),
            key=lambda item: (-(item[1].get(SCORE_KEY) or float("-inf")), item[0]),
        )
        if max_pockets is not None:
            ordered = ordered[: max(0, int(max_pockets))]

        pockets: list[DetectedPocket] = []
        for rank, (fpocket_index, props) in enumerate(ordered, start=1):
            points = sites.get(fpocket_index) or []
            if not points:
                warnings.append(f"fpocket pocket {fpocket_index} has no alpha-sphere file; skipped")
                continue
            center = centroid(points)
            pocket = DetectedPocket(
                rank=rank,
                score=props.get(SCORE_KEY),
                score_kind=self.score_kind,
                center=center,
                engine=self.engine,
                engine_version=self._engine_version(),
                site_points=points,
                druggability=props.get(DRUGGABILITY_KEY),
                volume_a3=props.get(VOLUME_KEY),
                radius_a=float(radius),
                properties=dict(props) | {"fpocket_index": float(fpocket_index)},
            )
            if out_dir is not None and rank <= max(0, int(n_prepared)):
                self._write_pocket(
                    pocket,
                    atoms=atoms,
                    out_dir=out_dir,
                    stem=f"{source.stem}_fpocket_p{rank}",
                    site_mode=site_mode,
                    source=source,
                )
            pockets.append(pocket)

        return PocketDetection(
            structure_path=str(source),
            engine=self.engine,
            engine_version=self._engine_version(),
            pockets=pockets,
            runtime_seconds=seconds,
            snapshot_id=None,  # geometry, not a model: there is no artifact to pin
            config={
                "site_mode": site_mode,
                "radius_a": float(radius),
                "min_alpha_size": min_alpha_size,
                "max_alpha_size": max_alpha_size,
                "min_spheres_per_pocket": min_spheres_per_pocket,
                "clustering_distance": clustering_distance,
                "binary": self.spec.binary,
            },
            warnings=warnings,
        )

    def _write_pocket(
        self,
        pocket: DetectedPocket,
        *,
        atoms: list[Atom],
        out_dir: Path,
        stem: str,
        site_mode: str,
        source: Path,
    ) -> None:
        """Cut and write this pocket, recording what it cost on the pocket itself."""
        points: list[Point] = pocket.site_points if site_mode == "alpha_spheres" else [pocket.center]
        path, cut = self.prepare_pocket_file(
            atoms,
            site_points=points,
            radius=pocket.radius_a,
            out_dir=out_dir,
            stem=stem,
            provenance={
                "detector": f"{self.engine} {pocket.engine_version}",
                "capability": self.capability_id,
                "source_structure": source.name,
                "rank": pocket.rank,
                self.score_kind: "unscored" if pocket.score is None else f"{pocket.score:g}",
                "center": ", ".join(f"{c:.3f}" for c in pocket.center),
                "radius_a": f"{pocket.radius_a:g}",
                "site_mode": site_mode,
            },
        )
        pocket.pdb_path = str(path)
        pocket.residues = cut.residues
        pocket.n_residues = cut.n_residues
        pocket.n_atoms = cut.n_atoms
        pocket.warnings.extend(cut.warnings)

    # --- argv ----------------------------------------------------------------
    @staticmethod
    def _detection_args(
        *,
        min_alpha_size: float | None,
        max_alpha_size: float | None,
        min_spheres_per_pocket: int | None,
        clustering_distance: float | None,
        model_number: int | None,
        keep_chains: str | None,
        drop_chains: str | None,
    ) -> list[str]:
        """fpocket's detection flags, defaults left to fpocket where unset.

        Passing fpocket's own defaults back to it explicitly would freeze this
        adapter to one release's tuning; an unset option here means "whatever this
        build considers standard", which is what the descriptors were calibrated on.
        """
        args: list[str] = []
        if min_alpha_size is not None:
            args += ["-m", f"{float(min_alpha_size):g}"]
        if max_alpha_size is not None:
            args += ["-M", f"{float(max_alpha_size):g}"]
        if clustering_distance is not None:
            args += ["-D", f"{float(clustering_distance):g}"]
        if min_spheres_per_pocket is not None:
            if int(min_spheres_per_pocket) < 1:
                raise PocketError("min_spheres_per_pocket must be >= 1")
            args += ["-i", str(int(min_spheres_per_pocket))]
        if model_number is not None:
            args += ["-l", str(int(model_number))]
        if keep_chains and drop_chains:
            raise PocketError("keep_chains and drop_chains are mutually exclusive in fpocket")
        if keep_chains:
            args += ["-k", keep_chains]
        if drop_chains:
            args += ["-c", drop_chains]
        return args

    # --- process --------------------------------------------------------------
    def _run(self, argv: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
        binary = self.spec.binary
        if binary is None:  # pragma: no cover - require_available() has already refused
            raise PocketError(f"no {FPOCKET_BINARY!r} executable found")
        try:
            proc = run_sandboxed(
                binary,
                argv,
                cwd=workdir,
                timeout=self.timeout_s,
                env=self.ext.child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise PocketError(
                f"fpocket exceeded its {self.timeout_s:.0f}s budget and was terminated; "
                f"raise DRUGAGENT_FPOCKET_TIMEOUT_SECONDS for a very large complex"
            ) from exc
        except OSError as exc:
            raise PocketError(f"could not launch {binary}: {exc}") from exc
        if proc.returncode != 0:
            detail = self._tail(proc.stderr or "") or self._tail(proc.stdout or "")
            raise PocketError(f"fpocket exited {proc.returncode}:\n{detail}")
        return proc

    def _engine_version(self) -> str:
        """What the binary calls itself, asked for once per adapter.

        fpocket has no ``--version``; it prints an ANSI-bold banner when it is given
        no input and exits non-zero. Note that the banner and the package version can
        disagree — the conda-forge build ``fpocket 4.2.3`` prints ``fpocket 4.0`` —
        so this reports what the executable says about itself, which is the thing
        that actually produced the numbers.
        """
        if self._version is not None:
            return self._version
        self._version = "unknown"
        binary = self.spec.binary
        if binary is None:  # pragma: no cover - require_available() has already refused
            return self._version
        try:
            proc = run_sandboxed(binary, [], timeout=60.0, env=self.ext.child_env())
        except (subprocess.SubprocessError, OSError, ValueError):
            return self._version
        text = _ANSI.sub("", f"{proc.stdout or ''}\n{proc.stderr or ''}")
        match = _VERSION_BANNER.search(text)
        if match:
            self._version = match.group(1)
        return self._version

    # --- parsing ---------------------------------------------------------------
    @staticmethod
    def _parse_info(text: str) -> dict[int, dict[str, float]]:
        """``<stem>_info.txt`` as ``{pocket number: {normalised key: value}}``.

        Every numeric descriptor is kept, not a chosen subset: fpocket reports
        nineteen of them and which ones matter is the caller's judgement, not this
        parser's. Keys are lowercased and punctuation-stripped so
        ``"Cent. of mass - Alpha Sphere max dist"`` becomes
        ``cent_of_mass_alpha_sphere_max_dist``.
        """
        pockets: dict[int, dict[str, float]] = {}
        current: dict[str, float] | None = None
        for line in text.splitlines():
            header = _INFO_HEADER.match(line)
            if header:
                current = {}
                pockets[int(header.group(1))] = current
                continue
            if current is None:
                continue
            field = _INFO_FIELD.match(line)
            if field:
                try:
                    current[_normalise_key(field.group("key"))] = float(field.group("value"))
                except ValueError:  # pragma: no cover - the pattern matched a number
                    continue
        return pockets

    @staticmethod
    def _read_alpha_spheres(out_root: Path, index: int) -> list[Point]:
        """Alpha-sphere centres of one pocket, from ``pockets/pocket<N>_vert.pqr``.

        These are the site. fpocket's ``_atm.pdb`` holds the *contacted protein
        atoms* instead, which is a different set: it is the wall of the cavity, and
        cutting a radius around it would grow the pocket outwards into the protein
        rather than filling it.
        """
        path = out_root / "pockets" / f"pocket{index}_vert.pqr"
        if not path.is_file():
            return []
        return parse_points(path.read_text())

    @staticmethod
    def _tail(text: str) -> str:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return "\n".join(lines[-_ERROR_TAIL_LINES:])


def _normalise_key(raw: str) -> str:
    """``"Mean alp. sph. solvent access"`` -> ``mean_alp_sph_solvent_access``."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", raw).strip().lower()
    return re.sub(r"\s+", "_", cleaned)


__all__ = [
    "DRUGGABILITY_KEY",
    "FPOCKET_ADAPTER_VERSION",
    "FPOCKET_BINARY",
    "FPOCKET_ENV",
    "SCORE_KEY",
    "SITE_MODES",
    "SPEC",
    "VOLUME_KEY",
    "FpocketAdapter",
    "find_fpocket",
]
