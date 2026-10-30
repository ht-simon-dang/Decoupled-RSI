"""P2Rank and PRANK — one program, one model directory, two things to point it at.

P2Rank (Krivák & Hoksza, *J. Cheminformatics* 2018) does not look for cavities. It
covers the protein's solvent-accessible surface with points, describes each point
from the chemistry of the atoms around it, asks a random forest how *ligandable*
that point is, and clusters the high-scoring points into pockets. The pocket score
is the sum of its points' ligandability, so the ranking is learned rather than
geometric — which is the complementary failure mode to fpocket's, and the reason
both are worth having.

**PRANK is the same program in its other mode, and that is why this is one
capability.** ``prank predict`` runs the pipeline above from scratch;
``prank rescore`` takes *somebody else's* pockets — fpocket's ``<stem>_out.pdb``,
ConCavity's, PUResNet's — and re-ranks them with the same point-scoring machinery
against the ``default_rescore`` model. They are two commands of one 275 MB release
that ships both model directories, they need the same JRE, they fail for the same
reasons, and their readiness is one filesystem answer. Splitting them into two
capability cards would create the one state that is actually wrong: an install where
``pocket.detect.p2rank`` is AVAILABLE and ``pocket.rescore.prank`` is not, from the
same directory. So they are :meth:`P2RankAdapter.detect` and
:meth:`P2RankAdapter.rescore` on one adapter, exactly as
:class:`~drug_agent.tools.structure.gnina_tool.GninaAdapter` carries ``dock`` and
``score_pose``. What the two modes are *not* is interchangeable: ``rescore``
consumes an fpocket run and finds nothing fpocket did not, and its output says so
by carrying fpocket's own pocket names (``pocket.1``) and the rank it moved from.

**No network at run time; one download at setup.** P2Rank publishes a self-contained
release archive — the JAR, its dependencies, and the trained forests under
``models/`` — and the ``default`` model is what ``config/default.groovy`` selects.
``scripts/external/fetch_pockets.py --only p2rank`` fetches release 2.5.1 once,
verifies size and gzip magic, unpacks it to ``.external/p2rank`` and writes a
``drugagent_provenance.json`` beside it recording the URL, the release tag, the
SHA-256 of the archive and the fetch timestamp. The detection path reads the model
off local disk and nothing else; the provenance record is what
:meth:`~drug_agent.tools.pockets.PocketToolAdapter.artifact_snapshot` turns into a
content-hashed ``snapshot_id`` a run can cite.

**It needs a Java runtime, and this adapter checks rather than assumes.** P2Rank
2.5.1's release notes state supported Java versions 17 to 24. The probe looks for
``java`` on PATH and, when it finds one, reads ``JAVA_VERSION`` out of the JDK's
``release`` file — a file read, not a process launch, because readiness is computed
on every ``drugagent tools`` and once per adapter construction. A missing JRE is
reported as a blocker naming the install command; an out-of-range JRE is reported as
a blocker naming the version found and the supported range, because P2Rank's failure
on an unsupported JVM is a class-file error a long way from its cause.

**Measured against this repository's own target.** On ``.external/pdb/3I28.pdb``
(sEH, 4,316 ATOM records, 2,689 exposed atoms) P2Rank scores 7,510 surface points,
finds 197 ligandable ones, and reports 7 pockets in **2.5 s** with visualisations
off. Pocket 1 scores 53.21 at probability 0.983 over 99 surface atoms, centred at
(76.290, -8.279, 64.778) — which is **3.53 A** from the centroid of the
hand-prepared catalytic-site file ``.external/pockets/seh_3i28_pocket10.pdb``, and
4.51 A from where fpocket independently puts its own rank-1 pocket. Two consecutive
runs produced byte-identical ``_predictions.csv``, which is why the card says
DETERMINISTIC where fpocket's says STOCHASTIC. Rescoring fpocket's 29 pockets with
PRANK keeps fpocket's pocket 1 at rank 1 (score 38.70, probability 0.912) and
promotes its pocket 14 from rank 14 to rank 2, which is the whole point of the mode:
the disagreement is between the geometry ranking and the learned one.

**On the size of the pocket this writes.** P2Rank reports a *centre*, not an extent,
so the cut is the single-point one — ``scripts/prepare_pocket.py --center`` exactly.
That is systematically tighter than a cut around a ligand: on this structure, 10 A
around P2Rank's rank-1 centre selects 286 atoms / 32 residues, while 10 A around the
crystallographic ligand 34N selects the reference file's 752 atoms / 89 residues.
Both are "a few hundred atoms" and both are valid pocket files; a caller who wants
the reference's extent should pass ``radius=14``, which selects 755 atoms around the
same centre. The default stays 10 A because that is the number every other pocket in
this repository was cut at, and a default that varied per detector would make two
pockets incomparable for a reason that has nothing to do with the detectors.
"""

from __future__ import annotations

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
from ._pdb import Atom, Point

#: This adapter's contract with P2Rank, not P2Rank's own version.
P2RANK_ADAPTER_VERSION = "1.0.0"

#: The release this adapter was written against, and the archive it comes in.
P2RANK_RELEASE = "2.5.1"
P2RANK_ARCHIVE_URL = (
    f"https://github.com/rdk/p2rank/releases/download/{P2RANK_RELEASE}/p2rank_{P2RANK_RELEASE}.tar.gz"
)
P2RANK_ARCHIVE_BYTES = 275_625_956

#: Java runtimes P2Rank 2.5.1 supports, from its own release notes.
JAVA_MIN_MAJOR = 17
JAVA_MAX_MAJOR = 24

INSTALL_HINT = "python scripts/external/fetch_pockets.py --only p2rank"

#: How to get a JRE when there is none. Named separately from ``install_hint``
#: because "download P2Rank again" is the wrong advice for a missing JVM.
JRE_HINT = (
    "install a Java runtime, e.g. `sudo apt-get install -y openjdk-21-jre-headless` "
    "or `conda create -y -n drug-agent-p2rank -c conda-forge openjdk=21`"
)

#: ``<basename>_predictions.csv`` header, as ``PredictionSummary`` writes it. The
#: parser matches on these names rather than on column positions, because the
#: residue and atom id columns are themselves space-separated lists and a
#: position-based reader silently mis-aligns the moment upstream adds a column.
PREDICTION_COLUMNS = (
    "name",
    "rank",
    "score",
    "probability",
    "sas_points",
    "surf_atoms",
    "center_x",
    "center_y",
    "center_z",
    "residue_ids",
    "surf_atom_ids",
)

#: Columns of ``<basename>_rescored.csv``, written by ``rescore`` only.
RESCORE_COLUMNS = ("name", "score", "rank", "old_rank", "change")

#: ``JAVA_VERSION="21.0.12"`` in a JDK's ``release`` file.
_JAVA_RELEASE_VERSION = re.compile(r'^JAVA_VERSION="?([0-9][0-9._]*)', re.MULTILINE)

#: How many lines of a failed run's output go into the raised error.
_ERROR_TAIL_LINES = 20


def java_report() -> tuple[str | None, str | None]:
    """``(java path, version string)``. Either may be ``None``.

    The version comes from ``<java_home>/release`` — a text file every OpenJDK build
    ships — reached by resolving the ``java`` symlink and stepping up out of ``bin``.
    ``java -version`` would be more direct and is deliberately not used: this runs on
    every readiness probe, and :class:`~drug_agent.tools._ext.backend.ExternalBackend`
    does filesystem work only for exactly that reason. A JRE whose ``release`` file
    cannot be read still counts as present — an unreadable version is not evidence of
    a wrong one.
    """
    found = shutil.which("java")
    if found is None:
        return None, None
    try:
        home = Path(found).resolve().parent.parent
        text = (home / "release").read_text()
    except OSError:
        return found, None
    match = _JAVA_RELEASE_VERSION.search(text)
    return found, match.group(1) if match else None


def _java_major(version: str) -> int | None:
    """Major version of ``"21.0.12"`` or the legacy ``"1.8.0_402"``."""
    parts = version.replace("_", ".").split(".")
    try:
        first = int(parts[0])
    except (ValueError, IndexError):
        return None
    if first == 1 and len(parts) > 1:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return first


def _platform_blockers() -> tuple[str, ...]:
    """Blockers true of this machine whatever else is installed: the JVM.

    Evaluated per adapter construction rather than at import, so a JRE installed
    after the process started is noticed on the next ``tool_pool()`` call.
    """
    java, version = java_report()
    if java is None:
        return (
            f"no 'java' executable on PATH; P2Rank {P2RANK_RELEASE} is a JVM program and "
            f"supports Java {JAVA_MIN_MAJOR}-{JAVA_MAX_MAJOR}. To fix: {JRE_HINT}.",
        )
    if version is None:
        return ()
    major = _java_major(version)
    if major is None or JAVA_MIN_MAJOR <= major <= JAVA_MAX_MAJOR:
        return ()
    return (
        f"{java} is Java {version}, outside the {JAVA_MIN_MAJOR}-{JAVA_MAX_MAJOR} range "
        f"P2Rank {P2RANK_RELEASE} supports; it fails with a class-file version error a long "
        f"way from its cause. To fix: {JRE_HINT}.",
    )


def build_spec() -> ExternalSpec:
    """The spec, with the JVM blockers evaluated now rather than at import."""
    return ExternalSpec(
        name="P2Rank",
        env_prefix="P2RANK",
        checkout_relpath=".external/p2rank",
        # The launcher, the JAR it puts on the classpath, the config `predict`
        # loads, and the config `rescore` loads on top of it. A directory without
        # all four is not a P2Rank release, whatever else is in it.
        sentinel_files=(
            "prank",
            "bin/p2rank.jar",
            "config/default.groovy",
            "config/default_rescore.groovy",
        ),
        # No driver and no runtime modules: P2Rank is a JVM program launched
        # through a shell script, so there is no interpreter to select and nothing
        # to import. `binary` stays None because the thing that must be on PATH is
        # `java`, whose absence is reported as a blocker that names the JRE install
        # command rather than the P2Rank download.
        driver_name=None,
        selection_modules=(),
        runtime_modules=(),
        # The trained forests. `default` is what `predict` uses and
        # `default_rescore` is what `rescore` uses; naming both means an archive
        # that unpacked halfway cannot report AVAILABLE.
        weight_relpaths=(
            "models/default/model.zst",
            "models/default_rescore/model.zst",
        ),
        conda_envs=(),
        binary=None,
        upstream_url="https://github.com/rdk/p2rank",
        upstream_ref=P2RANK_RELEASE,
        license="P2Rank MIT (Krivák & Hoksza, J. Cheminformatics 2018, 10:39)",
        install_hint=INSTALL_HINT,
        needs_gpu=False,
        platform_blockers=_platform_blockers(),
    )


SPEC = build_spec()


class P2RankAdapter(PocketToolAdapter):
    """Ligandability prediction over solvent-accessible surface points."""

    capability_id = "pocket.detect.p2rank"
    spec = SPEC
    engine = "p2rank"
    version = P2RANK_ADAPTER_VERSION
    score_kind = "p2rank_score"

    def __init__(self, *, settings: Any = None, timeout_s: float | None = None) -> None:
        self.spec = build_spec()
        # 11 s on a 4,300-atom monomer; the ceiling is for a large multimer whose
        # SAS point count grows with surface area, not for the common case.
        super().__init__(settings=settings, timeout_s=timeout_s if timeout_s is not None else 900.0)
        self._version: str | None = None

    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Ligandability-based binding-site prediction (P2Rank / PRANK)",
            description=(
                "Scores solvent-accessible surface points with a trained random forest and "
                "clusters the ligandable ones into ranked pockets, writing each as a prepared "
                "pocket PDB. The same model directory also re-ranks another detector's pockets "
                "(PRANK mode), which is a different claim and is reported as one."
            ),
            input_schema_ref="ProteinStructure",
            output_schema_ref="PocketSpec",
            # A trained forest evaluated over deterministic surface points: the same
            # structure and the same model give the same pockets and the same scores.
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.MODERATE,
            known_failure_modes=[
                "no Java runtime, or one outside the 17-24 range P2Rank 2.5.1 supports",
                "a structure P2Rank's parser rejects — malformed mmCIF, no protein chain",
                "rescore given an fpocket directory that is not the one the protein came from",
                "a large multimer's surface point count pushes the run past its budget",
                "the release archive unpacked without models/, so the forest cannot be loaded",
            ],
            validation_checks=[
                "score is P2Rank's pocket score; probability is its fitted transform, not a duplicate",
                "scores are never compared with fpocket's — different axes, geometric check only",
                "every prepared pocket file is a whole-residue cut of the input structure",
                "rescore output carries the upstream detector's pocket names and previous rank",
                "no network access on the prediction path; the model is read from local disk",
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
        max_pockets: int | None = None,
        config: str | None = None,
        model: str | None = None,
        threads: int | None = None,
    ) -> PocketDetection:
        """Predict pockets in *structure_pdb* with ``prank predict``, best first.

        When *out_dir* is given the top *n_prepared* pockets are written there as
        prepared pocket PDBs, cut around the predicted centre with the same
        whole-residue rule as ``scripts/prepare_pocket.py --center``, and each
        path lands on :attr:`~drug_agent.tools.pockets.DetectedPocket.pdb_path`.
        """
        self.require_available()
        with measure(self.capability_id, "detect", backend=self.backend) as metered:
            detection = self._predict(
                structure_pdb,
                out_dir=Path(out_dir) if out_dir is not None else None,
                n_prepared=n_prepared,
                radius=radius,
                max_pockets=max_pockets,
                config=config,
                model=model,
                threads=threads,
            )
            for key, value in detection.metrics().items():
                metered[key] = value
        return detection

    def rescore(
        self,
        structure_pdb: str | Path,
        *,
        fpocket_out_pdb: str | Path,
        out_dir: str | Path | None = None,
        n_prepared: int = DEFAULT_N_PREPARED,
        radius: float = DEFAULT_POCKET_RADIUS,
        max_pockets: int | None = None,
        config: str = "default_rescore.groovy",
        threads: int | None = None,
    ) -> PocketDetection:
        """Re-rank pockets another detector already found (``prank rescore``, PRANK).

        *fpocket_out_pdb* is the ``<stem>_out/<stem>_out.pdb`` an fpocket run wrote —
        the annotated structure carrying its alpha spheres — and *structure_pdb* is
        the protein those pockets were computed from. PRANK finds no new pockets: it
        scores the ones it is given, so ``old_rank`` and ``rank_change`` ride on each
        result and the pocket names are fpocket's own.
        """
        self.require_available()
        with measure(self.capability_id, "rescore", backend=self.backend) as metered:
            detection = self._rescore(
                structure_pdb,
                fpocket_out_pdb=Path(fpocket_out_pdb),
                out_dir=Path(out_dir) if out_dir is not None else None,
                n_prepared=n_prepared,
                radius=radius,
                max_pockets=max_pockets,
                config=config,
                threads=threads,
            )
            for key, value in detection.metrics().items():
                metered[key] = value
        return detection

    # --- real backend -------------------------------------------------------
    def _predict(
        self,
        structure_pdb: str | Path,
        *,
        out_dir: Path | None,
        n_prepared: int,
        radius: float,
        max_pockets: int | None,
        config: str | None,
        model: str | None,
        threads: int | None,
    ) -> PocketDetection:
        source, atoms = self.read_structure(structure_pdb)
        started = time.perf_counter()
        with self.scratch("p2rank") as workdir:
            staged = self.stage_structure(source, workdir)
            results = workdir / "out"
            argv = ["predict", "-f", str(staged), "-o", str(results)]
            argv += self._common_args(config=config, model=model, threads=threads)
            self._run(argv, workdir)
            rows = self._read_predictions(results, staged.name)
        seconds = time.perf_counter() - started
        return self._assemble(
            source=source,
            atoms=atoms,
            rows=rows,
            rescored={},
            out_dir=out_dir,
            n_prepared=n_prepared,
            radius=radius,
            max_pockets=max_pockets,
            seconds=seconds,
            operation="predict",
            config={"config": config, "model": model, "threads": threads},
        )

    def _rescore(
        self,
        structure_pdb: str | Path,
        *,
        fpocket_out_pdb: Path,
        out_dir: Path | None,
        n_prepared: int,
        radius: float,
        max_pockets: int | None,
        config: str,
        threads: int | None,
    ) -> PocketDetection:
        source, atoms = self.read_structure(structure_pdb)
        if not fpocket_out_pdb.is_file():
            raise PocketError(
                f"fpocket output {fpocket_out_pdb} does not exist; PRANK rescoring needs the "
                f"<stem>_out/<stem>_out.pdb an fpocket run wrote, not the pocket files under it"
            )
        started = time.perf_counter()
        with self.scratch("prank_rescore") as workdir:
            dataset = workdir / "rescore.ds"
            dataset.write_text(
                "PARAM.PREDICTION_METHOD=fpocket\n\n"
                "HEADER: prediction protein\n\n"
                f"{fpocket_out_pdb.resolve()}  {source.resolve()}\n"
            )
            results = workdir / "out"
            argv = ["rescore", str(dataset), "-o", str(results)]
            argv += self._common_args(config=config, model=None, threads=threads)
            self._run(argv, workdir)
            rows = self._read_predictions(results, source.name)
            rescored = self._read_rescored(results, source.name)
        seconds = time.perf_counter() - started
        return self._assemble(
            source=source,
            atoms=atoms,
            rows=rows,
            rescored=rescored,
            out_dir=out_dir,
            n_prepared=n_prepared,
            radius=radius,
            max_pockets=max_pockets,
            seconds=seconds,
            operation="rescore",
            config={"config": config, "threads": threads, "upstream_method": "fpocket"},
        )

    def _assemble(
        self,
        *,
        source: Path,
        atoms: list[Atom],
        rows: list[dict[str, str]],
        rescored: dict[str, dict[str, str]],
        out_dir: Path | None,
        n_prepared: int,
        radius: float,
        max_pockets: int | None,
        seconds: float,
        operation: str,
        config: dict[str, Any],
    ) -> PocketDetection:
        warnings: list[str] = []
        if not rows:
            warnings.append("P2Rank found no ligandable pocket on this structure")
        if max_pockets is not None:
            rows = rows[: max(0, int(max_pockets))]

        pockets: list[DetectedPocket] = []
        for rank, row in enumerate(rows, start=1):
            center = self._center(row)
            residue_ids = tuple((row.get("residue_ids") or "").split())
            properties: dict[str, float] = {}
            for key in ("sas_points", "surf_atoms", "rank"):
                value = _as_float(row.get(key))
                if value is not None:
                    properties[f"p2rank_{key}"] = value
            label = (row.get("name") or "").strip() or f"pocket{rank}"
            extra = rescored.get(row.get("name", ""), {})
            for key, out_key in (("old_rank", "p2rank_old_rank"), ("change", "p2rank_rank_change")):
                value = _as_float(extra.get(key))
                if value is not None:
                    properties[out_key] = value
            pocket = DetectedPocket(
                rank=rank,
                score=_as_float(row.get("score")),
                score_kind=self.score_kind,
                center=center,
                engine=self.engine,
                engine_version=self._engine_version(),
                site_points=[center],
                label=label,
                probability=_as_float(row.get("probability")),
                radius_a=float(radius),
                properties=properties,
            )
            if out_dir is not None and rank <= max(0, int(n_prepared)):
                path, cut = self.prepare_pocket_file(
                    atoms,
                    site_points=[center],
                    radius=float(radius),
                    out_dir=out_dir,
                    stem=f"{source.stem}_p2rank_{operation}_p{rank}",
                    provenance={
                        "detector": f"{self.engine} {pocket.engine_version} ({operation})",
                        "capability": self.capability_id,
                        "source_structure": source.name,
                        "upstream_pocket": label,
                        "rank": rank,
                        self.score_kind: row.get("score", "").strip(),
                        "probability": row.get("probability", "").strip(),
                        "center": ", ".join(f"{c:.3f}" for c in center),
                        "radius_a": f"{float(radius):g}",
                    },
                )
                pocket.pdb_path = str(path)
                pocket.residues = cut.residues
                pocket.n_residues = cut.n_residues
                pocket.n_atoms = cut.n_atoms
                pocket.warnings.extend(cut.warnings)
            if not pocket.residues and residue_ids:
                # P2Rank reports the surface residues it scored even when no file was
                # cut. They are its own selection, not a 10 A cut, and are kept so a
                # caller who asked for no files still learns which residues were hit.
                pocket.residues = residue_ids
                pocket.n_residues = len(residue_ids)
            pockets.append(pocket)

        return PocketDetection(
            structure_path=str(source),
            engine=self.engine,
            engine_version=self._engine_version(),
            pockets=pockets,
            runtime_seconds=seconds,
            snapshot_id=self.snapshot_id(),
            config={"operation": operation, "radius_a": float(radius), **config},
            warnings=warnings,
        )

    # --- argv ----------------------------------------------------------------
    def _common_args(self, *, config: str | None, model: str | None, threads: int | None) -> list[str]:
        """Flags shared by both commands.

        Visualisations are off: they are PyMOL scripts plus a copy of the input
        structure, which is tens of megabytes per run and is not something a
        programmatic caller reads.
        """
        args = ["-visualizations", "0"]
        if config:
            args += ["-c", config]
        if model:
            args += ["-m", model]
        if threads is not None:
            if int(threads) < 1:
                raise PocketError("threads must be >= 1")
            args += ["-threads", str(int(threads))]
        return args

    # --- process --------------------------------------------------------------
    def _launcher(self) -> Path:
        root = self.ext.root
        if root is None:  # pragma: no cover - require_available() has already refused
            raise PocketError("no P2Rank checkout resolved")
        return root / "prank"

    def _run(self, argv: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
        launcher = self._launcher()
        try:
            proc = run_sandboxed(
                str(launcher),
                argv,
                cwd=workdir,
                timeout=self.timeout_s,
                env=self.ext.child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise PocketError(
                f"P2Rank exceeded its {self.timeout_s:.0f}s budget and was terminated; "
                f"raise DRUGAGENT_P2RANK_TIMEOUT_SECONDS, or lower -threads on a loaded host"
            ) from exc
        except OSError as exc:
            raise PocketError(f"could not launch {launcher}: {exc}") from exc
        if proc.returncode != 0:
            detail = self._tail(proc.stderr or "") or self._tail(proc.stdout or "")
            raise PocketError(f"P2Rank exited {proc.returncode}:\n{detail}")
        return proc

    def _engine_version(self) -> str:
        """The release this checkout is, taken from the spec rather than the JVM.

        ``prank`` prints its version in a banner, but only after starting a JVM and
        loading the config — around a second, on every call. The release tag is
        already pinned by the spec and verified by ``fetch_pockets.py`` against the
        archive it downloaded, so there is nothing a banner would add.
        """
        if self._version is None:
            self._version = self.spec.upstream_ref or "unknown"
        return self._version

    # --- parsing ---------------------------------------------------------------
    def _read_predictions(self, results: Path, structure_name: str) -> list[dict[str, str]]:
        path = results / f"{structure_name}_predictions.csv"
        if not path.is_file():
            candidates = sorted(p.name for p in results.glob("*_predictions.csv"))
            raise PocketError(
                f"P2Rank exited cleanly but wrote no {path.name}"
                + (f" (found {', '.join(candidates)})" if candidates else "")
            )
        return _read_csv(path.read_text(), PREDICTION_COLUMNS)

    def _read_rescored(self, results: Path, structure_name: str) -> dict[str, dict[str, str]]:
        path = results / f"{structure_name}_rescored.csv"
        if not path.is_file():
            return {}
        return {row["name"]: row for row in _read_csv(path.read_text(), RESCORE_COLUMNS)}

    @staticmethod
    def _center(row: dict[str, str]) -> Point:
        try:
            return (
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["center_z"]),
            )
        except (KeyError, ValueError) as exc:
            raise PocketError(f"P2Rank prediction row has no usable centre: {row!r}") from exc

    @staticmethod
    def _tail(text: str) -> str:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return "\n".join(lines[-_ERROR_TAIL_LINES:])


def _read_csv(text: str, expected: tuple[str, ...]) -> list[dict[str, str]]:
    """P2Rank's whitespace-padded CSV as a list of stripped ``{column: value}``.

    Not :mod:`csv`: the file is comma-separated with every field padded to a fixed
    width, and two of the columns are themselves space-separated lists, so the useful
    step is stripping rather than quoting. The header is checked against *expected*
    so a column added upstream is a loud failure here instead of a silently shifted
    coordinate.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = [cell.strip() for cell in lines[0].split(",")]
    missing = [name for name in expected if name not in header]
    if missing:
        raise PocketError(
            f"P2Rank CSV is missing the column(s) {', '.join(missing)}; header was {', '.join(header)}"
        )
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < len(header):
            continue
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


def _as_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return None


__all__ = [
    "INSTALL_HINT",
    "JAVA_MAX_MAJOR",
    "JAVA_MIN_MAJOR",
    "JRE_HINT",
    "P2RANK_ADAPTER_VERSION",
    "P2RANK_ARCHIVE_BYTES",
    "P2RANK_ARCHIVE_URL",
    "P2RANK_RELEASE",
    "PREDICTION_COLUMNS",
    "RESCORE_COLUMNS",
    "SPEC",
    "P2RankAdapter",
    "build_spec",
    "java_report",
]
