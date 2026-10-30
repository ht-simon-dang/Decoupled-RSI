"""De-novo molecule generation: the PFM adapter. There is no second backend.

**PFM** (*Perturbed Flow Matching for Structure-Based Drug Design*,
https://github.com/kurisu92725/PFM) is a 3D, pocket-conditioned generative model
in the TargetDiff lineage: it samples ligand atom positions and types inside a
protein binding site by integrating a perturbed flow-matching ODE, then rebuilds
a molecular graph from the point cloud. It is a *generator*, not a scorer — every
molecule it proposes is screened by the same RDKit/sklearn pipeline as any other
candidate, and carries the trust label ``PROPOSED`` until it is.

One backend:

``pfm``
    The real model. Requires a PFM checkout (weights included) and torch +
    torch-geometric + ``torch_scatter``/``torch_cluster`` + OpenBabel, plus a GPU.
    Those deps install cleanly *alongside* this platform — verified on Python
    3.12 / torch 2.14+cu130 (and earlier on 3.13 / torch 2.11) — so the usual
    setup is a single environment and ``DRUGAGENT_PFM_PYTHON`` can be left unset.
    ``torch_scatter``/``torch_cluster`` have no wheel on any platform and compile
    from source, which is why :data:`PFM_RUNTIME_MODULES` checks for them by name:
    they are what a half-finished install leaves out.

    Sampling nonetheless runs through a subprocess
    (``scripts/pfm_sample_pocket.py``), defaulting to ``sys.executable``. That
    costs ~1.6 s of import + checkpoint load against ~1.5 s *per molecule* of
    sampling, and buys two things worth far more: a CUDA fault or OOM inside PFM
    cannot take down the API server, and an operator who *does* want PFM in a
    separate environment (a different CUDA build, an older pinned stack) just
    points ``DRUGAGENT_PFM_PYTHON`` at that interpreter.

There is no second backend. A BRICS-recombination surrogate used to stand in when
PFM was absent, tagging its output ``pfm_mock``; it was removed deliberately.
Labelling fake molecules honestly still put them in front of people who wanted
real ones, and every downstream consumer — ranking, reporting, the evolver's
corpus — had to keep remembering which kind it held. A deployment without PFM now
reports ``UNAVAILABLE`` and generates nothing, which is the answer that cannot be
misread.

Tests keep their own stub (``tests/fakes.py``) so the suite stays hermetic and
GPU-free; it lives on the test side of the boundary precisely so it can never be
selected in production.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from ..contracts.brief import PocketSpec
from ..contracts.capability import CapabilityProfile
from ..contracts.enums import CapabilityStatus, CostClass, Determinism, TrustLabel
from ..observability.toolmeter import measure, note
from .base import ToolAdapter

RDLogger.DisableLog("rdApp.*")

#: Version of *this adapter's* contract with PFM, not of PFM's weights.
PFM_ADAPTER_VERSION = "1.0.0"

#: Origin tag written onto every candidate this adapter produces. The only one:
#: the surrogate that used to write ``pfm_mock`` was removed, so an ``origin`` of
#: ``pfm`` now means the model actually ran.
ORIGIN_PFM = "pfm"

#: The driver executed inside the PFM environment.
DRIVER_NAME = "pfm_sample_pocket.py"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class GeneratedMolecule:
    """One molecule proposed by a generator, before standardization."""

    smiles: str
    origin: str
    sample_index: int
    model_id: str
    model_version: str
    backend: str
    pocket_id: str | None = None
    n_heavy_atoms: int | None = None
    sdf_block: str | None = None
    qc_flags: list[str] = field(default_factory=list)
    #: Generated structures are hypotheses until screened — never CALCULATED.
    trust_label: TrustLabel = TrustLabel.PROPOSED


@dataclass
class GenerationBatch:
    """The outcome of one generation call, with the QC numbers a run must record."""

    molecules: list[GeneratedMolecule] = field(default_factory=list)
    backend: str = "pfm"
    model_id: str = "molecule.generate.pfm"
    model_version: str = PFM_ADAPTER_VERSION
    n_requested: int = 0
    n_raw: int = 0
    n_valid: int = 0
    n_unique: int = 0
    runtime_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def validity(self) -> float:
        """Fraction of raw samples RDKit could parse (the standard SBDD metric)."""
        return round(self.n_valid / self.n_raw, 4) if self.n_raw else 0.0

    @property
    def uniqueness(self) -> float:
        """Fraction of valid samples that are distinct canonical structures."""
        return round(self.n_unique / self.n_valid, 4) if self.n_valid else 0.0

    def metrics(self) -> dict[str, float]:
        return {
            "n_generated_requested": float(self.n_requested),
            "n_generated_raw": float(self.n_raw),
            "n_generated_valid": float(self.n_valid),
            "n_generated_unique": float(self.n_unique),
            "generation_validity": self.validity,
            "generation_uniqueness": self.uniqueness,
            "generation_runtime_seconds": round(self.runtime_seconds, 3),
        }


class GenerationError(RuntimeError):
    """Raised when the real PFM backend was selected and failed."""


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------
@dataclass
class PFMPaths:
    """Where the PFM checkout, its interpreter, and its weights live."""

    root: Path | None = None
    python: str = sys.executable
    checkpoint: Path | None = None
    training_yml: Path | None = None
    x_predictor: Path | None = None
    h_predictor: Path | None = None
    device: str = "cuda:0"
    pocket_pdb: Path | None = None
    driver: Path | None = None
    #: A built target manifest (``target_manifest.json``). When a deployment
    #: declares one, the pocket handed to PFM is checked against it before any
    #: sampling happens — see :meth:`PFMGeneratorAdapter._sample_pfm`. Left unset
    #: the check cannot run, and the tool says so in its metrics rather than
    #: reporting a silent pass.
    target_manifest: Path | None = None

    def missing(self) -> list[str]:
        """Human-readable reasons the real backend cannot be used (empty = usable)."""
        problems: list[str] = []
        if self.root is None:
            return ["DRUGAGENT_PFM_ROOT is not set (path to a PFM checkout)"]
        if not self.root.is_dir():
            return [f"PFM root {self.root} does not exist"]
        for rel in ("scripts/sampling.py", "egnn/DotGat.py", "utils/reconstruct.py"):
            if not (self.root / rel).is_file():
                problems.append(f"{self.root / rel} missing (not a PFM checkout?)")
        if self.checkpoint is None or not self.checkpoint.is_file():
            problems.append(f"PFM checkpoint not found at {self.checkpoint}")
        if self.training_yml is None or not self.training_yml.is_file():
            problems.append(f"PFM training config not found at {self.training_yml}")
        if self.driver is None or not self.driver.is_file():
            problems.append(f"driver script {DRIVER_NAME} not found at {self.driver}")
        if not (self.python and (Path(self.python).is_file() or shutil.which(self.python))):
            problems.append(f"PFM interpreter {self.python!r} not executable")
        elif absent := _interpreter_missing_modules(self.python):
            # Being executable is not enough, and neither is having torch: the compiled
            # extensions are what a half-finished install leaves out. Catching that here
            # turns a mid-run subprocess crash into an upfront MOCKED with a real reason.
            problems.append(
                f"PFM interpreter {self.python} cannot import {', '.join(absent)} — run "
                f"`make pfm-install`, or set DRUGAGENT_PFM_PYTHON to an interpreter that has them"
            )
        return problems


def _default_driver() -> Path | None:
    """Locate ``scripts/pfm_sample_pocket.py`` in an installed or source tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "scripts" / DRIVER_NAME
        if cand.is_file():
            return cand
    return None


#: Where ``make pfm-clone`` puts the checkout, relative to the repo root.
DEFAULT_PFM_RELPATH = ".external/PFM"

#: The conda environment the project is installed into (see ``environment.yml``).
DEFAULT_CONDA_ENV = "drug-agent"


def _conda_env_pythons() -> tuple[str, ...]:
    """Interpreters to try when the running one cannot import torch.

    PFM's dependencies are installed into the project's conda environment by
    ``make pfm-install``. The process that launched the console may nevertheless
    be a *different* interpreter — the base environment's ``python``, first on
    PATH, is the usual culprit — and that one has no torch. So: look in the
    currently activated environment, then in the project environment by name
    under the same conda installation.
    """
    names = [os.environ.get("DRUGAGENT_CONDA_ENV") or DEFAULT_CONDA_ENV]
    candidates: list[str] = []
    roots: list[Path] = []

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        active = Path(prefix)
        candidates += [str(active / "bin" / "python"), str(active / "python.exe")]
        # <root>/envs/<name> for a named env; <root> itself for the base env.
        roots.append(active.parent.parent if active.parent.name == "envs" else active)
    conda_exe = os.environ.get("CONDA_EXE")  # <root>/bin/conda
    if conda_exe:
        roots.append(Path(conda_exe).parent.parent)

    for root in roots:
        for name in names:
            env = root / "envs" / name
            candidates += [str(env / "bin" / "python"), str(env / "python.exe")]

    seen: set[str] = set()
    unique: list[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            unique.append(cand)
    return tuple(unique)


#: Every third-party module PFM's *sampling* path imports (``scripts/sampling.py``,
#: ``egnn/Dot_inter_utils.py``, ``datasets/pl_data.py``). ``torch_scatter`` and
#: ``torch_cluster`` are compiled CUDA extensions with no prebuilt wheel on any
#: platform, so an install that dies during the build leaves exactly this shape behind:
#: torch present, extensions absent. Probing torch alone would then report AVAILABLE and
#: crash at sample time — the very failure this check exists to prevent.
PFM_RUNTIME_MODULES: tuple[str, ...] = ("torch", "torch_geometric", "torch_scatter", "torch_cluster")

#: The largest seed PFM will accept. Its driver calls ``misc.seed_all``, which calls
#: ``numpy.random.seed``, and numpy refuses anything outside ``[0, 2**32 - 1]`` with
#: ``ValueError: Seed must be between 0 and 2**32 - 1``.
#:
#: WHY THIS IS CHECKED HERE RATHER THAN LEFT TO NUMPY. A campaign that chunks a large draw forms a
#: continuation seed per chunk, and a date-like base seed overflows almost immediately: a real run on
#: 2026-09-25 formed 20260925001, which is 4.7x the limit, and the draw died at 297 molecules of 8,000
#: -- PARTIALLY, because the early chunks had already succeeded with in-range seeds. A partial draw is
#: the worst failure this adapter can produce: it looks like a completed sample of a smaller pool, so
#: a campaign can report a fixed-budget yield over a denominator it never actually asked for. Refusing
#: at the boundary, before the subprocess and before any GPU work, turns that into one clear error.
#:
#: The seed is NOT silently reduced modulo the limit. A seed is an identity in this programme --
#: campaign records declare one and reproducibility claims rest on it -- so a run whose effective seed
#: differed from its declared seed would corrupt the record permanently, where a refusal costs a cycle.
PFM_MAX_SEED: int = 2**32 - 1

#: How long a PFM sampling subprocess is given to tear its CUDA context down on SIGTERM before it is
#: SIGKILLed. A context released on SIGTERM costs nothing; one stranded by SIGKILL costs a GPU reset.
_PFM_SIGTERM_GRACE_SECONDS: float = 30.0


def _interpreter_missing_modules(python: str, modules: tuple[str, ...] = PFM_RUNTIME_MODULES) -> list[str]:
    """Which of *modules* ``python`` cannot import — without importing or spawning it.

    Importing torch costs about a second and spawning a probe costs more, and this runs
    on every ``doctor`` and every health poll. For the running interpreter ``find_spec``
    answers for free; for any other one, look for the package in its ``site-packages``.
    """
    # Deliberately abspath, never resolve(): an environment's bin/python is often a
    # symlink to another interpreter, so resolving it walks out of the environment,
    # probes the wrong site-packages, and equates the environment with its base.
    try:
        candidate = Path(os.path.abspath(python))  # noqa: PTH100 - resolve() would follow the symlink
    except (OSError, ValueError):
        return list(modules)
    if candidate == Path(os.path.abspath(sys.executable)):  # noqa: PTH100 - same reason
        absent: list[str] = []
        for mod in modules:
            try:
                found = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                absent.append(mod)
        return absent
    prefix = candidate.parent.parent
    return [
        mod
        for mod in modules
        if not any(prefix.glob(f"lib/python*/site-packages/{mod}/__init__.py"))
        and not any(prefix.glob(f"Lib/site-packages/{mod}/__init__.py"))
    ]


def _interpreter_has_torch(python: str) -> bool:
    """Whether ``python`` can import torch.

    Interpreter *selection* asks only this much: an environment with torch is the one
    PFM was installed into, and whether that install ran to completion is readiness'
    question, not selection's. Keeping them apart means a half-built environment is
    still chosen over one with no torch at all — and then reported honestly, module by
    module, by :meth:`PFMPaths.missing`.
    """
    return not _interpreter_missing_modules(python, ("torch",))


def _default_pfm_python() -> str:
    """The interpreter to run the driver in, when the operator has not named one.

    ``sys.executable`` is the natural default and is right whenever the platform runs
    from the environment PFM was installed into. It is *wrong* when the console was
    started by some other interpreter — the base conda env's python on PATH, say — and
    the failure then lands at sample time as "No module named 'torch'", long after the
    capability reported AVAILABLE. Prefer the running interpreter, fall back to the
    project's conda environment when that one does have torch, and leave ``missing()``
    to explain when neither does.
    """
    if _interpreter_has_torch(sys.executable):
        return sys.executable
    for cand in _conda_env_pythons():
        if Path(cand).is_file() and _interpreter_has_torch(cand):
            return cand
    return sys.executable


def _default_pfm_root() -> Path | None:
    """Find a PFM checkout at the conventional location, if one is really there.

    ``make pfm-clone`` clones into ``.external/PFM``. Requiring an operator who ran
    the documented install to *also* export ``DRUGAGENT_PFM_ROOT`` in every shell
    that launches a server is a trap: the console reports "PFM is not installed"
    for a PFM that is installed. Looking in the place we put it closes that loop.

    The probe is a real PFM checkout, not just a directory, so a stale or empty
    ``.external/PFM`` cannot make the platform claim a model it cannot run.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / DEFAULT_PFM_RELPATH
        if (cand / "scripts" / "sampling.py").is_file():
            return cand.resolve()
    return None


def resolve_pfm_paths(settings: Any = None) -> PFMPaths:
    """Resolve PFM locations from settings, falling back to the environment.

    Settings win when set; ``DRUGAGENT_*`` environment variables are the escape
    hatch for operators who cannot edit a profile YAML.
    """

    def _get(attr: str, env: str) -> str | None:
        val = getattr(settings, attr, None) if settings is not None else None
        if val:
            return str(val)
        raw = os.environ.get(env)
        return raw or None

    root_s = _get("pfm_root", "DRUGAGENT_PFM_ROOT")
    # Explicit configuration wins; otherwise fall back to where `make pfm-clone` puts it.
    root = Path(root_s).expanduser().resolve() if root_s else _default_pfm_root()

    ckpt_s = _get("pfm_checkpoint", "DRUGAGENT_PFM_CHECKPOINT")
    train_s = _get("pfm_training_yml", "DRUGAGENT_PFM_TRAINING_YML")
    xpre_s = _get("pfm_x_predictor", "DRUGAGENT_PFM_X_PREDICTOR")
    hpre_s = _get("pfm_h_predictor", "DRUGAGENT_PFM_H_PREDICTOR")
    # Two spellings exist in the wild: the docs and .env.example say
    # DRUGAGENT_PFM_POCKET, while pydantic derives DRUGAGENT_PFM_POCKET_PDB from the
    # settings field name. Accept both rather than silently honouring one.
    pocket_s = _get("pfm_pocket_pdb", "DRUGAGENT_PFM_POCKET") or os.environ.get("DRUGAGENT_PFM_POCKET_PDB")
    # The target this deployment is actually working on. The pocket path above is
    # operator configuration and nothing used to check it against the brief: two
    # recorded ZAK runs were generated, scored and ACCEPTed against a soluble
    # epoxide hydrolase pocket because of exactly that gap.
    manifest_s = _get("target_manifest", "DRUGAGENT_TARGET_MANIFEST")

    def _shipped_predictor(explicit: str | None, root: Path | None, name: str) -> Path | None:
        if explicit:
            return Path(explicit).expanduser().resolve()
        cand = (root / "model_params" / name) if root else None
        return cand if (cand and cand.is_file()) else None

    def _under_root(explicit: str | None, default_rel: str) -> Path | None:
        if explicit:
            return Path(explicit).expanduser().resolve()
        return (root / default_rel) if root else None

    return PFMPaths(
        root=root,
        python=_get("pfm_python", "DRUGAGENT_PFM_PYTHON") or _default_pfm_python(),
        checkpoint=_under_root(ckpt_s, "model_params/pfm.pth"),
        training_yml=_under_root(train_s, "model_params/training.yml"),
        # PFM's training.yml points at these, but its shipped copy hard-codes the
        # authors' absolute paths. The checkout ships the weights next to pfm.pth, so
        # prefer those when present — otherwise discovering a root would only move the
        # failure from "not installed" to a puzzling missing-file error at sample time.
        x_predictor=_shipped_predictor(xpre_s, root, "x_predictor.pth"),
        h_predictor=_shipped_predictor(hpre_s, root, "h_predictor.pth"),
        device=_get("pfm_device", "DRUGAGENT_PFM_DEVICE") or "cuda:0",
        pocket_pdb=Path(pocket_s).expanduser().resolve() if pocket_s else None,
        driver=_default_driver(),
        target_manifest=Path(manifest_s).expanduser().resolve() if manifest_s else None,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class PFMGeneratorAdapter(ToolAdapter):
    """Pocket-conditioned de-novo generation. PFM or nothing — never a stand-in."""

    capability_id = "molecule.generate.pfm"
    #: PFM predates `GenerativeToolAdapter` and does not inherit from it, so it
    #: states the same three facts here. Without them it is the one installed
    #: generator that describes nothing about itself, and the master agent's
    #: catalogue would show a blank row for the very model it defaults to.
    conditioning = "pocket"
    synthesis_aware = False
    required_inputs: tuple[str, ...] = ("pocket_pdb",)

    def __init__(
        self,
        backend: str = "auto",
        *,
        settings: Any = None,
        paths: PFMPaths | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.paths = paths or resolve_pfm_paths(settings)
        self._missing = self.paths.missing()
        requested = (backend or "auto").lower()
        if requested not in {"auto", "pfm"}:
            # 'mock' used to be accepted here and is the one an old config will
            # carry, so name it rather than letting it read as a typo.
            raise ValueError(
                f"unknown generator backend {backend!r}; the only backend is 'pfm' "
                f"('auto' selects it). The BRICS surrogate was removed — a deployment "
                f"without PFM now generates nothing instead of something fake."
            )
        # There is nothing to choose between: 'auto' and 'pfm' are the same request.
        # Whether it can actually run is `self._missing`, reported by the profile.
        self.backend = "pfm"
        self.timeout_s = float(
            timeout_s if timeout_s is not None else getattr(settings, "pfm_timeout_seconds", None) or 1800.0
        )

    # --- capability --------------------------------------------------------
    @property
    def profile(self) -> CapabilityProfile:
        real = not self._missing
        return CapabilityProfile(
            capability_id=self.capability_id,
            version=PFM_ADAPTER_VERSION,
            title="De-novo molecule generation (PFM)",
            description=(
                "Structure-conditioned 3D generative model (Perturbed Flow Matching) "
                "proposing novel ligands for a binding pocket."
            ),
            input_schema_ref="PocketSpec",
            output_schema_ref="Candidate",
            # PFM samples an SDE/ODE from a random prior, so a fixed seed pins a
            # run but the model itself is stochastic.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            # UNAVAILABLE, not MOCKED: there is nothing to fall back to. The tool
            # either runs the model or does not run.
            status=CapabilityStatus.AVAILABLE if real else CapabilityStatus.UNAVAILABLE,
            provides_mock=False,
            backend=self.backend,
            timeout_seconds=self.timeout_s,
            runtime_requirements={
                "gpu": True,
                "interpreter": self.paths.python,
                "packages": [
                    "torch",
                    "torch-geometric",
                    "torch_scatter",
                    "torch_cluster",
                    "openbabel",
                ],
                "checkpoint": str(self.paths.checkpoint) if self.paths.checkpoint else None,
                "env_var": "DRUGAGENT_PFM_ROOT",
            },
            license="PFM MIT (optional external model)",
            known_failure_modes=[
                "pocket PDB missing or not a pocket (whole protein)",
                "reconstruction failure from the generated point cloud",
                "checkpoint/architecture mismatch",
                "CUDA out of memory at large batch_size",
            ],
            validation_checks=[
                "SMILES parses under RDKit",
                "heavy-atom count within the configured window",
                "duplicates removed by canonical SMILES",
            ],
            unavailable_reason=None if real else _unavailable_reason(self._missing),
        )

    def available(self) -> bool:
        return not self._missing

    # --- generation --------------------------------------------------------
    def generate(
        self,
        *,
        n_molecules: int,
        pocket: PocketSpec | dict[str, Any] | None = None,
        context_smiles: list[str] | None = None,
        seed: int = 0,
        batch_size: int = 10,
        min_heavy_atoms: int = 10,
        max_heavy_atoms: int = 70,
        keep_invalid: bool = False,
        max_runtime_seconds: float | None = None,
    ) -> GenerationBatch:
        """Sample ``n_molecules`` ligands, de-duplicated and validity-filtered.

        ``context_smiles`` seeds the surrogate backend only; PFM conditions on the
        pocket alone and ignores it.

        ``pocket`` accepts either the typed :class:`PocketSpec` or its ``model_dump``.
        The pipeline passes the dict; direct callers reach for the contract object, and
        making that work here beats an ``AttributeError`` thrown three frames deeper.
        """
        if not 0 <= int(seed) <= PFM_MAX_SEED:
            raise GenerationError(
                f"PFM cannot sample with seed={seed}: its driver seeds numpy, which accepts only "
                f"0..{PFM_MAX_SEED} (2**32 - 1). This is refused BEFORE any sampling because the "
                f"alternative is worse than an error -- an out-of-range seed reached mid-draw kills "
                f"that chunk while earlier chunks stand, leaving a PARTIAL pool that looks like a "
                f"complete smaller one. If you are chunking a large draw, derive each chunk's seed "
                f"inside the range (for example `(base + index) % {PFM_MAX_SEED + 1}`) and record the "
                f"derivation, or make one call for the whole draw. The seed is not reduced for you: "
                f"campaign records declare a seed and reproducibility rests on it."
            )
        with measure(self.capability_id, "generate", backend=self.backend) as metered:
            return self._generate(
                metered,
                n_molecules=n_molecules,
                pocket=pocket,
                context_smiles=context_smiles,
                seed=seed,
                batch_size=batch_size,
                min_heavy_atoms=min_heavy_atoms,
                max_heavy_atoms=max_heavy_atoms,
                keep_invalid=keep_invalid,
                max_runtime_seconds=max_runtime_seconds,
            )

    def _generate(
        self,
        metered: dict[str, float],
        *,
        n_molecules: int,
        pocket: PocketSpec | dict[str, Any] | None = None,
        context_smiles: list[str] | None = None,
        seed: int = 0,
        batch_size: int = 10,
        min_heavy_atoms: int = 10,
        max_heavy_atoms: int = 70,
        keep_invalid: bool = False,
        max_runtime_seconds: float | None = None,
    ) -> GenerationBatch:
        started = time.perf_counter()
        if isinstance(pocket, PocketSpec):
            pocket = pocket.model_dump(mode="json")
        cfg = {
            "backend": self.backend,
            "n_molecules": int(n_molecules),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "min_heavy_atoms": int(min_heavy_atoms),
            "max_heavy_atoms": int(max_heavy_atoms),
            "pocket": pocket or {},
            "device": self.paths.device if self.backend == "pfm" else "cpu",
            "checkpoint": str(self.paths.checkpoint) if self.backend == "pfm" else None,
        }
        if self._missing:
            # No surrogate to fall back to, by design: producing nothing is the
            # only honest answer when the model cannot run.
            raise GenerationError(_unavailable_reason(self._missing))
        raw, warnings, errors = self._sample_pfm(
            n_molecules=n_molecules,
            pocket=pocket or {},
            seed=seed,
            batch_size=batch_size,
            timeout_s=max_runtime_seconds or self.timeout_s,
        )
        origin, backend = ORIGIN_PFM, "pfm"

        batch = GenerationBatch(
            backend=backend,
            model_id=self.capability_id,
            model_version=PFM_ADAPTER_VERSION,
            n_requested=int(n_molecules),
            n_raw=len(raw),
            warnings=warnings,
            errors=errors,
            config=cfg,
        )
        pocket_id = (pocket or {}).get("receptor_id") or (pocket or {}).get("pdb_id")
        seen: set[str] = set()
        n_valid = 0
        for idx, item in enumerate(raw):
            smiles = item.get("smiles") or ""
            rdmol = Chem.MolFromSmiles(smiles) if smiles else None
            if rdmol is None:
                if keep_invalid:
                    batch.molecules.append(
                        GeneratedMolecule(
                            smiles=smiles,
                            origin=origin,
                            sample_index=idx,
                            model_id=self.capability_id,
                            model_version=PFM_ADAPTER_VERSION,
                            backend=backend,
                            pocket_id=pocket_id,
                            sdf_block=item.get("sdf"),
                            qc_flags=["unparseable_smiles"],
                        )
                    )
                continue
            n_valid += 1
            canon = Chem.MolToSmiles(rdmol)
            if canon in seen:
                continue
            seen.add(canon)
            heavy = rdmol.GetNumHeavyAtoms()
            flags: list[str] = list(item.get("qc_flags") or [])
            if heavy < min_heavy_atoms:
                flags.append("below_min_heavy_atoms")
            if heavy > max_heavy_atoms:
                flags.append("above_max_heavy_atoms")
            if flags and not keep_invalid:
                continue
            batch.molecules.append(
                GeneratedMolecule(
                    smiles=canon,
                    origin=origin,
                    sample_index=idx,
                    model_id=self.capability_id,
                    model_version=PFM_ADAPTER_VERSION,
                    backend=backend,
                    pocket_id=pocket_id,
                    n_heavy_atoms=heavy,
                    sdf_block=item.get("sdf"),
                    qc_flags=flags,
                )
            )
        batch.n_valid = n_valid
        batch.n_unique = len(seen)
        batch.molecules = batch.molecules[: int(n_molecules)]
        batch.runtime_seconds = time.perf_counter() - started
        # De-novo generation is the most expensive and least predictable tool in
        # the pool, so its own quality numbers are the ones worth carrying across
        # runs: a validity or uniqueness collapse is how a broken checkpoint or a
        # wrong pocket announces itself.
        metrics = batch.metrics()
        metered["n_kept"] = float(len(batch.molecules))
        metered["n_requested"] = float(n_molecules)
        for key in ("generation_validity", "generation_uniqueness"):
            value = metrics.get(key)
            if value is not None:
                metered[key] = float(value)
        # Always 0 now that the surrogate is gone, and kept for exactly that reason:
        # journals written before its removal carry 1.0 here, and dropping the key
        # would make old and new ledgers silently incomparable rather than visibly
        # different. `n_kept` is what proves a run actually produced something.
        metered["is_surrogate"] = 0.0
        if errors:
            note(self.capability_id, "generate", errors[0][:200])
        return batch

    def _require_pocket_matches_target(self, pocket_pdb: str) -> None:
        """Refuse to generate against a pocket that is not this target's.

        The defect this closes is recorded: two runs in ``running-history/`` were
        handed ``.external/pockets/seh_3i28_pocket10.pdb`` — soluble epoxide
        hydrolase, PDB 3I28 — while the brief named ZAK and 6JUU. Nothing
        compared the two, so the platform generated molecules for the wrong
        protein, scored them, and reported ACCEPT. The computation was fine; it
        answered a different question.

        The check runs only when the deployment declares a target manifest
        (``DRUGAGENT_TARGET_MANIFEST`` or the ``target_manifest`` setting).
        Without one there is nothing to compare against, and an undeclared target
        is reported as unchecked rather than treated as agreement.
        """
        manifest_path = self.paths.target_manifest
        if manifest_path is None:
            return
        from ..selfevolve.target.data import pocket_matches_target

        try:
            manifest = json.loads(Path(manifest_path).read_text())
        except (OSError, ValueError) as exc:
            raise GenerationError(
                f"target manifest {manifest_path} is declared but unreadable ({exc!r}); "
                "refusing to generate against an unverifiable target identity"
            ) from exc
        verdict = pocket_matches_target(pocket_pdb, manifest)
        if not verdict["match"]:
            raise GenerationError(
                f"pocket {pocket_pdb} does not belong to target "
                f"{manifest.get('alias')!r} ({manifest.get('uniprot_accession')}, "
                f"{manifest.get('structure', {}).get('pdb_id')}): {verdict['reason']}, "
                f"residue identity {verdict['identity']:.3f} over "
                f"{verdict['n_residues_checked']} residues "
                f"(threshold {verdict['threshold']}). Generating here would produce "
                "molecules for a different protein under this target's name. "
                "Cut a pocket from the target structure with "
                "`python scripts/prepare_pocket.py <target>.pdb --ligand <code> -o pocket.pdb`."
            )

    # --- real backend ------------------------------------------------------
    @staticmethod
    def _terminate_pfm_group(proc: subprocess.Popen) -> tuple[str, str]:
        """SIGTERM the child's process group, then SIGKILL what is left, and drain its pipes.

        Returns whatever the child had written, so the caller can still report its tail. Every step
        is suppressed individually: this runs on the failure path, and an error here would mask the
        exception that brought us into it.

        The grace period is :data:`_PFM_SIGTERM_GRACE_SECONDS` and is deliberately NOT a parameter:
        the caller used to pass the whole sampling timeout, which read at the call site as if that
        were the grace period while the body ignored it.
        """
        import signal

        streams: tuple[str, str] = ("", "")
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            streams = proc.communicate(timeout=_PFM_SIGTERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                streams = proc.communicate(timeout=_PFM_SIGTERM_GRACE_SECONDS)
        except Exception:  # pragma: no cover - the pipes may already be gone
            pass
        return streams

    def _sample_pfm(
        self, *, n_molecules: int, pocket: dict[str, Any], seed: int, batch_size: int, timeout_s: float
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Run PFM in a subprocess (``sys.executable`` unless overridden) and parse it."""
        # The brief's pocket wins; the operator-configured default fills in for
        # briefs (like the shipped fixture) that cannot hardcode a local path.
        pocket_pdb = pocket.get("pdb_path") or (str(self.paths.pocket_pdb) if self.paths.pocket_pdb else None)
        if not pocket_pdb or not Path(pocket_pdb).is_file():
            raise GenerationError(
                f"PFM requires a prepared pocket PDB; generation.pocket.pdb_path={pocket_pdb!r} "
                "is missing or unreadable. Cut one with "
                "`python scripts/prepare_pocket.py <structure>.pdb --ligand <code> -o pocket.pdb`, "
                "then set it on the brief or in DRUGAGENT_PFM_POCKET."
            )
        self._require_pocket_matches_target(pocket_pdb)
        workdir = Path(tempfile.mkdtemp(prefix="drugagent_pfm_"))
        try:
            request = {
                "pocket_pdb": str(Path(pocket_pdb).resolve()),
                "n_molecules": int(n_molecules),
                "batch_size": int(batch_size),
                "seed": int(seed),
                "device": self.paths.device,
                "checkpoint": str(self.paths.checkpoint),
                "training_yml": str(self.paths.training_yml),
                "x_predictor": str(self.paths.x_predictor) if self.paths.x_predictor else None,
                "h_predictor": str(self.paths.h_predictor) if self.paths.h_predictor else None,
                "out_dir": str(workdir / "out"),
            }
            req_path = workdir / "request.json"
            req_path.write_text(json.dumps(request, indent=2))
            result_path = workdir / "out" / "result.json"

            root = str(self.paths.root)
            env = dict(os.environ)
            # PFM's modules import both from the repo root (utils/, datasets/,
            # egnn/) and from scripts/ (sampling, prior_distributions, ...).
            env["PYTHONPATH"] = os.pathsep.join(
                [root, str(Path(root) / "scripts"), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            env.setdefault("PYTHONHASHSEED", str(seed))

            # --- why this is not subprocess.run(timeout=...) -------------------------------------
            # It used to be, and that cost a machine. `subprocess.run` implements its timeout by
            # SIGKILLing the direct child, and SIGKILL to a process that is inside a CUDA call is
            # precisely what strands its context: the main thread dies, a driver thread does not, and
            # the task becomes an UNREAPABLE zombie -- state Z with live threads -- that systemd
            # cannot collect. Its GPU allocation is never released. Measured on this host on
            # 2026-09-27: 43 such zombies had accumulated over about 15 hours, holding 97,245 MiB of
            # 97,887 on the GPU and pinning tens of GB of host RAM in the driver, until the kernel OOM
            # killer reaped the user session slice and took every tmux session with it. No signal
            # recovers that state; only a GPU reset or a reboot does.
            #
            # So two things change. The child leads its OWN PROCESS GROUP, so the whole group can be
            # signalled rather than just the direct child -- the driver's helpers and any grandchild
            # the sampler spawned included. And termination ESCALATES: SIGTERM first, which lets
            # Python run its exit handlers and tear the CUDA context down properly, and only then
            # SIGKILL. That ordering is the fix; the process group is what makes it reach everything.
            proc = subprocess.Popen(  # fixed argv; paths come from operator config, not user input
                [self.paths.python, str(self.paths.driver), str(req_path)],
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except BaseException as exc:
                # ANY interruption must clean up, not just the timeout. `subprocess.run` used Popen as
                # a context manager, which killed and reaped on any exception; switching to a bare
                # Popen for the process-group fix quietly dropped that, and only TimeoutExpired was
                # handled. A KeyboardInterrupt, the worker's own deadline, or an OOM then left the
                # sampler running -- and because start_new_session=True puts it in a FRESH session it
                # is reached neither by Ctrl-C nor by killing the campaign worker's process group. So
                # the exact CUDA context this change exists to protect would be stranded by every
                # interruption that is not a timeout. BaseException deliberately, to cover
                # KeyboardInterrupt and SystemExit.
                stdout, stderr = self._terminate_pfm_group(proc)
                if not isinstance(exc, subprocess.TimeoutExpired):
                    raise
                raise GenerationError(
                    f"PFM sampling exceeded {timeout_s:.0f}s and its process group was terminated "
                    f"(SIGTERM, then SIGKILL after {_PFM_SIGTERM_GRACE_SECONDS:.0f}s). Lower "
                    f"generation.n_molecules or raise generation.max_runtime_seconds. If the GPU "
                    f"still shows memory held by a defunct process afterwards, that context is "
                    f"stranded and only a GPU reset or a reboot releases it."
                ) from None

            if proc.returncode != 0 or not result_path.is_file():
                tail = (stderr or stdout or "").strip().splitlines()[-20:]
                raise GenerationError(f"PFM sampling failed (exit {proc.returncode}):\n" + "\n".join(tail))
            payload = json.loads(result_path.read_text())
            molecules = payload.get("molecules", [])
            warnings = list(payload.get("warnings", []))
            if payload.get("n_reconstruction_failures"):
                warnings.append(
                    f"{payload['n_reconstruction_failures']} generated point clouds could not be "
                    "reconstructed into a valid molecule (expected for a fraction of samples)"
                )
            return molecules, warnings, list(payload.get("errors", []))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _unavailable_reason(missing: list[str]) -> str:
    """Why de-novo generation cannot run here — the whole reason, not a summary.

    This string is what an operator sees in ``drugagent tools``, in the capability
    profile, and in the error raised at call time, so it has to be actionable on
    its own: there is no surrogate quietly covering for it any more.
    """
    detail = "; ".join(missing) if missing else "unknown"
    return (
        "PFM is not usable here, so de-novo generation is UNAVAILABLE and produces "
        "nothing — there is no surrogate backend. Install it with `make pfm-clone && "
        f"make pfm-install`, or drop `pfm` from the brief's candidate sources. Reason: {detail}."
    )
