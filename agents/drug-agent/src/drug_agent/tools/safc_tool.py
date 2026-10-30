"""SAFC: a target-specific score read off an MD-derived receptor ensemble.

What the number is
------------------
SAFC does not score a molecule against *a* structure. Every candidate is docked
into each conformation of a receptor ensemble sampled from molecular dynamics,
each docked complex is encoded by the frozen official EquiScore graph encoder
into 128 dimensions, the per-pose embeddings are averaged per ligand, and a
target-specific head trained on that representation turns the one vector into a
score. On the bundle installed here that ensemble is **41 ZAK conformations**
drawn from an active-apo arm (origin 6JUT) and an inactive arm (origin 5HES),
and the head is a five-member ExtraTrees ensemble fitted with repeated balanced
bootstrap.

Nothing is fitted at call time. The encoder checkpoint, the 41 receptor
PDB/PDBQT pairs with their own box centres, and ``primary_model.joblib`` all
ship inside the bundle and are checksummed; this adapter only runs them.

Why it is its own capability and not a second docking backend
-------------------------------------------------------------
``structure.dock`` answers "how well does this pose score in this pocket" and
returns poses. This answers "how does this molecule rank for this target across
its receptor's accessible conformations", and the two are not substitutes: the
ensemble is the measurement, not an implementation detail of it. The adapter
also costs what the ensemble costs — one Vina run per ligand per conformation,
so 41 docks per molecule here — which is why ``cost_class`` is ``EXPENSIVE``
and why the driver keeps a resumable pose cache in a caller-owned directory.

The applicability domain is one protein, and it is checked
----------------------------------------------------------
A SAFC head is trained for one target. The bundle declares which one
(``targets/<id>/config/target.json``: UniProt, gene symbol, ChEMBL target), and
:meth:`SafcAdapter.score` refuses when that disagrees with the target manifest
this deployment declares. The defect this closes is the one ``pfm_tool`` already
names: a run that computes cleanly and answers a different question than the
brief asked. Scoring ZAK candidates with a head trained for another kinase would
produce numbers no reader could tell from real ones, which is exactly what
ADR-0011 forbids. An absent manifest is left unchecked rather than treated as
agreement, and a manifest that cannot be read is a refusal.

What a reader must not conclude from the score
----------------------------------------------
The bundle's own guidance is copied rather than paraphrased: *treat scores as
ranking or prioritization evidence, not calibrated binding probabilities.* Two
further limits travel on every profile here because they are properties of the
installed artifact, not opinions about it:

* the ZAK head was fitted on **67 ligands, of which only 8 are negatives**, under
  a label rule of pActivity >= 7 for actives and <= 5 for inactives, and its own
  model card states that the reported cross-validation differences carry
  substantial uncertainty. Its published metrics are internal out-of-fold
  estimates; no independent external test was retained.
* its documented controls span a narrow band — the held-out 6JUU ligand C9R at
  0.976 against paclitaxel, the deliberate non-kinase sanity control, at 0.745 —
  so the usable signal is the ordering, and a small absolute difference between
  two candidates is not evidence that one binds better.

``pose_count`` therefore travels with every score. A score averaged over twelve
surviving poses is a different measurement from one averaged over forty-one, and
a reader who cannot see which they hold cannot judge either.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import CostClass, Determinism
from ..observability.toolmeter import measure
from ._ext.adapter import ExternalToolAdapter
from ._ext.spec import ExternalSpec
from .vina_tool import OBABEL_BINARY, VINA_BINARY, find_obabel, find_vina

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..settings.config import Settings

INSTALL_HINT = "make safc-install"


def _find_obabel(settings: Any = None) -> str | None:
    """Open Babel for the *ensemble* stage, which runs outside this interpreter.

    :func:`~drug_agent.tools.vina_tool.find_obabel` looks on PATH only, and it is
    right to: for ``structure.dock`` Open Babel ships beside the platform
    interpreter, so its absence means an incomplete primary environment. SAFC is
    the case that assumption does not cover — its stages run in the bundle's own
    pinned environment, which has RDKit and DGL and no Open Babel, so the binary
    has to be found wherever it really is and handed over as an absolute path.
    Hence the same conda-root search :func:`find_vina` already performs, after
    the PATH lookup rather than instead of it.
    """
    on_path = find_obabel(settings)
    if on_path:
        return on_path
    from ._ext.probe import conda_env_pythons

    for interpreter in conda_env_pythons(("drug-agent", "safc-equiscore-tl")):
        candidate = Path(interpreter).parent / OBABEL_BINARY
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None

SPEC = ExternalSpec(
    name="SAFC",
    env_prefix="SAFC",
    checkout_relpath=".external/safc-agent-zak",
    # The bundle is a transfer archive, not a git checkout, so the sentinels are
    # the artifacts that make it the real thing: its manifest, the four pipeline
    # stages this adapter drives, the EquiScore encoder it freezes, and the
    # target's own conformation manifest. A directory that merely exists under
    # the expected path is not a SAFC bundle.
    sentinel_files=(
        "VERSION",
        "bundle_manifest.json",
        "SHA256SUMS",
        "safc/scripts/05c_build_official_equiscore_inputs.py",
        "safc/scripts/05d_extract_official_equiscore_embeddings.py",
        "safc/scripts/07_predict_official_equiscore_tl.py",
        "safc/third_party/EquiScore/model/equiscore.py",
        "targets/zak/config/target.json",
        "targets/zak/structures/conformation_manifest.csv",
    ),
    driver_name="safc_score.py",
    # dgl is the narrow probe: it is the one dependency no other environment on a
    # typical host has, it is pinned to a version that stopped being published to
    # PyPI, and EquiScore's layers import it directly. torch alone would select an
    # environment that cannot build a single graph.
    selection_modules=("dgl", "torch"),
    runtime_modules=("dgl", "torch", "rdkit", "pandas", "numpy", "sklearn", "joblib"),
    weight_relpaths=(
        "safc/third_party/EquiScore/workdir/official_weight/save_model_screen.pt",
        "targets/zak/models/official_equiscore_tl/primary_model.joblib",
    ),
    conda_envs=("safc-equiscore-tl",),
    upstream_url="https://github.com/HimansuKumarIIITA/DrugDesignAI-Benchmark",
    upstream_ref="zak-final-20260919-r2",
    license="EquiScore MIT; SAFC bundle NOTICE.md",
    install_hint=INSTALL_HINT,
    needs_gpu=False,
)


class SafcError(RuntimeError):
    """SAFC was installed, ran, and could not produce a score."""


@dataclass
class SafcScore:
    """One molecule's ensemble score, with the evidence behind it."""

    sample_id: str
    inchikey: str
    canonical_smiles: str
    #: The head's mean probability over its ensemble members. A ranking signal.
    safc_score: float
    #: How many of the ensemble's conformations produced a usable pose. The
    #: denominator of the embedding average, and never silently dropped.
    pose_count: int
    #: The frozen encoder's own pre-head output, carried for diagnosis only.
    equiscore_zero_shot: float | None = None
    #: Mean Vina affinity over the surviving poses, kcal/mol, lower better.
    vina_ensemble_mean_kcal_mol: float | None = None


@dataclass
class SafcRun:
    """Everything one scoring pass produced, including what it failed to score."""

    scores: list[SafcScore] = field(default_factory=list)
    unscored: list[dict[str, Any]] = field(default_factory=list)
    n_requested: int = 0
    n_dock_failures: int = 0
    workdir: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


class SafcAdapter(ExternalToolAdapter):
    """The SAFC ensemble score, as the installed bundle computes it."""

    capability_id = "activity.safc"
    spec = SPEC
    engine = "safc"
    version = "1.0.0"
    produces_nothing = "scores nothing"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        timeout_s: float | None = None,
        target: str = "zak",
    ) -> None:
        self.settings = settings
        self.target = target
        self._vina = find_vina(settings)
        self._obabel = _find_obabel(settings)
        super().__init__(settings=settings, timeout_s=timeout_s)
        if blockers := self._local_blockers():
            self._missing = list(self._missing) + blockers
            self.backend = "none"

    # --- readiness ---------------------------------------------------------
    def _local_blockers(self) -> list[str]:
        """Reasons the ensemble stage could not run even with the bundle present.

        The encoder and the head live in the bundle's own environment, but the
        docking stage is two executables, and a host with neither cannot produce
        a single pose for the encoder to read. Reporting them here rather than at
        call time keeps the card honest about a tool that would otherwise claim
        AVAILABLE and fail on its first molecule.
        """
        problems: list[str] = []
        if self._vina is None:
            problems.append(
                f"no {VINA_BINARY!r} executable on PATH, which is what docks each "
                f"candidate into the receptor ensemble; `conda install -c conda-forge vina`"
            )
        if self._obabel is None:
            problems.append(
                f"no {OBABEL_BINARY!r} executable on PATH, which is what converts an "
                f"embedded ligand into the PDBQT vina requires; `pip install openbabel-wheel`"
            )
        return problems

    @property
    def unavailable_reason(self) -> str | None:
        """The whole reason, including blockers the shared probe cannot see.

        The base class reports the bundle, its weights and its interpreter. The
        two docking executables are this adapter's own requirement, and a reason
        that omitted them would send an operator to reinstall a bundle that is
        already checksummed and intact.
        """
        if not self._missing:
            return None
        return (
            f"{self.spec.name} is not usable here, so this capability is UNAVAILABLE and "
            f"{self.produces_nothing} — there is no surrogate backend. "
            f"Install it with `{INSTALL_HINT}`. Reason: {'; '.join(self._missing)}."
        )

    # --- profile -----------------------------------------------------------
    @property
    def profile(self) -> CapabilityProfile:
        return self.build_profile(
            title="Ensemble activity score over an MD receptor ensemble (SAFC)",
            description=(
                "Docks each candidate into every conformation of the bundle's receptor "
                "ensemble, encodes each docked complex with the frozen official EquiScore "
                "graph encoder, averages the 128-dimensional per-pose embeddings per "
                "ligand, and applies the bundle's trained target head. Returns a ranking "
                "score with the pose count behind it; fits nothing at call time."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="SafcScore",
            # Vina's search is stochastic, so the poses the embedding is averaged
            # over vary between runs at a fixed seed only as far as vina does. The
            # encoder and the head are deterministic given those poses.
            determinism=Determinism.STOCHASTIC,
            cost_class=CostClass.EXPENSIVE,
            known_failure_modes=[
                "one Vina run per ligand per conformation: 41 docks per molecule on the "
                "installed ZAK ensemble, and large flexible ligands cost an order of "
                "magnitude more per dock than rigid ones",
                "a ligand RDKit cannot embed, or Open Babel cannot convert, is never docked "
                "and is returned in 'unscored' rather than scored from fewer poses",
                "a pose the complex builder or the encoder rejects lowers pose_count, so a "
                "score can rest on fewer conformations than the ensemble holds",
                "the installed ZAK head was fitted on 67 ligands with 8 negatives; its "
                "published metrics are out-of-fold estimates and no external test was kept",
                "scores occupy a narrow band (documented controls 0.745 to 0.976), so small "
                "absolute differences are not evidence of a binding difference",
                "a head trained for another target refuses rather than scoring, so a bundle "
                "whose declared target disagrees with this deployment's manifest is a refusal",
            ],
            validation_checks=[
                "the bundle's declared target matches the deployment's target manifest",
                "every score carries the pose count its embedding was averaged over",
                "a molecule that was not scored appears in 'unscored' with its reason",
                "the encoder checkpoint sha256 recorded by the run matches the bundle manifest",
            ],
        )

    # --- applicability domain ---------------------------------------------
    def _bundle_target(self) -> dict[str, Any]:
        """The identity the installed bundle declares for its own head."""
        if self.ext.root is None:
            return {}
        config = self.ext.root / "targets" / self.target / "config" / "target.json"
        if not config.is_file():
            return {}
        try:
            return dict((json.loads(config.read_text()).get("target") or {}))
        except (OSError, json.JSONDecodeError) as exc:
            raise SafcError(
                f"the bundle's target declaration at {config} is unreadable ({exc}); "
                "an unreadable applicability domain is never a pass"
            ) from exc

    def _require_target_matches(self) -> dict[str, Any]:
        """Refuse when this head was trained for a different protein.

        A SAFC head is one target's model. Handed candidates for another, every
        stage still computes — the ligands dock, the encoder encodes, the head
        returns probabilities — and the numbers mean nothing, while looking
        exactly like the numbers that do. That is the failure ``pfm_tool`` was
        given a pocket-identity gate to stop, and this is the same gate at the
        level SAFC is scoped to: the protein the bundle names.

        A deployment that declares no target manifest is left unchecked. An
        absent check is not agreement, and it is reported as ``unchecked`` in the
        provenance rather than being quietly read as a match.
        """
        declared = self._bundle_target()
        manifest_path = getattr(self.settings, "target_manifest", None)
        if manifest_path is None:
            return {"status": "unchecked", "bundle_target": declared}
        try:
            manifest = json.loads(Path(str(manifest_path)).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SafcError(
                f"target manifest {manifest_path} is unreadable ({exc}); refusing to score, "
                "because a manifest nobody could read cannot agree with anything"
            ) from exc

        for field_name, manifest_key in (
            ("uniprot", "uniprot_accession"),
            ("chembl_target", "chembl_target_id"),
        ):
            mine = declared.get(field_name)
            theirs = manifest.get(manifest_key)
            if mine and theirs and str(mine) != str(theirs):
                raise SafcError(
                    f"this SAFC head is trained for {field_name} {mine!r} and this deployment "
                    f"declares {theirs!r} ({manifest.get('alias') or 'unnamed target'}). "
                    "Scoring here would rank candidates for a different protein under this "
                    "target's name, so it refuses instead."
                )
        return {
            "status": "matched",
            "bundle_target": declared,
            "manifest_alias": manifest.get("alias"),
            # A built target_manifest.json spells it `manifest_hash`; the registry's
            # own records spell the same value `target_manifest_hash`. Reading one
            # name only is how this field silently became null in the provenance of
            # a run whose domain really had been checked.
            "manifest_hash": manifest.get("manifest_hash")
            or manifest.get("target_manifest_hash"),
        }

    # --- operations --------------------------------------------------------
    def score(
        self,
        molecules: list[tuple[str, str]] | list[dict[str, str]],
        *,
        workdir: str | Path,
        jobs: int = 10,
        vina_cpu: int = 2,
        exhaustiveness: int = 8,
        num_modes: int = 3,
        seed: int = 42,
        dock_timeout_seconds: float = 900.0,
        batch_size: int = 8,
        threads: int = 8,
        device: str = "cpu",
        pocket_cutoff: float = 8.0,
        timeout_s: float | None = None,
    ) -> SafcRun:
        """Score *molecules* against the installed ensemble.

        *workdir* is a directory the caller owns and keeps. It holds the pose
        cache, and it is what makes an interrupted pass resumable: a dock whose
        pose is already there is not repeated. It is deliberately not the
        driver's ``out_dir``, which the external-tool helper deletes on return.
        """
        self.require_available()
        domain = self._require_target_matches()

        payload: list[dict[str, str]] = []
        for item in molecules:
            if isinstance(item, dict):
                sample_id = str(item.get("sample_id") or item.get("molecule_id") or "")
                smiles = str(item.get("smiles") or item.get("canonical_smiles") or "")
            else:
                sample_id, smiles = str(item[0]), str(item[1])
            if not sample_id or not smiles:
                raise SafcError(f"a molecule arrived without an id or a SMILES: {item!r}")
            payload.append({"sample_id": sample_id, "smiles": smiles})
        if not payload:
            raise SafcError("no molecules were given to score")

        workdir = Path(workdir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        with measure(self.capability_id, "score", backend=self.backend) as metered:
            metered["safc_n_molecules"] = float(len(payload))
            run = self.run_driver(
                {
                    "task": "score",
                    "target": self.target,
                    "molecules": payload,
                    "workdir": str(workdir),
                    "vina": self._vina or VINA_BINARY,
                    "obabel": self._obabel or OBABEL_BINARY,
                    "jobs": jobs,
                    "vina_cpu": vina_cpu,
                    "exhaustiveness": exhaustiveness,
                    "num_modes": num_modes,
                    "seed": seed,
                    "dock_timeout_seconds": dock_timeout_seconds,
                    "batch_size": batch_size,
                    "threads": threads,
                    "device": device,
                    "pocket_cutoff": pocket_cutoff,
                },
                error_cls=SafcError,
                timeout_s=timeout_s,
            )
            # Recorded inside the block so the journal carries what the pass
            # actually achieved, not just what it was asked for. A pass that
            # scored 90 of 150 molecules is the interesting case, and the
            # difference is invisible from the request alone.
            metered["safc_n_scored"] = float(len(run.payload.get("scores") or []))
            metered["safc_n_dock_failures"] = float(run.payload.get("n_dock_failures") or 0)

        parsed = run.payload
        scores = [
            SafcScore(
                sample_id=str(row["sample_id"]),
                inchikey=str(row.get("inchikey", "")),
                canonical_smiles=str(row.get("canonical_smiles", "")),
                safc_score=float(row["safc_score"]),
                pose_count=int(row.get("pose_count") or 0),
                equiscore_zero_shot=(
                    float(row["equiscore_zero_shot"])
                    if row.get("equiscore_zero_shot") is not None
                    else None
                ),
                vina_ensemble_mean_kcal_mol=(
                    float(row["vina_ensemble_mean_kcal_mol"])
                    if row.get("vina_ensemble_mean_kcal_mol") is not None
                    else None
                ),
            )
            for row in parsed.get("scores") or []
        ]
        provenance = dict(parsed.get("provenance") or {})
        provenance["applicability_domain"] = domain
        return SafcRun(
            scores=scores,
            unscored=list(parsed.get("unscored") or []),
            n_requested=int(parsed.get("n_requested") or len(payload)),
            n_dock_failures=int(parsed.get("n_dock_failures") or 0),
            workdir=str(parsed.get("workdir") or workdir),
            provenance=provenance,
        )

    def describe_ensemble(self) -> dict[str, Any]:
        """The receptor ensemble this bundle scores against, read off its manifest.

        Cheap and offline: it opens two files the bundle ships. It exists because
        "how many conformations, from which states" is the first question a reader
        of a SAFC score asks, and answering it should not require a scoring run.
        """
        self.require_available()
        assert self.ext.root is not None  # require_available() proved the checkout
        target_root = self.ext.root / "targets" / self.target
        config = json.loads((target_root / "config" / "target.json").read_text())
        manifest = target_root / "structures" / "conformation_manifest.csv"

        import csv

        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        arms: dict[str, int] = {}
        for row in rows:
            arms[row["state_arm"]] = arms.get(row["state_arm"], 0) + 1
        structures = config.get("structures") or {}
        return {
            "target": config.get("target") or {},
            "n_conformations": len(rows),
            "declared_n_conformations": structures.get("n_conformations"),
            "state_arms": arms,
            "active_apo_origin": structures.get("active_apo_origin"),
            "inactive_origin": structures.get("inactive_origin"),
            "reference_ligand": structures.get("reference_ligand_resname"),
            "label_rule": config.get("task") or {},
            "representation": (config.get("model") or {}).get("representation"),
        }


__all__ = ["SPEC", "SafcAdapter", "SafcError", "SafcRun", "SafcScore"]
