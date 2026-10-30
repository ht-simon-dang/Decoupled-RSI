"""Single-cell perturbation — real differential expression, no invented drug effect.

Two different things live behind this capability, and only one of them was ever
a computation.

**Real, and kept.** :meth:`SingleCellAdapter.disease_signature` computes a
genuine differential-expression signature from the fixture expression matrix:
per-gene mean difference between disease and healthy cells, scaled by pooled
standard deviation, ranked by absolute effect. :meth:`qc` and :meth:`donor_split`
are likewise real. The dataset is a synthetic fixture — that belongs in
``license``, not in a status flag — but the statistics over it are honest.

**Fabricated, and removed.** ``score_candidate`` claimed to say how a molecule
perturbs the transcriptome. It had no model of that. It built the molecule's
per-gene effect as::

    pert[g] = (fingerprint[hash(g)] * 2 - 1) * (1 + (hash(g + smiles)/100 - 0.5))

— a hash of the gene name and the SMILES — then reported the cosine against the
real disease signature as ``restoration_score``. The signature was real, which
is exactly what made the product look like a measurement. Nothing in that
pipeline observed a molecule doing anything to a cell.

That matters more than the docking mock did, because ``restoration_score`` is
the **primary objective of the phenotype brief** and a key the LLM planner can
write success criteria against. Under the make-it-or-fail rule
(:mod:`drug_agent.tools.base`) it is gone: :meth:`score_candidate` raises, the
capability reports ``UNAVAILABLE``, and phenotype programs fail loudly instead of
ranking candidates on a hash.

Making it real needs a perturbation model — a trained LINCS/L1000-style
signature predictor, or measured profiles for these compounds. Neither is in
this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts.capability import CapabilityProfile
from ..contracts.enums import (
    CapabilityStatus,
    CostClass,
    Determinism,
)
from ..contracts.perturbation import (
    DifferentialGene,
    PerturbationSignature,
    QCSummary,
)
from ..util import stable_hash_int
from .base import ToolAdapter
from .rdkit_tool import Cheminformatics

_UNAVAILABLE_REASON = (
    "Scoring a candidate's effect on the transcriptome needs a perturbation-response "
    "model (LINCS/L1000-style) or measured profiles; this deployment has neither. The "
    "previous implementation derived the effect from a hash of the SMILES and is removed. "
    "The disease signature itself is real and still computed."
)


@dataclass
class SCDataset:
    X: np.ndarray  # cells x genes (log-normalized)
    gene_names: list[str]
    condition: np.ndarray  # per-cell: 'disease' | 'healthy'
    donor: np.ndarray  # per-cell donor id
    cell_context: str = "unknown"
    dataset_id: str = "fixture"


def _has_anndata() -> bool:
    try:
        import anndata  # noqa: F401

        return True
    except Exception:
        return False


class SingleCellAdapter(ToolAdapter):
    capability_id = "perturb.assess"

    def __init__(self, backend: str = "baseline") -> None:
        self.backend = backend
        self.chem = Cheminformatics()
        self._anndata = _has_anndata()

    @property
    def profile(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.capability_id,
            version="1.0.0",
            title="Single-cell perturbation",
            description=(
                "Differential-expression signature over a single-cell fixture. Candidate "
                "restoration scoring has no model and is not available."
            ),
            input_schema_ref="Molecule",
            output_schema_ref="PerturbationSignature",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.MODERATE,
            status=CapabilityStatus.UNAVAILABLE,
            provides_mock=False,
            backend="none",
            runtime_requirements={"model": "a trained perturbation-response model"},
            license="synthetic fixture; demonstration-only",
            known_failure_modes=["empty dataset", "single condition"],
            validation_checks=["restoration in [-1,1]", "donor split disjoint"],
            unavailable_reason=_UNAVAILABLE_REASON,
        )

    def available(self) -> bool:
        return False

    # --- loading -----------------------------------------------------------
    def load_fixture(self, path: str | Path) -> SCDataset:
        path = Path(path)
        if path.suffix == ".h5ad" and self._anndata:  # pragma: no cover - optional
            import anndata

            adata = anndata.read_h5ad(path)
            X = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)  # type: ignore[union-attr]
            return SCDataset(
                X=X,
                gene_names=list(adata.var_names),
                condition=np.asarray(adata.obs.get("condition", ["disease"] * X.shape[0])),
                donor=np.asarray(adata.obs.get("donor", ["d0"] * X.shape[0])),
                cell_context=str(adata.uns.get("cell_context", "unknown")),
                dataset_id=path.stem,
            )
        # npz fixture
        data = np.load(path, allow_pickle=True)
        return SCDataset(
            X=data["X"],
            gene_names=list(data["gene_names"]),
            condition=data["condition"],
            donor=data["donor"],
            cell_context=str(data["cell_context"]) if "cell_context" in data else "unknown",
            dataset_id=path.stem,
        )

    # --- QC ----------------------------------------------------------------
    def qc(self, ds: SCDataset) -> QCSummary:
        counts = ds.X.sum(axis=1)
        flags = []
        if ds.X.shape[0] < 10:
            flags.append("low_cell_count")
        if len(set(ds.condition.tolist())) < 2:
            flags.append("single_condition")
        return QCSummary(
            n_cells=int(ds.X.shape[0]),
            n_genes=int(ds.X.shape[1]),
            n_donors=len(set(ds.donor.tolist())),
            median_counts=float(np.median(counts)),
            pct_mito=None,
            flags=flags,
        )

    # --- donor-aware split -------------------------------------------------
    @staticmethod
    def donor_split(ds: SCDataset, holdout_frac: float = 0.34) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_mask, test_mask) with disjoint donors (no leakage)."""
        donors = sorted(set(ds.donor.tolist()))
        n_hold = max(1, round(len(donors) * holdout_frac))
        # Deterministic donor selection by stable hash order.
        ordered = sorted(donors, key=lambda d: stable_hash_int(str(d)))
        holdout = set(ordered[:n_hold])
        test_mask = np.array([d in holdout for d in ds.donor])
        return ~test_mask, test_mask

    # --- disease signature -------------------------------------------------
    def disease_signature(
        self, ds: SCDataset, mask: np.ndarray | None = None, top_k: int = 15
    ) -> list[DifferentialGene]:
        m = mask if mask is not None else np.ones(ds.X.shape[0], dtype=bool)
        is_disease = (ds.condition == "disease") & m
        is_healthy = (ds.condition == "healthy") & m
        if is_disease.sum() == 0 or is_healthy.sum() == 0:
            return []
        mean_d = ds.X[is_disease].mean(axis=0)
        mean_h = ds.X[is_healthy].mean(axis=0)
        std_pool = ds.X[m].std(axis=0) + 1e-6
        log2fc = mean_d - mean_h
        score = log2fc / std_pool
        order = np.argsort(-np.abs(score))[:top_k]
        genes = []
        for i in order:
            genes.append(
                DifferentialGene(
                    gene=ds.gene_names[i],
                    log2fc=round(float(log2fc[i]), 4),
                    score=round(float(score[i]), 4),
                    direction="up" if log2fc[i] > 0 else "down",
                )
            )
        return genes

    # --- candidate scoring -------------------------------------------------
    def score_candidate(
        self, candidate_id: str, smiles: str, ds: SCDataset, signature: list[DifferentialGene]
    ) -> PerturbationSignature:
        """Raise :class:`ToolUnavailable`: there is no perturbation-response model.

        See the module docstring for what the removed implementation did.
        """
        self.require_available()
        raise AssertionError("unreachable")  # pragma: no cover
