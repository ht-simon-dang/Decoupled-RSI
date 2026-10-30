"""scikit-learn — the baseline activity model (``model.activity``).

Named for its backend like the rest of the pool, with one caveat worth stating:
``safety_tool`` also uses scikit-learn, for its calibrated toxicity endpoints.
This file is the *activity* model specifically.

A deliberately simple, honest baseline (§12.1 baseline-first): Morgan-fingerprint
features into a RandomForest. Uncertainty is the ensemble standard deviation
across trees; applicability domain is nearest-neighbour Tanimoto similarity to
the training set. Everything here is a CALCULATED/PREDICTED value — never sourced
from an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..contracts.enums import ApplicabilityDomain
from ..observability.toolmeter import measure
from .rdkit_tool import Cheminformatics


@dataclass
class PredictionOutput:
    value: float
    std: float
    applicability: ApplicabilityDomain
    applicability_score: float
    in_domain_similarity: float


@dataclass
class BaselineActivityModel:
    """RandomForest on Morgan fingerprints. Reports ensemble-std uncertainty."""

    #: Stable id in the tool pool. This is not a ``ToolAdapter`` — it is fitted
    #: per run from the run's own assay snapshot rather than being a standing
    #: service — but it is a tool by every other measure, so it carries an id and
    #: is metered like one.
    capability_id: ClassVar[str] = "model.activity"

    endpoint: str = "pIC50"
    n_estimators: int = 200
    max_depth: int | None = 12
    seed: int = 1729
    ad_threshold_in: float = 0.35
    ad_threshold_border: float = 0.2
    model_version: str = "baseline-rf-1.0.0"

    chem: Cheminformatics = field(default_factory=Cheminformatics)
    _model: RandomForestRegressor | None = field(default=None, init=False)
    _train_fps: np.ndarray | None = field(default=None, init=False)
    _y_mean: float = field(default=0.0, init=False)
    _y_std: float = field(default=1.0, init=False)
    _fitted: bool = field(default=False, init=False)

    def _featurize(self, smiles_list: list[str]) -> np.ndarray:
        return np.vstack([self.chem.fingerprint(s) for s in smiles_list]).astype(np.float32)

    def fit(self, smiles_list: list[str], y: list[float]) -> BaselineActivityModel:
        with measure(self.capability_id, "fit", backend="sklearn-rf") as metered:
            metered["n_train"] = float(len(smiles_list))
            return self._fit(smiles_list, y)

    def _fit(self, smiles_list: list[str], y: list[float]) -> BaselineActivityModel:
        X = self._featurize(smiles_list)
        y_arr = np.asarray(y, dtype=np.float64)
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std() or 1.0)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=1,
        )
        self._model.fit(X, y_arr)
        self._train_fps = X.astype(np.uint8)
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _nn_similarity(self, fp: np.ndarray) -> float:
        if self._train_fps is None or len(self._train_fps) == 0:
            return 0.0
        fp8 = fp.astype(np.uint8)
        inter = np.bitwise_and(self._train_fps, fp8).sum(axis=1).astype(np.float64)
        union = np.bitwise_or(self._train_fps, fp8).sum(axis=1).astype(np.float64)
        sims = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        return float(sims.max()) if len(sims) else 0.0

    def _domain(self, sim: float) -> ApplicabilityDomain:
        if sim >= self.ad_threshold_in:
            return ApplicabilityDomain.IN_DOMAIN
        if sim >= self.ad_threshold_border:
            return ApplicabilityDomain.BORDERLINE
        return ApplicabilityDomain.OUT_OF_DOMAIN

    def predict_one(self, smiles: str) -> PredictionOutput:
        """Predict for one molecule, with its applicability domain.

        Metered here rather than in :meth:`predict_many` because this is the call
        the pipeline actually makes — once per candidate. Aggregating the
        per-call ``in_domain`` flag gives the in-domain *fraction* for the run,
        which is the number that decides whether a predicted potency means
        anything at all.
        """
        with measure(self.capability_id, "predict_one", backend="sklearn-rf") as metered:
            out = self._predict_one(smiles)
            metered["in_domain"] = 1.0 if str(out.applicability) == "IN_DOMAIN" else 0.0
            metered["uncertainty"] = float(out.std)
        return out

    def _predict_one(self, smiles: str) -> PredictionOutput:
        if not self._fitted or self._model is None:
            raise RuntimeError("model not fitted")
        fp = self.chem.fingerprint(smiles)
        X = fp.astype(np.float32).reshape(1, -1)
        tree_preds = np.array([est.predict(X)[0] for est in self._model.estimators_])
        value = float(tree_preds.mean())
        std = float(tree_preds.std())
        sim = self._nn_similarity(fp)
        domain = self._domain(sim)
        # Out-of-domain predictions carry inflated uncertainty.
        if domain == ApplicabilityDomain.OUT_OF_DOMAIN:
            std = max(std, self._y_std)
        return PredictionOutput(
            value=round(value, 4),
            std=round(std, 4),
            applicability=domain,
            applicability_score=round(sim, 4),
            in_domain_similarity=round(sim, 4),
        )

    def predict_many(self, smiles_list: list[str]) -> list[PredictionOutput]:
        """Predict for a batch. Metering happens per molecule in ``predict_one``."""
        return [self.predict_one(s) for s in smiles_list]
