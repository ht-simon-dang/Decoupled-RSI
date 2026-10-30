"""Property models: what a molecule will do to a body, predicted rather than measured.

Three upstream projects live here and they are not interchangeable, though a
scorecard would happily print all three in the same column.

**ADMET-AI** is a fixed panel. It ships trained models for a published set of
absorption, distribution, metabolism, excretion and toxicity endpoints and predicts
all of them for any molecule you hand it. Nothing is fitted at call time, which is
its strength — the numbers are reproducible and the training data is known — and
also its limit: the endpoints are the ones the authors chose, and a molecule far
from their training distribution gets a confident number anyway.

**Chemprop** is the opposite. It learns *your* endpoint from *your* labels and
predicts nothing until it has been trained. It is the right tool once real assay
data exists and the wrong one before that, and it should be compared against a
cheap fingerprint baseline before anyone concludes the message-passing network
earned its cost.

**Uni-Mol** is neither. It is a 3D molecular representation that property models can
be built *on*, not a pretrained ADMET oracle. Treating its embeddings as
predictions would be a category error, so this package exposes representation and
fine-tuned prediction as separate operations with separate names.

:class:`PropertyToolAdapter` owns what the three share: taking a list of SMILES,
dropping the ones RDKit cannot parse *before* they reach a model that would happily
return a number for them, and assembling a :class:`PropertyPanel` that keeps every
endpoint next to the model that produced it.

**The applicability problem is not solved here.** Every one of these models will
return a number for a molecule nothing like its training set. None of them is
calibrated on this project's chemistry. So a panel carries ``limitation`` text and
the endpoints carry their model id and version, and the specialist agent that turns
a panel into a :class:`~drug_agent.contracts.predictions.Prediction` is responsible
for setting applicability. A property adapter never claims a domain it has not
checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, RDLogger

from .._ext.adapter import ExternalToolAdapter

RDLogger.DisableLog("rdApp.*")


class PropertyModelError(RuntimeError):
    """The selected property backend ran and failed.

    Distinct from :class:`~drug_agent.tools.base.ToolUnavailable`: the model was
    installed, inference started, and it did not return usable values.
    """


@dataclass
class PropertyPanel:
    """Predicted endpoints for one molecule, with the model that produced them."""

    candidate_id: str = ""
    smiles: str = ""
    #: Endpoint name to predicted value. Names are the upstream model's own, kept
    #: verbatim so a value can always be traced to a published definition.
    endpoints: dict[str, float] = field(default_factory=dict)
    #: Physicochemical descriptors the backend computed alongside the predictions.
    #: Separated from ``endpoints`` because these are calculated, not predicted.
    descriptors: dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    model_version: str = "0.0.0"
    qc_flags: list[str] = field(default_factory=list)
    limitation: str = ""

    @property
    def ok(self) -> bool:
        """Whether anything was actually predicted for this molecule."""
        return bool(self.endpoints) and "unparseable_smiles" not in self.qc_flags


class PropertyToolAdapter(ExternalToolAdapter):
    """Shared input screening and panel assembly for the property backends."""

    produces_nothing = "predicts nothing"
    #: What this backend is honestly for. Used in the card's description and to keep
    #: a representation model from being read as a property oracle.
    modality: str = "predicted property panel"

    @staticmethod
    def screen(smiles_list: list[str]) -> tuple[list[str], dict[int, str]]:
        """Split *smiles_list* into parseable inputs and per-index rejection reasons.

        A model handed an unparseable SMILES will not always refuse it — some
        featurizers produce an all-zero vector and the network returns a perfectly
        confident number for nothing at all. Screening here means such a molecule
        gets a ``qc_flag`` instead of a prediction.
        """
        keep: list[str] = []
        rejected: dict[int, str] = {}
        for idx, smi in enumerate(smiles_list):
            if not smi or Chem.MolFromSmiles(smi) is None:
                rejected[idx] = "unparseable_smiles"
            else:
                keep.append(smi)
        return keep, rejected

    def empty_panel(self, candidate_id: str, smiles: str, reason: str) -> PropertyPanel:
        """A panel that predicted nothing, and says why.

        Not an exception: one molecule the featurizer cannot handle is a fact about
        that molecule, and the rest of the batch is still valid.
        """
        return PropertyPanel(
            candidate_id=candidate_id,
            smiles=smiles,
            model_id=self.capability_id,
            model_version=self.version,
            qc_flags=[reason],
        )

    @staticmethod
    def coverage(panels: list[PropertyPanel]) -> dict[str, float]:
        """Ledger metrics: how much of the batch actually got predictions."""
        if not panels:
            return {}
        ok = [p for p in panels if p.ok]
        endpoints = float(sum(len(p.endpoints) for p in ok) / len(ok)) if ok else 0.0
        return {
            "property_n_molecules": float(len(panels)),
            "property_coverage": float(len(ok) / len(panels)),
            "property_endpoints_per_molecule": endpoints,
        }

    @staticmethod
    def numeric(value: Any) -> float | None:
        """Coerce an upstream value to a finite float, or ``None``.

        Upstream frames mix numpy scalars, Python floats, strings and NaN. A NaN
        that reaches the ledger is silently dropped by the metering layer, and a NaN
        that reaches a scorecard reads as a missing measurement rather than a failed
        one, so it is filtered at the boundary instead.
        """
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if out == out and abs(out) != float("inf") else None


__all__ = ["PropertyModelError", "PropertyPanel", "PropertyToolAdapter"]
