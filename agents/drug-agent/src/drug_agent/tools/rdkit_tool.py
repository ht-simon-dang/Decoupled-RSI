"""RDKit — standardization, descriptors, fingerprints, scaffolds, alerts.

The platform's main RDKit surface and the ``chem.standardize`` capability. A
*real* scientific capability (trust label CALCULATED), fully deterministic.

It is not, however, the only module that imports RDKit: ``safety_tool``,
``synthesis_tool``, ``vina_tool``, ``mmff_tool`` and ``pfm_tool`` each reach
for it directly for a handful of descriptor calls. Everything *outside*
``tools/`` consumes typed :class:`Molecule` objects and never sees RDKit — that
is the boundary that actually holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..contracts.base import Provenance
from ..contracts.capability import CapabilityProfile
from ..contracts.chem import Molecule
from ..contracts.enums import CapabilityStatus, CostClass, Determinism, TrustLabel
from ..observability.toolmeter import instrument, measure
from ..util import content_hash
from .base import ToolAdapter

RDLogger.DisableLog("rdApp.*")

STANDARDIZER_VERSION = f"rdkit-{Chem.rdBase.rdkitVersion}"

# A small, transparent reactive/undesirable substructure alert set (SMARTS).
# Deliberately conservative and documented; supplements RDKit's PAINS catalog.
_ALERT_SMARTS: dict[str, str] = {
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "nitro_aromatic": "[a][NX3+](=O)[O-]",
    "thiol": "[SX2H]",
    "peroxide": "[OX2][OX2]",
    "azide": "[NX2]=[NX2+]=[NX1-]",
    "quaternary_nitrogen": "[NX4+]",
}


@dataclass
class StandardizationResult:
    molecule: Molecule
    mol: Any  # RDKit Mol (parent)


@lru_cache(maxsize=1)
def _fp_generator(radius: int = 2, fp_size: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


class Cheminformatics(ToolAdapter):
    """RDKit-backed cheminformatics operations."""

    capability_id = "chem.standardize"

    def __init__(self, fingerprint_bits: int = 2048, fingerprint_radius: int = 2) -> None:
        self.fp_bits = fingerprint_bits
        self.fp_radius = fingerprint_radius
        self._uncharger = rdMolStandardize.Uncharger()
        self._normalizer = rdMolStandardize.Normalizer()
        self._lfc = rdMolStandardize.LargestFragmentChooser()

    @property
    def profile(self) -> CapabilityProfile:
        return CapabilityProfile(
            capability_id=self.capability_id,
            version=STANDARDIZER_VERSION,
            title="RDKit cheminformatics",
            description="Sanitize, standardize, descriptors, fingerprints, alerts.",
            input_schema_ref="Molecule",
            output_schema_ref="Molecule",
            determinism=Determinism.DETERMINISTIC,
            cost_class=CostClass.CHEAP,
            status=CapabilityStatus.AVAILABLE,
            provides_mock=False,
            license="RDKit BSD-3-Clause",
            known_failure_modes=["unparseable SMILES", "sanitization failure"],
            validation_checks=["sanitizable", "inchikey computable", "descriptors finite"],
            backend="rdkit",
        )

    # --- parsing / standardization ----------------------------------------
    def standardize(
        self, smiles: str, *, source: str | None = None, lineage: list[str] | None = None
    ) -> Molecule:
        """Parse, sanitize and standardize a SMILES into a :class:`Molecule`.

        Preserves the input verbatim; standardized identity is used downstream.
        """
        with measure(self.capability_id, "standardize", backend="rdkit") as metered:
            mol = self._standardize(smiles, source=source, lineage=lineage)
            metered["standardized"] = 1.0 if mol.standardized else 0.0
            metered["n_structural_alerts"] = float(len(mol.structural_alerts))
        return mol

    def _standardize(
        self, smiles: str, *, source: str | None = None, lineage: list[str] | None = None
    ) -> Molecule:
        mol = Molecule(input_smiles=smiles, source=source, lineage=lineage or [])
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            mol.sanitization_ok = False
            mol.errors.append("unparseable_smiles")
            mol.standardized = False
            return mol

        # Detect mixtures/salts before choosing the parent fragment.
        frags = Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
        mol.is_mixture = len(frags) > 1

        try:
            clean = rdMolStandardize.Cleanup(rdmol)
            parent = self._lfc.choose(clean)
            parent = self._normalizer.normalize(parent)
            parent = self._uncharger.uncharge(parent)
            Chem.SanitizeMol(parent)
        except Exception as exc:
            mol.sanitization_ok = False
            mol.errors.append(f"standardize_error:{type(exc).__name__}")
            return mol

        mol.is_salt = mol.is_mixture and Chem.MolToSmiles(parent) != Chem.MolToSmiles(clean)
        mol.canonical_smiles = Chem.MolToSmiles(parent, isomericSmiles=False)
        mol.isomeric_smiles = Chem.MolToSmiles(parent, isomericSmiles=True)
        mol.formula = rdMolDescriptors.CalcMolFormula(parent)

        # Stereochemistry status.
        stereo = Chem.FindMolChiralCenters(parent, includeUnassigned=True, useLegacyImplementation=False)
        mol.has_stereo = len(stereo) > 0
        mol.stereo_defined = bool(stereo) and all(s[1] != "?" for s in stereo)

        try:
            inchi = Chem.MolToInchi(parent)
            mol.inchi = inchi or None
            mol.inchikey = Chem.InchiToInchiKey(inchi) if inchi else None
            mol.parent_inchikey = mol.inchikey
        except Exception:
            mol.errors.append("inchikey_unavailable")

        mol.descriptors = self.descriptors(parent)
        mol.structural_alerts = self.structural_alerts(parent)
        mol.standardized = True
        mol.standardizer_version = STANDARDIZER_VERSION
        mol.provenance = Provenance(
            source=source or "input",
            method="chem.standardize",
            method_version=STANDARDIZER_VERSION,
            trust_label=TrustLabel.CALCULATED,
            content_hash=content_hash({"can": mol.canonical_smiles}),
        )
        return mol

    # --- descriptors -------------------------------------------------------
    def descriptors(self, rdmol: Any) -> dict[str, float]:
        d = {
            "mw": Descriptors.MolWt(rdmol),
            "logp": Crippen.MolLogP(rdmol),
            "tpsa": rdMolDescriptors.CalcTPSA(rdmol),
            "hbd": float(Lipinski.NumHDonors(rdmol)),
            "hba": float(Lipinski.NumHAcceptors(rdmol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(rdmol)),
            "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(rdmol)),
            "rings": float(rdMolDescriptors.CalcNumRings(rdmol)),
            "heavy_atoms": float(rdmol.GetNumHeavyAtoms()),
            "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(rdmol),
            "qed": Descriptors.qed(rdmol),
            "formal_charge": float(Chem.GetFormalCharge(rdmol)),
        }
        return {k: round(float(v), 5) for k, v in d.items()}

    def descriptors_from_smiles(self, smiles: str) -> dict[str, float]:
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return {}
        return self.descriptors(rdmol)

    # --- fingerprints ------------------------------------------------------
    @instrument("chem.standardize", "fingerprint")
    def fingerprint(self, smiles: str) -> np.ndarray:
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return np.zeros(self.fp_bits, dtype=np.uint8)
        gen = _fp_generator(self.fp_radius, self.fp_bits)
        fp = gen.GetFingerprint(rdmol)
        arr = np.zeros(self.fp_bits, dtype=np.uint8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        return arr

    @staticmethod
    def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
        inter = float(np.bitwise_and(fp_a, fp_b).sum())
        union = float(np.bitwise_or(fp_a, fp_b).sum())
        return inter / union if union else 0.0

    # --- scaffolds / alerts ------------------------------------------------
    def scaffold(self, smiles: str) -> str | None:
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return None
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(rdmol)
            return Chem.MolToSmiles(scaf)
        except Exception:
            return None

    def structural_alerts(self, rdmol: Any) -> list[str]:
        hits: list[str] = []
        for name, smarts in _ALERT_SMARTS.items():
            patt = _compile_smarts(smarts)
            if patt is not None and rdmol.HasSubstructMatch(patt):
                hits.append(name)
        return hits

    def structural_alerts_from_smiles(self, smiles: str) -> list[str]:
        rdmol = Chem.MolFromSmiles(smiles)
        return self.structural_alerts(rdmol) if rdmol is not None else []

    # --- deterministic analogue enumeration (generation baseline) ---------
    def enumerate_analogues(self, smiles: str, *, max_per_seed: int = 24) -> list[str]:
        """Enumerate a small, deterministic analogue set by single-point aromatic
        substitutions (F, Cl, Br, methyl, methoxy). Returns canonical SMILES,
        de-duplicated and excluding the seed. Not a generative model — a
        transparent, reproducible baseline enumerator.
        """
        with measure(self.capability_id, "enumerate_analogues", backend="rdkit") as metered:
            out = self._enumerate_analogues(smiles, max_per_seed=max_per_seed)
            metered["n_analogues"] = float(len(out))
        return out

    def _enumerate_analogues(self, smiles: str, *, max_per_seed: int = 24) -> list[str]:
        from rdkit.Chem import AllChem

        seed = Chem.MolFromSmiles(smiles)
        if seed is None:
            return []
        seed_can = Chem.MolToSmiles(seed)
        rxn_smarts = [
            "[cH:1]>>[c:1]F",
            "[cH:1]>>[c:1]Cl",
            "[cH:1]>>[c:1]Br",
            "[cH:1]>>[c:1]C",
            "[cH:1]>>[c:1]OC",
        ]
        products: dict[str, None] = {}
        for sm in rxn_smarts:
            rxn = AllChem.ReactionFromSmarts(sm)
            for prod_set in rxn.RunReactants((seed,)):
                for prod in prod_set:
                    try:
                        Chem.SanitizeMol(prod)
                        can = Chem.MolToSmiles(prod)
                    except Exception:
                        continue
                    if can and can != seed_can:
                        products[can] = None
        # Deterministic order + cap.
        return sorted(products)[:max_per_seed]


@lru_cache(maxsize=256)
def _compile_smarts(smarts: str):
    return Chem.MolFromSmarts(smarts)
