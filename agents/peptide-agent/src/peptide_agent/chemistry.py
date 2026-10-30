"""Explicit head-to-tail amide closure and topology-aware descriptors."""
import math

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def molecule(sequence, topology):
    mol = Chem.MolFromSequence(sequence)
    if mol is None:
        raise ValueError("Cannot construct canonical peptide")
    if topology == "head_to_tail":
        def atom(name, residue):
            return next(a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo()
                        and a.GetPDBResidueInfo().GetName().strip() == name
                        and a.GetPDBResidueInfo().GetResidueNumber() == residue)
        n, c, oxygen = atom("N", 1), atom("C", len(sequence)), atom("OXT", len(sequence))
        edit = Chem.RWMol(mol)
        edit.AddBond(n, c, Chem.BondType.SINGLE)
        edit.RemoveAtom(oxygen)
        mol = edit.GetMol()
        Chem.SanitizeMol(mol)
    return mol


def cyclic_charge(sequence, ph):
    # Same side-chain pKas used by Biopython's Bjellqvist implementation; no terminal groups.
    positive = {"K": 10.0, "R": 12.0, "H": 5.98}
    negative = {"D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
    return sum(sequence.count(a) / (1 + 10**(ph-pk)) for a, pk in positive.items()) - sum(
        sequence.count(a) / (1 + 10**(pk-ph)) for a, pk in negative.items())


def properties(sequence, topology, ph):
    mol = molecule(sequence, topology)
    result = {"topology": topology, "smiles": Chem.MolToSmiles(mol),
              "molecular_weight_da": Descriptors.MolWt(mol), "formula": rdMolDescriptors.CalcMolFormula(mol),
              "tpsa_A2": rdMolDescriptors.CalcTPSA(mol)}
    if topology == "head_to_tail":
        low, high = 0.0, 14.0
        if not any(a in sequence for a in "KRHDECY") or cyclic_charge(sequence, low)*cyclic_charge(sequence, high) > 0:
            pi = None
        else:
            for _ in range(60):
                mid = (low+high)/2
                if cyclic_charge(sequence, mid) > 0: low = mid
                else: high = mid
            pi = (low+high)/2
        result.update(charge_at_ph=cyclic_charge(sequence, ph), isoelectric_point=pi,
                      instability_index=None, trypsin_sites=None)
    return result
