from __future__ import annotations
import io
import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.Data.PDBData import protein_letters_3to1


def chain_records(pdb, chain="A"):
    structure = PDBParser(QUIET=True).get_structure("input", io.StringIO(pdb))
    if chain not in structure[0]:
        raise ValueError(f"Missing chain {chain}")
    residues = [r for r in structure[0][chain] if r.id[0] == " " and r.resname in protein_letters_3to1]
    seq = "".join(protein_letters_3to1[r.resname] for r in residues)
    if not seq:
        raise ValueError("No canonical protein residues in selected chain")
    return structure, residues, seq


def validated_mapping(reference):
    if not reference.pdb:
        raise ValueError("A reference structure is required")
    structure, residues, sequence = chain_records(reference.pdb, reference.chain)
    if sequence != reference.sequence:
        raise ValueError("PDB chain does not exactly match mature sequence; resolve missing residues/numbering before design")
    if any(any(a not in r for a in ("N", "CA", "C", "O")) for r in residues):
        raise ValueError("Reference backbone has missing atoms")
    labels = [f"{reference.chain}{r.id[1]}{r.id[2].strip()}" for r in residues]
    return structure, residues, labels


def surface_positions(reference, count=8):
    structure, residues, _ = validated_mapping(reference)
    ShrakeRupley(n_points=100).compute(structure[0], level="R")
    protected = set(reference.protected_positions + reference.catalytic_positions)
    critical = [residues[p-1]["CA"].coord for p in protected]
    choices = []
    for i, residue in enumerate(residues, 1):
        if i in protected or reference.sequence[i-1] in "CGP":
            continue
        if critical and min(np.linalg.norm(residue["CA"].coord - c) for c in critical) < 10:
            continue
        choices.append((residue.sasa, i))
    return [p for _,p in sorted(choices, reverse=True)[:count]]


def compare_structures(candidate_pdb, reference, catalytic_positions):
    _, target, sequence = chain_records(candidate_pdb, "A")
    _, parent, refseq = chain_records(reference.pdb, reference.chain)
    if len(sequence) != len(refseq):
        return {"status": "not_applicable", "reason": "Sequence-length mismatch requires explicit alignment"}
    a = np.array([r["CA"].coord for r in target])
    b = np.array([r["CA"].coord for r in parent])
    ac, bc = a-a.mean(0), b-b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.eye(3); correction[-1,-1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    diff = (ac @ rotation-bc)
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2,axis=1))))
    local = [p-1 for p in catalytic_positions]
    sidechain_differences=[]
    atom_names=[]
    for p in catalytic_positions:
        if sequence[p-1]!=refseq[p-1]: continue
        for atom in target[p-1]:
            if atom.name in {'N','CA','C','O','OXT'} or atom.element=='H': continue
            if atom.name in parent[p-1]:
                transformed=(atom.coord-a.mean(0))@rotation+b.mean(0)
                sidechain_differences.append(transformed-parent[p-1][atom.name].coord)
                atom_names.append(f'{p}:{atom.name}')
    sidechain_rmsd=float(np.sqrt(np.mean(np.sum(np.array(sidechain_differences)**2,axis=1)))) if sidechain_differences else None
    return {"status": "ok", "backbone_rmsd_angstrom": rmsd,
            "catalytic_ca_rmsd_angstrom": float(np.sqrt(np.mean(np.sum(diff[local]**2, axis=1)))) if local else None,
            "catalytic_sidechain_rmsd_angstrom":sidechain_rmsd,"compared_catalytic_atoms":atom_names,
            "mapping": "equal-length mature-chain indices; whole-chain Kabsch fit", "unit": "angstrom",
            "limitations": "Reference-relative static geometry; ligand, transition-state and conformational-ensemble compatibility remain unknown"}
