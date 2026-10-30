import io
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.SeqUtils import seq1

from .contracts import BoltzGenerate, Structure, ToolResult, Unavailable
from .models import model_root

BOLTZ_REV = "c1be29e1f82ffcc72264f64b993c43fb4e0d17f0"
MOLS_REV = "c3d36fd276e9caf098c75d4113c6d5eb320b1a4c"


def boltz_paths():
    root = model_root() / "boltzgen"
    weights = root / "models--boltzgen--boltzgen-1" / "snapshots" / BOLTZ_REV
    moldir = root / "datasets--boltzgen--inference-data" / "snapshots" / MOLS_REV / "mols.zip"
    design, fold = weights / "boltzgen1_diverse.ckpt", weights / "boltz2_conf_final.ckpt"
    if not design.is_file() or not fold.is_file() or not moldir.is_file():
        raise Unavailable("BoltzGen assets missing. Run peptide-agent download boltzgen")
    return design, fold, moldir


def generate_boltz(request: BoltzGenerate):
    design, fold, moldir = boltz_paths()
    with tempfile.TemporaryDirectory(prefix="peptide-boltz-") as temp:
        root = Path(temp)
        entities = [{"protein": {"id": "B", "sequence": f"{request.length}..{request.length}",
                                  "cyclic": request.topology == "head_to_tail"}}]
        if request.target_pdb:
            model = PDBParser(QUIET=True).get_structure("target", io.StringIO(request.target_pdb))[0]
            if request.target_chain not in model:
                raise ValueError("Target chain absent from PDB")
            (root / "target.pdb").write_text(request.target_pdb)
            entities.append({"file": {"path": "target.pdb", "include": [{"chain": {"id": request.target_chain}}]}})
        else:
            entities.append({"protein": {"id": request.target_chain, "sequence": request.target_sequence}})
        # Upstream assigns output chain IDs in entity order.
        entities = entities[1:] + entities[:1]
        spec = root / "design.yaml"
        spec.write_text(yaml.safe_dump({"entities": entities}))
        output = root / "output"
        command = [sys.executable, "-m", "peptide_agent.boltz_runner", str(request.seed), "run", str(spec),
                   "--output", str(output), "--protocol", "peptide-anything", "--num_designs", str(request.count),
                   "--budget", str(request.count), "--devices", "1", "--num_workers", "0",
                   "--design_checkpoints", str(design), "--folding_checkpoint", str(fold),
                   "--moldir", str(moldir), "--skip_inverse_folding", "--use_kernels", "false", "--no_subprocess",
                   "--config", "design", f"sampling_steps={request.sampling_steps}",
                   "--steps", "design"]
        if request.refold:
            command.append("folding")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
        proc = subprocess.run(command, text=True, capture_output=True, timeout=1800, env=env)
        if proc.returncode:
            raise ValueError("BoltzGen failed: " + (proc.stderr + proc.stdout)[-4000:])
        all_cifs = sorted(output.rglob("*.cif"))
        cifs = [p for p in all_cifs if p.parent.name == "refold_cif"] if request.refold else []
        if request.refold and not cifs:
            raise ValueError("BoltzGen refolding requested but no refolded structures produced")
        if not cifs:
            cifs = [p for p in all_cifs if p.parent.name == "intermediate_designs"]
        candidates = []
        for path in cifs:
            model = MMCIFParser(QUIET=True).get_structure(path.stem, path)[0]
            # Verify chain identity against upstream design metadata, never by length.
            design_cifs = [p for p in all_cifs if p.parent.name == "intermediate_designs" and p.stem == path.stem]
            metadata = path.with_suffix(".npz") if path.parent.name == "intermediate_designs" else (design_cifs[0].with_suffix(".npz") if design_cifs else None)
            if metadata is None or not metadata.is_file():
                raise ValueError("Missing BoltzGen design-mask metadata")
            with np.load(metadata, allow_pickle=False) as data:
                mask = data["design_mask"].reshape(-1).astype(bool)
            chains = list(model)
            if len(chains) != 2 or len(mask) != sum(len(c) for c in chains):
                raise ValueError("Unexpected BoltzGen chain/token layout")
            offset, designed = 0, []
            for chain in chains:
                flags = mask[offset:offset + len(chain)]
                if flags.all(): designed.append(chain)
                elif flags.any(): raise ValueError("Partially designed chain unsupported")
                offset += len(chain)
            if len(designed) != 1:
                raise ValueError("Ambiguous BoltzGen designed chain")
            binder = designed[0]
            receptor = next(c for c in chains if c is not binder)
            original_mapping = {binder.id: "B", receptor.id: request.target_chain}
            for chain in chains: model.detach_child(chain.id)
            binder.id, receptor.id = "B", request.target_chain
            model.add(receptor); model.add(binder)
            sequence = "".join(seq1(r.resname) for r in model["B"] if r.id[0] == " ")
            if len(sequence) != request.length or "X" in sequence:
                raise ValueError(f"Unexpected BoltzGen peptide sequence {sequence!r}; residues="+
                                 str([(r.resname, r.id) for r in model['B']]))
            stream = io.StringIO()
            writer = PDBIO(); writer.set_structure(model); writer.save(stream)
            pdb_text = stream.getvalue()
            if request.topology == "head_to_tail":
                atoms = [line for line in pdb_text.splitlines() if line.startswith("ATOM") and line[21] == "B"]
                first_n = next(int(line[6:11]) for line in atoms if line[12:16].strip() == "N")
                last_c = next(int(line[6:11]) for line in reversed(atoms) if line[12:16].strip() == "C")
                pdb_text = pdb_text.rsplit("END", 1)[0] + f"CONECT{first_n:5d}{last_c:5d}\nCONECT{last_c:5d}{first_n:5d}\nEND   \n"
            candidates.append({"sequence": sequence, "pdb": pdb_text, "cif": path.read_text(),
                               "topology": request.topology, "cif_to_pdb_chains": original_mapping,
                               "closure_distance_angstrom": float(np.linalg.norm(list(binder)[-1]["C"].coord - list(binder)[0]["N"].coord)) if request.topology == "head_to_tail" else None,
                               "refolded": path.parent.name == "refold_cif"})
        if len(candidates) != request.count:
            raise ValueError(f"BoltzGen returned {len(candidates)} of {request.count} requested structures")
        return ToolResult(tool="generation.boltzgen", status="ok", trust="PREDICTED",
                          data={"sequences": [c["sequence"] for c in candidates], "complexes": candidates},
                          provenance={"model": "BoltzGen-1 diverse", "revision": BOLTZ_REV, "molecules_revision": MOLS_REV,
                                      "package": importlib.metadata.version("boltzgen"), "seed": request.seed,
                                      "sampling_steps": request.sampling_steps, "topology": request.topology},
                          warnings=["Generated structures are predictions, not experimental complexes.",
                                    "Reduced sampling steps are for smoke testing; default research setting is 500."])


def affinity(request: Structure):
    from prodigy_prot.modules.prodigy import Prodigy
    if request.receptor_chain == request.peptide_chain:
        raise ValueError("Distinct receptor and peptide chains required")
    model = PDBParser(QUIET=True).get_structure("complex", io.StringIO(request.pdb))[0]
    for chain in (request.receptor_chain, request.peptide_chain):
        if chain not in model:
            raise ValueError(f"Chain {chain} absent")
    for chain in list(model):
        if chain.id not in {request.receptor_chain, request.peptide_chain}:
            model.detach_child(chain.id)
    for chain in model:
        for residue in list(chain):
            if residue.id[0] != " ": chain.detach_child(residue.id)
    predictor = Prodigy(model, selection=[request.receptor_chain, request.peptide_chain], temp=25.0)
    predictor.predict()
    return ToolResult(tool="affinity.prodigy", status="ok", trust="PREDICTED",
                      data={"delta_g_kcal_mol": float(predictor.ba_val), "kd_M": float(predictor.kd_val),
                            "temperature_C": 25.0, "contacts": len(predictor.ic_network), "calibrated_for_peptides": False},
                      provenance={"algorithm": "PRODIGY IC-NIS", "version": importlib.metadata.version("prodigy-prot")},
                      warnings=["Protein-complex empirical model; short/cyclic-peptide extrapolation is not calibrated.",
                                "Predicted geometry adds uncertainty. Kd is not a measured binding constant."])
