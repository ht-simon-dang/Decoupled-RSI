"""Real model adapters. No synthetic fallback when a model fails."""
from __future__ import annotations
import importlib.metadata
import json
import random
import subprocess
import sys
from pathlib import Path

from .contracts import AA, Brief, Unavailable
from .models import MODELS, model_path
from .structure import validated_mapping


def editable_positions(brief):
    if not brief.reference:
        raise ValueError("Conservative engineering requires a reference enzyme")
    if not brief.design_positions:
        raise ValueError("Specify design_positions or supply a complete structure for surface selection")
    return brief.design_positions


def generate_esm3(brief, count, work):
    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig, ESMProteinError
    import esm.pretrained as pretrained
    import esm.utils.constants.esm3 as constants
    path = model_path("esm3")
    # The upstream builder otherwise performs an unpinned Hub download. Resolve
    # its data root to the explicitly downloaded snapshot for this worker only.
    pretrained.data_root = lambda model: path
    constants.data_root = lambda model: path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(brief.seed)
    model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device(device)).eval()
    positions = editable_positions(brief)
    masked = list(brief.reference.sequence)
    for p in positions:
        masked[p-1] = "_"
    sequences = []
    for index in range(count):
        torch.manual_seed(brief.seed + index)
        result = model.generate(ESMProtein(sequence="".join(masked)), GenerationConfig(track="sequence", num_steps=min(8, len(positions)), temperature=brief.temperature))
        if isinstance(result, ESMProteinError):
            raise RuntimeError(f"ESM3 generation failed: {result}")
        sequences.append(result.sequence)
    return {"sequences": sequences, "provenance": {**MODELS["esm3"], "package": importlib.metadata.version("esm"), "masked_positions": positions, "seed": brief.seed,
            "temperature": brief.temperature, "design_conditioning": "sequence masks; fixed positions unchanged", "device": device}}


def generate_mpnn(brief, count, work, ligand=True):
    base = model_path("ligandmpnn")
    structure, residues, labels = validated_mapping(brief.reference)
    positions = editable_positions(brief)
    if ligand:
        ligands = [r for r in structure[0].get_residues() if r.id[0].startswith("H_") and r.resname not in {"HOH", "WAT"}]
        if not ligands:
            raise ValueError("LigandMPNN requires a ligand/cofactor context; explicitly choose ProteinMPNN otherwise")
    # Keep all chains as context and redesign only the mapped selected chain.
    pdb = work / "parent.pdb"
    pdb.write_text(brief.reference.pdb)
    kind = "ligand_mpnn" if ligand else "protein_mpnn"
    weight = "ligandmpnn_v_32_010_25.pt" if ligand else "proteinmpnn_v_48_020.pt"
    args = [sys.executable, "-m", "enzyme_agent.mpnn_runner", str(base / "source/run.py"), "--model_type", kind,
            "--checkpoint_" + kind, str(base / weight), "--pdb_path", str(pdb),
            "--out_folder", str(work / "output"), "--batch_size", "1", "--number_of_batches", str(count),
            "--seed", str(brief.seed), "--temperature", str(brief.temperature),
            "--chains_to_design", brief.reference.chain, "--parse_these_chains_only", brief.reference.chain,
            "--redesigned_residues", " ".join(labels[p-1] for p in positions), "--pack_side_chains", "0"]
    result = subprocess.run(args, cwd=base / "source", capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError("LigandMPNN worker failed: " + (result.stderr + result.stdout)[-2500:])
    from Bio import SeqIO
    fasta = work / "output/seqs/parent.fa"
    records = list(SeqIO.parse(fasta, "fasta"))
    sequences = [str(record.seq).replace(":", "") for record in records[1:]]
    if len(sequences) != count:
        raise RuntimeError("MPNN output candidate count mismatch")
    manifest = json.loads((base / "manifest.json").read_text())
    return {"sequences": sequences, "provenance": {"model": kind, "repo": manifest["repo"], "revision": manifest["revision"], "weights_sha256": manifest["files"][weight],
            "seed": brief.seed, "temperature": brief.temperature, "design_positions": positions, "residue_map": dict(zip(map(str, range(1,len(labels)+1)),labels)),
            "ligand_context": ligand, "likelihood_is_catalysis": False, "compatibility": "process-local np.int=int for pinned OpenFold on NumPy 2"}}


def generate_zymctrl(brief, count, work):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not brief.reaction.ec:
        raise ValueError("ZymCTRL requires a complete EC class")
    path = model_path("zymctrl")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tok(brief.reaction.ec, return_tensors="pt").to(model.device)
    sequences = []
    raw = []
    for i in range(count):
        torch.manual_seed(brief.seed+i)
        with torch.inference_mode():
            output = model.generate(**inputs, max_length=1024, do_sample=True, top_k=9, repetition_penalty=1.2,
                temperature=brief.temperature, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        text = tok.decode(output[0], skip_special_tokens=False)
        raw.append(text)
        if "<start>" not in text or "<end>" not in text:
            raise RuntimeError("ZymCTRL did not produce a complete <start>…<end> enzyme")
        sequence = text.split("<start>",1)[1].split("<end>",1)[0].replace(" ", "")
        if not sequence or set(sequence) - set(AA):
            raise RuntimeError("ZymCTRL returned noncanonical residues")
        sequences.append(sequence)
    return {"sequences": sequences, "raw": raw, "provenance": {**MODELS["zymctrl"], "ec": brief.reaction.ec, "seed": brief.seed,
            "scope": "EC-conditioned exploratory seeds; substrate compatibility and processing unverified"}}


def generate_simple(brief, count, work):
    rng = random.Random(brief.seed)
    positions = editable_positions(brief)
    groups = ["AVILM", "DE", "KR", "STNQ", "FYWH"]
    sequences = []
    for i in range(count):
        seq = list(brief.reference.sequence)
        p = positions[i % len(positions)]-1
        choices = next((g for g in groups if seq[p] in g), AA)
        seq[p] = rng.choice([c for c in choices if c != seq[p]])
        sequences.append("".join(seq))
    return {"sequences": sequences, "provenance": {"model": "simple_conservative_mutation", "version": "1", "seed": brief.seed,
            "scope": "Explicit baseline only; substitutions are hypotheses, not optimized predictions"}}


def generate(name, args, work):
    brief = Brief.model_validate(args["brief"])
    count = args["count"]
    if name == "esm3":
        return generate_esm3(brief, count, work)
    if name in {"ligandmpnn", "proteinmpnn"}:
        return generate_mpnn(brief, count, work, name == "ligandmpnn")
    if name == "zymctrl":
        return generate_zymctrl(brief, count, work)
    if name == "simple":
        return generate_simple(brief, count, work)
    if name == 'rfdiffusion3':
        from .rfdiffusion import generate_rfd3
        return generate_rfd3(brief, count, work)
    raise Unavailable(f"{name} is registered but no validated runtime is installed; no replacement model was used")
