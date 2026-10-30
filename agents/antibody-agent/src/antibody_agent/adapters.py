"""Native scientific adapters. Run only in the isolated science worker."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .contracts import Binder, Brief, Chain, NumberedResidue, Unavailable
from .structure import audit, coordinates


def number(arguments, config, directory):
    from anarcii import Anarcii

    binder = Binder.model_validate(arguments["binder"])
    if arguments.get("scheme", "IMGT") != "IMGT":
        raise Unavailable("Native numbering uses IMGT; other schemes require a validated mapping bridge")
    model = Anarcii(seq_type="antibody", cpu=config.get("cpu", True), ncpu=4)
    for chain in binder.chains:
        result = next(iter(model.number([chain.sequence]).values()))
        if result.get("error") or not result.get("numbering"):
            raise ValueError("ANARCII could not number the complete variable domain")
        kind = result["chain_type"]
        if (chain.role in {"VH", "VHH"} and kind != "H") or (chain.role == "VL" and kind not in {"K", "L"}):
            raise ValueError("Predicted chain type does not match the supplied chain role")
        ungapped = [(label, aa) for label, aa in result["numbering"] if aa != "-"]
        if "".join(aa for _, aa in ungapped) != chain.sequence:
            raise ValueError("ANARCII trimmed the input; supply the exact variable domain explicitly")
        rows = []
        for index, ((position, insertion), _) in enumerate(ungapped, 1):
            region = (
                "FR1"
                if position < 27
                else "CDR1"
                if position <= 38
                else "FR2"
                if position < 56
                else "CDR2"
                if position <= 65
                else "FR3"
                if position < 105
                else "CDR3"
                if position <= 117
                else "FR4"
            )
            rows.append(NumberedResidue(index=index, label=str(position) + insertion.strip(), region=region))
        chain.numbering_source = "ANARCII " + config["version"]
        chain.numbering = rows
        chain.scheme = "IMGT"
    return {"binder": binder.model_dump()}


def fold(arguments, config, directory):
    from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2
    from ImmuneBuilder.util import are_weights_ready

    binder = Binder.model_validate(arguments["binder"])
    path = Path(config.get("weights_dir", ""))
    prefix = "nanobody" if binder.chains[0].role == "VHH" else "antibody"
    if not path.is_dir() or not all(
        are_weights_ready(str(path / f"{prefix}_model_{i}")) for i in (1, 2, 3, 4)
    ):
        raise Unavailable("ImmuneBuilder requires preinstalled weights_dir; no implicit model download")
    cls = NanoBodyBuilder2 if binder.chains[0].role == "VHH" else ABodyBuilder2
    predictor = cls(weights_dir=str(path))
    sequences = {"H" if c.role in {"VH", "VHH"} else "L": c.sequence for c in binder.chains}
    result = predictor.predict(sequences)
    output = directory / "fold.pdb"
    result.save(str(output))
    pdb = output.read_text()
    return {"pdb": pdb, **audit(pdb, binder), "method": cls.__name__, "templates_used": False}


def complex_prediction(arguments, config, directory):
    import yaml

    from .contracts import AntigenState

    binder = Binder.model_validate(arguments["binder"])
    state = AntigenState.model_validate(arguments["state"])
    if not state.sequence:
        raise ValueError("Complex prediction requires the explicit antigen sequence")
    cache = Path(config.get("cache", ""))
    checkpoint = Path(config.get("checkpoint", ""))
    if not cache.is_dir() or not checkpoint.is_file():
        raise Unavailable("Boltz-2 requires an installed cache and checkpoint")
    sequences = [
        {"protein": {"id": "H" if c.role in {"VH", "VHH"} else "L", "sequence": c.sequence, "msa": "empty"}}
        for c in sorted(binder.chains, key=lambda c: c.role)
    ]
    sequences.append({"protein": {"id": "A", "sequence": state.sequence, "msa": "empty"}})
    input_path = directory / "complex.yaml"
    input_path.write_text(yaml.safe_dump({"version": 1, "sequences": sequences}))
    # No template, contact constraint, affinity property, or remote MSA upload.
    executable = str(Path(sys.executable).parent / "boltz")
    subprocess.run(
        [
            executable,
            "predict",
            str(input_path),
            "--out_dir",
            str(directory / "boltz"),
            "--cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--output_format",
            "pdb",
            "--diffusion_samples",
            "1",
            "--recycling_steps",
            "3",
            "--sampling_steps",
            "200",
            "--accelerator",
            config.get("accelerator", "gpu"),
        ],
        check=True,
    )
    outputs = list((directory / "boltz").rglob("complex_model_0.pdb"))
    confidences = list((directory / "boltz").rglob("confidence_complex_model_0.json"))
    if len(outputs) != 1 or len(confidences) != 1:
        raise ValueError("Expected exactly one Boltz structure and confidence record")
    pdb = outputs[0].read_text()
    scores = json.loads(confidences[0].read_text())
    antigen_index = str(len(binder.chains))
    pairs = scores["pair_chains_iptm"]
    values = [pairs[str(i)][antigen_index] for i in range(len(binder.chains))]
    return {
        "pdb": pdb,
        "antigen_iptm": max(values),
        "complex_plddt": scores["complex_plddt"],
        "templates_used": False,
        "forced_contacts": False,
        "native_context_checked": False,
        "context_id": arguments["context_id"],
        "msa_mode": "single_sequence",
        "limitation": "Single-sequence mode has reduced accuracy; no glycans, membrane, or antibody KD prediction",
        "raw_confidence": scores,
    }


def antifold(arguments, config, directory):
    from Bio import SeqIO

    brief = Brief.model_validate(arguments["brief"])
    parent = brief.reference
    if (
        not parent
        or not parent.structure_pdb
        or any(not c.numbering or c.scheme != "IMGT" for c in parent.chains)
    ):
        raise ValueError("AntiFold requires a mapped IMGT variable-domain structure")
    weights = Path(config.get("checkpoint", ""))
    if not weights.is_file():
        raise Unavailable("AntiFold checkpoint not installed")
    coords = coordinates(parent.structure_pdb)
    for chain in parent.chains:
        name = "L" if chain.role == "VL" else "H"
        rows = coords.get(name, [])
        if "".join(r["aa"] for r in rows) != chain.sequence:
            raise ValueError("AntiFold input structure differs from the paired reference")
        labels = {r.index: r.label for r in chain.numbering}
        if [r["residue"] for r in rows] != [labels[i] for i in range(1, len(rows) + 1)]:
            raise ValueError("AntiFold PDB residue ids must match the explicit IMGT map")
    regions = []
    for role, labels in brief.constraints.editable.items():
        if any(not re.fullmatch(r"[0-9]+", label) for label in labels):
            raise Unavailable(
                "Native AntiFold position selection requires numeric IMGT labels; insertion positions need a mapping bridge"
            )
        if labels:
            regions.append(("L:" if role == "VL" else "H:") + ",".join(labels))
    if not regions:
        raise ValueError("Declare editable IMGT positions for AntiFold")
    pdb = directory / "parent.pdb"
    pdb.write_text(parent.structure_pdb)
    argv = [
        sys.executable,
        "-m",
        "antifold.main",
        "--pdb_file",
        str(pdb),
        "--out_dir",
        str(directory / "output"),
        "--model_path",
        str(weights),
        "--num_seq_per_target",
        str(arguments["count"]),
        "--sampling_temp",
        "0.2",
        "--regions",
        " ".join(regions),
        "--seed",
        str(brief.seed),
        "--skip_anarcii_numbering",
    ]
    argv += (
        ["--nanobody_chain", "H"]
        if parent.chains[0].role == "VHH"
        else ["--heavy_chain", "H", "--light_chain", "L"]
    )
    subprocess.run(argv, check=True)
    outputs = []
    for path in sorted((directory / "output").glob("*.fasta")):
        for record in SeqIO.parse(path, "fasta"):
            if not record.id.startswith("seq"):
                continue  # The first FASTA record is the parent, not a generated sequence.
            seqs = str(record.seq).split("/")
            roles = ["VHH"] if parent.chains[0].role == "VHH" else ["VH", "VL"]
            if len(seqs) != len(roles):
                raise ValueError("AntiFold output lost heavy/light pairing")
            outputs.append(
                {
                    "name": record.id,
                    "chains": [
                        Chain(role=r, sequence=s).model_dump() for r, s in zip(roles, seqs, strict=True)
                    ],
                    "pairing_source": "model_proposed",
                    "source": "AntiFold",
                    "product": brief.product.model_dump(),
                }
            )
    if len(outputs) != arguments["count"]:
        raise ValueError("AntiFold did not return the requested number of sampled pairs")
    return {"binders": outputs, "sequence_score_meaning": "Inverse-folding likelihood, not affinity"}


def rfantibody(arguments, config, directory):
    brief = Brief.model_validate(arguments["brief"])
    parent = brief.reference
    states = [s for s in brief.states if s.role == "positive"]
    if not parent or not parent.structure_pdb or len(states) != 1 or not states[0].structure_pdb:
        raise ValueError(
            "Native RFantibody needs one target PDB and an HLT framework; multi-state design uses a bridge"
        )
    if brief.constraints.allow_length_change:
        raise Unavailable(
            "Native RFantibody uses fixed CDR lengths; length changes require a validated mapping bridge"
        )
    framework = coordinates(parent.structure_pdb)
    loops, offset, expected_remarks = [], 0, set()
    for chain in sorted(parent.chains, key=lambda c: c.role):
        code = "L" if chain.role == "VL" else "H"
        if "".join(r["aa"] for r in framework.get(code, [])) != chain.sequence:
            raise ValueError("HLT framework does not match the reference sequence")
        editable = set(brief.constraints.editable.get(chain.role, []))
        selected = set()
        for n in (1, 2, 3):
            rows = [r for r in chain.numbering if r.region == f"CDR{n}"]
            labels = {r.label for r in rows}
            expected_remarks.update((offset + r.index, code + str(n)) for r in rows)
            if editable & labels:
                if not labels <= editable:
                    raise ValueError(
                        "RFantibody backbone design requires entire declared CDR loops to be editable"
                    )
                loops.append(code + str(n) + ":" + str(len(rows)))
                selected |= labels
        if selected != editable:
            raise ValueError("Native RFantibody only edits complete CDR loops")
        offset += len(chain.sequence)
    remarks = {
        (int(i), label)
        for i, label in re.findall(r"REMARK PDBinfo-LABEL:\s+(\d+)\s+([HL][123])", parent.structure_pdb)
    }
    if remarks != expected_remarks or not loops:
        raise ValueError("HLT CDR remarks and declared numbering must agree exactly")
    state = states[0]
    target_coords = coordinates(state.structure_pdb).get(state.structure_chain, [])
    if "".join(r["aa"] for r in target_coords) != state.sequence:
        raise ValueError("Target PDB/sequence mismatch; provide a reversible full-context mapping")
    contacts = sorted({r for e in brief.epitopes if e.state_id == state.id for r in e.residues})
    if not contacts:
        raise ValueError("Declare RFantibody epitope hotspots")
    hotspots = ",".join(state.structure_chain + target_coords[i - 1]["residue"] for i in contacts)
    target, scaffold = directory / "target.pdb", directory / "framework.pdb"
    target.write_text(state.structure_pdb)
    scaffold.write_text(parent.structure_pdb)
    backbones, sequences, predicted = (directory / n for n in ("backbones", "sequences", "rf2"))
    for path in (backbones, sequences, predicted):
        path.mkdir()
    binaries = Path(config["bin_dir"])
    subprocess.run(
        [
            str(binaries / "rfdiffusion"),
            "-t",
            str(target),
            "-f",
            str(scaffold),
            "-o",
            str(backbones / "design"),
            "-n",
            str(arguments["count"]),
            "-l",
            ",".join(loops),
            "-h",
            hotspots,
            "--deterministic",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(binaries / "proteinmpnn"),
            "-i",
            str(backbones),
            "-o",
            str(sequences),
            "-n",
            "1",
            "-l",
            ",".join(loop.split(":")[0] for loop in loops),
            "--deterministic",
        ],
        check=True,
    )
    subprocess.run([str(binaries / "rf2"), "-i", str(sequences), "-o", str(predicted)], check=True)
    outputs = []
    for path in sorted(sequences.glob("*.pdb")):
        rows = coordinates(path.read_text())
        chains = [
            Chain(role=c.role, sequence="".join(r["aa"] for r in rows["L" if c.role == "VL" else "H"]))
            for c in parent.chains
        ]
        binder = Binder(
            chains=chains,
            pairing_source="model_proposed",
            source="RFantibody backbone + base ProteinMPNN",
            product=brief.product,
            structure_pdb=path.read_text(),
        )
        outputs.append(binder.model_dump())
    if len(outputs) != arguments["count"] or not list(predicted.glob("*.pdb")):
        raise ValueError("RFantibody pipeline returned incomplete sequence/structure outputs")
    return {
        "binders": outputs,
        "internal_rf2": "Generator self-consistency only; independent evaluation still required",
    }


NATIVE = {
    "numbering.anarcii": number,
    "fold.immunebuilder": fold,
    "complex.boltz2": complex_prediction,
    "generation.antifold": antifold,
    "generation.rfantibody": rfantibody,
}
