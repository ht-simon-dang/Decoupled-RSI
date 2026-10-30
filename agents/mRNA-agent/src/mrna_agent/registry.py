"""Explicit roles, provenance, scope and activation state for every blueprint model."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .storage import data_root, used_bytes, MAX_BYTES

QWEN = {"repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"}


def qwen_path():
    configured = os.environ.get("MRNA_QWEN_DIR")
    if configured:
        return Path(configured).resolve()
    shared_root = Path(__file__).resolve().parents[3] / "peptide-agent/.peptide-agent/models"
    shared_root = Path(os.environ.get("PEPTIDE_MODEL_DIR", str(shared_root))).expanduser()
    sibling = shared_root / "qwen" / QWEN["revision"]
    return sibling if sibling.is_dir() else data_root() / "models/qwen" / QWEN["revision"]


MODELS = {
    'qwen': dict(role='planner', regions=[], source='https://huggingface.co/Qwen/Qwen3.5-0.8B', license='Apache-2.0; same pinned revision as peptide-agent', scope='Planning only; no scientific scores; 8192-token application context', length_unit='tokens', max_length=8192),
    'gemorna': dict(role='generator', regions=['cds', 'utr5', 'utr3'], source='https://github.com/RainaBio/GEMORNA', license='Noncommercial research; no redistribution', scope='Region generation; canonical RNA; modification/cell response uncalibrated', length_unit='amino_acids', max_length=184),
    'lineardesign': dict(role='optimizer', regions=['cds'], source='https://github.com/LinearDesignSoftware/LinearDesign', license='Research terms; separate commercial/distribution permission', scope='Human codon table; CDS MFE/CAI reward, not protein yield', length_unit='amino_acids', max_length=2000),
    'synonymous': dict(role='baseline', regions=['cds'], source='local deterministic standard genetic code', license='Project license', scope='Explicit uniform synonymous sampling fallback; no learned biological efficacy', length_unit='amino_acids', max_length=6000),
    'ribonn': dict(role='predictor', regions=['utr5', 'cds', 'utr3'], source='https://github.com/Sanofi-Public/RiboNN', license='Academic/noncommercial code and separate model-weights license', scope='Human/mouse endogenous TE; IVT, delivery and modifications are domain shifts; 1381 nt 5UTR and 11937 nt CDS+3UTR', length_unit='nt', max_length=13318),
    'vienna': dict(role='structure', regions=['full'], source='https://www.tbi.univie.ac.at/RNA/', license='ViennaRNA license', scope='Unmodified-RNA thermodynamic prior at 37 C; full molecule MFE, ensemble and start accessibility; application cap 2000 nt', length_unit='nt', max_length=2000),
    'smart5utr': dict(role='generator', regions=['utr5'], source='https://github.com/deepomicslab/Smart5UTR', license='GPL-3.0 upstream; weights stored separately', scope='50 nt 5UTR, m1Psi, eGFP MPRA-derived MRL objective; reporter/cell transfer needs validation', length_unit='nt', max_length=50),
    'parade': dict(role='generator_predictor', regions=['utr5', 'utr3'], source='https://github.com/autosome-ru/parade', license='MIT code; track exact weights and data', scope='50 nt 5UTR / 240 nt 3UTR; author cell codes c1,c2,c4,c6,c17 (+c13 for 3UTR); reporter activity, not delivery or absolute protein', length_unit='nt', max_length=240),
    'optimus5': dict(role='predictor', regions=['utr5'], source='https://github.com/deepomicslab/Smart5UTR', license='Upstream GPL-3.0 distribution; author checkpoint', scope='50 nt 5UTR, m1Psi eGFP reporter MRL; same MPRA lineage as Smart5UTR, correlated evidence', length_unit='nt', max_length=50),
    'saluki': dict(role='predictor', regions=['full'], source='https://github.com/calico/basenji', license='Apache-2.0 code; official Zenodo 6326409 weights', scope='Full 50-member human/mouse ensemble; 6 channels, synthetic splice sites zero; normalized consensus, not hours', length_unit='nt', max_length=12288),
    'rnadegformer': dict(role='predictor', regions=['full'], source='https://github.com/Shujun-He/RNAdegformer', license='Official author Webapp release', scope='All 10 folds, short-RNA training/long-mRNA domain shift, native RNAsoft BPP at default 37 C (author Arnie ignores T); 5 per-base chemical-assay channels; unmodified RNA only', length_unit='nt', max_length=1024),
    'utr_lm': dict(role='predictor', regions=['utr5'], source='https://github.com/a96123155/UTR-LM', license='GPL-3.0', scope='Official unmodified egfp MRL fold0 head; exactly 50 nt; reporter transfer remains unvalidated', length_unit='nt', max_length=50),
    'codongpt': dict(role='generator', regions=['cds'], source='https://huggingface.co/naniltx/codonGPT', license='Free for research use', scope='GPT2 codon sampler constrained to exact target protein; no functional optimization claim', length_unit='codon_tokens', max_length=1024),
    'utrgan': dict(role='generator', regions=['utr5'], source='https://github.com/ciceklab/UTRGAN', license='CC BY-NC-SA 2.0', scope='Author pretrained WGAN prior; 5UTR diversity, no predictor reward optimization', length_unit='nt', max_length=128),
    'mrnabert': dict(role='representation', regions=['full'], source='https://huggingface.co/YYLY66/mRNABERT', license='Apache-2.0', scope='Mixed nucleotide/codon tokenization; mean hidden representation only; no deployed mammalian functional head', length_unit='tokens', max_length=1024),
    'joint': dict(role='proposal_import', regions=['utr5', 'cds', 'utr3'], source='local provenance-bearing proposal import', license='External provider terms supplied with each proposal', scope='User-supplied external full constructs, strictly evaluated before delivery; no remote generation service implied', length_unit='nt', max_length=None),
    'mrnautilus': dict(role='joint_generator', regions=['utr5', 'cds', 'utr3'], source='https://huggingface.co/atombio/mrnautilus', license='CC BY-NC 4.0; source README badge differs, retain upstream terms', scope='Official 150M weights; bounded 8-step masked diffusion, synonymous CDS; native torch SDPA; no MCTS/reward reranking', length_unit='mixed_tokens', max_length=1024),
    'degscore': dict(role='control_predictor', regions=['full'], source='https://github.com/eternagame/DegScore', license='Upstream license in pinned source', scope='DegScore 2.1 chemical-degradation ridge model, ViennaRNA structure; correlated OpenVaccine evidence, not independent vote', length_unit='nt', max_length=2000),
    'promorna': dict(role='joint_generator', regions=['full'], source='https://arxiv.org/html/2605.01513v2', license='No public deployable checkpoint identified', scope='Paper v2; no authenticated official model/API found in availability audit', length_unit='nt', max_length=None),
    'mrna_gpt': dict(role='joint_generator', regions=['full'], source='https://www.biorxiv.org/content/10.64898/2026.03.31.715707v1', license='No public Sanofi checkpoint identified', scope='Sanofi model; unrelated ZHymLumine/ZYMScott models are not substitutes', length_unit='nt', max_length=None),
}

OPTIONAL_PROBES={
 'saluki':'models/saluki/f0_c0/train/model0_best.h5','rnadegformer':'external/RNAdegformer-Webapp/RNA_Inference/best_weights/fold0top1.ckpt',
 'utr_lm':'external/UTR-LM/Scripts/esm/model/esm2_secondarystructure.py','codongpt':'models/codongpt/pytorch_model.bin',
 'utrgan':'external/UTRGAN/models/checkpoint_3000.h5','mrnabert':'models/mrnabert/pytorch_model.bin',
 'mrnautilus':'models/mrnautilus/best.ckpt','degscore':'external/DegScore/DegScore.py'}


def manifest():
    p = data_root() / "assets.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def capabilities():
    root = data_root()
    probes = {
        "qwen": qwen_path() / "config.json",
        "gemorna": root / "external/GEMORNA/checkpoints/gemorna_cds.pt",
        "lineardesign": root / "external/LinearDesign/bin/LinearDesign_2D",
        "ribonn": root / "models/ribonn/human/runs.csv",
        "smart5utr": root / "external/Smart5UTR/models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5",
        "parade": root / "external/parade/predictor/regression_multiple/saved_models/model-utr5-deltas-epoch=9-step=840.ckpt",
        "optimus5": root / "external/Smart5UTR/models/Optimus5-Prime/Optimus5_egfp_m1pseudo2_model_200k.h5",
    }
    probes.update({k:root/v for k,v in OPTIONAL_PROBES.items()})
    validation_path=root/"model-validation.json"
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    rows = []
    for name, card in MODELS.items():
        ready = name in {"synonymous", "vienna", "joint"} or (name in probes and probes[name].is_file() and probes[name].stat().st_size > 300)
        if name in OPTIONAL_PROBES:ready=ready and validation.get(name,{}).get("passed") is True
        rows.append({"id": name, **card, "status": "installed" if ready else "unavailable", "enabled": ready, "validation": validation.get(name),
                     "reason": None if ready else MODELS[name]["scope"] if name in {"promorna","mrna_gpt","joint"} else "Required checkpoint / inference validation not complete",
                     "version": QWEN if name == "qwen" else manifest().get({"gemorna": "GEMORNA", "ribonn": "RiboNN", "lineardesign": "LinearDesign", "smart5utr": "Smart5UTR", "saluki":"basenji", "utr_lm":"UTR-LM", "rnadegformer":"RNAdegformer-Webapp", "degscore":"DegScore", "utrgan":"UTRGAN"}.get(name, name), {})})
    return {"models": rows, "llm": QWEN, "data_root": str(root), "quota_bytes": MAX_BYTES,
            "data_bytes": used_bytes(), "workflow": ["target", "plan", "references", "generate", "assemble", "validate", "evaluate", "high_potential_filter", "portfolio", "feedback"],
            "evidence_states": ["calculated", "model_prediction", "mechanistic_hypothesis", "measured_this_context", "not_evaluated"]}
