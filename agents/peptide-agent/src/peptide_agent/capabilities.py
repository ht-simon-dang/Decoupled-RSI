import importlib.util
import json

from .models import MODELS, model_path, model_root
from .contracts import Unavailable
from .structural import boltz_paths


def capabilities():
    def probe(fn):
        try: fn(); return "ready"
        except (Unavailable, OSError): return "missing_weights"
    qwen = probe(lambda: model_path("qwen"))
    pep = probe(lambda: model_path("pepmlm"))
    bg = probe(boltz_paths)
    tox = "ready" if (model_root()/"toxinpred3/trees.npz").is_file() else "missing_weights"
    return {"planner": {"name": "Qwen3.5-0.8B", "status": qwen}, "tools": [
        {"id": "generation.pepmlm", "name": "PepMLM-650M", "status": pep, "domain": "Linear sequence design"},
        {"id": "generation.boltzgen", "name": "BoltzGen-1", "status": bg, "domain": "Linear / cyclic all-atom design + optional Boltz2 refolding"},
        {"id": "sequence.analyze", "name": "ProtParam + RDKit", "status": "ready", "domain": "Topology-aware canonical peptide chemistry"},
        {"id": "structure.contacts", "name": "Contact geometry", "status": "ready", "domain": "PDB heavy-atom contacts and clashes"},
        {"id": "safety.toxinpred3", "name": "ToxinPred3", "status": tox, "domain": "Linear canonical peptides; ML-only"},
        {"id": "affinity.prodigy", "name": "PRODIGY", "status": "ready" if importlib.util.find_spec("prodigy_prot") else "missing_package",
         "domain": "Structure-based ΔG / Kd estimate; peptide extrapolation"},
        {"id": "affinity.calibrated", "name": "Measured pKd calibration", "status": "ready" if list((model_root()/"calibrations").glob("*.json")) else "needs_data",
         "domain": "Group-disjoint train / calibration / test"}],
        "research_backends": [{"name": "PepTri / PepFGLD", "status": "not_integrated",
            "reason": "No verified downloadable inference weights in reviewed upstream releases; not advertised as executable."}]}
