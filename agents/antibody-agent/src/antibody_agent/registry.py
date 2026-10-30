"""Capabilities are configured runtimes, not claims that a paper is integrated."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from .config import data_root
from .contracts import Unavailable

QWEN = {
    "repo": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "parameters": "0.8B",
    "role": "planner",
    "license": "Apache-2.0",
}
MODELS = {
    "rfantibody": (
        "generator",
        "RosettaCommons/RFantibody",
        "MIT",
        "CDR backbone + ProteinMPNN + RF2; variable domains",
    ),
    "boltzgen": ("generator", "HannesStark/boltzgen", "MIT", "Antibody/nanobody CDR sequence and structure"),
    "iggm": ("generator", "TencentAI4S/IgGM", "MIT", "Epitope-conditioned variable-domain design"),
    "igdesign": ("generator", "AbSciBio/igdesign", "MIT", "CDR inverse folding; requires a trusted complex"),
    "antifold": ("generator", "oxpig/AntiFold", "BSD-3-Clause", "Fixed-backbone antibody sequence redesign"),
    "p_iggen": (
        "generator",
        "OliverT1/p-IgGen",
        "BSD-3-Clause",
        "Paired sequence prior; no native antigen conditioning",
    ),
    "lichen": ("generator", "oxpig/LICHEN", "BSD-3-Clause", "Light-chain and common-light-chain proposals"),
    "sapiens": ("generator", "Merck/Sapiens", "MIT", "Humanization proposals; not clinical immunogenicity"),
    "humatch": ("generator", "oxpig/Humatch", "BSD-3-Clause", "Paired humanization"),
    "immunebuilder": (
        "structure_predictor",
        "oxpig/ImmuneBuilder",
        "BSD-3-Clause",
        "ABodyBuilder2 / NanoBodyBuilder2; unbound fold",
    ),
    "boltz2": (
        "structure_predictor",
        "jwohlwend/boltz",
        "MIT",
        "Independent complex prediction; no antibody affinity head",
    ),
    "chai1": (
        "structure_predictor",
        "chaidiscovery/chai-lab",
        "Apache-2.0",
        "Complementary complex prediction",
    ),
    "tap": (
        "property_predictor",
        "oxpig/TAP",
        "verify deployed code and service terms",
        "Therapeutic antibody profiling; format-specific reference",
    ),
    "tnp": (
        "property_predictor",
        "oxpig/TNP",
        "verify deployed revision",
        "Nanobody profiling; not an IgG benchmark",
    ),
    "anarcii": ("numbering", "oxpig/ANARCII", "BSD-3-Clause", "IMGT chain numbering with reversible mapping"),
}
BUILTINS = {
    "rfantibody": "rfantibody",
    "immunebuilder": "ImmuneBuilder",
    "boltz2": "boltz",
    "anarcii": "anarcii",
    "antifold": "antifold",
}


def qwen_path():
    return Path(
        os.environ.get("ANTIBODY_QWEN_DIR", str(data_root() / "models/qwen" / QWEN["revision"]))
    ).expanduser()


def planner_runtime():
    return {
        **QWEN,
        "python": os.environ.get("ANTIBODY_LLM_PYTHON", sys.executable),
        "checkpoint": str(qwen_path()),
    }


def runtimes():
    path = Path(os.environ.get("ANTIBODY_RUNTIMES", str(data_root() / "runtimes.json")))
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) - set(MODELS):
        raise ValueError("Runtime configuration contains unknown models")
    for name, cfg in value.items():
        if not isinstance(cfg, dict) or not cfg.get("version") or not cfg.get("weights_version"):
            raise ValueError(f"{name}: record code and weight versions")
        argv = cfg.get("command")
        if argv is not None and (
            not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv)
        ):
            raise ValueError(f"{name}: command must be a nonempty argv array, never a shell string")
        if argv is None and name not in BUILTINS:
            raise ValueError(f"{name}: an operator-installed JSON bridge command is required")
    return value


def runtime(name):
    cfg = runtimes().get(name)
    if not cfg:
        raise Unavailable(
            f"{name} is not configured; install its backend and register code/weight versions in runtimes.json"
        )
    return cfg


def capabilities():
    configured = runtimes()
    cards = []
    for name, (role, repo, license_, scope) in MODELS.items():
        cfg = configured.get(name)
        available = False
        if cfg:
            if cfg.get("command"):
                available = bool(shutil.which(cfg["command"][0]))
            else:
                python = cfg.get("python", sys.executable)
                available = bool(shutil.which(python)) and (
                    python != sys.executable or importlib.util.find_spec(BUILTINS[name]) is not None
                )
        source = (
            "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/tap"
            if name == "tap"
            else "https://github.com/" + repo
        )
        cards.append(
            {
                "id": name,
                "role": role,
                "source": source,
                "license": license_,
                "scope": scope,
                "status": "configured_unverified" if available else "unavailable",
                "native_adapter": name in BUILTINS,
                "version": cfg.get("version") if cfg else None,
                "weights_version": cfg.get("weights_version") if cfg else None,
                "training_scope": (cfg or {}).get("training_scope", "unknown; potential contamination"),
                "runtime_status_note": "Installation presence is not a successful inference or scientific validation",
            }
        )
    path = qwen_path()
    return {
        "planner": {
            **QWEN,
            "status": "weights_present" if (path / "config.json").is_file() else "unavailable",
        },
        "models": cards,
        "data_root": str(data_root()),
        "delivery": "Independent evaluation before human review; fail closed",
    }
