# Enzyme Task Agent

[English](README.md) | [简体中文](README_zh.md)

**Experimental and incomplete. Further evolution and validation are needed.**

A Task Agent for therapeutic enzyme research: typed reaction and product
requirements, sequence design, independent structural assessment, experimental
feedback, and local strategy review. The package does not yet provide a complete
adapter to the shared Self-Evolve Agent.

## Install

Use Linux. Install the sibling `peptide-agent` Python 3.11 runtime and its Qwen
model first. From this directory:

```bash
export PEPTIDE_PYTHON="$(cd ../peptide-agent && pwd)/.venv/bin/python"
export ENZYME_DATA_DIR="$HOME/.cache/decoupled-rsi/enzyme"
bash install.sh
bash run.sh doctor
bash run.sh serve --port 2006
```

The installer uses the peptide interpreter to create a venv, ensures its declared
base packages are available, installs the additional packages in `requirements.txt`,
downloads pinned core models, and fetches the
DNase reference from UniProt/PDB. It requires network access and substantial
storage. Scientific generation and folding need a compatible GPU runtime.
Model downloads and generated records count toward the application's
50,000,000,000-byte storage limit.

Third-party source, weights, reference structures, and prior runs are not bundled.
Model URLs and revisions are in `src/enzyme_agent/models.py`, `extended_models.py`,
and [model sources](docs/model-sources.md). Downloaded components retain their
upstream licenses.

## Reference example and optional backends

`scripts/prepare_dnase.py` generates `examples/dnase_reference.json` and
`examples/dnase_design.json` from the public reference resources during installation:

```bash
bash run.sh run examples/dnase_reference.json
```

Required evaluation backends may keep a workflow blocked until installed and
validated. RFdiffusion3 and ThermoMPNN use `scripts/bootstrap_extended.sh`;
set `ENZYME_FOUNDRY_BASE` to a compatible Python 3.12 interpreter with CUDA Torch
before invoking it. This separate install downloads additional assets.

Readiness gates are preserved. Internal inference-validation scripts and historic
readiness records are not published, so downloading extended models alone does
not mark them ready. Local validation and complete fresh-install support remain
work in progress; missing evidence produces an unavailable state.

## Configuration and code

- `ENZYME_DATA_DIR`: environments, models, and new run records; defaults to
  `~/.cache/decoupled-rsi/enzyme`.
- `PEPTIDE_PYTHON`: base interpreter; defaults to the sibling peptide `.venv`.
- `ENZYME_PYTHON`: override the installed enzyme interpreter used by `run.sh`.
- `ENZYME_QWEN_DIR`: override the shared Qwen snapshot.
- `ENZYME_FOUNDRY_BASE`: required base interpreter for the optional Foundry env.
- `workflow.py`, `gateway.py`, and worker modules: execution and tool boundaries.
- `evaluation.py`, `tool_evaluation.py`, and `evolution.py`: scientific evaluation
  and local selection-policy evolution.

See [architecture](docs/architecture.md). Structure confidence, static geometry,
and predicted stability do not establish catalytic activity or therapeutic
benefit. The public copy has not been installed or scientifically revalidated
as part of the packaging step.

## Third-party installation inventory

See the [component/stage installer](../../third_party/enzyme/README.md) for
source revisions, model/data layouts, offline plan/check commands, and explicit
acquisition entrypoints. `plan` and `check` never download or run models.
