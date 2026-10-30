# mRNA Task Agent

[English](README.md) | [简体中文](README_zh.md)

**Experimental and incomplete. Further evolution and validation are needed.**

A Task Agent for mRNA design, evaluation, and experimental feedback. Typed planning,
regional generators, scientific predictors, independent evaluation, and versioned
run records are separate components. Its local strategy review is not a complete
integration with the shared Self-Evolve Agent.

## Install

Use Linux. Install the sibling `peptide-agent` Python 3.11 runtime and its Qwen
model first. Then, from this directory:

```bash
export PEPTIDE_PYTHON="$(cd ../peptide-agent && pwd)/.venv/bin/python"
export MRNA_DATA_DIR="$HOME/.cache/decoupled-rsi/mrna"
bash install.sh
"$MRNA_DATA_DIR/venv/bin/python" -m mrna_agent.cli doctor
```

The installer creates a venv using the peptide interpreter, installs its declared
runtime packages, downloads pinned upstream code and model assets, builds LinearDesign, and creates separate
compatibility environments. It needs network access, conda for the Python 3.10
GEMORNA runtime, a C/C++ build toolchain, and substantial storage. Installation
is subject to the application's 50,000,000,000-byte storage limit. Check upstream
licenses before downloading; code and model terms differ by dependency.

The public copy contains dependency declarations and download adapters, not the
upstream sources, weights, datasets, or prior run records. Base dependencies are
in `requirements.txt`, legacy dependencies in `requirements-legacy.txt`, and
source revisions in `scripts/bootstrap.py` and `scripts/optional-pins.json`.

## Run

```bash
"$MRNA_DATA_DIR/venv/bin/python" -m mrna_agent.cli run examples/hbb.json
"$MRNA_DATA_DIR/venv/bin/python" -m mrna_agent.cli serve --port 2005
```

The supplied JSON files are input templates, not saved experimental results.
Tools can remain unavailable after download until their required readiness
checks are satisfied. The internal validation scripts and records are excluded
from this release; the installer does not invent replacement readiness records.
Optional model activation and fresh-install validation remain incomplete.

## Configuration and code

- `MRNA_DATA_DIR`: environments, assets, and new run records; defaults to
  `~/.cache/decoupled-rsi/mrna`.
- `PEPTIDE_PYTHON`: the peptide base interpreter; the installer defaults to the
  sibling project's `.venv/bin/python`.
- `MRNA_QWEN_DIR`: explicit Qwen snapshot; otherwise use the sibling peptide
  installation or the local model directory.
- `MRNA_ESMFOLD_DIR`: optional shared ESMFold2 snapshot. Its default follows
  `ENZYME_DATA_DIR` (or `~/.cache/decoupled-rsi/enzyme`).
- `MRNA_STRUCTURE_GPU_LOCK`: override the GPU lock shared with Enzyme.
- `src/mrna_agent/workflow.py`: task lifecycle; `gateway.py` and worker modules:
  scientific adapters; `evaluation.py`, `delivery.py`, and `benchmark.py`:
  evaluation and strategy review.

See [architecture](docs/architecture.md), [optional models](docs/optional-models.md),
[delivery policy](docs/high-potential-policy.md), and [history preview](docs/history-preview.md).
Predictor scores, RNA folding measures, and protein structure confidence do not
establish expression or efficacy in a new biological setting.

## Third-party installation inventory

See the [component/stage installer](../../third_party/mrna/README.md) for
source revisions, model/data layouts, offline plan/check commands, and explicit
acquisition entrypoints. `plan` and `check` never download or run models.
