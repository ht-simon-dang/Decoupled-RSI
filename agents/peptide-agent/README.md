# Peptide Task Agent

[English](README.md) | [简体中文](README_zh.md)

**Experimental and incomplete. Further evolution and validation are needed.**

A Task Agent for peptide design and analysis, included in Decoupled RSI Agent.
It separates typed planning, scientific tool calls, independent checks, and
persistent evidence. The package has its own workflow and evaluation machinery;
it is not yet a complete adapter to the shared Self-Evolve Agent.

## Install and run

Use Linux and Python 3.11. From this directory:

```bash
bash install.sh
.venv/bin/peptide-agent doctor
.venv/bin/peptide-agent run examples/library.yaml
.venv/bin/peptide-agent serve --port 2004
```

`install.sh` creates `.venv` and installs the runtime dependencies. Set
`PEPTIDE_BOOTSTRAP_PYTHON` or `PEPTIDE_ENV_DIR` to choose another interpreter or
virtual environment. A conda alternative is `conda env create -f environment.yml`,
followed by `conda run -n peptide-agent python -m pip install -e . --no-deps`.
The package metadata alone does not install the scientific dependency stack.

The library example analyzes supplied sequences. Planning and learned generation
require separate model downloads; for example:

```bash
.venv/bin/peptide-agent download qwen
.venv/bin/peptide-agent download pepmlm
```

The source of each model and its pinned revision are declared in
`src/peptide_agent/models.py` and `downloads.py`. Additional backends include
BoltzGen and ToxinPred3. No model weights, third-party source checkouts, or previous
run records are included. Upstream licenses apply to downloaded components.

## Configuration and code

- `PEPTIDE_MODEL_DIR`: model directory; defaults to `.peptide-agent/models`.
- `--store PATH`: local run store; defaults to `.peptide-agent`.
- `configs/mcp.json`: MCP configuration template; replace its placeholder paths.
- `src/peptide_agent/workflow.py`: task execution and checkpointing.
- `gateway.py`, `policy.json`, and `skills/`: typed tool boundary.
- `analysis.py`, `benchmark.py`, and `calibration.py`: runtime evaluation.

Keep this directory beside `mRNA-agent` and `Enzyme-Agent` if they share its Qwen
installation. When models live elsewhere, configure `MRNA_QWEN_DIR` and
`ENZYME_QWEN_DIR` explicitly.

See [architecture](docs/architecture.md) and [model selection](docs/model-selection.md).
The adapters return unavailable when required assets are absent. Computational
outputs do not establish biological efficacy. The public copy excludes the
internal test suite and validation records; a fresh scientific installation has
not been validated as part of this packaging step.

## Third-party installation inventory

See the [component/stage installer](../../third_party/peptide/README.md) for
source revisions, model/data layouts, offline plan/check commands, and explicit
acquisition entrypoints. `plan` and `check` never download or run models.
