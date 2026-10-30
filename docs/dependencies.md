# Dependencies and installation

[English](dependencies.md) | [简体中文](dependencies_zh.md)

The public snapshot contains source, dependency declarations, and installation
entry points. It does not contain installed environments, third-party source
trees, model checkpoints, the original research datasets, or their local
validation records.

Use a standalone checkout path without a colon, such as `decoupled-rsi-agent`,
for installation and candidate execution. The title-named staging directory is
for publication review. The PFM aggregation replay runs there without installing
any scientific dependencies.

## Primary runtime

The main runtime uses conda and Python 3.12. From the repository root:

```bash
./install.sh --help
./install.sh core
conda activate drug-agent
make doctor
```

`core` enters `agents/drug-agent`, creates or updates the environment from its
`environment.yml`, and installs
the local package with `pip install --no-deps -e .`. The environment does not
include development test dependencies. Review the declaration before installing
into an existing environment.

The equivalent explicit sequence for a new environment is:

```bash
cd agents/drug-agent
conda env create -f environment.yml
conda activate drug-agent
python -m pip install --no-deps -e .
```

## Optional PFM execution

The [recorded PFM example](../examples/pfm_replay/README.md) needs only standard
Python. Running the actual generator requires additional resources:

```bash
./install.sh pfm
./install.sh pfm-source
```

The first command installs the dependency overlay and builds the two graph
extensions against the installed PyTorch. It may require a compiler and a
compatible accelerator toolchain. The second command obtains the upstream PFM
source at the recorded integration revision
`33be6c1f3dc9f42deb7a27be974ab9f5fc241497`.

Acquire PFM's weights and predictor assets separately following its upstream
instructions, and provide a prepared pocket. `.env.example` lists the supported
`DRUGAGENT_PFM_*` settings in `agents/drug-agent/.env.example`. The source installer does not supply the original
target inputs or establish scientific readiness.

## Other optional drug tools

Select a source or environment explicitly:

```bash
./install.sh list
./install.sh external REINVENT4
./install.sh side-env vina
```

`external NAME` clones one entry from
[`CHECKOUTS.tsv`](../agents/drug-agent/scripts/external/CHECKOUTS.tsv) and checks out its recorded
revision. Automatic Git LFS downloads are disabled; ordinary files committed
to the selected upstream repository still come with that checkout.

`side-env NAME` supports `aizynth`, `boltz`, `synflownet`, `unimol`, and `vina`.
Their declarations are in [`envs/`](../agents/drug-agent/envs/). These environments are not
interchangeable, and the SynFlowNet recipe is specifically CUDA/Linux x86-64.
Weights, data, licenses, and backend-specific setup remain separate steps.

The project-owned Python files under `agents/drug-agent/scripts/external/` are adapter drivers,
not bundled third-party plugins. They invoke an upstream installation only when
that installation is configured and available. Consult
[THIRD_PARTY.md](../THIRD_PARTY.md) for source references.

## Component installers and file layouts

The [`third_party/`](../third_party/) directory has one installer and a
machine-readable dependency manifest per Agent family. Each manifest lists
source revisions, environment dependencies, expected files, configuration
variables, and any unavailable installation stages. The adjacent README expands
these into a directory structure and executable commands.

```bash
./install.sh dependency drug --help
./install.sh dependency peptide --help
./install.sh dependency mrna --help
./install.sh dependency enzyme --help
```

Planning and file checks do not download or import models. A present file is not
proof that a backend has passed its scientific validation. Components requiring
separately licensed assets or local validation report those requirements rather
than being marked ready.

## Experimental Task Agents

Use the per-project installation guide and environment rather than adding all
dependencies to the primary conda environment:

- [Peptide Agent](../agents/peptide-agent/README.md)
- [mRNA Agent](../agents/mRNA-agent/README.md)
- [Enzyme Agent](../agents/Enzyme-Agent/README.md)

Their install scripts may fetch substantial external assets when explicitly
invoked. Local readiness evidence for some scientific backends is intentionally
not distributed. A backend may remain unavailable pending validation even after
its dependencies and weights have been installed.

## What has been checked for this copy

Publication checks cover file scope, syntax and declarations, installation help,
and the standard-library PFM replay. A fresh scientific-environment installation
and new model/GPU executions have not been performed for this copy. The full
platform evolution loop still requires the regression gate described in
[self-evolution.md](self-evolution.md).

