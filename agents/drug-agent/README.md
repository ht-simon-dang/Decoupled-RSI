# Drug Agent

[English](README.md) | [简体中文](README_zh.md)

The primary Task Agent in Decoupled RSI Agent. This project also contains the
current Python implementation of the independent Self-Evolve lifecycle under
[`src/drug_agent/selfevolve/`](src/drug_agent/selfevolve/).

## Install and run

From this project directory, in a checkout whose path contains no colon:

```bash
./install.sh --help
./install.sh core
conda activate drug-agent
drugagent doctor
make demo
```

The package name remains `drug-agent`, the import is `drug_agent`, and the CLI
is `drugagent`. The offline demo uses [synthetic fixtures](fixtures/data/DATA_CARD.md).
Optional scientific backends require their own resources and may report
unavailable on a fresh installation.

## Project files

```text
drug-agent/
  src/drug_agent/       task execution, tools, evaluation, and self-evolution
  configs/             runtime configuration templates
  system_prompts/      packaged prompts
  schemas/             public interface contracts
  fixtures/            small offline demonstration inputs
  scripts/external/    first-party drivers for separately installed backends
  scripts/research/    campaign dispatch and target preparation entrypoints
  environment.yml      primary runtime dependencies
  environment-pfm.yml  optional PFM dependency overlay
  envs/                optional isolated environment recipes
  install.sh           explicit installation stages
```

Use [third_party/drug](../../third_party/drug/README.md) for the component
installers, source references, required file trees, and readiness checks.
No external source checkout, weight, or original campaign data is bundled.

The repository-level Git checkout contains this project at
`agents/drug-agent`. Candidate workspaces distinguish that project root from
the Git root. The full evolution loop retains its regression gate; the private
development suite is not distributed and the gate cannot pass without a
compatible supplied suite. See the [evolution guide](../../docs/self-evolution.md).

[Repository overview](../../README.md) · [License](LICENSE)
