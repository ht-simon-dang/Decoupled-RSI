# EMA4D Antibody Agent

[简体中文](README_zh.md)

An independent package implementing the antibody blueprint's project, generation,
evaluation, evidence, feedback and human-review workflow. All code and default
runtime state live in this directory. It does not modify sibling agents or the
repository-wide launcher.

```bash
./agents/antibody-agent/install.sh
./agents/antibody-agent/run.sh serve --port 2007
./agents/antibody-agent/run.sh run agents/antibody-agent/examples/disease.json
./agents/antibody-agent/run.sh doctor
```

Open **http://127.0.0.1:2007**. The workbench has structured input, Qwen conversation,
candidate dossiers, a local structure viewer, independent evaluation, explicit
human decisions, experiment feedback and model availability.

The planner is pinned to **Qwen/Qwen3.5-0.8B**, revision
`2fc06364715b967f1860aea9cf38778875588b17`. Structured briefs do not require an LLM.
Install the optional planner dependencies and weights with:

```bash
./agents/antibody-agent/install.sh llm
./agents/antibody-agent/run.sh download-qwen
./agents/antibody-agent/run.sh instruct 'Explore a therapeutic antibody research hypothesis and its evidence gaps.'
```

Existing installations can be reused read-only with `ANTIBODY_QWEN_DIR` and
`ANTIBODY_LLM_PYTHON`. The launcher reads trusted, ignored machine settings from
`.antibody-agent/runtime.env`. Do not put machine-specific paths in public files.

**Scientific backends are separate installations.** Native adapters are provided
for RFantibody, AntiFold, ANARCII, ImmuneBuilder and Boltz-2. The other registered
generators, Chai-1 and format-specific TAP/TNP profiles use the versioned JSON
bridge protocol in [model-adapters.md](docs/model-adapters.md). Registration alone
does not mean a model is installed, its inference has succeeded, or its scientific
claims have been validated. Missing mandatory tools withhold candidates.

The human delivery boundary is shared by CLI, WebUI, MCP, JSON/CSV/FASTA/report
exports, structures and active-learning recommendations. It rechecks immutable
chain/product identities, antigen context, tool-input/output hashes and the
frozen evaluator. Generation results, failed evaluations and private checkpoints
cannot bypass this boundary. A successful computational nomination still needs a
human decision and experimental validation in the final format.

For installation contracts, supported routes and current scientific limitations,
see the [Chinese guide](README_zh.md), [architecture](docs/architecture_zh.md) and
[model adapters](docs/model-adapters.md).

```bash
./agents/antibody-agent/install.sh test
cd agents/antibody-agent
make test
```

Tests use explicitly synthetic protocol fixtures; these are never production
models or scientific demonstrations. Source distributions and wheels include
only the explicit public resources listed in `pyproject.toml`.
