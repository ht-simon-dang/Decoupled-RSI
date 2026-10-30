# Antibody dependencies

[English](README.md) | [简体中文](README_zh.md)

This inventory describes the implemented Antibody Agent, using its
[package dependencies](../../agents/antibody-agent/pyproject.toml),
[model registry](../../agents/antibody-agent/src/antibody_agent/registry.py), and
[native adapters](../../agents/antibody-agent/src/antibody_agent/adapters.py).
[dependencies.json](dependencies.json) lists package requirements, the pinned
0.8B planner, upstream sources, runtime fields, expected assets and integration
limits. It is a declarative inventory, not an executable installer/checker.

No source checkout, environment, checkpoint, assay data or local validation record
is redistributed here. Model licenses below reproduce the agent's registry, not
a new license review; deployed code, weights, reference panels and services can
have separate terms.

## Existing installation entrypoints

Run these from the repository root only when installation is wanted:

```bash
# Lightweight agent and WebUI dependencies; no molecular models:
./agents/antibody-agent/install.sh base
# Optional local 0.8B planner dependencies and explicit weight download:
./agents/antibody-agent/install.sh llm
./agents/antibody-agent/run.sh download-qwen
# Optional MCP transport or developer checks:
./agents/antibody-agent/install.sh mcp
./agents/antibody-agent/install.sh test
# Inspect declared model configuration without running inference:
./agents/antibody-agent/run.sh doctor
```

The installer requires Python `>=3.11,<3.13`; `ANTIBODY_BOOTSTRAP_PYTHON` selects
the environment-creation interpreter. Runtime packages are Pydantic, FastAPI,
Uvicorn, PyYAML, Biopython and NumPy. The `llm` extra adds Torch, Transformers,
LM Format Enforcer and Hugging Face Hub. Exact version ranges are recorded in the
JSON inventory and remain authoritative in `pyproject.toml`.

No `third_party/antibody/install.sh`, generic `plan/check/install` command, or root
dispatcher integration is introduced by this dependency description. The above
commands use the agent's existing implementation. The molecular backends below
need separate compatible environments, assets and configuration.

## Planner and scientific backends

| Component | Integration | Required resources / scope |
| --- | --- | --- |
| Qwen3.5-0.8B | Native planner | `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`; full config, tokenizer and safetensors snapshot |
| ANARCII | Native numbering | Installed `anarcii` package and its model assets; complete IMGT variable-domain numbering |
| RFantibody | Native generation | Compatible `rfdiffusion`, `proteinmpnn`, `rf2` executables; diffusion, base ProteinMPNN and RF2 weights; HLT framework and target input |
| AntiFold | Native generation | Installed `antifold`, configured checkpoint, IMGT-labelled paired/VHH structure and explicit editable positions |
| ImmuneBuilder | Native fold | ABodyBuilder2 / NanoBodyBuilder2, numbering dependencies, OpenMM/pdbfixer and four format-specific weights |
| Boltz-2 | Native complex prediction | `boltz` executable beside the selected Python, populated cache and explicit checkpoint; full target sequence |
| BoltzGen, IgGM, IgDesign | JSON generation bridges | Separately deployed generator, its weights and a protocol-compatible bridge; registration is not native integration |
| p-IgGen, LICHEN | JSON generation bridges | Paired/light-chain model and bridge; sequence priors alone do not establish antigen binding |
| Sapiens, Humatch | JSON generation bridges | Humanization model and bridge; variants still need independent evaluation |
| Chai-1 | JSON evaluation bridge | Independently deployed complex predictor and bridge; used for BoltzGen-origin candidates |
| TAP / TNP | JSON profiling bridges | Actual format-appropriate profiler, reference panel and fixed interpretation rules; sequence motifs are not a substitute |

Choose generators for the declared route; all generators are not mandatory in one
project. Candidate delivery requires valid numbering, independent folding,
format-specific TAP/TNP evidence, and independent positive-state complex/epitope
evaluation. Missing required resources withhold candidates. A configured runtime,
an available weight file or a successful smoke test does not certify binding,
function or scientific readiness.

The Boltz-2 adapter uses single-sequence mode, without templates, forced contacts,
remote MSA requests or the small-molecule affinity head. Its protein-only output
does not establish membrane/glycan compatibility. For native-context checks and
each adapter's input boundaries, see the
[adapter contract](../../agents/antibody-agent/docs/model-adapters.md).

## Configuration and file layout

Checkout defaults:

```text
agents/antibody-agent/
  .venv/bin/python
  configs/runtimes.example.json          # public schema example, not installed assets
  .antibody-agent/                       # ANTIBODY_DATA_DIR
    runtime.env                         # trusted, ignored launcher settings
    runtimes.json                       # actual per-model runtime configuration
    models/qwen/2fc06364715b967f1860aea9cf38778875588b17/
      config.json
      <complete tokenizer/config files>
      <all safetensors shards and index if sharded>
    runs/                               # local evidence; not a dependency download
    property_models/                    # models trained from actual project measurements
```

Backend asset locations are explicitly configured, not hardcoded under this
tree. Examples of the native asset contracts are:

```text
<rfantibody.bin_dir>/{rfdiffusion,proteinmpnn,rf2}
<antifold.checkpoint>
<immunebuilder.weights_dir>/antibody_model_{1,2,3,4}  # VH/VL
<immunebuilder.weights_dir>/nanobody_model_{1,2,3,4}  # VHH
<boltz2.python directory>/boltz
<boltz2.cache>/<backend-required cache assets>
<boltz2.checkpoint>
<bridge command executable> [<bridge script and arguments>]
```

Every configured scientific model requires actual `version` and
`weights_version` values. Native adapters use their own Python or the active
interpreter; each must import the agent's base dependencies and backend packages.
Bridge `command` values are argv arrays, never shell strings. Asset paths should
be absolute. Copy and complete only installed entries from
[runtimes.example.json](../../agents/antibody-agent/configs/runtimes.example.json).

| Override | Meaning |
| --- | --- |
| `ANTIBODY_PYTHON` | Agent interpreter; default `<agent>/.venv/bin/python` |
| `ANTIBODY_BOOTSTRAP_PYTHON` | Interpreter used by `install.sh` to create the environment |
| `ANTIBODY_DATA_DIR` | Runtime assets/state; launcher defaults to `<agent>/.antibody-agent` |
| `ANTIBODY_RUNTIMES` | Model configuration; default `<data>/runtimes.json` |
| `ANTIBODY_QWEN_DIR` | Existing complete 0.8B snapshot, optionally reused read-only |
| `ANTIBODY_LLM_PYTHON` | Existing compatible planner interpreter; default active Python |

The launcher reads `<agent>/.antibody-agent/runtime.env`, not the repository-wide
runtime file. Reusing another agent's Qwen snapshot/environment requires these
explicit overrides; sibling agents are not mandatory dependencies. Direct wheel
use without the launcher falls back to the XDG/user cache data directory when
there is no source checkout. `--root` changes only the run journal directory.

## Project inputs and readiness

Target sequences/states, paired VH/VL or VHH, mapped framework coordinates,
functional epitopes, constraints and reference-panel provenance are supplied for
each project. They are not bundled model assets. Full-length IgG, advanced-format
assembly, insertion/deletion mappings and corresponding validation must not be
inferred from a variable-domain-only dependency setup.

The experimental learning baseline uses NumPy and real, condition-matched assay
data with frozen train/calibration/test splits. It does not require or pretend
to provide pretrained LaMBO-2 project-property weights. Generated models, assay
records and hardware-specific readiness evidence remain private local artifacts.

`doctor` reports configuration availability (`unavailable`,
`configured_unverified`) and planner weight-config presence. It does not run GPU
models, check every checkpoint shard or produce scientific readiness records.
Validate each actual backend with its declared inputs before relying on it.
