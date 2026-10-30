# Third-party installation and file dependencies

[English](THIRD_PARTY.md) | [简体中文](THIRD_PARTY_zh.md)

Each Agent family has an executable installer, a machine-readable dependency
manifest, and a file-layout guide. The installers connect to the actual adapter
requirements and existing resource fetchers. Help, plans, and presence checks do
not install environments or download models.

No third-party source, checkpoint, dataset, or installed environment is bundled.
The first-party adapters and installation scripts remain in this repository.
External components retain their own licenses and asset-access requirements.

## Installation entrypoints

| Agent family | Installer | Inspect a plan | Check expected files | Explicit installation example |
| --- | --- | --- | --- | --- |
| Drug | [`install.sh`](third_party/drug/install.sh) | `./third_party/drug/install.sh --plan pfm` | `./third_party/drug/install.sh --check pfm` | `./third_party/drug/install.sh pfm source` |
| Peptide | [`install.sh`](third_party/peptide/install.sh) | `./third_party/peptide/install.sh plan qwen` | `./third_party/peptide/install.sh check qwen` | `./third_party/peptide/install.sh install qwen` |
| mRNA | [`install.sh`](third_party/mrna/install.sh) | `./third_party/mrna/install.sh plan gemorna` | `./third_party/mrna/install.sh check gemorna` | `./third_party/mrna/install.sh install core` |
| Enzyme | [`install.sh`](third_party/enzyme/install.sh) | `./third_party/enzyme/install.sh plan ligandmpnn` | `./third_party/enzyme/install.sh check ligandmpnn` | `./third_party/enzyme/install.sh install ligandmpnn` |

Run an installer's `--help` first. The root dispatcher also forwards these
commands, for example `./install.sh dependency drug --plan pfm`.
The Self-Evolve entrypoint shares the [Drug Agent runtime](agents/drug-agent/environment.yml)
and separately configured reasoning backends; see its
[dependency guide](agents/self-evolve-agent/README.md).

Drug integrations expose separate `source`, `environment`, and `assets` stages.
A stage that needs a manually obtained asset refuses with an explanation. For
example, PFM's source and environment are installable, while its checkpoint and
prepared pocket must be supplied according to the manifest.

The biological Agent installers name individual components or explicit grouped
stages. Some mRNA backends share the `core` or `optional` installation stage;
`plan` reports the entire group before it is run. Placeholder research backends
and missing local validation are recorded as unavailable requirements, not
successful installations. Prerequisites appear in each component's manifest.

## Repository and runtime resource layout

The checked-in dependency machinery is:

```text
third_party/
  drug/
    install.sh
    installer.py
    dependencies.json
    README.md
  peptide/                 # mrna/ and enzyme/ use the same file organization
    install.sh
    manage.py
    dependencies.json
    README.md
agents/
  drug-agent/              # project-owned adapters, scripts, and environment declarations
  peptide-agent/
  mRNA-agent/
  Enzyme-Agent/
  self-evolve-agent/       # invocation entrypoint using the Drug Agent runtime
```

The files below are expected after the selected installation/resource steps.
They are **not distributed in this source snapshot**. Full component trees,
revisions, alternatives, environment overrides, and readiness records are listed
in the linked family manifests.

### PFM example

```text
agents/drug-agent/.external/PFM/
  scripts/sampling.py
  egnn/DotGat.py
  utils/reconstruct.py
  model_params/
    pfm.pth
    training.yml
    x_predictor.pth         # when used by the configured sampler
    h_predictor.pth         # when used by the configured sampler
<prepared-pocket>.pdb       # supplied via the task or DRUGAGENT_PFM_POCKET
```

`pfm source` retrieves the pinned upstream checkout. `pfm environment` installs
the dependency overlay and compiled extensions after the main runtime exists.
The manifest maps `DRUGAGENT_PFM_ROOT`, checkpoint, training configuration,
predictor, interpreter, device, and pocket overrides to the expected inputs.
The standard-library [PFM replay](examples/pfm_replay/README.md) needs none of
these external resources.

### Shared Peptide/Qwen example

```text
agents/peptide-agent/
  .venv/bin/python
  .peptide-agent/models/
    qwen/<pinned-revision>/
      config.json
      <tokenizer files>
      <weight files>
    pepmlm/<pinned-revision>/
      config.json
      <weight files>
```

The manifests use `PEPTIDE_ENV_DIR`, `PEPTIDE_PYTHON`, and `PEPTIDE_MODEL_DIR`
where supported. mRNA and Enzyme can reuse the Qwen snapshot or use their
explicit model-path overrides. Installation does not imply that every shared
backend has passed local inference validation.

### mRNA example

```text
${MRNA_DATA_DIR}/            # default: ~/.cache/decoupled-rsi/mrna
  venv/bin/python
  envs/gemorna/bin/python
  external/
    GEMORNA/
      src/generate.py
      src/shared/*.so
      vocab/*.pkl
      checkpoints/gemorna_{cds,5utr,3utr}.pt
    RiboNN/
  models/ribonn/human/
    runs.csv
    <ensemble member>/state_dict.pth
  assets.json
```

Additional predictors have their own pinned code, model files, and local
readiness requirements in [the mRNA manifest](third_party/mrna/dependencies.json).
The published installer does not fabricate the excluded historical validation
records.

### Enzyme example

```text
${ENZYME_DATA_DIR}/          # default: ~/.cache/decoupled-rsi/enzyme
  env/bin/python
  models/
    ligandmpnn/
      source/{run.py,model_utils.py,data_utils.py,openfold/}
      ligandmpnn_v_32_010_25.pt
      manifest.json
    thermompnn/
      source/transfer_model.py
      source/models/thermoMPNN_default.pt
      source/vanilla_model_weights/v_48_020.pt
      manifest.json
      runtime-validation.json   # local validation evidence, not supplied or invented
```

The MPNN fetcher retrieves a shared bundle covering LigandMPNN and ProteinMPNN;
its plan declares that scope. RFdiffusion3 uses a separate Foundry environment.
See [the Enzyme manifest](third_party/enzyme/dependencies.json) for its exact
source, checkpoint, environment, and validation requirements.

## Full component indexes

### Drug Agent

Installer: [`third_party/drug/install.sh`](third_party/drug/install.sh). [File layouts and usage](third_party/drug/README.md) · [Machine-readable dependencies](third_party/drug/dependencies.json).

Declared components: `admetai`, `aizynthfinder`, `boltz`, `chemprop`, `depmap`, `diffdock`, `drugflow`, `e3moldiffusion`, `experimental`, `flowdock`, `flowmol`, `flowr`, `flowr_root`, `fpocket`, `gnina`, `molcraft`, `moljo`, `molpilot`, `omtra`, `openfe`, `opentargets`, `p2rank`, `pfm`, `pocket2mol`, `pocketflow`, `pocketxmol`, `posebench`, `posebusters`, `prexsyn`, `protenix`, `reasyn`, `reinvent4`, `rxnflow`, `safc`, `shepherd`, `shepherd-score`, `synflownet`, `synformer`, `targetdiff`, `uni-mol`, `vina`, `vnegnn`.

### Peptide Agent

Installer: [`third_party/peptide/install.sh`](third_party/peptide/install.sh). [File layouts and usage](third_party/peptide/README.md) · [Machine-readable dependencies](third_party/peptide/dependencies.json).

Declared components: `runtime`, `qwen`, `pepmlm`, `boltzgen`, `toxinpred3`, `prodigy`, `sequence-chemistry`, `calibrated-affinity`, `research-backends`.

### mRNA Agent

Installer: [`third_party/mrna/install.sh`](third_party/mrna/install.sh). [File layouts and usage](third_party/mrna/README.md) · [Machine-readable dependencies](third_party/mrna/dependencies.json).

Declared components: `peptide-runtime`, `runtime`, `core`, `optional`, `qwen`, `gemorna`, `lineardesign`, `synonymous`, `ribonn`, `vienna`, `smart5utr`, `parade`, `optimus5`, `saluki`, `rnadegformer`, `utr_lm`, `codongpt`, `utrgan`, `mrnabert`, `joint`, `mrnautilus`, `degscore`, `promorna`, `mrna_gpt`, `references`, `esmfold2-preview`.

### Enzyme Agent

Installer: [`third_party/enzyme/install.sh`](third_party/enzyme/install.sh). [File layouts and usage](third_party/enzyme/README.md) · [Machine-readable dependencies](third_party/enzyme/dependencies.json).

Declared components: `peptide-runtime`, `runtime`, `core`, `foundry`, `qwen`, `esm3`, `esmfold2`, `zymctrl`, `ligandmpnn`, `proteinmpnn`, `rfdiffusion3`, `thermompnn`, `dnase-reference`, `rfdiffusion2`, `placer`, `catpred`, `catapro`, `ezspecificity`, `netsolp`, `netmhciipan`, `progen3`, `genzyme`.

## What the checks establish

`check` verifies declared files and, where supported, installed package metadata
or existing readiness-record fields. It does not import scientific models,
execute GPU inference, grant access to licensed assets, or certify scientific
validity. Missing resources return a nonzero status. Present resources remain
unvalidated unless the runtime's own requirements are separately met.

The installers and plans were inspected offline. This publication work did not
perform fresh full environment installations or model downloads. Some backends
need upstream setup or manually supplied data beyond the automated stages; those
limits are part of the manifests rather than hidden behind a successful exit.
