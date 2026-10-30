# Drug Agent dependency installer

[English](README.md) | [简体中文](README_zh.md)

This directory provides executable entry points and a machine-readable dependency
layout for the Drug Task Agent. It contains installation orchestration only;
third-party source trees, environments, weights and study datasets are not bundled.

## Inspect before installing

The entry point works from any working directory. From the repository root:

```bash
./third_party/drug/install.sh --help
./third_party/drug/install.sh --list
./third_party/drug/install.sh pfm --plan
./third_party/drug/install.sh pocketxmol assets --plan
./third_party/drug/install.sh pfm --check
./third_party/drug/install.sh all source --check
```

`--plan` reads the manifest and expands paths without running a subprocess.
`--check` performs offline file, Git-revision, executable and module-location
checks. It does not install anything or import model packages. A check reports
`MISSING` and exits 1 when prerequisites are absent. `PRESENT` means the declared
presence checks passed; it does not establish import/ABI compatibility, GPU
support, successful inference or scientific validity.

Actual installation requires **one component and one explicit stage**:

```bash
# Core runtime is installed separately.
./agents/drug-agent/install.sh core

./third_party/drug/install.sh pfm source
./third_party/drug/install.sh pfm environment
./third_party/drug/install.sh pfm assets --check

./third_party/drug/install.sh pocketxmol source
./third_party/drug/install.sh pocketxmol environment
./third_party/drug/install.sh pocketxmol assets
```

Use a checkout path without `:` for installation and execution. The publication
review folder may contain that character; planning and checking still work there.
There is no bulk install command. A `manual` or `not_required` installation stage
refuses with exit 2 and an explanation; it does not print a fake success. Commands
that actually run propagate errors, and source/asset stages check their declared
outputs after completion.

## Files and expected layout

```text
third_party/drug/
  install.sh                 executable shell entry point
  installer.py               standard-library orchestration and offline checks
  dependencies.json          component specifications and stage recipes
  README.md
agents/drug-agent/
  install.sh                 core, PFM and pinned-checkout installation primitives
  environment.yml            main runtime dependency declaration
  environment-pfm.yml        PFM overlay; compiled extensions are separate
  envs/*.yml                 AiZynth, Boltz, SynFlowNet, Uni-Mol and Vina recipes
  scripts/external/          first-party drivers and explicit asset fetchers
  .external/                 downloaded sources/assets, created only on request
    PFM/
      scripts/sampling.py
      model_params/pfm.pth
      model_params/training.yml
    PocketXMol/
      data/trained_models/pxm/checkpoints/pocketxmol.ckpt
    aizynthfinder/            source reference checkout
    aizynthfinder-data/       runtime models, templates, config and stock
```

The manifest specifies each integration's source URL/revision, runtime module
names, candidate environments, driver, source sentinels, checkpoint paths,
additional scientific inputs, configuration variables and installation support.
Its `integration.source_file` identifies the adapter from which requirements were
extracted, with a SHA-256 of that source snapshot. A few platform-dependent
expressions are retained as unresolved expressions rather than evaluated or
invented; they concern platform blockers, not checkpoint names.

Paths in the manifest use `{drug}`, `{external}`, `{source}` and `{runtime_root}`
tokens. They are expanded by `--plan`. Source and runtime data roots may differ:
AiZynthFinder is one example. Boltz uses `BOLTZ_CACHE` (default `~/.boltz`), while
Protenix uses its configured root or `PROTENIX_ROOT_DIR` (default `~/.protenix`).
Neither cache is part of the Git repository.

## Configuration

- `CONDA_EXE`: conda executable; `DRUGAGENT_CONDA_ENV`: main environment name.
- `DRUGAGENT_EXTERNAL_DIR`: checkout/download directory, defaulting to
  `agents/drug-agent/.external`. Relative values are interpreted under the Drug
  Agent directory.
- `PYTHON`: Python used to run this small installer, default `python3`.
- Per-tool `DRUGAGENT_<PREFIX>_ROOT` and `_PYTHON`: runtime root/interpreter
  overrides listed explicitly in the manifest. Set the per-tool runtime root
  when relocating sources; the global download-root variable alone is not a
  universal runtime override.
  After installing an isolated upstream environment, set the listed `_PYTHON`
  variable to that environment's interpreter. The installer prints this export
  when it can resolve the created environment. Some isolated environment names
  differ from the adapter's legacy auto-discovery names; installing the packages
  alone does not configure the Task Agent to use them.
- PFM also accepts explicit checkpoint/training-config paths and a prepared
  pocket via the variables listed in its manifest entry. These paths are inputs
  supplied by the operator, not downloads invented by this installer.

The check command respects the documented PFM artifact overrides. A shared
source/asset recipe refuses a conflicting per-tool root override instead of
writing to one location and claiming another was installed. No `.env` file is
read by this installer.

## Stages and limits

`source` fetches the recorded revision with Git LFS smudging disabled, or explains
why a package/data-backed component has no Git stage. An LFS pointer does not
satisfy an asset check. Existing checkout revisions are not silently replaced.

`environment` uses a retained environment file, an explicitly named upstream
recipe after source checkout, or the existing Drug Agent installation primitive.
Upstream recipes may contain platform-specific pins and build steps. Their
compatibility has not been revalidated by this publication preparation. Manual
entries identify the missing recipe or required operator choice.

`assets` invokes an existing explicit downloader only where the repository has
one or a documented upstream helper. Otherwise it reports the required layout
and what must be supplied. PFM's checkpoint inputs remain manual; SAFC requires
an independently supplied and verified transfer bundle; Chemprop requires a
trained checkpoint for the chosen endpoint; SynFlowNet requires reward-specific
training and building-block data. Installing a Python package cannot substitute
for these artifacts. This installer does not start training campaigns.

The selected PFM paths and all 26 entries from `CHECKOUTS.tsv` are represented.
Additional adapters, including ADMET-AI, Chemprop, Protenix, SAFC, Vina and pocket
or target-data tools, have their own entries. Source IDs are case-insensitive.

| Component ID | Source reference | Source | Environment | Assets |
| --- | --- | --- | --- | --- |
| `admetai` | package / supplied data | not_required | supported | manual |
| `aizynthfinder` | aizynthfinder | supported | supported | supported |
| `boltz` | boltz | supported | supported | supported |
| `chemprop` | package / supplied data | not_required | supported | manual |
| `depmap` | package / supplied data | manual | manual | manual |
| `diffdock` | DiffDock | supported | supported | manual |
| `drugflow` | DrugFlow | supported | supported | manual |
| `e3moldiffusion` | e3moldiffusion | supported | supported | manual |
| `experimental` | package / supplied data | manual | manual | manual |
| `flowdock` | FlowDock | supported | supported | manual |
| `flowmol` | FlowMol | supported | supported | manual |
| `flowr` | flowr_root | supported | supported | manual |
| `flowr_root` | flowr_root | supported | supported | manual |
| `fpocket` | package / supplied data | not_required | supported | not_required |
| `gnina` | gnina | supported | manual | supported |
| `molcraft` | MolCRAFT | supported | manual | manual |
| `moljo` | MolCRAFT | supported | manual | manual |
| `molpilot` | MolCRAFT | supported | manual | manual |
| `omtra` | OMTRA | supported | manual | manual |
| `openfe` | openfe | supported | supported | not_required |
| `opentargets` | package / supplied data | manual | manual | manual |
| `p2rank` | package / supplied data | manual | manual | supported |
| `pfm` | PFM | supported | supported | manual |
| `pocket2mol` | Pocket2Mol | supported | manual | supported |
| `pocketflow` | PocketFlow | supported | supported | manual |
| `pocketxmol` | PocketXMol | supported | supported | supported |
| `posebench` | PoseBench | supported | supported | manual |
| `posebusters` | package / supplied data | not_required | supported | not_required |
| `prexsyn` | prexsyn | supported | manual | manual |
| `protenix` | package / supplied data | not_required | manual | supported |
| `reasyn` | reasyn | supported | manual | manual |
| `reinvent4` | REINVENT4 | supported | manual | supported |
| `rxnflow` | RxnFlow | supported | manual | manual |
| `safc` | package / supplied data | manual | supported | manual |
| `shepherd` | shepherd | supported | supported | manual |
| `shepherd-score` | shepherd-score | supported | supported | not_required |
| `synflownet` | synflownet | supported | supported | manual |
| `synformer` | synformer | supported | manual | manual |
| `targetdiff` | targetdiff | supported | supported | supported |
| `uni-mol` | Uni-Mol | supported | supported | supported |
| `vina` | package / supplied data | not_required | supported | not_required |
| `vnegnn` | package / supplied data | supported | manual | manual |

`not_required` is scoped to that stage: a package without a Git checkout may
still require an environment and model files. An auxiliary source such as
`shepherd-score` has no standalone model-asset contract; inspect its consuming
adapter as well. Request-specific pockets, prepared receptors, MSA/template
inputs and target identifiers are documented as additional inputs and are not
claimed to be verified by the generic presence check.

External software, datasets and weights retain their upstream terms. The
recorded URLs and revisions describe this integration snapshot; download links
can move, and failed downloads remain failures requiring review.
