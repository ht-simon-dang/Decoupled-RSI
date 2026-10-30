# Peptide dependencies

[English](README.md) | [简体中文](README_zh.md)

This directory contains dependency declarations and installation entrypoints, not
third-party source, model weights, reference complexes, or previous run records.
The executable adapter contracts are in `agents/peptide-agent/src/peptide_agent`.

From the repository root:

```bash
bash third_party/peptide/install.sh --help
bash third_party/peptide/install.sh plan
bash third_party/peptide/install.sh check qwen
# Explicit installation operations; these can use the network:
bash third_party/peptide/install.sh install runtime
bash third_party/peptide/install.sh install qwen
bash third_party/peptide/install.sh install pepmlm
bash third_party/peptide/install.sh install boltzgen
bash third_party/peptide/install.sh install toxinpred3
```

`runtime` delegates to the project's installer. The model components delegate to
its real download/export commands, keeping the pinned source revisions and
checksum checks. `prodigy` and `sequence-chemistry` map to the runtime dependency
stage. `calibrated-affinity` requires user measurements and local calibration;
`research-backends` has no integrated automatic installation and returns missing.

Default layout (override `PEPTIDE_MODEL_DIR` with an absolute directory):

```text
agents/peptide-agent/
  .venv/bin/python
  .peptide-agent/models/
    qwen/<revision>/{config.json, tokenizer files, weight shards}
    pepmlm/<revision>/{config.json, tokenizer files, weights}
    boltzgen/
      models--boltzgen--boltzgen-1/snapshots/<revision>/*.ckpt
      datasets--boltzgen--inference-data/snapshots/<revision>/mols.zip
    toxinpred3/{model.pkl, trees.npz, manifest.json}
    calibrations/*.json                  # supplied/fitted by the operator
```

`PEPTIDE_PYTHON` selects an existing runtime for component downloads; the runtime
installer also accepts `PEPTIDE_BOOTSTRAP_PYTHON` and `PEPTIDE_ENV_DIR`. The sibling
mRNA and Enzyme agents now also honor `PEPTIDE_MODEL_DIR` for shared Qwen assets.
The exact revisions, URLs, path probes, install commands and limitations are in
[dependencies.json](dependencies.json). Downloaded code, models and data retain
their separate upstream terms; local project licensing does not grant those rights.

`plan` and `check` use only the standard library and local filesystem. They never
import scientific packages, download, install, or run inference. `check` returns
2 for missing assets or unresolved manual requirements, 0 for
`present_unvalidated`; it never reports a backend as scientifically ready.
No-argument `install` and `install all` are rejected. Select an explicit component.
