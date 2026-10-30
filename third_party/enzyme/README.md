# Enzyme dependencies

[English](README.md) | [简体中文](README_zh.md)

This inventory follows the actual Enzyme model registry and downloaders. It
includes runtime environments, source checkouts, model weights, reference data,
and the routes that still require manual integration. Downloaded artifacts are
not redistributed here. See [dependencies.json](dependencies.json) for exact
revisions, file contracts, commands, sources, and remaining requirements.

```bash
bash third_party/enzyme/install.sh --help
bash third_party/enzyme/install.sh plan esm3
bash third_party/enzyme/install.sh check rfdiffusion3
# Explicit installation operations; these can use the network:
bash third_party/enzyme/install.sh install peptide-runtime
bash third_party/enzyme/install.sh install runtime
bash third_party/enzyme/install.sh install qwen
bash third_party/enzyme/install.sh install ligandmpnn
bash third_party/enzyme/install.sh install esm3
bash third_party/enzyme/install.sh install esmfold2
bash third_party/enzyme/install.sh install zymctrl
bash third_party/enzyme/install.sh install dnase-reference
```

`core` installs the runtime, all core models, and the DNase reference as one
explicit stage. `ligandmpnn` and `proteinmpnn` share an upstream downloader and
retrieve both checkpoints. Individual model commands use the existing downloader;
all installation paths are passed as argv elements rather than shell-expanded
source strings.

For extended models:

```bash
bash third_party/enzyme/install.sh plan foundry
# Set this to an existing Python 3.12 interpreter with compatible CUDA Torch:
export ENZYME_FOUNDRY_BASE=/absolute/path/to/python
bash third_party/enzyme/install.sh install foundry
```

The Foundry stage downloads RFdiffusion3 and ThermoMPNN sources/weights and builds
the separate runtime. `install rfdiffusion3` or `install thermompnn` downloads the
selected source/weight bundle only. Neither command creates local readiness
evidence. Their check remains missing until the required runtime and matching
`runtime-validation.json` evidence exist; the internal validation scripts and
historical records are not included in this public snapshot.

Default layout (`ENZYME_DATA_DIR`, default `~/.cache/decoupled-rsi/enzyme`):

```text
<ENZYME_DATA_DIR>/
  env/bin/python
  foundry-env/bin/{python,rfd3}
  models/
    qwen/<revision>/                    # fallback for the shared Qwen model
    esm3/<revision>/{config.json,manifest.json,data/weights/*.pth}
    esmfold2/<revision>/{config.json,manifest.json,*.safetensors}
    zymctrl/<revision>/{config.json,manifest.json,pytorch_model.bin}
    ligandmpnn/
      source/{run.py,model_utils.py,data_utils.py,openfold/}
      {ligandmpnn_v_32_010_25.pt,proteinmpnn_v_48_020.pt,manifest.json}
    rfdiffusion3/{source/,rfd3_latest.ckpt,manifest.json,runtime-validation.json}
    thermompnn/
      source/{transfer_model.py,models/thermoMPNN_default.pt,
              vanilla_model_weights/v_48_020.pt}
      {manifest.json,runtime-validation.json}
  references/cf_dnase_reference.json
agents/Enzyme-Agent/examples/dnase_{reference,design}.json
                                         # generated locally and Git-ignored
```

`PEPTIDE_PYTHON` chooses the base interpreter; `ENZYME_PYTHON` chooses the Enzyme
interpreter. `ENZYME_QWEN_DIR` overrides the shared Qwen snapshot. Without that
override, the adapter honors `PEPTIDE_MODEL_DIR` or the sibling peptide model
location, with a fallback to its own data directory.

RFdiffusion2, PLACER, CatPred, CataPro, EZSpecificity, NetSolP, NetMHCIIpan,
ProGen3 and GENzyme are declared manual/incomplete routes. Their `plan` includes
upstream acquisition links; `check` and `install` return missing. License-gated
access, suitable input domains, and an integrated validated adapter must be
provided before activation. No stand-in files or ready flags are generated.
Downloaded components retain their own licenses, including any separate model
and noncommercial terms.

`plan` and `check` use only standard-library file inspection and do not invoke
models, downloads, package installers, or services. Results are `missing` (exit 2)
or `present_unvalidated` (exit 0); file presence does not certify execution or
scientific readiness. Installation can finish asset acquisition but return 2
while readiness evidence is missing. `install all` is intentionally rejected.
