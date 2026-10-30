# 第三方安装与文件依赖

[English](THIRD_PARTY.md) | [简体中文](THIRD_PARTY_zh.md)

每类 Agent 均提供可执行安装器、机器可读的依赖清单和文件结构指南。安装器对接实际适配器要求及现有资源获取脚本。帮助、计划和文件存在检查不会安装环境或下载模型。

仓库不附带第三方源码、checkpoint、数据集或已安装环境。本项目自己的适配器与安装脚本保留在仓库中。外部组件适用各自的许可证与资源访问要求。

## 安装入口

| Agent 类别 | 安装器 | 查看计划 | 检查预期文件 | 显式安装示例 |
| --- | --- | --- | --- | --- |
| Drug | [`install.sh`](third_party/drug/install.sh) | `./third_party/drug/install.sh --plan pfm` | `./third_party/drug/install.sh --check pfm` | `./third_party/drug/install.sh pfm source` |
| Peptide | [`install.sh`](third_party/peptide/install.sh) | `./third_party/peptide/install.sh plan qwen` | `./third_party/peptide/install.sh check qwen` | `./third_party/peptide/install.sh install qwen` |
| mRNA | [`install.sh`](third_party/mrna/install.sh) | `./third_party/mrna/install.sh plan gemorna` | `./third_party/mrna/install.sh check gemorna` | `./third_party/mrna/install.sh install core` |
| Enzyme | [`install.sh`](third_party/enzyme/install.sh) | `./third_party/enzyme/install.sh plan ligandmpnn` | `./third_party/enzyme/install.sh check ligandmpnn` | `./third_party/enzyme/install.sh install ligandmpnn` |

请先运行安装器的 `--help`。根分发脚本也可转发这些命令，例如 `./install.sh dependency drug --plan pfm`。Self-Evolve 入口共用 [Drug Agent 运行环境](agents/drug-agent/environment.yml)，并使用单独配置的推理后端；详见其[依赖指南](agents/self-evolve-agent/README_zh.md)。

Drug 集成提供独立的 `source`、`environment` 和 `assets` 阶段。需要手工获取资源的阶段会拒绝执行并给出说明。例如，PFM 的源码和环境支持安装，但 checkpoint 与准备好的口袋须按清单自行提供。

其他生物领域 Agent 的安装器使用单个组件名或显式分组阶段。部分 mRNA 后端共用 `core` 或 `optional` 安装阶段；运行前，`plan` 会展示整个分组的范围。占位研究后端和缺失的本地验证会被记录为尚未满足的要求，不会被当作安装成功。各组件清单列明先决条件。

## 仓库与运行资源结构

仓库中提交的依赖管理文件如下：

```text
third_party/
  drug/
    install.sh
    installer.py
    dependencies.json
    README.md
  peptide/                 # mrna/ 和 enzyme/ 采用相同文件组织
    install.sh
    manage.py
    dependencies.json
    README.md
agents/
  drug-agent/              # 本项目的适配器、脚本和环境声明
  peptide-agent/
  mRNA-agent/
  Enzyme-Agent/
  self-evolve-agent/       # 使用 Drug Agent 运行环境的调用入口
```

以下文件应在执行所选安装/资源获取步骤后存在，**不随本源码副本分发**。完整组件树、版本、替代路径、环境覆盖变量及就绪记录见链接的各类依赖清单。

### PFM 示例

```text
agents/drug-agent/.external/PFM/
  scripts/sampling.py
  egnn/DotGat.py
  utils/reconstruct.py
  model_params/
    pfm.pth
    training.yml
    x_predictor.pth         # 配置的采样器使用时需要
    h_predictor.pth         # 配置的采样器使用时需要
<prepared-pocket>.pdb       # 通过任务或 DRUGAGENT_PFM_POCKET 提供
```

`pfm source` 获取固定版本的上游 checkout。主要运行环境已存在时，`pfm environment` 安装额外依赖与编译扩展。清单将 `DRUGAGENT_PFM_ROOT`、checkpoint、训练配置、预测器、解释器、设备和口袋覆盖变量映射到对应输入。标准库 [PFM 回放](examples/pfm_replay/README_zh.md) 不需要这些外部资源。

### Peptide/Qwen 共享示例

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

清单在受支持的位置使用 `PEPTIDE_ENV_DIR`、`PEPTIDE_PYTHON` 和 `PEPTIDE_MODEL_DIR`。mRNA 和 Enzyme 可以复用 Qwen 快照，也可以显式覆盖模型路径。安装完成不表示所有共享后端均已通过本地推理验证。

### mRNA 示例

```text
${MRNA_DATA_DIR}/            # 默认：~/.cache/decoupled-rsi/mrna
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

其他预测器各自的固定代码、模型文件与本地就绪要求见 [mRNA 清单](third_party/mrna/dependencies.json)。公开安装器不会伪造已排除的历史验证记录。

### Enzyme 示例

```text
${ENZYME_DATA_DIR}/          # 默认：~/.cache/decoupled-rsi/enzyme
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
      runtime-validation.json   # 本地验证证据；不提供，也不伪造
```

MPNN 获取脚本下载同时覆盖 LigandMPNN 和 ProteinMPNN 的共享资源包；计划会声明这一范围。RFdiffusion3 使用独立的 Foundry 环境。准确的源码、checkpoint、环境和验证要求见 [Enzyme 清单](third_party/enzyme/dependencies.json)。

## 完整组件索引

### Drug Agent

安装器：[`third_party/drug/install.sh`](third_party/drug/install.sh)。[文件结构与用法](third_party/drug/README_zh.md) · [机器可读依赖](third_party/drug/dependencies.json)。

声明的组件：`admetai`、`aizynthfinder`、`boltz`、`chemprop`、`depmap`、`diffdock`、`drugflow`、`e3moldiffusion`、`experimental`、`flowdock`、`flowmol`、`flowr`、`flowr_root`、`fpocket`、`gnina`、`molcraft`、`moljo`、`molpilot`、`omtra`、`openfe`、`opentargets`、`p2rank`、`pfm`、`pocket2mol`、`pocketflow`、`pocketxmol`、`posebench`、`posebusters`、`prexsyn`、`protenix`、`reasyn`、`reinvent4`、`rxnflow`、`safc`、`shepherd`、`shepherd-score`、`synflownet`、`synformer`、`targetdiff`、`uni-mol`、`vina`、`vnegnn`。

### Peptide Agent

安装器：[`third_party/peptide/install.sh`](third_party/peptide/install.sh)。[文件结构与用法](third_party/peptide/README_zh.md) · [机器可读依赖](third_party/peptide/dependencies.json)。

声明的组件：`runtime`、`qwen`、`pepmlm`、`boltzgen`、`toxinpred3`、`prodigy`、`sequence-chemistry`、`calibrated-affinity`、`research-backends`。

### mRNA Agent

安装器：[`third_party/mrna/install.sh`](third_party/mrna/install.sh)。[文件结构与用法](third_party/mrna/README_zh.md) · [机器可读依赖](third_party/mrna/dependencies.json)。

声明的组件：`peptide-runtime`、`runtime`、`core`、`optional`、`qwen`、`gemorna`、`lineardesign`、`synonymous`、`ribonn`、`vienna`、`smart5utr`、`parade`、`optimus5`、`saluki`、`rnadegformer`、`utr_lm`、`codongpt`、`utrgan`、`mrnabert`、`joint`、`mrnautilus`、`degscore`、`promorna`、`mrna_gpt`、`references`、`esmfold2-preview`。

### Enzyme Agent

安装器：[`third_party/enzyme/install.sh`](third_party/enzyme/install.sh)。[文件结构与用法](third_party/enzyme/README_zh.md) · [机器可读依赖](third_party/enzyme/dependencies.json)。

声明的组件：`peptide-runtime`、`runtime`、`core`、`foundry`、`qwen`、`esm3`、`esmfold2`、`zymctrl`、`ligandmpnn`、`proteinmpnn`、`rfdiffusion3`、`thermompnn`、`dnase-reference`、`rfdiffusion2`、`placer`、`catpred`、`catapro`、`ezspecificity`、`netsolp`、`netmhciipan`、`progen3`、`genzyme`。

## 检查能说明什么

`check` 验证声明的文件，并在支持时检查已安装包元数据或已有就绪记录字段。它不会导入科学模型、执行 GPU 推理、授予受许可资源的访问权，或认证科学有效性。资源缺失时返回非零状态。即使资源存在，在单独满足运行时要求之前仍属未验证状态。

安装器和计划已进行离线检查。本次发布准备没有重新进行完整环境安装或模型下载。部分后端除自动阶段外还需要上游配置或手工提供数据；清单会明确这些限制，不会以成功退出掩盖它们。

