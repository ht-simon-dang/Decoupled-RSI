# Drug Agent 依赖安装器

[English](README.md) | [简体中文](README_zh.md)

本目录为 Drug Task Agent 提供可执行入口与机器可读的依赖结构，仅包含安装编排逻辑，不附带第三方源码树、运行环境、权重或研究数据集。

## 安装前检查

入口可从任意工作目录调用。在仓库根目录执行：

```bash
./third_party/drug/install.sh --help
./third_party/drug/install.sh --list
./third_party/drug/install.sh pfm --plan
./third_party/drug/install.sh pocketxmol assets --plan
./third_party/drug/install.sh pfm --check
./third_party/drug/install.sh all source --check
```

`--plan` 读取清单并展开路径，不启动子进程。`--check` 离线检查文件、Git revision、可执行程序及模块位置，不安装任何内容，也不导入模型包。先决条件缺失时，检查报告 `MISSING` 并以 1 退出。`PRESENT` 仅表示声明的存在检查通过，不代表导入/ABI 兼容、GPU 支持、推理成功或科学有效。

实际安装要求**一个组件和一个明确阶段**：

```bash
# 核心运行环境需单独安装。
./agents/drug-agent/install.sh core

./third_party/drug/install.sh pfm source
./third_party/drug/install.sh pfm environment
./third_party/drug/install.sh pfm assets --check

./third_party/drug/install.sh pocketxmol source
./third_party/drug/install.sh pocketxmol environment
./third_party/drug/install.sh pocketxmol assets
```

安装和执行请使用不含 `:` 的 checkout 路径。发布审阅目录可以包含该字符，计划和检查仍可运行。没有批量安装命令。`manual` 或 `not_required` 安装阶段会以退出码 2 拒绝，并给出解释，不会输出虚假的成功。实际执行的命令会传递错误，源码/资源阶段在完成后检查声明的输出。

## 文件与预期结构

```text
third_party/drug/
  install.sh                 可执行 shell 入口
  installer.py               标准库编排与离线检查
  dependencies.json          组件规格与阶段配方
  README.md
agents/drug-agent/
  install.sh                 核心、PFM 和固定 checkout 的基础安装操作
  environment.yml            主要运行依赖声明
  environment-pfm.yml        PFM 额外依赖；编译扩展另行处理
  envs/*.yml                 AiZynth、Boltz、SynFlowNet、Uni-Mol 和 Vina 配方
  scripts/external/          本项目自己的驱动与显式资源获取脚本
  .external/                 按需创建的源码/资源下载目录
    PFM/
      scripts/sampling.py
      model_params/pfm.pth
      model_params/training.yml
    PocketXMol/
      data/trained_models/pxm/checkpoints/pocketxmol.ckpt
    aizynthfinder/            源码参考 checkout
    aizynthfinder-data/       运行模型、模板、配置和原料库
```

清单指定每项集成的源码 URL/revision、运行时模块名、候选环境、驱动、源码标志文件、checkpoint 路径、附加科学输入、配置变量和安装支持程度。`integration.source_file` 标明提取要求时使用的适配器，并附该源码快照的 SHA-256。少量平台相关表达式以未解析形式保留，不会执行或编造；它们描述平台阻碍条件，不涉及 checkpoint 名称。

清单路径使用 `{drug}`、`{external}`、`{source}` 和 `{runtime_root}` 占位符，由 `--plan` 展开。源码与运行数据根目录可以不同，AiZynthFinder 就是一个例子。Boltz 使用 `BOLTZ_CACHE`（默认 `~/.boltz`），Protenix 使用配置的根目录或 `PROTENIX_ROOT_DIR`（默认 `~/.protenix`）。两者缓存均不属于 Git 仓库。

## 配置

- `CONDA_EXE`：conda 可执行程序；`DRUGAGENT_CONDA_ENV`：主要环境名。
- `DRUGAGENT_EXTERNAL_DIR`：checkout/下载目录，默认 `agents/drug-agent/.external`。相对值按 Drug Agent 目录解释。
- `PYTHON`：运行这个小型安装器的 Python，默认为 `python3`。
- 每个工具的 `DRUGAGENT_<PREFIX>_ROOT` 和 `_PYTHON`：清单中明确列出的运行根目录/解释器覆盖变量。移动源码位置时需设置对应工具的运行根目录；全局下载根目录变量不是通用运行时覆盖变量。安装独立上游环境后，将列出的 `_PYTHON` 变量设为该环境解释器。能够解析已创建环境时，安装器会输出相应 export。部分独立环境名与适配器旧版自动发现的名称不同；仅安装包不会自动配置 Task Agent 使用它们。
- PFM 还可通过清单列出的变量显式指定 checkpoint、训练配置路径及准备好的口袋。这些路径是操作者提供的输入，不是安装器自行编造的下载。

检查命令遵循文档中的 PFM 资源覆盖变量。共享源码/资源配方遇到冲突的单工具根目录覆盖时会拒绝执行，不会写入一个位置却声称另一个位置已安装。本安装器不读取 `.env` 文件。

## 阶段与限制

`source` 在禁用 Git LFS smudging 的情况下获取记录的版本，或解释以包/数据为基础的组件为何没有 Git 阶段。LFS 指针不能通过资源检查。已有 checkout 的 revision 不会被静默替换。

`environment` 使用保留的环境文件、源码 checkout 后显式指定的上游配方，或已有 Drug Agent 基础安装操作。上游配方可能包含平台特定固定依赖和构建步骤；本次发布准备未重新验证其兼容性。手工条目会指出缺少的配方或需要操作者作出的选择。

`assets` 仅在仓库已有显式下载器或文档明确的上游辅助工具时调用它；否则报告所需结构和必须提供的内容。PFM checkpoint 输入仍需手工提供；SAFC 需要独立提供并验证的转移包；Chemprop 需要针对所选端点训练的 checkpoint；SynFlowNet 需要奖励特定的训练和构建块数据。安装 Python 包不能替代这些产物。该安装器不会启动训练活动。

清单包含所选 PFM 路径及 `CHECKOUTS.tsv` 中全部 26 个条目。ADMET-AI、Chemprop、Protenix、SAFC、Vina，以及口袋或靶点数据工具等额外适配器也有独立条目。源码 ID 不区分大小写。

| 组件 ID | 源码参考 | 源码 | 环境 | 资源 |
| --- | --- | --- | --- | --- |
| `admetai` | 软件包 / 提供的数据 | not_required | supported | manual |
| `aizynthfinder` | aizynthfinder | supported | supported | supported |
| `boltz` | boltz | supported | supported | supported |
| `chemprop` | 软件包 / 提供的数据 | not_required | supported | manual |
| `depmap` | 软件包 / 提供的数据 | manual | manual | manual |
| `diffdock` | DiffDock | supported | supported | manual |
| `drugflow` | DrugFlow | supported | supported | manual |
| `e3moldiffusion` | e3moldiffusion | supported | supported | manual |
| `experimental` | 软件包 / 提供的数据 | manual | manual | manual |
| `flowdock` | FlowDock | supported | supported | manual |
| `flowmol` | FlowMol | supported | supported | manual |
| `flowr` | flowr_root | supported | supported | manual |
| `flowr_root` | flowr_root | supported | supported | manual |
| `fpocket` | 软件包 / 提供的数据 | not_required | supported | not_required |
| `gnina` | gnina | supported | manual | supported |
| `molcraft` | MolCRAFT | supported | manual | manual |
| `moljo` | MolCRAFT | supported | manual | manual |
| `molpilot` | MolCRAFT | supported | manual | manual |
| `omtra` | OMTRA | supported | manual | manual |
| `openfe` | openfe | supported | supported | not_required |
| `opentargets` | 软件包 / 提供的数据 | manual | manual | manual |
| `p2rank` | 软件包 / 提供的数据 | manual | manual | supported |
| `pfm` | PFM | supported | supported | manual |
| `pocket2mol` | Pocket2Mol | supported | manual | supported |
| `pocketflow` | PocketFlow | supported | supported | manual |
| `pocketxmol` | PocketXMol | supported | supported | supported |
| `posebench` | PoseBench | supported | supported | manual |
| `posebusters` | 软件包 / 提供的数据 | not_required | supported | not_required |
| `prexsyn` | prexsyn | supported | manual | manual |
| `protenix` | 软件包 / 提供的数据 | not_required | manual | supported |
| `reasyn` | reasyn | supported | manual | manual |
| `reinvent4` | REINVENT4 | supported | manual | supported |
| `rxnflow` | RxnFlow | supported | manual | manual |
| `safc` | 软件包 / 提供的数据 | manual | supported | manual |
| `shepherd` | shepherd | supported | supported | manual |
| `shepherd-score` | shepherd-score | supported | supported | not_required |
| `synflownet` | synflownet | supported | supported | manual |
| `synformer` | synformer | supported | manual | manual |
| `targetdiff` | targetdiff | supported | supported | supported |
| `uni-mol` | Uni-Mol | supported | supported | supported |
| `vina` | 软件包 / 提供的数据 | not_required | supported | not_required |
| `vnegnn` | 软件包 / 提供的数据 | supported | manual | manual |

表中的 `supported` 表示该阶段支持自动安装，`manual` 表示需要手工处理，`not_required` 表示不需要该阶段。`not_required` 仅针对所在阶段：无需 Git checkout 的包仍可能需要环境和模型文件。`shepherd-score` 这类辅助源码没有独立模型资源契约，也应检查调用它的适配器。请求专用口袋、准备好的受体、MSA/模板输入和靶点标识符均作为额外输入记录，通用存在检查不宣称已经验证它们。

外部软件、数据集和权重保留各自上游条款。记录的 URL 和 revision 描述本次集成快照；下载链接可能迁移，下载失败仍需检查处理。

