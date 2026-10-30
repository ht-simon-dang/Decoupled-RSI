# 依赖与安装

[English](dependencies.md) | [简体中文](dependencies_zh.md)

公开副本包含源码、依赖声明和安装入口，不包含已安装环境、第三方源码树、模型 checkpoint、原始研究数据集及其本地验证记录。

安装和候选执行请使用不含冒号的独立 checkout 路径，例如 `decoupled-rsi-agent`。以项目标题命名的暂存目录用于发布审阅；PFM 聚合回放可在该目录运行，无需安装科学计算依赖。

## 主要运行环境

主要运行环境使用 conda 和 Python 3.12。在仓库根目录执行：

```bash
./install.sh --help
./install.sh core
conda activate drug-agent
make doctor
```

`core` 进入 `agents/drug-agent`，根据其中的 `environment.yml` 创建或更新环境，再通过 `pip install --no-deps -e .` 安装本地包。该环境不包含开发测试依赖。安装到已有环境前，请先检查依赖声明。

创建新环境的等价显式步骤为：

```bash
cd agents/drug-agent
conda env create -f environment.yml
conda activate drug-agent
python -m pip install --no-deps -e .
```

## 可选的 PFM 执行环境

[已记录的 PFM 示例](../examples/pfm_replay/README_zh.md) 只需标准 Python。实际运行生成器还需要额外资源：

```bash
./install.sh pfm
./install.sh pfm-source
```

第一条命令安装额外依赖，并针对已安装的 PyTorch 构建两个图计算扩展，可能需要编译器和兼容的加速器工具链。第二条命令获取集成时记录的 PFM 上游源码版本：
`33be6c1f3dc9f42deb7a27be974ab9f5fc241497`。

请按上游说明单独获取 PFM 权重及预测器资源，并提供准备好的口袋。`agents/drug-agent/.env.example` 列出了受支持的 `DRUGAGENT_PFM_*` 设置。源码安装器不提供原靶点输入，也不代表科学计算功能已经就绪。

## 其他可选药物工具

显式选择一个源码或环境：

```bash
./install.sh list
./install.sh external REINVENT4
./install.sh side-env vina
```

`external NAME` 克隆 [`CHECKOUTS.tsv`](../agents/drug-agent/scripts/external/CHECKOUTS.tsv) 中的一个条目，并检出记录的 revision。自动 Git LFS 下载已禁用，但选定上游仓库中正常提交的文件仍会随 checkout 获取。

`side-env NAME` 支持 `aizynth`、`boltz`、`synflownet`、`unimol` 和 `vina`，对应声明位于 [`envs/`](../agents/drug-agent/envs/)。这些环境不能互换，其中 SynFlowNet 配方专用于 CUDA/Linux x86-64。权重、数据、许可证和后端特定配置仍需另行处理。

`agents/drug-agent/scripts/external/` 中的 Python 文件是本项目自己的适配驱动，不是随仓库打包的第三方插件。只有上游安装已配置且可用时，这些驱动才会调用它。源码来源参见 [THIRD_PARTY_zh.md](../THIRD_PARTY_zh.md)。

## 组件安装器与文件结构

[`third_party/`](../third_party/) 为每类 Agent 提供一个安装器和机器可读的依赖清单。清单列出源码版本、环境依赖、预期文件、配置变量和不可用的安装阶段。相邻 README 提供对应目录结构与可执行命令。

```bash
./install.sh dependency drug --help
./install.sh dependency peptide --help
./install.sh dependency mrna --help
./install.sh dependency enzyme --help
```

计划和文件检查不会下载或导入模型。文件存在不代表后端已通过科学验证。需要单独许可资源或本地验证的组件会报告这些要求，而不会直接标为就绪。

## 实验性 Task Agent

请使用各项目自己的安装指南和环境，不要将所有依赖都装入主要 conda 环境：

- [Peptide Agent](../agents/peptide-agent/README_zh.md)
- [mRNA Agent](../agents/mRNA-agent/README_zh.md)
- [Enzyme Agent](../agents/Enzyme-Agent/README_zh.md)

显式执行这些安装脚本时，可能会下载大量外部资源。部分科学后端的本地就绪证据不随仓库发布。即使依赖和权重已经安装，后端也可能因尚未验证而保持不可用。

## 本副本已完成的检查

发布检查覆盖文件范围、语法与声明、安装帮助，以及标准库 PFM 回放。本次没有重新安装科学环境，也没有执行新的模型/GPU 运行。完整平台进化循环仍需要 [self-evolution_zh.md](self-evolution_zh.md) 所述的回归关卡。

