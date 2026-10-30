# mRNA Task Agent

[English](README.md) | [简体中文](README_zh.md)

**实验性实现，尚不完善，需要继续进化和验证。**

这是用于 mRNA 设计、评估和实验反馈的 Task Agent。类型化规划、分区域生成器、科学预测器、独立评估和版本化运行记录分别作为独立组件。其本地策略审查尚未与共享 Self-Evolve Agent 完整整合。

## 安装

使用 Linux。先安装同级 `peptide-agent` 的 Python 3.11 运行环境及 Qwen 模型，然后在本目录执行：

```bash
export PEPTIDE_PYTHON="$(cd ../peptide-agent && pwd)/.venv/bin/python"
export MRNA_DATA_DIR="$HOME/.cache/decoupled-rsi/mrna"
bash install.sh
"$MRNA_DATA_DIR/venv/bin/python" -m mrna_agent.cli doctor
```

安装器使用 Peptide 解释器创建 venv，安装声明的运行包，下载固定版本的上游代码和模型资源，构建 LinearDesign，并创建独立兼容环境。它需要网络、用于 Python 3.10 GEMORNA 运行环境的 conda、C/C++ 构建工具链及足够存储。安装受应用的 50,000,000,000 字节存储上限约束。下载前请检查上游许可证；不同依赖的代码和模型条款不同。

公开副本包含依赖声明和下载适配器，不包含上游源码、权重、数据集或以往运行记录。基础依赖见 `requirements.txt`，旧版依赖见 `requirements-legacy.txt`，源码版本见 `scripts/bootstrap.py` 和 `scripts/optional-pins.json`。

## 运行

```bash
"$MRNA_DATA_DIR/venv/bin/python" -m mrna_agent.cli run examples/hbb.json
"$MRNA_DATA_DIR/venv/bin/python" -m mrna_agent.cli serve --port 2005
```

随仓库提供的 JSON 是输入模板，不是保存的实验结果。下载完成后，工具仍可能在满足就绪检查前保持不可用。本次发布不包含内部验证脚本和记录；安装器不会生成虚假的替代就绪记录。可选模型激活与全新安装验证仍不完善。

## 配置与代码

- `MRNA_DATA_DIR`：环境、资源和新运行记录，默认为 `~/.cache/decoupled-rsi/mrna`。
- `PEPTIDE_PYTHON`：Peptide 基础解释器；安装器默认使用同级项目的 `.venv/bin/python`。
- `MRNA_QWEN_DIR`：显式指定 Qwen 快照；否则使用同级 Peptide 安装或本地模型目录。
- `MRNA_ESMFOLD_DIR`：可选的共享 ESMFold2 快照，默认跟随 `ENZYME_DATA_DIR`（或 `~/.cache/decoupled-rsi/enzyme`）。
- `MRNA_STRUCTURE_GPU_LOCK`：覆盖与 Enzyme 共享的 GPU 锁路径。
- `src/mrna_agent/workflow.py`：任务生命周期；`gateway.py` 和 worker 模块：科学适配器；`evaluation.py`、`delivery.py` 和 `benchmark.py`：评估与策略审查。

详见[架构](docs/architecture_zh.md)、[可选模型](docs/optional-models_zh.md)、[交付规则](docs/high-potential-policy_zh.md)和[历史预览](docs/history-preview_zh.md)。预测器分数、RNA 折叠指标和蛋白结构置信度不能证明新生物情境下的表达或效力。

## 第三方安装清单

[组件/阶段安装器](../../third_party/mrna/README_zh.md) 提供源码版本、模型/数据结构、离线 plan/check 命令及显式资源获取入口。`plan` 和 `check` 不会下载或运行模型。

