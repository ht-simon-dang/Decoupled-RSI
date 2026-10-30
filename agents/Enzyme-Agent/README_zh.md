# Enzyme Task Agent

[English](README.md) | [简体中文](README_zh.md)

**实验性实现，尚不完善，需要继续进化和验证。**

这是用于治疗性酶研究的 Task Agent，包含类型化反应与产物要求、序列设计、独立结构评估、实验反馈和本地策略审查。该包尚未提供与共享 Self-Evolve Agent 完整连接的适配器。

## 安装

使用 Linux。先安装同级 `peptide-agent` 的 Python 3.11 运行环境及 Qwen 模型，然后在本目录执行：

```bash
export PEPTIDE_PYTHON="$(cd ../peptide-agent && pwd)/.venv/bin/python"
export ENZYME_DATA_DIR="$HOME/.cache/decoupled-rsi/enzyme"
bash install.sh
bash run.sh doctor
bash run.sh serve --port 2006
```

安装器使用 Peptide 解释器创建 venv，确保声明的基础包可用，安装 `requirements.txt` 中的附加包，下载固定版本的核心模型，并从 UniProt/PDB 获取 DNase 参考。它需要网络与足够存储。科学生成和折叠需要兼容的 GPU 运行环境。模型下载和生成记录计入应用的 50,000,000,000 字节存储上限。

仓库不附带第三方源码、权重、参考结构或过往运行。模型 URL 和版本见 `src/enzyme_agent/models.py`、`extended_models.py` 及[模型来源](docs/model-sources_zh.md)。下载组件保留上游许可证。

## 参考示例与可选后端

安装过程中，`scripts/prepare_dnase.py` 根据公开参考资源生成 `examples/dnase_reference.json` 和 `examples/dnase_design.json`：

```bash
bash run.sh run examples/dnase_reference.json
```

所需评估后端可能在完成安装与验证前使工作流保持阻塞。RFdiffusion3 和 ThermoMPNN 使用 `scripts/bootstrap_extended.sh`；调用前请将 `ENZYME_FOUNDRY_BASE` 设置为带兼容 CUDA Torch 的 Python 3.12 解释器。该独立安装会下载额外资源。

就绪关卡保持不变。内部推理验证脚本和历史就绪记录不公开，因此仅下载扩展模型不会将其标为就绪。本地验证与完整的全新安装支持仍在完善中；缺少证据会返回不可用状态。

## 配置与代码

- `ENZYME_DATA_DIR`：环境、模型和新运行记录，默认为 `~/.cache/decoupled-rsi/enzyme`。
- `PEPTIDE_PYTHON`：基础解释器，默认使用同级 Peptide 的 `.venv`。
- `ENZYME_PYTHON`：覆盖 `run.sh` 使用的已安装 Enzyme 解释器。
- `ENZYME_QWEN_DIR`：覆盖共享 Qwen 快照路径。
- `ENZYME_FOUNDRY_BASE`：可选 Foundry 环境所必需的基础解释器。
- `workflow.py`、`gateway.py` 和 worker 模块：执行与工具边界。
- `evaluation.py`、`tool_evaluation.py` 和 `evolution.py`：科学评估与本地选择策略进化。

详见[架构](docs/architecture_zh.md)。结构置信度、静态几何和预测稳定性不能证明催化活性或治疗获益。本次整理没有重新安装公开副本，也没有重新进行科学验证。

## 第三方安装清单

[组件/阶段安装器](../../third_party/enzyme/README_zh.md) 提供源码版本、模型/数据结构、离线 plan/check 命令及显式资源获取入口。`plan` 和 `check` 不会下载或运行模型。

