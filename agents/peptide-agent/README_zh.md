# Peptide Task Agent

[English](README.md) | [简体中文](README_zh.md)

**实验性实现，尚不完善，需要继续进化和验证。**

这是 Decoupled RSI Agent 中用于肽设计与分析的 Task Agent，将类型化规划、科学工具调用、独立检查和持久化证据分开处理。该包拥有自己的工作流和评估机制，尚未成为共享 Self-Evolve Agent 的完整适配器。

## 安装与运行

使用 Linux 和 Python 3.11。在本目录执行：

```bash
bash install.sh
.venv/bin/peptide-agent doctor
.venv/bin/peptide-agent run examples/library.yaml
.venv/bin/peptide-agent serve --port 2004
```

`install.sh` 创建 `.venv` 并安装运行依赖。可设置 `PEPTIDE_BOOTSTRAP_PYTHON` 或 `PEPTIDE_ENV_DIR` 选择其他解释器或虚拟环境。也可使用 conda：先执行 `conda env create -f environment.yml`，再执行 `conda run -n peptide-agent python -m pip install -e . --no-deps`。仅安装包元数据中声明的依赖不足以安装整个科学计算依赖栈。

library 示例分析给定序列。规划与基于学习模型的生成需要另行下载模型，例如：

```bash
.venv/bin/peptide-agent download qwen
.venv/bin/peptide-agent download pepmlm
```

各模型的来源和固定 revision 声明在 `src/peptide_agent/models.py` 和 `downloads.py` 中。其他后端包括 BoltzGen 和 ToxinPred3。仓库不包含模型权重、第三方源码 checkout 或过往运行记录。下载组件适用各自的上游许可证。

## 配置与代码

- `PEPTIDE_MODEL_DIR`：模型目录，默认为 `.peptide-agent/models`。
- `--store PATH`：本地运行存储，默认为 `.peptide-agent`。
- `configs/mcp.json`：MCP 配置模板，需要替换其中的占位路径。
- `src/peptide_agent/workflow.py`：任务执行与检查点保存。
- `gateway.py`、`policy.json` 和 `skills/`：类型化工具边界。
- `analysis.py`、`benchmark.py` 和 `calibration.py`：运行时评估。

如果 mRNA 和 Enzyme 共享本项目的 Qwen 安装，请保持本目录与 `mRNA-agent`、`Enzyme-Agent` 同级。模型位于其他位置时，显式配置 `MRNA_QWEN_DIR` 和 `ENZYME_QWEN_DIR`。

详见[架构](docs/architecture_zh.md)和[模型选择](docs/model-selection_zh.md)。所需资源缺失时，适配器返回不可用。计算输出不能证明生物有效性。公开副本不包含内部测试套件及验证记录；本次整理未验证全新安装的科学环境。

## 第三方安装清单

[组件/阶段安装器](../../third_party/peptide/README_zh.md) 提供源码版本、模型/数据结构、离线 plan/check 命令及显式资源获取入口。`plan` 和 `check` 不会下载或运行模型。

