# Task Agent

[English](task-agents.md) | [简体中文](task-agents_zh.md)

公开源码副本包含四个 Task Agent。Drug Agent 是主要实现；Peptide、mRNA 和 Enzyme Agent 均处于实验阶段，尚不完善，需要继续进化与验证。将它们纳入仓库，并不代表已实现统一的跨领域进化接口，也不构成性能比较证据。

## Drug Agent

项目位于 `agents/drug-agent`，Python 包名为 `drug_agent`，命令为 `drugagent`。在仓库根目录执行：

```bash
./install.sh core
conda activate drug-agent
make doctor
make demo
```

fixtures 是合成演示输入。可选后端需要自己的源码、模型权重、运行环境，有时还需要准备数据。不可用的工具会保持不可用，不会用编造的科学输出代替。请查看本地安装的能力报告。

根 Makefile 会先进入 Drug Agent 项目，再分发命令。`make demo` 运行确定性工作流，`make serve` 启动本地网页控制台；LLM 控制台还要求配置相应模型后端。

运行时会在本地记录新的执行证据。原项目的请求记录和训练产物不在公开范围内。

## 实验性 Agent

各包拥有独立的依赖约束和安装说明：

| Agent | 指南 | 源码 |
| --- | --- | --- |
| Peptide | [README](../agents/peptide-agent/README_zh.md) | [`peptide_agent`](../agents/peptide-agent/src/peptide_agent/) |
| mRNA | [README](../agents/mRNA-agent/README_zh.md) | [`mrna_agent`](../agents/mRNA-agent/src/mrna_agent/) |
| Enzyme | [README](../agents/Enzyme-Agent/README_zh.md) | [`enzyme_agent`](../agents/Enzyme-Agent/src/enzyme_agent/) |

这些项目在 `agents/` 下同级放置。使用默认的共享肽模型位置时，请保持这一结构。通过文档中的环境变量设置可写数据目录。后端就绪可能依赖这里未分发的本地证据与资源；仅安装依赖并不能证明科学功能已验证。

## 将架构用于其他任务

新的 Task Agent 需要执行反馈、明确的可编辑范围、可重复的评估输入和采纳机制。当前代码提供了这些职责的参考实现。抽取可复用、与领域无关的进化包，并连接全部四个应用，仍是后续工作。

