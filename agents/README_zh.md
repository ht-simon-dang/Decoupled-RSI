# Agent

[English](README.md) | [简体中文](README_zh.md)

所有 Agent 入口集中在这里。每个 Task Agent 在自己的项目目录内保留运行配置、依赖声明、安装脚本和源码包。

| Agent | 职责 | 入口 |
| --- | --- | --- |
| [Self-Evolve Agent](self-evolve-agent/README_zh.md) | 独立的改进生命周期 | `self-evolve-agent/run.sh` |
| [Drug Agent](drug-agent/README_zh.md) | 主要的计算药物发现 Task Agent | `drugagent` |
| [Peptide Agent](peptide-agent/README_zh.md) | 实验性肽 Task Agent | `peptide-agent` |
| [mRNA Agent](mRNA-agent/README_zh.md) | 实验性 mRNA Task Agent | `mrna-agent` |
| [Enzyme Agent](Enzyme-Agent/README_zh.md) | 实验性酶 Task Agent | `enzyme-agent` |

Self-Evolve 目录提供独立的调用与文档入口。当前 Python 实现仍在 Drug Agent 包内共享，以避免复制框架代码。其他三个 Task Agent 尚不完善，也尚未接入统一的进化接口。

第三方安装器及预期资源结构的索引见 [THIRD_PARTY_zh.md](../THIRD_PARTY_zh.md)。

