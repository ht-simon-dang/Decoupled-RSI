# Self-Evolve Agent

[English](README.md) | [简体中文](README_zh.md)

本目录为独立的改进生命周期提供自己的入口，与 Task Agent 并列。当前实现为 [`drug_agent.selfevolve`](../drug-agent/src/drug_agent/selfevolve/)，与 Drug Agent 包共享，没有复制成第二个包。

```bash
./install.sh --help
./install.sh
./run.sh --help
./run.sh doctor
```

`install.sh` 安装 Drug Agent 运行环境，不会安装所有科学后端或下载全部模型权重。`run.sh` 进入 Drug Agent 项目，在配置的 conda 环境中将命令转发给 `drugagent self-evolve`。

循环读取执行经验、提出候选改动、独立评估候选，并记录采纳决策。它可以在任务执行之外单独调用。与实验性肽、mRNA 和酶项目的自动跨 Agent 整合仍是后续工作。

默认完整循环的 gate 0 要求私有回归套件。该套件不公开，此入口也不会绕过关卡。启动循环前请阅读[进化指南](../../docs/self-evolution_zh.md)。

依赖包括 [Drug Agent 运行环境](../drug-agent/environment.yml)、[`self_evolve.yaml`](../drug-agent/configs/self_evolve.yaml) 中配置的提案/推理后端，以及所选评估需要的科学后端。详见[第三方安装说明](../../THIRD_PARTY_zh.md)。

