# Drug Agent

[English](README.md) | [简体中文](README_zh.md)

这是 Decoupled RSI Agent 的主要 Task Agent 项目，同时包含独立 Self-Evolve 生命周期的当前 Python 实现，位于 [`src/drug_agent/selfevolve/`](src/drug_agent/selfevolve/)。

## 安装与运行

请在路径不含冒号的 checkout 中，进入本项目目录执行：

```bash
./install.sh --help
./install.sh core
conda activate drug-agent
drugagent doctor
make demo
```

包名仍为 `drug-agent`，导入名为 `drug_agent`，CLI 为 `drugagent`。离线演示使用[合成 fixtures](fixtures/data/DATA_CARD_zh.md)。可选科学后端需要自己的资源，新安装时可能报告不可用。

## 项目文件

```text
drug-agent/
  src/drug_agent/       任务执行、工具、评估和自进化
  configs/             运行配置模板
  system_prompts/      随包分发的提示模板
  schemas/             公开接口契约
  fixtures/            小型离线演示输入
  scripts/external/    本项目编写的驱动，用于调用单独安装的后端
  scripts/research/    研究活动调度和靶点准备入口
  environment.yml      主要运行依赖
  environment-pfm.yml  可选 PFM 额外依赖
  envs/                可选独立环境配方
  install.sh           显式选择的安装阶段
```

组件安装器、源码引用、所需文件树和就绪检查见 [third_party/drug](../../third_party/drug/README_zh.md)。仓库不附带外部源码 checkout、权重或原研究活动数据。

本项目在仓库级 Git checkout 中的路径是 `agents/drug-agent`。候选工作区区分项目根目录和 Git 根目录。完整进化循环保留回归关卡；私有开发测试套件不随仓库发布，未提供兼容套件时该关卡无法通过。详见[进化指南](../../docs/self-evolution_zh.md)。

[仓库概览](../../README_zh.md) · [许可证](LICENSE)

