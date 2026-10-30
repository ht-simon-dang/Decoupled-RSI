# 架构

[English](architecture.md) | [简体中文](architecture_zh.md)

Task Agent 与 Self-Evolve Agent 各有明确职责和独立入口。任务执行产生证据；进化过程使用这些证据、提出候选方案、进行评估，并记录候选是否满足采纳条件。运行时读取显式选定的策略版本。

| 职责 | 实现 |
| --- | --- |
| 任务规划与执行 | [`agents/`](../agents/drug-agent/src/drug_agent/agents/)、[`orchestration/`](../agents/drug-agent/src/drug_agent/orchestration/) |
| 科学工具接口 | [`tools/`](../agents/drug-agent/src/drug_agent/tools/) 及 [`scripts/external/`](../agents/drug-agent/scripts/external/) 中本项目自己的调用驱动 |
| 执行经验 | [`selfevolve/experience/`](../agents/drug-agent/src/drug_agent/selfevolve/experience/) |
| 失败分析与方案提出 | [`analysis/`](../agents/drug-agent/src/drug_agent/selfevolve/analysis/)、[`hypotheses/`](../agents/drug-agent/src/drug_agent/selfevolve/hypotheses/) |
| 进化生命周期与预算 | [`controller/loop.py`](../agents/drug-agent/src/drug_agent/selfevolve/controller/loop.py)、[`config.py`](../agents/drug-agent/src/drug_agent/selfevolve/config.py) |
| 候选隔离与谱系 | [`candidates/`](../agents/drug-agent/src/drug_agent/selfevolve/candidates/)、[`archive/`](../agents/drug-agent/src/drug_agent/selfevolve/archive/) |
| 独立评估 | [`evaluation/`](../agents/drug-agent/src/drug_agent/selfevolve/evaluation/) |
| 采纳决策 | [`promotion/`](../agents/drug-agent/src/drug_agent/selfevolve/promotion/) 及 `promote` CLI |
| 运行时更新接口 | [`llm/policy.py`](../agents/drug-agent/src/drug_agent/llm/policy.py) |

## 解耦了什么

进化器可以在任务服务之外单独调用，并拥有自己的配置、经验、候选归档、模型预算和评估记录。提出更新不会立即替换 Task Agent 的策略。面向操作者的晋升命令先检查评估证据，再写入策略覆盖层（overlay）；部署端随后显式选择该覆盖层。

本次发布提出的架构重点是这种生命周期分离。这并不意味着后台调度、热部署或通用的跨 Agent 更新接口已经实现。

## 两种进化实现

平台循环通过候选覆盖层和代码改动改变 Drug Agent 的行为。另一个 [`selfevolve/target/`](../agents/drug-agent/src/drug_agent/selfevolve/target/) 实现用于组织针对具体靶点的科学研究活动，PFM 示例来自后者。二者共享设计思路，但不能互相替代，作为同一实验的证据。

肽、mRNA 和酶项目各自拥有任务流程及与进化相关的工作流。它们作为实验性应用纳入仓库；接入共享 Self-Evolve Agent 仍需进一步整合。

## Agent 项目与候选工作区

所有 Agent 入口均位于 [`agents/`](../agents/README_zh.md)。Drug Agent 的 `src`、运行配置、脚本和包元数据都保留在 `agents/drug-agent` 内。其余 Task Agent 与它同级，以保留共享模型资源的发现机制。

候选工作区区分 Git checkout 根目录和 Drug Agent 项目根目录。导入与执行使用候选项目；补丁和受保护路径按项目边界进行规范化。这样既支持共享仓库结构，也避免误评父项目已安装的代码，或利用仓库路径前缀掩盖对受保护文件的修改。

Self-Evolve Agent 有独立调用目录，但 Python 实现仍通过 Drug Agent 包共享，尚未复制或抽取成单独的发行包。

## 与 Harness² 的关系

[Harness²](https://github.com/google-research/harness2) 已经区分执行 harness 与改进 harness，其论文描述了任务内优化和跨任务经验复用。本项目强调可单独调用的 Self-Evolve Agent，以及明确的评估与采纳边界。这里尚未通过条件匹配的消融实验确立两者的相对效率。

