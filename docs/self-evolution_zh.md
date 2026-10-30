# Self-Evolve Agent

[English](self-evolution.md) | [简体中文](self-evolution_zh.md)

入口是 [`agents/self-evolve-agent/run.sh`](../agents/self-evolve-agent/run.sh)，它从 Drug Agent 项目运行平台接口 `drugagent self-evolve`。共享源码保留在 [`agents/drug-agent/src/drug_agent/selfevolve/`](../agents/drug-agent/src/drug_agent/selfevolve/)，默认配置位于 [`configs/self_evolve.yaml`](../agents/drug-agent/configs/self_evolve.yaml)。

循环读取本地产生的请求日志，对测得的失败进行分组，提出候选策略或补丁，并独立评估候选。归档保存候选谱系和决策。仓库包含日志记录与存储的**代码**，但不包含原操作者的日志和存储内容。用户运行软件时会生成自己的记录。

## 可用源码接口

- `doctor`：展示配置和后端可用性。
- `analyze`：读取配置的日志目录中的经验。
- `baseline`、`step` 和 `run`：在所需运行环境与回归检查齐备时，执行评估和改进循环。
- `report`：解释某个候选的评估结果及采纳决策。
- `promote`：要求评估证据和显式操作者标志，然后写入覆盖层。部署端读取 `DRUGAGENT_EVOLVE_OVERLAY`。

这些命令可能产生本地状态或调用已配置的模型后端，不属于标准库 PFM 回放示例。

## Gate 0 与未公开的开发测试套件

现有内核调用 `python -m pytest -q -x tests` 作为回归关卡。本次发布按范围要求排除私有 `tests/` 目录和开发依赖，因此仅凭本副本，默认完整平台进化循环无法通过 gate 0。

发布版保留这一失败行为。实际部署必须提供兼容的回归套件及依赖，或有意识地接入自己的不变量检查。缺少套件不等于评估成功。运行时评估内核、准入规则和晋升检查属于框架组件，因此仍以源码形式保留在仓库中。

## 按靶点组织的研究活动

[`scripts/research/`](../agents/drug-agent/scripts/research/) 保留研究活动调度与靶点准备入口。历史研究注册、协议锁定文件、资格验证记录、留出分区及外部科学数据集不随仓库发布。开展新研究需要显式提供兼容的输入和评估配置；本仓库不宣称旧研究命令可以直接运行。

[PFM 示例](../examples/pfm_replay/README_zh.md) 是对已记录评估聚合结果的整理回放，不提供原始研究存储，也不提供重复分子生成和评分所需的全部资源。

## 隔离与采纳

候选 worktree 隔离候选文件，并明确当前评估的是哪份代码。但让 worker 以同一操作系统用户运行，并不能使受保护数据不可访问。部署层面的访问控制与内核完整性检查是不同的机制。

本公开副本有独立的 Git 仓库，历史从已审阅的源码副本开始，不包含外层源仓库历史。本地发布副本尚未配置 remote。候选工作区同时跟踪 Git checkout 根目录和嵌套的 `agents/drug-agent` 项目根目录。审阅完成后，使用候选 worktree 前，请先将项目放到路径不含 `:` 的独立 checkout，并使用已提交的基线版本。

