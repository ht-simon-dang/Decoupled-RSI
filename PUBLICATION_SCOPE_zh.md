# 公开副本人工审计清单

[原文](PUBLICATION_SCOPE.md) | [简体中文](PUBLICATION_SCOPE_zh.md)

状态：**公开文件及新版 README 已由用户审阅；本地 Git 历史从本公开副本开始，未配置 remote、未上传 GitHub。**

源仓库保留原样。本目录按明确清单复制，再仅在副本内整理依赖、可移植路径和说明。
原始基线为 `285b23b7b80edf8a36b42b6fde9645c9d810e24e`，同时纳入选定的未提交代码。
该 commit 不代表本副本的完整内容，具体来源见 `PROVENANCE.json`。本次为第 4 版整理，正式 README 已采用审阅通过的中英文版本及四张架构/实例配图；所有 Agent 位于 `agents/`。

文件级清单已在用户完成审阅后移除；本说明与 `PROVENANCE.json` 保留范围和来源记录。
本目录使用独立 `.git/`，初始提交仅包含已审阅的公开内容，不包含源仓库的 commit history。

## 纳入范围

| 范围 | 内容与保留理由 |
| --- | --- |
| `agents/drug-agent/src/drug_agent/` | Drug Task Agent、Self-Evolve Agent、领域工具适配器、运行时评估、晋升逻辑、CLI/API/UI |
| `agents/peptide-agent/` | 肽任务框架、运行配置、示例输入、模型声明与安装入口；标为 experimental |
| `agents/mRNA-agent/` | mRNA 任务框架、运行配置、示例输入、模型声明与安装入口；标为 experimental |
| `agents/Enzyme-Agent/` | 酶任务框架、依赖与资源获取入口；标为 experimental；不附已下载 PDB 示例 |
| `agents/drug-agent/system_prompts/`、`agents/drug-agent/schemas/` | 运行提示模板与公开接口契约 |
| `agents/drug-agent/configs/*.yaml` | 可移植运行配置模板；移除原机器解释器路径和本机部署设置 |
| 各 Agent 的 `environment*.yml`、`envs/*.yml` | 核心运行依赖与可选第三方环境声明 |
| `agents/drug-agent/scripts/external/` | 本项目自己的第三方调用驱动与 upstream revision 清单，不是第三方源码树 |
| `agents/drug-agent/scripts/*.py` | 资源获取、PFM 调用、pocket 准备、demo 和报告入口 |
| 选定 `agents/drug-agent/scripts/research/*.py` | campaign 调度、独立实验执行和 target 数据准备源码；不附原研究记录或冻结配置 |
| `agents/drug-agent/fixtures/` | 默认离线示例必需的小型合成输入及生成器；已排除故意无效的 brief |
| `third_party/` | 按 Agent 家族提供可执行安装入口、机器可读依赖清单、资源文件结构和离线检查 |
| `agents/self-evolve-agent/` | 独立调用入口，复用 Drug Agent 包中的现有演化实现 |
| `examples/pfm_replay/` | 已跑 PFM 的最小脱敏聚合回放及双语说明，600 行 gate 判定 |
| README、`docs/`、`THIRD_PARTY.md` | 架构、运行边界、安装与第三方来源说明；包含四张配图的 SVG/PNG 及可移植绘图脚本 |
| 本说明、`PROVENANCE.json` | 公开范围与来源记录 |

所有 Agent 入口已统一到 `agents/`：Self-Evolve、Drug、Peptide、mRNA、Enzyme。
Drug 项目保留自己的 src/configs/scripts 等内部结构；候选工作区新增 Git 根与项目根
的坐标区分，修复候选导入、补丁应用和保护路径规范化。mRNA/Enzyme 与 Peptide
仍为同级目录。Self-Evolve 目录提供独立入口，底层实现继续共用 Drug 包，未复制一份
相同源码，也没有宣称已完成通用独立 Python 包抽取。

## 明确不公开

- 原 `.git` 历史、`.env` 凭据、个人工具配置和缓存。
- `running-history/`、`.drugagent*/`、`.research/`、原 `0925-pfm-evolve/` 运行目录。
- 原始 chat/transcripts/history/logs、worker debug 导出、截图、验收记录、原始候选列表。
- 所有开发 `tests/`、测试工作流、validation/smoke 专用脚本、测试依赖与测试目标。
- 原 `configs/research/` 下 campaign、baseline、qualification、protocol lock 和其他研究记录。
- 历史统计、回填、转录抽取和机器专用分析脚本。
- `.external/` 第三方 checkout、已安装环境、模型权重、checkpoint、科学输入数据库。
- 原模型转移包、zip/tar.gz、paper/grant/presentation、handoff 和历史研究草稿。
- 带本机 editable 路径或构建机 `file://` URL 的环境 lock 文件。
- salvaged PFM sampler 第三方源码，以及 Enzyme 示例内嵌的已下载完整 PDB/UniProt 记录。

**运行时 evaluation、benchmark、history、validation 函数仍保留。** 它们是框架
运行组件，不是开发测试或原始运行记录。比如 `history.py` 是读取用户未来产生记录的
代码，`selfevolve/evaluation/` 是判断候选改进的执行器。

## PFM 示例的精确边界

- c04/c10：seed 20261001，qualifying scaffolds `77 → 89`。
- c06/c13：seed 20260929，qualifying scaffolds `87 → 88`。
- 每组提交 150 行，四组共 600 行；同一冻结 checkpoint、development 分区。
- 只导出 gate 布尔值、scaffold SHA-256、少量采样计数、改动参数及源记录哈希。
- 不公开分子 SMILES/ID、原始日志、对话、绝对机器路径或权重。
- `replay.py` 重算已记录判定的聚合；不会重新跑 PFM、docking 或科学评分。
- 这不是解耦与非解耦消融，也不证明候选已采纳部署或有生物活性。

## 副本内做过的适配

1. 根 README 与架构导航围绕独立 Self-Evolve Agent；三个扩展标明未完善、待进化与验证。
2. 根依赖去掉开发测试组件；安装入口显式选择 core、PFM、单个第三方来源或独立环境。
3. 根 Makefile 与子项目安装入口移除未发布测试/验收目标；帮助命令不触发安装。
4. mRNA/Enzyme 默认数据目录改为用户 cache 目录，并保留环境变量覆盖。
5. bootstrap 的个人绝对解释器路径参数化；修正对 sibling 虚拟环境依赖的安装处理。
6. Enzyme 参考数据改为用户显式获取时生成，不把已下载第三方结构带入副本。
7. 两处工具说明字符串中的个人/上游私有路径泛化。
8. 主包移至 `agents/drug-agent`；候选 workspace 区分 Git 根与项目根，支持项目相对和
   Git 根相对补丁，拒绝越界及受保护路径变更。路径模块加入评估完整性哈希。
9. 新增分组件第三方安装、计划与检查入口；文件存在检查不冒充模型就绪或科学验证。
10. 保留原 gate 0 回归要求；旧 PFM 记录保留旧 evaluator hash，不冒充新代码重新执行。
11. 用户审阅后移除文件级清单；增加说明文档的 `_zh` 简体中文版及语言切换链接，不改动运行提示模板或配置；提示校验脚本仅增加对 `README_zh.md` 的跳过规则。

12. 将审阅通过的 README 初稿正式纳入公开根目录，修正仓库相对链接并加入四张配图；绘图脚本直接读取公开仓库内的 PFM 摘录。

## 人工审阅覆盖的重点

- [x] 四个 Task Agent 的范围、实验性标签和当前整合程度是否准确。
- [x] 确认所有 Agent 入口统一到 `agents/`；检查 Self-Evolve 入口与共享底层实现的说明。
- [x] 是否接受公开这 600 行脱敏 PFM 评估摘录和候选改动参数。
- [x] 是否接受离线药物 demo 的合成 fixtures；它们不属于原研究运行数据。
- [x] 是否接受保留选定 target 准备与 campaign 入口源码，但不附历史配置与数据。
- [x] 核对第三方组件的 install.sh、依赖清单、预期文件结构及需要手工提供的资源。
- [x] 核对公开分发时适配的文件，尤其依赖、默认路径和安装脚本。
- [x] 确认下面的运行限制符合首发预期，再进行提交和上传。

## 已检查与尚未验证

已完成 Python AST、JSON/TOML/YAML、shell 语法检查；核心 CLI 与自演化模块从副本
导入成功；安装帮助与 Makefile 帮助可读取；PFM 聚合回放通过。另检查了排除目录、
符号链接、凭据特征、个人路径、相对 Markdown 链接，以及首次复制的 506 个源文件
哈希保持不变。目录迁移还在临时 Git 仓库中通过 25 项检查，验证候选补丁、实际导入、保护路径、兄弟项目输出边界和父仓库不变性；这些临时校验脚本未放入公开副本。凭据特征扫描不等同于完整安全审计。

未重新安装科学环境、未调用 LLM/GPU、未运行新的科研实验或私有回归套件。

当前完整 platform self-evolve 仍要求被排除的私有回归套件，因此无法通过 gate 0；
部分实验性后端也需要未公开的本地 readiness 验证记录。原 ZAK campaign 不构成
可直接重跑的公开复现包。这些限制均在公开 README/文档中明示。

用户指定的文件夹名含 `:`；本目录可供审计及标准库 PFM 回放。实际安装与演化应放在
不含冒号的独立 checkout（如 `decoupled-rsi-agent`），并使用已提交的基线版本。

