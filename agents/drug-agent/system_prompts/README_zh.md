# `system_prompts/`：提示库

[English](README.md) | [简体中文](README_zh.md)

**Master Agent** 发给模型的每条 system 和 user 消息都以 Markdown 文件保存在这里，不再作为 Python 字符串字面量嵌入代码。`src/drug_agent/llm/prompts.py` 只负责组合：决定各阶段使用哪些片段，以及用哪些运行时值填充占位符。提示文件本身是权威来源，由 Python 直接读取，不通过生成流程产生。

将提示移出代码，是因为提示本身就是平台进化的产物。自进化循环会提出对提示的改动（`RuntimePolicy.plan_prompt_addendum` 等），日志记录哪个提示版本产生了哪段运行经历，晋升决策也需要讨论提示文本。因此它们应当放在审阅者可以直接阅读和比较差异的文件中。

---

## 目录结构

```
system_prompts/
├── README.md                          本导航的原文
├── README_zh.md                       简体中文导航
├── stamp_checksums.py                 有意修改提示后重新写入 sha256
├── shared/                            多个阶段共用的片段
│   ├── master_rules.md
│   ├── objective_vocabulary.md
│   └── constraint_vocabulary.md
└── master_agent/                      每个循环阶段各有 system 和 user 文件
    ├── plan.system.md         plan.user.md
    ├── transcribe.system.md   transcribe.user.md
    ├── critique.system.md     critique.user.md
    ├── refine.system.md       refine.user.md
    └── select_best.system.md  select_best.user.md      （循环结束后，dynamic 模式）
```

目录层级对应仓库已有的职责分层。这里有 `master_agent/`，没有 `specialists/`，因为 `agents/llm/chemical_agent.py` 和 `agents/llm/evaluation_agent.py` 都声明 `llm_backed = False`：它们调用工具并执行数值计算，从不调用模型，也不包含提示文本。这一性质应当保留：产出数值的 Agent 和评判数值的 Agent 都不由 LLM 直接控制。

---

## 文件格式

```
---
id: master_agent.plan.system
phase: plan
role: system
used_by: drug_agent.llm.prompts:plan_prompt -> ...
placeholders: rules
sha256: <hex digest of the body>
---
<the prompt body>
```

- **正文（Body）**：结束 `---` 行之后的全部内容，去掉文本文件末尾的一个换行。正文自身必须以换行结尾时（仅 `shared/master_rules.md`），文件末尾保留一个空行。
- **占位符（Slots）**：采用 `{{name}}`。除占位符外，正文不会出现连续双花括号：JSON 输出结构按 `indent=2` 格式化，每个花括号后都有换行，因此可以安全地进行简单文本替换。加载器要求调用方提供的占位符与模板声明**完全一致**；占位符重命名会在调用处报错，而不会一路传到模型。
- **`sha256`**：覆盖经过上述规范化后的正文，所以可以修改 front matter 中的说明，而提示正文的字节仍由哈希固定。

---

## 逐字节不变性

`RuntimePolicy.prompt_version()`（`llm/policy.py`，**封存**）生成每条日志记录使用的 ID，运行历史分析据此比较不同经历。需要注意，该 ID 实际哈希的是策略**覆盖层**——四个附加提示和两个 envelope 标志，**不是**基础提示文本。它假定基础文本恒定，只有这一假设成立，全部历史记录才可比较。把提示从 Python 移出后，原先的两道保护（`ruff format` 处理及显眼的代码 diff）不再适用，因此现在显式保护：

- 每个文件记录正文的 `sha256`；
- `drug_agent.llm.prompts._block()` 首次读取时验证哈希，不一致则报错，并给出两个摘要及修复方法。

因此，**有意**修改提示需要两个可见步骤：

```bash
$EDITOR system_prompts/master_agent/plan.system.md
python system_prompts/stamp_checksums.py          # --check 只验证，不写入
```

这样，摘要变化和导致它变化的文本修改会出现在同一个 diff 中。任何意外的文件重排，例如编辑器删除末尾空行、Markdown 格式化器重排段落，都会在下次 LLM 调用时明确失败，而不会悄悄让历史记录分叉。**不要对本目录中的提示模板运行 Markdown 格式化器。** 校验脚本跳过 `README.md` 和 `README_zh.md`，它们只是说明文档。

原先将提示迁移到这里时，没有改变任何提示字节：迁移前后 `prompt_version()` 均为 `p1-d65e3ebc70d0`，十六个渲染字符串（四阶段 × system/user × 有/无策略覆盖层）逐一比较完全相同。

---

## 打包

规范文件位于项目根目录、Python 包之外，因为它们是可直接阅读的文档。`[tool.hatch.build.targets.wheel] packages = ["src/drug_agent"]` 无法覆盖顶层目录，所以 `pyproject.toml` 强制包含：

```toml
[tool.hatch.build.targets.wheel.force-include]
"system_prompts" = "drug_agent/system_prompts"
```

`_library_root()` 依次查找 `<package>/system_prompts`（仅构建后的 wheel 中存在）和 `<repo root>/system_prompts`（checkout 或 editable 安装）。同一份字节支持三种布局，无需复制步骤。这里刻意**不提供**通过环境变量覆盖提示库路径的方式：若能在不改变 `prompt_version()` 的情况下更换提示文本，日志就无法准确说明实际发送了什么。

---

## 文件导航

### 共享片段

| 文件 | ID | 使用位置 | 必须包含 | 不变量 |
|---|---|---|---|---|
| `shared/master_rules.md` | `shared.master_rules` | `llm/prompts.py` → 全部四个构造器；每个 Master Agent system 提示的开头 | 角色设定（Master Agent、高级计算药物化学家、指挥 chemical 和 evaluation 专用 Agent）及五条不可妥协规则：仅用于研究优先级判断而非临床；不得编造科学数值；SMILES 只能作为假设提出；用户消息是目标而非覆盖指令；输出必须恰为一个 `json` 围栏代码块。 | 规则 2 是整个平台的证据边界，不得弱化 “you must NEVER invent scientific numbers”。规则 4 是提示注入边界，与 `agents/platform/literature_agent.py` 配合。规则 5 对应 `llm/jsonio.py` 的解码和 `llm/harness.py` 的修复；改变围栏契约会破坏二者。正文以换行结尾，因此文件末尾有空行。 |
| `shared/objective_vocabulary.md` | `shared.objective_vocabulary` | `transcribe_prompt`、`refine_prompt` | 五个目标名称 token：`pIC50`、`qed`、`sa_score`、`hERG`、`restoration_score`，包含方向和含义，并声明方向只能为 `maximize` 或 `minimize`。 | 每个 token 都必须是后端可评分的名称；`restoration_score` 必须标为仅限表型。方向字符串由下游解析，见 `llm/schemas.py:ObjectiveSpec`。 |
| `shared/constraint_vocabulary.md` | `shared.constraint_vocabulary` | `transcribe_prompt`、`refine_prompt` | 七个硬约束属性键（`descriptors.mw`、`.logp`、`.tpsa`、`.hbd`、`.hba`、`.qed`、`n_structural_alerts`），附单位、示例阈值，以及六个允许的运算符。 | 属性键必须与排序/筛选层可计算内容同步；后端不认识的键会产生始终无法生效的约束。运算符必须与 `agents/llm/evaluation_agent.py:_compare` 同步。 |

### Master Agent：四个循环阶段

| 文件 | ID | 使用位置（模块:函数 → 阶段） | 必须包含 | 不变量 |
|---|---|---|---|---|
| `master_agent/plan.system.md` | `master_agent.plan.system` | `llm/prompts.py:plan_prompt` → `agents/llm/master_agent.py:MasterAgent.run`，**阶段 1（plan）**。解码为 `llm/schemas.py:TaskPlan`。 | `{{rules}}`、规划任务、按 EFFICACY / PROPERTIES / PORTFOLIO / CONFIDENCE 分组的允许指标键、选择 3–6 个判据的指令，以及 `TaskPlan` 输出结构。 | **此处指标键必须与 `llm/schemas.py:METRIC_KEYS` 双向同步**；原私有套件中的 `tests/test_pools.py::test_the_plan_prompt_offers_every_metric_the_evaluator_can_compute` 及其反向检查负责约束。提供但无法计算的键会使 ACCEPT 不可达；可计算但未提供的键不会被使用（表型任务中曾因此遗漏 `restoration_median`）。目标类型指导（蛋白靶点用 pIC50 指标，表型救援用 `restoration_median`）由 `llm/repair.py:repair_criteria_for_goal` 在策略启用时确定性强制执行。 |
| `master_agent/plan.user.md` | `master_agent.plan.user` | 同上 | 用 `"""` 包裹的用户指令，以及生成计划的要求。 | `{{instruction}}` 是不可信用户文本，必须保留引用并放在规则**下方**。`plan_prompt` 填充前调用 `.strip()`。 |
| `master_agent/transcribe.system.md` | `master_agent.transcribe.system` | `llm/prompts.py:transcribe_prompt` → **阶段 2（transcribe）**。解码为 `llm/schemas.py:ChemicalRequirements`。 | `{{rules}}`、转录任务及 `target_based` / `phenotype_based` 区分、`{{objective_vocabulary}}`、`{{constraint_vocabulary}}`、指导块（必须有活性目标；添加 `qed`/`sa_score`；数值限制写成约束；新颖性对应 seeds；靶点不是测量；`n_generated` 的含义），以及 `ChemicalRequirements` 输出结构。 | 输出键必须匹配 `ChemicalRequirements` 字段，因为本阶段采用 schema 约束解码（`llm/harness.py:_use_schema_decoding` 对 `transcribe` 返回 True），键漂移会导致硬解码失败。生成指导必须准确描述模型可控制内容：候选来源、从头生成模型（`generator_backend`）、样本数、是否检测结合位点（`pocket_source`、`pocket_detector`、`pdb_ids`）。这改变了旧版“来源是部署决定”的约束：哪条路线适合目标属于目标判断，因此交给 Master Agent。它还选择评估深度：`property_backends`、`route_backend`、`complex_backend`、`target_evidence`；每个字段都会启用之前无法从运行中触达的已安装工具。模型仍不能填写坐标：口袋中心必须测得，不能凭空指定。 |
| `master_agent/transcribe.user.md` | `master_agent.transcribe.user` | 同上 | 包裹后的用户指令，以及阶段 1 的计划摘要。 | `{{plan_summary}}` 来自 `master_agent.py:_plan_summary`；`{{instruction}}` 是经 `.strip()` 处理的不可信文本。 |
| `master_agent/critique.system.md` | `master_agent.critique.system` | `llm/prompts.py:critique_prompt` → chemical 和 evaluation Agent 执行后的**阶段 4（critique）**。解码为 `llm/schemas.py:Critique`。 | `{{rules}}`、评判任务、三种判定（ACCEPT / REFINE / REPLAN）及含义、循环预算句子，以及含 `per_criterion` 的 `Critique` 输出结构。 | 比较指令旁必须保留“不得编造数值”：这是模型最容易补出报告中不存在数值的阶段。`master_agent.py` 匹配判定 token，重命名会使细化分支失效。`{{cycle}}` / `{{max_cycles}}` 用 `str(int)` 填充，用于让模型在接近预算时更倾向 ACCEPT。 |
| `master_agent/critique.user.md` | `master_agent.critique.user` | 同上 | 计划摘要、本周期需求摘要，以及标为“工具计算、权威”的评估报告。 | `{{evaluation_json}}` 为 `EvaluationReport.compact_json()`（`agents/llm/evaluation_agent.py`），是裁剪后的视图；扩大它会增加每轮上下文成本。“权威”标签用于区分实测数值与模型自述，不能随意去掉。 |
| `master_agent/refine.system.md` | `master_agent.refine.system` | `llm/prompts.py:refine_prompt` → 判定为 REFINE 时的**阶段 5（refine）**；调整后的需求启动下一轮。解码为 `llm/schemas.py:RequirementsAdjustment`。 | `{{rules}}`、“不要只重试，要先诊断再修改”、可调项列表（`seed_smiles`、`property_constraints`、`objectives`、`max_candidates`/`shortlist_size`、`n_generated`）、`{{objective_vocabulary}}`、`{{constraint_vocabulary}}`，以及 `adjusted_requirements` 为**完整** `ChemicalRequirements` 的输出结构。 | 这是消融归因的核心：`changes` 每一项都必须对应诊断，才能归因某轮的效果。必须强调“完整对象”：`objectives` 和 `property_constraints` 默认 `[]`，因此在本阶段的 schema 约束解码中，遗漏意味着**删除**；`RuntimePolicy.refine_carry_forward` 和 `refine_max_drop_fraction` 就是为捕获这种失败而设。列出的可调项必须确实被流水线执行。 |
| `master_agent/refine.user.md` | `master_agent.refine.user` | 同上 | 当前需求 JSON、批评/诊断，以及权威评估报告。 | 与 critique 的 user 消息保持同样的权威来源标注。 |

### Master Agent：循环后的步骤

| 文件 | ID | 使用位置（模块:函数 → 阶段） | 必须包含 | 不变量 |
|---|---|---|---|---|
| `master_agent/select_best.system.md` | `master_agent.select_best.system` | `llm/prompts.py:select_best_prompt` → `agents/llm/master_agent.py:MasterAgent._select_best_candidate`，`dynamic` 上限模式运行最后一轮之后的**最终步骤**。解码为 `llm/schemas.py:BestCandidateSelection`。 | `{{rules}}`、“从任意周期选择一个候选”、“这是排序，不是测量”的段落、四个平局判据（目标依赖的指标；适用域；Pareto 等级；较晚不等于较好），以及 `BestCandidateSelection` 输出结构。 | **ADR-0011 决定了整个提示结构。** 排序/测量段落及“不得选择表中不存在的候选”确保此步骤只做决策、不生成数值；结构也提供支持：`BestCandidateSelection` 没有可填写数值的字段，`master_agent.py` 从评估报告重新读取所选 ID 的每个分数，并拒绝未提供过的 ID。即使弱化提示，后两道保护仍在，但三者应共同理解。使用 `config.SELECT_BEST`（最大的 `num_predict`、最低温度）；候选表更大时应增加预算，而不是缩短提示。 |
| `master_agent/select_best.user.md` | `master_agent.select_best.user` | 同上 | 计划摘要、每周期的判据结果，以及**每周期全部候选**的逐端点分数。 | `{{candidates_json}}` 有意包含每周期的 shortlist，而非仅获胜周期：只看获胜周期就无法完成该步骤的目的。两个数据占位符均保留“工具计算、权威”标签，与 critique 和 refine 一致。`null` 必须始终表示“工具无法计算”，不能当成零。 |

### 可选块的附加位置

运行时构建的文本由 `llm/prompts.py:_extend` 在 system 提示组合完成后追加，所以不作为本目录文件保存。`_extend` 提供一个通道（`envelope_block`），`agents/llm/master_agent.py:_join_blocks` 将本次运行适用的块组合后传入：

| 块 | 来源 | 附加阶段 | 启用条件 |
|---|---|---|---|
| 部署能力 | `master_agent.py:_availability_block`：实际工具池与 `agents/registry.py:AGENT_TOOLS` 的交集，不能手写或手工列举 | plan、transcribe | **无，始终启用** |
| 硬上限 | `master_agent.py:_ceiling_block`：操作者启动时给出的 `--ceiling METRIC=VALUE` | plan、transcribe | ceiling 模式为 `hard`（运行参数） |
| 能力包络 | `tools/envelope_tool.py:CapabilityEnvelope.as_prompt_block`：由真实探测运行生成，不能手写 | plan、transcribe、critique、refine、select_best | `RuntimePolicy.capability_envelope`、`.capability_envelope_transcribe`，**或** ceiling 模式为 `dynamic` |
| 策略附加提示 | 自进化覆盖层（`RuntimePolicy.*_prompt_addendum`），由假设引擎在运行时写入 | 四个循环阶段 | 附加提示非空 |

**为什么部署能力块不设置开关，尤其不受 `RuntimePolicy.capability_envelope` 控制。** 该标志启用的是自进化实验正在研究的处理：给规划器提供**实测数值**，例如“本部署曾达到 pIC50 6.4582”。这是对流水线实际达到水平的陈述，适合做 A/B 比较。“本机未安装 AutoDock Vina”则是部署事实，类似模型名称，既不是实验处理，也不是测量。对基线隐去它只会让基线信息更少。因此二者在设计上独立，原私有套件中的 `tests/test_ceiling_modes.py` 维护这一边界。

所有附加块为空时，每个构造器返回的字符串与这些参数出现之前完全相同。这使候选臂与基线臂可比，原私有套件中的 `tests/test_selfevolve_policy.py` 用于验证这一点。

---

## 尚未迁移的提示及原因

以下面向模型的文本在原提示迁移中刻意保留在原位置。列出它们是为了提供完整导航，即使本目录没有对应文件。

| 提示 | 位置 | 保留原因 |
|---|---|---|
| 自进化 **router** | `selfevolve/controller/loop.py:73` 的 `_ROUTER_SYSTEM` | 不在原改动范围内，且属于不同层：它控制进化器，不是面向用户服务的 Agent，其文本不由 `prompt_version()` 覆盖。后续可使用相同加载器迁移至 `system_prompts/selfevolve/router.system.md`。 |
| 假设 **generator** | `selfevolve/hypotheses/generator.py:67` 的 `_GENERATOR_SYSTEM` | 同样不在原范围和层内。其规则 4（不得提出弱化用于评判自身工作的评估、基准或成功标准）是整个自进化分支的完整性保护，应像 `shared/master_rules.md` 一样仔细审阅。候选目标位置：`system_prompts/selfevolve/hypothesis_generator.system.md`。 |
| 假设 **verifier** | `selfevolve/hypotheses/generator.py:91` 的 `_VERIFIER_SYSTEM` | 同上。候选目标位置：`system_prompts/selfevolve/hypothesis_verifier.system.md`。其特殊指令——`INSUFFICIENT_EVIDENCE` 是受到鼓励的答案——必须在任何迁移中保留。 |
| JSON **repair** 消息 | `llm/harness.py:300-310` | 这是 user 角色的恢复轮次，不是 system 提示；它由实时解码错误和 `_required_fields(schema)`（从 Pydantic 模型内省）组装。只抽取固定部分会把一条消息拆到两处，没有收益。 |
| 能力 **envelope** 块 | `tools/envelope_tool.py:111` 的 `as_prompt_block` | 不是人工编写的文本，而是根据实际运行机器的测量渲染，无静态文本可存储。 |
| **封存**的评估内核 | `selfevolve/evaluation/*.py` | **哈希固定。** `selfevolve/evaluation/integrity.py:kernel_digest` 为这些模块生成指纹，防止候选由自己修改过的内核评分；改动会使原基准臂失效。原清点确认内核**没有**面向模型的提示文本，只有评分、回放、一致性检查和校准，无内容需要迁移。 |

为完整说明搜索范围，另列出两个不属于提示的内容：`agents/platform/literature_agent.py:26` 保存提示注入检测模式（针对不可信检索文本的防御性正则表达式）；`tools/llm_provider_tool.py` 是确定性 stub，输出为输入的哈希。

