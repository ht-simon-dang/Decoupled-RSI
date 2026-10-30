# EMA4D Antibody Agent

按本目录的《疾病导向抗体 AI Design Agent 功能蓝本》实现的独立研究 agent。
界面参考 Enzyme、mRNA、peptide agent 的工作台、Evaluation、反馈和 Agent map，
默认运行文件、模型配置与日志均保存在本目录，不依赖修改仓库主启动器。

## 启动

从仓库根目录执行：

```bash
./agents/antibody-agent/install.sh
./agents/antibody-agent/run.sh serve --port 2007
```

访问 <http://127.0.0.1:2007>。结构化机制探索可立即使用：

```bash
./agents/antibody-agent/run.sh run agents/antibody-agent/examples/disease.json
./agents/antibody-agent/run.sh doctor
./agents/antibody-agent/run.sh schema
```

需要 Python 3.11 或 3.12。安装脚本的缓存与虚拟环境也位于本目录。
`ANTIBODY_BOOTSTRAP_PYTHON` 可指定创建虚拟环境的 Python。

## 0.8B 主控

固定使用 `Qwen/Qwen3.5-0.8B`，权重版本为
`2fc06364715b967f1860aea9cf38778875588b17`。使用 JSON Schema 约束解码与严格语义校验，
只推断任务、方向与路线。VH/VL/VHH 和抗原序列只能从用户原文提取；主控不编写序列、
测量值或文献证据。

```bash
./agents/antibody-agent/install.sh llm
./agents/antibody-agent/run.sh download-qwen
./agents/antibody-agent/run.sh instruct '梳理一个自身免疫病抗体项目的机制与证据缺口，仅做研究规划。'
```

也可以复用已有权重和推理解释器，在本目录的 `.antibody-agent/runtime.env` 中设置：

```bash
export ANTIBODY_QWEN_DIR=/absolute/path/to/Qwen3.5-0.8B
export ANTIBODY_LLM_PYTHON=/absolute/path/to/llm-env/bin/python
```

这个文件是受信任的本地 shell 配置，已被 Git 忽略。它不读取仓库全局的 runtime.env。
结构化任务不要求额外经过 LLM；两种入口最终使用同一份 Brief 合约。

## 设计与强制评估

执行顺序为：项目档案与机制检查 → 编号/编辑约束 → 分子生成 → 独立折叠 →
TAP/TNP 开发画像与复合物复核 → 多状态/表位/格式筛选 → 不同家族组合 → human 审阅。

支持疾病、已有靶点、配对序列、复合物、实验反馈五种入口。疾病输入不完整时先建立
研究草案，明确列出重要缺项。科学工具只在其输入条件成立时运行。

提交完整任务可使用 WebUI 中的 JSON 编辑器，或 `run brief.json`。主要对象为：

- `states`：实际靶点实体、物种、亚型、状态、序列、天然呈递、正/负靶与来源。
- `epitopes`：功能理由、允许/禁止接触、可达性证据；残基编号为抗原序列的 1-based 位置。
- `reference` / `library`：完整配对 VH/VL 或明确标识的 VHH，配对来源、框架结构与编号。
- `constraints`：IMGT/AHo/Kabat 标签、可编辑/保护位置和最大突变数。原生编号使用 IMGT。
- `product`：格式、价态、Fc 需求、链组织、融合与偶联版本。交付的序列范围明确为可变区。
- `budget`：候选、工具调用、总时间、审阅组合和实验名额。恢复不会重置已消耗预算。
- `evaluation_policy`：预先声明的计算门槛；默认阈值未按项目对照校准，不是结合概率。

同一可变区更换轻链、Fc、连接方式或格式后，完整分子身份会变化；抗原状态、测量条件、
批次和糖型各自绑定证据。五种成功分别记录，技术失败和删失不充当分子阴性。

候选只有在流水线完成并通过所有必需计算检查后，才会出现在 API、CLI、WebUI、FASTA、
CSV、JSON、报告、结构下载和下一批推荐中。每次读取会重新核对评价器版本和证据；
修改已保存的 `pass` 或候选列表不能跳过评价。未通过原因可见，原始未筛选数据保存在本地 SQLite。

原生复核使用无设计模板/强制接触的 ImmuneBuilder 和 Boltz-2。BoltzGen 来源的候选改用
Chai-1 桥接复核，避免使用同家族模型作为唯一独立判断。Boltz-2 原生适配器使用
single-sequence 模式，记录其精度局限，不调用针对小分子的 affinity 模块。

## 科学模型配置

`install.sh` 安装 agent，不自动下载所有分子模型。请按
[适配器文档](docs/model-adapters.md) 配置 `.antibody-agent/runtimes.json`。
`configs/runtimes.example.json` 是结构示例，其中的路径和版本需要替换，不能直接当成已安装模型。

| 能力 | 本版执行路径 |
| --- | --- |
| RFantibody | 原生三阶段 CLI：RFdiffusion → 基础 ProteinMPNN → RF2；RF2 仅为内部自洽检查 |
| AntiFold | 原生固定骨架、明确 IMGT 位点的配对采样 |
| ANARCII | 原生 IMGT 编号与链类型检查，禁止静默截短 |
| ImmuneBuilder | 原生 ABodyBuilder2 / NanoBodyBuilder2；检查四个已安装权重 |
| Boltz-2 | 原生独立复合物预测，实际坐标检查与抗原界面 ipTM |
| BoltzGen、IgGM、IgDesign、p-IgGen、LICHEN、Sapiens、Humatch | 已注册、受约束的版本化 JSON 模型桥接，需部署相应 bridge |
| Chai-1、TAP、TNP | 独立复核/格式画像桥接；缺失时不放行候选 |

每次运行记录实际模型版本、权重版本、请求与输出哈希、种子和工具成本。
`configured_unverified` 仅表示配置存在；不是模型推理或科学有效性已验收。

本版原生设计支持完整配对或 VHH 的固定长度编辑。单 VH 补全缺失轻链、插入/删除、
编号体系转换和高级分子装配仍需专门的映射/装配实现；缺少这些条件会明确阻塞。
双抗、ADC、条件激活和融合体不能用单臂证据通过整体验收。膜和糖链已知时，需要
桥接器明确执行天然背景检查；未知时保留未知，不声称已验证无糖链或膜遮挡。

## 实验反馈、学习与系统改进

```bash
./agents/antibody-agent/run.sh feedback RUN_ID --data measurements.json
./agents/antibody-agent/run.sh review RUN_ID CANDIDATE_ID accept --note '记录本轮审阅理由'
./agents/antibody-agent/run.sh train-property training.json
./agents/antibody-agent/run.sh next-batch RUN_ID objectives.json
./agents/antibody-agent/run.sh benchmark RUN_ID
```

测量记录通过 `candidate_id` 与 `context_id` 绑定分子/抗原，另存条件、批次、糖型、重复、
对照与误差。观察 ID 不可覆盖；纠错追加新 ID。WebUI 可生成测量模板，模板中的空值必须
替换为真实数据后才能入库。结合但不阻断、开发风险和技术失败会产生不同的下一轮建议。

属性学习是明确的 **配对链组成 Ridge + 按谱系分组的 split conformal 基线**，不是预置
LaMBO-2 抗体性质权重。要求至少 8 条训练、4 条校准、4 条测试实测记录，校准至少四个谱系；
谱系、序列和原始测量不能跨拆分。测试集不参与拟合。80% 名义区间、实测测试覆盖率、MAE
与均值基线分开报告。删失/技术失败计数和完整数据哈希仍保留。

多目标选批比较 Pareto 前沿并保留家族探索名额，仅能从当前已通过独立评估的候选中选择。
`benchmark` 固定同一候选池、预算和评价器，对比置信度排序与家族覆盖。它不会通过更改
合格定义或删除失败来取得收益，也不会把计算重放冒充独立生物学增益或自动升级默认模型。
系统级代码补丁自动产生/部署和跨 Agent 权重迁移不在本版运行路径中。

## 运行记录和开发

```bash
./agents/antibody-agent/run.sh history
./agents/antibody-agent/run.sh show RUN_ID
./agents/antibody-agent/run.sh resume RUN_ID
./agents/antibody-agent/run.sh reevaluate RUN_ID
./agents/antibody-agent/run.sh export RUN_ID --format fasta
./agents/antibody-agent/install.sh mcp
./agents/antibody-agent/run.sh mcp
```

`reevaluate` 建立新运行、明确新预算，保留原始生成谱系和历史证据；不会覆盖原项目。
默认 `ANTIBODY_DATA_DIR=.antibody-agent`，`--root` 仅覆盖运行日志目录。
WebUI 默认只监听 localhost，拒绝跨站写入，没有任意 shell 执行或原始检查点下载接口。

```bash
./agents/antibody-agent/install.sh test
cd agents/antibody-agent
make test
.venv/bin/ruff check src tests
```

测试夹具是明确的合成软件协议数据，不是成功设计的抗体或科学模型替身。
`pyproject.toml` 明确列出发布资源，模型、运行记录、测试与本地输入不会混入 wheel/sdist。
见[架构与验收说明](docs/architecture_zh.md)。
