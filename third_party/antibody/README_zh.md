# Antibody 依赖

[English](README.md) | [简体中文](README_zh.md)

本清单对应 Antibody Agent 当前的
[Python 依赖声明](../../agents/antibody-agent/pyproject.toml)、
[模型注册表](../../agents/antibody-agent/src/antibody_agent/registry.py)及
[原生适配器](../../agents/antibody-agent/src/antibody_agent/adapters.py)。
[dependencies.json](dependencies.json) 记录包版本范围、固定 0.8B 权重版本、上游来源、
运行配置字段、预期文件和集成边界。它是声明式依赖描述，不是自动安装器或检查器。

这里不重新分发第三方源码、环境、权重、实验数据或本地验证记录。许可字段沿用当前
agent 注册表，不代表重新审查了上游条款；具体代码、权重、参考面板和服务应分别核对。

## 已有安装入口

需要安装时，在仓库根目录执行：

```bash
# 基础运行环境与 WebUI；不安装分子模型：
./agents/antibody-agent/install.sh base
# 可选 0.8B 主控环境与显式权重下载：
./agents/antibody-agent/install.sh llm
./agents/antibody-agent/run.sh download-qwen
# 可选 MCP 或开发检查：
./agents/antibody-agent/install.sh mcp
./agents/antibody-agent/install.sh test
# 查看模型配置状态，不启动推理：
./agents/antibody-agent/run.sh doctor
```

基础包要求 Python `>=3.11,<3.13`，即 3.11 或 3.12。
`ANTIBODY_BOOTSTRAP_PYTHON` 可指定创建虚拟环境的解释器。
基础依赖为 Pydantic、FastAPI、Uvicorn、PyYAML、Biopython、NumPy；`llm` 额外安装
Torch、Transformers、LM Format Enforcer 和 Hugging Face Hub。
准确版本范围见 JSON 清单，以 `pyproject.toml` 为准。

本目录没有新增 `install.sh`、通用 `plan/check/install` 命令或仓库根分发器入口。
上述命令均复用 agent 已有实现。科学后端需要另行准备兼容环境、资产与配置。

## 主控与科学后端

| 组件 | 当前接入方式 | 所需资源与职责 |
| --- | --- | --- |
| Qwen3.5-0.8B | 原生主控 | `Qwen/Qwen3.5-0.8B`；固定 revision `2fc06364715b967f1860aea9cf38778875588b17`；完整配置、tokenizer 和 safetensors 快照 |
| ANARCII | 原生编号 | `anarcii` 包及配套模型资产；完整可变区 IMGT 编号 |
| RFantibody | 原生生成 | 兼容的 `rfdiffusion`、`proteinmpnn`、`rf2` 可执行文件；扩散、基础 ProteinMPNN、RF2 权重；HLT 框架与靶标输入 |
| AntiFold | 原生生成 | `antifold`、显式 checkpoint、IMGT 编号的配对/VHH 结构与可编辑位点 |
| ImmuneBuilder | 原生独立折叠 | ABodyBuilder2 / NanoBodyBuilder2、编号依赖、OpenMM/pdbfixer，以及对应格式的四个权重 |
| Boltz-2 | 原生复合物复核 | 所选 Python 同目录的 `boltz`、完整 cache 和显式 checkpoint；完整靶标序列 |
| BoltzGen、IgGM、IgDesign | JSON 生成桥接 | 单独部署的模型、权重与兼容桥接器；登记名称不等于已有原生实现 |
| p-IgGen、LICHEN | JSON 生成桥接 | 配对/轻链模型与桥接器；序列先验不代表抗原结合证据 |
| Sapiens、Humatch | JSON 生成桥接 | 人源化模型与桥接器；生成后仍需独立评价 |
| Chai-1 | JSON 评价桥接 | 独立复合物预测环境与桥接器；用于 BoltzGen 来源候选的复核 |
| TAP / TNP | JSON 开发画像桥接 | 实际格式适配的分析工具、参考面板与固定判定规则；不能用序列 motif 检查冒充 |

根据项目路线选择生成器，不要求每轮运行所有生成模型。候选交付需要有效编号、独立折叠、
格式对应的 TAP/TNP 证据，以及正向抗原状态的复合物/表位评价。必需依赖缺失时不放行。
配置存在、权重存在或单次 smoke test 成功均不代表已证明结合、功能或科学有效性。

原生 Boltz-2 使用 single-sequence 模式，不注入设计模板、强制接触、远程 MSA 请求，
也不调用小分子 affinity 模块。蛋白序列复核不能证明膜或糖链背景兼容。各模型输入和
天然状态检查要求见[适配器合同](../../agents/antibody-agent/docs/model-adapters.md)。

## 配置与文件布局

源码工作区默认结构：

```text
agents/antibody-agent/
  .venv/bin/python
  configs/runtimes.example.json          # 公开配置示例，不代表资源已安装
  .antibody-agent/                       # ANTIBODY_DATA_DIR
    runtime.env                         # 受信任的本地启动配置，Git 忽略
    runtimes.json                       # 各模型实际运行配置
    models/qwen/2fc06364715b967f1860aea9cf38778875588b17/
      config.json
      <完整 tokenizer / 配置文件>
      <全部 safetensors 分片及必要索引>
    runs/                               # 本地证据账本，不属于依赖下载
    property_models/                    # 由真实项目实验训练的属性模型
```

科学后端位置由配置显式指定，不强制使用某个固定目录：

```text
<rfantibody.bin_dir>/{rfdiffusion,proteinmpnn,rf2}
<antifold.checkpoint>
<immunebuilder.weights_dir>/antibody_model_{1,2,3,4}  # VH/VL
<immunebuilder.weights_dir>/nanobody_model_{1,2,3,4}  # VHH
<boltz2.python 所在目录>/boltz
<boltz2.cache>/<后端所需完整缓存资产>
<boltz2.checkpoint>
<桥接命令可执行文件> [<桥接脚本与参数>]
```

所有科学模型配置都要记录实际 `version` 和 `weights_version`。原生适配器使用配置的
Python 或当前解释器，环境中需要能导入 agent 基础依赖及对应科学包。
桥接 `command` 是 argv 数组，不是 shell 命令字符串。资产路径使用绝对路径；只复制并
填写[配置示例](../../agents/antibody-agent/configs/runtimes.example.json)中已安装的条目。

| 环境变量 | 含义 |
| --- | --- |
| `ANTIBODY_PYTHON` | agent 解释器，默认 `<agent>/.venv/bin/python` |
| `ANTIBODY_BOOTSTRAP_PYTHON` | `install.sh` 创建环境时使用的解释器 |
| `ANTIBODY_DATA_DIR` | 资产与运行目录，启动器默认 `<agent>/.antibody-agent` |
| `ANTIBODY_RUNTIMES` | 模型配置，默认 `<data>/runtimes.json` |
| `ANTIBODY_QWEN_DIR` | 已有完整 0.8B 快照，可只读复用 |
| `ANTIBODY_LLM_PYTHON` | 已有兼容主控环境，默认当前 Python |

启动器读取 `<agent>/.antibody-agent/runtime.env`，不读取仓库全局 runtime.env。
复用其他 agent 的 Qwen 快照或环境需要显式设置覆盖变量，不把同级 agent 当成强制依赖。
直接使用 wheel 且没有源码工作区时，默认数据目录回退到 XDG/用户缓存。
`--root` 只覆盖运行日志目录。

## 项目数据与就绪状态

靶标序列/状态、配对 VH/VL 或 VHH、已映射的框架坐标、功能表位、约束和参考面板来源由
项目提供，不是打包的模型资产。完整 IgG、高级格式装配、插入/删除映射及对应验收不能
从“可变区模型依赖已经安装”自动推断出来。

实验学习基线依赖 NumPy 与真实、条件匹配、冻结 train/calibration/test 拆分的测量数据。
没有预置或隐式下载 LaMBO-2 项目属性权重。训练产物、实验记录及硬件相关验证证据保留为
本地私有运行资产。

`doctor` 报告配置可用性（`unavailable` / `configured_unverified`）及主控配置文件是否存在。
它不执行 GPU 推理，不检查每个权重分片，也不生成科学就绪证明。部署后应按各后端输入合同
独立完成实际验证。
