# mRNA 依赖

[English](README.md) | [简体中文](README_zh.md)

源码注册表声明核心后端、可选后端、内置操作和不可用研究路线。[dependencies.json](dependencies.json) 为每一项记录资源获取阶段和适配器实际使用的文件路径。仓库不附带上游源码、权重、数据集或历史就绪记录。

```bash
bash third_party/mrna/install.sh --help
bash third_party/mrna/install.sh plan gemorna
bash third_party/mrna/install.sh check optional
# 显式操作；可能安装包和下载资源：
bash third_party/mrna/install.sh install peptide-runtime
bash third_party/mrna/install.sh install runtime
bash third_party/mrna/install.sh install qwen
bash third_party/mrna/install.sh install core
bash third_party/mrna/install.sh install optional
```

`core` 是一个显式分组阶段，包含 GEMORNA、LinearDesign、RiboNN、Smart5UTR、Optimus5、PARADE 和参考记录。它创建 Python 3.10 兼容环境，获取固定源码，构建 LinearDesign 并下载声明的权重。选择其中单个模型 ID 安装时，会解析为整个 core 阶段；`plan MODEL` 会在安装前展示这一范围。

`optional` 包含 codonGPT、UTRGAN、mRNAutilus、Saluki、RNAdegformer、UTR-LM、mRNABERT 和 DegScore，还会获取固定源码支持库，并创建 Transformers 4 兼容环境。单个可选模型 ID 会解析为这一完整阶段。安装器不创建推理就绪记录：在所需本地证据存在之前，可选模型的就绪检查仍会报告缺失。仅下载不能激活它们。

`references` 只获取公开的 NCBI/UniProt 参考记录。`esmfold2-preview` 委托 Enzyme 安装器获取资源（需先安装其运行环境）。`synonymous`、`vienna` 和 `joint` 映射到运行包；`joint` 仍要求调用方提供带来源记录的提案。`promorna` 和 `mrna_gpt` 没有已集成的可执行获取路径，会返回缺失；清单列明上游研究链接和后续所需工作。

默认结构（`MRNA_DATA_DIR`，默认为 `~/.cache/decoupled-rsi/mrna`）：

```text
<MRNA_DATA_DIR>/
  venv/bin/python                       # 现代科学运行环境
  envs/gemorna/bin/python                # Python 3.10 兼容运行环境
  envs/transformers4/bin/python          # Transformers 4 兼容运行环境
  external/
    GEMORNA/{src/shared/*.so, vocab/*.pkl, checkpoints/*.pt}
    LinearDesign/{bin/LinearDesign_2D, codon_usage_freq_table_human.csv}
    RiboNN/src/
    Smart5UTR/models/{Smart5UTR, Optimus5-Prime}/*.h5
    parade/predictor/{model, regression_multiple/saved_models}
    basenji/   UTRGAN/   UTR-LM/   DegScore/   arnie/
    RNAdegformer-Webapp/{RNA_Inference/best_weights, MultiRNAFold}
  models/
    qwen/<revision>/                     # 缺少共享 Qwen 时的后备位置
    ribonn/{human,mouse}/{runs.csv,<run>/state_dict.pth}
    codongpt/   mrnabert/   mrnautilus/
    saluki/f*_c*/train/{params.json,model0_best.h5,model1_best.h5}
  datasets/{references.json, NM_000518.5.gb, P42212.fasta, P08659.fasta}
  assets.json                           # 获取脚本生成的哈希/版本
  model-validation.json                 # 必需的本地证据；不随仓库提供
```

`PEPTIDE_PYTHON` 选择基础运行环境。`MRNA_PYTHON` 选择这些安装封装使用的运行环境；应用本身仍使用启动它的解释器。共享 Qwen 依次查找 `MRNA_QWEN_DIR`、`PEPTIDE_MODEL_DIR` / 同级 Peptide 模型目录，最后使用本地后备路径。蛋白预览使用 `MRNA_ESMFOLD_DIR` 或 Enzyme 模型目录。

所有源码和模型条款均由各自上游决定，包括源码注册表记录的仅限研究和非商业条款。获取/使用前请检查链接中的条款。安装器不能授予受许可资源的访问权。

`plan` / `check` 是离线标准库操作，不会导入模型或创建输出目录。检查结果是 `missing`（退出码 2）或 `present_unvalidated`（退出码 0），不表示推理或生物学验证已经通过。资源成功下载后，如果所需就绪证据仍缺失，安装也可能返回 2。不存在隐式的 `install all`。

