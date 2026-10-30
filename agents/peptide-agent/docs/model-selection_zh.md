# 模型与算法选择——核查日期：2026-10-18

[English](model-selection.md) | [简体中文](model-selection_zh.md)

“前沿”取决于任务：仅序列的线性结合肽、口袋条件下的全原子协同设计和大环肽，不能共用一个缺乏证据的通用 SOTA 声明。本项目选择可下载的序列与全原子后端，将未发布的研究候选单独列出。

| 模型 | 主要来源 | 决定 |
| --- | --- | --- |
| Qwen3.5-0.8B | [官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 必需的本地规划器；严格验证 brief，不是科学预测器 |
| PepMLM-650M | [作者实现](https://github.com/programmablebio/pepmlm)、[权重](https://huggingface.co/ChatterjeeLab/PepMLM-650M) | 已选择、实现并固定版本；由靶点序列生成线性肽，使用常规 Transformers 运行时 |
| BoltzGen-1 | [官方实现](https://github.com/HannesStark/boltzgen)、[权重](https://huggingface.co/boltzgen/boltzgen-1) | 原环境已安装 0.3.2；全原子线性/环状设计及可选 Boltz2 重折叠 |
| PepGLAD | [官方代码 / 已发布 checkpoint](https://github.com/THUNLP-MT/PepGLAD) | 已有的全原子口袋条件比较方法；未安装；较旧的 torch/scatter 栈需要在同一环境内另行验证兼容性 |
| PepTri，ICLR 2026 | [官方仓库](https://github.com/aigensciences/PepTri) | 近期结构方向的前沿候选；README 描述训练和本地 checkpoint 结构，但核查的发布页中未找到经确认可直接下载的 checkpoint；此处不是可执行后端 |
| PepFGLD，ICML 2026 | [官方仓库](https://github.com/NiuDongjiang/PepFGLD) | 核查时 README 表示代码、数据和预训练模型将发布；不能宣称推理已可用 |
| PepEDiff，2026 | [官方仓库](https://github.com/LabJunBMI/PepEDiff-Seq-Only) | 新的序列生成研究路线；checkpoint/数据通过 SharePoint 链接提供，尚未本地验证；暂缓接入 |

PepMLM 作者模型卡链接了其 Nature Biotechnology 论文。选择它是为了可复现的当代结合肽生成，并不意味着本项目已证明它优于所有 2026 年方法。其似然不是亲和力估计。适配器实现同时预测被 mask 的片段及逐位置 top-k softmax 采样，并将输出限定为 20 种标准氨基酸；这一显式限制记录在来源信息中。

固定模型 revision：

- Qwen/Qwen3.5-0.8B：`2fc06364715b967f1860aea9cf38778875588b17`。
- ChatterjeeLab/PepMLM-650M：`898fca941a9057aebdd1a6164b5ee09a1a71780e`。

Qwen 模型卡声明 Apache-2.0。PepMLM 模型卡元数据标为 MIT，同时在 YAML 元数据中保留额外使用相关字段；分发权重或商业使用生成结果前，请检查上游条款。本仓库不改变或授予模型权利。核查时，公开 HF API 显示两个所选仓库均无需访问审批。

## 计算分析与评估

[Biopython ProtParam](https://biopython.org/docs/latest/api/Bio.SeqUtils.ProtParam.html) 提供已有的分子量、pI、电荷与疏水性计算。蛋白不稳定性指数附有明确适用域限制，不作为肽半衰期预测器宣传。序列唯一性/编辑距离和重原子几何提供透明的基线，不作学习模型能力声明。

[PepBenchmark](https://github.com/ZGCI-AI4S-Pep/PepBenchmark/) 是肽性质评估的公开参考。本项目可执行的 benchmark harness 在用户提供、组间不重叠的划分上提供明确的 Dummy、Logistic 和 RandomForest 基线。这里未下载或评分 PepBenchmark 数据集，也不宣称复现其榜单。

后续结构基准中，PepGLAD 的 PepBench 是独立的结构数据集，与 PepBenchmark 不是同一项目。只有提供天然复合物及明确的链/残基映射后，才应加入基于参考的 DockQ/iRMSD/Fnat。Boltz/AlphaFold 置信度、MD 能量和 Rosetta 界面分数必须保留各自证据类型，不能替代实验 Kd/IC50 或毒性。

[DockQ](https://github.com/wallnerlab/DockQ) 2.1.3 曾进行依赖求解检查，但未安装：其 `numpy<2.0` 要求与本环境的 NumPy 2 分析栈冲突。加入它需要经过测试的统一依赖修订，而不是静默新建另一环境或强行安装不兼容包。

## 结构生成、毒性与亲和力集成

BoltzGen 使用固定权重 `c1be29e1f82ffcc72264f64b993c43fb4e0d17f0` 和推理分子归档 `c3d36fd276e9caf098c75d4113c6d5eb320b1a4c`。0.3.2 版本在同一 Python 运行时中运行。适配器执行全原子序列设计，无独立逆折叠阶段。规范化前会依据上游 design-mask 元数据检查输出链身份。

[ToxinPred3](https://github.com/raghavagps/toxinpred3) 的纯 ML 权重采用 revision `dcdba833cd48cc8a2090ac90f3319fd9bcb87a1c`。其 sklearn 1.2.2 pickle 经校验和验证后转换为 NumPy 树数组，不重训、不改变阈值，从而无需第二个旧版环境。运行时从不加载用户 pickle。特征为 20 个 AAC 加 400 个重叠 DPC，均以百分比表示。0.38 是已发布分类器的判定阈值，不是临床校准值。

[PRODIGY](https://github.com/haddocking/prodigy) 2.4.0 提供经验性的复合物 ΔG/Kd。独立的本地 Ridge/conformal 校准 API 要求实测 pKd，以及组间不重叠的训练/校准/测试记录。二者都不被宣传为对候选完成了实验验证。

