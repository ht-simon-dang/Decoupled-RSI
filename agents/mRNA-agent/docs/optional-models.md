# Optional mRNA backends

[原文](optional-models.md) | [简体中文](optional-models_zh.md)

**Experimental integrations. Downloaded assets do not establish readiness.**

`make optional-models` fetches the sources and weights declared in
`scripts/optional-pins.json` and `scripts/fetch_optional.py`. Assets are stored
under `MRNA_DATA_DIR` (default `~/.cache/decoupled-rsi/mrna`) within the application's
50,000,000,000-byte quota. Upstream component licenses apply separately.

| 模型 | 已接入能力 | 实际边界 |
|---|---|---|
| [codonGPT](https://huggingface.co/naniltx/codonGPT) | 密码子级自回归生成 | 精确同义约束；不伪称完成作者 RL 优化 |
| [UTRGAN](https://github.com/ciceklab/UTRGAN) | WGAN 5′UTR 采样 | 官方 generator prior；不把随机样本称为优化后的表达增益 |
| [mRNAutilus](https://huggingface.co/atombio/mrnautilus) | 全长联合生成 | 官方 150M 权重；8 步 absorbing-mask 采样、同义 CDS 约束；原投影/RoPE + PyTorch SDPA；本适配器未启用 MCTS 奖励搜索 |
| [Saluki](https://github.com/calico/basenji/tree/master/manuscripts/saluki) | 完整转录本稳定性预测 | 官方 Zenodo 权重，人/鼠各 50 模型；合成 IVT 的 splice channel 显式为零；归一化稳定性共识，不是小时 |
| [RNAdegformer](https://github.com/Shujun-He/RNAdegformer-Webapp) | 5 通道逐碱基化学降解预测 | 全部 10 folds；RNAsoft/BLstar 特征；不等于细胞内半衰期 |
| [UTR-LM](https://github.com/a96123155/UTR-LM) | 50 nt 未修饰 RNA 的 MRL | 官方 GSM3130435/egfp_unmod_1 fold0 下游头；保留 reporter/细胞迁移限制 |
| [mRNABERT](https://huggingface.co/YYLY66/mRNABERT) | 完整构建体表征 | UTR 单碱基、CDS 密码子分词，1024 token 上限；没有可用于当前哺乳动物任务的已验证功能头，不输出虚构功能分数 |
| [DegScore](https://github.com/eternagame/DegScore) | 化学降解对照 | 官方 2.1 系数 + ViennaRNA 结构；与 OpenVaccine 证据相关，不算独立预测器票数 |


The RNAdegformer adapter preserves the pinned upstream feature semantics:
its RNAsoft feature path uses an effective temperature of 37 degrees C. The five
output channels describe distinct training assay conditions; feature temperature
is not the same quantity. Predictor outputs retain their original task scales.

ProMORNA and Sanofi mRNA-GPT remain unavailable routes in this implementation;
activating them requires an appropriate upstream integration and accessible
model assets. Similar model names do not establish interchangeability.

Some adapters require a local `model-validation.json` readiness record before
execution. The public snapshot excludes internal validation scripts and prior
records, and the installer does not create a replacement record. These adapters
therefore remain blocked until local readiness verification is implemented.
The runtime checks are retained rather than bypassed. Biological transfer across
proteins, cells, modifications, and delivery settings remains unestablished.
