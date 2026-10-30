# 可选 mRNA 后端

[原文](optional-models.md) | [简体中文](optional-models_zh.md)

**实验性集成。下载资源不代表后端已经就绪。**

`make optional-models` 获取 `scripts/optional-pins.json` 和 `scripts/fetch_optional.py` 声明的源码与权重。资源存放在 `MRNA_DATA_DIR` 下（默认为 `~/.cache/decoupled-rsi/mrna`），受应用 50,000,000,000 字节配额约束。各上游组件许可证独立适用。

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

RNAdegformer 适配器保留固定上游版本的特征语义：RNAsoft 特征路径使用的有效温度为 37°C。五个输出通道对应不同训练测定条件，特征温度不是同一个量。预测器输出保留各自原始任务尺度。

ProMORNA 和 Sanofi mRNA-GPT 在本实现中仍为不可用路线；激活需要合适的上游集成及可访问模型资源。模型名称相似不代表可以互换。

部分适配器执行前要求本地 `model-validation.json` 就绪记录。公开副本不包含内部验证脚本和过往记录，安装器也不会创建替代记录。因此，在实现本地就绪验证前，这些适配器会保持阻塞。运行时检查保留，不予绕过。跨蛋白、细胞、修饰和递送条件的生物学迁移能力仍未确立。

