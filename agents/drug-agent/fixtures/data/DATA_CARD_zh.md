# 数据卡：离线演示 fixtures

[English](DATA_CARD.md) | [简体中文](DATA_CARD_zh.md)

本目录所有文件均由 [`../generate_fixtures.py`](../generate_fixtures.py) 确定性生成，使平台无需访问外部数据即可完成端到端离线运行。

> **来源与可信度。** 化学**结构**采用广为人知的公开药物/类药物分子（SMILES）。所有**生物活性标签、靶点–疾病分数、结构元数据、文献摘要和单细胞计数都是合成的**，由文档所述的潜在规则及固定种子的噪声产生。它们仅用于演示，绝不能视为实测或真实数据。

| 文件 | 类型 | 记录数 | 可信度 | 说明 |
| ---- | ---- | ------- | ----- | ----- |
| `seh_bioactivity.csv` | 生物活性（类似 ChEMBL） | 50 | 合成标签 | `IC50` 单位为 nM，含 `pchembl_value`；基于描述符的潜在构效关系规则 |
| `compound_library.csv` | 化合物（类似 PubChem） | 50 | 公开结构 | 候选库（公开 SMILES） |
| `targets.json` | 靶点–疾病（类似 Open Targets） | 1 个靶点 / 3 个关联 | 合成分数 | sEH / EPHX2（UniProt P34913） |
| `structures.json` | 结构元数据（类似 RCSB PDB） | 2 | 合成元数据 | 引用 PDB ID，不含坐标 |
| `literature.jsonl` | 文献 | 3 | 合成；其中 1 条为对抗样本 | 包含一条供安全测试使用的提示注入摘要 |
| `sc_perturbation.npz` | 单细胞 | 120 个细胞 × 24 个基因 | 合成 | 来自 4 位供者的疾病/健康数据；按供者划分 |

- **演示靶点：** 可溶性环氧化物水解酶（sEH / EPHX2），一种研究充分、非病原体来源、适合小分子干预的酶。选择它不意味着提出新的发现或临床主张。
- **许可证：** 结构为公共领域标识符；合成标签按仓库的 Apache-2.0 许可证发布。
- **快照 ID / 哈希：** 每个连接器对其记录计算内容哈希（`DataSnapshot.snapshot_id`），并记录到每次运行的 `manifest.json` 中。
- **重新生成：** `make fixtures`（确定性；给定种子时字节稳定）。

