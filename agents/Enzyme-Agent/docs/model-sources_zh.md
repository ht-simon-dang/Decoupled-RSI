# 原始资料与固定版本

[原文](model-sources.md) | [简体中文](model-sources_zh.md)

这是实验性公开副本的依赖清单。下载时会在 `ENZYME_DATA_DIR/models/` 下记录源码版本和哈希。仓库不附带模型文件或历史验证记录；使用前请检查上游许可证及本地 doctor 输出。

| 组件 | 固定 revision | 来源与用途 |
| --- | --- | --- |
| Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | [官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)，与 peptide-agent 共用本地文件；路由规划 |
| LigandMPNN / ProteinMPNN | `26ec57ac976ade5379920dbd43c7f97a91cf82de` | [作者仓库](https://github.com/dauparas/LigandMPNN)，原子环境 / 骨架条件序列设计；代码与所选权重 MIT |
| ESM3 small open | `47f0545b2b6daf26a93439a3cd610f4f7f3d5478` | [权重](https://huggingface.co/biohub/esm3-sm-open-v1)、[官方用法](https://github.com/Biohub/esm/blob/main/_assets/ESM3_README.md)；sequence track 掩码补全，MIT |
| ESMFold2-Fast | `45fe8656f5b3ef493c17fcf9abe9a2968902e712` | [模型卡](https://huggingface.co/biohub/ESMFold2-Fast)、[系列说明](https://huggingface.co/biohub/ESMFold2)；使用 Transformers 5.20.1 内置实现，单链推理，MIT |
| ZymCTRL | `3c532ef172b9cd2e95238baadf5167ebb89fbc32` | [作者模型卡](https://huggingface.co/AI4PD/ZymCTRL)；EC 条件蛋白种子，Apache-2.0 |

LigandMPNN 使用 `ligandmpnn_v_32_010_25.pt`，ProteinMPNN 使用 `proteinmpnn_v_48_020.pt`。固定上游文件不作代码修改；旧 OpenFold 的 `np.int` 仅通过 worker 内兼容别名修复。模型 likelihood 只属生成来源，不进入催化合格判据。

ESM3 库使用兼容 Python 3.11 的 3.2.1.post1。上游最新版对 Python/Transformers 依赖不同，因此本版没有未经验证地更新同环境基础依赖。模型快照来自现有 Biohub 官方仓库；仅 sequence track 的已用接口经过实测，其他 API 不在声明范围内。

ESMFold2-Fast 采用 `infer_protein`，默认 3 loops、50 diffusion steps、1 sample。输出 pLDDT 的本实现尺度为 `[0,1]`；没有误标成 `[0,100]`。输入只有序列，未传设计模板或强制接触条件。没有把单链折叠称为底物复合物、真实修饰、寡聚或催化验证。

DNase 参考： [UniProt P24855](https://www.uniprot.org/uniprotkb/P24855/entry)、[PDB 4AWN](https://www.rcsb.org/structure/4AWN)。UniProt 前体 1–22 为信号肽，成熟链 23–282；PDB 链 A 的 260 aa 与该成熟链一致。原始 UniProt JSON 保留每项证据类型，含基于相似性的注释，不把所有审阅条目解释为逐位实验验证。

疾病模板依据原蓝本引用的 [Fabrazyme 标签](https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/103979s5313lbl.pdf)、[Palynziq 标签](https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/761079s034lbl.pdf)、[Pulmozyme 标签](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=d8c78a7e-ff99-48f3-8952-643ec2ea0f86)。模板中的治疗机制不自动转移为新变体的有效性结论。

未启用的大模型和预测器详见能力登记；这些模型的输入、许可、资源和适用域需在具体接入时重新验证。网络资料支持各组件能力，不支持本系统已验证疾病疗效的说法。

## RFdiffusion3 与 ThermoMPNN

- RFdiffusion3: [官方 Foundry](https://github.com/RosettaCommons/foundry/tree/d078b24d6d0e6f8607e9a0311128f4a897f83953)，commit `d078b24d6d0e6f8607e9a0311128f4a897f83953`，checkpoint `rfd3_foundry_2025_12_01_remapped.ckpt`（2,690,316,669 bytes）。使用 Python 3.12、兼容 CUDA Torch 的隔离 venv，固定 AtomWorks 3.0.1（3.0.4 更改了共价修饰变换参数，与该 Foundry revision 不兼容），固定 `inference_mode=legacy`，默认 200 steps、batch=1；关闭轨迹/编译缓存生成以控制空间。仅接入参考酶受约束原子与序列重设计，未将生成坐标作为独立评价。
- ThermoMPNN: [官方源码](https://github.com/Kuhlman-Lab/ThermoMPNN/tree/370f76ec62bd929f7425e311d8df04a0d094990f)，commit `370f76ec62bd929f7425e311d8df04a0d094990f`，官方 `thermoMPNN_default.pt` 与 `v_48_020.pt`。使用原始 TransferModel、原始特征解析与严格 state_dict 加载；去除 Lightning 的 model. 名称前缀，不改变权重。正 ddG 表示去稳定化；多替换逐个报告，不计算可误导的加和净 ddG。
- 下载会将模型和源码哈希记录在 `${ENZYME_DATA_DIR}/models/<model>/manifest.json` 中。扩展模型就绪还要求 `runtime-validation.json` 中的本地推理证据；公开版既不包含历史记录，也不包含生成这些记录的内部脚本。全新安装后的激活流程仍不完善。可选安装器将本地环境快照写入仓库之外的 `ENZYME_DATA_DIR` 下。
