# Peptide 依赖

[English](README.md) | [简体中文](README_zh.md)

本目录包含依赖声明和安装入口，不包含第三方源码、模型权重、参考复合物或过往运行记录。可执行适配器的契约位于 `agents/peptide-agent/src/peptide_agent`。

在仓库根目录执行：

```bash
bash third_party/peptide/install.sh --help
bash third_party/peptide/install.sh plan
bash third_party/peptide/install.sh check qwen
# 显式安装操作；可能使用网络：
bash third_party/peptide/install.sh install runtime
bash third_party/peptide/install.sh install qwen
bash third_party/peptide/install.sh install pepmlm
bash third_party/peptide/install.sh install boltzgen
bash third_party/peptide/install.sh install toxinpred3
```

`runtime` 委托项目安装器执行。模型组件调用项目实际的下载/导出命令，保留固定源码版本与校验和检查。`prodigy` 和 `sequence-chemistry` 映射到运行依赖阶段。`calibrated-affinity` 需要用户测量和本地校准；`research-backends` 尚无集成的自动安装流程，会返回缺失。

默认结构（可使用绝对目录覆盖 `PEPTIDE_MODEL_DIR`）：

```text
agents/peptide-agent/
  .venv/bin/python
  .peptide-agent/models/
    qwen/<revision>/{config.json, tokenizer files, weight shards}
    pepmlm/<revision>/{config.json, tokenizer files, weights}
    boltzgen/
      models--boltzgen--boltzgen-1/snapshots/<revision>/*.ckpt
      datasets--boltzgen--inference-data/snapshots/<revision>/mols.zip
    toxinpred3/{model.pkl, trees.npz, manifest.json}
    calibrations/*.json                  # 由操作者提供或拟合
```

`PEPTIDE_PYTHON` 选择用于下载组件的现有运行环境；运行环境安装器还接受 `PEPTIDE_BOOTSTRAP_PYTHON` 和 `PEPTIDE_ENV_DIR`。同级 mRNA 和 Enzyme Agent 也支持通过 `PEPTIDE_MODEL_DIR` 共享 Qwen 资源。准确的版本、URL、路径探测、安装命令和限制见 [dependencies.json](dependencies.json)。下载的代码、模型和数据保留各自上游条款，本项目许可证不会授予这些权利。

`plan` 和 `check` 仅使用标准库和本地文件系统，不会导入科学包、下载、安装或执行推理。资源缺失或手工要求尚未解决时，`check` 返回 2；状态为 `present_unvalidated` 时返回 0，绝不会宣称后端已达到科学就绪状态。不带参数的 `install` 和 `install all` 均会拒绝执行，请显式选择组件。

