# Enzyme 依赖

[English](README.md) | [简体中文](README_zh.md)

本清单对应实际 Enzyme 模型注册表和下载器，涵盖运行环境、源码 checkout、模型权重、参考数据及仍需手工集成的路线。下载产物不在此重新分发。准确版本、文件契约、命令、来源和剩余要求见 [dependencies.json](dependencies.json)。

```bash
bash third_party/enzyme/install.sh --help
bash third_party/enzyme/install.sh plan esm3
bash third_party/enzyme/install.sh check rfdiffusion3
# 显式安装操作；可能使用网络：
bash third_party/enzyme/install.sh install peptide-runtime
bash third_party/enzyme/install.sh install runtime
bash third_party/enzyme/install.sh install qwen
bash third_party/enzyme/install.sh install ligandmpnn
bash third_party/enzyme/install.sh install esm3
bash third_party/enzyme/install.sh install esmfold2
bash third_party/enzyme/install.sh install zymctrl
bash third_party/enzyme/install.sh install dnase-reference
```

`core` 将运行环境、全部核心模型和 DNase 参考作为一个显式阶段安装。`ligandmpnn` 与 `proteinmpnn` 共用上游下载器，会同时获取两个 checkpoint。单模型命令使用现有下载器；所有安装路径均作为 argv 元素传递，而不是拼接成由 shell 展开的源码字符串。

扩展模型：

```bash
bash third_party/enzyme/install.sh plan foundry
# 设置为已有的 Python 3.12 解释器，需带兼容 CUDA Torch：
export ENZYME_FOUNDRY_BASE=/absolute/path/to/python
bash third_party/enzyme/install.sh install foundry
```

Foundry 阶段下载 RFdiffusion3 和 ThermoMPNN 的源码/权重，并构建独立运行环境。`install rfdiffusion3` 或 `install thermompnn` 只下载所选源码/权重包，均不会生成本地就绪证据。在所需运行环境和匹配的 `runtime-validation.json` 证据存在之前，检查仍报告缺失；公开副本不包含内部验证脚本和历史记录。

默认结构（`ENZYME_DATA_DIR`，默认为 `~/.cache/decoupled-rsi/enzyme`）：

```text
<ENZYME_DATA_DIR>/
  env/bin/python
  foundry-env/bin/{python,rfd3}
  models/
    qwen/<revision>/                    # 共享 Qwen 模型的后备位置
    esm3/<revision>/{config.json,manifest.json,data/weights/*.pth}
    esmfold2/<revision>/{config.json,manifest.json,*.safetensors}
    zymctrl/<revision>/{config.json,manifest.json,pytorch_model.bin}
    ligandmpnn/
      source/{run.py,model_utils.py,data_utils.py,openfold/}
      {ligandmpnn_v_32_010_25.pt,proteinmpnn_v_48_020.pt,manifest.json}
    rfdiffusion3/{source/,rfd3_latest.ckpt,manifest.json,runtime-validation.json}
    thermompnn/
      source/{transfer_model.py,models/thermoMPNN_default.pt,
              vanilla_model_weights/v_48_020.pt}
      {manifest.json,runtime-validation.json}
  references/cf_dnase_reference.json
agents/Enzyme-Agent/examples/dnase_{reference,design}.json
                                         # 本地生成，已被 Git 忽略
```

`PEPTIDE_PYTHON` 选择基础解释器，`ENZYME_PYTHON` 选择 Enzyme 解释器。`ENZYME_QWEN_DIR` 覆盖共享 Qwen 快照路径。未显式覆盖时，适配器使用 `PEPTIDE_MODEL_DIR` 或同级 Peptide 模型位置，再回退到自己的数据目录。

RFdiffusion2、PLACER、CatPred、CataPro、EZSpecificity、NetSolP、NetMHCIIpan、ProGen3 和 GENzyme 被声明为需手工处理/不完整的路线。其 `plan` 包含上游获取链接，`check` 和 `install` 返回缺失。激活前需获得受许可访问权限、确认适用输入域，并提供已集成且验证过的适配器。不会生成占位文件或就绪标志。下载组件保留各自许可证，包括独立的模型条款和非商业条款。

`plan` 和 `check` 仅使用标准库检查文件，不会调用模型、下载器、包安装器或服务。结果为 `missing`（退出码 2）或 `present_unvalidated`（退出码 0）；文件存在不能认证可执行性或科学就绪状态。安装可能完成资源获取，但因就绪证据缺失而返回 2。`install all` 会被明确拒绝。

