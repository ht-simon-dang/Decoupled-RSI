# 历史任务与结构预览（v0.2.1）

[原文](history-preview.md) | [简体中文](history-preview_zh.md)

顶部“打开过往任务”和底部“任务记录”均可打开已保存的 run。页面自动选择第一条构建体；点击其他行切换证据卡。地址栏包含 `run`、`candidate`、`mode` 和 `structure`，可复制链接、刷新或通过浏览器前进/后退恢复。

## 历史记录与推荐

原问题来自历史列表和详情使用不同的数据边界：历史列表统计所有原始构建体，详情却读取只含 high-potential 的交付接口。因此旧版任务或零推荐任务看起来为空。

新增 `GET /api/runs/{run}/preview` 提供明确的只读证据视图。对照、未评价、旧版未验证、未入选与已推荐分别标记。读取不会写入 journal、改写评价或触发重评估。原始工具输入输出仍留在本地日志，预览只返回构建体和评价。

“High-potential 推荐”继续使用原 `/api/runs/{run}`。JSON、CSV、FASTA、MCP、反馈推荐仍经过当前强制评价和筛选；历史视图中的导出按钮也遵守这一规则。需要让旧版候选进入新推荐时，使用“重新评价”，生成独立运行记录。

## 两类结构

- **RNA 二级结构示意**：读取该构建体历史评价保存的全分子 `mfe_dot_bracket`，校验长度、括号平衡，使用 ViennaRNA 的平面坐标布局。配对、分区和核苷酸位置可检查，视图可在三维视角中旋转。这是拓扑示意，坐标单位任意，不是 RNA 原子坐标或三级结构预测。缺少完整历史结构时显示不可用，不补造坐标。
- **编码蛋白 3D**：从所选构建体 CDS 严格翻译蛋白，使用与 Enzyme Agent 相同的 ESMFold2-Fast 预测方式。显示真实模型输出的 Cα 骨架、逐残基 pLDDT、平均 pLDDT 与 pTM；支持 PDB 下载。它不是 RNA 结构，也不代表翻译效率或实验效力。同义 CDS 编码同一蛋白时共享结果。

视图支持拖动旋转、滚轮/按钮缩放、方向键旋转、`+/-` 缩放、`0` 复位、全屏以及悬停查看位置。新蛋白点击“生成蛋白预测结构”进入后台任务，切换页面不会丢失结果。当前交互预算限制为 16–600 aa，预测与等待 GPU 各最多 10 分钟；超范围、缺模型、失败均明确展示。

## 本地运行与缓存

模型：`biohub/ESMFold2-Fast`，固定 revision `45fe8656f5b3ef493c17fcf9abe9a2968902e712`；seed 42、1 loop、50 sampling steps、单样本，无模板约束。Transformer 推理使用既有 mRNA Python 环境。

默认只读复用 `${ENZYME_DATA_DIR}/models/esmfold2/<revision>`，不复制约 25 GB 权重。`MRNA_ESMFOLD_DIR` 可指向同一固定模型的安装目录，需要匹配的 Enzyme manifest 和完整文件大小检查。所有新结果写入 `${MRNA_DATA_DIR}/structure-previews` 并受现有 50 GB 配额锁约束。

按蛋白序列与完整预测协议生成缓存键；成功结果经 PDB 序列一致性、完整 Cα、有限坐标及逐残基置信度检查后才显示。缓存读取再次检查。失败可重试；缺坐标不会绘制替代蛋白结构。不同进程通过缓存锁去重，默认复用 `${ENZYME_DATA_DIR}/.gpu.lock` 避免与 Enzyme 抢占同一推理锁（可由 `MRNA_STRUCTURE_GPU_LOCK` 指定）。

接口均限定 run 内的构建体成员：

```text
GET  /api/runs/{run}/preview
GET  /api/runs/{run}/preview/candidates/{candidate}/rna
GET  /api/runs/{run}/preview/candidates/{candidate}/protein
POST /api/runs/{run}/preview/candidates/{candidate}/protein
GET  /api/runs/{run}/preview/candidates/{candidate}/protein.pdb
```

GET 不启动预测；POST 无任意命令、路径或模型参数输入，沿用同源写入约束。前端仅使用本地 Canvas，不加载外部 CDN，也不放宽 CSP。


默认数据目录为 `~/.cache/decoupled-rsi/enzyme` 和
`~/.cache/decoupled-rsi/mrna`。本次发布包含预览实现，不包含过往运行记录、截图或验证报告。
