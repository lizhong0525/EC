# MentalHealthBench 外部回应评测协议与实测记录

日期：2026-10-25。当前执行版本：`mentalhealthbench-external-v1.1.0`。
状态：**真实两例工程冒烟已完成；全量公开数据未运行；不是论文官方结果复现。**

## 1. 与 EcoTriage 的边界

本轨道回答“已有对话前文后的下一条回复，是否满足源数据集专家撰写的评分准则”。
它不回答 EcoTriage Track A 的“当前节点选择何种支持或转介动作”，也不提供行动、
时窗、转介对象或证据引用 Gold。源 rubric 不会写入专家库，不改变 Gold 0、
 reviewed Silver 0、Unresolved 280 的项目状态。

公开源数据可重建，不是独立保密盲测。材料以英文为主，只有 1 条中文记录；
不能外推为中文青少年、诊断、治疗效果或真实部署安全性。

## 2. 固定来源与评分

- 数据归档 SHA-256：
  `a3e94c7786bd514952c542d6a5694bb24055f136b417f15c5c6ab942976d7474`。
- 数据：1,215 任务，5,262 条源专家撰写准则。快照逐文件摘要与数量均经核验。
- 官方论文：本地 `artifacts/external/mentalhealthbench-paper/`，PDF SHA-256
  `f4ce5da8db2c7c3ec3512363f57b82a07367b30eb1b5ff20f0349f09bfcabd75`。
- 原方法：每任务四个独立采样回复；GPT-5.6 Sol、高推理强度、每准则单独判分。
- 原公式：满足准则的有符号点数之和 / 该任务全部正分；任务回复分数在 0 截断。
  先对同一任务的采样求均值，再对任务等权宏平均。
- 另行报告正向贡献和归一化惩罚负担；行为轴也除以该任务总正分，而不是轴内正分。

实现中的 `GRADER_PROMPT` 是附录 B.2 的逐字文本，并采用显式占位符替换，
不会把示例 JSON 的花括号误当格式化字段。真实 PDF 附录的评分主体和公式已再次阅读。

## 3. 可复现工作流

在 `ArtificalPsychologist` 根目录运行。所有输出必须选择新路径。

```powershell
$py = ".research-venv\Scripts\python.exe"
$snapshot = "artifacts/external/mentalhealthbench/a3e94c7786bd514952c542d6a5694bb24055f136b417f15c5c6ab942976d7474"
$paper = "artifacts/external/mentalhealthbench-paper/f4ce5da8db2c7c3ec3512363f57b82a07367b30eb1b5ff20f0349f09bfcabd75"

# 已有论文快照可复用；只有显式执行下载命令才访问公开 CDN。
& $py -m ecocompanion.research mhb freeze-paper --output <新的论文目录>

# 当前已冻结且可验证的公开包；prepare 本身不调用模型。
& $py -m ecocompanion.research mhb prepare --snapshot $snapshot --paper $paper `
  --output <新的公开包目录> --scope public
& $py -m ecocompanion.research mhb verify --packet <公开包目录>

# 先显式启动既有本地运行时；服务仅监听回环地址。
& ./scripts/start_local_judge.ps1 -Output <新的运行时日志目录>
& $py -m ecocompanion.research mhb bind-local --packet <任务包> `
  --model qwen2.5:7b-instruct-q4_K_M --output <新的绑定.json>
& $py -m ecocompanion.research mhb run --packet <任务包> `
  --config <绑定.json> --output <新的运行目录>
& $py -m ecocompanion.research mhb report --packet <任务包> `
  --config <绑定.json> --run <运行目录> --output <新的独立重算.json>
```

`tasks.jsonl` 只含无语义 ID 和原始可见消息。`private/references.jsonl` 保存准则，
`private/mapping.jsonl` 保存源 ID、索引和分组元数据。生成模型不接收准则、
源 ID、风险类别、用户画像、数据划分、canary 或项目动作标签。判分器按任务需要
接收可见对话、待评回复和当前一条准则，这是源方法的必要输入。

加载报告时从源快照重新构造所有任务和私有映射，逐项核对完整请求、模型身份、
摘要、token、生成记录来源和判分结果。任何变动要求新协议和新目录。
论文按其字节身份复用，不与评测代码版本不必要地绑定；历史 paper manifest 不覆盖。

## 4. 采样、预算与失败语义

- `scope=smoke`：在任何模型输出前按可见字符数、源 ID 选择最短两例，仅用于工程链路。
  不是随机代表样本，不可用于总体性能估计。每例一次生成，temperature 0。
- `scope=public`：固定全量 1,215 例，每例四次预注册种子的本地采样，temperature 1。
  这是本地诊断采样，不声称与官方采样分布相同。
- 当前公开包最多 4,860 次生成和 21,048 次判分。准备任务包不会自动执行它们。
- 生成 `num_predict=2048`，判分 1200，上下文 16384；判分 temperature 0。
- v1.1 单次读取超时 1200 秒，禁止模型回退、隐式重试或修改已完成记录。
- 显式 `--resume` 仅允许执行中断前尚未发起的调用，不重试失败；已封存完整运行
  重复 resume 只做重放验证。已封存的失败运行不能被改名成成功运行。
- 格式错误、模型身份不符、token 账目缺失、超时和截断为不完整，不静默当准则未满足。
- 真正空回复按源协议直接记零，不再调用 Judge；这类豁免在报告中单独计数。
- 主分只有在全部任务完整时给出。完整子集统计明确标为观察性描述，不替代全量主分。
- 以源任务为聚类单位；少于 30 个完整任务不出 bootstrap 区间。任务不等于真实独立人物。

## 5. 本轮实际执行

### v1.0 失败留痕

首次真实冒烟在第一条 1609-token 对话上运行 301.47 秒后触发 300 秒读取超时。
记录 1 次真实调用、0 次判分、0/2 完成，失败 token 未知。旧目录和报告未修改。
这不是模型质量分，也没有拿超时回复参与评分。

### v1.1 完成的工程冒烟

模型为既有 `qwen2.5:7b-instruct-q4_K_M`，模型摘要：
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`。
运行时为 Ollama 0.40.8，仅本机回环，未下载新模型、未调用远程模型。

- 产物：`artifacts/research/mhb-external-v1/smoke-run-v1.1/`。
- 输入：`smoke-packet-v1.1-r2/`；最短两例、6 条 rubric、一次生成/例。
- 结果：2/2 完整，2 次生成 + 6 次判分 = **8 次真实调用**，零重试。
- 成功生成输入/输出 token：69 / 442；成功判分输入/输出 token：7740 / 292。
- 生成延迟分别约 3.96、314.15 秒；判分约 1.51–2.96 秒。
- 本地未校准诊断分：0.1875，正向贡献 0.1875、惩罚负担 0。
  此数仅说明两条已执行记录如何按源公式得分，不是总体模型水平。
- 完整任务仅 2 个，置信区间不可用；没有进行人工逐条质量校准。
- `smoke-independent-rescore-v1.1-final.json` 与原报告完整相等，重放验证通过。

### 全量状态

`public-packet-v1.1/` 已验证 1,215 任务、5,262 准则，状态 NOT_EXECUTED。
没有全量真实模型分，没有论文官方 GPT-5.6 Sol 结果，也没有项目行动 Gold。
旧的未执行任务包和失败运行只作为版本历史保留；当前复现应使用 v1.1 路径。

## 6. 验证

- 新增 `backend/tests/test_research_mhb.py`：12 项通过。
- 首轮相关回归：58 项通过。
- 最终全量后端：601 通过、1 跳过（`tests-full-final.xml`）；跳过原因为 Windows
  符号链接权限，POSIX 权限隔离仍需相应环境验证。
- 前端：3 项通过，生产构建成功。
- Ruff：生产源码、脚本及新增测试通过。
- Mypy：100 个生产源码文件通过。
- 项目秘密扫描通过；扫描器补充排除 `.research-venv` 第三方依赖目录，
  并用新增测试覆盖。没有读取或输出任何密钥。
- 测试使用本地模拟 HTTP 服务验证泄漏、严格输出、空回复、失败、摘要漂移、
  篡改重放、显式续跑和 CLI；真实冒烟与模拟测试分开记录。

## 7. 尚需外部条件

1. 完整公开包的 25,908 次最大计划调用需要单独时间预算，不能由两例外推耗时。
2. 本地 Qwen Judge 尚未对独立专家判分校准，不能称为官方 Judge 或验证合格的评分器。
3. 若需论文官方可比结果，需另行获授权的官方模型/配置与独立成本审批，本轮未调用。
4. EcoTriage 行动 Gold 仍需真实合格专家、独立数据管理员及独立盲测材料。
5. 中文迁移、人物级独立性、真实世界结局、伦理与未成年人部署仍未验证。
6. 原始材料只供本地研究；本流程不提供公开再分发功能。
