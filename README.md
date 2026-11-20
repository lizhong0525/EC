# EcoCompanion

EcoCompanion 是一个中文、文本优先的合成研究平台，包含：

- `EcoTriage-Bench`：纵向青少年心理风险识别与分级转介 Benchmark；
- `ECO-SAFE`：联合考虑风险、证据、不确定性和关系边界的决策策略；
- 陪伴 Agent 演示：持续人格、可管理记忆、主动关怀限制、流式 Web 对话与模拟交接。

系统只支持合成环境和成年研究测试，不是诊断、治疗或实时紧急响应服务，也不能直接
部署给真实未成年人。完整范围与验收标准见
[开发计划](./EcoCompanion_完整Benchmark与陪伴Agent开发计划.md)。
新增公开来源材料仅在隔离的离线研究路径处理，不进入陪伴服务或真人部署。

当前接续状态见 [2026-10-25 状态页](docs/PROJECT_STATUS_20261025.md)；
Web 同轮安全确认、风险后陪伴和语言蒸馏边界见
[Web 陪伴策略 v2](docs/WEB_SUPPORTIVE_DIALOGUE_V2.md)。
表达蒸馏的冻结任务、指标、来源隔离与人工盲评设计见
[评估规范](docs/RENDERER_EVALUATION_V1.md)；目前只锁定设计，
独立评估病例和真人标签仍为 0，不能宣称已经完成独立验证。

## 2026-10-20 可信研究入口

新增独立版本 **`ecotriage-v2.0.0`**，第一阶段只评估 Track A 的当前证据下下一步动作。
旧 v1 数据和历史报告保留不改；原 Hidden Test 可由公开生成器/种子重建，
只能作为公开开发或回归材料，不能作为保密盲测。Track B/C 暂为探索性/测试夹具。

```powershell
python -m ecocompanion.research --help
python -m ecocompanion.research serve --packet artifacts/research/ecotriage-v2-pilot-r2 --port 8765
```

20 例待标任务包与角色隔离表单、逐节点锁定、双标分歧/裁决、Gold 门禁、
两个参考标准分栏评分及离线规则实验入口见
[操作手册](docs/ANNOTATION_OPERATIONS_V2.md)。
研究边界和统计约定见 [v2 协议](docs/RESEARCH_PROTOCOL_V2.md)，
实际缺陷与保密威胁模型见 [可信度审计](docs/TRUST_AUDIT_V2.md)，
实际执行结果见 [验证记录](docs/VERIFICATION_V2.md)。
尚无真实专家 Gold、独立盲测验证或真实模型主实验；不得把工程测试当作这些结果。

### P0/P1 测量修订

新增独立测量层 `ecotriage-measurement-v2.1.0`：
`validity-audit` 执行评分器投机挑战，`diagnostic-plan → diagnostic-run → diagnostic-report`
冻结并重放六条件单因素规则实验。旧数据和报告不覆盖，输入改变后的标签适用范围
单独标注，工程夹具从不生成专家 Gold。
方法映射、命令和限制见 [P0/P1 协议](docs/P0_P1_PROTOCOL.md)。

### CRADLE-Dialogue 本地外部检查

新增 `cradle-download → cradle-audit → cradle-plan → cradle-run → cradle-report`，
固定原始提交并隔离原生危机事件标签与项目行动评分。仅运行本地规则、零模型调用；
没有从外部标签生成项目 Gold。数据许可证尚待确认，不再分发。
命令、实际数据异常与结论边界见 [本地接入协议](docs/CRADLE_LOCAL_PROTOCOL.md)。

### 中文外部数据入口

`python -m ecocompanion.research zh --help` 提供 PsyCrisis 与 CPsyCoun 的固定快照、
审计、离线规则执行、全量重放及待审任务包。PsyCrisis原作者评分与项目行动Gold
分开；CPsyCoun只建开发试标，不假定年龄/人物独立性，不伪造专家或真实模型结果。
见 [协议](docs/CHINESE_EXTERNAL_PROTOCOL_V1.md) 与
[试标说明](docs/CHINESE_EXTERNAL_ANNOTATION.md)。所有历史数据与报告保留。
本轮实际执行及未完成事项见 [结果记录](docs/CHINESE_EXTERNAL_RESULTS_20261021.md)。
后续技术复核与40条AI辅助初评见 [复核记录](docs/CHINESE_REVIEW_AND_PILOT_20261021.md)；
正式专家/隐私/授权确认仍待完成，不能将AI初评当Gold或模型安全证明。

### 无新增专家的外部参考路线

新增隔离脚本 `scripts.reference_sources`、`scripts.external_reference`、
`scripts.local_reference_judge`。复用 PsyCrisis 原回复人工评分、VERA-MH 原整段对话
临床评分，并保持 CRADLE 原生事件与项目行动规范分离。仅连接本机的 Judge 入口
要求先校准、后公开留出、再按门禁开放项目回复；不调用付费模型、不伪造 Gold。
固定版本、统计、执行命令和限制见
[外部参考协议](docs/EXTERNAL_REFERENCE_PROTOCOL_V1.md)。
实际执行、原始评分一致性及未完成状态见
[本轮结果](docs/EXTERNAL_REFERENCE_RESULTS_20261021.md)。
新增脚本不修改已冻结 backend 源码，旧协议与历史实验结果保留。

### 当前架构与 P0 启动检查

主系统实际是规则评估、ECO-SAFE 动作决策、模板或可选 LLM 表达的混合架构；
外部 Judge 与对话模型分开。新增 `python -m scripts.p0_validation doctor --output <新目录>`
检查冻结任务包、本机服务和逐维样本支持量；`run --model <精确已安装名称>` 按顺序
冻结并执行校准、公开留出，不自动下载模型或绕过门禁。
见 [架构与命令](docs/CURRENT_ARCHITECTURE_P0.md)、
[本轮实际结果](docs/P0_READINESS_RESULTS_20261021.md)。

### 2026-10-22 外部参考补全

新增 VERA 评分表分支审计、严格答案路径重放、逐维原人工一致性与人物模板
聚类敏感性分析，以及 PsyCrisis 参考支持量诊断。两项均为离线审计，
不产生专家 Gold，不自动运行模型，也不解除既有门禁。
运行方法见 [补全协议](docs/COMPLETION_PROTOCOL_V1.md)，
实际测试、复跑和阻断项见 [本轮验收](docs/COMPLETION_RESULTS_20261022.md)。

### 本地模型实跑

经用户确认后，已安装固定版本 Ollama 与 Qwen2.5 7B Q4_K_M，
只开放本机回环服务，不使用付费接口或云模型。
安装摘要、固定缓存配置、额外加载检查及真实校准执行状态见
[本地运行记录](docs/LOCAL_JUDGE_RUN_20261022.md)。
Judge 实跑不是目标 Agent 主实验，也不会生成行动规范 Gold。

### 2026-10-23 Gold 准备与指南参考层

新增 `python -m ecocompanion.research references --help`。
指南原文快照、条款适用性、局部映射、第二人复核、不可覆盖纠错与 Silver
单独评分均与现有专家库隔离。没有真实专家时 Gold 保持 0；没有实际项目复核时
Silver 也不自动生成。指南下载失败、未知条件与分歧保留为 Unresolved。
操作命令、20 例新包和人工待办见
[指南参考与 Gold 操作手册](docs/GUIDELINE_REFERENCE_OPERATIONS_V1.md)。
既有专家手册、历史实验报告和 Judge 实验均不覆盖。
实际产物、11 项新增测试、全量 560 通过/2 失败及 Windows 权限限制见
[本轮验收](docs/GOLD_READINESS_RESULTS_20261023.md)；
参考分级与统计边界见 [新版本协议](docs/GUIDELINE_REFERENCE_PROTOCOL_V1.md)。

## 当前实现

- Track A：144 个规范病例、288 条冻结表层轨迹和 72/24/24/24 固定划分；
- Track B：24 个分层病例、每病例 5 个固定种子；
- Track C：12 个关系安全场景、4 种配置、每配置 5 个固定种子；
- 六类公共 Schema、确定性 `WorldEngine`、ECO-SAFE、安全审计、记忆、主动关怀和
  幂等模拟交接；
- 可恢复的 `run → resume → report` 实验链路；
- 模型发现、20 条资格测试、模型身份锁定、缓存、成本账本和脱敏 Trace；
- FastAPI、SQLite、SSE、React/Vite 研究界面及 Benchmark 报告面板；
- 带校验和的研究预发布包。

尚未完成的外部验收包括真实 LiteLLM 模型资格快照与主实验、专家双标和裁决、成年
参与者伦理流程，以及安装 Docker 的机器上的真实镜像冷启动测试。不得把本地
Mock/规则实验解释为这些工作的替代品。

## 环境

- Python 3.11；
- Node.js 22；
- npm 10；
- 可选 Docker Compose；
- 真实模型实验需要由运行环境提供 `LITELLM_API_KEY`。

运行时和开发依赖分别固定在 `requirements.lock` 与
`requirements-dev.lock`，前端依赖固定在 `web/package-lock.json`。

## 安装

干净环境：

```bash
make install
npm --prefix web ci
```

`make install` 会快速拒绝覆盖已有 `.venv`。已有隔离环境需要同步时使用：

```bash
make sync
```

重新解析锁文件需要预先安装 `uv`：

```bash
make lock
```

## 质量门禁

```bash
make check
```

该命令依次覆盖 Ruff、mypy、后端测试、核心覆盖率门槛、Schema 导出、秘密扫描、
Web 测试和生产构建。也可以单独执行：

```bash
make lint
make typecheck
make test
make coverage-core
make schemas
make secret-scan
make web-test
make web-build
```

CI 只使用 Mock LLM，不会自动调用付费中转。

## Benchmark

验证病例、固定划分、精确泄漏与字符 3-gram 近重复：

```bash
eco benchmark validate --output artifacts/validation/report.json
```

运行、恢复和生成报告：

```bash
eco benchmark run \
  --track A \
  --split public_test \
  --policy random,always-listen,always-escalate,keyword-triage,history-aware-rule \
  --surface all \
  --repeats 3 \
  --run-dir artifacts/runs/track-a-local

eco benchmark resume artifacts/runs/track-a-local
eco report build artifacts/runs/track-a-local
```

Track B 和 Track C 使用相同入口：

```bash
eco benchmark run --track B --policy history-aware-rule \
  --run-dir artifacts/runs/track-b-local

eco benchmark run --track C --configuration all \
  --run-dir artifacts/runs/track-c-local
```

默认的本地模式用于确定性回归。模型支持的策略必须显式提供资格快照和模型 ID，禁止
在正式 Benchmark 中静默切换模型：

```bash
eco benchmark run \
  --track A \
  --policy current-message-llm,full-history-llm,eco-safe \
  --model-snapshot artifacts/models/model_snapshot.json \
  --agent-model QUALIFIED_MODEL_ID \
  --run-dir artifacts/runs/track-a-api
```

研究预发布包：

```bash
eco benchmark package \
  --output artifacts/release/ecotriage-v1-preview
```

公开包只包含 Train/Dev 标签和 Public Test 输入；Hidden Test 以及
Public/Hidden 参考标签文件不会进入发布目录，但源码可重建它们，不能据此声称保密。
包还会附带评测/统计源文件、规则基线、
Prompt 身份、Docker 与锁文件、逐文件校验和，以及在没有真实模型时的
`models/MODEL_SNAPSHOT_PENDING.md` 占位说明。生成后可先运行：

```bash
python3 artifacts/release/ecotriage-v1-preview/scripts/verify_release.py
```

真实模型快照和运行 Trace 只有在脱敏、审核后才应显式加入发布包；占位文件不构成
模型能力或临床安全证据。

## 模型发现与资格测试

配置只允许放在进程环境或被忽略的 `.env.local`：

```bash
cp .env.example .env.local
```

不要把密钥写入命令历史、源码、文档、镜像或 Trace。配置完成后：

```bash
eco models discover
eco models qualify
```

资格测试覆盖中文、多轮和 system role、真实 SSE 流、JSON/JSON Schema、20 条心理
安全微测验、延迟与成本。生成的模型快照是正式实验的一部分；模型发生变化时必须创建
新的实验版本。

## Web 演示

```bash
eco web serve
npm --prefix web run dev
```

Web 支持会话恢复、SSE 回复、普通记忆查看/纠正/删除、研究 Trace 和 Benchmark
报告查看。Benchmark 页面可直接新建本地规则测试、查看进度并下载报告，
无需先运行命令行；见 [网页报告操作](docs/WEB_BENCHMARK_OPERATIONS_20261026.md)。
未配置模型时对话只使用本地审核模板安全降级。

Docker 配置及尚待完成的真实容器验收见
[docker/README.md](./docker/README.md)。

## 外部合成日志

外部模拟器只能通过
[可观察日志契约](./docs/EXTERNAL_LOG_IMPORT.md)导入：

```bash
eco benchmark import-log synthetic_events.jsonl \
  --output artifacts/imports/expert_relabel_drafts.jsonl
```

输出固定标记为待专家重标草稿，不生成隐藏风险状态、临床标签、动作标签或干预窗口，
也不能直接并入正式 Benchmark。

## 研究材料

- [Benchmark Card](./docs/BENCHMARK_CARD.md)
- [Datasheet](./docs/DATASHEET.md)
- [专家标注手册](./docs/ANNOTATION_GUIDELINES.md)
- [发布文档索引](./docs/README.md)
- [版本变更记录](./CHANGELOG.md)

论文只能声明“合成环境中的风险识别与转介性能”，不得声明临床诊断、治疗效果或真实
未成年人部署安全性。
