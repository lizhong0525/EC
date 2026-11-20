# EcoCompanion 发布文档

## v2 本地研究改造

- [2026-10-25 接续状态](PROJECT_STATUS_20261025.md)：Web 修复、真实本地调用、
  当前预览、工程验证及仍未完成的工作。
- [Web 陪伴策略与蒸馏交接](WEB_SUPPORTIVE_DIALOGUE_V2.md)：同轮共情与安全确认、
  风险后继续陪伴、L4 保留、独立语言蒸馏路线与未执行事项。
- [数据来源盘点](DATA_SOURCE_INVENTORY_20261025.md)：历史用途、独立病例与校准集缺口。
- [指标与计算链路](REPORT_METRICS_AND_PIPELINES_20261024.md)：风险评分、动作阈值、
  Track A / Gold / Judge / Track B·C 与产品层评估的公式、分母与口径陷阱。
- [可信度审计](TRUST_AUDIT_V2.md)：实际实现、泄漏与威胁模型；
- [研究协议](RESEARCH_PROTOCOL_V2.md)：Track A、两种参考标准、聚类统计与发布边界；
- [专家试标操作](ANNOTATION_OPERATIONS_V2.md)：20 例任务、独立双标、裁决与 Gold 门禁；
- [实际验证记录](VERIFICATION_V2.md)：已执行测试、失败项、产物与外部待办。
- [P0/P1 协议](P0_P1_PROTOCOL.md)：评分投机挑战、单因素阶段诊断、
  实际输入重放与输入匹配的参考标准。
- [CRADLE 本地接入](CRADLE_LOCAL_PROTOCOL.md)：固定公开快照、原生事件评测与
  项目策略跨语言诊断，保留源数据异常，不伪造行动 Gold。
- [CRADLE 实测结果](CRADLE_LOCAL_RESULTS_20261021.md)：600段对话的实际本地输出、
  重放与工程测试结果，以及跨语言缺口和待办。
- [中文外部数据协议](CHINESE_EXTERNAL_PROTOCOL_V1.md)：PsyCrisis固定快照、按输入分组
  的评分校准材料与CPsyCoun开发试标，不把原作者评分当项目Gold。
- [中文试标说明](CHINESE_EXTERNAL_ANNOTATION.md)：20例有损输入视图、来源审查、
  独立双标/裁决与复用Gold评分的操作条件。
- [中文本地执行结果](CHINESE_EXTERNAL_RESULTS_20261021.md)：实际3870条规则输出重放、
  任务包、数据重叠及全量测试仍有2项Windows失败的记录。
- [中文复核与AI初评](CHINESE_REVIEW_AND_PILOT_20261021.md)：技术复核、隐私/上下文队列、
  首批40条真实AI判断与校准分歧；不冒充专家或自动放行Gold。
- [外部参考复现协议](EXTERNAL_REFERENCE_PROTOCOL_V1.md)：无新增专家条件下复用
  PsyCrisis/VERA-MH 原人工评审，CRADLE 原任务隔离，以及本地 Judge 的分阶段门禁。
- [外部参考实际结果](EXTERNAL_REFERENCE_RESULTS_20261021.md)：原始评分一致性、
  数据支持量限制、实际模型服务状态，以及没有完成的新 Judge 实验。
- [当前架构与 P0 入口](CURRENT_ARCHITECTURE_P0.md)：混合 Agent、直接 LLM 基线、
  独立 Judge 的边界，以及不改历史协议的本机启动检查与顺序运行。
- [P0 实际结果](P0_READINESS_RESULTS_20261021.md)：新入口实测、阻断原因和未完成实验。
- [外部参考补全协议](COMPLETION_PROTOCOL_V1.md)：评分表路由、逐维一致性与聚类
  敏感性、参考支持量和不得自动放行的边界。
- [2026-10-22 补全验收](COMPLETION_RESULTS_20261022.md)：本轮真实测试、
  审计重放和仍需模型、数据管理员及专家的事项。
- [本地 Judge 安装与真实运行](LOCAL_JUDGE_RUN_20261022.md)：官方运行时/权重摘要、
  实际加载记录、额外调用账目和冻结校准状态。
- [DeepSeek 接入与 Track A 主实验](TRACK_A_DEEPSEEK_20261023.md)：provider 兼容修复、
  资格结果、24 例两政策指标（含 3 例急性危机）与 `full-history` 证据契约失败；
  非正式主实验，非专家 Gold。
- [Judge 外部校准（DeepSeek）](JUDGE_CALIBRATION_DEEPSEEK_20261023.md)：60 条 CounselBench
  抽样上的两轮判分、共情锚点一致性与分回复者差异；不是专家 Gold 或独立盲测。
- [Windows 测试限制与处理](WINDOWS_TEST_LIMITS_20261023.md)：符号链接与 0600 权限
  的 Windows 差异、条件断言的依据，以及仍需在 POSIX 验证的安全属性。
- [外部数据接入与冻结](EXTERNAL_DATASET_INTAKE_20261023.md)：按优先级注册五个外部来源，
  冻结 MentalHealthBench 与 CounselBench-Eval，记录 MENTAT/ReachOut 未取得与
  SOS-HL-1K 暂缓的原因；不把外部标签转换成本项目行动 Gold。
- [MentalHealthBench 外部回应评测](MENTALHEALTHBENCH_EXTERNAL_V1.md)：固定论文与
  数据、可见输入/私有准则隔离、本地真实两例冒烟、独立重放与全量未执行状态。
- [Judge V2 历史诊断](JUDGE_DIAGNOSTIC_V2.md)：30 例用户开发标签、V2.5 的失败与
  历史产物；不能按有效子集宣称满分。
- [Judge V3 候选诊断](JUDGE_DIAGNOSTIC_V3.md)：分维证据输出与逐问题风险分类、
  失败保留、完整审计重放、开发结果和未完成的独立人工验证。

以下为 v1 研究预发布路径。v1 Hidden 可由公开源码重建，不是独立保密盲测。
v2 独立盲测材料不得进入旧发布路径；当前没有执行任何外部数据发布。

## 历史发布文档

- [Benchmark Card](./BENCHMARK_CARD.md)：用途、评测协议、指标和限制；
- [Datasheet](./DATASHEET.md)：数据生成、构成、划分、质量和维护；
- [专家标注手册](./ANNOTATION_GUIDELINES.md)：试标、标签定义、盲评和一致性；
- [外部合成事件日志导入契约](./EXTERNAL_LOG_IMPORT.md)：中立 JSON/JSONL 输入、
  脱敏、幂等导入和专家重标草稿边界；
- [版本变更记录](../CHANGELOG.md)：代码与 Benchmark 版本变化。
- [本地 Docker 演示](../docker/README.md)：启动、健康检查、数据挂载和验收边界。

使用 `eco benchmark package` 生成的研究预发布包还会包含：

- `evaluation/`：指标、泄漏审计、报告和秘密扫描实现；
- `baselines/`：本地规则基线与受控模型策略源文件；
- `prompts/`：静态 Prompt 文本、版本和 SHA-256 身份；
- `models/`：经审核的模型快照，或明确的待补快照占位说明；
- `runs/`：仅在显式提供运行目录时复制的脱敏原始输出与 AgentTrace；
- `docker/`、`reproducibility/` 和 `scripts/`：容器配置、锁文件及校验脚本。

包根目录的 `RELEASE_MANIFEST.json` 记录每项材料是已包含、占位、未提供还是被保留，
并列出逐文件 SHA-256。运行 `python3 scripts/verify_release.py` 可在安装依赖前校验
包完整性。没有真实 LiteLLM 快照时，包不会伪造模型结果；专家标注、真实 API 主实验、
许可证和 Docker 冷启动仍属于外部发布门槛。

这些文档当前属于研究预发布材料。精确泄漏和字符 3-gram 近重复审计已经进入
`eco benchmark validate` 与发布包验证报告；专家复核、公开许可证、真实 API
主实验和真实 Docker 验收仍是首次公开 v1 前的阻断项。
