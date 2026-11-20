# v2 实际验证记录

日期：2026-10-20。这是工程执行记录，不是专家实验结果或临床有效性报告。

## 环境

本机 Windows，实际使用 Python **3.13.7** 的 `.research-venv`。
该环境使用 system-site-packages 加补充安装，不等于项目计划的 Python 3.11 锁定冷启动。
没有改写原锁文件；完整解析后包清单及平台信息保存在
`artifacts/research/ecotriage-v2-pilot-r2/runtime-inventory.json`。
安装时补充了 SQLAlchemy asyncio 所需 greenlet；首次全套测试因其缺失而未能收集，
安装后才完成下述测试。不能把首次失败写作通过。

## 已执行

| 执行 | 实际结果 |
|---|---|
| `pytest backend/tests/test_research_v2.py -q -p no:cacheprovider` | 31 passed；最后一次 17.86s |
| `pytest backend/tests -q -p no:cacheprovider --junitxml=artifacts/research/backend-tests-final.xml` | **325 passed，2 failed，71.10s**；不是全绿 |
| `mypy backend/src/ecocompanion` | 87 个源码文件，无问题 |
| Ruff：research 包、根 CLI、release.py、新测试及 UI fixture | 通过 |
| Ruff：全 backend/src | 未通过；未触及的 `benchmark/leakage.py:3` 有 I001 导入排序问题 |
| Node + Playwright + 本机 Edge | 桌面 1440×1100、手机 390×844；登录、凭据清空、提交锁定、下一节点、双角色隔离、无横向溢出、无 pageerror 通过 |
| 本地标注服务 | `127.0.0.1:8765/health` 实际返回 ok |
| `prepare` / `schemas` / `run-local` / `agreement` / `label-status` / `audit` / `blind-replies` | 实际运行，详见下文产物 |
| `sensitivity --max-cases 24` | 24 个公开病例、3 参数设置×4 本地规则×5 种子，共 1440 次；合理性待专家审核 |
| 最终试标协议与当前源码/环境指纹比较 | 匹配；程序中的漂移拒绝另有回归测试 |

浏览器第一次尝试因缺少 Playwright 自带 Chromium 而失败，之后明确改用本机 Edge 并实际通过。
UI 操作只提交到临时 TEST-ONLY 库，没有写入实际试标库。截图：
`artifacts/research/ui-qa/desktop.png`、`mobile.png`。
可复验脚本为 `scripts/research_ui_test.cjs`，要求 Node 可解析 playwright，
本机 Edge 可启动，8766 未占用；不下载浏览器、不调用模型。

全套失败位置（保留原断言，没有通过跳过或放宽测试获得全绿）：

1. `backend/tests/test_benchmark_catalog.py:69`：Windows 创建符号链接缺少权限，WinError 1314。
2. `backend/tests/test_benchmark_log_import.py:118`：断言 POSIX 0600，Windows 返回 0666。

这两项与新增路径无关，但整体 CI 仍未通过。不得将 POSIX mode 测试失败解读为
已经完成 Windows ACL 验收。真实 ACL、3.11/Linux 锁定冷启动仍待验证。
未运行现有 React 前端完整构建、Docker 冷启动或真实远端模型实验；
新增静态标注页面已单独浏览器验收。

## 最终交付包

当前应使用 **`artifacts/research/ecotriage-v2-pilot-r2`**：

- 20 例、280 节点；均来自 legacy Train，人工 judgments=0、attestations=0。
- 六种风险轨迹各 3–4 例，四种生态各 5 例；表达风格分配为 6/7/7。
- `private/`：空专家库、独立角色访问凭据、固定身份映射、模拟参考、手册快照。
- `public/inputs.json`：公开开发输入，仅供开发者；不可提前交给独立标注者看完整未来。
- `protocol.json`、`runtime-inventory.json`、`pilot-manifest.json`：冻结协议、环境、覆盖与审计。
- 分层/生成器元数据仅用于管理员组织任务，不属于模型/独立专家输入。

早先的 `ecotriage-v2-pilot` 保留为中间冻结包，没有覆盖；最终类型修正造成源码指纹变化，
故另建 r2。它们是同一批公开规范病例的两次打包，**不是 40 例独立样本**。
不要用旧包配当前源码静默重跑，也不要修改旧协议摘要。

其他实际产物：

| 路径（均相对项目根） | 内容与状态 |
|---|---|
| `artifacts/research/local-run-01` | 四个离线规则策略各 280 条成功输出，共 1120；模型调用 0，已知成本 0 |
| `artifacts/research/local-run-01/report.json` | 专家指标全部 `UNAVAILABLE_NO_APPLICABLE_GOLD` / null；模拟恢复独立列示 |
| `artifacts/research/pilot-agreement-01.json` | INCOMPLETE，绝非专家一致性通过 |
| `artifacts/research/pilot-status-01.json` | 280 个 UNLABELED，未自动提升 Gold |
| `artifacts/research/reply-pilot-01` | 1120 条匿名随机排序的本地模板回复，PENDING_EXPERT_REVIEW |
| `artifacts/research/schemas-v2` | v2 数据、协议、标注和预测 Schema |
| `artifacts/research/track-b-sensitivity-v2.json` | 已测参数网格无严格排名反转；不证明参数合理、真实有效或全面稳健 |
| `artifacts/research/backend-tests-final.xml` | 最终全后端测试的机器可读结果，含两项失败 |

统计报告将这 20 例按声明的骨架等关系合为 **6 个主聚类**，另外报告 14 个模板族。
它不把 280 个节点、1120 条输出或新 UUID 当成新的独立病例。
试标全部属于 development，跨划分近重复队列为空不构成独立性证据；
语义插件未运行，状态明确 NOT_RUN。

`scripts/freeze_research_source.py --output artifacts/research/source-snapshot-v2`
生成本地源码快照及逐文件 SHA256 清单。白名单只含源码/测试/文档/锁文件，
不遍历任何 artifacts/private、病例或标注文件；没有外部发布动作。
将当前代码与协议/运行清单一起保留，之后新增版本不得覆盖它。

## 回归测试覆盖

覆盖完整 decision/render provider 封套中的隐含字段/嵌套消息与编码 ID；
当前节点可见性、未来引用拒绝、提交锁定、专家隔离、审计篡改；
三人不同身份、无变异 κ 不可用、回顾性意见不改原判断；
Gold 动作/等级/证据/时窗变化及模拟指标不变、置信度不作模拟真值；
无 Gold 指标不可用、缺失结果留在分母、伪造引用计工程失败；
分组划分、语义插件接口、独立材料复核/私有输出；
匿名回复、严重违规队列、源码漂移拒绝及离线 CLI 集成。

测试中的“专家”数据和核验均标记 TEST_FIXTURE，只存在临时目录，
不得解释为任何真实专家参与或金标准证据。

## 必须由外部完成

- 两位真实专家与第三裁决者的资格核验、手册/码表审查、独立试标、裁决及一致性达标。
- 独立管理员获取与开发者隔离的新材料，核实来源、完整语义/人工复核与访问权限。
- 所需伦理/同意/数据治理流程；当前未进行真人研究，更未研究真实未成年人。
- 真正冻结模型身份与 Prompt 的传输层审计、资格/成本日志、真实 API 主实验。
  现有 `qualified_api` 仅为严格导入契约，不是已验证的付费运行器。
- 真实回复专家校准、严重违规人工审查、Track B 参数合理性审查。
- 正式预注册、独立病例量规划、跨平台锁定复现、生产权限和发布审批。

验收含义：本地研究流程与防误用门禁已实现并有可执行检查；
Benchmark 的外部效度、专家共识、真正盲测和真实模型优势仍未被验证。
