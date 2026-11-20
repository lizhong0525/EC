# 2026-10-22 补全验收记录

日期：2026-10-22。新增范围：`external-reference-completion-v1.0.0`。
本记录只报告本机实际执行结果；历史实验、冻结任务包和旧报告未覆盖。

## 本轮已完成

| 项目 | 实际结果 | 结论边界 |
|---|---|---|
| VERA 快照完整性 | 通过 | 固定 Git 对象、清单和文件摘要一致 |
| VERA 评分表结构 | 33 个问题、72 条回答边、全部可达、无循环 | 不是临床效度，也不是论文完整复现 |
| VERA 路由契约 | 12/12 通过 | 是程序分支夹具，不是专家或模型答案 |
| VERA 严格重放 | 已实现 | 只重放外部精确答案，不从对话推断 |
| VERA 原始人工诊断 | 5 个维度分别统计 | 是原评分者之间的一致性，不是 Judge 性能 |
| PsyCrisis 支持量 | 600 校准 + 200 留出已复核 | 风险询问留出阳性仍只有 7 条，原门禁阻断 |
| 审计重复性 | 两次新目录输出的 JSON SHA-256 相同 | 仅证明同环境确定性重放 |
| 前端测试 | 3/3 通过 | 不是浏览器跨视口验收 |
| 前端构建 | Vite/TypeScript 通过 | Docker 冷启动仍未验证 |
| 新脚本静态检查 | Ruff 通过，mypy 通过 | 不替代行为和数据审计 |
| 专项回归 | 77/77 通过，90.32 秒 | 新增 32 项及既有 P0/外部参考 45 项，非新增模型实验 |

## 关键统计

VERA 每维的原始评分者 alpha、按对话聚类区间、按人物模板聚类敏感性区间如下。
区间是描述性 bootstrap，不是临床置信区间；人物模板只有 10 个，且是模拟人物，
不代表 10 个真实独立受试者。

| 维度 | alpha | 对话聚类 95% 区间 | 人物模板敏感性区间 | 多数参考无支持类别 |
|---|---:|---:|---:|---|
| Detects Risk | 0.5541 | 0.3731–0.7113 | 0.2293–0.6456 | High Potential for Harm |
| Clarifies Risk | 0.7261 | 0.6334–0.8116 | 0.6313–0.8060 | 无 |
| Guides to Human Support | 0.6250 | 0.5047–0.7243 | 0.5346–0.7014 | 无 |
| Collaborates and Validates Appropriately | 0.4529 | 0.2595–0.6035 | 0.1243–0.5328 | Best Practice、High Potential for Harm |
| Maintains Safe Boundaries | 0.7395 | 0.6516–0.8225 | 0.6334–0.8395 | 无 |

PsyCrisis 原五维工程门禁保持不变。风险询问为校准 21 条阳性、公开留出 7 条阳性，
总计 28 条；两阶段各要求至少 20 条阳性，因此当前数据不可能通过原五维联合支持量
检查。没有移动、复制、删除或自动生成标签。

## 实际产物

- VERA 审计：`artifacts/research/completion-20261022-v3/vera-flow/audit.json`
- VERA 私有结构：`artifacts/research/completion-20261022-v3/vera-flow/private/rubric-structure.json`
- PsyCrisis 支持量：`artifacts/research/completion-20261022-v3/reference-support.json`
- 本轮全量后端 JUnit：`artifacts/research/completion-20261022-v3/tests-full.xml`
- 本轮新增目标测试 JUnit：`artifacts/research/completion-20261022-v3/tests-targeted.xml`
- 前端 JUnit：`artifacts/research/completion-20261022-v3/tests-web.xml`
- P0 预检：`artifacts/research/completion-20261022-v3/doctor/summary.json`

两次独立执行的 `audit.json` SHA-256 均为：
`A7546F46BD848BB9C4BAAAB786B5E0C4E106F9318501196D540C78736B4FCFC3`。
两次独立执行的 `reference-support.json` SHA-256 均为：
`0421E04EC598A571637E22DBEEB436824D8196E278B61D9E20EFE1B9A3E03BF1`。

## 测试结果

实际全量命令使用进程内配置隔离，先移除配置环境变量，再禁用默认 dotenv，
避免测试导入应用时读取用户凭据文件。测试自行创建的临时配置夹具仍可使用。
这不修改磁盘配置或当前终端的环境。

```powershell
.research-venv/Scripts/python.exe -X utf8 -c "import os; [os.environ.pop(k, None) for k in tuple(os.environ) if k.startswith(('LITELLM_', 'ECO_'))]; from ecocompanion.config import Settings; Settings.model_config['env_file'] = None; import pytest; raise SystemExit(pytest.main(['backend/tests', '-q', '--junitxml=artifacts/research/completion-20261022-v3/tests-full.xml']))"
```

实际结果：**519 passed，2 failed，0 skipped**。
其中本轮新增的 32 项测试全部通过：VERA 25 项、参考支持量 7 项。
用例覆盖严格答案匹配、缺失答案、跳题、循环、无效评分者、未决判断、
单一类别下 alpha 不可定义，以及假设性区间的输入检查。

专项重跑命令：

```powershell
.research-venv/Scripts/python.exe -X utf8 -m pytest `
  backend/tests/test_vera_flow_audit.py `
  backend/tests/test_reference_support_analysis.py `
  backend/tests/test_p0_validation.py `
  backend/tests/test_external_reference_workflow.py `
  -q --junitxml=artifacts/research/completion-20261022-v3/tests-targeted.xml
```

实际结果：**77 passed，0 failed**。此前一次命令误写既有测试文件名，
以文件不存在退出、没有运行测试；不计作通过，也不混入这次专项结果。

保留的两个失败：

1. `test_catalog_lists_only_valid_non_symlink_runs`：Windows 当前账户创建目录
   符号链接失败，`WinError 1314`。没有通过跳过测试来伪造安全验证。
2. `test_json_import_creates_observable_only_expert_draft`：Windows 文件模式报告
   `0666` 而非 POSIX 断言的 `0600`。这不能证明 Windows ACL 已经安全，未将模式位
   断言改成通过。

前端命令：

```powershell
cd web
npm.cmd test -- --reporter=default --reporter=junit `
  --outputFile.junit=../artifacts/research/completion-20261022-v3/tests-web.xml
npm.cmd run build
```

实际结果：**3 个前端测试通过，生产构建通过**。
没有运行未安装 Playwright 的旧 UI 脚本，也没有覆盖历史 UI 截图。

## 尚未完成

- 本机 Ollama 不可用，精确模型未选择，Judge 新推理为 0 次；
- 未进行 600+200 真实模型校准/留出，也未开放 3,040 条项目回复评审；
- 未完成真实目标 Agent 主实验；
- 没有项目特有的行动、禁止动作、时间窗、证据合理性 Gold；
- 没有独立保密盲测、独立数据管理员确认、授权/隐私复核；
- VERA 的论文级客户端、完整历史请求和精确研究版本等价性尚未建立；
- Docker daemon 未运行，未完成干净环境镜像冷启动；
- 没有读取密钥、付费调用、上传材料、下载模型或开展真人研究。

因此，本轮结论是：**可信流程的离线审计与可复现检查已补齐；Benchmark
整体不能标记为完成或已验证安全。**下一步需要真实模型运行、适用数据补充、
独立数据管理和合格专家 Gold，不能由本地规则或 LLM Judge 自动替代。
