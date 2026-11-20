# CRADLE-Dialogue 本地验证结果

日期：2026-10-21。接入版本：`cradle-local-v1.0.0`。
执行环境：Windows，现有 `.research-venv`，Python 3.13.7；
httpx 0.28.1、pyarrow 23.0.1、numpy 2.1.3、pydantic 2.12.5。
不是项目声明的 Python 3.11 干净锁定环境冷启动，未改变原依赖锁文件。

## 本轮真正完成的工作

- 下载固定提交的 Train/Validation/Test 原始 Parquet，共 6,552,363 字节，
  每个数据文件均匹配上游 LFS SHA-256，保存原数据卡与元数据。
- 解析审计三个划分，原件只读使用，没有修补源文件、改写标签或覆盖历史结果。
- 冻结600段测试对话的4,527个用户节点，分别执行两个原生事件弱基线、五个项目策略。
- 实际保存31,689条输出，全部逐条确定性重放通过，无缺失、无执行错误。
- 另外执行一次独立报告重算，两份 JSON 报告逐字节一致。
- 模型调用、重试、token、费用均为0；没有读取模型配置、上传对话或部署大型权重。

## 数据审计

| 划分 | 恢复对话数 | 总行数 | 无标准角色前缀行 | 标签来源声明 |
|---|---:|---:|---:|---|
| Train | 3,058 | 48,557 | 65 | 来源方模型合成 |
| Validation | 420 | 6,679 | 8 | 来源方模型合成 |
| Test | 600 | 8,975 | 0 | 来源方临床专家标注 |

实际文件缺少文档提及的 `dialogue_id`，由连续回合和零重置恢复边界。
Test 有121段以 Listener 开场，全部保留；有2个 Listener 行标签，保留原件与审计，
不擅自移到用户行。因此4,527个用户节点参考共有711个标签实例，而不是原件713个。
没有发现跨划分完全相同的整段对话，但未实施语义去重或人工泄漏复核。
缺少源人物、帖子、模板映射，600段不能自动解释成600名独立被试。

## 原生事件弱基线

采用精确用户回合、类型与 ongoing/past 一致才算命中的16类事件 Micro-F1；
不是“有风险/无风险”准确率，也不是可接受行动准确率。

| 本地基线 | TP | FP | FN | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 始终不输出事件 | 0 | 0 | 711 | 不适用 | 0% | 0% |
| 固定英文词组首次提及 | 29 | 101 | 682 | 22.31% | 4.08% | 6.90% |

英文弱基线的恢复对话聚类 bootstrap 95%区间为4.70%–9.17%，对话集合 Micro-F1
为11.51%。这是本地弱基线的描述，不是项目大模型成绩、官方论文复现或临床能力。
源帖/人物依赖未知，区间可能偏窄。该词表未实现可靠否定、人物归属、时间和 Alert
推理，不在测试集上根据结果增补词表，低分不能推导本任务本身不可完成。

## 原项目策略在英文上的表现

| 策略 | 实际主动作 | 有规则证据节点 | 成功执行 |
|---|---|---:|---:|
| always-listen | 全部 SUPPORTIVE_LISTENING | 0 | 4,527/4,527 |
| always-escalate | 全部 URGENT_ESCALATION | 4,527个无条件基线标记 | 4,527/4,527 |
| keyword-triage | 全部 SUPPORTIVE_LISTENING | 0 | 4,527/4,527 |
| history-aware-rule | 全部 SUPPORTIVE_LISTENING | 0 | 4,527/4,527 |
| eco-safe | 全部 CLARIFY_SYMPTOMS | 0 | 4,527/4,527 |

这里的“成功”仅指返回契约输出，不代表行动正确。无条件升级的证据码是基线标记，
不是临床证据。现有风险规则和回复模板主要为中文，直接接英语后未识别出风险证据；
这是明确的跨语言能力缺口，不应为制造高分而修改测试标签或偷偷替换检测器。

原生事件不提供可接受动作集合、禁止动作、安全询问要求或转介对象，所有项目专家
行动指标保持不可用；不存在模拟隐藏状态，因此不计算隐藏状态恢复分数。

## 工程测试

- 新增接入单元/集成/CLI测试：**32通过**，从本轮后端全量 XML 逐项核对。
- 后端全量：**386通过、2失败、0跳过**，pytest 耗时131.32秒。
- 研究包 mypy：17个源码文件通过；本轮改动文件 Ruff 检查通过。
- 两项失败与历史记录一致，未修改或跳过原断言：
  `test_benchmark_catalog.py:69` 受 Windows 符号链接权限限制；
  `test_benchmark_log_import.py:118` 的 POSIX0600 权限断言在 Windows 返回0666。
- 未执行前端构建、Docker冷启动、真实模型推理或真人/专家研究。

初次数据审计因训练集角色格式异常拒绝继续，随后新增“仅审计未知角色、不猜测角色”
处理；初次冻结因对话开场角色检查过严而拒绝，核实121段 Listener 开场后修正并
增加回归测试。失败发生在正式计划生成之前，最终完整运行没有删例或补造标签。
早期29项/91项测试报告保留，最终32项与全量388项以 `backend-tests.xml` 为准。

## 本地入口与产物

项目根目录下：

- 原始快照：`artifacts/external/cradle-dialogue/49d5f40912e9afeadf4d0c1aa0ffd22c8d89c6a3/`。
- 最终冻结计划：`artifacts/research/cradle-20261021-v1/test-packet/plan.json`。
- 最终数据审计：`artifacts/research/cradle-20261021-v1/test-packet/data-audit.json`。
- 原始输出：`artifacts/research/cradle-20261021-v1/run/predictions.jsonl`。
- 最终报告：`artifacts/research/cradle-20261021-v1/run/report.json`。
- 独立重算：`artifacts/research/cradle-20261021-v1/report-recomputed.json`。
- 全量测试：`artifacts/research/cradle-20261021-v1/backend-tests.xml`。
- 协议与限制：[本地接入协议](CRADLE_LOCAL_PROTOCOL.md)。

报告 SHA-256：`588c030a1e440ba20892f1c87abff1f8a26e867bcb9f83a79d3ccd8ba467b017`。

快照已在本地，之后无需重新下载。只读审计或另存重算报告：

```powershell
$py = '.research-venv/Scripts/python.exe'
& $py -m ecocompanion.research cradle-report `
  --packet artifacts/research/cradle-20261021-v1/test-packet `
  --run artifacts/research/cradle-20261021-v1/run `
  --output artifacts/research/cradle-20261021-v1/report-another.json
```

已有文件拒绝覆盖；源码或环境变化应新建 `cradle-plan`，不能更新旧指纹伪装复现。

## 下一阶段待办

先确认作者授权和源数据歧义，再选择合格的英文/多语模型，冻结模型身份与预算后进行
真实事件检测实验；公开测试集不用于调参。要测试“下一步动作是否适当”，仍需要
输入匹配的专家独立行动标注，不能把临床事件标签直接映射为行动 Gold。
