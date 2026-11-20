# Track A 真实模型主实验：DeepSeek 接入记录（2026-10-23）

状态：**开发性主实验，不是正式盲测、不是专家 Gold、不是临床验证**。
模型：DeepSeek `deepseek-flash`（官方 OpenAI 兼容端点）；判分器与本次 Track A 分开。

## 1. 做了什么

在既有 provider 层（OpenAI 兼容网关）接入 DeepSeek，完成三件事：

1. 让资格探针适配“思考型”模型（原探针预算过小，会把正常模型判成不可用）；
2. 让结构化输出在托管端不支持 `json_schema` 时回退到 `json_object`；
3. 增加进程级 provider 额外参数，用于关闭思考模式，避免结构化输出被推理内容挤空。

随后完成模型发现、资格测试、三次试跑和一次 8 例主实验（政策：`eco-safe` 与
`current-message-llm`）。

## 2. 代码改动（均为兼容性修复，不改变评分口径）

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `providers/registry.py` | 探针预算 24→64（基础）、24→128（流式）、48→256（结构化） | 思考型模型在 24 token 内只输出推理、正文为空，被误判为不可用 |
| `providers/client.py` | 把 “unavailable / not available” 计入能力不支持标记 | DeepSeek 返回 “This response_format type is unavailable now”，原逻辑不触发回退 |
| `providers/client.py` | 新增 `default_extra_body`，禁止覆盖受保护字段 | 需要传 `{"thinking":{"type":"disabled"}}` |
| `config.py` | 新增 `ECO_PROVIDER_EXTRA_BODY`（JSON） | 进程级配置，不写文件、不落盘 |
| `cli.py` / `api.py` / `provider_runtime.py` / `interactive_api_runtime.py` | 透传额外参数 | 保持所有入口一致 |

验证：provider/config 相关 **68 项通过**，新增 4 项（额外参数下发、受保护字段拒绝、
JSON 解析、非法 JSON 拒绝）；Ruff 通过。

## 3. 资格结果

`eco models discover`：2 个模型（`deepseek-flash`、`deepseek-v4-pro`）。

`deepseek-flash` 在修复后 **ready**：

- 基础对话、流式、结构化（回退到 `json_object`）、用量上报均通过；
- 中文安全微套件通过率 **0.1**，边界拒答率 **0.5**；
- 平均延迟 3.83 秒，P95 5.57 秒。

即：**模型可用，但在这套中文安全微套件上表现很差**。这是关于模型的证据，
不是接入 bug；正式报告必须单列，不能用它反推系统安全。

`deepseek-v4-pro` 在旧探针下判为 ineligible（24 token 内正文为空），本轮未重新
资格化，故不参与实验。

## 4. 三次试跑的失败模式

| 试跑 | 现象 | 判定 |
| --- | --- | --- |
| pilot2（full-history-llm） | 决策引用 `obs-01`，不在该步允许证据集合内 | 契约拒绝，属模型不遵守证据 ID 约束 |
| pilot3（current-message-llm） | 结构化输出正文为空，`json_object` 校验失败 | 思考模式默认开启，推理吃掉正文；关闭思考后修复 |
| pilot4（三政策） | eco-safe 与 current-message 完成 4 个单元；full-history 在 step-0 重复引用当前观测 ID | full-history 仍不满足证据契约，暂停该政策 |

关键结论：**full-history-llm 在本任务上 0 个单元通过证据契约**；这不是运行器缺陷，
而是模型无法区分“当前观测”和“可引用证据”。该政策需要单独设计兼容方案，不能
悄悄放宽约束后仍然叫 Track A 结果。

## 5. 24 例主实验（public_test 全量，两政策，2 表面，重复 1）

运行：96/96 单元完成，1,344 次模型调用，**¥8.04**，约 75 分钟。
样本覆盖全部六条风险轨迹，其中 ACUTE_CRISIS 3 例、SEVERE_NON_ACUTE 6 例。

### 5.1 总体

| 指标（单元均值） | `current-message-llm` | `eco-safe`（规则+模型表达） |
| --- | --- | --- |
| acceptable_action_rate | **0.720** | 0.698 |
| preferred_action_rate | 0.362 | 0.353 |
| prohibited_action_rate | 0.000 | 0.000 |
| critical_action_recall | **0.125** | 0.000 |
| immediate_risk_recall | **0.125** | 0.000 |
| intervention_window_hit_rate | 0.031 | 0.000 |
| late_or_missing_intervention_rate | 0.521 | 0.427 |
| premature_intervention_rate | 0.417 | 0.542 |
| evidence_code_precision | 0.043 | 0.063 |
| depression Brier | **0.151** | 0.364 |
| depression ECE | 0.333 | 0.404 |

### 5.2 急性危机（ACUTE_CRISIS，3 例 / 6 单元）

| 指标 | current-message-llm | eco-safe |
| --- | --- | --- |
| acceptable_action_rate | 0.857 | 0.643 |
| critical_action_recall | **1.000** | **0.000** |
| immediate_risk_recall | **1.000** | **0.000** |
| late_or_missing_intervention_rate | 0.500 | 0.750 |
| intervention_window_hit_rate | 0.250 | 0.000 |
| preferred_action_rate | 0.429 | 0.357 |
| premature_intervention_rate | 0.000 | 0.000 |

这是本轮最值得注意的差异：**关键词规则在危机轨迹上完全没触发（recall 0），模型触发全部
危机（recall 1.0）**。但样本只有 3 个病例、6 个单元，且模型仍有 50% 的窗口未命中，
不能据此宣称“模型更安全”；只能说明规则基线在该合成分布上存在系统性漏检。

### 5.3 其他轨迹摘录

| 轨迹 | 观察 |
| --- | --- |
| SEVERE_NON_ACUTE | 规则 acceptable 1.000 vs 模型 0.613；两者 preferred 均为 0，late 均为 1.0 |
| GRADUAL_WORSENING | 两者 acceptable 接近（0.762 vs 0.786），模型 evidence 0.082 vs 规则 0 |
| PERSISTENT_SUBCLINICAL | 两者 preferred 均 0.833，提前干预率均 1.0 |
| NON_DEPRESSIVE_DISTRESS | 模型 acceptable 0.780 vs 规则 0.500，规则提前干预 1.0 |
| TRANSIENT_LOW_MOOD | 两者 acceptable 都低（0.524 / 0.321），提前干预均 1.0 |

### 5.4 怎么读

- **两张表不能合并成“谁更好”**：规则在 SEVERE_NON_ACUTE 上 acceptable 更高，模型在
  ACUTE_CRISIS 上召回更高，二者互补而非替代。
- **证据契约仍是弱项**：模型 evidence_code_precision 0.043，与 `full-history-llm`
  首步违规一致——模型会输出 JSON，但不遵守证据 ID 约束。
- **校准仍差**：模型 ECE 0.333、规则 0.404，都属于明显过度自信。
- **preferred_action_rate 普遍低**（0.35 左右）：合成规范里的“优选动作”较严格，
  不能直接读成临床正确性。
- 小样本：3 个危机病例、1 次重复、无置信区间；所有差异都需要更多重复才能下结论。

### 5.5 与 8 例试跑的关系

8 例试跑（32 单元，¥2.60）只覆盖前两条低风险轨迹，recall 无分母，已被本节 24 例结果取代。

### 5.6 重复实验：3 次重复的方差

为检验 24 例结果是否只是单次波动，对 `current-message-llm` 在相同 24 例上重复 3 次
（单表面，72 单元，1,008 次调用，¥6.66）。

| 指标 | 重复 1 | 重复 2 | 重复 3 | 范围 |
| --- | --- | --- | --- | --- |
| 全局 critical_action_recall | 0.125 | 0.125 | 0.125 | 恒定 |
| 全局 acceptable_action_rate | 0.732 | 0.744 | 0.744 | 0.012 |
| 全局 late_or_missing | 0.521 | 0.521 | 0.521 | 恒定 |
| **危机 critical_action_recall** | **1.000** | **1.000** | **1.000** | **恒定** |
| 危机 acceptable_action_rate | 0.833 | 0.857 | 0.833 | 0.024 |
| 危机 late_or_missing | 0.500 | 0.500 | 0.500 | 恒定 |
| 危机窗口命中率 | 0.167 | 0.250 | 0.167 | 0.083 |

逐病例（3 次均值）：三个危机病例的 critical/immediate recall 都是 **1.000**。

**必须说清楚的口径**：3 次重复测的是**解码稳定性**，不是病例抽样不确定性。
重复次数再多也不能把“3 个危机病例”变成“危机病例总体”。召回率的置信区间需要
更多**不同**危机病例（例如 dev 的 3 例、hidden 的 6 例），而不是同一批病例的重复。

可以说的：在这 3 个公开危机病例上，模型的危机识别是稳定复现的（9/9 单元命中），
且窗口命中率仍在 0.17–0.25、迟到/漏干预仍为 0.5——**识别到 ≠ 及时行动**。

## 6. 成本

| 运行 | 调用次数 | 金额（按声明价） |
| --- | --- | --- |
| pilot2 | 31 | ¥0.14 |
| pilot4 | 58 | ¥0.33 |
| 8 例试跑 | 448 | ¥2.60 |
| 24 例主实验 | 1,344 | ¥8.04 |
| **24 例 × 3 重复** | **1,008** | **¥6.66** |
| 合计 | 2,889 | ≈ ¥17.8 |

单价按 DeepSeek 官方峰值价折算（输入 ¥2/M、输出 ¥8/M）；平台实际账单可能不同。

## 7. 复现

```powershell
$env:LITELLM_BASE_URL = "https://api.deepseek.com/v1"
$env:LITELLM_API_KEY = "<进程内设置，不写文件>"
$env:ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION = "2.0"
$env:ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION = "8.0"
$env:ECO_MODEL_PRICING_VERSION = "deepseek-flash-20261023-declared"
$env:ECO_API_BUDGET_CNY = "40"
$env:ECO_PROVIDER_EXTRA_BODY = '{"thinking":{"type":"disabled"}}'

eco models discover -o artifacts/models/deepseek-discovery-20261023.json
eco models qualify -i artifacts/models/deepseek-discovery-20261023.json --models deepseek-flash -o artifacts/models/deepseek-flash-snapshot-2.json
eco benchmark run --track A --policy eco-safe,current-message-llm --limit 24 ^
  --agent-model deepseek-flash --model-snapshot artifacts/models/deepseek-flash-snapshot-2.json ^
  --run-dir artifacts/runs/deepseek-trackA24-20261023
```

## 8. 非声明

- 不是独立盲测：公开数据可能已进入预训练语料。
- 不是专家 Gold：参考动作来自合成病例规范，未经过真实专家复核。
- 不是正式主实验：样本量、模型数量和重复次数都远低于协议要求。
- 价格是声明值，不是账单核对结果。
- full-history 的失败应保留为负面证据，不得通过放宽证据约束“修复”后当作通过。
