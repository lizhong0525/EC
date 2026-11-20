# Judge 外部校准：DeepSeek 托管模型（2026-10-23）

状态：**开发性外部校准，不是专家 Gold，也不是独立盲测**。
数据：CounselBench-Eval 冻结快照 `8d56a96e…`，抽样 60 条（四类回复者各 15 条）。
判分器：本项目五维 Judge Prompt（v2.5）+ 托管模型结构补充，DeepSeek `deepseek-chat`。

## 1. 设置

| 项 | 值 |
| --- | --- |
| 抽样 | 分层，seed `20261023`，per_responder=15 |
| 材料 | 100 个咨询问题 × 4 类回复者（gpt4/llama3/gemini/human），每题 5 位评审 |
| 判分调用 | 每条 1 次，temperature 0，max_tokens 1400，无重试、无修复 |
| 运行 | 两轮独立运行，各 118 秒 |
| 成本 | 每轮约 99,876 prompt + 13.3k completion tokens |
| 密钥 | 仅进程内环境变量，不写入仓库或快照 |
| 结构约束 | 托管 API 的 json_object 不强制 Schema，提示词内重述精确结构；1/60 因截断未被接受 |

判分器只输出五维 0/1 与原文证据，评测材料（user/assistant）被视为数据而非指令。

## 2. 结果（两轮）

| 维度 | 第一轮阳性率 | 第二轮阳性率 |
| --- | --- | --- |
| empathy | 0.542（32/59） | 0.492（29/59） |
| regulation | 0.051（3/59） | 0.051（3/59） |
| exploration | 0.051（3/59） | 0.034（2/59） |
| risk_inquiry | 0.000（0/59） | 0.000（0/59） |
| external_support | 0.797（47/59） | 0.797（47/59） |

有效输出 59/60（两轮相同）；两次失败均为 `INVALID_JSON`（输出被 max_tokens 截断，
按协议保留为失败，不重试、不修复）。

### 2.1 与 CounselBench 人类共情分的对齐

锚点规则：5 位评审的 `empathy_score` 均值 ≥ 4 记为“人类认为有共情”。

| 指标 | 第一轮 | 第二轮 |
| --- | --- | --- |
| 判分器阳性 | 32 | 29 |
| 锚点阳性 | 24 | 25 |
| 一致率 | 72.9% | 72.9% |
| Cohen κ | 0.466 | 0.456 |
| 命中/误报/漏报/真阴 | 20 / 12 / 4 / 23 | 19 / 10 / 6 / 24 |
| 召回 | 83.3% | 76.0% |
| 精确率 | 62.5% | 65.5% |

解读：**判分器倾向多判“有共情”**（误报约为漏报的 2–3 倍），但能抓住大部分
人类标注的共情回复。

### 2.2 分回复者（第一轮，n=15，llama3 因 1 条失败为 14）

| 回复者 | 判分器阳性 | 锚点阳性 | 一致率 | 召回 | 精确率 |
| --- | --- | --- | --- | --- | --- |
| gpt4 | 7 | 8 | 93.3% | 87.5% | 100% |
| llama3 | 10 | 12 | 71.4% | 75.0% | 90.0% |
| gemini | 5 | 1 | 73.3% | 100% | 20.0% |
| human（真实咨询师） | 10 | 3 | 53.3% | 100% | 30.0% |

真实咨询师回复上的精确率只有 30%：判分器把大量“治疗性套话”当成共情，而人类评审
没有给到 4 分。这是需要人工逐条复核的构造效度问题，不能简单说成模型更好或更差。

### 2.3 两轮自一致（58 条共同有效）

| 维度 | 一致率 | κ |
| --- | --- | --- |
| empathy | 94.8% | 0.897 |
| regulation | 100% | 1.000 |
| exploration | 100% | 1.000 |
| risk_inquiry | 100% | 1.000 |
| external_support | 100% | 1.000 |

注意：后四维在本样本中阳性极少（0–3 条），100% 一致主要由全 0 造成，**不能**当作
该四维判分能力已被验证。

## 3. 能说什么与不能说什么

可以说：

- 该判分器在 60 条 CounselBench 回复上重复运行高度稳定；
- 其共情维度与人类评分中等一致（κ≈0.46，召回 76–83%，精确率 63–66%）；
- 一致率随回复来源显著变化，真实咨询师回复最差。

不能说：

- 不能说判分器“整体有效”；regulations/exploration/risk_inquiry 在本样本里几乎没有阳性；
- 不能把 CounselBench 的人类分当成我们模型回复的专家分；
- 不能把 60 条英文材料的结果外推到中文青少年场景；
- 不能把托管模型的一次运行当作独立盲测或临床结论。

## 4. 复现

```powershell
$env:DEEPSEEK_API_KEY = "<仅在当前进程设置>"
$env:PYTHONPATH = "backend/src;."
& $py scripts/external_calibration.py sample --snapshot artifacts/external/counselbench-eval/<rev> --output artifacts/research/human-calibration-20261023-v1
& $py scripts/deepseek_judge.py run --packet artifacts/research/human-calibration-20261023-v1 --output .../judge-deepseek-run1.jsonl
& $py scripts/deepseek_judge.py report --packet ... --judge .../judge-deepseek-run1.jsonl --output .../judge-report-run1.json
& $py scripts/deepseek_judge.py compare --packet ... --first .../run1.jsonl --second .../run2.jsonl --output .../judge-consistency.json
```

原始输出：`judge-deepseek-run1.jsonl`、`judge-deepseek-run2.jsonl`、
`judge-report-run1.json`、`judge-report-run2.json`、`judge-consistency.json`、
`responder-breakdown.json`。

## 5. 下一步

1. 把 `blind.csv` 交给 3–5 位同学盲评，得到我们五维口径下的人类一致性（Fleiss κ），
   再与本次判分器结果比较；当前“判分器 vs 我们的人类评审者”仍是空白。
2. 逐条复核 12 条误报，确认是判分器过宽、还是人类评审过严（或两者都有）。
3. 需要风险阳性样本（本次 0 条），否则 risk_inquiry 维度完全没有校准证据。
4. 换成本地中文材料前，不要引用本页数字作为中文场景结论。
