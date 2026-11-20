# 外部数据接入记录：按优先级冻结与审计

日期：2026-10-23
工具：`scripts/external_datasets.py`（版本 `external-datasets-v1.0.0`）
状态登记：`artifacts/external/registry.json`
测试：`backend/tests/test_external_datasets.py`

## 1. 本轮范围

按既定顺序接入 4 个外部数据源，SOS-HL-1K 暂缓：

| 顺序 | 数据 | 优先补什么 |
| --- | --- | --- |
| 1 | MentalHealthBench | 上下文中的回应适当性与安全约束 |
| 2 | MENTAT 分诊子集 | 专业分诊参考下的行动紧迫程度 |
| 3 | CounselBench-Eval | Judge 与已有专家评分的一致性 |
| 4 | ReachOut | 青年社区场景的外部迁移验证 |
| 暂缓 | SOS-HL-1K | 先排查与既有数据的来源重叠 |

本轮完成的是“取得原始字节 → 校验身份 → 冻结快照 → 记录许可与原生分布”。
没有把任何外部标签转换成本项目的逐节点行动 Gold，没有运行模型评分，也没有做
专家复核。外部数据可以成为独立参考轨道，不能自动成为中文青少年行动 Gold。

## 2. 实际状态

| 顺序 | 数据 | 状态 | 实际结果 |
| --- | --- | --- | --- |
| 1 | MentalHealthBench | 已冻结 | 2.1 MB 归档，SHA-256 与成员哈希双验，1215 条记录 |
| 2 | MENTAT 分诊子集 | 未取得 | 官方项目页在本机不可达，未下载任何字节，未假设 schema |
| 3 | CounselBench-Eval | 已冻结 | 3.2 MB CSV，固定 revision + git blob 校验，2000 条人工评价 |
| 4 | ReachOut | 未取得 | 数据受控访问，需向管理方申请，本轮未提交申请 |
| 暂缓 | SOS-HL-1K | 暂缓 | 与在用的 PsyCrisis 存在来源重叠，先做来源审计 |

## 3. 已冻结快照

### 3.1 MentalHealthBench（第 1 优先）

路径：`artifacts/external/mentalhealthbench/a3e94c7786bd514952c542d6a5694bb24055f136b417f15c5c6ab942976d7474/`

- 归档：`archive.zip`，2,106,277 字节，SHA-256 `a3e94c7786bd514952c542d6a5694bb24055f136b417f15c5c6ab942976d7474`
- 数据文件：`mentalhealthbench_eval.jsonl`，6,492,410 字节，SHA-256 `a112166bd0809567ac785643acd3adcebf209db53487787c73883a3bcd9a6e7d`
- 许可：MIT（归档内 LICENSE，与数据卡 `license: mit` 一致）

原生分布（由工具逐条校验后统计，不是抽样）：

| 维度 | 计数 |
| --- | --- |
| 记录 | 1,215（ID 唯一，均为 32 位十六进制） |
| 评分准则 | 5,262 条，合计 8,422 分 |
| 语言 | en 903、es 105、hi 54、ar 34、pt 29、de 25、fa 17、id 17、it 17、tr 13、**zh 1** |
| 紧急度 | non_acute 650、high_acuity 221、emergent 344 |
| 用户画像 | adult 828、teen 257、clinician 70、caregiver 60 |
| 有实质前文 | 70 条（其余只有年龄等提示） |
| 对话长度 | 1–16 条消息 |
| 行为轴 | context_seeking 1124、actionable_guidance 933、clinical_reasoning 853、interpretation 659、empathy 596、reality_testing 363、over_alarmism 285、dual_use 230、collaboration 170、communication 49 |

边界：原版只有 1 条中文对话；rubric 是“评分准则”，不是本项目的动作、时窗或
转介对象标签；257 条青少年情境不是中文青少年材料。

### 3.2 CounselBench-Eval（第 3 优先）

路径：`artifacts/external/counselbench-eval/8d56a96ea1de3f3f190f77f4ca9bc3503d731af7/`

- 数据文件：`counselbench_eval.csv`，3,168,928 字节
- 身份：HuggingFace revision `8d56a96ea1de3f3f190f77f4ca9bc3503d731af7`，
  git blob `7b5dc395091779e1724cbc03a9e166743d719365`（本地重算一致），
  SHA-256 `3754607cafb0af64a5e8fc863444da5845dd6499237b94b563b6a3dfea982130`
- 许可：CC-BY-NC-ND-4.0（数据卡），本地研究用，禁止改编与再分发

原生分布：

| 维度 | 计数 |
| --- | --- |
| 行 | 2,000（100 个问题 × gpt4/llama3/gemini/human 各 500 行） |
| 话题 / 问卷 | 20 个话题，100 个 survey_id |
| overall_score | 2,000 个数值，均值 3.357，范围 1–5 |
| empathy_score | 均值 3.268；specificity 均值 3.719；toxicity 均值 1.835 |
| factual_consistency_score | 1,930 个数值 + 70 条 “I am not sure” |
| medical_advice_score | 全为文本：No 1,729、Yes 220、“I am not sure” 51 |

注意：`medical_advice_score` 不是 1–5 数值列，任何按数值处理的脚本都会失败；
工具保留原文并单列计数，不做静默转换。

取得方式说明：本机无法直连 `huggingface.co`（连接超时），CSV 经
`hf-mirror.com` 取得。身份校验用固定 revision 和 git blob 内容寻址，
能证明字节与上游仓库对象一致；镜像元数据本身不是独立授权证明。

## 4. 未取得的三个来源

- **MENTAT（第 2）**：官方项目页 `robotics.stanford.edu/blog/mentat/` 在本机
  处于不可达状态（HTTP 连接失败），没有取得任何字节，也没有依据论文描述
  推测字段结构。解锁条件：可访问的官方数据地址，或作者直接提供数据与许可。
- **ReachOut（第 4）**：Reddit 来源数据受控访问，需要向管理方提交申请并获批；
  本轮没有提交申请，因此没有数据、没有标签空间可核对。
- **SOS-HL-1K（暂缓）**：与在用的 PsyCrisis 存在来源关系，必须先做样本重叠审计，
  再决定是否有新增可用材料；其许可与底层来源授权也仍未核对。

## 5. 能支持与不能支持的判断

| 数据 | 能支持 | 不能支持 |
| --- | --- | --- |
| MentalHealthBench | 回应适当性、安全约束、上下文使用的分轴评分 | 中文人群结论；逐节点行动/时窗 Gold |
| CounselBench-Eval | Judge 与原始回复上专家评分的维度一致性 | 该评分不能迁移到本项目模型的新回复 |
| MENTAT（未取得） | 尚未接入 | 任何指标、任何标签映射 |
| ReachOut（未取得） | 尚未接入 | 任何指标、任何标签映射 |

两者都没有“下一步做什么动作、何时升级、转介给谁”的逐节点参考，因此不改变
专家 Gold 与独立盲测仍然缺失的结论。

## 6. 复现

```powershell
$py = ".research-venv\Scripts\python.exe"
& $py scripts/external_datasets.py list
& $py scripts/external_datasets.py plan
& $py scripts/external_datasets.py verify --snapshot artifacts/external/mentalhealthbench/a3e94c7786bd514952c542d6a5694bb24055f136b417f15c5c6ab942976d7474
& $py scripts/external_datasets.py inventory --snapshot artifacts/external/counselbench-eval/8d56a96ea1de3f3f190f77f4ca9bc3503d731af7
```

`verify` 会重算每个文件的 SHA-256、git blob 与全部原生计数；文件改动或计数漂移
都会直接报错，不会静默通过。快照目录一经创建即拒绝覆盖。

## 7. 非声明

- 没有专家复核、没有独立盲测、没有真实模型实验，也没有任何临床或部署结论。
- 快照仅限本地学术研究；CC-BY-NC-ND-4.0 材料不得改编或再分发。
- 公开数据可能已进入预训练语料，不能当作保密盲测。
- 外部数据集的许可与底层来源权利没有独立核验，只记录数据卡原文。
- 本轮没有修改历史实验、旧评分、专家数据库或发布包。
