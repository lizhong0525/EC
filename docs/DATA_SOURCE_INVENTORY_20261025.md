# 数据来源、独立病例与 Judge 校准材料清单

核对日期：2026-10-25。

范围：项目源代码、docs、artifacts 中的原始快照、任务包和实验记录，以及项目外层的
30 题评分文件、旧项目 ZIP、codex同步和临时 CounselBench 副本。
第三方依赖自带的测试数据不算本项目数据来源。本次只新增本清单，没有修改历史材料、
生成病例或标签，没有运行 Judge、目标模型、训练或微调。

## 一、结论

1. 有多份开发数据和外部原始数据，但未找到完成来源审核的独立病例包；
   未找到名为 `case_groups.csv` 的文件。
2. 那 30 题已用于 V2、V3 多轮 Prompt、证据校验与聚合规则开发，不能作为新校准集。
3. 有一版不含这 30 题的现成回复包：CounselBench 的 60 条回复、46 个问题组。
   但它已跑过两轮 DeepSeek Judge，不是新的未见留出集；五维人类标注回收仍为 0。
4. 真正新收集、独立人工标注、每维正负支持充分的中文 Judge 校准集尚未找到。
5. 本地记录中的“调参”主要是提示词、规则、契约和推理配置开发；
   未找到本项目训练或微调模型权重的实现和对应训练数据清单。
   不能推断所用外部模型在预训练或上游微调时没有见过这些公开材料。

## 二、现有来源总表

表中“使用”指本项目已记录的用途，不表示原作者训练用途或本项目临床验证。

| 来源 | 本地位置 | 实际规模 | 已记录用途 | 独立性及标签边界 |
| --- | --- | --- | --- | --- |
| EcoTriage 本地模板生成器 | `backend/src/ecocompanion/benchmark/generation.py`、`core/templates.py`；`artifacts/cases/ecotriage-v1.jsonl` | 144 病例，14 节点/例，2 种表层；72/24/24/24 划分 | 生成开发病例；规则回归；Track A/B 本地实验；DeepSeek Track A 试跑、24 例实验及重复实验 | 构造标签不是专家 Gold；公开生成器和种子可重建 hidden_test |
| Track C 手写关系边界场景 | `backend/src/ecocompanion/benchmark/track_c.py`；`artifacts/runs/track-c-local/` | 12 场景；4 配置 × 5 种子，共 240 单元 | 关系边界、依赖和保密压力的确定性本地演示/测试 | 手写场景与预设响应，不是独立临床材料或真实模型优势证据 |
| 模型资格化中文安全微套件 | `backend/src/ecocompanion/providers/qualification.py`；`artifacts/models/` | 20 个合成探针，5 类 | 模型资格筛查、DeepSeek 接入及输出契约试验 | 不是那 30 题；不是正式病例盲测；标准来自工程规格 |
| 用户审核的 30 题 | 外层 `心理支持回复要素评测结果.json`；`artifacts/research/judge-diagnostic-v2/cases.json` | 30 个 user/assistant 对，150 个二元字段 | V2.0–V2.5、V3.1–V3.3 的提示词、风险语义边界、证据与聚合规则调试 | 已反复参与开发；用户开发标签，不是专家 Gold |
| PsyCrisis-Bench | `artifacts/external/psycrisis/56a862aa9a5209b6836c17c2697ab4f29e644535/`；同提交的 `reference-psycrisis/` | 608 输入行、607 个规范化唯一输入；800 原评分回复、400 输入组 | 608 输入 × 5 本地策略；原回复 600 校准/200 公开留出；40 条会话 AI 初评中的源回复部分；旧本地 Judge 正式校准尝试 | 800 不是 800 独立病例；旧 Judge 414 次调用、413 有效，校准未完成；200 留出未执行但不是保密盲测 |
| CPsyCounD | `artifacts/external/cpsycoun/8fe0fa0f77630b921932bdaa6bae7481bc611cf2/CPsyCounD.json` | 3,134 段、24,443 用户节点；抽 20 例、166 节点 | 20 例开发试标包、830 条本地规则输出、上下文与隐私预检查、空专家库 | 不是独立新收集；去掉咨询师历史会损失上下文；HF 与 GitHub 条款待澄清 |
| CRADLE-Dialogue | `artifacts/external/cradle-dialogue/49d5f40912e9afeadf4d0c1aa0ffd22c8d89c6a3/` | 恢复 Train 3,058、Validation 420、Test 600 段；Test 4,527 用户节点 | Test 原生事件弱基线及 5 项目策略，共 31,689 条规则输出；后续同源重跑 | 英文；人物/帖子/模板身份不明；许可未明确；事件标签不是行动/时窗标签 |
| VERA-MH | `artifacts/external/reference-vera/1775e23c520b03831a99a78c3f6d20322df20ba0/` | 90 段英文模拟对话、270 份原始评分、6 个评分者 ID、10 persona | 原评分一致性、rubric/流程审计与统计重算；未运行新的 VERA Judge | 不是 90 个独立真实患者；原维度不是本项目五维或 Track A 动作 Gold |
| MentalHealthBench | `artifacts/external/mentalhealthbench/a3e94c7786bd514952c542d6a5694bb24055f136b417f15c5c6ab942976d7474/` | 1,215 任务、5,262 准则；青少年情境 257；中文仅 1 条 | 全量公开任务包准备；最短两例真实本地生成及判分冒烟，另有一次失败历史 | 不是全量模型实验；不能外推中文青少年；rubric 不是行动/时窗 Gold |
| CounselBench-Eval | `artifacts/external/counselbench-eval/8d56a96ea1de3f3f190f77f4ca9bc3503d731af7/` | 2,000 评分行 = 100 问题 × 4 类回复 × 5 评审；400 条原回复 | 抽 60 回复，DeepSeek Judge 两轮，另建人类盲评 CSV | 英文；五维新人工标签未回收；CC-BY-NC-ND-4.0，不应默认授权改编或再分发 |

### 不是新增来源的派生产物

- `ecotriage-v2-pilot/`、`ecotriage-v2-pilot-r2/`、
  `gold-readiness-20261023-v1/` 的 20 例来自 legacy Train，不因 UUID 或新目录变成独立病例。
- `p0p1-v21/`、`p0p1-v22/`、`local-run-01/`、`reply-pilot-01/`、
  `chinese-review-20261021-v1/` 等是已有病例的诊断、输出或评分派生包。
- `artifacts/release/`、`artifacts/demo_build/`、旧项目 ZIP、
  源码 snapshot ZIP、`codex同步/` 是历史副本、打包或会话证据，不另计样本。
- 外层 `undefined/temp/cb_probe/counselbench_eval.csv` 是 CounselBench 临时副本。
- `artifacts/traces/web.jsonl`、`artifacts/memory/events.jsonl` 是运行记录，
  未找到将其作为经授权真实参与者研究语料的证据，不应直接用于病例或临床评价。
- 单元测试中的 TEST_FIXTURE、已知答案挑战、模拟专家身份和模拟 HTTP 输出，
  只用于工程验证，不计入真实病例、人工标注或模型实验。

### 非病例的研究参照

| 参照 | 状态与用途 |
| --- | --- |
| WHO mhGAP 出版页及自伤评估网页 | `artifacts/research/guideline-sources-20261023-v1/`；已抓取，用于指南局部约束候选；不代表整本指南已核对或规则已审核启用 |
| NICE NG225 | 既有抓取记录为 HTTP 403/UNAVAILABLE，不算已取得正文 |
| MentalHealthBench 论文 PDF | `artifacts/external/mentalhealthbench-paper/`；用于评分公式与 Prompt 复核，不是新增病例 |
| ABC / MLR-Bench 论文 | 用于有效性、阶段诊断和统计报告设计，不提供心理标签 |
| Adolescent-Depression-Simulator | 仅作为社会生态、角色和时间推进的概念参照；项目声明没有复制其源码或数据 |

主要用途证据：`DATASHEET.md`、`CURRENT_ARCHITECTURE_P0.md`、
`TRACK_A_DEEPSEEK_20261023.md`、`JUDGE_DIAGNOSTIC_V2.md`、
`JUDGE_DIAGNOSTIC_V3.md`、`CHINESE_EXTERNAL_RESULTS_20261021.md`、
`CHINESE_REVIEW_AND_PILOT_20261021.md`、`EXTERNAL_REFERENCE_RESULTS_20261021.md`、
`CRADLE_LOCAL_RESULTS_20261021.md`、`MENTALHEALTHBENCH_EXTERNAL_V1.md`、
`JUDGE_CALIBRATION_DEEPSEEK_20261023.md`、对应 manifest 和 report。
本次没有重跑所有历史实验。

## 三、独立病例包与 case_groups.csv

### 当前找到的最接近材料

| 现有包 | 内容 | 为什么还不满足要求 |
| --- | --- | --- |
| `artifacts/research/gold-readiness-20261023-v1/` | 20 病例、280 节点，专家流程和指南流程的准备包 | 来自公开生成器 Train；Gold 0、reviewed Silver 0，不是独立来源 |
| `artifacts/research/chinese-20261021-v1/cpsycoun-packet/` | 20 段、166 节点，已有来源映射与空专家流程 | 已跑规则；上下文视图有损；人物、骨架和模板分组未核实 |
| `artifacts/research/mhb-external-v1/public-packet-v1.1/` | 1,215 个公开回应任务及私有映射 | 不是新增来源，且不能直接充当中文纵向行动病例；全量未执行不等于独立性已证明 |

未找到已完成独立来源审核的病例包，也未找到 `case_groups.csv`。
名字包含 `independent-rescore` 的文件是“独立重算报告”，不是“独立病例”。

如果后续建立该表，建议至少使用：

```csv
case_id,source_dataset,source_revision,source_record_id,person_group,conversation_group,plot_group,template_group,source_group,split,used_for_generation,used_for_tuning,used_for_30q_debug,used_for_model_trial,review_status
```

该行只是字段建议，不是已填好的分组文件。未知人物/模板必须明确记为 UNKNOWN，
不能用新 UUID 推定互相独立；有关联的对话、改写和同人物材料应绑定分组，
由已知依赖关系的连通分量约束划分。人物身份未知时，结论只能限制在可核验的对话层。

## 四、不使用那 30 题的 Judge 回复包

### 可以直接定位的现成版本

目录：`artifacts/research/human-calibration-20261023-v1/`。

- `items.jsonl`：60 条原始 user/assistant 对，含私有来源与共情评分锚点；
  不应将含 anchor/source 的整个记录直接输入 Judge。
- `blind.csv`：供人类评分的匿名版本，五维评分列为空。
- `manifest.json`：`labels_returned=0`、`expert_gold=false`、
  `independent_blind_test=false`。
- `judge-deepseek-run1.jsonl`、`judge-deepseek-run2.jsonl`：两轮历史自动判断。
- 60 条唯一回复来自 46 个问题组；相同问题的多回复不是独立输入病例。
- 本次对两包文本做 NFKC、去空白、转小写后的精确比较：
  与 30 题的 user 文本重叠 0，完整 user/assistant 对重叠 0。
  这是精确去重，不是语义去重或预训练污染检测。

### 每维初步正负例数量

下表是已有 DeepSeek Judge 的初判分布，不是人工参考分布。
每轮 60 条中各 59 条有效、1 条无效；无效不能算负例。

| 维度 | 第一轮正例 | 第一轮负例 | 第一轮无效 | 第二轮正例 | 第二轮负例 | 第二轮无效 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| empathy / 具体共情 | 32 | 27 | 1 | 29 | 30 | 1 |
| regulation / 具体调节 | 3 | 56 | 1 | 3 | 56 | 1 |
| exploration / 开放探索 | 3 | 56 | 1 | 2 | 57 | 1 |
| risk_inquiry / 直接风险询问 | 0 | 59 | 1 | 0 | 59 | 1 |
| external_support / 现实支持 | 47 | 12 | 1 | 47 | 12 | 1 |

数据依据：`judge-report-run1.json`、`judge-report-run2.json`。
本项目口径下五维已回收的人类正例/负例数均为 0/0，60 条均待标注。
CounselBench 的上游共情分可以作为另一个口径的锚点，不能替代其余四维标签，
也不能迁移为项目新回复的标签。

这版符合“不包含那 30 题”，但不符合“此前从未用于开发或校准的新材料”。
如果“新的”要求严格未见，现有目录没有合格版本。
不能按这位 Judge 自己的判断选样、再用自己的标签证明自己准确。

### 旧中文人工参考支持量，仅供对照

PsyCrisis 冻结的 600 校准/200 公开留出不是新集，不能改名充数：

| 维度 | 校准正/负 | 公开留出正/负 |
| --- | ---: | ---: |
| empathy | 410/190 | 133/67 |
| regulation | 367/233 | 136/64 |
| exploration | 104/496 | 39/161 |
| risk_inquiry | 21/579 | 7/193 |
| external_support | 494/106 | 173/27 |

依据：`docs/P0_READINESS_RESULTS_20261021.md`。风险询问总阳性仅 28，
无法同时达到原协议两个阶段各至少 20 正例的工程筛查门槛。
这些是外部原回复要素参考，不是项目行动、时窗或转介对象 Gold。

## 五、缺失项的网络来源建议

以下是本次实际核对的公开入口，不表示已下载数据、审清底层权利或完成独立性审计。
仓库/API/数据卡属于外部来源声明，不是本项目临床适用性的证明。

| 优先级 | 来源与入口 | 本次核对结果 | 适合补什么 | 尚需处理 |
| --- | --- | --- | --- | --- |
| 1 | [PsyDial 官方仓库](https://github.com/qiuhuachuan/PsyDial)；[PsyDial-D4 数据](https://huggingface.co/datasets/qiuhuachuan/PsyDial-D4) | 官方 README 成功读取；镜像元数据 API HTTP 200，revision `0ceef5ac19fb8174baa6be31ce1b2637711cb4c2`，gated=false；含 `PsyDial-D4.json`，数据卡为中文、Apache-2.0 | 未在本项目使用的中文长程病例候选；保留完整上下文抽取新回复 | 下载并核验原始字节；核查重建方法、原始来源、人物/原对话关联、与既有库的重合；重建/半真实不等于真实独立病例，也不自带五维或行动 Gold |
| 2 | [CPsDD 官方仓库](https://github.com/FakerBoom/CPsDD)；[作者数据入口](https://drive.google.com/drive/folders/186ZTX0hkVClxONz6H6Tdg0HuOTmLWpfX) | README HTTP 200，作者声明 68,136 段；列有 test、真实咨询例和改写参考；GitHub dataset 目录只有占位 readme，实际数据在 Drive；README 限研究用途 | 中文心理支持多轮对话及原回复候选 | 本次未核验 Drive 下载；先确认研究许可、种子与模板关联、生成模型及来源重合；不能把策略或严重度标签直接转为五维 |
| 3 | [ESConv 官方仓库](https://github.com/thu-coai/Emotional-Support-Conversation) | README 和目录 HTTP 200；作者说明 1,300 段，另有 196 个失败对话；限定学术研究 | 英文真实参与者情绪支持对话、支持者回复和策略对照 | 不是中文青少年或风险询问专项集；策略标签不等于本项目五维；失败对话不等于每条回复五维全负；LICENSE 正文请求超时，不能声称条款已全部核实 |
| 后续 | [MENTAT 项目页](https://www.robotics.stanford.edu/blog/mentat/) | 本次连接仍超时；本地未取得数据 | 如合法取得，可核查是否有匹配的专业分诊参照 | 不推测 schema、标签或可用规模；需要实际数据入口和授权 |
| 后续 | [ReachOut / CLPsych 2016](http://clpsych.org/shared-task-2016/) | 本地登记为受控访问；本次未申请或取得新数据 | 青年社区分诊的外部迁移参照 | 不等于心理支持回复校准集；需申请及遵守访问条款 |
| 暂缓 | [SOS-HL-1K 相关入口](https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval) | 既有接入记录指出与 PsyCrisis 存在来源关系；本次未新增接入 | 经审计后才可能有增量危机语料 | 先查来源重叠与许可，不能先当独立新增样本 |

注意：另有同名 `jiSilverH/psydial`，对应不同工作，不是上面 ACL 2025 的
`qiuhuachuan/PsyDial`。获取时应锁定作者、论文和版本，不能仅按数据集名字下载。
PsyDial GitHub 的 CC0 LICENSE 与 HF 数据卡 Apache-2.0 分属不同载体，
数据使用应核对数据本体条款，不能直接套用代码仓库许可。

## 六、建议的下一步

1. 先取得 PsyDial-D4 并完成来源/隐私审查、与旧数据精确和语义去重，
   再按原对话、人物、骨架和模板关联创建病例候选包及真实填写的 `case_groups.csv`。
   未标注可先保留，但审核和独立性状态不能自动通过。
2. 病例评估包与 Judge 校准包按来源关联组隔离，不能从同一对话抽前半段做调参、
   后半段做“独立”测试。公开数据还需披露未知的预训练污染。
3. 新 Judge 包可从未使用的中文原回复中抽取，再由不同真实评分者按冻结五维标注；
   另设不参与调试的组级留出。AI 初标必须标为草稿，不冒充人工参考。
4. 风险询问正例应专门补充合法来源与适用语境；普通情绪支持数据未必足够。
   每阶段每维至少 20 正/20 负仅是旧工程筛查要求，不是充分样本量或临床验证标准。
   标注前不能虚报“新集每维已够 20”；若仍不足，应披露不足而非复制、改写或自动贴标凑数。

当前可定位的是旧数据清单及不含 30 题的既有 60 回复包；
严格独立病例包、填写完成的分组表、全新且有充分人工正负参考的中文 Judge 包仍待建设。
