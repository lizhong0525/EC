# EcoCompanion 规范 Agent 预标注与专家校正标注方案

> 版本：v0.1（研究方案）  
> 日期：2026-09-04  
> 适用范围：EcoTriage-Bench 合成中学生病例及其陪伴 Agent 回复评测

## 1. 核心思路

采用“**规范 Agent 预标注 + 专家校正 + 独立金标准冻结**”的三级标注方案：

```text
规范 Agent 生成候选标签
          ↓
独立专家盲标与风险路由
          ↓
高风险/分歧案例由专家复核和裁决
          ↓
Gold 金标准 + Silver 扩展标签 + Unresolved 未决标签
          ↓
用 Gold 校准规范 Agent 和自动 Judge
```

规范 Agent 是标注加速器，不是专家的替代物。专家负责高风险安全判断、规范边界
确认和最终金标准；Agent 产生的标签不得伪装成独立专家标签。

## 2. 标注目标与边界

标注目标不是诊断学生是否患有抑郁症，而是判断：

- 当前可见证据是否需要进一步澄清或直接安全询问；
- 陪伴 Agent 当前轮应采取哪一种支持、边界或现实转介动作；
- 何时开始转介、何时仍可观察、何时延迟会产生安全代价；
- 回复是否引用了真实证据，是否存在排他依赖、虚假保密、医疗承诺或虚假交接。

所有标签都针对合成病例中的“行动规范”，不能被解释为临床诊断、治疗效果或真实
未成年人安全证明。

## 3. 三种标签层级

| 层级 | 生成方式 | 主要用途 | 论文地位 |
|---|---|---|---|
| `SILVER` | 规范 Agent、规则和 WorldEngine 生成 | 全量预标注、调试、候选筛选、训练辅助 | 不作为最终金标准 |
| `GOLD` | 专家独立标注、专家校正和裁决 | Hidden Test、主结果、核心安全指标 | 论文主要依据 |
| `UNRESOLVED` | 专家仍有合理分歧或证据不足 | 不确定性分析、边界案例报告 | 不强行转成单一标签 |

`SILVER` 标签必须记录来源模型、模型快照、Prompt 版本和置信度。`GOLD` 标签必须
记录参与专家数量、裁决状态和校正原因。`UNRESOLVED` 节点应从强制单标签排名中排除，
但可作为安全压力测试单独报告。

## 4. 角色分工

### 4.1 规范 Agent

规范 Agent 只读取当前节点允许观察到的内容，不得读取隐藏状态、未来节点或程序
生成的优选动作。它输出结构化候选标注：

- 风险等级及置信度；
- 支撑证据 ID 和受控证据代码；
- 可接受动作集合；
- 优选动作和禁止动作；
- 是否需要直接安全询问；
- 转介对象和转介时间窗；
- `abstain`（无法判断）标记；
- 简短理由码，不保存自由形式 Chain-of-Thought。

### 4.2 专家 A、B

- 独立完成规定病例和决策节点；
- 第一轮不得看到规范 Agent 标签；
- 只使用当前节点可见信息；
- 对高风险、家庭不安全、掩饰和前后矛盾案例给出置信度；
- 按标注手册填写证据、动作和时间窗。

### 4.3 专家 C（裁决者）

负责：

- 所有 L4 或潜在危机节点；
- 专家 A、B 的标签分歧；
- 规范 Agent 与专家共识的重大分歧；
- 随机抽检的普通节点；
- 最终 `GOLD` / `UNRESOLVED` 状态确认。

### 4.4 数据管理员与统计人员

数据管理员负责匿名任务包、版本锁定、完整性和权限，不参与独立判断。统计人员在
一致性统计完成前不得修改原始标注，只能生成派生结果。

## 5. 分阶段工作流

### 阶段 0：冻结协议

在生成候选标签前冻结：

1. `CaseSpecV1`、`ObservationV1` 和标注 Schema；
2. 风险等级、动作本体、证据代码和转介窗口定义；
3. 标注手册版本；
4. 规范 Agent 的模型快照、Prompt、解码参数和输出校验规则；
5. 数据划分和随机种子。

协议冻结后，改变风险定义、动作集合或标签语义必须创建新的 Benchmark 版本。

### 阶段 1：规范 Agent 全量预标注

规范 Agent 对 2,016 个规范决策节点生成候选标签。每条结果必须：

- 通过结构化 Schema；
- 只引用已出现的 `observation_id`；
- 不引用未来事实或隐藏变量；
- 对无法判断的节点主动 `abstain`；
- 记录模型、Prompt、时间、成本、重试和校验结果。

预标注失败时不能静默补写标签，应进入重试或 `UNRESOLVED_CANDIDATE` 队列。

### 阶段 2：专家盲标试点

从不同风险轨迹、社会生态和表达方式中抽取 20 个病例进行试标。专家此时看不到
规范 Agent 建议，目的是测量独立人类共识，并发现标注手册中的歧义。

根据试标结果修订手册，然后冻结正式版本；试点标签不能直接混入正式 Gold。

### 阶段 3：风险导向路由

将节点分为四类：

| 路由 | 进入条件 | 标注要求 |
|---|---|---|
| `CRITICAL` | L4 候选、明确/隐式自伤信号、无法保证安全 | 两名专家独立标注，专家 C 必须裁决 |
| `DISAGREEMENT` | Agent 与专家初标冲突，或 A/B 标签冲突 | 两名专家复核，必要时 C 裁决 |
| `EDGE_CASE` | 玩笑、否认、掩饰、家庭可能不安全、依赖/身份越界 | 专家双标，至少抽样裁决 |
| `ROUTINE` | 低风险且证据清晰的普通节点 | 分层抽样双标，其余可保留 Silver |

建议正式数据中对所有 `CRITICAL` 和 `EDGE_CASE` 节点做全量专家审核；普通节点采用
分层抽样时，必须报告抽样比例和加权方式。

### 阶段 4：专家校正

只有在专家完成独立判断后，才向专家展示规范 Agent 候选标签。专家逐字段选择：

- 接受；
- 修改；
- 拒绝；
- 信息不足，标记 `UNRESOLVED`。

每次修改必须记录 `correction_reason`，例如：

- 漏掉了纵向证据；
- 误读了玩笑或否认；
- 引用了未来信息；
- 过早或过晚转介；
- 把家庭不安全误判为可信任成人；
- 缺少必要的直接安全询问；
- 回复存在依赖或保密边界问题。

专家不能只给“整体同意/不同意”，高风险字段必须逐项确认。

### 阶段 5：裁决与冻结

专家 C 查看：

- 当前可见病例材料；
- 专家 A、B 的独立标签；
- 规范 Agent 候选标签；
- 各方证据 ID 和理由码。

裁决后生成不可变的 `GOLD` 记录。若证据不足或存在合理规范分歧，则生成
`UNRESOLVED`，保留分歧信息，不使用任意投票结果掩盖不确定性。

### 阶段 6：校准与蒸馏

Gold 集冻结后，才能用于：

- 校准规范 Agent 的置信度；
- 训练或蒸馏更稳定的候选标注器；
- 校准 LLM Judge；
- 发现系统性漏报、过度升级和证据幻觉。

不能用规范 Agent 参与生成的标签，反过来证明该 Agent 自己正确；必须使用其未参与
生成的 Gold 节点进行验证。

## 6. 标注内容

每个“病例—决策节点”至少包含：

1. 风险等级 `L0–L4/B`；
2. 置信度和是否需要澄清；
3. 最小充分证据集合；
4. 可接受动作集合；
5. 优选动作；
6. 禁止动作；
7. 是否需要直接安全询问；
8. 转介目标；
9. 最早、最佳起点、最佳终点和最迟安全节点；
10. 理由码、校正原因和标签状态。

风险等级表示行动强度，不表示医学诊断。若关系边界问题与 L2–L4 同时出现，安全
风险动作优先，边界澄清作为次动作。

## 7. 建议的数据结构

当前 `NodeAnnotationV1` 面向人类判断。规范 Agent 不能直接伪装成
`AnnotationRole.INDEPENDENT`，建议新增两个版本化对象：

```text
CandidateAnnotationV1
  schema_version
  benchmark_version
  case_id
  step_id
  source_type: simulator | normative_agent
  source_model_snapshot
  prompt_identity
  risk_level
  confidence
  evidence
  acceptable_actions
  preferred_action
  prohibited_actions
  direct_safety_check_required
  referral_window
  abstain
  validation_status
  generated_at
```

```text
AdjudicatedAnnotationV1
  schema_version
  benchmark_version
  case_id
  step_id
  final_label_status: GOLD | UNRESOLVED
  final_annotation
  independent_annotator_count
  adjudicator_id
  correction_reason
  disagreement_codes
  expert_agreement
  finalized_at
```

建议的目录：

```text
artifacts/annotations/
├── candidate/       # 规范 Agent 预标注
├── independent/     # 专家 A/B 原始独立标注
├── adjudicated/     # 专家 C 裁决后的 Gold/Unresolved
├── audits/          # 一致性、修正率和抽样审计
└── manifests/       # 版本、模型快照、Prompt 和哈希
```

## 8. 质量门槛

正式 Gold 冻结前至少检查：

- 危机标签 Cohen’s κ 或 Krippendorff’s α ≥ 0.80；
- 风险等级加权 κ ≥ 0.75；
- 动作集合平均 Jaccard 一致性 ≥ 0.70；
- 所有高风险节点都有两名专家标注和第三方裁决；
- 证据只能引用当前或历史可见观察；
- 不允许可接受动作与禁止动作相交；
- 转介窗口满足：

```text
earliest ≤ optimal_start ≤ optimal_end ≤ latest_safe
```

- 记录 Agent 接受率、字段修正率、专家间分歧率和未决率；
- 报告各风险轨迹、社会生态和表达方式的分层覆盖；
- Gold、Silver 和 Unresolved 不得混用或隐式合并。

若一致性未达标，应先修订手册并重新标注对应子集，不能只保留有利样本。

## 9. 论文实验设计

将标注流程本身作为一个可验证的方法学实验，比较：

1. `Agent-only`：只使用规范 Agent；
2. `Expert-only`：只使用专家独立标注和裁决；
3. `Agent-assisted`：规范 Agent 预标注，专家按风险路由校正。

比较以下指标：

- 与 Gold 的风险等级一致性；
- L4 高风险召回和漏报率；
- 证据 Precision/Recall/F1；
- 转介时间窗命中率；
- 禁止动作率；
- 专家每病例耗时；
- 每个修正字段的成本；
- Agent 置信度的校准误差；
- 不同路由策略的人工工作量。

核心假设是：

> Agent-assisted 在保持高风险安全性能的前提下，能够降低专家标注成本，并通过
> 不确定性路由把人工注意力集中到最需要判断的节点。

若只完成抽样专家审核，论文应明确将数据命名为：

> `Silver Benchmark with Expert-Audited Gold Subset`

不能将未审核的全量 Silver 结果表述为专家金标准结果。

## 10. 安全与方法边界

- 规范 Agent 不能读取隐藏风险状态，也不能把模拟器标签当作临床事实；
- 高风险节点必须由确定性规则和专家审核共同兜底；
- Agent 低置信度不能自动降级为低风险；
- 专家校正阶段不得泄露真实个人信息；
- 回复盲评和病例标签标注应分开，避免模型名称或回复风格影响风险判断；
- 自动 Judge 必须用独立专家评分子集校准；
- 该流程仍不能证明真实临床诊断、治疗效果或真实未成年人部署安全性。

## 11. 在 EcoCompanion 中的落地顺序

1. 保持现有 `NodeAnnotationV1` 作为人类标注契约；
2. 新增 `CandidateAnnotationV1` 和 `AdjudicatedAnnotationV1`；
3. 实现规范 Agent 的结构化预标注器、`abstain` 和证据校验；
4. 增加候选标签与专家标签的字段级差异报告；
5. 实现 CRITICAL / DISAGREEMENT / EDGE_CASE / ROUTINE 路由；
6. 增加 Gold/Silver/Unresolved 的数据划分和报告字段；
7. 完成 20 个病例试标并冻结标注手册；
8. 完成高风险全量审核、普通节点分层抽样和第三专家裁决；
9. 用冻结 Gold 校准规范 Agent、LLM Judge 和主实验指标；
10. 在 Benchmark Card、Datasheet 和论文中分别报告三类标签来源。

## 12. 最终决策

本项目采用以下原则：

> **规范 Agent 负责“先看一遍、提出候选”；专家负责“确认风险、纠正边界、裁决高危”；
> Gold 负责论文结论；Silver 负责扩展规模；Unresolved 负责保留真实不确定性。**

这比纯专家全量标注更节省成本，也比纯 Agent 标注更适合心理健康高风险场景。
