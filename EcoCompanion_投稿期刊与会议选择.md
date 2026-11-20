# EcoCompanion 投稿期刊与会议选择

> 项目：EcoTriage-Bench + ECO-SAFE  
> 信息核验日期：2026年9月4日  
> 时间说明：AoE 为 UTC−12，AoE 当日 23:59 对应北京时间次日 19:59。

## 1. 核心结论

当前论文的核心是：

1. 面向抑郁风险中学生的长时程陪伴式安全 Benchmark；
2. 兼顾情感支持、风险识别、主动追问和分级转介的序贯决策算法。

按照不同的论文重心，投稿方向如下：

| 论文重心 | 首选投稿方向 |
|---|---|
| 长时对话 Benchmark、风险识别和安全评测 | ACL、NAACL、COLING、TACL |
| 多智能体模拟、长期决策和转介策略 | AAMAS、IJCAI |
| 情感状态、关系建模和陪伴安全 | IEEE Transactions on Affective Computing |
| 未成年人依赖、隐私、偏差和责任审计 | FAccT |
| 主动陪伴、Avatar、信任和用户交互 | ACM TiiS、IUI、CHI、ACII |
| 临床价值和真实学校转介流程 | JMIR Mental Health、Artificial Intelligence in Medicine、npj Digital Medicine |

结合当前项目仍处于设计阶段的实际情况：

> **会议首选 ACL 2027，期刊首选 IEEE Transactions on Affective Computing。**

不建议为了赶近期会议，将工作压缩成“少量合成病例 + Prompt 对比”。这不足以支撑 Benchmark 和算法两个核心贡献。

---

## 2. 会议选择

| 会议 | 推荐的论文包装 | 截稿时间 | 当前判断 |
|---|---|---|---|
| **ACL 2027** | EcoTriage-Bench + ECO-SAFE；突出长时对话、主动信息获取和安全评测 | 对应 ARR 周期计划在 **2027年1月**，具体日期尚未公布 | **最推荐，时间相对合理** |
| **AAMAS 2027** | 多 Agent 社会生态模拟、关系感知状态建模、风险约束序贯决策 | 注册：2026年9月17日 AoE；摘要：10月1日 AoE；全文：**10月8日 AoE** | 内容高度匹配，但当前阶段不宜仓促投稿 |
| **NAACL 2027** | NLP Benchmark、对话 Agent、心理健康语言安全 | ARR 全文：**2026年10月12日 AoE** | 时间过紧；只有已有数据、代码和实验时才考虑 |
| **COLING 2027** | 中文青少年语境、对话资源和 Benchmark | ARR 全文：**2026年10月12日 AoE** | 中文资源方向合适，但时间同样过紧 |
| **FAccT 2027** | 未成年人 AI 陪伴审计、错误转介、依赖、隐私和责任边界 | 摘要：10月27日 AoE；全文：**2026年11月3日 AoE** | 需要加入专家、利益相关者或真实产品审计 |
| **IJCAI 2027** | 不确定性感知 POMDP、约束规划、安全策略优化 | 截稿日期尚未公布 | ECO-SAFE 算法创新足够强时值得重点考虑 |
| **EMNLP 2027** | 大规模长时对话评测、模型行为分析 | 截稿日期尚未公布 | 可作为 ACL 之后的后备目标 |
| **ACII 2027** | 语音、表情、情绪变化、Avatar 和多模态陪伴 | 官方 CFP 尚未公布 | 加入多模态情感计算后很合适 |

### 不建议冲刺的当前会议

- **CHI 2027**：全文截止 2026年9月10日 AoE，而且需要扎实的人体或 HCI 研究。
- **ICLR 2027**：摘要截止 9月18日、全文截止 9月25日 AoE，要求较强的通用机器学习创新。
- **IUI 2027**：主论文已经于 2026年8月20日截止；Poster/Demo 截止 11月10日 AoE，但不能替代正式论文。
- **AAAI 2027**：主论文已经截止。

---

## 3. 期刊选择

| 期刊 | 投稿时间 | 适合的项目版本 | 推荐程度 |
|---|---|---|---|
| **IEEE Transactions on Affective Computing** | 常规稿滚动投稿；相关多模态情感计算专题截止 **2026年12月30日** | 长时情感状态、关系边界、风险变化和情感 Agent 安全评测 | **当前首选期刊** |
| **TACL** | 每月1日 23:59 Honolulu time；下一批约为北京时间 **2026年10月2日17:59** | 成熟的 NLP Benchmark、方法和全面实验 | 很匹配，但门槛高，不宜仓促投稿 |
| **ACM Transactions on Interactive Intelligent Systems** | 常规稿滚动投稿 | 主动陪伴、长期记忆、信任、依赖和人机交互研究 | 加入用户或专家交互实验后推荐 |
| **Artificial Intelligence in Medicine** | 常规稿滚动投稿 | 风险状态估计、不确定性校准、分级转介和临床决策支持 | 需要明确算法创新和临床专家验证 |
| **JMIR Mental Health** | 常规稿滚动；相关专题截止 **2027年1月31日** | 数字心理健康长期安全、效果、伤害和真实工作流 | 需要专家、外部验证或真实应用证据 |
| **npj Digital Medicine** | 常规稿滚动；合成临床数据与隐私专题截止 **2027年6月3日** | 隐私保护的纵向合成心理健康轨迹和临床转化 | 高目标，当前纯合成版本证据不足 |

### 期刊投稿时的关键限制

1. **IEEE TAC**：论文需要突出情感计算问题，不能只写成校园心理产品。
2. **TACL**：需要高质量数据、严格标注、多模型基线、消融和泛化实验。经过 ARR 评审的同一工作通常不能立即转投 TACL。
3. **ACM TiiS**：如果只有“模拟学生 Agent 与被测 Agent 对话”，缺少真实人机交互研究，贡献会偏弱。
4. **AI in Medicine**：仅使用 Prompt 和规则路由不够，ECO-SAFE 应形成明确的状态估计、约束决策或安全优化算法。
5. **JMIR Mental Health**：不适合只有原型和少量案例的形成性研究，需要更充分的专业验证。
6. **npj Digital Medicine**：需要临床人员、真实流程或外部机构验证，不适合作为第一篇论文的直接目标。

---

## 4. 推荐投稿路线

### 路线一：首篇会议论文

目标：**ACL 2027**

论文主体：

> EcoTriage-Bench：长时程陪伴式风险识别与分级转介 Benchmark  
> +  
> ECO-SAFE：关系与不确定性感知的安全陪伴决策算法

提交前至少完成：

- 专家定义的风险轨迹和转介时间窗；
- 冻结轨迹与交互式轨迹；
- 多种 LLM 和陪伴 Agent 基线；
- ECO-SAFE 完整算法；
- 风险识别、发现时间、漏报、过度转介和依赖诱导指标；
- 心理专业人员标注及一致性分析；
- 算法消融、反事实和跨模型泛化实验。

如果错过 ACL 2027，则依次考虑：

1. EMNLP 2027；
2. IJCAI 2027；
3. IEEE Transactions on Affective Computing 常规稿。

### 路线二：期刊扩展论文

目标：**IEEE Transactions on Affective Computing**

在会议版本基础上增加：

- 更大规模的长时程人物和社会生态轨迹；
- 情绪、风险、关系和依赖状态的联合建模；
- 语音或 Avatar 等多模态实验；
- 更多商业或开源陪伴 Agent 审计；
- 心理专家和交互专家评价；
- 更完整的安全边界与失效分析。

### 路线三：临床转化论文

目标：

1. Artificial Intelligence in Medicine；
2. JMIR Mental Health；
3. npj Digital Medicine。

该路线需要在合成 Benchmark 之外增加：

- 儿童青少年精神科或临床心理专家参与；
- 学校心理服务或转介工作流验证；
- 外部数据或回顾性案例；
- 公平性、校准、隐私和人工接管评价；
- 伦理审批和未成年人保护方案。

---

## 5. 最终建议

当前阶段不要冲刺 AAMAS、NAACL、COLING、CHI 或 ICLR。

最稳妥的计划是：

1. 以 **ACL 2027 的2027年1月 ARR周期**作为首个内部目标；
2. 将 Benchmark、专家标注和 ECO-SAFE 实验完整性放在截稿速度之前；
3. 如果 ACL 时间不足，改投 **EMNLP 2027**或等待 **IJCAI 2027**；
4. 完成更完整的情感、关系和多模态实验后，扩展投稿 **IEEE Transactions on Affective Computing**；
5. 有真实专家和学校工作流证据后，再考虑心理健康和数字医疗期刊。

最终优先级：

```text
会议首选：ACL 2027
算法方向：IJCAI 2027 / 后续 AAMAS
安全治理：FAccT
期刊首选：IEEE Transactions on Affective Computing
临床扩展：Artificial Intelligence in Medicine / JMIR Mental Health
高阶临床目标：npj Digital Medicine
```

## 6. 官方信息入口

- [ACL 2027](https://2027.aclweb.org/)
- [AAMAS 2027 Main Track](https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/)
- [NAACL 2027 Main Conference Papers](https://2027.naacl.org/calls/main_conference_papers/)
- [COLING 2027](https://www.aclweb.org/portal/content/32nd-international-conference-computational-linguistics-coling-2027)
- [FAccT 2027 Call for Papers](https://facctconference.org/2027/cfp.html)
- [TACL Submission Guidelines](https://transacl.org/index.php/tacl/about/submissions)
- [IEEE Transactions on Affective Computing](https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers)
- [ACM TiiS Author Guidelines](https://dl.acm.org/journal/tiis/author-guidelines)
- [Artificial Intelligence in Medicine Guide for Authors](https://www.sciencedirect.com/journal/artificial-intelligence-in-medicine/publish/guide-for-authors)
- [JMIR Mental Health专题](https://mental.jmir.org/announcements/726)
- [npj Digital Medicine合成临床数据专题](https://www.nature.com/collections/gggjgihcjj)
