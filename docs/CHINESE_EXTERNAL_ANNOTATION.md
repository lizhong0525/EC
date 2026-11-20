# 中文外部材料试标操作说明

手册版本：`chinese-external-v1.0.0-pilot`；日期：2026-10-21。
状态：待专家审查与来源隐私复核。不能把本文件当作临床操作指南。

## 本批材料是什么

本批从 CPsyCoun 的固定公开 HF 版本中选出20段重建对话，作为开发试标材料。
按对话长度四分位轮选，各5例；不声称按风险、学派或年龄分层。
先按规范化开场用户文本分组，同组只选一例；这不等于已知独立人物。
材料不是原始临床记录，年龄未知，不能直接称青少年病例。
未知人物、骨架、模板族采用保守的共同 UNKNOWN 分组，不虚构独立性。

本轮展示的是**当前及以前的用户表达，不含咨询师回复**。这是为匹配既有策略输入
契约的有损视图，不是完整咨询对话。用户可能在回答被隐藏的问题；遇到不连贯处，
记录缺失信息或 Unresolved，不从原始完整记录补答案，不假定是用户症状。
如果专家认为删去咨询师文本使规范判断不可行，应结束该批的适用性审查、保留记录，
另建“完整可见历史”版本，重新冻结输入和标注；不能把原标签直接挪过去。

`turn:n` 是文本先后顺序，不是天数、持续时长或真实时间。
结构中兼容旧 Schema 的 STUDENT_SIMULATOR 角色名不证明材料人物是学生。
没有社会生态资料时不推断家庭/学校条件，不把学校或可信任成人作为默认可用对象。

## 管理员先做

1. 核对来源、去标识化情况、敏感材料处理和实际授权。HF 为 CC BY-SA 4.0，
   与 GitHub 的 CC BY 4.0 不同，未澄清前不公开衍生材料。原报告申请不在本轮范围。
2. 在受控本机审阅20例，确认有损视图是否适合试标。当前状态仍为待审，
   生成任务包不等于审查完成。涉及机构审批/同意的条件由机构确认。
3. 按 `ANNOTATION_OPERATIONS_V2.md` 核验并登记三位不同人员的真实身份、资格与培训。
   不具备专家时保持空库，不调用 attest，不编造资质或判断。
4. 只向专家开放角色隔离表单；不能把整个任务包、源码、private 或完整对话发给专家。
   private/access.json 由管理员私下分发，不能写入聊天、URL、截图或公开材料。

```powershell
$py = '.research-venv/Scripts/python.exe'
$packet = 'artifacts/research/chinese-20261021-v1/cpsycoun-annotation'
& $py -m ecocompanion.research serve --packet $packet --port 8766
& $py -m ecocompanion.research audit --packet $packet
```

服务只监听回环地址，不直接开放公网。旧8765任务不改变，使用前检查8766是否占用。
本地角色隔离不是对操作系统管理员的保密系统。

## 专家怎样标

沿用 `ANNOTATION_OPERATIONS_V2.md` 的逐节点表单、L0-L4/B、动作和证据码操作定义，
但要先审查这些定义是否适合年龄未知的外部材料。手册/码表实质调整须新版本新批次。

两位专家独立查看当前节点和已出现的用户表达，填写证据引用、缺失信息、
行动等级、可接受/优选/禁止动作、必要安全询问、转介对象、置信度和理由。
置信度是判断把握，不是患病概率。信息不足不等于安全，保留待澄清事项。
不要将原作者咨询师回复、原始分类、模型预标注或未来剧情当作答案。

每次提交后锁定，才能进入下一节点。原始独立判断不能回写；
误填采用新的可追踪纠错批次，不编辑数据库。第三人仅在双标完成后裁决。
有序等级统计不把 B 排在 L4 之后；类别不足、未双标或分歧未解决都不能声称达标。
不得删除难例来提高一致性。

```powershell
& $py -m ecocompanion.research agreement --packet $packet --output <新的一致性.json>
& $py -m ecocompanion.research label-status --packet $packet --output <新的标签状态.json>
& $py -m ecocompanion.research freeze-gold --packet $packet --output <该包private中的新Gold.json>
```

Gold 命令必须通过真实资质记录、双标、一致性和必要裁决门禁。目前没有这些结果。
前瞻时间窗由已锁定节点判断推导；回顾性窗口只追加独立事件，不覆盖前瞻标签。

## 标完后怎样进入评分

本包提供兼容既有 v2 的 protocol.json，可用既有 run-local、report 及显式 --gold
加载经门禁冻结的标签。不需要补造 simulation-truth.json。
该兼容入口沿用既有四种规则基线，不包含 zh run 中额外的 eco-safe；
两条运行的协议摘要和输出文件不同，不能混用。
未有 Gold 时规范指标保持不可用；外部真实结局、模拟隐藏状态始终没有参考答案。
本轮 zh run 的动作分布是独立诊断报告，不自动变成 Gold 成绩。

```powershell
& $py -m ecocompanion.research run-local --packet $packet --output <新的规则运行目录>
& $py -m ecocompanion.research report --packet $packet --predictions <该运行的predictions.jsonl> --gold <实际冻结Gold.json> --output <新的Gold报告.json>
```

20例只用于流程试标。已知人物/来源独立性不足，不将病例级区间解释为独立人物总体
区间，不作临床结论或真实模型优势声明。

## 与其他任务隔离

PsyCrisis 的匿名回复盲评与800条原作者评分校准是另外的任务，不能发给本轮
行动规范标注者作为预标答案。其原作者评分不是本项目新专家 Gold。
回复评审遵循 reply-v2，严重违规需人工复核，LLM Judge 不能替代独立专家。
本轮不运行任何 Judge，也不生成主观评分。
