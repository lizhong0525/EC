# CRADLE-Dialogue 本地外部评测协议

日期：2026-10-21。接入版本：`cradle-local-v1.0.0`。
这是外部公开数据的本地探索性测试，不覆盖已有 EcoTriage 数据、Gold、协议或报告。

## 来源与授权

- 原始仓库：<https://huggingface.co/datasets/SungJoo/Cradle-Dialogue>。
- 固定提交：`49d5f40912e9afeadf4d0c1aa0ffd22c8d89c6a3`。
- 直接下载固定提交的 Parquet，与截图 `load_dataset` 所指的是同一仓库；不执行远程代码。
- 保存原始 README、元数据、树及逐文件 SHA-256，Parquet 对照上游 LFS SHA-256。
- 此提交的数据卡与元数据未声明许可证。用户要求下仅作本地研究检查；不据此认定已获
  商用、再分发、衍生集公开或真实未成年人部署许可，发表/共享前须向作者确认授权。
- 只发出公开数据下载请求，不读 HF token、`.env` 或模型配置，不上传材料，不调用模型。
- 数据含自伤、自杀、暴力和虐待描述；本地目录不是对不可信操作系统管理员的安全沙箱。

## 实测结构与保留差异

原始列只有 `turn_id/text/labels`，文档声称的 `dialogue_id` 在文件中缺失。
本工具依赖行序、`turn_id=0` 重置与连续递增恢复边界，角色未知或序列不连续时拒绝
正式适配，不声称恢复了原人物、帖子或剧情模板分组。

- Train：3,058 段、48,557 行，65 行无标准角色前缀。
- Validation：420 段、6,679 行，8 行无标准角色前缀。
- Test：600 段、8,975 行，其中 4,527 用户行、4,448 倾听者行。
- Test 中 121 段由 Listener 开场，原样保留，不删除这些病例或猜测角色。
- Test 共 713 个标签实例，其中 2 个在 Listener 行，保留在原件和异常审计；
  本轮用户节点事件指标采用剩余 711 个实例。不移动标签到相邻用户、不删除原行。
- Train/Validation 是来源方模型合成标签；Test 的临床专家标注身份来自上游声明，
  本项目没有重新审核专家资质，也没有完成独立行动规范标注。
- 精确整段去重不等于语义无泄漏；上游帖子/人物/模板分组与授权仍待作者提供。

## 两条互不混分的测试路径

### 原生危机事件

评估16种原生 Alert/Confirm 类别在精确用户回合的匹配，保持 ongoing/past 和类型区别。
标签是稀疏事件，不是持续状态；空标签不代表此人无风险，不做前向填充。

固定两个本地基线：始终无事件、英文词组首次提及。后者只是透明的弱工程基线，
不处理可靠的否定、人物归属、歧义预警或时间推理，不是临床检测器。
它会漏掉 Alert；不能把其分数称为项目模型分数或论文复现。

主指标为用户回合多标签 Micro-F1，另报逐类和对话集合 F1；零分母为 null，不给满分。
缺失、错误和非法输出保留在分母并单列失败。完整运行报告必须通过全部记录重放。
采用固定种子、1,000次恢复对话聚类 bootstrap；这不是已核实的独立人物/源帖聚类，
存在组间依赖时区间可能偏窄。不作显著性、临床能力或跨系统优劣结论。

### 项目策略迁移诊断

运行原有 always-listen、always-escalate、keyword-triage、history-aware-rule、ECO-SAFE。
不改中文规则来适配测试答案，不自动翻译，不把英语词表偷偷替换进项目策略。
采用既有 v2 当前及历史用户消息约定，不传原 Listener 回复给这些策略；这一上下文
差异在计划中固定，不能将两路径视为相同条件下的模型竞争。

保留真实动作、证据、回复与失败，仅统计分布。原生事件没有可接受动作集合、
转介目标和证据行动标签，所以专家行动评分显示不可用，模拟隐藏状态评分也不可用。
例如 `confirm_SH_past` 不等于“现在必须紧急转介”，严禁建立这种未经验证的映射。
中文回复模板在英文输入上的回复质量未评估，不能作为可用英文产品展示。

## 冻结与复现

每用户节点只含当前及先前可见文本，原始 Listener 文本仅在原生可见前缀保留。
所有派生 ID 随机无语义；原标签与行映射留在单独 private 文件，不传入策略函数。
冻结计划在运行前保存实际输入、原生标签、上下文、源码、环境、预算及统计约定。
每节点零模型调用、零 token、零费用，没有重试和回退。
源码/环境变化须新建计划；文件已存在就拒绝覆盖，旧结果保留。

```powershell
$py = '.research-venv/Scripts/python.exe'
$snapshot = 'artifacts/external/cradle-dialogue/49d5f40912e9afeadf4d0c1aa0ffd22c8d89c6a3'
$packet = 'artifacts/research/cradle-20261021-v1/test-packet'
& $py -m ecocompanion.research cradle-download --output $snapshot
& $py -m ecocompanion.research cradle-audit --snapshot $snapshot --output <新的审计.json>
& $py -m ecocompanion.research cradle-plan --snapshot $snapshot --output $packet
& $py -m ecocompanion.research cradle-run --packet $packet --output <新的运行目录>
& $py -m ecocompanion.research cradle-report --packet $packet --run <运行目录> --output <新的报告.json>
```

下载后的所有处理都可以离线运行；不需要模型或 `datasets` 的隐式在线加载。
重新运行已有快照时跳过 download，使用新的计划和运行目录。原始与派生数据目录
加入 `.gitignore`；本轮没有改动发布打包器，不应将这些目录手工加入发布包。

## 尚未完成

- 作者明确数据许可、原始角色异常/Listener标签解释、源帖/人物/模板组映射。
- 与作者官方推理提示词、代码和事件匹配口径逐项核对后的论文复现。
- 冻结真实模型、英文能力资格测试、调用预算、真实推理及匿名回复人工审核。
- 中文迁移若需要翻译，应建新数据版本并由双语专家审核，不能沿用未审标签。
- 输入匹配的独立行动规范 Gold 与新盲测集、真实世界效果和伦理审批。

本地数据可读取、契约可验证、代码可重放，不等于这些外部科学与授权条件已经完成。
