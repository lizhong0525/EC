# 中文外部数据本地接入协议

协议版本：`chinese-external-v1.0.0`；日期：2026-10-21。
本轮是公开开发材料的探索性工程诊断，不是预注册主实验或独立盲测。
旧数据、协议、标签与报告不改写。

## 优先级与来源

第一优先 PsyCrisis-Bench：检查中文高风险输入上的本地规则行为，为回复盲评做准备。
第二优先 CPsyCoun：抽取20例多轮开发材料，接入逐节点独立专家流程。
PsyDial-D4 后续再审；SOS-HL-1K 许可及 CPsDD 下载/研究授权未确认前不导入。

### PsyCrisis

- 来源仓库：mental-health-llm-safety-eval/psycrisis-bench。
- 固定提交：56a862aa9a5209b6836c17c2697ab4f29e644535。
- 仅下载 README、LICENSE、608输入工作簿及800回复评分工作簿。
- 上游 LICENSE/README 声明 CC BY-NC 4.0、非商业学术研究使用。
  不据上游陈述假定已独立核验全部底层来源授权、隐私或研究伦理。
- 608条输入：自杀类289、自伤类115、存在性困扰204。全部是危机相关材料，
  规范化后607个唯一输入；保留原始608行，不称608个独立人物。
  没有一般低风险对照，不能估计一般人群误报率。
- 800条评分实际上是400个不同输入，每输入两条回复；其中398输入与608条集合重合。
  两份文件不能当作独立验证集，800也不是800个独立病例。
- 只保留五个二元维度：共情/empathy、建议/regulation、担忧/exploration、
  风险评估/risk_inquiry、外部干预/external_support；英文名是适配器字段，
  不自动扩张原表评分定义。评分标准需专家逐项核对。
- 上游程序会调用远程 Judge，本轮不下载或执行它；不复现或认同未经核对的
  0-5安全总分，也不根据关键词自动给上述主观维度分数。
- 原作者评分没有工作簿内的逐人资质、独立原始双标和分歧记录，
  仅称“上游提供的人类评分”，不是本项目行动 Gold。

### CPsyCoun

- 本轮用作者 HF 格式化数据 CAS-SIAT-XinHai/CPsyCoun，
  固定提交 8fe0fa0f77630b921932bdaa6bae7481bc611cf2。
- JSON 实际3134段、24443个用户节点；另一个 GitHub 提交的逐文件目录是3084，
  不静默混用，也不混入独立的 CPsyCounE。
- 每条 history 是用户/咨询师对；instruction/output 是末对；input 必须为空。
  不识别的新字段/格式应拒绝，不能丢掉后宣称正确适配。
- 不申请原始咨询报告。公开材料是重建对话，不是原始临床记录，不推断年龄。
- HF 卡声明 CC BY-SA 4.0，GitHub 声明 CC BY 4.0。保留冲突，
  依据实际下载源记录条款，发表/共享前向作者澄清；本轮不公开材料。
- 没有逐例学派/学校/人物/模板标签，不能把介绍中的统计当逐例元数据。
- 实测有3条重复用户轨迹、37个被多段共享的规范化开场组；保留审计，
  不通过换ID声称独立。

## 完整性和版本

下载只请求公开固定版本，不读取 token、API 配置或 .env，不执行远程代码。
HF 对照文件 Git blob/LFS身份与大小；保存元数据、目录树和逐文件 SHA-256。
若 GitHub API 403/429，本工具明确降为固定 raw HTTPS 地址+本地 SHA-256，
清单标记 NOT_UPSTREAM_TREE_VERIFIED，不能称上游树校验已通过。
其他错误不冒充限流成功。全部文件下载成功后才写快照目录。
这些摘要能检查后续意外改动，不是来源合法性或管理员不可篡改的证明。

工作簿只读取值，拒绝公式、宏、外部工作簿链接、额外工作表和未知列。
所有计划、运行、评分、任务包拒绝覆盖已有输出。源码/实际运行环境改变要冻结新计划，
不能更新旧摘要“修复”重放。文件/ID映射保存在 private；它不是硬盘加密或ACL系统。

## 本地输入和运行

派生病例/节点使用随机不透明ID，映射稳定保存在本次包中。
策略只接收当前及先前用户表达；源分类、原作者回复、行号、未来剧情、划分不传入。
保留既有 v2 Schema，未知年龄写 unspecified，语言写 zh-CN。
turn:n 只表顺序，不暗示真实时间。CP输入隐藏了咨询师问题，是有损视图，
专家适用性审查若不通过，应建新版本，不把不连贯误判为用户风险。

冻结既有5种本地策略：always-listen、always-escalate、keyword-triage、
history-aware-rule、eco-safe。零模型调用、零token、零费用、零重试、无跨模型回退。
不为对齐外部标签调整规则。本地模板不能代表真实 LLM。
输出保留执行错误/契约违规；报告前完整重放每条输出。

只报告动作、证据和契约计数，不报告临床诊断准确率或由分类硬映射的转介正确率。
没有输入匹配行动 Gold 时，风险/动作/证据/时窗规范指标不可用。
这些数据不提供模拟隐藏状态，模拟状态恢复和真实世界结局不可用。
人物/来源依赖未知，不构造有独立总体含义的置信区间，不作显著性比较。

## 校准与人工流程

800回复任务按规范化用户文本分400组，SHA-256排序后100组作公开留出、300组校准。
同输入的两个回复始终在同一分区。留出分组可从公开资料重建，不是保密盲测。
任务中不展示参考分数、来源行、模型身份或分区；匿名ID与随机呈现顺序持久化，
可比对任务摘要。不声称回复风格无法猜测身份。

calibrate 只导入实际发生的外部评审，不调用 Judge。每行要求 candidate_id、status、
scores（恰好五个0/1整数，不能用bool）、judge_id、verification_reference。
身份/引用是外部记录定位，不是工具已验证评审身份的证明。
重复或未知ID拒绝；缺失/格式错输出留在准确率分母及正例召回分母。
逐维报告准确率、P/R/F1和有效对 kappa，缺少类别变化时 kappa不可用；
缺失输出下的有效对kappa不得包装成全样本结果。不计算总安全分和显著性。
目前没有真实Judge预测文件，也不拿参考答案复制成假Judge实验。

CPsyCoun 20例只创建空标注库，操作见 CHINESE_EXTERNAL_ANNOTATION.md。
独立A/B、逐节点锁定、C裁决、证据与理由、资质引用和Gold门禁复用现有流程。
原始双标不能覆盖，未具备真实专家就保持待完成。
PsyCrisis真实来源输入不强行伪装成仅接受合成数据的 ResearchCase。
其匿名回复任务直接采用既有reply-v2格式，与CP行动规范标注人员/材料隔离。

## 本机命令

在项目根目录的现有 .research-venv 使用。新环境需显式安装 `.[external-data]`；
本轮仅新增可选依赖声明，没有改写原依赖锁。实际环境 inventory 随计划保存。

```powershell
$py = '.research-venv/Scripts/python.exe'
$root = 'artifacts/research/chinese-20261021-v1'
$psy = 'artifacts/external/psycrisis/56a862aa9a5209b6836c17c2697ab4f29e644535'
$cp = 'artifacts/external/cpsycoun/8fe0fa0f77630b921932bdaa6bae7481bc611cf2'
& $py -m ecocompanion.research zh --help
& $py -m ecocompanion.research zh download psycrisis --output $psy
& $py -m ecocompanion.research zh download cpsycoun --output $cp
& $py -m ecocompanion.research zh plan --snapshot $psy --output "$root/psycrisis-packet"
& $py -m ecocompanion.research zh run --packet "$root/psycrisis-packet" --output "$root/psycrisis-run"
& $py -m ecocompanion.research zh calibration-packet --snapshot $psy --output "$root/psycrisis-calibration"
& $py -m ecocompanion.research zh reply-packet --packet "$root/psycrisis-packet" --run "$root/psycrisis-run" --output "$root/psycrisis-replies"
& $py -m ecocompanion.research zh plan --snapshot $cp --output "$root/cpsycoun-packet"
& $py -m ecocompanion.research zh run --packet "$root/cpsycoun-packet" --output "$root/cpsycoun-run"
& $py -m ecocompanion.research zh annotation-packet --packet "$root/cpsycoun-packet" --output "$root/cpsycoun-annotation"
```

已下载时跳过download；所有已存在路径必须改用新版本，不能删除历史产物。
运行报告可用 zh report 在新路径重新生成，并再次全量重放。
下载后其余步骤可离线，所有原始/派生目录忽略提交，不自动进入发布包。

## 仍需外部完成

- 来源/隐私与重建材料适用性审查，许可证冲突、源帖/人物/模板分组澄清。
- 三名真实专家、手册审查培训、完整双标和裁决、输入匹配Gold。
- 真实评审/Judge校准、严重违规人工复核，而非本轮不存在的自动分数。
- 冻结真实模型身份/提示词/调用预算并获授权后执行的新实验。
- 独立低风险对照和新盲测集；伦理/同意及真实世界部署安全性另行研究。
