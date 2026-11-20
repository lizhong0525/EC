# Qwen 证据提取微调与数据登记

日期：2026-10-27。目标是改善本项目已有的 `QwenAssessment` 结构化证据提取，
不是训练疾病诊断器、治疗者或替代危机服务。模型仍受本地规则、Jev 建议校验和
输出审查约束，不能降低规则的安全等级。

## 当前完成范围

- 固定任务 Schema：本人当前线索、证据逐字引用、支持重点、是否需要澄清。
- `training/data/reviewed_examples.jsonl` 是 29 条项目自写的虚构工程样例。
  许可就是本项目源码交接许可，不包含真实个案、用户聊天或外部受控数据。
- 来源登记、逐字引用校验、去重分组、训练/验证/测试组隔离、SFT 数据准备、
  Ollama 基线脚本和 QLoRA 脚本已提供。
- 小样本工程准备成功，不代表临床有效性、人口泛化或危机识别达到上线要求。
  需要更多独立双人审核样例和真实基线/适配器对照。

## 2026-10-27 本机基线

已下载并通过 Ollama 校验 `qwen2.5:7b`。使用与生产一致的系统提示、Schema、
温度和 29 条全部工程样例执行真实本地推理，结果存于本机
`training/results/baseline.json`（不会进入源码 ZIP）：

- 29/29 输出可解析；28/29 的证据引用逐字匹配，1 条虚构引用被本地校验识别。
- `safety_signal` 精确一致 9/29（31.0%）。主要问题为普通表达过度标成 `uncertain`。
- 这不是临床准确率，也不允许用总体数字掩盖当前安全线索漏检；完整场景结果
  与输入 SHA-256 一同保留。当前模型不能被当作疾病筛查或心理判定器。

QLoRA 使用官方 `Qwen/Qwen2.5-7B-Instruct`，Hugging Face 核验 revision 为
`a09a35458c702b33eeacc393d103063234e8bc28`。本机 Hugging Face 下载遇到
SSL/连接中断，改用 ModelScope 官方同名仓库下载到忽略的 `tmp/models/`。
ModelScope 不提供相同 Hugging Face commit 的完整一致性保证，因此本地运行记录
必须保留来源和实际权重哈希，不能把下载成功等同于两个仓库逐字节等价。
7B 下载曾因耗时中止，其缓存不视为完整权重。另行使用官方
`Qwen/Qwen2.5-0.5B-Instruct` 验证 CUDA、4bit 加载、训练与适配器保存；该冒烟
实验不替换 7B，也不作为心理识别效果证据。

实际冒烟已完成：`Qwen2.5-0.5B-Instruct`（ModelScope 官方仓库），CUDA 4bit、
LoRA、17 条训练与 3 条验证、1 epoch、最大长度 512。训练 13.93 秒，
`train_loss=3.3771`、`eval_loss=2.3668`，成功保存约 35 MB 的适配器。
`training/runs/qwen-0.5b-qlora-smoke/RUN.json` 记录依赖、数据哈希、GPU、指标和
适配器 SHA-256，已复核哈希并确认未读测试集。revision 没有固定，因此该运行不是
可复现发布产物；不合并、不导入 Ollama、不部署到应用。

## 2026-10-29 同济 HPC 实际训练

在同济 HPC 的隔离目录 `/share/home/u24121/ecocompanion_qwen_v1` 使用 L40
完成了官方 `Qwen2.5-7B-Instruct` 固定 revision 的四分片 SHA-256 校验与完整加载。
ModelScope 官方 Qwen 仓库用于权重传输，Hugging Face 固定 revision 的 API 元数据
用于哈希校验。下载采用完整临时文件，不再使用镜像断点追加；训练记录见
`scripts/hpc/README.md`。

`v1` 两轮 QLoRA 的训练损失为 2.397，验证损失为 1.337，但 3 条开发样例均未输出
项目完整 Schema。随后 `v2` 独立训练 10 轮，验证损失为 0.389，开发样例 3/3
可解析；不是覆盖旧运行，也没有用用户聊天训练。

冻结的 9 条工程测试在相同 Transformers 自由生成条件下比较：`v2` 为 9/9 JSON
有效、9/9 安全信号一致、8/9 证据逐字落地，未适配基座为 0/9 JSON 有效。基座在
Ollama 的 JSON Schema 约束下则为 9/9 有效、9/9 证据落地、4/9 安全信号一致。
这两种推理条件不同，不能把改善全部解释为泛化能力或微调单独贡献。

`v2` 在 `quotation-001` 中把原文概括为“摘抄歌词”，不是逐字引用，因此没有通过
部署门槛。**两个适配器均未合并、未导入 Ollama、未替换现有 `qwen2.5:7b`。**
本地 `training/runs/` 保存两个适配器包、RUN 记录及对照结果，包内文件哈希已核验。
`v2` 包曾因打包脚本固定读取 `v1` 验证文件而嵌入错误的开发集结果；已在包内登记并
更正为实际 `v2` 开发集结果，未改动适配器权重或训练数据。打包脚本现强制指定
`--validation`，避免把其他运行的结果混入交付物。
应用 `backend/tests/test_pet.py` 36 项通过。此结果只是少量虚构工程样例，不是
临床心理评估验证。测试集已经用于本轮判断，不继续据其调参；下一轮需扩充独立、
审核过的训练/开发数据，并加强证据逐字约束与真实 Ollama 适配器回归。

## 权威来源的实际核验

以下仅登记来源及其当前适用性。访问日期均为 2026-10-27；没有因为“公开可访问”
就把数据加入训练。指南不是可自由抓取的训练语料，量表分数也不是本任务标签。

| 来源 | 官方入口与本次事实 | 当前处理 |
| --- | --- | --- |
| PsyDial / PsyDial-D4 | [作者仓库](https://github.com/qiuhuachuan/PsyDial)、[作者数据卡](https://huggingface.co/datasets/qiuhuachuan/PsyDial-D4)、[ACL 2025 论文](https://aclanthology.org/2025.acl-long.1049/)。API revision `0ceef5ac19fb8174baa6be31ce1b2637711cb4c2`，数据卡 Apache-2.0、中文、2,382 段。作者说明为真实长程咨询重建的半真实数据。 | 可作为非临床支持对话候选。尚未导入训练。需保留数据版本，核查底层来源、隐私、相关对话分组及既有数据重合。代码仓库许可不能替代数据许可。 |
| ESConv | [清华 CoAI 官方仓库](https://github.com/thu-coai/Emotional-Support-Conversation)、[许可](https://github.com/thu-coai/Emotional-Support-Conversation/blob/main/LICENSE)、[论文](https://arxiv.org/abs/2106.01144)。README 声明学术研究使用，LICENSE 为 CC BY-NC 4.0。 | 英文参与者情绪支持材料，限非商业且需署名。未导入。策略标签不能当成本项目安全信号或诊断标签。 |
| DAIC-WOZ | [USC ICT 官方页](https://dcapswoz.ict.usc.edu/)。包含访谈转写、音视频等；页面要求学术邮箱申请，因知情同意仅向学术及其他非营利研究者分发。 | 受控访问，未申请、未下载、未训练。临床访谈严重度材料不能直接转写为本任务 `current` 标签。 |
| WHO 心理支持指南 | [WHO 出版物入口](https://www.who.int/publications/i/item/9789241548205)。用于支持边界与术语的参考入口。 | 不是模型诊断标签库。引用/再利用须逐项遵守版权和许可，不批量抓取作为训练数据。 |

官方来源只说明数据出处和条款，不证明模型适合用户所在地区、语言或用途。
真实危机提示的部署还需要本地紧急资源运营、人工复核、测试与责任边界。

## 数据进入规则

每条训练样本必须含 `source_id`、`group_id`、审核人、审核状态、场景和符合
`QwenAssessment` 的目标。来源必须 `training_allowed: true`，并填写许可入口和
复核日期。相同原话或改写必须共享分组；禁止从同一对话切不同段落分别训练和测试。
本人/他人、当前/过去、否定、隐喻、文学引用、身体不适、工作压力、当前安全线索
均需专项覆盖。测试分组不得参与调参；报告必须披露哪些场景没有独立留出样本。

用户聊天、渠道消息、未审核 AI 生成数据默认禁止进入训练集。外部数据放在项目外或
被忽略的数据目录，不把受控数据和真实个案打入 ZIP。

## 本地执行

```powershell
.\Install-LocalQwen.ps1
$env:PYTHONPATH = "backend/src"
.\.research-venv\Scripts\python.exe scripts\prepare_qwen_training.py
.\.research-venv\Scripts\python.exe scripts\qwen_baseline.py --model qwen2.5:7b --output training/results/baseline.json
```

基线应在微调前冻结。后续从官方未量化 `Qwen/Qwen2.5-7B-Instruct` 权重执行小型
QLoRA 实验（约 16GB 模型下载，8GB 显存仍需以实测为准）：

```powershell
.\.research-venv\Scripts\python.exe scripts\train_qwen_qlora.py --model tmp/models/qwen2.5-7b-instruct --output training/runs/qwen-evidence-v1
```

该脚本只读准备好的 train/validation，不读 test，也不接受用户聊天路径。
对量化权重启用 QLoRA 准备和梯度检查点，用户原文与系统提示共同参与模板编码，
只学习 assistant 目标 token。需先完成短上下文/小模型冒烟，再运行 7B 实验。
`--model` 可指定已核验的本地权重目录。必须记录基础模型 revision、依赖、GPU、
训练数据哈希、配置和输出适配器哈希。训练后合并与量化导入 Ollama，再以完全相同的
留出集、系统提示、Schema 和温度运行基线对照；任何证据虚构或安全回退都阻止替换
现有模型。适配器或量化权重不放入源码交付 ZIP。

## 隐私、误判与删除

- 报告是用户主动记录，不展示疾病概率或诊断；1-5 分不是标准量表。
- 报告 AI 摘要逐次同意，只发送所选本人原话至本机 Qwen，不调用云端模型。
- 删除报告删除其快照和引用；删除会话会移除相关报告原话及非用户摘要。
- 微调适配器不是对原对话的可靠删除机制，因此绝不能用真实用户聊天直接试验。

## 来源与许可核验记录

| 项目 | 实际状态 |
| --- | --- |
| PsyDial-D4 | 2026-10-27 访问作者 Hugging Face API/数据卡，revision 与 Apache-2.0 已登记；未导入。 |
| ESConv | 2026-10-27 访问作者 GitHub 仓库与 LICENSE，CC BY-NC 4.0；未导入。 |
| DAIC-WOZ | 2026-10-27 访问 USC ICT 官方页，需受控申请；未申请、未下载。 |
| WHO | 2026-10-27 HTTP 200 核验《Psychological first aid: Guide for field workers》出版页；未抓取为训练语料。 |
| Qwen 基座 | Hugging Face API 核验 commit；Ollama 权重已完成；7B 未量化权重下载未完成，0.5B 官方 ModelScope 权重完成。 |

## 2026-10-29 v3 外部数据再训练

首次自动关键词候选没有直接用于训练。逐条检查发现“压力减轻”“担心”的
否定语境和他人身体感受容易被误标，因此改为固定对话/轮次的入选清单。
最终 100 条数据含 39 条逐条审阅的 PsyDial-D4 来访者消息、32 条新边界样例
及原 29 条边界样例；83 条用于训练、17 条用于开发，没有新测试样本。
外部数据只采用当前消息原文，不把咨询师回复、量表或疾病词转换成安全标签。
每条证据均通过逐字子串检查，外部样本以完整对话为单位隔离。

来源为作者发布的 ACL 2025 `PsyDial-D4`，固定 revision 和 SHA-256、逐样本
对话/轮次、Apache-2.0 数据卡与论文引用均记录在 `training/data/v3/`。
原始 35.5 MB 整库位于被忽略的 `training/external/`，不进入源码交付 ZIP。
CPsyCoun 已核验作者数据卡 CC BY-SA-4.0，但未导入；本轮没有混入多来源数据。
这些标签仍是工程支持需求标注，不是心理专家标注、临床结论或人群效度证据。

### v3/v4 实际结果与停止点

`v3` 在同济 A800 上完成 6 轮 QLoRA（83 条训练、17 条开发），纯训练 226.5
秒。开发集 17/17 JSON 有效，但证据与安全标签各 16/17：一处轻微改写证据，
一处将普通强烈恐慌标成 `uncertain`。依赖作业据此自动拒绝冻结回归。

随后只进行一轮基于开发问题的修正。`v4` 新增 12 个未使用的完整来源对话
和 8 个独立边界组，共 120 条（97 训练、23 开发），同样 6 轮，A800 纯训练
264.5 秒。开发集达到 23/23 JSON 有效、23/23 证据落地、23/23 安全标签一致。
这不是用冻结集调参；修正脚本不读取冻结测试或其结果。

最后运行的同一冻结回归中，`v4` 为 9/9 JSON 有效、9/9 安全标签一致、
8/9 证据落地。当前意图样例中，原文“已经准备好了药”被生成成
“我已经准备好了药”，多出原文没有的“我”。因此 `v3` 与 `v4` **均不部署**，
现有 `qwen2.5:7b` 不变，不合并、不导入 Ollama，也不称为已验证心理模型。

两个适配器包均已下载，7 个列入清单的适配器文件哈希通过；训练与验证记录
保存在 `training/runs/` 和 `training/results/`。HPC 无遗留作业，应用相关
41 项测试通过。后续需要独立的新训练/开发数据加强逐字跨度抽取，并建立新的
留出集和量化 Ollama 回归，不能继续针对上述 9 条样例修补。
