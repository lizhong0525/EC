# 回复渲染蒸馏执行记录

日期：2026-10-25。阶段：离线准备完成，等待安全配置教师凭据。
本记录不修改旧 Benchmark 阈值、旧 30 题评分或历史实验结论。

## 本轮路线

```text
新编虚构成人场景 → 先划分数据 → 固定本地策略与可见输入
    → DeepSeek 非思考模式生成草稿 → 自动筛查
    → 人工逐条复核 → 仅导出批准的 train 样本
    → 独立训练环境做 LoRA/SFT → 四条件对照与安全回归
```

仅学习自然语言表达，不把风险等级、动作选择或危机升级交给学生学习。
本轮先验证小规模数据流水线，不把 8 条草稿当成足够训练专用模型的数据量。
不采集教师隐式思维链，不上传真实患者资料或既有盲测材料。

## 已落地

- 实现入口：`scripts/distillation.py`，包含 doctor、freeze-sources、prepare、
  generate、export；没有训练命令，不会下载模型或更新权重。
- 新编种子：`docs/DISTILLATION_PILOT_SEEDS_V1.json`，24 条虚构成人场景。
  八类为寒暄、日常分享、实际困难、情绪支持、身体不适、误解修复、
  关系边界、话题切换。首轮排除危机、自伤及本地 L2/L3/L4 场景。
- 数据包：`artifacts/research/distillation-20261025-v1/packet/`，
  train/dev/heldout 各 8 条；教师调用只选 train。
- 冻结真实 Web renderer 输入，使用
  `web-conversation-policy-v2.0.0` / `web-response-renderer-v3.0.0`。
  输入不包含 split、category、group、未来节点、专家答案或 Judge 分数。
- 哈希校验覆盖输入和官方资料；调用时还会重建 renderer 输入并检查一致性。
  Windows 文本换行导致的快照哈希问题已修复并补回归测试。
- 教师返回失败、拒答、截断及自动拒绝均保留状态；异常不记录响应体或
  可能含凭据的异常字符串；响应元数据也过滤凭据。
- 导出必须有 `APPROVE`、匹配的候选摘要、reviewer_id 和 review_reference，
  并重新通过自动关卡。人工身份只是整理者声明，不是系统认证的专家资格。

## 官方条款与价格

本次真实抓取并冻结在
`artifacts/research/distillation-20261025-v1/sources-r3/`：

1. [用户协议](https://cdn.deepseek.com/policies/zh-CN/deepseek-terms-of-use.html)：
   更新及生效日期均为 2025-09-05。第 4.2 条明确列出
   “训练其他模型（如模型蒸馏）”；仍须满足法律、其他协议和第 3.6 条等限制。
   这不是医疗、心理高风险用途的全面授权或法律意见。
2. [模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)：
   选择 `deepseek-flash`，页面记载版本 `DeepSeek-V4.1-Flash-1017`。
   别名并非不可变检查点，实际返回的模型名称与指纹会另存。
3. [思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)：
   显式发送 `thinking.type=disabled`，只提取回复正文。

按高峰、缓存未命中输入 2 元/百万 tokens，输出 8 元/百万 tokens 预留。
最多 8 次请求，每次输出最多 384 tokens，输入保守预留 16,384 tokens，
零重试，整轮预留 0.280576 元，计划预算 1 元。
这是应用层估算，不是服务商强制扣费上限；超时也可能产生未知费用，
用量缺失或调用契约异常时停止并保留未知成本标志。

早期 `sources/` 抓取因非规范 URL 重定向中断，`sources-r2/` 有换行哈希问题；
两者原样保留，正式 packet 只引用校验通过的 `sources-r3/`。

## 数据隔离及其限制

与 `artifacts/research/judge-diagnostic-v2/cases.json` 的旧 30 题做离线检查：
24 条新场景及其历史共 36 段文本，对比旧题 60 段用户/助手文本，
按 Unicode NFKC 与去空白规范化后的精确重合为 0。
可复核摘要见 `artifacts/research/distillation-20261025-v1/dedup-check.json`。

精确去重不等于语义独立。当前分组是新编情节的作者声明，
未经过独立语义关联组审核，各分区仍共享八类任务及 renderer 模板。
因此 heldout 只是开发留出，不是独立盲测，更不是专家 Gold 或临床验证。
扩大数据规模前必须完成语义近重复与来源关联组复核。

## 当前环境与真实执行量

- RTX 5060 Laptop，8151 MiB 显存，预检时约 5948 MiB 空闲。
- 磁盘预检：C 盘约 13.60 GiB、D 盘约 20.94 GiB 空闲；
  未预留底座、缓存及检查点空间，不能据此直接启动 7B 训练。
- 已发现 torch、transformers、peft、datasets、bitsandbytes；未发现 TRL。
  包存在不等于已经验证 GPU/CUDA/量化训练兼容性。
- 没有安装依赖、下载 Transformers 底座或建立训练环境。
  已有 Ollama GGUF 仅用于推理，不作为常规 LoRA 训练底座。
- 本进程 `DEEPSEEK_API_KEY` 缺失；真实生成入口在发送前停止，
  未创建 teacher 目录。这不是 API 鉴权失败，没有到达 API。
- **真实教师请求 0；人工批准样本 0；训练导出 0；训练权重 0。**
- 网络隔离的流水线测试 16 passed；模拟教师和模拟批准只出现在测试临时目录，
  不计作真实教师数据、人工标签或训练产物。
- 最终完整后端回归：697 passed / 1 skipped / 0 failed，381.44 秒。
  跳过项仍为 Windows 符号链接权限限制；未修改旧 Benchmark 测试断言。
  Ruff、格式检查、脚本 Mypy 通过；本轮 25 个实现/文档/快照文件凭据模式扫描 clean。

## 安全启动

先撤销已经贴入聊天的密钥，再创建替换密钥。不要把替换密钥粘贴到聊天、
脚本、命令行字面量或 `.env`。在项目目录打开自己的 PowerShell，
运行以下命令后，在隐藏输入提示中输入替换密钥：

```powershell
$previousKey = $env:DEEPSEEK_API_KEY
$secret = Read-Host "DeepSeek replacement API key" -AsSecureString
try {
    $credential = [System.Management.Automation.PSCredential]::new("deepseek", $secret)
    $env:DEEPSEEK_API_KEY = $credential.GetNetworkCredential().Password
    .research-venv/Scripts/python.exe -m scripts.distillation generate `
      --packet artifacts/research/distillation-20261025-v1/packet `
      --output artifacts/research/distillation-20261025-v1/teacher `
      --limit 8 --allow-paid
} finally {
    $env:DEEPSEEK_API_KEY = $previousKey
    $secret.Dispose()
    Remove-Variable credential, secret, previousKey -ErrorAction SilentlyContinue
}
```

这会实际付费调用，不启动训练。不要重复执行到不同目录以绕过首轮 8 次限制。
若发生错误，先查看保留的状态及未知费用标志，不自动重跑。

成功后，`teacher/review.csv` 默认全部为 PENDING。
复核应检查具体回应、自然度、边界、身体不适处理、历史误解修复和固定动作遵从；
不得以教师自评或自动过关代替人工复核。填入批准决定及可追溯审核依据后，
才可运行 export。不得让系统替真人填写“已审核”。

## 后续进入训练的条件

表达评估设计已先行锁定，见 [冻结评估规范](RENDERER_EVALUATION_V1.md)。
新的独立评估集不可从本轮 8 条 heldout 改名取得；
当前独立病例与真人标签均为 0，取得材料和运行评估仍有明确门禁。

先完成人工小批评审，再决定是否扩大 train 数据；dev/heldout 不做梯度更新。
单独建立训练环境，验证新显卡、CUDA 和量化依赖兼容后再选底座大小与训练框架。
比较原学生、仅改 Prompt、蒸馏学生、教师四种条件，
统计失败、截断、重复、安全动作遵从、延迟与成本，并进行独立人工盲评。
最终量化产物也必须重跑冻结安全回归；保留原推理模型用于回滚。
