# 本地 Judge 安装与真实校准记录

日期：2026-10-22。记录版本：`local-judge-session-v1.0.0`。
用户已明确允许安装本地推理运行时及下载模型。
本记录是新实验，不覆盖 2026-10-21 的 NOT_RUN 或 2026-10-22 的离线审计结果。

## 固定计划

- 运行时：官方 Ollama `v0.40.8`，Windows amd64 独立包。
- 模型：`qwen2.5:7b-instruct-q4_K_M`，不是云模型。
- 模型 manifest SHA-256：
  `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`。
- 主权重 SHA-256：
  `2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730`。
- 运行时 ZIP SHA-256：
  `04d1d43ac62c2deb0393307bb49fed7caf6b192de41bdd497aaa9488089d5552`。
- 运行时与模型存储在 `D:\EcoCompanionRuntime`，不占用系统盘存放大权重。
- 仅监听 `127.0.0.1:11434`；禁用云功能、环境代理及远程模型回退。
- 并行推理数 1，最多加载模型数 1，Flash Attention 开启，KV cache 为 `q8_0`。
  缓存精度是明确的实验配置，不等价于默认 f16 运行；不声称质量无损。
- 沿用冻结 Prompt、temperature 0、seed 20261021、
  num_predict 1200、num_ctx 16384、每题一次调用、零重试。
- 先做独立通用输入加载检查；额外调用另记，不计入校准或留出。
- 再用既有 `scripts.p0_validation` 依次运行 600 条校准和 200 条公开留出。
  不选择部分结果、不改阈值、不把公开留出称为独立保密盲测。

## 安装机制与安全边界

`scripts.local_runtime_download` 只包含两个固定公开资产的地址、字节数和摘要，
不接受任意来源，不执行下载的脚本。严格检查 HTTP 范围响应，
合并后必须满足完整 SHA-256 才将文件提升为最终资产。
分段缓存只按大小复用，但不跳过最后的完整摘要校验。

初次普通网络路径发生超时、连接失败和范围不匹配；随后保留完成分段，
以禁用环境代理的直接公开 HTTPS 继续下载。下载重试与模型推理重试分开，
前者最多三次/分段并留痕，后者仍是每题零重试。
所有缓存仅作下载材料，不能当作已验证运行时或模型。

`scripts/start_local_judge.ps1` 是显式启动入口，不注册系统服务、开机启动或
改写机器级环境；后台窗口隐藏，配置、进程号和日志写入新目录。
没有读取项目密钥文件或使用远程模型；不能据此宣称整个操作系统无任何遥测。
不读取或输出运行时自行管理的身份密钥。

## 当前执行状态

主权重 4,683,073,952 字节及全部辅助 blob、manifest 已下载并通过摘要校验。
运行时 ZIP 1,582,537,058 字节也已通过摘要，归档路径检查通过并解压。
`ollama.exe` Authenticode 为 Valid，签名者为 Ollama Inc.。
服务实际版本为 0.40.8，仅监听 127.0.0.1:11434。
运行时可执行文件 SHA-256：
`034a26a71836df581ef8b9ab4db489154f548c81b53aa8816d0e843b9e385c2b`。
下载器 12 项、通用加载检查器 4 项网络隔离夹具测试共 16 项通过，
两个 Python 脚本 Ruff 和 mypy 检查通过。
完整后端回归：535 passed、2 failed、0 skipped，352.41 秒。
两项失败仍为既有 Windows 符号链接权限和 POSIX0600 断言，未跳过或改成通过。
对应 JUnit：`artifacts/research/local-judge-20261022-v1/tests-full.xml`。

GitHub 默认入口 IP 后续连接超时；使用另一入口 IP 取得官方 302 响应，
TLS 主机名仍是 `github.com`，没有关闭证书验证。
后续仅允许 `https://release-assets.githubusercontent.com/github-production-release-asset/`
路径继续分段下载，最终仍使用同一 ZIP 字节数和 SHA-256。
该只读下载路由不改变模型或评测协议，公开临时下载查询不输出到报告。

## 真实调用

独立通用加载检查已完成：1 次调用、零重试，约 108.16 秒，
包括首次 GPU 加载与预热；返回符合五维 JSON 契约。
`/api/ps` 显示该固定摘要模型以 16,384 上下文运行，
`size_vram = size = 5,032,304,966` 字节；这不是速度或临床性能保证。
此输入是独立通用测试，不使用校准或留出材料，不计算准确率。
实际记录：`artifacts/research/local-judge-20261022-v1/smoke/result.json`。

正式校准已运行并按原失败停止规则结束。实际命令：

```powershell
.research-venv/Scripts/python.exe -u -X utf8 -m scripts.p0_validation run `
  --model qwen2.5:7b-instruct-q4_K_M `
  --output artifacts/research/local-judge-20261022-v1/experiment
```

没有提供 `--include-project`，不会自动对项目回复打分。
校准不完整时按原流程停止；不得隐藏失败、续跑覆盖或改写历史分母。
完整请求、回复、token、延迟和错误记录在各阶段 `records/` 下。

## 实际校准结果

状态：`STOPPED_INCOMPLETE_PHASE_NO_RETRY`。
正式调用 414 次：413 条符合输出契约，1 条结构失败；剩余 186 条未调用。
有效覆盖为 `413/600 = 68.83%`。失败与未调用的 187 条全部保留在 600 条分母中。
200 条公开留出未运行，项目回复评审未运行，未重试、换模型或续写旧目录。

失败记录 ID：`87caf4f8623447eb8dd390975a9514aa`。
服务器返回 `done=true`、`done_reason=stop`，不是本次记录中的超时或 token 上限。
JSON 语法可解析，但 `human_review_flags` 被嵌套进 `reasons`，
导致外层缺少必需字段且理由字段集合错误，严格 Schema 检查抛出 ValueError。
本轮没有自动挪字段、修复答案或把无效回复记作有效。
最初进度更新笼统称“无效 JSON”，此处明确更正为“JSON 字段层级违反契约”。

以下是保留失败/缺失分母的**未完成运行诊断**，不是完整校准性能：

| 维度 | 正例 F1 | 正例召回 | 已观察假阳性 FP |
|---|---:|---:|---:|
| 具体共情 | 0.6807 | 0.6317 | 92 |
| 具体调节建议 | 0.6208 | 0.4796 | 24 |
| 开放探索 | 0.3761 | 0.6058 | 168 |
| 直接风险询问 | 0.1189 | 0.8095 | 248 |
| 现实外部支持 | 0.7721 | 0.6619 | 26 |

指标衡量与源人工“回复要素”标签的相符程度，不是患病识别或临床安全。
已观察到的分歧并不都由中途停止造成，例如风险询问有 248 个 FP，
说明不能只解决 JSON 层级问题就声称该 Judge 合格。
不对未运行部分做性能外推，也不以有效前缀替换全部分母生成通过声明。
383 条有效记录带有模型提出的人工复核标志；它们不是 383 个已确认违规，
尚无合格专业人员完成复核，不能据此生成 Gold。

额外通用加载检查 1 次，加正式推理 414 次，共 **415 次真实生成调用**。
正式成功调用的输入/输出 token 为 273,748 / 77,708。
失败回复保存了服务报告的 618 / 184 token；加载检查为 308 / 167 token。
原冻结评分器对失败 token 的统一字段仍保留 `UNKNOWN_NOT_ASSUMED_ZERO`，
这里单列该失败响应已实际保存的计数，不改写评分器或旧报告。
远程模型调用和远程 API 费用均为 0；本地电力及算力成本未测量。
正式请求延迟合计约 1,744.30 秒，中位数约 4.08 秒，不含安装、下载和外部编排耗时。

## 独立重算与产物

重新读取原记录、重验完整请求、模型身份和冻结任务包，执行：

```powershell
.research-venv/Scripts/python.exe -X utf8 -m scripts.local_reference_judge score `
  --packet artifacts/research/chinese-external-reference-20261021-v2/psycrisis `
  --config artifacts/research/local-judge-20261022-v1/experiment/judge-config.json `
  --run artifacts/research/local-judge-20261022-v1/experiment/calibration `
  --output artifacts/research/local-judge-20261022-v1/calibration-independent-rescore.json
```

独立重算 JSON 与阶段原报告完整相等。
原任务包摘要在运行结束再次验证，没有变化。
运行后再次执行 `doctor --model qwen2.5:7b-instruct-q4_K_M`，
`calibration_execution_ready=true`、运行时阻断为空，确认安装条件已经解决。
此字段仅表示可开始新实验，不表示本次校准通过；
数据支持量阻断仍为 `public_holdout:risk_inquiry:INSUFFICIENT_POSITIVE_SUPPORT`。

- 运行时配置：`artifacts/research/local-judge-20261022-v1/runtime/runtime.json`
- 加载检查：`artifacts/research/local-judge-20261022-v1/smoke/result.json`
- 冻结配置：`artifacts/research/local-judge-20261022-v1/experiment/judge-config.json`
- 总结：`artifacts/research/local-judge-20261022-v1/experiment/summary.json`
- 阶段报告：`artifacts/research/local-judge-20261022-v1/experiment/calibration/report.json`
- 逐题记录：`artifacts/research/local-judge-20261022-v1/experiment/calibration/records/`
- 独立重算：`artifacts/research/local-judge-20261022-v1/calibration-independent-rescore.json`
- 新增 16 项测试：`artifacts/research/local-judge-20261022-v1/tests-install-smoke.xml`
- 运行后预检：`artifacts/research/local-judge-20261022-v1/post-run-doctor/preflight.json`

本机服务保持可用；未配置开机启动，重启机器后须显式启动。
在项目根目录执行下列入口，日志目录必须是新路径：

```powershell
& ./scripts/start_local_judge.ps1 -Output artifacts/research/local-runtime-next
```

不得直接重跑旧实验目录，也不应仅为凑完整覆盖在新目录隐形重试同一实验。
后续若改用更严格结构化输出、修改 Judge 提示词或调整推理配置，
须新建协议/任务包和记录，并披露此前校准材料已被使用。
公开留出虽未推理，也始终不是独立保密盲测。

## 不变的限制

风险询问的原人工阳性为校准 21 条、公开留出 7 条。
因此，即使本模型对现有样本全答对，也不能通过原五维联合支持量门禁。
不会自动放行项目回复评审；没有新专家 Gold，也没有目标 Agent 主实验。
本模型充当外部回复要素 Judge，不是原研究临床专家，也不替代行动规范标注。
模型家族独立性尚未验证，原公开材料是否被预训练见过也不能排除。
