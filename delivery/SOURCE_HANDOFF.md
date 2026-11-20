# EcoCompanion 心伴桌宠与消息渠道：源码交接版

2026-10-29 更新：微信公众号文本客服已移除，包括设置面板、网关回调与连接预检。
QQ 和个人微信测试桥接保持不变。下方旧版公众号描述仅为历史记录，不代表当前可用功能。

日期：2026-10-27。用于开发接手、代码审阅和已授权素材下的本地演示。
本包不是免安装独立程序，不是临床研究产品，也不包含真实账号或部署凭证。

## 1. 本版包含

- Electron 透明桌宠、拖拽、点击展开对话、7 段指定角色动画。
- DeepSeek 文本对话与实际生成验证；Qwen/Ollama 本地评估；TypeSafe Jev 建议。
- 多组会话的自动保存、独立草稿、搜索、分页、切换、重命名与删除。
- QQ 官方机器人 C2C。
- 设置页统一启动/停止消息网关；QQ 只填 AppID/AppSecret，复用桌面 DeepSeek。
- 独立、默认关闭的个人微信私聊白名单测试桥接，不自动读取或操作微信。
- 渠道隐私同意、会话隔离、限流、删除命令、回执去重和长回复分段。
- 应用内渠道配置页、独立网关启动脚本、React 前端源码及构建产物。
- 自我记录、本人原话选择、可选本机 Qwen 摘要、报告快照、独立删除与原生打印/PDF。
- Qwen 本地基线脚本、来源登记、分组隔离数据准备与 QLoRA 流程；不含模型权重。
- 针对性后端测试、协议模拟测试、桌面 QA 脚本与交付文档。

## 2. Windows 启动

需要 64 位 Windows、Python 3.11+、Node.js 22+ 和 npm。首次安装依赖需联网。

```powershell
py -3.11 -m venv .research-venv
.\.research-venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
.\.research-venv\Scripts\python.exe -m pip install --no-deps -e .
cd web
npm.cmd ci
npm.cmd run build
cd ..
.\Build-DesktopPetLauncher.ps1
.\EcoCompanion.exe --check
.\EcoCompanion.exe
```

`EcoCompanion.exe` 是快速启动器，会检查本地环境并启动后端与 Electron。
它不是可单独复制运行的打包安装程序。可选浏览器预览：

```powershell
$env:PYTHONPATH = "backend/src"
.\.research-venv\Scripts\python.exe -m uvicorn ecocompanion.pet.api:app --host 127.0.0.1 --port 8001
```

打开 `http://127.0.0.1:8001/`。启动不会自动下载 Ollama 权重、调用付费
模型或连接消息平台。模型与消息渠道均由用户显式配置。

## 3. 模型与渠道

DeepSeek 是云端回复模型，默认关闭云端传输。API Key、人格提示词、生成参数
在“对话服务”和“人格与表达”页配置。Qwen 使用本机 Ollama，当前仍是通用
权重，不是已完成微调的心理评估模型；Jev 使用实际 `jev-1.14.0` 接口，
但中文决策效果仍需专项评估。模型建议不能降低本地规则的安全等级。

“消息渠道”页保存 QQ 官方机器人与公众号配置，可启动/停止网关；保存本身
不启动联网。个人微信另有隔离测试开关、wxid 白名单、独立 Token 和可选脚本，
详见 `docs/ASTRBOT_CHANNEL_COMPATIBILITY.md`。它不是官方个人微信 API，
未完成真实账号验证，不能默认接管当前微信。

## 4. 验证与限制

会话、草稿、标题与选择均保存在 SQLite / 本机状态文件中，重启后端可恢复。
每次渠道事件在发送前保存最终回复，平台重试恢复同一内容，不重复调用模型。
未发送桌面草稿也在本机持久化，没有标题 AI 总结或跨设备同步。

当前完成模拟协议与本地网关测试，没有真实 QQ 或微信公众号上线验证。
公开部署前还需权限审批、HTTPS 反向代理、用户协议、审计和独立安全评审。
详细结果见 `delivery/VALIDATION.md`。工程验证不等于心理或临床有效性验证。

## 5. 隐私与素材

包内不含密钥、开发者对话、SQLite 数据库、轨迹、模型权重、虚拟环境或
`node_modules`。运行时对话和调试轨迹未加密，本版不适用于真实敏感个案。

角色 OC「溟月」和动画来自指定参考工程，原始声明见
`web/public/pet/ATTRIBUTION.md`。素材有个人非商业使用限制，署名不自动
取得公开分发或商业授权；公开比赛与发布前须确认权利方许可。
原始 Python 项目未附根目录 LICENSE，本包不推定新增开源许可。

## 6. Qwen 与报告验证

Ollama `qwen2.5:7b` 已实际运行 29 条工程样例：29 条 JSON 可解析，28 条证据
逐字匹配，安全线索标签一致 9 条。这个小样本不构成临床或上线有效性证据。
QLoRA 的权重下载、训练和适配器结果不放入 ZIP，实际状态见
`docs/QWEN_FINETUNING.md`；通用模型仍须经过本地规则约束。
已完成 0.5B 隔离 QLoRA 冒烟，但 7B 正式微调未完成，未将适配器接入应用。

报告中的 1-5 分是用户自评，不是量表。AI 摘要逐次同意，只发送所选原话到
本机 Qwen，生成后可修订；删除会话会同步清除报告中的相关原话和自动摘要。
`delivery/VALIDATION.md` 记录实际执行的验证，不把模拟平台协议称为真实上线。
