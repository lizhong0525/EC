# EcoCompanion 项目提交说明

版本日期：2026-10-29。此 ZIP 用于代码审阅、项目展示和继续开发，不是免安装程序。

## 项目概述

EcoCompanion 是中文、文本优先的心理陪伴 Agent 与合成研究平台，包含
EcoTriage-Bench、ECO-SAFE 规则决策、桌宠对话、消息渠道、报告功能和本地小游戏。
系统面向合成数据和成年研究测试，不是医疗诊断、治疗或实时危机响应服务。

## 主要功能

- 桌宠陪伴：DeepSeek 对话、会话保存、人格与模型设置、Qwen 本地评估、记录与打印。
- 游戏间：用户与 DeepSeek 下国际象棋；用户与三个独立 DeepSeek 玩家进行谁是卧底。
- 国际象棋：标准规则、合法落子校验、将军与终局、认输/提和、PGN 下载和本地存档。
- 谁是卧底：四人局、私有词语隔离、描述/讨论/投票、平票辩护、淘汰与身份揭晓。
- 消息渠道：QQ 官方机器人和默认关闭的个人微信测试桥接。
- 研究平台：合成 Benchmark、模型评估、规则审计、标注工具、来源登记及离线训练脚本。

## 目录导览

- `backend/src/ecocompanion/`：Python 后端、研究流程、桌宠 API 与游戏引擎。
- `backend/tests/`：后端测试，含国际象棋和谁是卧底规则及 API 测试。
- `web/src/`：React/TypeScript 界面；`web/src/games/` 为游戏间。
- `web/public/games/`：用户提供的大肥鱼图片，四个 AI 玩家使用不同造型。
- `web/desktop/`：Electron 桌宠与游戏窗口入口。
- `web/dist/`：已构建网页；依赖安装完成后可直接由本地后端提供。
- `docs/`、`training/data/`、`artifacts/cases/`：研究说明、公开规范样例与 Schema。
- `scripts/`：研究、评估、训练与打包工具；`delivery/VALIDATION.md` 为验证记录。
- `preview/`：合成桌宠与游戏界面截图，仅供界面审阅。

## 安装与启动

需要 Windows、Python 3.11+、Node.js 22+ 和 npm。首次安装依赖需要联网。

```powershell
py -3.11 -m venv .research-venv
.\.research-venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
.\.research-venv\Scripts\python.exe -m pip install --no-deps -e .
npm.cmd --prefix web ci
npm.cmd --prefix web run build
.\.research-venv\Scripts\python.exe -m pip install pytest pytest-asyncio
.\.research-venv\Scripts\python.exe -m pytest backend/tests/test_games.py
.\进入小游戏.cmd
```

打开 `http://127.0.0.1:8765/games`。启动本身不会调用付费模型。
在游戏页右上角设置 DeepSeek API Key，并明确开启云端授权后才可进行 AI 对局。
不要使用远程训练环境保存的个人密钥；请在接收项目后自行配置。

桌宠可按 `delivery/SOURCE_HANDOFF.md` 安装与启动。API Key 不会随 ZIP 提供。

微信公众号文本客服已移除，不再提供设置、回调路由或连接预检。
为兼容旧配置和历史测试，源码仍保留公众号协议工具与配置字段，但应用网关不再使用。

## 打包范围与限制

包含完整研究/应用源码、测试、文档、网页构建、游戏图片及合成示例数据。
不包含 API 密钥、真实聊天或数据库、虚拟环境、`node_modules`、模型权重、
远程训练输出、下载的外部数据集、临时缓存或真实账号配置。

三个卧底玩家由后端依次独立请求 DeepSeek，不是 Codex 子 Agent，也不是共享上下文
的单一回复。规则与私有信息在本地处理，AI 仅返回规定的结构化动作。

项目没有新增开源许可证。角色动画与游戏图片的公开比赛分发授权须另行确认，
详见 `delivery/THIRD_PARTY.md` 与 `web/public/pet/ATTRIBUTION.md`。
`MANIFEST.json` 记录每个文件的 SHA-256；历史验证不代替当前环境的重新验证。
