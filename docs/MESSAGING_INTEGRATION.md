# 官方消息渠道接入

2026-10-29 更新：应用已移除微信公众号文本客服的设置、回调和预检入口。
当前仅保留 QQ 与个人微信隔离测试桥接。本文后续公众号内容为历史协议说明。

日期：2026-10-27。本版实现 QQ 官方机器人私聊与微信公众号文本回调，
QQ 不登录个人账号；另有默认关闭的个人微信隔离测试桥接，详见
`ASTRBOT_CHANNEL_COMPATIBILITY.md`。不支持 QQ 群、频道或个人微信群聊。

## 架构与数据边界

平台事件 -> 官方签名或 AccessToken 验证 -> 事件去重与每用户限流 ->
独立渠道会话 -> 本地规则 -> 可选 Qwen 证据提取 -> 可选 Jev 决策建议 ->
DeepSeek 回复 -> 本地输出审查 -> 平台文本回复。

渠道复用桌面端的模型、人格和密钥设置，但使用独立的 `conversations.db`。
每个 `(channel, external_user)` 有独立会话，不能跨用户读取或删除历史。
首次消息显示用途、平台与云端传输告知，用户回复“同意”后才创建会话。

支持“新对话”“删除本机记录”“确认删除”“取消”“帮助”。删除只删除本机
会话、草稿、绑定与渠道回执，不承诺撤回或删除平台已有消息。单条用户消息
最长 2000 字；默认每用户每分钟 6 条。渠道当前仅用于成年测试者预览。

## 配置与启动

在应用设置的“消息渠道”页填写并保存凭证，或者从 `docs/messaging.example.json`
创建 `%LOCALAPPDATA%/EcoCompanion/pet/messaging/messaging.json`。界面不会回传
已保存密钥；Windows 下密钥通过 DPAPI 绑定当前 Windows 用户。非 Windows
配置使用权限为 0600 的本地文件，仍应由部署者妥善保护。

保存配置不会自动联网。可以在设置页启动/停止已保存配置，或显式运行：

```powershell
.\Start-MessagingGateway.ps1
```

该脚本调用 `python -m ecocompanion.messaging.runtime`。`ECO_MESSAGING_HOME`
可覆盖渠道配置与数据库目录；`ECO_PET_HOME` 可覆盖共享模型设置目录。
桌宠本地 `/api/pet` 端口绝不能暴露到公网。

个人微信桥接需要在上述网关启动后，再从设置页单独启动。界面显示
“监听已就绪”才表示数据库、窗口和监听初始化完成；只显示进程运行不代表可收发。
测试好友应在就绪后发送新消息，首次回复“同意”。桥接仅注册白名单私聊，
不轮询其他会话；回复时从内部 wxid 核对并转换为好友显示名，发送回读也复用
同一测试账号数据库。失败阶段会显示在设置页，不记录聊天正文或密钥。

## QQ 官方机器人

仅处理 `C2C_MESSAGE_CREATE`。使用官方 `getAppAccessToken`、`/gateway`
WebSocket、Identify / Heartbeat / Resume 及 `/v2/users/{openid}/messages`。
回复以 `msg_id` 关联原事件，1800 字分段，使用递增 `msg_seq`；模型处理前
先发送“正在整理回复”。凭证由开发者在 QQ 开放平台创建，私聊权限、
沙箱用户及发布状态由开发者控制台决定。
应用只要求 AppID/AppSecret，DeepSeek、人格和评估配置直接继承桌面端。

- https://bot.q.qq.com/wiki/develop/api-v2/dev-prepare/access-token.html
- https://bot.q.qq.com/wiki/develop/api-v2/dev-prepare/interface-framework/event-emit.html
- https://bot.q.qq.com/wiki/develop/api-v2/autogen/event/c2c_message_create.html
- https://bot.q.qq.com/wiki/develop/api-v2/autogen/api/v2_users_user_openid_messages.post.html

## 微信公众号

实现官方公众号 XML 文本回调、SHA1 签名检查、报文大小限制与 DTD/ENTITY
拒绝。回调先返回 `success`，后台完成处理后调用客服文本接口，长回复按
2000 字分段。仅支持明文回调；AES 兼容模式尚未实现。配置 Token、AppID、
AppSecret 后启动网关，回调地址为：

`https://你的受管域名/wechat/callback`

服务只监听 `127.0.0.1:8765`（端口可调）。部署者必须使用受管理的 HTTPS
反向代理、限制访问、验证平台侧白名单和客服消息权限，不能把本机端口直接
映射到公网。公众号是否具备对应客服接口权限，以账号后台和官方条款为准。

官方入口：https://kf.weixin.qq.com/api/doc/path/93304

## 验证范围与上线前事项

已测试模拟官方协议、Token、分段、签名、快速 ACK、去重、用户隔离、删除、
网关启动与关闭。没有真实账号、密钥、线上回调或平台沙箱，因此不代表真实
平台上线、权限审批或可用性验证完成。QQ 被动回复存在平台时限，正式上线前
需用真实沙箱验证长回复与进度提示的投递策略；若平台只允许有限被动回复，
应改为合规的主动消息产品或减少长推理等待。

本版不适合真实敏感个案。公开多用户部署还需正式隐私政策、账号级授权、
审计、备份删除策略、内容安全运营、危机服务边界及独立安全评审。
