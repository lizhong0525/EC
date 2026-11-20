# 小溟游戏间

## 启动与接入

双击项目根目录 `进入小游戏.cmd`，或执行 `./Start-Games.ps1`，然后访问
`http://127.0.0.1:8765/games`。网页服务仅监听本机，独立于训练进程，复用桌宠 API。
桌面版主窗口与桌宠工具条新增游戏手柄入口；运行桌面版前需重新构建前端并重启。

游戏沿用工程 `PetSettings` 和 DeepSeek 客户端。已使用本机桌宠 API 密钥且开启云端授权时
可直接游戏；否则在页面右上角设置密钥和云端授权。密钥仅保存于本机 DPAPI 配置。
每步游戏会携带 DeepSeek 调用费用。游戏发言仅用于游戏回合，不会进入心理对话或训练数据目录。

## 规则与实现边界

- 国际象棋：用户执白或执黑，对 DeepSeek。`python-chess` 是合法走子、将军、将死、
  易位、吃过路兵、升变、自然和棋的唯一裁判；DeepSeek 从规则库提供的合法走子中选一步。
  可认输，符合条件时可提和；AI 可直接提出可主张和棋时按提和处理。无计时或引擎 Elo 宣称。
- 谁是卧底：用户与三个独立 DeepSeek 子 Agent，四人三平民一卧底，无白板。
  每人只知道自己的词与公开发言，不直接知道自己的阵营。每轮依次描述、自由讨论、
  匿名投票；最高票出局。平票先由候选人加赛辩护，再投一次；再平票无人出局。
  卧底出局则平民胜，剩两人则卧底胜。八轮仍未分胜负则平局。
- 投票在后台按序独立请求，仅向后续投票者公开已完成的票数，不公开具体票型。
  投票结束后统一公布。淘汰的人类玩家只能观战。AI 发言或投票失败不会由本地随机操作替代。
- 所有动作以服务端状态、版本号与 SQLite 原子更新为准；每步后保存。网页可恢复当前存档。
  异常提示不含 API 密钥。旧存档在进程意外退出后按最近已保存的动作恢复。

调研依据：
- [FIDE Laws of Chess, 2023](https://handbook.fide.com/chapter/E012023)
- [python-chess documentation](https://python-chess.readthedocs.io/en/latest/core.html)
- [Undercover official How to Play](https://www.yanstarstudio.com/undercover-how-to-play)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode)
