"""Explicit, secret-free official-channel connectivity checks."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from ecocompanion.messaging.qq import QQApi
from ecocompanion.messaging.settings import MessagingSettings
from ecocompanion.pet.settings import PetSettings


async def check_messaging(
    settings: MessagingSettings,
    pet: PetSettings | None = None,
    *,
    deepseek_ready: bool = False,
    timeout: float = 8,
) -> dict[str, Any]:
    checks = []
    if not any((settings.qq.enabled, settings.personal_wechat.enabled)):
        return {"ok": False, "checks": [], "detail": "请先启用并保存至少一个消息渠道"}

    async def run(name: str, check: Callable[[], Awaitable[str]]) -> None:
        try:
            detail = await asyncio.wait_for(check(), timeout)
            checks.append({"channel": name, "ok": True, "detail": detail})
        except (httpx.HTTPError, OSError, ValueError, RuntimeError, TimeoutError):
            checks.append(
                {
                    "channel": name,
                    "ok": False,
                    "detail": (
                        "连接或鉴权失败，请核对官方控制台、凭证、权限和网络；"
                        "详细错误未回传以免泄露信息"
                    ),
                }
            )

    if settings.qq.enabled:

        async def qq_check() -> str:
            api = QQApi(settings.qq)
            try:
                await api.access_token()
                await api.gateway()
                return "QQ 凭证与 Gateway 可用；启动网关后测试私聊收发"
            finally:
                await api.close()

        await run("qq", qq_check)
    if settings.personal_wechat.enabled:
        try:
            with socket.create_connection(
                (settings.callback_host, settings.callback_port), timeout=2
            ):
                pass
            detail = "本机桥接入口已监听；仍需启动独立微信桥接并验证测试会话"
            ok = True
        except OSError:
            detail = "本机桥接入口未监听，请先启动消息网关"
            ok = False
        checks.append({"channel": "personal_wechat", "ok": ok, "detail": detail})
    if pet is not None:
        ready = pet.cloud_consent and deepseek_ready
        checks.append(
            {
                "channel": "deepseek",
                # Local audited fallback is an intentional supported mode.
                "ok": True,
                "detail": (
                    "已继承桌面 DeepSeek 配置"
                    if ready
                    else "云端回复未启用，将使用本地审查回复"
                ),
            }
        )
    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "detail": "仅检查连接与鉴权，不代表消息收发或平台上线已验证",
    }
