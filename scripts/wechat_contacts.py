"""Explicit, read-only account and private-contact discovery for test setup."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from typing import Any


def discovery_dependency_error() -> str | None:
    if os.name != "nt":
        return "账号与好友读取仅支持 Windows 微信测试环境"
    if importlib.util.find_spec("wechatauto") is None:
        return "尚未安装可选依赖 wechatauto-replica，请先运行 Install-WeChatTestBridge.ps1"
    return None


def discover_contacts(
    account_wxid: str = "",
    query: str = "",
    *,
    confirmed: bool,
    database_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Return identities and display names only, never message or session content."""
    if not confirmed:
        raise ValueError("请先确认使用隔离测试账号与测试聊天")
    dependency_error = discovery_dependency_error()
    if dependency_error:
        raise RuntimeError(dependency_error)
    account_wxid = account_wxid.strip()
    query = query.strip()
    if len(account_wxid) > 128 or any(char.isspace() for char in account_wxid):
        raise ValueError("测试账号 ID 格式无效")
    if len(query) > 80:
        raise ValueError("好友搜索词不能超过 80 个字符")
    if database_factory is None:
        from wechatauto import WeChatDB

        from scripts.wechat_test_bridge import open_test_database

        database = open_test_database(account_wxid, WeChatDB)
    else:
        database = database_factory(**({"account": account_wxid} if account_wxid else {}))
    actual_wxid = str(getattr(database, "wxid", "")).strip()
    if not actual_wxid or (account_wxid and actual_wxid != account_wxid):
        raise RuntimeError("读取到的账号与所选测试账号不一致，已停止")

    self_info = database.get_self_info() or {}
    self_name = str(self_info.get("remark") or self_info.get("nick_name") or "").strip()
    names = database.nickname_map() or {}
    candidates: list[str] = []
    if query:
        candidates.extend(
            str(item.get("username") or "")
            for item in (database.search_contact(query) or [])
        )
    else:
        # Contact identities and names are sufficient; do not read message summaries.
        candidates.extend(str(wxid) for wxid in names)

    contacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for wxid in candidates:
        if (
            not wxid
            or wxid in seen
            or wxid == actual_wxid
            or len(wxid) > 128
            or any(char.isspace() for char in wxid)
            or wxid.endswith("@chatroom")
            or wxid.startswith("gh_")
        ):
            continue
        seen.add(wxid)
        name = str(names.get(wxid) or "").strip()
        contacts.append({"wxid": wxid, "name": name[:80]})
    contacts.sort(key=lambda item: (item["name"] or item["wxid"]).casefold())
    return {
        "account": {"wxid": actual_wxid, "name": self_name[:80]},
        "contacts": contacts,
    }
