"""Isolated, whitelist-only Windows WeChat test transport.

This optional process is never started by importing the application. It uses
wechatauto-replica's local DB listener and UI sender, not an official personal
WeChat API. Account terms, client-version and wrong-recipient risks remain.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from ecocompanion.messaging.settings import load_settings, messaging_home

LOG = logging.getLogger("ecopet.wechat-test-bridge")
STATUS_VERSION = 1
IGNORE_REASONS = {
    "chat is not in the test whitelist": "该会话不在网关白名单，请保存渠道配置并重启微信桥接",
    "duplicate": "这条消息已处理，不会重复回复",
    "rate_limited": "消息过于频繁，请稍后再试",
}


class BridgeStatus:
    """A small, secret-free lifecycle report for the desktop settings panel."""

    def __init__(self) -> None:
        self.phase = "starting"
        self.detail = "正在初始化微信测试桥接"

    def set(self, phase: str, detail: str) -> None:
        self.phase = phase
        self.detail = detail
        print(
            json.dumps(
                {"wechat_bridge": STATUS_VERSION, "phase": phase, "detail": detail},
                ensure_ascii=False,
            ),
            flush=True,
        )


def gateway_ready(url: str) -> bool:
    try:
        response = httpx.get(f"{url}/healthz", timeout=2, trust_env=False)
        return response.status_code == 200 and response.json().get("ok") is True
    except (httpx.HTTPError, ValueError, TypeError):
        return False


def resolve_recipient(database: Any, username: str) -> str:
    """Resolve a stable internal ID to the exact contact used by WeChat search."""
    nickname = str(database.get_nickname(username) or "").strip()
    if not nickname or nickname == username:
        raise RuntimeError("无法取得测试好友的备注或昵称，已停止以避免发错会话")
    matches = [
        contact
        for contact in database.search_contact(nickname) or []
        if contact.get("username") == username
        and str(contact.get("remark") or contact.get("nick_name") or "").strip()
        == nickname
    ]
    same_name = [
        contact
        for contact in database.search_contact(nickname) or []
        if contact.get("username") != username
        and str(contact.get("remark") or contact.get("nick_name") or "").strip()
        == nickname
    ]
    if len(matches) != 1 or same_name:
        raise RuntimeError("测试好友备注或昵称已变化，请重新读取好友并保存")
    return nickname


def reply_watermark(database: Any, username: str) -> dict[str, Any]:
    rows = database.get_messages(username, limit=20) or []
    return {
        "seq": max((int(row.get("sort_seq") or 0) for row in rows), default=0),
        "ids": {
            (
                int(row.get("sort_seq") or 0),
                str(row.get("shard") or ""),
                int(row.get("local_id") or 0),
            )
            for row in rows
        },
    }


def reply_confirmed(
    database: Any, username: str, text: str, watermark: dict[str, Any]
) -> bool:
    """Accept only a new, exact own message; numeric sender IDs vary by shard."""
    for row in database.get_messages(username, limit=20) or []:
        seq = int(row.get("sort_seq") or 0)
        identity = (seq, str(row.get("shard") or ""), int(row.get("local_id") or 0))
        if seq < watermark["seq"] or identity in watermark["ids"]:
            continue
        if str(row.get("sender_username") or "") != str(database.wxid):
            continue
        if row.get("content") == text:
            return True
    return False


def confirm_sent_reply(
    database: Any,
    username: str,
    text: str,
    watermark: dict[str, Any],
    *,
    timeout: float = 12,
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if reply_confirmed(database, username, text, watermark):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.5, remaining))


def send_reply(gui: Any, database: Any, username: str, display_name: str, text: str) -> None:
    """Send once, then verify using the locked account instead of sender_id=2."""
    # WeChatGUI otherwise creates an unscoped WeChatDB for its delivery receipt.
    gui._cached_db = database
    watermark = reply_watermark(database, username)
    result = gui.send_msg(text, display_name, verify=False)
    if not getattr(result, "is_success", False):
        message = str(result.get("message") or "") if isinstance(result, dict) else ""
        raise RuntimeError(message[:180] or "微信窗口未能完成回复操作")
    if not confirm_sent_reply(database, username, text, watermark):
        raise RuntimeError("微信已操作发送，但未能确认新回复；请勿重发，先检查测试会话")


def open_test_database(account_wxid: str, database_factory: Callable[..., Any]) -> Any:
    """Resolve an internal account ID to WeChat's suffixed on-disk directory."""
    from wechatauto.db import _find_account_dirs, auto_detect_db_dir

    root = auto_detect_db_dir()
    if not root:
        raise RuntimeError("未找到微信数据库目录，请确认 Windows 微信已登录")
    candidates = [
        path
        for path in _find_account_dirs(root)
        if os.path.basename(path) == account_wxid
        or os.path.basename(path).startswith(f"{account_wxid}_")
    ]
    if len(candidates) != 1:
        raise RuntimeError("无法唯一匹配测试账号目录，请重新登录专用测试账号后重试")
    database = database_factory(db_dir=root, account=os.path.basename(candidates[0]))
    if str(getattr(database, "wxid", "")) != account_wxid:
        raise RuntimeError("当前微信账号与设置中的测试账号不一致，已停止")
    if not getattr(database, "_db_files", None):
        raise RuntimeError("测试账号数据库未加载，请确认微信已登录并有本地消息记录")
    return database


def database_summary(database: Any) -> dict[str, int]:
    return {
        "database_files": len(getattr(database, "_db_files", [])),
        "received": 0,
        "ignored_nontext": 0,
        "ignored_own": 0,
        "ignored_identity": 0,
        "gateway_requests": 0,
        "gateway_ignored": 0,
        "queued_replies": 0,
        "sent_replies": 0,
    }


def preflight() -> dict[str, Any]:
    try:
        settings = load_settings(messaging_home()).personal_wechat
    except (ValueError, OSError) as exc:
        return {"ok": False, "detail": str(exc)}
    if os.name != "nt":
        return {"ok": False, "detail": "此桥接仅支持 Windows 微信测试环境"}
    if importlib.util.find_spec("wechatauto") is None:
        return {"ok": False, "detail": "尚未安装可选依赖 wechatauto-replica"}
    if not settings.enabled:
        return {"ok": False, "detail": "请先在设置中启用并保存个人微信测试桥接"}
    return {"ok": True, "detail": "配置与依赖已就绪，启动后仍需核对测试账号和窗口"}


def run() -> None:
    settings = load_settings(messaging_home())
    personal = settings.personal_wechat
    status = BridgeStatus()
    if personal.allow_groups:
        raise RuntimeError("当前桥接只支持私聊白名单；请关闭群聊选项")
    allowed = set(personal.allowed_chats)
    base_url = f"http://127.0.0.1:{settings.callback_port}"
    if not gateway_ready(base_url):
        raise RuntimeError("消息网关未启动，请先点击“启动已保存配置”，再启动微信桥接")
    # Import only after an explicit run. A missing optional dependency cannot
    # affect the desktop app, official QQ, or Official Account adapters.
    from wechatauto import WeChatDB, WeChatGUI
    from wechatauto.db import Listener

    status.set("starting", "正在核对测试账号并准备消息监听，请勿操作微信窗口")
    database = open_test_database(personal.account_wxid, WeChatDB)
    names = database.nickname_map() or {}
    if personal.account_wxid not in names or any(
        username not in names for username in allowed
    ):
        raise RuntimeError("测试账号或好友不在当前数据库，请重新读取账号与全部好友后保存")
    metrics = database_summary(database)
    gui = WeChatGUI()
    outgoing: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=32)
    finished = threading.Event()
    url = f"{base_url}/webhook/wechat"

    def receive(message: dict[str, Any], _: Any) -> None:
        username = str(message.get("username") or "")
        if username not in allowed:
            return
        metrics["received"] += 1
        if message.get("type") != "文本":
            metrics["ignored_nontext"] += 1
            return
        sender = str(message.get("sender_username") or "")
        if sender == personal.account_wxid:
            metrics["ignored_own"] += 1
            return
        if not sender:
            metrics["ignored_identity"] += 1
            status.set("listening", "收到消息但无法确认发送者，已忽略；请报告此状态")
            return
        text = str(message.get("content") or "").strip()
        local_id = message.get("local_id")
        if not text or local_id is None:
            return
        group = username.endswith("@chatroom")
        # The adapter must provide an explicit mention flag; guessing from a
        # display name can cause a message for someone else to trigger a reply.
        mentioned = bool(message.get("at_me") or message.get("is_at"))
        payload = {
            "chat_id": username,
            "chat_type": "group" if group else "private",
            "user_id": sender,
            "text": text,
            "msg_id": f"{username}:{message.get('shard') or ''}:{local_id}",
            "at_me": mentioned,
        }
        try:
            metrics["gateway_requests"] += 1
            response = httpx.post(
                url,
                json=payload,
                headers={"x-bot-token": personal.bridge_token},
                timeout=150,
                trust_env=False,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("action") != "reply" or not result.get("reply"):
                metrics["gateway_ignored"] += 1
                reason = str(result.get("reason") or "未生成回复")
                reason = IGNORE_REASONS.get(reason, reason)
                status.set("listening", f"消息已到达网关但未回复：{reason[:100]}")
                return
            # A single text reply, not the upstream proactive/bubble features.
            outgoing.put_nowait((username, str(result["reply"])))
            metrics["queued_replies"] += 1
            status.set(
                "listening",
                "网关已生成回复，正在发送；请暂时不要移动鼠标或切换微信会话",
            )
        except httpx.ConnectError:
            status.set("error", "消息网关连接中断，请启动已保存配置后重启微信桥接")
        except httpx.TimeoutException:
            status.set("error", "消息网关处理超时，请检查模型连接；可稍后发送新消息重试")
        except httpx.HTTPStatusError:
            status.set("error", "消息网关拒绝了测试消息，请保存配置并重启消息网关")
        except (httpx.HTTPError, ValueError, TypeError):
            status.set("error", "消息网关返回无效结果，请检查本机服务与渠道配置")
        except queue.Full:
            status.set("error", "待回复消息较多，本次消息未处理，请稍后发送新消息")

    def send() -> None:
        while not finished.is_set():
            try:
                username, text = outgoing.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if username not in allowed:
                    continue
                send_reply(
                    gui, database, username, resolve_recipient(database, username), text
                )
                metrics["sent_replies"] += 1
                status.set("listening", "监听已就绪；最近一条测试回复已确认发送")
            except Exception as exc:
                # Do not expose a recipient, message body, token, or local DB path.
                status.set("error", f"微信回复未完成：{str(exc)[:180]}")
            finally:
                outgoing.task_done()

    # Register only approved private chats. add_all reads unrelated sessions even
    # though its callback later discards them.
    listener = Listener(database, interval=0.6, watermark_file="")
    for username in allowed:
        listener.add_listener(username, receive)
    sender = threading.Thread(target=send, name="wechat-test-sender", daemon=True)
    sender.start()
    status.set("listening", "监听已就绪；请白名单好友发送新消息，首次需回复“同意”")
    listener.start()
    last_metrics = ""
    try:
        while True:
            time.sleep(1)
            current_metrics = json.dumps(metrics, sort_keys=True)
            if current_metrics != last_metrics:
                last_metrics = current_metrics
                print(
                    json.dumps({"wechat_bridge_metrics": STATUS_VERSION, **metrics}),
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        finished.set()
        listener.stop()
        sender.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--confirm-test-account", action="store_true")
    parser.add_argument("--account", default="")
    parser.add_argument("--query", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.discover:
        if args.check or args.run:
            parser.error("--discover cannot be combined with --check or --run")
        from scripts.wechat_contacts import discover_contacts

        result = discover_contacts(
            args.account, args.query, confirmed=args.confirm_test_account
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    status = preflight()
    if args.check:
        print(json.dumps(status, ensure_ascii=False))
        return 0 if status["ok"] else 2
    if args.run:
        if not status["ok"]:
            raise RuntimeError(status["detail"])
        try:
            run()
        except Exception as exc:
            BridgeStatus().set("error", str(exc)[:220])
            return 1
        return 0
    parser.error("choose --check or --run")
    return 2


if __name__ == "__main__":
    sys.exit(main())
