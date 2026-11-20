import json
import sys
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest

from scripts import wechat_test_bridge as transport


def test_recipient_resolution_rejects_internal_id_and_ambiguous_contacts():
    class Database:
        def __init__(self, nickname, contacts):
            self.nickname = nickname
            self.contacts = contacts

        def get_nickname(self, username):
            return self.nickname

        def search_contact(self, nickname):
            return self.contacts

    contact = {"username": "wxid_test_friend", "nick_name": "测试好友", "remark": ""}
    assert transport.resolve_recipient(
        Database("测试好友", [contact]), contact["username"]
    ) == "测试好友"
    with pytest.raises(RuntimeError, match="无法取得"):
        transport.resolve_recipient(Database("wxid_test_friend", []), "wxid_test_friend")
    with pytest.raises(RuntimeError, match="已变化"):
        transport.resolve_recipient(
            Database("测试好友", [contact, {"username": "other", "nick_name": "测试好友"}]),
            "wxid_test_friend",
        )


def test_sender_requires_explicit_success_and_keeps_scoped_database():
    class Database:
        wxid = "wxid_test_account"

        def get_messages(self, username, limit=20):
            return []

    class Gui:
        def send_msg(self, text, display_name, verify):
            assert (text, display_name, verify) == ("回复", "测试好友", False)
            class Failure(dict):
                is_success = False

            return Failure(message="会话没打开")

    gui = Gui()
    database = Database()
    with pytest.raises(RuntimeError, match="会话没打开"):
        transport.send_reply(gui, database, "wxid_test_friend", "测试好友", "回复")
    assert gui._cached_db is database


def test_delivery_confirmation_uses_account_identity_and_fresh_watermark():
    class Database:
        wxid = "wxid_test_account"

        def __init__(self):
            self.rows = [{
                "sender_username": "wxid_test_friend",
                "sender_id": 2,
                "content": "相同的历史文本",
                "sort_seq": 100,
                "local_id": 1,
                "shard": "message_0",
            }]

        def get_messages(self, username, limit):
            assert username == "wxid_test_friend"
            return self.rows

    database = Database()
    watermark = transport.reply_watermark(database, "wxid_test_friend")
    # An incoming message with identical text must never become a send receipt.
    database.rows.insert(0, {
        "sender_username": "wxid_test_friend",
        "sender_id": 2,
        "content": "回复",
        "sort_seq": 101,
        "local_id": 2,
        "shard": "message_0",
    })
    assert not transport.reply_confirmed(database, "wxid_test_friend", "回复", watermark)
    # This account uses sender_id=1 for itself; an equal sort_seq is still new.
    database.rows.insert(0, {
        "sender_username": "wxid_test_account",
        "sender_id": 1,
        "content": "回复",
        "sort_seq": 100,
        "local_id": 3,
        "shard": "message_0",
    })
    assert transport.reply_confirmed(database, "wxid_test_friend", "回复", watermark)
    assert not transport.reply_confirmed(
        database, "wxid_test_friend", "相同的历史文本", watermark
    )


def test_account_directory_resolution_includes_wechat_hash_suffix(monkeypatch):
    class Database:
        wxid = "wxid_test_account"
        _db_files: ClassVar[list] = [("contact.db", "test-path", 100)]

        def __init__(self, db_dir, account):
            assert db_dir == "test-root"
            assert account == "wxid_test_account_ab12"

    monkeypatch.setattr(
        "wechatauto.db.auto_detect_db_dir", lambda: "test-root"
    )
    monkeypatch.setattr(
        "wechatauto.db._find_account_dirs",
        lambda root: [
            "test-root/other_account",
            "test-root/wxid_test_account_ab12",
        ],
    )
    database = transport.open_test_database("wxid_test_account", Database)
    assert database.wxid == "wxid_test_account"
    with pytest.raises(RuntimeError, match="唯一匹配"):
        transport.open_test_database("missing", Database)


def test_synthetic_whitelisted_message_reaches_gateway_and_display_name_sender(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("ECO_MESSAGING_HOME", str(tmp_path))
    monkeypatch.setattr(transport, "gateway_ready", lambda url: True)
    monkeypatch.setattr(
        transport,
        "load_settings",
        lambda _: SimpleNamespace(
            callback_port=8765,
            personal_wechat=SimpleNamespace(
                allow_groups=False,
                allowed_chats=["wxid_test_friend"],
                account_wxid="wxid_test_account",
                bridge_token="synthetic-only-token-that-is-long-enough",
            ),
        ),
    )
    sent = threading.Event()
    created_gui = None

    class Database:
        wxid = "wxid_test_account"
        _db_files: ClassVar[list] = [("contact.db", "test-path", 100)]
        instance = None

        def __init__(self, db_dir, account):
            assert account == "wxid_test_account_ab12"
            self.rows = []
            Database.instance = self

        def get_messages(self, username, limit=20):
            return self.rows

        def nickname_map(self):
            return {
                "wxid_test_account": "测试账号",
                "wxid_test_friend": "测试好友",
            }

        def get_nickname(self, username):
            assert username == "wxid_test_friend"
            return "测试好友"

        def search_contact(self, nickname):
            assert nickname == "测试好友"
            return [{"username": "wxid_test_friend", "nick_name": nickname, "remark": ""}]

    class Gui:
        def __init__(self):
            nonlocal created_gui
            self.calls = []
            created_gui = self

        def send_msg(self, text, display_name, verify):
            self.calls.append((text, display_name, verify))
            Database.instance.rows = [{
                "sender_username": "wxid_test_account",
                "sender_id": 1,
                "content": text,
                "sort_seq": 1,
                "local_id": 2,
                "shard": "message_0",
            }]
            class Success(dict):
                is_success = True

            return Success(message="已确认")

    class Listener:
        instance = None

        def __init__(self, database, interval, watermark_file):
            assert watermark_file == ""
            self.registrations = []
            self.stopped = False
            Listener.instance = self

        def add_listener(self, username, callback):
            self.registrations.append(username)
            self.callback = callback

        def start(self):
            self.callback(
                {
                    "username": "wxid_test_friend",
                    "sender_username": "wxid_test_friend",
                    "type": "文本",
                    "content": "测试消息",
                    "local_id": 1,
                    "shard": "synthetic",
                },
                self,
            )

        def stop(self):
            self.stopped = True

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"action": "reply", "reply": "测试回复"}

    def post(url, **kwargs):
        assert url == "http://127.0.0.1:8765/webhook/wechat"
        assert kwargs["json"]["text"] == "测试消息"
        return Response()

    monkeypatch.setattr(transport.httpx, "post", post)
    report_status = transport.BridgeStatus.set

    def wait_for_delivery(self, phase, detail):
        report_status(self, phase, detail)
        if phase == "listening" and "已确认发送" in detail:
            sent.set()

    monkeypatch.setattr(transport.BridgeStatus, "set", wait_for_delivery)
    monkeypatch.setitem(
        sys.modules,
        "wechatauto",
        SimpleNamespace(WeChatDB=Database, WeChatGUI=Gui),
    )
    monkeypatch.setitem(
        sys.modules, "wechatauto.db", SimpleNamespace(Listener=Listener)
    )
    monkeypatch.setattr(transport, "open_test_database", lambda account, factory: factory(
        db_dir="test-root", account="wxid_test_account_ab12"
    ))

    def sleep(_):
        if sent.is_set():
            raise KeyboardInterrupt
        sent.wait(0.05)

    monkeypatch.setattr(transport.time, "sleep", sleep)
    transport.run()

    listener = Listener.instance
    assert listener is not None
    assert listener.registrations == ["wxid_test_friend"]
    assert listener.stopped
    assert created_gui is not None
    assert created_gui.calls == [("测试回复", "测试好友", False)]
    status = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if json.loads(line).get("wechat_bridge") == 1
    ]
    assert status[-1]["phase"] == "listening"
    assert "已确认发送" in status[-1]["detail"]
def test_gateway_is_required_before_wechat_is_touched(tmp_path, monkeypatch):
    monkeypatch.setenv("ECO_MESSAGING_HOME", str(tmp_path))
    monkeypatch.setattr(transport, "gateway_ready", lambda _: False)
    monkeypatch.setattr(
        transport,
        "load_settings",
        lambda _: SimpleNamespace(
            callback_port=8765,
            personal_wechat=SimpleNamespace(
                allow_groups=False,
                allowed_chats=["wxid_test_friend"],
                account_wxid="wxid_test_account",
                bridge_token="synthetic-only-token-that-is-long-enough",
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="消息网关未启动"):
        transport.run()
