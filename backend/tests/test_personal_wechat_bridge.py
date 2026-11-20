import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.tests.test_messaging import build_processor
from ecocompanion.messaging.bridge import PersonalWeChatBridge
from ecocompanion.messaging.personal_wechat import (
    bridge_message,
    create_personal_wechat_app,
)
from ecocompanion.messaging.settings import PersonalWeChatSettings
from ecocompanion.pet.api import create_pet_app

TOKEN = "test-only-bridge-token-that-is-long-enough"


@pytest.mark.asyncio
async def test_contact_read_requires_confirmation_and_a_stopped_bridge():
    bridge = PersonalWeChatBridge()
    with pytest.raises(ValueError, match="确认"):
        await bridge.discover("", "", confirmed=False)
    bridge.process = type("RunningProcess", (), {"poll": lambda self: None})()
    with pytest.raises(RuntimeError, match="停止"):
        await bridge.discover("", "", confirmed=True)


@pytest.mark.asyncio
async def test_component_install_is_independent_from_account_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    bridge = PersonalWeChatBridge()
    monkeypatch.setattr(
        "ecocompanion.messaging.bridge.importlib.util.find_spec", lambda _: object()
    )
    result = await bridge.install()
    assert result["ok"]
    assert "已安装" in result["detail"]
    assert not bridge.installation_status()["running"]


def test_bridge_status_is_not_ready_before_listener_confirmation():
    bridge = PersonalWeChatBridge()
    bridge.process = type("RunningProcess", (), {"poll": lambda self: None})()
    bridge.process_status = {"phase": "starting", "detail": "正在初始化"}
    assert bridge.status() == {
        "running": True, "ready": False, "detail": "正在初始化", "metrics": {}
    }
    bridge.process_status = {"phase": "listening", "detail": "监听已就绪"}
    assert bridge.status()["ready"]
    bridge.process = None
    bridge.process_status = {"phase": "error", "detail": "消息网关未启动"}
    assert bridge.status() == {
        "running": False, "ready": False, "detail": "消息网关未启动", "metrics": {}
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process lifecycle")
def test_windows_job_closes_isolated_bridge_child():
    job = PersonalWeChatBridge._process_job()
    assert job is not None
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    bridge = PersonalWeChatBridge()
    try:
        bridge._assign_process_job(job, process)
        bridge.process_job = job
        bridge._close_process_job()
        assert process.wait(timeout=4) is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        bridge._close_process_job()


@pytest.mark.asyncio
async def test_bridge_start_failure_preserves_safe_process_detail(monkeypatch):
    bridge = PersonalWeChatBridge()

    async def check():
        return {"ok": True, "detail": ""}

    monkeypatch.setattr(bridge, "check", check)
    monkeypatch.setattr(bridge, "_process_job", lambda: None)

    def read_status(process):
        bridge.process_status = {"phase": "error", "detail": "消息网关未启动"}

    monkeypatch.setattr(bridge, "_read_process_status", read_status)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: type(
            "FailedProcess",
            (),
            {
                "pid": 123,
                "stdout": None,
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout=None: 1,
            },
        )(),
    )
    with pytest.raises(RuntimeError, match="消息网关未启动"):
        await bridge.set_running(True)
    assert not bridge.status()["running"]
    assert bridge.status()["detail"] == "消息网关未启动"


def settings(**changes):
    return PersonalWeChatSettings(
        enabled=True,
        bridge_token=TOKEN,
        account_wxid="wxid_test_account",
        allowed_chats=["wxid_test_friend"],
        test_account_confirmed=True,
        **changes,
    )


def payload(**changes):
    return {
        "chat_id": "wxid_test_friend",
        "user_id": "wxid_test_friend",
        "text": "同意",
        "msg_id": "wxid_test_friend:0:1",
        **changes,
    }


def test_reference_json_is_normalized_without_missing_identity():
    incoming = bridge_message(
        {
            "chatId": "wxid_test_friend",
            "senderId": "wxid_test_friend",
            "content": "hello",
            "msgId": "stable-message",
            "atMe": False,
        },
        settings(),
    )
    assert incoming.channel == "personal_wechat"
    assert incoming.event_id == "stable-message"
    for invalid in (payload(msg_id=""), payload(chat_id="unknown"), payload(from_bot=True)):
        with pytest.raises(ValueError):
            bridge_message(invalid, settings())
    with pytest.raises(ValueError, match="mention"):
        bridge_message(payload(chat_type="group", at_me=True), settings())


def test_bridge_auth_whitelist_consent_and_duplicate(tmp_path: Path):
    processor, _ = __import__("asyncio").run(build_processor(tmp_path))
    with TestClient(create_personal_wechat_app(processor, settings())) as client:
        assert client.post("/webhook/wechat", json=payload()).status_code == 403
        headers = {"x-bot-token": TOKEN}
        response = client.post("/webhook/wechat", headers=headers, json=payload())
        assert response.status_code == 200
        assert response.json()["action"] == "reply"
        assert "同意" in response.json()["reply"]
        assert client.post("/webhook/wechat", headers=headers, json=payload()).json() == {
            "ok": True, "action": "ignore", "reason": "duplicate"
        }
        ignored = client.post(
            "/webhook/wechat", headers=headers, json=payload(chat_id="not-allowed")
        )
        assert ignored.json()["action"] == "ignore"
    assert TOKEN not in json.dumps(response.json())
    __import__("asyncio").run(processor.service.close())


def test_desktop_bridge_check_never_imports_or_controls_wechat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ECO_MESSAGING_HOME", str(tmp_path / "messaging"))
    with TestClient(create_pet_app(tmp_path)) as client:
        token = client.get("/api/pet/bootstrap").json()["token"]
        status = client.get("/api/pet/messaging/wechat-bridge").json()
        assert not status["running"]
        check = client.post(
            "/api/pet/messaging/wechat-bridge/check",
            headers={"X-Pet-Token": token},
            json={},
        )
        assert check.status_code == 200
        assert not check.json()["ok"]
        start = client.post(
            "/api/pet/messaging/wechat-bridge",
            headers={"X-Pet-Token": token},
            json={"running": True},
        )
        assert start.status_code == 400
        assert not client.get("/api/pet/messaging/wechat-bridge").json()["running"]
