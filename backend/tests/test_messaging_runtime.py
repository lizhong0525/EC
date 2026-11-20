import asyncio
import socket
from pathlib import Path

import httpx
import pytest

from ecocompanion.messaging.processor import MessageWorker
from ecocompanion.messaging.runtime import run
from ecocompanion.messaging.settings import (
    MessagingSettings,
    MessagingSettingsStore,
    MessagingSettingsUpdate,
)
from ecocompanion.messaging.wechat import SyntheticTransport


@pytest.mark.asyncio
async def test_worker_close_cancels_pending_inference(tmp_path: Path):
    from backend.tests.test_messaging import build_processor, message

    processor, _ = await build_processor(tmp_path)
    worker = MessageWorker(processor)
    sink = SyntheticTransport()
    worker.start()
    worker.submit(message("同意"), sink)
    await asyncio.sleep(0)
    await worker.close()
    assert worker.task is not None and worker.task.done()
    await processor.service.close()


@pytest.mark.asyncio
async def test_disabled_gateway_refuses_to_start(tmp_path: Path):
    with pytest.raises(RuntimeError, match="至少一个消息渠道"):
        await run(tmp_path)


@pytest.mark.asyncio
async def test_legacy_official_account_cannot_start_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ECO_PET_HOME", str(tmp_path / "desktop"))
    store = MessagingSettingsStore(tmp_path)
    store.update(
        MessagingSettingsUpdate(
            settings=MessagingSettings(
                wechat={"enabled": True, "app_id": "test-app"}
            ),
            wechat_token="test-only",
            wechat_app_secret="test-only",
        )
    )
    with pytest.raises(RuntimeError, match="至少一个消息渠道"):
        await run(tmp_path)


@pytest.mark.asyncio
async def test_personal_wechat_gateway_round_trip_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ECO_PET_HOME", str(tmp_path / "desktop"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    store = MessagingSettingsStore(tmp_path)
    view = store.update(
        MessagingSettingsUpdate(
            settings=MessagingSettings(
                wechat={"enabled": True, "app_id": "legacy-app"},
                personal_wechat={
                    "enabled": True,
                    "account_wxid": "wxid_test_bot",
                    "allowed_chats": ["wxid_test_friend"],
                    "test_account_confirmed": True,
                },
                callback_port=port,
            ),
            wechat_token="test-only",
            wechat_app_secret="test-only",
        )
    )
    assert view["personal_wechat_token_configured"]
    token = store.settings.personal_wechat.bridge_token
    from ecocompanion.messaging.runtime import MessagingRuntime

    runtime = MessagingRuntime(tmp_path)
    await runtime.start()
    assert runtime.channels == ["personal_wechat"]
    url = f"http://127.0.0.1:{port}/webhook/wechat"
    message = {
        "chat_id": "wxid_test_friend",
        "user_id": "wxid_test_friend",
        "msg_id": "stable-test-1",
        "text": "同意",
    }
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(50):
                try:
                    response = await client.post(
                        url, json=message, headers={"x-bot-token": token}
                    )
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("personal bridge did not start")
            assert "同意" in response.json()["reply"]
            assert (await client.get(f"http://127.0.0.1:{port}/wechat/callback")).status_code == 404
            duplicate = await client.post(url, json=message, headers={"x-bot-token": token})
            assert duplicate.json()["reason"] == "duplicate"
    finally:
        await runtime.stop()
    assert not runtime.running
