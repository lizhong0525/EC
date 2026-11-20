import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ecocompanion.application.repository import SqlAlchemySessionRepository
from ecocompanion.application.service import CompanionService
from ecocompanion.infrastructure import CostLedger, JsonlTraceStore
from ecocompanion.messaging.processor import (
    NOTICE,
    InboundMessage,
    MessageProcessor,
    MessageWorker,
)
from ecocompanion.messaging.qq import GatewayPayload, QQApi, QQRunner, c2c_message
from ecocompanion.messaging.settings import QQSettings, WeChatSettings
from ecocompanion.messaging.wechat import (
    CallbackError,
    SyntheticTransport,
    WeChatApi,
    create_wechat_app,
    parse_text_message,
    verify_signature,
)
from ecocompanion.pet.pipeline import PetRenderer
from ecocompanion.pet.settings import PetSettings, SettingsStore, SettingsUpdate


def message(text="今天有一点累", user="QQ-USER_1", event="ROBOT.1_event"):
    return InboundMessage("qq", user, event, text)


async def build_processor(tmp_path: Path):
    store = SettingsStore(tmp_path / "desktop")
    store.update(SettingsUpdate(settings=PetSettings(qwen_enabled=False)))
    repository = SqlAlchemySessionRepository(f"sqlite+aiosqlite:///{tmp_path / 'messages.db'}")
    service = CompanionService(
        repository=repository,
        renderer=PetRenderer(store),
        trace_store=JsonlTraceStore(tmp_path / "trace.jsonl"),
        cost_ledger=CostLedger(path=tmp_path / "cost.jsonl"),
    )
    await service.initialize()
    return MessageProcessor(service, store, repository=repository), repository


@pytest.mark.asyncio
async def test_consent_new_conversation_isolation_and_confirmed_deletion(tmp_path: Path):
    processor, repository = await build_processor(tmp_path)
    first = message()
    assert "同意" in await processor.handle(first)
    assert "同意" in await processor.handle(first)
    assert "可以" in await processor.handle(message("同意", event="event-consent"))
    original = await processor.handle(message(event="event-1"))
    assert original
    assert await processor.handle(message(event="event-1")) == original
    assert await processor.handle(message("新对话", event="event-new")) == (
        "已新建一组对话。之前的本地记录仍然保留。"
    )
    binding = await repository.get_channel_binding("qq", first.external_user)
    assert binding is not None
    assert (await processor.handle(message("删除本机记录", event="event-delete"))).startswith(
        "此操作"
    )
    assert "确认删除" in await processor.handle(message("取消", event="event-cancel"))
    assert "平台已有消息" in await processor.handle(message("确认删除", event="event-confirm"))
    assert await repository.get_channel_binding("qq", first.external_user) is None
    assert await repository.get(binding.session_id) is None
    assert await processor.handle(first) == NOTICE
    await processor.service.close()


@pytest.mark.asyncio
async def test_different_users_never_share_history_or_deletion(tmp_path: Path):
    processor, repository = await build_processor(tmp_path)
    for user in ("USER-A", "USER-B"):
        await processor.handle(message("同意", user, f"consent-{user}"))
    await processor.handle(message("我的旅行预算", "USER-A", "a1"))
    await processor.handle(message("我的读书笔记", "USER-B", "b1"))
    a = await repository.get_channel_binding("qq", "USER-A")
    b = await repository.get_channel_binding("qq", "USER-B")
    assert a is not None and b is not None and a.session_id != b.session_id
    await processor.handle(message("删除本机记录", "USER-A", "a2"))
    await processor.handle(message("确认删除", "USER-A", "a3"))
    assert await repository.get_channel_binding("qq", "USER-A") is None
    assert await repository.get(b.session_id) is not None
    assert (await repository.list_messages(b.session_id)) == ()
    await processor.service.close()


@pytest.mark.asyncio
async def test_every_command_retry_recovers_the_exact_saved_reply(tmp_path: Path):
    processor, _ = await build_processor(tmp_path)
    for text, event in (
        ("同意", "consent"),
        ("新对话", "new"),
        ("删除本机记录", "delete"),
        ("取消", "cancel"),
        ("帮助", "help"),
    ):
        incoming = message(text, event=event)
        first = await processor.handle(incoming)
        assert first
        assert await processor.handle(incoming) == first
    await processor.service.close()


@pytest.mark.asyncio
async def test_rate_limit_is_per_user(tmp_path: Path):
    processor, _ = await build_processor(tmp_path)
    for index in range(6):
        assert processor.accept(message(event=f"rate-{index}"), 101)
    assert not processor.accept(message(event="limited"), 101)
    assert processor.accept(message(user="other"), 101)
    assert processor.accept(message(event="later"), 162)
    await processor.service.close()


def test_qq_parse_only_c2c_text_and_reject_bot_senders():
    payload = GatewayPayload.parse(
        json.dumps(
            {
                "op": 0,
                "s": 22,
                "t": "C2C_MESSAGE_CREATE",
                "id": "outer",
                "d": {
                    "id": "ROBOT.1_event",
                    "content": "你好",
                    "author": {"user_openid": "QQ-USER_1"},
                },
            }
        )
    )
    assert c2c_message(payload) == message("你好")
    assert (
        c2c_message(GatewayPayload.parse('{"op":0,"t":"GROUP_AT_MESSAGE_CREATE","d":{}}')) is None
    )
    bot = GatewayPayload.parse(
        json.dumps(
            {
                "op": 0,
                "t": "C2C_MESSAGE_CREATE",
                "d": {"id": "id", "author": {"id": "bot", "bot": True}},
            }
        )
    )
    assert c2c_message(bot) is None
    with pytest.raises(ValueError):
        GatewayPayload.parse(b"binary")


@pytest.mark.asyncio
async def test_qq_token_gateway_and_chunked_reply_use_official_contract():
    requests = []

    def transport(request: httpx.Request):
        requests.append(request)
        if request.url.path == "/app/getAppAccessToken":
            return httpx.Response(200, json={"access_token": "fake-token", "expires_in": 7200})
        if request.url.path == "/gateway":
            assert request.headers["authorization"] == "QQBot fake-token"
            return httpx.Response(200, json={"url": "wss://api.bot.qq.com/websocket"})
        assert request.url.path == "/v2/users/USER-1/messages"
        assert request.headers["authorization"] == "QQBot fake-token"
        return httpx.Response(200, json={"id": "reply"})

    api = QQApi(
        QQSettings(app_id="test-app", client_secret="test-only"),
        transport=httpx.MockTransport(transport),
    )
    assert await api.access_token() == "fake-token"
    assert await api.access_token() == "fake-token"
    assert await api.gateway() == "wss://api.bot.qq.com/websocket"
    await api.send_text("USER-1", "message-id", "a" * 1801)
    replies = [item for item in requests if item.url.path.endswith("/messages")]
    assert [json.loads(item.content)["msg_seq"] for item in replies] == [1, 2]
    assert len(requests) == 4
    await api.close()


class FakeSocket:
    def __init__(self, *messages):
        self.messages = list(messages)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)

    async def send(self, text):
        self.sent.append(json.loads(text))


@pytest.mark.asyncio
async def test_qq_identifies_and_resumes_with_sequence(tmp_path: Path):
    processor, _ = await build_processor(tmp_path)
    runner = QQRunner(QQApi(QQSettings()), processor)
    socket = FakeSocket(
        '{"op":10,"d":{"heartbeat_interval":45000}}',
        '{"op":0,"t":"READY","s":1,"d":{"session_id":"saved-session"}}',
        '{"op":0,"t":"RESUMED","s":2,"d":{}}',
    )
    await runner._listen(socket, "token", asyncio.Event())
    assert socket.sent[0]["op"] == 2
    assert socket.sent[0]["d"]["intents"] == 1 << 25
    assert runner.sequence == 2
    runner.session_id = "saved-session"
    resumed = FakeSocket('{"op":10,"d":{"heartbeat_interval":45000}}')
    await runner._listen(resumed, "token", asyncio.Event())
    assert resumed.sent == [
        {
            "op": 6,
            "d": {"token": "QQBot token", "session_id": "saved-session", "seq": 2},
        }
    ]
    await runner.api.close()
    await processor.service.close()


def wechat_xml(content="今天有一点累", user="wx-user", event="12345"):
    return (
        f"<xml><ToUserName><![CDATA[official-account]]></ToUserName>"
        f"<FromUserName><![CDATA[{user}]]></FromUserName>"
        f"<CreateTime>1793059200</CreateTime><MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content><MsgId>{event}</MsgId></xml>"
    ).encode()


def signed_query(content: bytes, token="test-only"):
    del content
    timestamp, nonce = "1793059200", "nonce"
    signature = hashlib.sha1("".join(sorted((token, timestamp, nonce))).encode()).hexdigest()
    return f"?signature={signature}&timestamp={timestamp}&nonce={nonce}"


def wait_for_sink(sink: SyntheticTransport):
    for _ in range(200):
        if sink.sent:
            return
        time.sleep(0.025)
    raise AssertionError("callback worker did not send its synthetic reply")


def wait_for_empty_queue(worker: MessageWorker):
    for _ in range(200):
        if worker.queue.empty():
            return
        time.sleep(0.025)
    raise AssertionError("callback queue did not settle")


def test_wechat_signature_and_xml_validation():
    signature = hashlib.sha1("".join(sorted(("test-only", "123", "n"))).encode()).hexdigest()
    verify_signature("test-only", signature, "123", "n")
    with pytest.raises(CallbackError):
        verify_signature("test-only", "bad", "123", "n")
    assert parse_text_message(wechat_xml()).external_user == "wx-user"
    for raw in (
        b"<xml><!DOCTYPE bad [<!ENTITY xxe SYSTEM 'file:///secret'>]><MsgType>text</MsgType></xml>",
        b"not-xml",
    ):
        with pytest.raises(CallbackError):
            parse_text_message(raw)


def test_wechat_signature_ack_dedup_and_no_double_send(tmp_path: Path):
    processor, _ = asyncio.run(build_processor(tmp_path))
    worker = MessageWorker(processor)
    sink = SyntheticTransport()
    app = create_wechat_app(
        processor, WeChatSettings(enabled=True, token="test-only"), worker, sink
    )
    query = signed_query(wechat_xml("同意"))
    with TestClient(app) as client:
        assert (
            client.post(
                "/wechat/callback?signature=bad&timestamp=1&nonce=n",
                content=wechat_xml("同意"),
            ).status_code
            == 403
        )
        assert client.post(f"/wechat/callback{query}", content=wechat_xml("同意")).text == "success"
        assert client.post(f"/wechat/callback{query}", content=wechat_xml("同意")).text == "success"
        wait_for_sink(sink)
        wait_for_empty_queue(worker)
        assert len(sink.sent) == 1
        assert "同意" in sink.sent[0][1]
    asyncio.run(processor.service.close())


@pytest.mark.asyncio
async def test_wechat_customer_service_api_uses_token_and_text_contract():
    requests = []

    def transport(request: httpx.Request):
        requests.append(request)
        if request.url.path == "/cgi-bin/token":
            return httpx.Response(200, json={"access_token": "fake-token", "expires_in": 7200})
        assert request.url.params["access_token"] == "fake-token"
        assert json.loads(request.content)["text"]["content"] == "hello"
        return httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})

    api = WeChatApi(
        WeChatSettings(app_id="test-app", app_secret="test-only"),
        transport=httpx.MockTransport(transport),
    )
    await api.send_text("wx-user", "hello")
    assert len(requests) == 2
    await api.close()


@pytest.mark.asyncio
async def test_messaging_settings_do_not_expose_secrets_and_survive_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ecocompanion.messaging.settings import (
        MessagingSettings,
        MessagingSettingsStore,
        MessagingSettingsUpdate,
        load_settings,
    )

    monkeypatch.setenv("ECO_MESSAGING_HOME", str(tmp_path / "unrelated"))
    store = MessagingSettingsStore(tmp_path / "messages")
    settings = MessagingSettings(qq={"app_id": "test-app"}, wechat={"app_id": "wx-app"})
    view = store.update(
        MessagingSettingsUpdate(
            settings=settings,
            qq_client_secret="test-only",
            wechat_token="test-only",
            wechat_app_secret="test-only",
        )
    )
    assert view["qq_secret_configured"]
    assert "test-only" not in json.dumps(view)
    assert "test-only" not in (tmp_path / "messages/messaging.json").read_text() or os.name != "nt"
    restored = load_settings(tmp_path / "messages")
    assert restored.qq.client_secret == "test-only"
    assert restored.wechat.token == "test-only"


def test_wechat_ack_stays_fast_while_model_is_processing(tmp_path: Path):
    processor, _ = asyncio.run(build_processor(tmp_path))

    class SlowProcessor(type(processor)):
        async def handle(self, incoming):
            await asyncio.sleep(0.4)
            return await super().handle(incoming)

    slow = SlowProcessor(processor.service, processor.store, repository=processor.repository)
    sink = SyntheticTransport()
    app = create_wechat_app(
        slow, WeChatSettings(enabled=True, token="test-only"), MessageWorker(slow), sink
    )
    query = signed_query(wechat_xml("同意"))
    with TestClient(app) as client:
        start = time.monotonic()
        assert client.post(f"/wechat/callback{query}", content=wechat_xml("同意")).text == "success"
        assert client.post(f"/wechat/callback{query}", content=wechat_xml("同意")).text == "success"
        assert time.monotonic() - start < 0.3
        assert not sink.sent
        wait_for_sink(sink)
        assert len(sink.sent) == 1
    asyncio.run(processor.service.close())
