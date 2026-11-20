"""QQ official bot C2C transport using the documented AccessToken/WebSocket flow."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from time import monotonic
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import websockets

from ecocompanion.messaging.processor import (
    InboundMessage,
    MessageProcessor,
    MessageWorker,
    ReplyTransport,
)
from ecocompanion.messaging.settings import QQSettings

API_ORIGIN = "https://api.bot.qq.com"


class QQApi:
    def __init__(
        self,
        settings: QQSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._http = httpx.AsyncClient(transport=transport, trust_env=False, timeout=15)
        self._token = ""
        self._expires = 0.0
        self._token_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._http.aclose()

    async def access_token(self) -> str:
        if self._token and monotonic() < self._expires - 60:
            return self._token
        async with self._token_lock:
            if self._token and monotonic() < self._expires - 60:
                return self._token
            response = await self._http.post(
                f"{API_ORIGIN}/app/getAppAccessToken",
                json={"appId": self.settings.app_id, "clientSecret": self.settings.client_secret},
            )
            response.raise_for_status()
            data = response.json()
            token = data.get("access_token")
            if not isinstance(token, str) or not token:
                raise ValueError("QQ access token response was rejected")
            self._token = token
            self._expires = monotonic() + min(float(data.get("expires_in", 7200)), 7200)
            return token

    async def _headers(self) -> dict[str, str]:
        return {"Authorization": f"QQBot {await self.access_token()}"}

    async def gateway(self) -> str:
        response = await self._http.get(f"{API_ORIGIN}/gateway", headers=await self._headers())
        response.raise_for_status()
        url = response.json().get("url", "")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "wss"
            or parsed.hostname not in {"api.bot.qq.com", "api.sgroup.qq.com"}
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("QQ returned an untrusted WebSocket endpoint")
        return url

    async def send_text(
        self, user_openid: str, message_id: str, text: str, *, first_seq: int = 1
    ) -> None:
        if not user_openid or any(
            character not in ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
            for character in user_openid
        ):
            raise ValueError("invalid QQ user_openid")
        for index in range(0, len(text), 1800):
            response = await self._http.post(
                f"{API_ORIGIN}/v2/users/{quote(user_openid, safe='')}/messages",
                headers=await self._headers(),
                json={
                    "msg_type": 0,
                    "content": text[index : index + 1800],
                    "msg_id": message_id,
                    "msg_seq": index // 1800 + first_seq,
                },
            )
            response.raise_for_status()


class QQTransport(ReplyTransport):
    def __init__(self, api: QQApi) -> None:
        self.api = api

    async def send(self, message: InboundMessage, text: str) -> None:
        await self.api.send_text(message.external_user, message.event_id, text)


class QQProgressTransport(ReplyTransport):
    """Use the first passive-reply slot before slow local inference."""

    def __init__(self, transport: QQTransport) -> None:
        self.transport = transport
        self.progress_sent = False

    async def send(self, message: InboundMessage, text: str) -> None:
        if not self.progress_sent:
            await self.begin(message)
        await self.transport.api.send_text(
            message.external_user, message.event_id, text, first_seq=2
        )

    async def begin(self, message: InboundMessage) -> None:
        if not self.progress_sent:
            await self.transport.send(message, "正在整理回复，请稍等。")
            self.progress_sent = True


@dataclass(frozen=True, slots=True)
class GatewayPayload:
    op: int
    data: Any
    sequence: int | None
    event: str | None
    event_id: str | None

    @classmethod
    def parse(cls, raw: str | bytes) -> GatewayPayload:
        if isinstance(raw, bytes) or len(raw) > 200_000:
            raise ValueError("oversized or binary QQ gateway payload")
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("op"), int):
            raise ValueError("invalid QQ gateway payload")
        sequence = data.get("s")
        if sequence is not None and (not isinstance(sequence, int) or sequence < 0):
            raise ValueError("invalid QQ gateway sequence")
        return cls(data["op"], data.get("d"), sequence, data.get("t"), data.get("id"))


def c2c_message(payload: GatewayPayload) -> InboundMessage | None:
    if (
        payload.op != 0
        or payload.event != "C2C_MESSAGE_CREATE"
        or not isinstance(payload.data, dict)
    ):
        return None
    data = payload.data
    author = data.get("author")
    if not isinstance(author, dict) or author.get("bot") is True:
        return None
    user = author.get("user_openid") or author.get("id")
    event_id = data.get("id")
    text = data.get("content")
    if not isinstance(user, str) or not isinstance(event_id, str) or not user or not event_id:
        raise ValueError("missing QQ private message identity")
    if len(event_id) > 160 or len(user) > 128 or not isinstance(text, str):
        raise ValueError("invalid QQ private message")
    return InboundMessage("qq", user, event_id, text, str(data.get("timestamp", "")))


class QQRunner:
    def __init__(
        self, api: QQApi, processor: MessageProcessor, worker: MessageWorker | None = None
    ) -> None:
        self.api = api
        self.processor = processor
        self.transport = QQTransport(api)
        self.worker = worker
        self.session_id = ""
        self.sequence: int | None = None

    async def run(self, stop: asyncio.Event) -> None:
        delay = 1
        while not stop.is_set():
            try:
                token = await self.api.access_token()
                url = await self.api.gateway()
                async with websockets.connect(
                    url, proxy=None, max_size=200_000, open_timeout=15
                ) as socket:
                    delay = 1
                    await self._listen(socket, token, stop)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Credentials and message bodies are intentionally not logged.
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except TimeoutError:
                    pass
                delay = min(delay * 2, 60)

    async def _listen(self, socket: Any, token: str, stop: asyncio.Event) -> None:
        heartbeat: asyncio.Task[None] | None = None
        try:
            async for raw in socket:
                payload = GatewayPayload.parse(raw)
                if payload.sequence is not None:
                    self.sequence = payload.sequence
                if payload.op == 10:
                    if not isinstance(payload.data, dict):
                        raise ValueError("invalid QQ hello")
                    interval = payload.data.get("heartbeat_interval", 45000)
                    if not isinstance(interval, (int, float)) or not 1000 <= interval <= 120000:
                        raise ValueError("invalid QQ heartbeat interval")
                    data: dict[str, Any] = {"token": f"QQBot {token}"}
                    if self.session_id:
                        data.update(session_id=self.session_id, seq=self.sequence or 0)
                        op = 6
                    else:
                        data.update(intents=self.api.settings.intents, shard=[0, 1], properties={})
                        op = 2
                    await socket.send(json.dumps({"op": op, "d": data}))
                    heartbeat = asyncio.create_task(self._heartbeat(socket, interval / 1000))
                elif payload.op == 1:
                    await socket.send(json.dumps({"op": 1, "d": self.sequence}))
                elif payload.op == 0 and payload.event == "READY":
                    if isinstance(payload.data, dict) and isinstance(
                        payload.data.get("session_id"), str
                    ):
                        self.session_id = payload.data["session_id"]
                elif payload.op == 9:
                    self.session_id = ""
                    self.sequence = None
                    if payload.data is True:
                        raise RuntimeError("QQ session is not resumable")
                    return
                elif payload.op == 7:
                    return
                elif payload.op == 0:
                    message = c2c_message(payload)
                    if message is not None and self.processor.accept(message, monotonic()):
                        await self._respond(message)
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass

    async def _heartbeat(self, socket: Any, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            await socket.send(json.dumps({"op": 1, "d": self.sequence}))

    async def _respond(self, message: InboundMessage) -> None:
        if await self.processor.seen(message):
            return
        if self.worker is not None:
            transport = QQProgressTransport(self.transport)
            await transport.begin(message)
            if not self.worker.submit(message, transport):
                await self.transport.send(message, "当前消息较多，请稍后重新发送。")
            return
        reply = await self.processor.handle(message)
        if reply:
            await self.transport.send(message, reply)
            await self.processor.mark_sent(message)
