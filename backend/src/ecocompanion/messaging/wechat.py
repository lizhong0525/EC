"""Official Account text callbacks with authentication, ACK and customer-service reply."""

from __future__ import annotations

import hashlib
import hmac
from contextlib import asynccontextmanager
from time import monotonic
from typing import Any
from xml.etree import ElementTree

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from ecocompanion.messaging.processor import (
    InboundMessage,
    MessageProcessor,
    MessageWorker,
    ReplyTransport,
)
from ecocompanion.messaging.settings import WeChatSettings

WECHAT_API = "https://api.weixin.qq.com"


class CallbackError(ValueError):
    pass


def verify_signature(token: str, signature: str, timestamp: str, nonce: str) -> None:
    if not timestamp.isdigit() or len(timestamp) > 20 or len(nonce) > 128:
        raise CallbackError("invalid WeChat callback fields")
    expected = hashlib.sha1("".join(sorted((token, timestamp, nonce))).encode()).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise CallbackError("invalid WeChat callback signature")


def parse_text_message(raw: bytes) -> InboundMessage:
    if len(raw) > 65_536:
        raise CallbackError("oversized WeChat callback")
    try:
        # ElementTree does not resolve external entities. DTD/entity declarations
        # are rejected explicitly rather than passed to a general-purpose parser.
        if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
            raise CallbackError("DTD is not accepted")
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise CallbackError("invalid XML") from exc
    if root.tag != "xml":
        raise CallbackError("invalid XML root")
    fields: dict[str, str] = {}
    for child in root:
        if (
            child.tag
            not in {
                "ToUserName",
                "FromUserName",
                "CreateTime",
                "MsgType",
                "Content",
                "MsgId",
            }
            or child.text is None
        ):
            continue
        fields[child.tag] = child.text
    if fields.get("MsgType") != "text":
        raise CallbackError("unsupported WeChat message type")
    user, event_id = fields.get("FromUserName"), fields.get("MsgId")
    if not user or not event_id or len(user) > 128 or len(event_id) > 64:
        raise CallbackError("missing message identity")
    if not event_id.isdigit():
        raise CallbackError("invalid WeChat message id")
    text = fields.get("Content", "")
    timestamp = fields.get("CreateTime", "")
    if timestamp and (not timestamp.isdigit() or int(timestamp) > 4102444800):
        raise CallbackError("invalid message timestamp")
    return InboundMessage("wechat", user, event_id, text, timestamp)


class WeChatApi:
    def __init__(
        self,
        settings: WeChatSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.http = httpx.AsyncClient(transport=transport, trust_env=False, timeout=15)
        self._token = ""
        self._expires = 0.0

    async def close(self) -> None:
        await self.http.aclose()

    async def access_token(self) -> str:
        if not self.settings.app_id or not self.settings.app_secret:
            raise RuntimeError("WeChat app credentials are required for customer-service replies")
        if self._token and monotonic() < self._expires - 60:
            return self._token
        response = await self.http.get(
            f"{WECHAT_API}/cgi-bin/token",
            params={
                "grant_type": "client_credential",
                "appid": self.settings.app_id,
                "secret": self.settings.app_secret,
            },
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body.get("access_token"), str):
            raise RuntimeError("WeChat access token request failed")
        self._token = body["access_token"]
        self._expires = monotonic() + min(float(body.get("expires_in", 7200)), 7200)
        return self._token

    async def send_text(self, user_openid: str, text: str) -> None:
        if not user_openid or len(user_openid) > 128:
            raise ValueError("invalid WeChat openid")
        for index in range(0, len(text), 2000):
            response = await self.http.post(
                f"{WECHAT_API}/cgi-bin/message/custom/send",
                params={"access_token": await self.access_token()},
                json={
                    "touser": user_openid,
                    "msgtype": "text",
                    "text": {"content": text[index : index + 2000]},
                },
            )
            response.raise_for_status()
            body = response.json()
            if body.get("errcode", 0) != 0:
                raise RuntimeError("WeChat customer-service message was rejected")


class WeChatTransport(ReplyTransport):
    def __init__(self, api: WeChatApi) -> None:
        self.api = api

    async def send(self, message: InboundMessage, text: str) -> None:
        await self.api.send_text(message.external_user, text)


class SyntheticTransport(ReplyTransport):
    """Local-only smoke-test sink; it never sends messages to a real account."""

    def __init__(self) -> None:
        self.sent: list[tuple[InboundMessage, str]] = []

    async def send(self, message: InboundMessage, text: str) -> None:
        self.sent.append((message, text))


def create_wechat_app(
    processor: MessageProcessor,
    settings: WeChatSettings,
    worker: MessageWorker,
    transport: ReplyTransport,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker.start()
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(
        title="EcoCompanion Official Account callback",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    if settings.allow_ip_cidr:
        import ipaddress

        network = ipaddress.ip_network(settings.allow_ip_cidr, strict=False)
    else:
        network = None

    @app.middleware("http")
    async def callback_boundary(request: Request, call_next: Any):
        if network is not None:
            import ipaddress

            try:
                if ipaddress.ip_address(request.client.host) not in network:
                    return Response(status_code=403)
            except ValueError:
                return Response(status_code=403)
        if request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > 65_536:
                    return Response(status_code=413)
            except ValueError:
                return Response(status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def authenticated(request: Request) -> None:
        try:
            verify_signature(
                settings.token,
                request.query_params.get("signature", ""),
                request.query_params.get("timestamp", ""),
                request.query_params.get("nonce", ""),
            )
        except CallbackError as exc:
            raise HTTPException(403, "callback authentication failed") from exc

    @app.get("/wechat/callback")
    async def validate(request: Request) -> str:
        if not settings.enabled:
            raise HTTPException(404)
        authenticated(request)
        return request.query_params.get("echostr", "")

    @app.post("/wechat/callback")
    async def receive(request: Request) -> Response:
        if not settings.enabled:
            raise HTTPException(404)
        authenticated(request)
        if request.query_params.get("encrypt_type") not in {None, "", "raw"}:
            # AES compatibility mode needs its own verified msg_signature flow.
            raise HTTPException(400, "use plaintext callback mode for this preview")
        try:
            message = parse_text_message(await request.body())
        except CallbackError:
            # Validly signed non-text messages receive an ACK without model inference.
            return Response("success", media_type="text/plain")
        if await processor.seen(message):
            return Response("success", media_type="text/plain")
        if not processor.accept(message, monotonic()):
            return Response("success", media_type="text/plain")
        if not worker.submit(message, transport):
            return Response("success", media_type="text/plain")
        return Response("success", media_type="text/plain")

    return app
