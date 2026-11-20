"""Authenticated local JSON bridge; no WeChat process or database access here."""

from __future__ import annotations

import hmac
from time import monotonic
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ecocompanion.messaging.processor import InboundMessage, MessageProcessor
from ecocompanion.messaging.settings import PersonalWeChatSettings


class BridgeMessage(BaseModel):
    """The small common subset of the whale_gril gateway JSON contract."""

    model_config = ConfigDict(extra="ignore")
    text: str = Field(max_length=8_000)
    chat_id: str = Field(max_length=128)
    user_id: str = Field(max_length=128)
    msg_id: str = Field(min_length=1, max_length=240)
    chat_type: str = "private"
    at_me: bool = False
    from_bot: bool = False

    @model_validator(mode="before")
    @classmethod
    def accept_reference_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data: dict[str, Any] = (
            dict(value["data"]) if isinstance(value.get("data"), dict) else dict(value)
        )
        aliases = {
            "text": ("text", "content", "message", "msg"),
            "chat_id": ("chat_id", "chatId", "room_id", "conversation_id"),
            "user_id": ("user_id", "sender_id", "senderId", "from", "wxid"),
            "msg_id": ("msg_id", "msgId", "message_id", "messageId"),
            "chat_type": ("chat_type", "chatType"),
            "at_me": ("at_me", "atMe", "is_at", "mentioned"),
            "from_bot": ("from_bot", "fromBot", "is_self", "isSelf"),
        }
        return {
            target: next(
                (data[key] for key in keys if key in data and data[key] is not None), None
            )
            for target, keys in aliases.items()
            if any(key in data and data[key] is not None for key in keys)
        }

    @model_validator(mode="after")
    def stable_identity(self) -> BridgeMessage:
        for name in ("chat_id", "user_id"):
            value = getattr(self, name).strip()
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"invalid {name}")
            setattr(self, name, value)
        self.msg_id = self.msg_id.strip()
        if self.chat_type not in {"private", "group"}:
            raise ValueError("unsupported chat type")
        return self


def bridge_message(payload: dict[str, Any], settings: PersonalWeChatSettings) -> InboundMessage:
    try:
        incoming = BridgeMessage.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("bridge message must have stable IDs and text") from exc
    if incoming.from_bot:
        raise ValueError("bot-originated messages are ignored")
    if incoming.chat_id not in settings.allowed_chats:
        raise ValueError("chat is not in the test whitelist")
    if incoming.chat_type == "group" and (not settings.allow_groups or not incoming.at_me):
        raise ValueError("groups require explicit enablement and an @ mention")
    return InboundMessage(
        "personal_wechat",
        incoming.user_id,
        incoming.msg_id,
        incoming.text,
    )


def create_personal_wechat_app(
    processor: MessageProcessor,
    settings: PersonalWeChatSettings,
) -> FastAPI:
    app = FastAPI(
        title="EcoCompanion personal WeChat test bridge",
        docs_url=None,
        redoc_url=None,
    )

    def authenticate(request: Request) -> None:
        if request.client is None or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(403, "loopback bridge only")
        supplied = request.headers.get("x-bot-token", "")
        if not hmac.compare_digest(supplied, settings.bridge_token):
            raise HTTPException(403, "bridge authentication failed")

    @app.post("/webhook/wechat")
    async def receive(request: Request) -> dict[str, Any]:
        authenticate(request)
        if int(request.headers.get("content-length", "0") or 0) > 32_768:
            raise HTTPException(413, "bridge message too large")
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            message = bridge_message(payload, settings)
        except (ValueError, TypeError) as exc:
            return {"ok": True, "action": "ignore", "reason": str(exc)[:160]}
        if await processor.seen(message):
            return {"ok": True, "action": "ignore", "reason": "duplicate"}
        if not processor.accept(message, monotonic()):
            return {"ok": True, "action": "ignore", "reason": "rate_limited"}
        try:
            reply = await processor.handle(message)
            if reply:
                await processor.mark_sent(message)
            return {
                "ok": True,
                "action": "reply" if reply else "ignore",
                "reply": reply,
                # One conservative text bubble; the transport must not fan it out.
                "bubbles": [reply] if reply else [],
            }
        finally:
            processor._pending_receipts.discard(
                (message.channel, message.external_user, message.event_id)
            )

    return app
