"""Durable, consent-gated message processing shared by official channels."""

from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from ecocompanion.application.models import CreateSessionRequest, TurnRequest
from ecocompanion.application.repository import (
    ChannelBinding,
    SqlAlchemySessionRepository,
)
from ecocompanion.application.service import CompanionService, SessionNotFound, TurnConflictError
from ecocompanion.pet.deepseek import failure_detail
from ecocompanion.pet.pipeline import TurnContext, turn_context
from ecocompanion.pet.settings import SettingsStore

NOTICE = (
    "我是心伴 AI 心理支持助手，仅用于成年测试者的支持性对话，不作诊断，也不是紧急救助服务。"
    "消息会经过所选平台；若桌面端已允许云端模型，内容还会发送给已启用的 DeepSeek / Jev 服务。"
    "请只发送你愿意保存和按上述方式处理的内容。回复“同意”开始；"
    "可发送“新对话”切换新对话，或“删除本机记录”申请删除本机副本。"
)
WELCOME = "已记录你的同意。可以开始聊了。发送“新对话”另起一组对话，或“删除本机记录”管理本机副本。"


@dataclass(frozen=True, slots=True)
class InboundMessage:
    channel: str
    external_user: str
    event_id: str
    text: str
    timestamp: str = ""


class ReplyTransport(Protocol):
    async def send(self, message: InboundMessage, text: str) -> None: ...


class MessageProcessor:
    def __init__(
        self,
        service: CompanionService,
        store: SettingsStore,
        *,
        repository: SqlAlchemySessionRepository | None = None,
        per_user_per_minute: int = 6,
    ) -> None:
        self.service = service
        self.store = store
        self.repository = repository or service._repository
        self.per_user_per_minute = per_user_per_minute
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._recent: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._pending_receipts: set[tuple[str, str, str]] = set()

    def accept(self, message: InboundMessage, now: float) -> bool:
        key = (message.channel, message.external_user)
        recent = self._recent[key]
        while recent and recent[0] <= now - 60:
            recent.popleft()
        if len(recent) >= self.per_user_per_minute:
            return False
        event_key = (message.channel, message.external_user, message.event_id)
        if event_key in self._pending_receipts:
            return False
        self._pending_receipts.add(event_key)
        recent.append(now)
        return True

    async def seen(self, message: InboundMessage) -> bool:
        return (
            await self.repository.get_channel_receipt(
                message.channel, message.external_user, message.event_id, self._request_id(message)
            )
            is not None
        )

    async def handle(self, message: InboundMessage) -> str:
        key = (message.channel, message.external_user)
        lock = self._locks.setdefault(key, asyncio.Lock())
        request_id = self._request_id(message)
        async with lock:
            saved = await self.repository.get_channel_receipt(*key, message.event_id, request_id)
            if saved is not None:
                return saved
            # A second process must not generate the same event concurrently.
            claimed = await self.repository.claim_channel_receipt(
                channel=message.channel,
                external_user=message.external_user,
                event_id=message.event_id,
                request_id=request_id,
                created_at=datetime.now(UTC),
            )
            if not claimed:
                return (
                    await self.repository.get_channel_receipt(*key, message.event_id, request_id)
                    or ""
                )
            try:
                text = " ".join(message.text.split())
                binding = await self.repository.get_channel_binding(*key)
                if binding is None or binding.delete_pending:
                    response = await self._without_consent(key, text, binding)
                else:
                    response = await self._consented(message, text, binding)
            except Exception:
                # Leave no completed claim for an unforeseen failure. Model
                # exceptions are converted into a reply by _consented below.
                await self.repository.delete_channel_receipt(*key, message.event_id)
                raise
            # Save before transport I/O so an uncertain send cannot cause another
            # command response or model invocation when the platform retries.
            await self.repository.save_channel_receipt(
                channel=message.channel,
                external_user=message.external_user,
                event_id=message.event_id,
                request_id=request_id,
                reply=response,
                created_at=datetime.now(UTC),
            )
            return response

    async def mark_sent(self, message: InboundMessage) -> None:
        await self.repository.save_channel_receipt(
            channel=message.channel,
            external_user=message.external_user,
            event_id=message.event_id,
            request_id=self._request_id(message),
            reply=await self.get_reply(message),
            created_at=datetime.now(UTC),
        )

    async def get_reply(self, message: InboundMessage) -> str:
        return (
            await self.repository.get_channel_receipt(
                message.channel,
                message.external_user,
                message.event_id,
                self._request_id(message),
            )
            or ""
        )

    @staticmethod
    def _request_id(message: InboundMessage) -> str:
        return hashlib.sha256(
            f"{message.channel}\0{message.external_user}\0{message.event_id}".encode()
        ).hexdigest()

    async def _without_consent(
        self,
        key: tuple[str, str],
        text: str,
        binding: ChannelBinding | None,
    ) -> str:
        if text == "确认删除":
            if binding is not None:
                await self.repository.delete_channel_history(*key)
            return (
                "已删除此平台用户在本机的对话、草稿与渠道绑定；"
                "平台已有消息不会因此撤回。重新使用需再次同意。"
            )
        if binding is not None and binding.delete_pending:
            return "删除尚未确认。发送“确认删除”删除本机记录，发送“取消”撤销本次申请。"
        if text in {"同意", "/agree"}:
            return await self._new_binding(*key, consent=True, message=WELCOME)
        return NOTICE

    async def _consented(self, message: InboundMessage, text: str, binding: ChannelBinding) -> str:
        key = (message.channel, message.external_user)
        if text in {"新对话", "/new"}:
            return await self._new_binding(
                *key, consent=True, message="已新建一组对话。之前的本地记录仍然保留。"
            )
        if text in {"删除本机记录", "/delete"}:
            await self.repository.save_channel_binding(
                ChannelBinding(
                    *key, session_id=binding.session_id, consent=True, delete_pending=True
                )
            )
            return (
                "此操作将删除本机对话与渠道绑定，不影响平台已有消息。"
                "发送“确认删除”继续，或“取消”撤销。"
            )
        if text in {"取消", "/cancel"}:
            return "当前没有待确认操作，可以继续对话。"
        if text in {"帮助", "/help", "/start"}:
            return NOTICE
        if not text:
            return "目前只支持文字消息。你可以重新发送一段文字。"
        if len(text) > 2000:
            return "这条消息较长，请先分成 2000 字以内的几段发送。"
        try:
            context = TurnContext(self.store.settings, dict(self.store.keys))
            marker = turn_context.set(context)
            try:
                await self.service.process_turn(
                    binding.session_id,
                    TurnRequest(
                        text=message.text,
                        request_id=self._request_id(message),
                    ),
                )
            finally:
                turn_context.reset(marker)
        except TurnConflictError:
            recovered = await self.service.recover_turn(
                binding.session_id,
                TurnRequest(
                    text=message.text,
                    request_id=self._request_id(message),
                ),
            )
            if recovered is None:
                return "这条消息已在处理，请稍后重试。"
            return recovered[1]
        except (SessionNotFound, KeyError):
            await self.repository.delete_channel_history(*key)
            return "本地绑定已失效，旧记录不可恢复。回复“同意”可以重新开始。"
        except Exception as exc:
            return f"DeepSeek 回复未完成：{failure_detail(exc)}"
        return await self._saved_reply(binding.session_id)

    async def _new_binding(
        self, channel: str, external_user: str, *, consent: bool, message: str
    ) -> str:
        session = await self.service.create_session(
            CreateSessionRequest(memory_enabled=False, research_mode=False)
        )
        await self.repository.save_channel_binding(
            ChannelBinding(channel, external_user, session.session_id, consent)
        )
        return message

    async def _saved_reply(self, session_id: str) -> str:
        messages = await self.service.list_messages(session_id)
        if not messages:
            raise RuntimeError("a committed messaging turn has no reply")
        return messages[-1].content


class MessageWorker:
    """Acknowledge callbacks quickly; a bounded worker owns model latency."""

    def __init__(
        self,
        processor: MessageProcessor,
        *,
        queue_size: int = 32,
    ) -> None:
        self.processor = processor
        self.queue: asyncio.Queue[tuple[InboundMessage, ReplyTransport] | None] = asyncio.Queue(
            maxsize=queue_size
        )
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run(), name="official-message-worker")

    def submit(self, message: InboundMessage, transport: ReplyTransport) -> bool:
        try:
            self.queue.put_nowait((message, transport))
            return True
        except asyncio.QueueFull:
            return False

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            message, transport = item
            try:
                reply = await self.processor.handle(message)
                if reply:
                    await transport.send(message, reply)
                    await self.processor.mark_sent(message)
            except Exception:
                # Do not log message contents or credentials. The platform may retry.
                continue
            finally:
                self.processor._pending_receipts.discard(
                    (message.channel, message.external_user, message.event_id)
                )
                self.queue.task_done()
