"""SQLite/SQLAlchemy session persistence with optimistic turn commits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from ecocompanion.application.models import ParticipantType
from ecocompanion.domain.enums import ActorRole
from ecocompanion.domain.schemas import SessionStateV1


class SessionConflictError(RuntimeError):
    """Raised when a session identifier is reused with different data."""


class ConcurrentSessionUpdate(RuntimeError):
    """Raised when a stale command attempts to overwrite newer session state."""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    participant_type: ParticipantType
    synthetic_case_id: str | None
    state: SessionStateV1
    model: str | None
    research_mode: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredMessage:
    message_id: str
    session_id: str
    ordinal: int
    role: ActorRole
    content: str
    simulated_time: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.message_id.strip():
            raise ValueError("message_id cannot be blank")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be blank")
        if self.ordinal < 0:
            raise ValueError("ordinal cannot be negative")
        if not self.content.strip():
            raise ValueError("message content cannot be blank")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


@dataclass(frozen=True, slots=True)
class ChannelBinding:
    channel: str
    external_user: str
    session_id: str
    consent: bool
    delete_pending: bool = False


class SessionRepository(Protocol):
    async def initialize(self) -> None:
        """Create storage structures without changing existing data."""

    async def create(self, record: SessionRecord) -> None:
        """Create one session idempotently."""

    async def get(self, session_id: str) -> SessionRecord | None:
        """Query one immutable session record."""

    async def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[StoredMessage, ...]:
        """Query messages in stable order."""

    async def commit_turn(
        self,
        *,
        session_id: str,
        expected_turn_index: int,
        next_state: SessionStateV1,
        messages: tuple[StoredMessage, ...],
        assessment: dict | None = None,
    ) -> None:
        """Atomically append messages and advance the authoritative state."""

    async def save_configuration(
        self,
        *,
        session_id: str,
        expected_turn_index: int,
        next_state: SessionStateV1,
    ) -> None:
        """Persist a non-turn configuration transition."""

    async def close(self) -> None:
        """Release repository resources."""


class _Base(DeclarativeBase):
    pass


class _SessionRow(_Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    participant_type: Mapped[str] = mapped_column(String(48), nullable=False)
    synthetic_case_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    research_mode: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _MessageRow(_Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("session_id", "ordinal", name="uq_message_session_ordinal"),)

    message_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    simulated_time: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _ConversationUiRow(_Base):
    __tablename__ = "conversation_ui"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), primary_key=True
    )
    title: Mapped[str | None] = mapped_column(String(48), nullable=True)
    draft: Mapped[str] = mapped_column(Text, nullable=False, default="")


class _ChannelBindingRow(_Base):
    __tablename__ = "channel_bindings"

    channel: Mapped[str] = mapped_column(String(16), primary_key=True)
    external_user: Mapped[str] = mapped_column(String(160), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    delete_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class _ChannelReceiptRow(_Base):
    __tablename__ = "channel_receipts"

    channel: Mapped[str] = mapped_column(String(16), primary_key=True)
    external_user: Mapped[str] = mapped_column(String(160), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reply: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _AssessmentRow(_Base):
    __tablename__ = "turn_assessments"

    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.message_id", ondelete="CASCADE"), primary_key=True
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SqlAlchemySessionRepository:
    """The sole persistent source for conversation state and message history."""

    def __init__(
        self,
        database_url: str,
        *,
        engine: AsyncEngine | None = None,
    ) -> None:
        if not database_url.startswith("sqlite+aiosqlite://"):
            raise ValueError("only sqlite+aiosqlite database URLs are supported in V1")
        self._owns_engine = engine is None
        if engine is None:
            engine_kwargs: dict[str, object] = {}
            if database_url.endswith(":memory:"):
                engine_kwargs["poolclass"] = StaticPool
                engine_kwargs["connect_args"] = {"check_same_thread": False}
            engine = create_async_engine(database_url, **engine_kwargs)
        self._engine = engine
        self._sessions = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def initialize(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)
            columns = await connection.run_sync(
                lambda sync: {
                    item["name"] for item in inspect(sync).get_columns("channel_receipts")
                }
            )
            if columns and "reply" not in columns:
                await connection.execute(
                    text("ALTER TABLE channel_receipts ADD COLUMN reply TEXT NOT NULL DEFAULT ''")
                )

    async def create(self, record: SessionRecord) -> None:
        self._validate_record(record)
        async with self._sessions.begin() as session:
            existing = await session.get(_SessionRow, record.session_id)
            if existing is not None:
                if self._record_from_row(existing) != record:
                    raise SessionConflictError(f"session_id {record.session_id!r} already exists")
                return
            session.add(
                _SessionRow(
                    session_id=record.session_id,
                    participant_type=record.participant_type.value,
                    synthetic_case_id=record.synthetic_case_id,
                    state_json=record.state.model_dump_json(),
                    model=record.model,
                    research_mode=record.research_mode,
                    created_at=record.created_at,
                    updated_at=record.state.updated_at,
                )
            )

    async def get(self, session_id: str) -> SessionRecord | None:
        if not session_id.strip():
            raise ValueError("session_id cannot be blank")
        async with self._sessions() as session:
            row = await session.get(_SessionRow, session_id)
            return None if row is None else self._record_from_row(row)

    async def search_sessions(
        self, *, query: str = "", limit: int = 40, offset: int = 0
    ) -> tuple[tuple[SessionSummary, ...], int]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("invalid pagination")
        first_user = (
            select(_MessageRow.content)
            .where(
                _MessageRow.session_id == _SessionRow.session_id,
                _MessageRow.role == ActorRole.USER.value,
            )
            .order_by(_MessageRow.ordinal)
            .limit(1)
            .correlate(_SessionRow)
            .scalar_subquery()
        )
        message_count = (
            select(func.count())
            .select_from(_MessageRow)
            .where(_MessageRow.session_id == _SessionRow.session_id)
            .correlate(_SessionRow)
            .scalar_subquery()
        )
        custom_title = (
            select(_ConversationUiRow.title)
            .where(_ConversationUiRow.session_id == _SessionRow.session_id)
            .correlate(_SessionRow)
            .scalar_subquery()
        )
        title = func.coalesce(custom_title, first_user, "新对话")
        conditions = []
        if query.strip():
            # Escape SQL wildcards; search literal text without loading transcripts.
            conditions.append(title.contains(query.strip(), autoescape=True))
        statement = (
            select(_SessionRow, title, message_count)
            .where(*conditions)
            .order_by(
                _SessionRow.updated_at.desc(),
                _SessionRow.created_at.desc(),
                _SessionRow.session_id,
            )
            .offset(offset)
            .limit(limit)
        )
        async with self._sessions() as session:
            total = await session.scalar(
                select(func.count()).select_from(_SessionRow).where(*conditions)
            )
            rows = (await session.execute(statement)).all()
        return (
            tuple(
                SessionSummary(
                    session_id=row.session_id,
                    title=" ".join(first.split())[:48] if first else "新对话",
                    created_at=self._aware(row.created_at),
                    updated_at=self._aware(row.updated_at),
                    message_count=count,
                )
                for row, first, count in rows
            ),
            total or 0,
        )

    async def get_draft(self, session_id: str) -> str:
        async with self._sessions() as session:
            return (
                await session.scalar(
                    select(_ConversationUiRow.draft).where(
                        _ConversationUiRow.session_id == session_id
                    )
                )
                or ""
            )

    async def save_draft(self, session_id: str, text: str) -> None:
        if len(text) > 8_000:
            raise ValueError("draft is too long")
        async with self._sessions.begin() as session:
            if await session.get(_SessionRow, session_id) is None:
                raise KeyError(session_id)
            row = await session.get(_ConversationUiRow, session_id)
            if row is None:
                row = _ConversationUiRow(session_id=session_id, draft=text)
                session.add(row)
            else:
                row.draft = text

    async def rename_session(self, session_id: str, title: str) -> None:
        title = " ".join(title.split())
        if not title or len(title) > 48:
            raise ValueError("title must contain 1-48 characters")
        async with self._sessions.begin() as session:
            if await session.get(_SessionRow, session_id) is None:
                raise KeyError(session_id)
            row = await session.get(_ConversationUiRow, session_id)
            if row is None:
                row = _ConversationUiRow(session_id=session_id, title=title, draft="")
                session.add(row)
            else:
                row.title = title

    async def delete_session(self, session_id: str) -> bool:
        async with self._sessions.begin() as session:
            row = await session.get(_SessionRow, session_id)
            if row is None:
                return False
            await session.execute(delete(_MessageRow).where(_MessageRow.session_id == session_id))
            await session.execute(
                delete(_AssessmentRow).where(_AssessmentRow.session_id == session_id)
            )
            await session.execute(
                delete(_ConversationUiRow).where(_ConversationUiRow.session_id == session_id)
            )
            await session.execute(
                delete(_ChannelBindingRow).where(_ChannelBindingRow.session_id == session_id)
            )
            await session.delete(row)
            return True

    async def get_channel_binding(self, channel: str, external_user: str) -> ChannelBinding | None:
        async with self._sessions() as session:
            row = await session.get(_ChannelBindingRow, (channel, external_user))
            return (
                None
                if row is None
                else ChannelBinding(
                    channel=row.channel,
                    external_user=row.external_user,
                    session_id=row.session_id,
                    consent=row.consent,
                    delete_pending=row.delete_pending,
                )
            )

    async def save_channel_binding(self, binding: ChannelBinding) -> None:
        async with self._sessions.begin() as session:
            if await session.get(_SessionRow, binding.session_id) is None:
                raise KeyError(binding.session_id)
            row = await session.get(_ChannelBindingRow, (binding.channel, binding.external_user))
            if row is None:
                row = _ChannelBindingRow(
                    channel=binding.channel,
                    external_user=binding.external_user,
                    session_id=binding.session_id,
                    consent=binding.consent,
                    delete_pending=binding.delete_pending,
                )
                session.add(row)
            else:
                row.session_id = binding.session_id
                row.consent = binding.consent
                row.delete_pending = binding.delete_pending

    async def list_channel_sessions(self) -> tuple[dict[str, str], ...]:
        """Expose session/channel labels, never personal-platform identifiers."""
        async with self._sessions() as session:
            rows = (await session.scalars(select(_ChannelBindingRow))).all()
        return tuple(
            {"session_id": row.session_id, "channel": row.channel} for row in rows
        )

    async def save_assessment(
        self, session_id: str, message_id: str, snapshot: dict, created_at: datetime
    ) -> None:
        async with self._sessions.begin() as session:
            row = await session.get(_AssessmentRow, message_id)
            if row is not None:
                return
            session.add(_AssessmentRow(
                session_id=session_id, message_id=message_id,
                snapshot_json=json.dumps(snapshot, ensure_ascii=False), created_at=created_at,
            ))

    async def list_assessments(
        self, session_id: str, message_ids: list[str] | None = None
    ) -> list[dict]:
        statement = select(_AssessmentRow).where(_AssessmentRow.session_id == session_id)
        if message_ids is not None:
            if not message_ids:
                return []
            statement = statement.where(_AssessmentRow.message_id.in_(message_ids))
        statement = statement.order_by(_AssessmentRow.created_at)
        async with self._sessions() as session:
            rows = (await session.scalars(statement)).all()
        return [json.loads(row.snapshot_json) for row in rows]

    async def message_count(self, session_id: str) -> int:
        async with self._sessions() as session:
            return await session.scalar(
                select(func.count()).select_from(_MessageRow).where(
                    _MessageRow.session_id == session_id
                )
            ) or 0

    async def get_channel_receipt(
        self, channel: str, external_user: str, event_id: str, request_id: str
    ) -> str | None:
        async with self._sessions() as session:
            row = await session.get(_ChannelReceiptRow, (channel, external_user, event_id))
            return row.reply if row is not None and row.request_id == request_id else None

    async def save_channel_receipt(
        self,
        *,
        channel: str,
        external_user: str,
        event_id: str,
        request_id: str,
        reply: str,
        created_at: datetime,
    ) -> None:
        async with self._sessions.begin() as session:
            row = await session.get(_ChannelReceiptRow, (channel, external_user, event_id))
            if row is not None:
                if row.request_id == request_id:
                    row.reply = reply
                return
            session.add(
                _ChannelReceiptRow(
                    channel=channel,
                    external_user=external_user,
                    event_id=event_id,
                    request_id=request_id,
                    reply=reply,
                    created_at=created_at,
                )
            )

    async def claim_channel_receipt(
        self,
        *,
        channel: str,
        external_user: str,
        event_id: str,
        request_id: str,
        created_at: datetime,
    ) -> bool:
        async with self._sessions.begin() as session:
            existing = await session.get(_ChannelReceiptRow, (channel, external_user, event_id))
            if existing is not None:
                return False
            session.add(
                _ChannelReceiptRow(
                    channel=channel,
                    external_user=external_user,
                    event_id=event_id,
                    request_id=request_id,
                    reply="",
                    created_at=created_at,
                )
            )
            return True

    async def delete_channel_receipt(self, channel: str, external_user: str, event_id: str) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                delete(_ChannelReceiptRow).where(
                    _ChannelReceiptRow.channel == channel,
                    _ChannelReceiptRow.external_user == external_user,
                    _ChannelReceiptRow.event_id == event_id,
                )
            )

    async def delete_channel_history(self, channel: str, external_user: str) -> None:
        binding = await self.get_channel_binding(channel, external_user)
        if binding is not None:
            await self.delete_session(binding.session_id)
        async with self._sessions.begin() as session:
            await session.execute(
                delete(_ChannelReceiptRow).where(
                    _ChannelReceiptRow.channel == channel,
                    _ChannelReceiptRow.external_user == external_user,
                )
            )

    async def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[StoredMessage, ...]:
        if not session_id.strip():
            raise ValueError("session_id cannot be blank")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        statement = (
            select(_MessageRow)
            .where(_MessageRow.session_id == session_id)
            .order_by(_MessageRow.ordinal.asc())
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            rows = list(result.scalars().all())
        selected = rows if limit is None else rows[-limit:]
        return tuple(self._message_from_row(row) for row in selected)

    async def commit_turn(
        self,
        *,
        session_id: str,
        expected_turn_index: int,
        next_state: SessionStateV1,
        messages: tuple[StoredMessage, ...],
        assessment: dict | None = None,
    ) -> None:
        if next_state.session_id != session_id:
            raise ValueError("next_state belongs to a different session")
        if next_state.turn_index != expected_turn_index + 1:
            raise ValueError("next_state must advance turn_index exactly once")
        if not messages:
            raise ValueError("turn commit requires at least one message")
        if any(message.session_id != session_id for message in messages):
            raise ValueError("all messages must belong to the committed session")
        ordinals = tuple(message.ordinal for message in messages)
        if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
            raise ValueError("turn message ordinals must be consecutive")

        async with self._sessions.begin() as session:
            row = await session.get(_SessionRow, session_id)
            if row is None:
                raise KeyError(session_id)
            current = SessionStateV1.model_validate_json(row.state_json)
            if current.turn_index != expected_turn_index:
                raise ConcurrentSessionUpdate(
                    f"session {session_id!r} advanced from turn "
                    f"{expected_turn_index} to {current.turn_index}"
                )

            for message in messages:
                existing = await session.get(_MessageRow, message.message_id)
                if existing is not None:
                    if self._message_from_row(existing) != message:
                        raise SessionConflictError(
                            f"message_id {message.message_id!r} already exists"
                        )
                    continue
                session.add(
                    _MessageRow(
                        message_id=message.message_id,
                        session_id=message.session_id,
                        ordinal=message.ordinal,
                        role=message.role.value,
                        content=message.content,
                        simulated_time=message.simulated_time,
                        created_at=message.created_at,
                    )
                )
            if assessment is not None:
                user_message = next(
                    (message for message in messages if message.role is ActorRole.USER), None
                )
                if user_message is None or assessment.get("message_id") != user_message.message_id:
                    raise ValueError("assessment must belong to the committed user message")
                session.add(
                    _AssessmentRow(
                        session_id=session_id,
                        message_id=user_message.message_id,
                        snapshot_json=json.dumps(assessment, ensure_ascii=False),
                        created_at=next_state.updated_at,
                    )
                )
            row.state_json = next_state.model_dump_json()
            row.updated_at = next_state.updated_at

    async def save_configuration(
        self,
        *,
        session_id: str,
        expected_turn_index: int,
        next_state: SessionStateV1,
    ) -> None:
        if next_state.session_id != session_id:
            raise ValueError("next_state belongs to a different session")
        if next_state.turn_index != expected_turn_index:
            raise ValueError("configuration changes cannot advance turn_index")
        async with self._sessions.begin() as session:
            row = await session.get(_SessionRow, session_id)
            if row is None:
                raise KeyError(session_id)
            current = SessionStateV1.model_validate_json(row.state_json)
            if current.turn_index != expected_turn_index:
                raise ConcurrentSessionUpdate(
                    f"session {session_id!r} changed during configuration update"
                )
            row.state_json = next_state.model_dump_json()
            row.updated_at = next_state.updated_at

    async def close(self) -> None:
        if self._owns_engine:
            await self._engine.dispose()

    @staticmethod
    def _record_from_row(row: _SessionRow) -> SessionRecord:
        return SessionRecord(
            session_id=row.session_id,
            participant_type=ParticipantType(row.participant_type),
            synthetic_case_id=row.synthetic_case_id,
            state=SessionStateV1.model_validate_json(row.state_json),
            model=row.model,
            research_mode=row.research_mode,
            created_at=SqlAlchemySessionRepository._aware(row.created_at),
        )

    @staticmethod
    def _message_from_row(row: _MessageRow) -> StoredMessage:
        return StoredMessage(
            message_id=row.message_id,
            session_id=row.session_id,
            ordinal=row.ordinal,
            role=ActorRole(row.role),
            content=row.content,
            simulated_time=row.simulated_time,
            created_at=SqlAlchemySessionRepository._aware(row.created_at),
        )

    @staticmethod
    def _validate_record(record: SessionRecord) -> None:
        if not record.session_id.strip():
            raise ValueError("session_id cannot be blank")
        if record.state.session_id != record.session_id:
            raise ValueError("session state belongs to a different session")
        if record.created_at.tzinfo is None or record.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")

    @staticmethod
    def _aware(value: datetime) -> datetime:
        # SQLite may return a naive value even for timezone=True.
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def serialize_record(record: SessionRecord) -> str:
    """Stable debugging helper that never includes provider credentials."""

    return json.dumps(
        {
            "session_id": record.session_id,
            "participant_type": record.participant_type.value,
            "synthetic_case_id": record.synthetic_case_id,
            "state": record.state.model_dump(mode="json"),
            "model": record.model,
            "research_mode": record.research_mode,
            "created_at": record.created_at.isoformat(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
