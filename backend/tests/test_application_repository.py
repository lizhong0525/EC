from datetime import UTC, datetime

import pytest

from ecocompanion.application.models import ParticipantType
from ecocompanion.application.repository import (
    ConcurrentSessionUpdate,
    SessionRecord,
    SqlAlchemySessionRepository,
    StoredMessage,
)
from ecocompanion.domain.enums import ActorRole
from ecocompanion.domain.schemas import SessionStateV1


@pytest.mark.asyncio
async def test_repository_commits_one_turn_atomically() -> None:
    repository = SqlAlchemySessionRepository("sqlite+aiosqlite:///:memory:")
    await repository.initialize()
    now = datetime(2026, 9, 3, tzinfo=UTC)
    state = SessionStateV1(session_id="session-1", updated_at=now)
    record = SessionRecord(
        session_id="session-1",
        participant_type=ParticipantType.ADULT_RESEARCH_TESTER,
        synthetic_case_id=None,
        state=state,
        model=None,
        research_mode=True,
        created_at=now,
    )
    await repository.create(record)
    await repository.create(record)

    next_state = state.model_copy(update={"turn_index": 1})
    messages = (
        StoredMessage(
            message_id="user-1",
            session_id="session-1",
            ordinal=0,
            role=ActorRole.USER,
            content="最近有点累。",
            simulated_time="day-01",
            created_at=now,
        ),
        StoredMessage(
            message_id="assistant-1",
            session_id="session-1",
            ordinal=1,
            role=ActorRole.COMPANION,
            content="愿意说说发生了什么吗？",
            simulated_time="day-01",
            created_at=now,
        ),
    )
    await repository.commit_turn(
        session_id="session-1",
        expected_turn_index=0,
        next_state=next_state,
        messages=messages,
    )

    stored = await repository.get("session-1")
    assert stored is not None
    assert stored.state.turn_index == 1
    assert await repository.list_messages("session-1") == messages

    with pytest.raises(ConcurrentSessionUpdate):
        await repository.commit_turn(
            session_id="session-1",
            expected_turn_index=0,
            next_state=next_state,
            messages=messages,
        )
    await repository.close()


@pytest.mark.asyncio
async def test_repository_returns_latest_history_in_chronological_order() -> None:
    repository = SqlAlchemySessionRepository("sqlite+aiosqlite:///:memory:")
    await repository.initialize()
    now = datetime(2026, 9, 3, tzinfo=UTC)
    state = SessionStateV1(session_id="session-2", updated_at=now)
    await repository.create(
        SessionRecord(
            session_id="session-2",
            participant_type=ParticipantType.ADULT_RESEARCH_TESTER,
            synthetic_case_id=None,
            state=state,
            model=None,
            research_mode=True,
            created_at=now,
        )
    )
    messages = tuple(
        StoredMessage(
            message_id=f"message-{index}",
            session_id="session-2",
            ordinal=index,
            role=ActorRole.USER,
            content=f"content-{index}",
            simulated_time="day-01",
            created_at=now,
        )
        for index in range(4)
    )
    await repository.commit_turn(
        session_id="session-2",
        expected_turn_index=0,
        next_state=state.model_copy(update={"turn_index": 1}),
        messages=messages,
    )

    latest = await repository.list_messages("session-2", limit=2)
    assert [message.ordinal for message in latest] == [2, 3]
    await repository.close()
