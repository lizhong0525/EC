from datetime import UTC, datetime, timedelta

import pytest

from ecocompanion.agent.proactivity import (
    ProactivityEligibilityReason,
    ProactivityNotAllowed,
    ProactivityPauseReason,
    ProactivityScheduler,
    PromptConflictError,
    PromptStatus,
)

DAY_ONE = datetime(2026, 9, 3, 9, 0, tzinfo=UTC)
DAY_TWO = DAY_ONE + timedelta(days=1)


def test_proactivity_is_disabled_by_default() -> None:
    scheduler = ProactivityScheduler()

    state = scheduler.get_state("session-001", now=DAY_ONE)
    eligibility = scheduler.eligibility("session-001", now=DAY_ONE)

    assert not state.enabled
    assert not eligibility.allowed
    assert eligibility.reason is ProactivityEligibilityReason.DISABLED


def test_scheduler_allows_at_most_one_prompt_per_day() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-001",
        session_id="session-001",
        sent_at=DAY_ONE,
    )
    scheduler.record_unanswered(
        "prompt-001",
        resolved_at=DAY_ONE + timedelta(hours=1),
    )

    eligibility = scheduler.eligibility(
        "session-001",
        now=DAY_ONE + timedelta(hours=2),
    )

    assert not eligibility.allowed
    assert eligibility.reason is ProactivityEligibilityReason.DAILY_LIMIT
    with pytest.raises(ProactivityNotAllowed, match="DAILY_LIMIT"):
        scheduler.record_sent(
            prompt_id="prompt-002",
            session_id="session-001",
            sent_at=DAY_ONE + timedelta(hours=2),
        )
    assert scheduler.eligibility("session-001", now=DAY_TWO).allowed


def test_scheduler_does_not_send_while_waiting_for_response() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-001",
        session_id="session-001",
        sent_at=DAY_ONE,
    )

    eligibility = scheduler.eligibility("session-001", now=DAY_TWO)

    assert not eligibility.allowed
    assert eligibility.reason is ProactivityEligibilityReason.AWAITING_RESPONSE


def test_two_consecutive_unanswered_prompts_pause_proactivity() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-001",
        session_id="session-001",
        sent_at=DAY_ONE,
    )
    scheduler.record_unanswered(
        "prompt-001",
        resolved_at=DAY_ONE + timedelta(hours=1),
    )
    scheduler.record_sent(
        prompt_id="prompt-002",
        session_id="session-001",
        sent_at=DAY_TWO,
    )
    scheduler.record_unanswered(
        "prompt-002",
        resolved_at=DAY_TWO + timedelta(hours=1),
    )

    state = scheduler.get_state("session-001", now=DAY_TWO)
    eligibility = scheduler.eligibility("session-001", now=DAY_TWO)

    assert state.paused
    assert state.enabled
    assert state.consecutive_unanswered == 2
    assert state.pause_reason is ProactivityPauseReason.CONSECUTIVE_NO_RESPONSE
    assert eligibility.reason is ProactivityEligibilityReason.PAUSED


def test_response_resets_no_response_streak() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-001",
        session_id="session-001",
        sent_at=DAY_ONE,
    )
    scheduler.record_unanswered(
        "prompt-001",
        resolved_at=DAY_ONE + timedelta(hours=1),
    )
    scheduler.record_sent(
        prompt_id="prompt-002",
        session_id="session-001",
        sent_at=DAY_TWO,
    )
    scheduler.record_response(
        "prompt-002",
        responded_at=DAY_TWO + timedelta(hours=1),
    )

    state = scheduler.get_state("session-001", now=DAY_TWO)

    assert state.consecutive_unanswered == 0
    assert scheduler.get_prompt("prompt-002").status is PromptStatus.RESPONDED


def test_prompt_events_are_idempotent_and_conflicts_are_rejected() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-001",
        session_id="session-001",
        sent_at=DAY_ONE,
    )
    scheduler.record_sent(
        prompt_id="prompt-001",
        session_id="session-001",
    )
    scheduler.record_unanswered(
        "prompt-001",
        resolved_at=DAY_ONE + timedelta(hours=1),
    )
    scheduler.record_unanswered(
        "prompt-001",
        resolved_at=DAY_ONE + timedelta(hours=2),
    )

    assert scheduler.get_state("session-001", now=DAY_ONE).consecutive_unanswered == 1
    with pytest.raises(PromptConflictError, match="different data"):
        scheduler.record_sent(
            prompt_id="prompt-001",
            session_id="other-session",
            sent_at=DAY_ONE,
        )


def test_paused_scheduler_requires_explicit_resume() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-001",
        session_id="session-001",
        sent_at=DAY_ONE,
    )
    scheduler.record_unanswered(
        "prompt-001",
        resolved_at=DAY_ONE + timedelta(hours=1),
    )
    scheduler.record_sent(
        prompt_id="prompt-002",
        session_id="session-001",
        sent_at=DAY_TWO,
    )
    scheduler.record_unanswered(
        "prompt-002",
        resolved_at=DAY_TWO + timedelta(hours=1),
    )

    with pytest.raises(ProactivityNotAllowed, match="explicit resume"):
        scheduler.enable("session-001")
    scheduler.resume("session-001")

    state = scheduler.get_state("session-001", now=DAY_TWO + timedelta(days=1))
    assert state.enabled
    assert not state.paused
    assert state.consecutive_unanswered == 0


def test_new_day_rollover_marks_unanswered_prompt_and_allows_next_prompt() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-rollover-1",
        session_id="session-001",
        sent_at=DAY_ONE,
    )

    scheduler.record_sent(
        prompt_id="prompt-rollover-2",
        session_id="session-001",
        sent_at=DAY_TWO,
    )

    assert scheduler.get_prompt("prompt-rollover-1").status is PromptStatus.UNANSWERED
    assert scheduler.get_state("session-001", now=DAY_TWO).consecutive_unanswered == 1
    assert (
        scheduler.get_state("session-001", now=DAY_TWO).outstanding_prompt_id
        == "prompt-rollover-2"
    )


def test_new_day_rollover_pauses_after_second_unanswered_prompt() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-rollover-3",
        session_id="session-001",
        sent_at=DAY_ONE,
    )
    scheduler.record_sent(
        prompt_id="prompt-rollover-4",
        session_id="session-001",
        sent_at=DAY_TWO,
    )

    with pytest.raises(ProactivityNotAllowed, match="PAUSED"):
        scheduler.record_sent(
            prompt_id="prompt-rollover-5",
            session_id="session-001",
            sent_at=DAY_TWO + timedelta(days=1),
        )

    assert scheduler.get_prompt("prompt-rollover-4").status is PromptStatus.UNANSWERED
    state = scheduler.get_state("session-001", now=DAY_TWO + timedelta(days=1))
    assert state.paused
    assert state.consecutive_unanswered == 2


def test_expire_outstanding_is_explicit_and_idempotent() -> None:
    scheduler = ProactivityScheduler()
    scheduler.enable("session-001")
    scheduler.record_sent(
        prompt_id="prompt-expire",
        session_id="session-001",
        sent_at=DAY_ONE,
    )

    scheduler.expire_outstanding("session-001", now=DAY_TWO)
    scheduler.expire_outstanding("session-001", now=DAY_TWO + timedelta(hours=1))

    assert scheduler.get_prompt("prompt-expire").status is PromptStatus.UNANSWERED
    assert scheduler.get_state("session-001", now=DAY_TWO).consecutive_unanswered == 1
