from datetime import UTC, datetime, timedelta

import pytest

from ecocompanion.agent.handoff import (
    HandoffConflictError,
    HandoffRequest,
    InvalidHandoffTransition,
    MockHandoffGateway,
)
from ecocompanion.domain.enums import EscalationTarget, HandoffStatus, RiskLevel

NOW = datetime(2026, 9, 3, 9, 0, tzinfo=UTC)


def handoff_request(
    *,
    request_id: str = "handoff-001",
    target: EscalationTarget = EscalationTarget.SCHOOL_SUPPORT,
) -> HandoffRequest:
    return HandoffRequest(
        request_id=request_id,
        session_id="session-001",
        target=target,
        risk_level=RiskLevel.L3,
        evidence_codes=("CURRENT_SAFETY_CONCERN",),
        requested_at=NOW,
    )


def test_handoff_requires_explicit_send_and_acknowledgement() -> None:
    gateway = MockHandoffGateway()
    request = handoff_request()

    gateway.request(request)
    assert gateway.get(request.request_id).status is HandoffStatus.REQUESTED
    assert not gateway.is_acknowledged(request.request_id)
    assert "尚未确认" in gateway.message_for_user(request.request_id)

    gateway.mark_sent(request.request_id, at=NOW + timedelta(seconds=1))
    assert gateway.get(request.request_id).status is HandoffStatus.SENT
    assert "尚未收到" in gateway.message_for_user(request.request_id)

    gateway.acknowledge(
        request.request_id,
        responder_id="school-counselor-01",
        at=NOW + timedelta(seconds=2),
    )
    record = gateway.get(request.request_id)
    assert record.status is HandoffStatus.ACKNOWLEDGED
    assert record.responder_id == "school-counselor-01"
    assert gateway.is_acknowledged(request.request_id)
    assert "已确认接收" in gateway.message_for_user(request.request_id)


def test_handoff_request_and_transitions_are_idempotent() -> None:
    gateway = MockHandoffGateway()
    request = handoff_request()
    equivalent_retry = HandoffRequest(
        request_id=request.request_id,
        session_id=request.session_id,
        target=request.target,
        risk_level=request.risk_level,
        evidence_codes=request.evidence_codes,
        requested_at=NOW + timedelta(milliseconds=500),
    )

    gateway.request(request)
    gateway.request(equivalent_retry)
    gateway.mark_sent(request.request_id, at=NOW + timedelta(seconds=1))
    gateway.mark_sent(request.request_id, at=NOW + timedelta(seconds=9))
    gateway.acknowledge(
        request.request_id,
        responder_id="responder-001",
        at=NOW + timedelta(seconds=2),
    )
    gateway.acknowledge(
        request.request_id,
        responder_id="responder-001",
        at=NOW + timedelta(seconds=5),
    )
    gateway.mark_sent(
        request.request_id,
        at=NOW + timedelta(seconds=6),
    )

    assert gateway.is_acknowledged(request.request_id)


def test_handoff_rejects_backwards_transition_time() -> None:
    gateway = MockHandoffGateway()
    request = handoff_request()
    gateway.request(request)

    with pytest.raises(ValueError, match="cannot move backwards"):
        gateway.mark_sent(
            request.request_id,
            at=NOW - timedelta(seconds=1),
        )


def test_handoff_rejects_conflicting_id_reuse() -> None:
    gateway = MockHandoffGateway()
    gateway.request(handoff_request())

    with pytest.raises(HandoffConflictError, match="different handoff data"):
        gateway.request(
            handoff_request(target=EscalationTarget.EMERGENCY_HUMAN)
        )


def test_handoff_cannot_claim_acknowledgement_before_send() -> None:
    gateway = MockHandoffGateway()
    request = handoff_request()
    gateway.request(request)

    with pytest.raises(InvalidHandoffTransition, match="REQUESTED to ACKNOWLEDGED"):
        gateway.acknowledge(
            request.request_id,
            responder_id="responder-001",
            at=NOW + timedelta(seconds=1),
        )


def test_failed_handoff_is_terminal_and_never_claims_success() -> None:
    gateway = MockHandoffGateway()
    request = handoff_request()
    gateway.request(request)
    gateway.fail(
        request.request_id,
        reason="mock transport unavailable",
        at=NOW + timedelta(seconds=1),
    )
    gateway.fail(
        request.request_id,
        reason="mock transport unavailable",
        at=NOW + timedelta(seconds=2),
    )

    assert gateway.get(request.request_id).status is HandoffStatus.FAILED
    assert "尚未接通" in gateway.message_for_user(request.request_id)
    assert not gateway.is_acknowledged(request.request_id)
    with pytest.raises(InvalidHandoffTransition, match="FAILED to SENT"):
        gateway.mark_sent(request.request_id, at=NOW + timedelta(seconds=3))


def test_handoff_rejects_none_target_and_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="cannot be NONE"):
        handoff_request(target=EscalationTarget.NONE)
    with pytest.raises(ValueError, match="timezone-aware"):
        HandoffRequest(
            request_id="handoff-002",
            session_id="session-001",
            target=EscalationTarget.PROFESSIONAL,
            risk_level=RiskLevel.L2,
            requested_at=datetime(2026, 9, 3, 9, 0),
        )
