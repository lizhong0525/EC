"""Explicit, idempotent human-handoff state transitions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import RLock

from ecocompanion.domain.enums import EscalationTarget, HandoffStatus, RiskLevel


class HandoffConflictError(ValueError):
    """Raised when an idempotency key is reused with different data."""


class InvalidHandoffTransition(RuntimeError):
    """Raised when a request attempts to skip or reverse a state."""


class UnknownHandoffRequest(KeyError):
    """Raised when a handoff request id is unknown."""


@dataclass(frozen=True, slots=True)
class HandoffRequest:
    request_id: str
    session_id: str
    target: EscalationTarget
    risk_level: RiskLevel
    evidence_codes: tuple[str, ...] = ()
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id cannot be blank")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be blank")
        if self.target is EscalationTarget.NONE:
            raise ValueError("handoff target cannot be NONE")
        _require_aware(self.requested_at, "requested_at")


@dataclass(frozen=True, slots=True)
class HandoffRecord:
    request: HandoffRequest
    status: HandoffStatus
    updated_at: datetime
    responder_id: str | None = None
    failure_reason: str | None = None


class MockHandoffGateway:
    """In-memory gateway used for deterministic benchmark and unit tests."""

    def __init__(self) -> None:
        self._records: dict[str, HandoffRecord] = {}
        self._lock = RLock()

    def request(self, request: HandoffRequest) -> None:
        """Register a request idempotently in REQUESTED state."""

        with self._lock:
            existing = self._records.get(request.request_id)
            if existing is None:
                self._records[request.request_id] = HandoffRecord(
                    request=request,
                    status=HandoffStatus.REQUESTED,
                    updated_at=request.requested_at,
                )
                return
            if not self._same_idempotent_request(existing.request, request):
                raise HandoffConflictError(
                    f"request_id {request.request_id!r} already has different handoff data"
                )

    def mark_sent(self, request_id: str, *, at: datetime | None = None) -> None:
        """Confirm transport delivery without claiming human acceptance."""

        timestamp = _normalize_time(at)
        with self._lock:
            current = self._require_record(request_id)
            if current.status in {
                HandoffStatus.SENT,
                HandoffStatus.ACKNOWLEDGED,
            }:
                return
            self._require_status(current, HandoffStatus.REQUESTED, HandoffStatus.SENT)
            self._require_monotonic_time(current, timestamp)
            self._records[request_id] = replace(
                current,
                status=HandoffStatus.SENT,
                updated_at=timestamp,
            )

    def acknowledge(
        self,
        request_id: str,
        *,
        responder_id: str,
        at: datetime | None = None,
    ) -> None:
        """Record an explicit human acknowledgement."""

        if not responder_id.strip():
            raise ValueError("responder_id cannot be blank")
        timestamp = _normalize_time(at)
        with self._lock:
            current = self._require_record(request_id)
            if current.status is HandoffStatus.ACKNOWLEDGED:
                if current.responder_id != responder_id:
                    raise HandoffConflictError(
                        f"request {request_id!r} was acknowledged by a different responder"
                    )
                return
            self._require_status(current, HandoffStatus.SENT, HandoffStatus.ACKNOWLEDGED)
            self._require_monotonic_time(current, timestamp)
            self._records[request_id] = replace(
                current,
                status=HandoffStatus.ACKNOWLEDGED,
                updated_at=timestamp,
                responder_id=responder_id,
            )

    def fail(
        self,
        request_id: str,
        *,
        reason: str,
        at: datetime | None = None,
    ) -> None:
        """Move a pending request to its terminal FAILED state."""

        if not reason.strip():
            raise ValueError("failure reason cannot be blank")
        timestamp = _normalize_time(at)
        with self._lock:
            current = self._require_record(request_id)
            if current.status is HandoffStatus.FAILED:
                if current.failure_reason != reason:
                    raise HandoffConflictError(
                        f"request {request_id!r} already failed for a different reason"
                    )
                return
            if current.status not in {HandoffStatus.REQUESTED, HandoffStatus.SENT}:
                raise InvalidHandoffTransition(
                    f"cannot transition handoff from {current.status.value} to FAILED"
                )
            self._require_monotonic_time(current, timestamp)
            self._records[request_id] = replace(
                current,
                status=HandoffStatus.FAILED,
                updated_at=timestamp,
                failure_reason=reason,
            )

    def get(self, request_id: str) -> HandoffRecord:
        """Return the current immutable record."""

        with self._lock:
            return self._require_record(request_id)

    def is_acknowledged(self, request_id: str) -> bool:
        """Return true only after explicit human acknowledgement."""

        return self.get(request_id).status is HandoffStatus.ACKNOWLEDGED

    def message_for_user(self, request_id: str) -> str:
        """Describe only the state that the gateway can prove."""

        status = self.get(request_id).status
        messages = {
            HandoffStatus.REQUESTED: "正在准备人工支持请求，目前尚未确认接通。",
            HandoffStatus.SENT: "人工支持请求已经发送，但尚未收到接收确认。",
            HandoffStatus.ACKNOWLEDGED: "人工支持人员已确认接收这次请求。",
            HandoffStatus.FAILED: (
                "人工支持请求尚未接通，请立即使用预先约定的备用现实支持渠道。"
            ),
        }
        try:
            return messages[status]
        except KeyError as error:
            raise InvalidHandoffTransition(
                f"handoff request cannot expose status {status.value}"
            ) from error

    def _require_record(self, request_id: str) -> HandoffRecord:
        try:
            return self._records[request_id]
        except KeyError as error:
            raise UnknownHandoffRequest(request_id) from error

    @staticmethod
    def _require_status(
        current: HandoffRecord,
        expected: HandoffStatus,
        target: HandoffStatus,
    ) -> None:
        if current.status is not expected:
            raise InvalidHandoffTransition(
                f"cannot transition handoff from {current.status.value} to {target.value}"
            )

    @staticmethod
    def _same_idempotent_request(
        existing: HandoffRequest,
        candidate: HandoffRequest,
    ) -> bool:
        return (
            existing.request_id == candidate.request_id
            and existing.session_id == candidate.session_id
            and existing.target is candidate.target
            and existing.risk_level is candidate.risk_level
            and existing.evidence_codes == candidate.evidence_codes
        )

    @staticmethod
    def _require_monotonic_time(current: HandoffRecord, timestamp: datetime) -> None:
        if timestamp < current.updated_at:
            raise ValueError("handoff transition time cannot move backwards")


def _normalize_time(value: datetime | None) -> datetime:
    effective = datetime.now(UTC) if value is None else value
    _require_aware(effective, "timestamp")
    return effective


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
