"""Opt-in proactive-message scheduling with conservative stop rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, tzinfo
from enum import StrEnum
from threading import RLock


class PromptStatus(StrEnum):
    SENT = "SENT"
    RESPONDED = "RESPONDED"
    UNANSWERED = "UNANSWERED"


class ProactivityPauseReason(StrEnum):
    CONSECUTIVE_NO_RESPONSE = "CONSECUTIVE_NO_RESPONSE"


class ProactivityEligibilityReason(StrEnum):
    ALLOWED = "ALLOWED"
    DISABLED = "DISABLED"
    PAUSED = "PAUSED"
    DAILY_LIMIT = "DAILY_LIMIT"
    AWAITING_RESPONSE = "AWAITING_RESPONSE"


class ProactivityNotAllowed(RuntimeError):
    """Raised when the scheduler refuses a proactive prompt."""


class PromptConflictError(ValueError):
    """Raised when a prompt id is reused with different data."""


class UnknownPrompt(KeyError):
    """Raised when a prompt id is unknown."""


@dataclass(frozen=True, slots=True)
class ProactivityState:
    session_id: str
    enabled: bool = False
    paused: bool = False
    pause_reason: ProactivityPauseReason | None = None
    consecutive_unanswered: int = 0
    daily_bucket: date | None = None
    sent_today: int = 0
    outstanding_prompt_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProactivityEligibility:
    allowed: bool
    reason: ProactivityEligibilityReason


@dataclass(frozen=True, slots=True)
class ProactivePrompt:
    prompt_id: str
    session_id: str
    sent_at: datetime
    status: PromptStatus = PromptStatus.SENT
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise ValueError("prompt_id cannot be blank")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be blank")
        _require_aware(self.sent_at, "sent_at")
        if self.resolved_at is not None:
            _require_aware(self.resolved_at, "resolved_at")


class ProactivityScheduler:
    """Consent-gated scheduler: one prompt/day and pause after two misses."""

    MAX_PER_DAY = 1
    MAX_CONSECUTIVE_UNANSWERED = 2

    def __init__(self, *, timezone: tzinfo = UTC) -> None:
        self._timezone = timezone
        self._states: dict[str, ProactivityState] = {}
        self._prompts: dict[str, ProactivePrompt] = {}
        self._lock = RLock()

    def enable(self, session_id: str) -> None:
        """Enable prompts for a session that is not safety-paused."""

        self._validate_session_id(session_id)
        with self._lock:
            current = self._states.get(session_id, ProactivityState(session_id=session_id))
            if current.paused:
                raise ProactivityNotAllowed(
                    "proactivity is paused; explicit resume is required"
                )
            self._states[session_id] = replace(current, enabled=True)

    def disable(self, session_id: str) -> None:
        """Disable new prompts without rewriting prior prompt outcomes."""

        self._validate_session_id(session_id)
        with self._lock:
            current = self._states.get(session_id, ProactivityState(session_id=session_id))
            self._states[session_id] = replace(current, enabled=False)

    def resume(self, session_id: str) -> None:
        """Explicitly resume after a consecutive-no-response pause."""

        self._validate_session_id(session_id)
        with self._lock:
            current = self._states.get(session_id, ProactivityState(session_id=session_id))
            self._states[session_id] = replace(
                current,
                enabled=True,
                paused=False,
                pause_reason=None,
                consecutive_unanswered=0,
            )

    def get_state(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> ProactivityState:
        """Return a day-normalized state without mutating scheduler data."""

        self._validate_session_id(session_id)
        effective_now = self._normalize_time(now)
        with self._lock:
            current = self._states.get(session_id, ProactivityState(session_id=session_id))
            return self._for_day(current, effective_now)

    def eligibility(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> ProactivityEligibility:
        """Explain whether a proactive prompt may be sent now."""

        state = self.get_state(session_id, now=now)
        if not state.enabled:
            return ProactivityEligibility(False, ProactivityEligibilityReason.DISABLED)
        if state.paused:
            return ProactivityEligibility(False, ProactivityEligibilityReason.PAUSED)
        if state.outstanding_prompt_id is not None:
            return ProactivityEligibility(
                False,
                ProactivityEligibilityReason.AWAITING_RESPONSE,
            )
        if state.sent_today >= self.MAX_PER_DAY:
            return ProactivityEligibility(False, ProactivityEligibilityReason.DAILY_LIMIT)
        return ProactivityEligibility(True, ProactivityEligibilityReason.ALLOWED)

    def record_sent(
        self,
        *,
        prompt_id: str,
        session_id: str,
        sent_at: datetime | None = None,
    ) -> None:
        """Record one send idempotently after checking all stop rules."""

        self._validate_session_id(session_id)
        if not prompt_id.strip():
            raise ValueError("prompt_id cannot be blank")
        with self._lock:
            existing = self._prompts.get(prompt_id)
            if existing is not None:
                if existing.session_id != session_id:
                    raise PromptConflictError(
                        f"prompt_id {prompt_id!r} already exists with different data"
                )
                return
            timestamp = self._normalize_time(sent_at)
            # A prompt left unanswered across a local-day boundary is a
            # conservative, deterministic miss.  Resolve it before checking
            # the next-send eligibility so the two-miss pause also works
            # through the public scheduling path.
            self._expire_outstanding_if_due(session_id, timestamp)
            candidate = ProactivePrompt(
                prompt_id=prompt_id,
                session_id=session_id,
                sent_at=timestamp,
            )
            eligibility = self.eligibility(session_id, now=timestamp)
            if not eligibility.allowed:
                raise ProactivityNotAllowed(
                    f"proactive prompt blocked: {eligibility.reason.value}"
                )
            current = self._for_day(
                self._states.get(session_id, ProactivityState(session_id=session_id)),
                timestamp,
            )
            self._prompts[prompt_id] = candidate
            self._states[session_id] = replace(
                current,
                daily_bucket=self._local_date(timestamp),
                sent_today=current.sent_today + 1,
                outstanding_prompt_id=prompt_id,
            )

    def expire_outstanding(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Mark a stale outstanding prompt unanswered after a day rollover.

        This is an explicit command so callers can run timeout handling even
        when they do not immediately send another proactive prompt.  Merely
        querying :meth:`get_state` or :meth:`eligibility` remains side-effect
        free.
        """

        self._validate_session_id(session_id)
        timestamp = self._normalize_time(now)
        with self._lock:
            self._expire_outstanding_if_due(session_id, timestamp)

    def record_response(
        self,
        prompt_id: str,
        *,
        responded_at: datetime | None = None,
    ) -> None:
        """Resolve a prompt as answered and reset the no-response streak."""

        timestamp = self._normalize_time(responded_at)
        with self._lock:
            prompt = self._require_prompt(prompt_id)
            if prompt.status is PromptStatus.RESPONDED:
                return
            if prompt.status is PromptStatus.UNANSWERED:
                return
            if timestamp < prompt.sent_at:
                raise ValueError("responded_at cannot be earlier than sent_at")
            current = self._states[prompt.session_id]
            self._prompts[prompt_id] = replace(
                prompt,
                status=PromptStatus.RESPONDED,
                resolved_at=timestamp,
            )
            outstanding_id = current.outstanding_prompt_id
            self._states[prompt.session_id] = replace(
                current,
                consecutive_unanswered=0,
                outstanding_prompt_id=(
                    None if outstanding_id == prompt_id else outstanding_id
                ),
            )

    def record_unanswered(
        self,
        prompt_id: str,
        *,
        resolved_at: datetime | None = None,
    ) -> None:
        """Count a missed prompt once and pause after the second consecutive miss."""

        timestamp = self._normalize_time(resolved_at)
        with self._lock:
            prompt = self._require_prompt(prompt_id)
            if prompt.status is PromptStatus.UNANSWERED:
                return
            if prompt.status is PromptStatus.RESPONDED:
                return
            if timestamp < prompt.sent_at:
                raise ValueError("resolved_at cannot be earlier than sent_at")
            current = self._states[prompt.session_id]
            unanswered_count = current.consecutive_unanswered + 1
            paused = unanswered_count >= self.MAX_CONSECUTIVE_UNANSWERED
            self._prompts[prompt_id] = replace(
                prompt,
                status=PromptStatus.UNANSWERED,
                resolved_at=timestamp,
            )
            outstanding_id = current.outstanding_prompt_id
            self._states[prompt.session_id] = replace(
                current,
                paused=paused,
                pause_reason=(
                    ProactivityPauseReason.CONSECUTIVE_NO_RESPONSE if paused else None
                ),
                consecutive_unanswered=unanswered_count,
                outstanding_prompt_id=(
                    None if outstanding_id == prompt_id else outstanding_id
                ),
            )

    def get_prompt(self, prompt_id: str) -> ProactivePrompt:
        """Return an immutable prompt record."""

        with self._lock:
            return self._require_prompt(prompt_id)

    def _normalize_time(self, value: datetime | None) -> datetime:
        effective = datetime.now(UTC) if value is None else value
        _require_aware(effective, "timestamp")
        return effective

    def _local_date(self, value: datetime) -> date:
        return value.astimezone(self._timezone).date()

    def _for_day(self, state: ProactivityState, now: datetime) -> ProactivityState:
        current_day = self._local_date(now)
        if state.daily_bucket is None or state.daily_bucket == current_day:
            return state
        return replace(state, daily_bucket=current_day, sent_today=0)

    def _expire_outstanding_if_due(
        self,
        session_id: str,
        timestamp: datetime,
    ) -> None:
        current = self._states.get(session_id)
        if current is None or current.outstanding_prompt_id is None:
            return
        prompt = self._prompts.get(current.outstanding_prompt_id)
        if prompt is None:
            raise UnknownPrompt(current.outstanding_prompt_id)
        if prompt.status is not PromptStatus.SENT:
            return
        if self._local_date(timestamp) <= self._local_date(prompt.sent_at):
            return
        # RLock makes this re-entrant while keeping the public transition
        # validation in one place.
        self.record_unanswered(prompt.prompt_id, resolved_at=timestamp)

    def _require_prompt(self, prompt_id: str) -> ProactivePrompt:
        try:
            return self._prompts[prompt_id]
        except KeyError as error:
            raise UnknownPrompt(prompt_id) from error

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id cannot be blank")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
