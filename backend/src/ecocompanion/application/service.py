"""One-turn application orchestration for the companion research prototype."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from ecocompanion.agent import (
    HandoffRequest,
    MemoryService,
    MockHandoffGateway,
    ProactivityNotAllowed,
    ProactivityScheduler,
    ProtectedMemoryError,
    SafetyAuditor,
)
from ecocompanion.application.conversation import (
    CONVERSATION_POLICY_VERSION,
    QUALITY_GATE_VERSION,
    RENDER_PROMPT_VERSION,
    WebConversationPolicy,
    classify_intent,
    fallback_reply,
    render_memory_snippets,
    response_quality_codes,
    web_audit_action,
)
from ecocompanion.application.models import (
    AuthorizedMemoryRequest,
    CreateSessionRequest,
    DecisionView,
    MemoryCorrectionRequest,
    MessageView,
    ParticipantType,
    ProactiveMessageView,
    SessionView,
    TurnRequest,
    TurnResult,
)
from ecocompanion.application.renderer import LLMResponseRenderer
from ecocompanion.application.repository import (
    SessionRecord,
    SessionRepository,
    StoredMessage,
)
from ecocompanion.core import (
    AssessmentResult,
    ECOSafePolicy,
    PolicyOutcome,
    RuleBasedRiskAssessor,
    SessionStateMachine,
)
from ecocompanion.core.state_machine import InvalidStateTransition
from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    ConversationPhase,
    EscalationTarget,
    HandoffStatus,
    MemoryKind,
    RiskLevel,
    Track,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    ConversationMessage,
    CurrentObservation,
    DecisionPlan,
    MemoryRecordV1,
    ModelCallUsage,
    ObservationV1,
    ResponsePayload,
    SafetyAudit,
    SessionStateV1,
    TraceEventV1,
    VisibleContext,
    VisibleProfile,
)
from ecocompanion.infrastructure import CostLedger, JsonlTraceStore
from ecocompanion.providers import ChatCompletion, ModelRole, ProviderError


class SessionNotFound(KeyError):
    """Raised when an API command targets an unknown session."""


class TurnConflictError(RuntimeError):
    """Raised when an idempotency key is reused after a turn was committed."""


class ModelSelectionError(ValueError):
    """Raised when a session requests a model outside the frozen runtime."""


class BudgetExceeded(RuntimeError):
    """Raised before a new provider call when the known budget is exhausted."""


@dataclass(frozen=True, slots=True)
class _SafeRender:
    text: str
    audit: SafetyAudit
    usages: tuple[ModelCallUsage, ...]


class CompanionService:
    """Coordinates pure decisions and command components without merging ownership."""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        renderer: LLMResponseRenderer,
        trace_store: JsonlTraceStore,
        cost_ledger: CostLedger,
        risk_assessor: RuleBasedRiskAssessor | None = None,
        policy: ECOSafePolicy | WebConversationPolicy | None = None,
        state_machine: SessionStateMachine | None = None,
        memory: MemoryService | None = None,
        auditor: SafetyAuditor | None = None,
        handoff: MockHandoffGateway | None = None,
        proactivity: ProactivityScheduler | None = None,
        known_synthetic_case_ids: frozenset[str] | None = None,
        api_budget_cny: float = 5_000.0,
        research_mode_enabled: bool = True,
        web_conversation_mode: bool = True,
        turn_advisor: Callable[
            [ObservationV1, AssessmentResult, SessionStateV1, PolicyOutcome],
            Awaitable[PolicyOutcome],
        ]
        | None = None,
        turn_snapshot: Callable[[str, bool, PolicyOutcome], dict | None] | None = None,
    ) -> None:
        if api_budget_cny <= 0:
            raise ValueError("api_budget_cny must be positive")
        self._repository = repository
        self._renderer = renderer
        self._trace_store = trace_store
        self._cost_ledger = cost_ledger
        self._risk_assessor = risk_assessor or RuleBasedRiskAssessor()
        self._policy = policy or (
            WebConversationPolicy() if web_conversation_mode else ECOSafePolicy()
        )
        self._web_conversation_mode = web_conversation_mode
        self._policy_version = (
            CONVERSATION_POLICY_VERSION if web_conversation_mode else "core-policy-v1"
        )
        self._quality_gate_version = QUALITY_GATE_VERSION
        self._state_machine = state_machine or SessionStateMachine()
        self._memory = memory or MemoryService()
        self._auditor = auditor or SafetyAuditor()
        self._handoff = handoff or MockHandoffGateway()
        self._proactivity = proactivity or ProactivityScheduler()
        self._known_synthetic_case_ids = known_synthetic_case_ids
        self._api_budget_cny = Decimal(str(api_budget_cny))
        self._research_mode_enabled = research_mode_enabled
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._turn_advisor = turn_advisor
        self._turn_snapshot = turn_snapshot

    @property
    def model(self) -> str | None:
        return self._renderer.model

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def close(self) -> None:
        await self._repository.close()

    async def create_session(self, request: CreateSessionRequest) -> SessionView:
        self._validate_session_request(request)
        session_id = uuid4().hex
        now = datetime.now(UTC)
        state = SessionStateV1(
            session_id=session_id,
            memory_enabled=request.memory_enabled,
            proactive_enabled=request.proactive_enabled,
            updated_at=now,
        )
        record = SessionRecord(
            session_id=session_id,
            participant_type=request.participant_type,
            synthetic_case_id=request.synthetic_case_id,
            state=state,
            model=self._renderer.model,
            research_mode=request.research_mode,
            created_at=now,
        )
        await self._repository.create(record)
        if request.proactive_enabled:
            self._proactivity.enable(session_id)
        return self._view(record)

    async def get_session(self, session_id: str) -> SessionView:
        return self._view(await self._require_session(session_id))

    async def delete_session(self, session_id: str) -> bool:
        """Delete one local conversation after its current turn has settled."""
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await self._repository.delete_session(session_id)

    async def recover_turn(
        self, session_id: str, request: TurnRequest
    ) -> tuple[SessionView, str] | None:
        """Read an identical committed turn without repeating provider effects."""
        if request.request_id is None:
            return None
        messages = await self._repository.list_messages(session_id)
        user_id = f"{session_id}:user:{request.request_id}"
        assistant_id = f"{session_id}:assistant:{request.request_id}"
        user = next((item for item in messages if item.message_id == user_id), None)
        assistant = next((item for item in messages if item.message_id == assistant_id), None)
        if user is None or assistant is None or user.content != request.text.strip():
            return None
        return await self.get_session(session_id), assistant.content

    async def process_turn(
        self,
        session_id: str,
        request: TurnRequest,
    ) -> TurnResult:
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            record = await self._require_session(session_id)
            request_id = request.request_id or uuid4().hex
            user_message_id = f"{session_id}:user:{request_id}"
            assistant_message_id = f"{session_id}:assistant:{request_id}"
            all_messages = await self._repository.list_messages(session_id)
            self._ensure_turn_request_is_new(
                request=request,
                request_id=request_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                messages=all_messages,
            )
            self._resolve_outstanding_proactive_prompt(record)
            history = all_messages[-20:]
            now = datetime.now(UTC)
            simulated_time = request.simulated_time or now.isoformat()
            observation = self._build_observation(
                record=record,
                history=history,
                message_id=user_message_id,
                text=request.text.strip(),
                simulated_time=simulated_time,
            )

            assessment = self._risk_assessor.assess(observation)
            policy_assessment = assessment
            if self._web_conversation_mode:
                intent = classify_intent(observation, assessment, record.state)
                if isinstance(self._policy, WebConversationPolicy):
                    web_decision = self._policy.decide_with_context(
                        assessment=assessment,
                        observation=observation,
                        session=record.state,
                    )
                    policy_assessment = web_decision.assessment
                    intent = web_decision.intent
                observation = observation.model_copy(
                    update={
                        "benchmark_version": CONVERSATION_POLICY_VERSION,
                        "current_observation": observation.current_observation.model_copy(
                            update={
                                "observable_features": {
                                    "conversation_intent": intent.value,
                                    "web_policy": CONVERSATION_POLICY_VERSION,
                                }
                            }
                        ),
                    }
                )
            if self._web_conversation_mode and isinstance(self._policy, WebConversationPolicy):
                outcome = web_decision.outcome
            else:
                outcome = self._policy.decide(
                    assessment=policy_assessment,
                    observation=observation,
                    session=record.state,
                )
            if self._turn_advisor is not None:
                outcome = await self._turn_advisor(
                    observation, policy_assessment, record.state, outcome
                )
            try:
                handoff_status = self._prepare_handoff(
                    record=record,
                    outcome=outcome,
                    evidence_codes=tuple(
                        evidence.evidence_code for evidence in assessment.evidence
                    ),
                    requested_at=now,
                )
            except (RuntimeError, ValueError):
                # A handoff transport failure must not erase the local safety
                # decision or let a generated response claim successful
                # contact.  Persist FAILED so the state machine remains
                # conservative and the renderer can offer a backup channel.
                handoff_status = (
                    HandoffStatus.FAILED
                    if outcome.plan.escalation_target is not EscalationTarget.NONE
                    else record.state.handoff_status
                )
            next_state, effective_outcome = self._advance_state(
                current=record.state,
                outcome=outcome,
                handoff_status=handoff_status,
                observation=observation,
            )
            safe_render = await self._render_safely(
                record=record,
                observation=observation,
                assessment=policy_assessment,
                outcome=effective_outcome,
                render_session=next_state,
            )

            decision = AgentDecisionV1(
                assessment=policy_assessment.estimate,
                uncertainty=policy_assessment.uncertainty,
                evidence=assessment.evidence,
                decision=effective_outcome.plan,
                response=ResponsePayload(text=safe_render.text),
                reason_codes=effective_outcome.reason_codes,
            )
            turn_messages = self._turn_messages(
                session_id=session_id,
                turn_index=record.state.turn_index,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                user_text=request.text.strip(),
                assistant_text=safe_render.text,
                simulated_time=simulated_time,
                created_at=now,
            )
            await self._repository.commit_turn(
                session_id=session_id,
                expected_turn_index=record.state.turn_index,
                next_state=next_state,
                messages=turn_messages,
                assessment=(
                    self._turn_snapshot(
                        user_message_id, safe_render.audit.used_fallback, effective_outcome
                    )
                    if self._turn_snapshot is not None
                    else None
                ),
            )

            memory_refs = self._record_memories(
                record=record,
                state=next_state,
                user_message=turn_messages[0],
                assistant_message=turn_messages[1],
                reason_codes=decision.reason_codes,
                created_at=now,
            )
            self._trace_store.append(
                TraceEventV1(
                    trace_id=f"trace:{session_id}:{request_id}",
                    benchmark_version=observation.benchmark_version,
                    scenario_id=record.synthetic_case_id or "adult-research-demo",
                    episode_id=session_id,
                    step_id=record.state.turn_index,
                    external_event={
                        "participant_type": record.participant_type.value,
                        "request_id": request_id,
                    },
                    observation=observation,
                    memory_refs=memory_refs,
                    assessment=decision.assessment,
                    selected_action=decision.decision.primary_action,
                    response_text=decision.response.text,
                    safety_audit=safe_render.audit,
                    random_seed=self._stable_seed(session_id, record.state.turn_index),
                    model_usage=safe_render.usages,
                )
            )
            updated_record = SessionRecord(
                session_id=record.session_id,
                participant_type=record.participant_type,
                synthetic_case_id=record.synthetic_case_id,
                state=next_state,
                model=record.model,
                research_mode=record.research_mode,
                created_at=record.created_at,
            )
            return TurnResult(
                session=self._view(updated_record),
                response=safe_render.text,
                decision=DecisionView(
                    primary_action=decision.decision.primary_action,
                    escalation_level=decision.decision.escalation_level,
                    escalation_target=decision.decision.escalation_target,
                    risk_estimate=decision.assessment,
                    uncertainty=decision.uncertainty,
                    evidence_codes=tuple(evidence.evidence_code for evidence in decision.evidence),
                    reason_codes=decision.reason_codes,
                    safety_passed=safe_render.audit.passed,
                    used_fallback=safe_render.audit.used_fallback,
                    audit_codes=safe_render.audit.violation_codes,
                    model_calls=safe_render.usages,
                    policy_version=self._policy_version,
                    prompt_version=(
                        RENDER_PROMPT_VERSION if self._renderer.provider_enabled else None
                    ),
                ),
            )

    async def list_memories(self, session_id: str) -> tuple[MemoryRecordV1, ...]:
        await self._require_session(session_id)
        return self._memory.list_for_session(
            session_id,
            kinds=(
                MemoryKind.WORKING,
                MemoryKind.EPISODIC,
                MemoryKind.PREFERENCE,
            ),
        )

    async def list_messages(self, session_id: str) -> tuple[MessageView, ...]:
        await self._require_session(session_id)
        messages = await self._repository.list_messages(session_id)
        return tuple(
            MessageView(
                message_id=message.message_id,
                ordinal=message.ordinal,
                role=message.role,
                content=message.content,
                simulated_time=message.simulated_time,
                created_at=message.created_at,
            )
            for message in messages
        )

    async def set_proactivity(
        self,
        session_id: str,
        *,
        enabled: bool,
    ) -> SessionView:
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            record = await self._require_session(session_id)
            if record.participant_type is not ParticipantType.ADULT_RESEARCH_TESTER:
                raise PermissionError("proactivity is only available to adult research testers")
            if enabled:
                # Check the in-memory scheduler before touching the
                # persistent session.  A paused scheduler requires the
                # explicit resume command and must not be silently re-enabled
                # by a normal opt-in update.
                scheduler_state = self._proactivity.get_state(session_id)
                if scheduler_state.paused:
                    raise ProactivityNotAllowed(
                        "proactivity is paused; explicit resume is required"
                    )
            next_state = self._state_machine.configure_proactivity(
                current=record.state,
                enabled=enabled,
            )
            # Persist first.  The repository is the authoritative source for
            # session configuration; mutating the process-local scheduler
            # before a failed database write would leave the two sources out
            # of sync.
            await self._repository.save_configuration(
                session_id=session_id,
                expected_turn_index=record.state.turn_index,
                next_state=next_state,
            )
            if enabled:
                self._proactivity.enable(session_id)
            else:
                self._proactivity.disable(session_id)
            return self._view(
                SessionRecord(
                    session_id=record.session_id,
                    participant_type=record.participant_type,
                    synthetic_case_id=record.synthetic_case_id,
                    state=next_state,
                    model=record.model,
                    research_mode=record.research_mode,
                    created_at=record.created_at,
                )
            )

    async def resume_proactivity(self, session_id: str) -> SessionView:
        """Explicitly resume an adult tester's paused proactive prompts.

        Resuming is an explicit opt-in action.  It is idempotent when the
        scheduler is already enabled and also repairs an in-memory scheduler
        after a process restart when the persisted session flag is enabled.
        The session flag is persisted before the scheduler transition so a
        failed configuration write cannot accidentally enable prompts.
        """

        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            record = await self._require_session(session_id)
            if record.participant_type is not ParticipantType.ADULT_RESEARCH_TESTER:
                raise PermissionError("proactivity is only available to adult research testers")
            scheduler_state = self._proactivity.get_state(session_id)
            next_state = self._state_machine.configure_proactivity(
                current=record.state,
                enabled=True,
            )
            await self._repository.save_configuration(
                session_id=session_id,
                expected_turn_index=record.state.turn_index,
                next_state=next_state,
            )
            if scheduler_state.paused:
                self._proactivity.resume(session_id)
            elif not scheduler_state.enabled:
                self._proactivity.enable(session_id)
            return self._view(
                SessionRecord(
                    session_id=record.session_id,
                    participant_type=record.participant_type,
                    synthetic_case_id=record.synthetic_case_id,
                    state=next_state,
                    model=record.model,
                    research_mode=record.research_mode,
                    created_at=record.created_at,
                )
            )

    async def claim_proactive_message(
        self,
        session_id: str,
    ) -> ProactiveMessageView:
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            record = await self._require_session(session_id)
            if record.participant_type is not ParticipantType.ADULT_RESEARCH_TESTER:
                raise PermissionError("proactivity is only available to adult research testers")
            if not record.state.proactive_enabled:
                raise PermissionError("proactivity is disabled for this session")
            scheduler_state = self._proactivity.get_state(session_id)
            if not scheduler_state.enabled and not scheduler_state.paused:
                self._proactivity.enable(session_id)

            prompt_id = f"proactive:{session_id}:{uuid4().hex}"
            sent_at = datetime.now(UTC)
            self._proactivity.record_sent(
                prompt_id=prompt_id,
                session_id=session_id,
                sent_at=sent_at,
            )
            snippets = tuple(
                memory.content
                for memory in self._memory.list_for_session(
                    session_id,
                    kinds=(MemoryKind.EPISODIC, MemoryKind.PREFERENCE),
                )[-4:]
            )
            completions: list[tuple[str, ChatCompletion]] = []
            used_fallback = False
            try:
                self._ensure_budget_available()
                rendered = await self._renderer.render_proactive(memory_snippets=snippets)
                if rendered.completion is not None:
                    completions.append(("proactive", rendered.completion))
                audited = self._auditor.enforce(
                    rendered.payload.text,
                    risk_level=RiskLevel.L0,
                    handoff_status=record.state.handoff_status,
                )
                text = audited.text
                used_fallback = not rendered.provider_used or audited.audit.used_fallback
            except (ProviderError, RuntimeError):
                text = self._auditor.fallback_for(RiskLevel.L0)
                used_fallback = True
            self._record_completions(
                session_id,
                record.state.turn_index,
                completions,
            )
            return ProactiveMessageView(
                prompt_id=prompt_id,
                text=text,
                sent_at=sent_at,
                used_fallback=used_fallback,
            )

    async def delete_memory(self, session_id: str, memory_id: str) -> None:
        await self._require_session(session_id)
        record = self._memory.get(memory_id, include_expired=True)
        if record is None:
            return
        if record.session_id != session_id:
            raise SessionNotFound(memory_id)
        try:
            self._memory.delete(memory_id)
        except ProtectedMemoryError:
            raise

    async def add_authorized_memory(
        self,
        session_id: str,
        request: AuthorizedMemoryRequest,
    ) -> MemoryRecordV1:
        session = await self._require_session(session_id)
        if not session.state.memory_enabled:
            raise PermissionError("memory is disabled for this session")
        if request.source_message_id is not None:
            messages = await self._repository.list_messages(session_id)
            if not any(message.message_id == request.source_message_id for message in messages):
                raise ValueError("source_message_id does not belong to this session")
        record = MemoryRecordV1(
            memory_id=f"memory:{session_id}:authorized:{uuid4().hex}",
            session_id=session_id,
            kind=request.kind,
            content=request.content.strip(),
            source_message_id=request.source_message_id,
            user_authorized=True,
        )
        self._memory.store(record)
        return record

    async def correct_memory(
        self,
        session_id: str,
        memory_id: str,
        request: MemoryCorrectionRequest,
    ) -> MemoryRecordV1:
        await self._require_session(session_id)
        current = self._memory.get(memory_id, include_expired=True)
        if current is None or current.session_id != session_id:
            raise SessionNotFound(memory_id)
        if current.kind is MemoryKind.SAFETY_LEDGER:
            raise ProtectedMemoryError(
                "safety-ledger memories cannot be corrected through the user API"
            )
        corrected_at = datetime.now(UTC)
        corrected = current.model_copy(
            update={
                "content": request.content.strip(),
                "user_authorized": (
                    True
                    if current.kind in {MemoryKind.EPISODIC, MemoryKind.PREFERENCE}
                    else current.user_authorized
                ),
                "created_at": corrected_at,
                "expires_at": (
                    corrected_at + timedelta(days=7)
                    if current.kind is MemoryKind.WORKING
                    else current.expires_at
                ),
            }
        )
        self._memory.replace(corrected)
        return corrected

    async def list_trace(self, session_id: str) -> tuple[TraceEventV1, ...]:
        session = await self._require_session(session_id)
        if not self._research_mode_enabled or not session.research_mode:
            raise PermissionError("trace access requires research_mode")
        return tuple(
            event for event in self._trace_store.read_all() if event.episode_id == session_id
        )

    def _validate_session_request(self, request: CreateSessionRequest) -> None:
        if request.research_mode and not self._research_mode_enabled:
            raise PermissionError("research_mode is disabled by the runtime configuration")
        if request.model is not None and request.model != self._renderer.model:
            raise ModelSelectionError("requested model is not the model frozen for this runtime")
        if (
            request.participant_type is ParticipantType.SYNTHETIC_CASE
            and self._known_synthetic_case_ids is not None
            and request.synthetic_case_id not in self._known_synthetic_case_ids
        ):
            raise ValueError("unknown synthetic_case_id")

    def _resolve_outstanding_proactive_prompt(self, record: SessionRecord) -> None:
        state = self._proactivity.get_state(record.session_id)
        if record.state.proactive_enabled and not state.enabled and not state.paused:
            self._proactivity.enable(record.session_id)
            state = self._proactivity.get_state(record.session_id)
        if state.outstanding_prompt_id is not None:
            self._proactivity.record_response(state.outstanding_prompt_id)

    async def _require_session(self, session_id: str) -> SessionRecord:
        if not session_id.strip():
            raise ValueError("session_id cannot be blank")
        record = await self._repository.get(session_id)
        if record is None:
            raise SessionNotFound(session_id)
        return record

    @staticmethod
    def _ensure_turn_request_is_new(
        *,
        request: TurnRequest,
        request_id: str,
        user_message_id: str,
        assistant_message_id: str,
        messages: tuple[StoredMessage, ...],
    ) -> None:
        """Reject a replay before any scheduler, handoff, or provider side effect.

        ``request_id`` is an optional idempotency key.  A completed turn is
        intentionally not re-rendered on replay: doing so could bill the
        provider twice or issue a second handoff.  Callers receive a conflict
        and can fetch the persisted session/messages instead.
        """

        if request.request_id is None:
            return
        existing_user = next(
            (message for message in messages if message.message_id == user_message_id),
            None,
        )
        existing_assistant = next(
            (message for message in messages if message.message_id == assistant_message_id),
            None,
        )
        if existing_user is None and existing_assistant is None:
            return
        if existing_user is not None and existing_user.content != request.text.strip():
            raise TurnConflictError(
                f"request_id {request_id!r} was already used with different text"
            )
        raise TurnConflictError(
            f"request_id {request_id!r} was already committed; "
            "fetch the existing session result instead of replaying it"
        )

    @staticmethod
    def _build_observation(
        *,
        record: SessionRecord,
        history: tuple[StoredMessage, ...],
        message_id: str,
        text: str,
        simulated_time: str,
    ) -> ObservationV1:
        age_group = (
            "13-17 synthetic"
            if record.participant_type is ParticipantType.SYNTHETIC_CASE
            else "adult research tester"
        )
        disclosed_background = (
            (f"synthetic_case_id:{record.synthetic_case_id}",) if record.synthetic_case_id else ()
        )
        conversation_history = tuple(
            ConversationMessage(
                message_id=message.message_id,
                role=message.role,
                content=message.content,
                simulated_time=message.simulated_time,
            )
            for message in history
        )
        return ObservationV1(
            track=Track.C,
            episode_id=record.session_id,
            step_id=record.state.turn_index,
            simulated_time=simulated_time,
            visible_profile=VisibleProfile(
                age_group=age_group,
                disclosed_background=disclosed_background,
            ),
            current_observation=CurrentObservation(
                observation_id=message_id,
                message=text,
            ),
            visible_context=VisibleContext(
                recent_events=(),
                relationship_cues=(),
                conversation_history=conversation_history,
            ),
            allowed_primary_actions=tuple(Action),
        )

    def _prepare_handoff(
        self,
        *,
        record: SessionRecord,
        outcome: PolicyOutcome,
        evidence_codes: tuple[str, ...],
        requested_at: datetime,
    ) -> HandoffStatus:
        target = outcome.plan.escalation_target
        # An unacknowledged L4 handoff is sticky.  Even if a later assessment
        # temporarily looks lower-risk, keep requesting the emergency-human
        # channel until the existing human handoff is explicitly acknowledged.
        if (
            record.state.risk_level is RiskLevel.L4
            and record.state.handoff_status is not HandoffStatus.ACKNOWLEDGED
        ):
            target = EscalationTarget.EMERGENCY_HUMAN
        if target is EscalationTarget.NONE:
            return record.state.handoff_status
        request_id = f"handoff:{record.session_id}:{record.state.turn_index}:{target.value}"
        self._handoff.request(
            HandoffRequest(
                request_id=request_id,
                session_id=record.session_id,
                target=target,
                risk_level=outcome.plan.escalation_level,
                evidence_codes=evidence_codes,
                requested_at=requested_at,
            )
        )
        self._handoff.mark_sent(request_id, at=requested_at)
        return self._handoff.get(request_id).status

    def _advance_state(
        self,
        *,
        current: SessionStateV1,
        outcome: PolicyOutcome,
        handoff_status: HandoffStatus,
        observation: ObservationV1 | None = None,
    ) -> tuple[SessionStateV1, PolicyOutcome]:
        try:
            next_state = self._state_machine.apply_policy(
                current=current,
                outcome=outcome,
                handoff_status=handoff_status,
            )
            return next_state, outcome
        except InvalidStateTransition:
            if current.risk_level is not RiskLevel.L4:
                raise
            if (
                self._web_conversation_mode
                and isinstance(self._policy, WebConversationPolicy)
                and observation is not None
                and not self._policy.has_explicit_immediate_evidence(observation)
            ):
                retained = replace(
                    outcome,
                    plan=outcome.plan.model_copy(
                        update={
                            "escalation_level": RiskLevel.L4,
                            "escalation_target": EscalationTarget.EMERGENCY_HUMAN,
                            "follow_up_required": True,
                        }
                    ),
                    next_phase=ConversationPhase.EMERGENCY,
                    reason_codes=("WEB_L4_CONTEXT_RETAINED", *outcome.reason_codes),
                )
                return self._state_machine.apply_policy(
                    current=current,
                    outcome=retained,
                    handoff_status=handoff_status,
                ), retained
            sticky = PolicyOutcome(
                plan=DecisionPlan(
                    primary_action=Action.URGENT_ESCALATION,
                    escalation_level=RiskLevel.L4,
                    escalation_target=EscalationTarget.EMERGENCY_HUMAN,
                    follow_up_required=True,
                ),
                next_phase=ConversationPhase.EMERGENCY,
                reason_codes=(
                    "EMERGENCY_REMAINS_ACTIVE_UNTIL_ACKNOWLEDGED",
                    *outcome.reason_codes,
                ),
            )
            return (
                self._state_machine.apply_policy(
                    current=current,
                    outcome=sticky,
                    handoff_status=handoff_status,
                ),
                sticky,
            )

    async def _render_safely(
        self,
        *,
        record: SessionRecord,
        observation: ObservationV1,
        assessment: AssessmentResult,
        outcome: PolicyOutcome,
        render_session: SessionStateV1,
    ) -> _SafeRender:
        memories = self._memory.list_for_session(
            record.session_id,
            kinds=(
                MemoryKind.WORKING,
                MemoryKind.EPISODIC,
                MemoryKind.PREFERENCE,
            ),
        )
        raw_snippets = (
            render_memory_snippets(
                memories,
                previous_responses=tuple(
                    message.content
                    for message in observation.visible_context.conversation_history
                    if message.role is ActorRole.COMPANION
                ),
            )
            if self._web_conversation_mode
            else tuple(memory.content for memory in memories[-6:])
        )
        current_text = observation.current_observation.message.strip()
        snippets = tuple(
            content
            for content in raw_snippets
            if content != current_text or not self._web_conversation_mode
        )
        completions: list[tuple[str, ChatCompletion]] = []
        try:
            self._ensure_budget_available()
            rendered = await self._renderer.render(
                observation=observation,
                assessment=assessment,
                policy=outcome,
                session=render_session,
                memory_snippets=snippets,
            )
        except (ProviderError, RuntimeError):
            degraded = self._auditor.degrade(outcome.plan.escalation_level)
            return _SafeRender(
                text=self._fallback_text(
                    observation=observation,
                    assessment=assessment,
                    outcome=outcome,
                    degraded_text=degraded.text,
                ),
                audit=degraded.audit,
                usages=(),
            )

        if rendered.completion is not None:
            completions.append(("render", rendered.completion))
        if not rendered.provider_used:
            text = (
                self._fallback_text(
                    observation=observation,
                    assessment=assessment,
                    outcome=outcome,
                    degraded_text=self._auditor.fallback_for(outcome.plan.escalation_level),
                )
                if self._web_conversation_mode
                else rendered.payload.text
            )
            audit = self._audited_candidate(
                text,
                observation=observation,
                outcome=outcome,
                render_session=render_session,
                candidate_index=0,
            )
            return _SafeRender(
                text=text,
                audit=SafetyAudit(
                    passed=audit.passed,
                    violation_codes=(
                        *audit.violation_codes,
                        "PROVIDER_UNAVAILABLE",
                    ),
                    used_fallback=True,
                ),
                usages=self._record_completions(
                    record.session_id,
                    render_session.turn_index,
                    completions,
                ),
            )

        initial_audit = self._audited_candidate(
            rendered.payload.text,
            observation=observation,
            outcome=outcome,
            render_session=render_session,
            candidate_index=0,
        )
        if initial_audit.passed:
            return _SafeRender(
                text=rendered.payload.text,
                audit=initial_audit,
                usages=self._record_completions(
                    record.session_id,
                    render_session.turn_index,
                    completions,
                ),
            )

        try:
            self._ensure_budget_available()
            rewritten = await self._renderer.rewrite(
                candidate_text=rendered.payload.text,
                violation_codes=initial_audit.violation_codes,
                observation=observation,
                policy=outcome,
                session=render_session,
            )
            if rewritten.completion is not None:
                completions.append(("rewrite", rewritten.completion))
            rewritten_audit = self._audited_candidate(
                rewritten.payload.text,
                observation=observation,
                outcome=outcome,
                render_session=render_session,
                candidate_index=1,
            )
            if rewritten_audit.passed:
                return _SafeRender(
                    text=rewritten.payload.text,
                    audit=rewritten_audit,
                    usages=self._record_completions(
                        record.session_id,
                        render_session.turn_index,
                        completions,
                    ),
                )
            violations = tuple(
                dict.fromkeys((*initial_audit.violation_codes, *rewritten_audit.violation_codes))
            )
        except (ProviderError, RuntimeError):
            violations = tuple(dict.fromkeys((*initial_audit.violation_codes, "REWRITE_FAILED")))

        return _SafeRender(
            text=self._fallback_text(
                observation=observation,
                assessment=assessment,
                outcome=outcome,
                degraded_text=self._auditor.fallback_for(outcome.plan.escalation_level),
            ),
            audit=SafetyAudit(
                passed=False,
                violation_codes=violations,
                used_fallback=True,
            ),
            usages=self._record_completions(
                record.session_id,
                render_session.turn_index,
                completions,
            ),
        )

    def _audited_candidate(
        self,
        text: str,
        *,
        observation: ObservationV1,
        outcome: PolicyOutcome,
        render_session: SessionStateV1,
        candidate_index: int,
    ) -> SafetyAudit:
        audit = self._auditor.audit(
            text,
            handoff_status=render_session.handoff_status,
            risk_level=outcome.plan.escalation_level,
            required_action=(
                web_audit_action(outcome)
                if self._web_conversation_mode
                else outcome.plan.primary_action
            ),
        )
        if not self._web_conversation_mode:
            return audit
        quality_codes = response_quality_codes(
            text,
            observation=observation,
            outcome=outcome,
            candidate_index=candidate_index,
        )
        return SafetyAudit(
            passed=audit.passed and not quality_codes,
            violation_codes=(*audit.violation_codes, *quality_codes),
            used_fallback=False,
        )

    def _fallback_text(
        self,
        *,
        observation: ObservationV1,
        assessment: AssessmentResult,
        outcome: PolicyOutcome,
        degraded_text: str,
    ) -> str:
        if not self._web_conversation_mode:
            return degraded_text
        return fallback_reply(
            observation=observation,
            assessment=assessment,
            outcome=outcome,
            safety_fallback=degraded_text,
        )

    def _record_completions(
        self,
        session_id: str,
        turn_index: int,
        completions: list[tuple[str, ChatCompletion]],
    ) -> tuple[ModelCallUsage, ...]:
        usages: list[ModelCallUsage] = []
        for kind, completion in completions:
            call_id = f"{session_id}:{turn_index}:{kind}:{completion.response_id or uuid4().hex}"
            self._cost_ledger.record_call(
                call_id=call_id,
                role=ModelRole.EVALUATED_AGENT,
                model=completion.response_model,
                usage=completion.usage,
                retry_index=completion.retry_count,
            )
            cost_record = next(
                record for record in self._cost_ledger.records() if record.call_id == call_id
            )
            usages.append(
                ModelCallUsage(
                    model=completion.response_model,
                    input_tokens=completion.usage.input_tokens,
                    output_tokens=completion.usage.output_tokens,
                    cached_input_tokens=completion.usage.cached_input_tokens,
                    estimated_cost_cny=(
                        float(cost_record.cost_cny) if cost_record.cost_cny is not None else 0.0
                    ),
                    retry_index=completion.retry_count,
                )
            )
        return tuple(usages)

    def _ensure_budget_available(self) -> None:
        summary = self._cost_ledger.summary()
        if summary.known_cost_cny >= self._api_budget_cny:
            raise BudgetExceeded("configured API budget has been exhausted")

    @staticmethod
    def _turn_messages(
        *,
        session_id: str,
        turn_index: int,
        user_message_id: str,
        assistant_message_id: str,
        user_text: str,
        assistant_text: str,
        simulated_time: str,
        created_at: datetime,
    ) -> tuple[StoredMessage, StoredMessage]:
        ordinal = turn_index * 2
        return (
            StoredMessage(
                message_id=user_message_id,
                session_id=session_id,
                ordinal=ordinal,
                role=ActorRole.USER,
                content=user_text,
                simulated_time=simulated_time,
                created_at=created_at,
            ),
            StoredMessage(
                message_id=assistant_message_id,
                session_id=session_id,
                ordinal=ordinal + 1,
                role=ActorRole.COMPANION,
                content=assistant_text,
                simulated_time=simulated_time,
                created_at=created_at,
            ),
        )

    def _record_memories(
        self,
        *,
        record: SessionRecord,
        state: SessionStateV1,
        user_message: StoredMessage,
        assistant_message: StoredMessage,
        reason_codes: tuple[str, ...],
        created_at: datetime,
    ) -> tuple[str, ...]:
        if not state.memory_enabled:
            return ()
        expires_at = created_at + timedelta(days=7)
        ordinary = (
            MemoryRecordV1(
                memory_id=f"memory:{user_message.message_id}",
                session_id=record.session_id,
                kind=MemoryKind.WORKING,
                content=user_message.content,
                source_message_id=user_message.message_id,
                created_at=created_at,
                expires_at=expires_at,
            ),
            MemoryRecordV1(
                memory_id=f"memory:{assistant_message.message_id}",
                session_id=record.session_id,
                kind=MemoryKind.WORKING,
                content=assistant_message.content,
                source_message_id=assistant_message.message_id,
                created_at=created_at,
                expires_at=expires_at,
            ),
        )
        for memory in ordinary:
            self._memory.store(memory)

        references = [memory.memory_id for memory in ordinary]
        if state.risk_level in {
            RiskLevel.L2,
            RiskLevel.L3,
            RiskLevel.L4,
            RiskLevel.B,
        }:
            ledger = MemoryRecordV1(
                memory_id=f"safety:{record.session_id}:{state.turn_index}",
                session_id=record.session_id,
                kind=MemoryKind.SAFETY_LEDGER,
                content=(
                    f"risk={state.risk_level.value}; "
                    f"reason_codes={','.join(reason_codes)}; "
                    f"handoff={state.handoff_status.value}"
                ),
                source_message_id=user_message.message_id,
                created_at=created_at,
            )
            self._memory.store(ledger)
            references.append(ledger.memory_id)
        return tuple(references)

    @staticmethod
    def _stable_seed(session_id: str, turn_index: int) -> int:
        digest = hashlib.sha256(f"{session_id}:{turn_index}".encode()).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)

    @staticmethod
    def _view(record: SessionRecord) -> SessionView:
        return SessionView(
            session_id=record.session_id,
            state=record.state,
            model=record.model,
            research_mode=record.research_mode,
            participant_type=record.participant_type,
            synthetic_case_id=record.synthetic_case_id,
        )
