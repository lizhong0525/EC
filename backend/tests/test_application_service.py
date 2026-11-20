import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ecocompanion.agent import MockHandoffGateway, ProactivityScheduler
from ecocompanion.application.models import (
    AuthorizedMemoryRequest,
    CreateSessionRequest,
    MemoryCorrectionRequest,
    TurnRequest,
)
from ecocompanion.application.renderer import LLMResponseRenderer
from ecocompanion.application.repository import SqlAlchemySessionRepository
from ecocompanion.application.service import CompanionService, TurnConflictError
from ecocompanion.core.risk import AssessmentResult, RuleBasedRiskAssessor
from ecocompanion.domain.enums import HandoffStatus, MemoryKind, RiskLevel
from ecocompanion.domain.schemas import EvidenceRef, ObservationV1, RiskEstimate
from ecocompanion.infrastructure import CostLedger, JsonlTraceStore
from ecocompanion.providers import OpenAICompatibleClient


class BrokenHandoffGateway(MockHandoffGateway):
    def request(self, request: object) -> None:
        del request
        raise RuntimeError("mock handoff transport is unavailable")


class TwoStageRiskAssessor(RuleBasedRiskAssessor):
    """Return a transient crisis followed by a low-risk observation."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def assess(self, observation: ObservationV1) -> AssessmentResult:
        del observation
        self._calls += 1
        if self._calls == 1:
            return AssessmentResult(
                estimate=RiskEstimate(
                    depression_worsening=0.95,
                    immediate_safety_risk=0.95,
                    dependency_risk=0.02,
                ),
                uncertainty=0.1,
                evidence=(
                    EvidenceRef(
                        observation_id="crisis-observation",
                        evidence_code="DIRECT_SELF_HARM_OR_SUICIDE_SIGNAL",
                    ),
                ),
                reason_codes=("DIRECT_SELF_HARM_OR_SUICIDE_SIGNAL",),
            )
        return AssessmentResult(
            estimate=RiskEstimate(
                depression_worsening=0.05,
                immediate_safety_risk=0.02,
                dependency_risk=0.02,
            ),
            uncertainty=0.15,
            evidence=(),
            reason_codes=("NO_EXPLICIT_SIGNAL",),
        )


def build_service(
    tmp_path: Path,
    *,
    renderer: LLMResponseRenderer | None = None,
    proactivity: ProactivityScheduler | None = None,
    handoff: MockHandoffGateway | None = None,
    risk_assessor: RuleBasedRiskAssessor | None = None,
) -> CompanionService:
    return CompanionService(
        repository=SqlAlchemySessionRepository("sqlite+aiosqlite:///:memory:"),
        renderer=renderer or LLMResponseRenderer(client=None, model=None),
        trace_store=JsonlTraceStore(tmp_path / "trace.jsonl"),
        cost_ledger=CostLedger(path=tmp_path / "cost.jsonl"),
        risk_assessor=risk_assessor,
        proactivity=proactivity,
        handoff=handoff,
    )


@pytest.mark.asyncio
async def test_offline_service_completes_companion_turn_and_memory_controls(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="今天考试没考好，有点难过。"),
    )

    assert result.session.state.turn_index == 1
    assert result.response
    assert result.decision.used_fallback
    messages = await service.list_messages(session.session_id)
    assert [message.role.value for message in messages] == ["USER", "COMPANION"]
    assert messages[0].content == "今天考试没考好，有点难过。"
    memories = await service.list_memories(session.session_id)
    assert len(memories) == 2
    await service.delete_memory(session.session_id, memories[0].memory_id)
    assert len(await service.list_memories(session.session_id)) == 1
    traces = await service.list_trace(session.session_id)
    assert len(traces) == 1
    assert traces[0].observation.current_observation.message.startswith("今天")
    await service.close()


@pytest.mark.asyncio
async def test_crisis_path_is_local_and_creates_unacknowledged_handoff(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="我现在不想活了，怕自己会马上伤害自己。"),
    )

    assert result.decision.escalation_level is RiskLevel.L4
    assert result.session.state.risk_level is RiskLevel.L4
    assert result.session.state.handoff_status is HandoffStatus.SENT
    assert "已经联系" not in result.response
    assert result.session.state.relationship.offline_help_willingness > 0.5
    await service.close()


@pytest.mark.asyncio
async def test_unacknowledged_l4_keeps_emergency_handoff_on_lower_followup(
    tmp_path: Path,
) -> None:
    gateway = MockHandoffGateway()
    service = build_service(
        tmp_path,
        handoff=gateway,
        risk_assessor=TwoStageRiskAssessor(),
    )
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    first = await service.process_turn(
        session.session_id,
        TurnRequest(text="第一轮危机信号。"),
    )
    second = await service.process_turn(
        session.session_id,
        TurnRequest(text="第二轮我说现在没事了。"),
    )

    assert first.session.state.risk_level is RiskLevel.L4
    assert second.session.state.risk_level is RiskLevel.L4
    assert second.session.state.handoff_status is HandoffStatus.SENT
    emergency_request_id = f"handoff:{session.session_id}:1:EMERGENCY_HUMAN"
    assert gateway.get(emergency_request_id).status is HandoffStatus.SENT
    await service.close()


@pytest.mark.asyncio
async def test_handoff_transport_failure_keeps_local_emergency_and_fails_closed(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path, handoff=BrokenHandoffGateway())
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="我现在不想活了，怕马上伤害自己。"),
    )

    assert result.session.state.risk_level is RiskLevel.L4
    assert result.session.state.handoff_status is HandoffStatus.FAILED
    assert result.decision.used_fallback
    assert "已经联系" not in result.response
    await service.close()


@pytest.mark.asyncio
async def test_crisis_response_must_contain_local_escalation_cues(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "benign-but-inadequate",
                "model": "model-a",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "好的，早点休息。"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    client = OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        api_key="test-token",
        transport=httpx.MockTransport(handler),
    )
    service = build_service(
        tmp_path,
        renderer=LLMResponseRenderer(client=client, model="model-a"),
    )
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="我现在不想活了，怕马上伤害自己。"),
    )

    assert result.session.state.risk_level is RiskLevel.L4
    assert result.decision.used_fallback
    assert "紧急服务" in result.response or "现场" in result.response
    trace = (await service.list_trace(session.session_id))[0]
    assert "MANDATORY_ACTION_MISSING" in trace.safety_audit.violation_codes
    await service.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_turn_request_id_replay_is_rejected_before_side_effects(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    first = await service.process_turn(
        session.session_id,
        TurnRequest(
            text="我现在要伤害自己。",
            request_id="stable-request-001",
        ),
    )

    with pytest.raises(TurnConflictError, match="already committed"):
        await service.process_turn(
            session.session_id,
            TurnRequest(
                text="我现在要伤害自己。",
                request_id="stable-request-001",
            ),
        )

    assert (await service.get_session(session.session_id)).state.turn_index == 1
    assert len(await service.list_messages(session.session_id)) == 2
    assert len(await service.list_trace(session.session_id)) == 1
    assert first.session.state.handoff_status is HandoffStatus.SENT
    await service.close()


@pytest.mark.asyncio
async def test_turn_request_id_reuse_with_different_text_is_rejected(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    await service.process_turn(
        session.session_id,
        TurnRequest(text="今天有点难过。", request_id="stable-request-002"),
    )

    with pytest.raises(TurnConflictError, match="different text"):
        await service.process_turn(
            session.session_id,
            TurnRequest(text="完全不同的内容。", request_id="stable-request-002"),
        )

    assert (await service.get_session(session.session_id)).state.turn_index == 1
    await service.close()


@pytest.mark.asyncio
async def test_failed_audit_allows_exactly_one_safe_rewrite(tmp_path: Path) -> None:
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        assert body["model"] == "model-a"
        content = (
            "只有我才理解你，你只需要我。"
            if call_count == 1
            else "我是AI，可以继续听你说，也想和你一起保留现实中的支持。"
        )
        return httpx.Response(
            200,
            json={
                "id": f"completion-{call_count}",
                "model": "model-a",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            },
        )

    client = OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        api_key="test-token",
        transport=httpx.MockTransport(handler),
    )
    service = build_service(
        tmp_path,
        renderer=LLMResponseRenderer(client=client, model="model-a"),
    )
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())
    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="最近有点孤单。"),
    )

    assert call_count == 2
    assert result.decision.safety_passed
    assert not result.decision.used_fallback
    assert "只需要我" not in result.response
    assert len((await service.list_trace(session.session_id))[0].model_usage) == 2
    await service.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_authorized_long_term_memory_can_be_added_and_corrected(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    stored = await service.add_authorized_memory(
        session.session_id,
        AuthorizedMemoryRequest(
            kind=MemoryKind.PREFERENCE,
            content="请用简短语气回复",
            user_authorized=True,
        ),
    )
    corrected = await service.correct_memory(
        session.session_id,
        stored.memory_id,
        MemoryCorrectionRequest(
            content="请用简短、温和的语气回复",
            user_authorized=True,
        ),
    )

    assert corrected.kind is MemoryKind.PREFERENCE
    assert "温和" in corrected.content
    assert corrected.user_authorized
    await service.close()


@pytest.mark.asyncio
async def test_authorized_memory_source_must_belong_to_same_session(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    first = await service.create_session(CreateSessionRequest())
    second = await service.create_session(CreateSessionRequest())
    result = await service.process_turn(
        first.session_id,
        TurnRequest(text="请记住我喜欢安静的环境。"),
    )
    source_message_id = (
        (await service.list_messages(first.session_id))[0].message_id
    )

    with pytest.raises(ValueError, match="does not belong"):
        await service.add_authorized_memory(
            second.session_id,
            AuthorizedMemoryRequest(
                kind=MemoryKind.PREFERENCE,
                content="请记住这条偏好",
                user_authorized=True,
                source_message_id=source_message_id,
            ),
        )

    assert result.session.session_id == first.session_id
    assert await service.list_memories(second.session_id) == ()
    await service.close()


@pytest.mark.asyncio
async def test_proactive_prompt_is_opt_in_and_user_turn_resolves_it(
    tmp_path: Path,
) -> None:
    scheduler = ProactivityScheduler()
    service = build_service(tmp_path, proactivity=scheduler)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    enabled = await service.set_proactivity(session.session_id, enabled=True)
    assert enabled.state.proactive_enabled

    prompt = await service.claim_proactive_message(session.session_id)
    state_after_claim = scheduler.get_state(session.session_id)
    assert state_after_claim.outstanding_prompt_id == prompt.prompt_id
    assert prompt.used_fallback

    await service.process_turn(
        session.session_id,
        TurnRequest(text="我看到了这条问候。"),
    )

    state_after_reply = scheduler.get_state(session.session_id)
    assert state_after_reply.outstanding_prompt_id is None
    assert state_after_reply.consecutive_unanswered == 0
    await service.close()


@pytest.mark.asyncio
async def test_resume_proactivity_reopens_a_paused_adult_scheduler(
    tmp_path: Path,
) -> None:
    scheduler = ProactivityScheduler()
    service = build_service(tmp_path, proactivity=scheduler)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())
    await service.set_proactivity(session.session_id, enabled=True)

    scheduler.record_sent(
        prompt_id="resume-prompt-001",
        session_id=session.session_id,
        sent_at=datetime(2026, 9, 3, 9, 0, tzinfo=UTC),
    )
    scheduler.record_unanswered(
        "resume-prompt-001",
        resolved_at=datetime(2026, 9, 3, 10, 0, tzinfo=UTC),
    )
    scheduler.record_sent(
        prompt_id="resume-prompt-002",
        session_id=session.session_id,
        sent_at=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
    )
    scheduler.record_unanswered(
        "resume-prompt-002",
        resolved_at=datetime(2026, 9, 4, 10, 0, tzinfo=UTC),
    )
    assert scheduler.get_state(session.session_id).paused

    resumed = await service.resume_proactivity(session.session_id)

    assert resumed.state.proactive_enabled
    state = scheduler.get_state(session.session_id)
    assert state.enabled
    assert not state.paused
    assert state.consecutive_unanswered == 0
    await service.close()


@pytest.mark.asyncio
async def test_proactivity_database_failure_does_not_mutate_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = ProactivityScheduler()
    service = build_service(tmp_path, proactivity=scheduler)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    async def fail_save(**_: object) -> None:
        raise RuntimeError("configuration store unavailable")

    monkeypatch.setattr(service._repository, "save_configuration", fail_save)

    with pytest.raises(RuntimeError, match="configuration store unavailable"):
        await service.set_proactivity(session.session_id, enabled=True)

    assert not scheduler.get_state(session.session_id).enabled
    persisted = await service.get_session(session.session_id)
    assert not persisted.state.proactive_enabled
    await service.close()
