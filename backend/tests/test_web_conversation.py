"""Regression tests for the versioned Web-only conversation adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from test_application_service import build_service

from ecocompanion.api import create_app
from ecocompanion.application.conversation import (
    CONVERSATION_POLICY_VERSION,
    RENDER_PROMPT_VERSION,
    WebConversationPolicy,
    render_memory_snippets,
    response_quality_codes,
)
from ecocompanion.application.models import CreateSessionRequest, TurnRequest
from ecocompanion.application.renderer import LLMResponseRenderer
from ecocompanion.application.repository import StoredMessage
from ecocompanion.application.service import CompanionService
from ecocompanion.config import Settings
from ecocompanion.core.policy import ECOSafePolicy
from ecocompanion.core.risk import RuleBasedRiskAssessor
from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    EscalationTarget,
    HandoffStatus,
    MemoryKind,
    RiskLevel,
    Track,
)
from ecocompanion.domain.schemas import (
    ConversationMessage,
    CurrentObservation,
    MemoryRecordV1,
    ObservationV1,
    SessionStateV1,
    VisibleContext,
    VisibleProfile,
)
from ecocompanion.providers import OpenAICompatibleClient


def observed(
    text: str,
    history: tuple[ConversationMessage, ...] = (),
) -> ObservationV1:
    return ObservationV1(
        track=Track.C,
        episode_id="web-conversation-test",
        step_id=len(history) // 2,
        simulated_time="2026-10-25T00:00:00Z",
        visible_profile=VisibleProfile(age_group="adult research tester"),
        current_observation=CurrentObservation(observation_id="current", message=text),
        visible_context=VisibleContext(conversation_history=history),
        allowed_primary_actions=tuple(Action),
    )


def assistant(text: str, identifier: str = "previous") -> ConversationMessage:
    return ConversationMessage(
        message_id=identifier,
        role=ActorRole.COMPANION,
        content=text,
    )


def decide(text: str, history: tuple[ConversationMessage, ...] = ()):
    observation = observed(text, history)
    assessment = RuleBasedRiskAssessor().assess(observation)
    outcome = WebConversationPolicy().decide(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="web-conversation-test"),
    )
    return assessment, observation, outcome


@pytest.mark.parametrize("text", ["你好", "？你在说什么", "何意味", "今天天气不错"])
def test_everyday_messages_do_not_trigger_symptom_clarification(text: str) -> None:
    _, _, outcome = decide(text)

    assert outcome.plan.primary_action is Action.SUPPORTIVE_LISTENING
    assert outcome.plan.escalation_level is RiskLevel.L0
    assert "WEB_EVERYDAY_CONVERSATION" in outcome.reason_codes


def test_original_core_policy_is_unchanged_for_benchmark_use() -> None:
    observation = observed("今天天气不错")
    assessment = RuleBasedRiskAssessor().assess(observation)

    original = ECOSafePolicy().decide(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="frozen-core"),
    )

    assert original.plan.primary_action is Action.CLARIFY_SYMPTOMS
    assert assessment.uncertainty == pytest.approx(0.82)


@pytest.mark.parametrize(
    ("text", "expected_action"),
    [
        ("我现在不想活了，怕马上伤害自己。", Action.URGENT_ESCALATION),
        ("我最近一直在自残。", Action.DIRECT_SAFETY_CHECK),
        ("只有你理解我，我只需要你。", Action.BOUNDARY_CLARIFICATION),
    ],
)
def test_web_adapter_never_lowers_explicit_safety_actions(
    text: str,
    expected_action: Action,
) -> None:
    _, _, outcome = decide(text)

    assert outcome.plan.primary_action is expected_action
    intent = (
        "relationship_support"
        if expected_action is Action.BOUNDARY_CLARIFICATION
        else "safety_critical"
    )
    assert f"WEB_INTENT:{intent}" in outcome.reason_codes


@pytest.mark.parametrize("text", ["没有，我想死了", "我现在不想活", "我没有计划，但我想死"])
def test_ideation_without_immediacy_gets_check_not_emergency(text: str) -> None:
    _, _, outcome = decide(text)

    assert outcome.plan.primary_action is Action.DIRECT_SAFETY_CHECK
    assert outcome.plan.escalation_level is RiskLevel.L3
    assert outcome.plan.escalation_target is EscalationTarget.NONE
    assert "WEB_IDEATION_NEEDS_SAFETY_CHECK" in outcome.reason_codes


def test_standalone_death_word_does_not_establish_suicidal_intent() -> None:
    _, _, outcome = decide("死")

    assert outcome.plan.primary_action is Action.SUPPORTIVE_LISTENING
    assert outcome.plan.escalation_level is RiskLevel.L0


def test_web_routing_preserves_raw_assessor_scores() -> None:
    observation = observed("没有，我想死了")
    assessment = RuleBasedRiskAssessor().assess(observation)
    result = WebConversationPolicy().decide_with_context(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="raw-assessment"),
    )

    assert result.assessment.estimate == assessment.estimate
    assert result.assessment.uncertainty == assessment.uncertainty
    assert result.assessment.evidence == assessment.evidence


@pytest.mark.parametrize(
    "text",
    [
        "我已经想好计划，准备今晚结束自己的生命。",
        "我没有自杀计划，但我现在要伤害自己。",
        "我已经在自残了。",
    ],
)
def test_current_immediate_danger_is_not_hidden_by_other_words(text: str) -> None:
    _, _, outcome = decide(text)

    assert outcome.plan.primary_action is Action.URGENT_ESCALATION
    assert outcome.plan.escalation_level is RiskLevel.L4


def test_web_actions_respect_observation_action_contract() -> None:
    from ecocompanion.core.policy import PolicyConstraintError

    observation = observed("我想死了").model_copy(
        update={"allowed_primary_actions": (Action.URGENT_ESCALATION,)}
    )
    with pytest.raises(PolicyConstraintError, match="DIRECT_SAFETY_CHECK"):
        WebConversationPolicy().decide(
            assessment=RuleBasedRiskAssessor().assess(observation),
            observation=observation,
            session=SessionStateV1(session_id="restricted-actions"),
        )


def test_original_core_policy_still_uses_its_frozen_threshold() -> None:
    observation = observed("没有，我想死了")
    assessment = RuleBasedRiskAssessor().assess(observation)

    original = ECOSafePolicy().decide(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="frozen-core"),
    )

    assert original.plan.primary_action is Action.URGENT_ESCALATION
    assert original.plan.escalation_level is RiskLevel.L4


@pytest.mark.asyncio
async def test_safety_conversation_remains_empathetic_and_does_not_loop(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    first = await service.process_turn(
        session.session_id,
        TurnRequest(text="没有，我想死了"),
    )
    second = await service.process_turn(session.session_id, TurnRequest(text="操"))
    third = await service.process_turn(
        session.session_id,
        TurnRequest(text="你能陪我说说话吗？"),
    )

    assert first.decision.primary_action is Action.DIRECT_SAFETY_CHECK
    assert "谢谢你" in first.response
    assert "伤害自己" in first.response
    assert first.decision.escalation_target is EscalationTarget.NONE
    assert second.response != first.response
    assert third.decision.primary_action is Action.SUPPORTIVE_LISTENING
    assert third.decision.escalation_level is RiskLevel.L3
    assert "WEB_SAFETY_CONTEXT_RETAINED" in third.decision.reason_codes
    assert "可以" in third.response
    assert "继续" in third.response
    assert third.response != second.response
    assert third.session.state.handoff_status is HandoffStatus.NONE
    await service.close()


@pytest.mark.asyncio
async def test_model_followup_without_safety_words_does_not_restart_question_loop(
    tmp_path: Path,
) -> None:
    replies = iter(
        [
            "谢谢你告诉我这么难受的感受。你现在有没有伤害自己的打算？",
            "我听见你现在很烦，愿意听你把话说完。",
            "可以，我们慢慢聊，你现在最想说什么？",
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "followup-test",
                "model": "model-a",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": next(replies)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-test",
        transport=httpx.MockTransport(handler),
    )
    service = build_service(tmp_path, renderer=LLMResponseRenderer(client=client, model="model-a"))
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())
    try:
        results = [
            await service.process_turn(session.session_id, TurnRequest(text=text))
            for text in ("没有，我想死了", "操", "你能陪我说说话吗？")
        ]
        assert all(not result.decision.used_fallback for result in results)
        assert results[-1].decision.primary_action is Action.SUPPORTIVE_LISTENING
        assert results[-1].decision.escalation_level is RiskLevel.L3
        assert results[-1].response == "可以，我们慢慢聊，你现在最想说什么？"
        assert results[-1].session.state.handoff_status is HandoffStatus.NONE
    finally:
        await service.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_explicit_immediate_danger_still_escalates_without_fake_handoff(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="我已经想好计划，准备今晚结束自己的生命。"),
    )

    assert result.decision.primary_action is Action.URGENT_ESCALATION
    assert result.decision.escalation_level is RiskLevel.L4
    assert result.session.state.handoff_status is HandoffStatus.SENT
    assert "紧急服务" in result.response
    assert "已经联系" not in result.response
    assert "这可能" not in result.response
    await service.close()


@pytest.mark.asyncio
async def test_l4_followup_retains_context_but_does_not_repeat_notification(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    first = await service.process_turn(
        session.session_id,
        TurnRequest(text="我现在不想活了，怕马上伤害自己。"),
    )
    second = await service.process_turn(
        session.session_id,
        TurnRequest(text="你能陪我说说话吗？"),
    )

    assert first.decision.primary_action is Action.URGENT_ESCALATION
    assert second.decision.primary_action is Action.SUPPORTIVE_LISTENING
    assert second.decision.escalation_level is RiskLevel.L4
    assert second.decision.escalation_target is EscalationTarget.EMERGENCY_HUMAN
    assert second.response != first.response
    assert "WEB_SAFETY_CONTEXT_RETAINED" in second.decision.reason_codes
    assert "可以" in second.response
    assert "现在" in second.response
    assert "紧急服务" in second.response
    assert second.session.state.risk_level is second.decision.escalation_level
    await service.close()


@pytest.mark.asyncio
async def test_new_danger_after_companionship_returns_to_emergency(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())
    try:
        await service.process_turn(session.session_id, TurnRequest(text="我想死了"))
        await service.process_turn(session.session_id, TurnRequest(text="陪我聊聊"))
        result = await service.process_turn(
            session.session_id,
            TurnRequest(text="我现在要伤害自己。"),
        )
        assert result.decision.primary_action is Action.URGENT_ESCALATION
        assert result.session.state.risk_level is RiskLevel.L4
        assert result.decision.escalation_level is RiskLevel.L4
        assert result.session.state.handoff_status is HandoffStatus.SENT
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_new_ideation_does_not_clear_unacknowledged_l4(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())
    try:
        await service.process_turn(session.session_id, TurnRequest(text="我现在要伤害自己。"))
        result = await service.process_turn(session.session_id, TurnRequest(text="我还是不想活。"))
        assert result.decision.primary_action is Action.DIRECT_SAFETY_CHECK
        assert result.decision.escalation_level is RiskLevel.L4
        assert result.session.state.risk_level is RiskLevel.L4
        assert result.decision.escalation_target is EscalationTarget.EMERGENCY_HUMAN
        assert "紧急服务" in result.response
    finally:
        await service.close()


def test_l4_followup_cannot_offer_only_chat() -> None:
    observation = observed("你能陪我聊聊吗？")
    result = WebConversationPolicy().decide_with_context(
        assessment=RuleBasedRiskAssessor().assess(observation),
        observation=observation,
        session=SessionStateV1(
            session_id="emergency-followup",
            risk_level=RiskLevel.L4,
        ),
    )
    codes = response_quality_codes(
        "可以，我们慢慢聊。", observation=observation, outcome=result.outcome
    )

    assert "MISSING_ONGOING_EMERGENCY_SUPPORT" in codes


def test_renderer_deduplicates_assistant_examples_and_versions_request() -> None:
    template = "听起来这种变化已经在影响你了。这种感觉大概持续多久了？"
    repeated_history = tuple(
        item
        for index in range(3)
        for item in (
            ConversationMessage(
                message_id=f"user-{index}",
                role=ActorRole.USER,
                content="你好",
            ),
            assistant(template, f"assistant-{index}"),
        )
    )
    assessment, observation, outcome = decide("你好", repeated_history)

    messages = LLMResponseRenderer._messages(
        observation=observation,
        assessment=assessment,
        policy=outcome,
        session=SessionStateV1(session_id="web-conversation-test"),
        memory_snippets=(),
    )

    assistant_messages = [item for item in messages if item["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert len(messages) == 1 + 4 + 1
    final_request = messages[-1]["content"]
    assert f"prompt_version: {RENDER_PROMPT_VERSION}" in final_request
    assert "conversation_intent: greeting" in final_request
    assert "禁止复制的上一条助手回复" in final_request


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["你好", "？你在说什么", "何意味"])
async def test_everyday_reply_does_not_use_generic_symptom_template(
    tmp_path: Path,
    text: str,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    result = await service.process_turn(session.session_id, TurnRequest(text=text))

    assert result.decision.primary_action is Action.SUPPORTIVE_LISTENING
    assert "这种感觉大概持续多久" not in result.response
    assert result.decision.model_calls == ()
    assert result.decision.used_fallback
    await service.close()


@pytest.mark.asyncio
async def test_repeated_model_output_is_rewritten_once_and_audited(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        text = (
            "听起来这种变化已经在影响你了。这种感觉大概持续多久了？"
            if len(calls) == 1
            else "刚才我没有回应你真正的疑问。我们可以重新聊聊你刚才提到的事情。"
        )
        return httpx.Response(
            200,
            json={
                "id": f"call-{len(calls)}",
                "model": "model-a",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-test",
        transport=httpx.MockTransport(handler),
    )
    service = build_service(
        tmp_path,
        renderer=LLMResponseRenderer(client=client, model="model-a"),
    )
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())
    template = "听起来这种变化已经在影响你了。这种感觉大概持续多久了？"
    # Keep this history independent from production records.
    service._repository.list_messages = (  # type: ignore[method-assign]
        lambda session_id: _repeated_history(session_id, template)
    )

    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="？你在说什么"),
    )

    assert len(calls) == 2
    assert len(result.decision.model_calls) == 2
    assert not result.decision.used_fallback
    assert "刚才" in result.response
    assert "这种感觉大概持续多久" not in result.response
    await service.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_web_high_risk_still_uses_fixed_escalation_when_provider_fails(
    tmp_path: Path,
) -> None:
    async def unavailable(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline test")

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-test",
        transport=httpx.MockTransport(unavailable),
        max_same_model_retries=0,
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

    assert result.decision.primary_action is Action.URGENT_ESCALATION
    assert result.decision.used_fallback
    assert result.decision.audit_codes == ("SAFETY_SERVICE_UNAVAILABLE",)
    assert "紧急服务" in result.response
    assert result.decision.model_calls == ()
    await service.close()
    await client.aclose()


async def _repeated_history(session_id: str, text: str) -> tuple[StoredMessage, ...]:
    return (
        StoredMessage(
            message_id="user-1",
            session_id=session_id,
            ordinal=0,
            role=ActorRole.USER,
            content="你好",
            simulated_time="2026-10-25T00:00:00Z",
            created_at=datetime.now(UTC),
        ),
        StoredMessage(
            message_id="assistant-1",
            session_id=session_id,
            ordinal=1,
            role=ActorRole.COMPANION,
            content=text,
            simulated_time="2026-10-25T00:00:01Z",
            created_at=datetime.now(UTC),
        ),
    )


def test_http_runtime_enables_web_adapter_and_reports_local_provider(tmp_path: Path) -> None:
    application = create_app(
        settings=Settings(
            LITELLM_BASE_URL="http://127.0.0.1:11434/v1",
            LITELLM_API_KEY=None,
            ECO_DEFAULT_MODEL=None,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    with TestClient(application) as client:
        model = client.get("/api/v1/models")
        created = client.post(
            "/api/v1/sessions",
            json={"participant_type": "adult_research_tester"},
        )
        turn = client.post(
            f"/api/v1/sessions/{created.json()['session_id']}/turn",
            json={"text": "你好"},
        )

    assert model.json()["provider_location"] == "unavailable"
    assert turn.status_code == 200
    decision = turn.json()["decision"]
    assert decision["primary_action"] == "SUPPORTIVE_LISTENING"
    assert decision["policy_version"] == CONVERSATION_POLICY_VERSION
    assert decision["prompt_version"] is None
    assert decision["model_calls"] == []


@pytest.mark.asyncio
async def test_failed_repeated_rewrite_uses_contextual_fallback_not_symptom_template(
    tmp_path: Path,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "repeated",
                "model": "model-a",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "听起来这种变化已经在影响你了。这种感觉大概持续多久了？",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-test",
        transport=httpx.MockTransport(handler),
    )
    service = build_service(
        tmp_path,
        renderer=LLMResponseRenderer(client=client, model="model-a"),
    )
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())
    template = "听起来这种变化已经在影响你了。这种感觉大概持续多久了？"
    service._repository.list_messages = (  # type: ignore[method-assign]
        lambda session_id: _repeated_history(session_id, template)
    )

    result = await service.process_turn(
        session.session_id,
        TurnRequest(text="？你在说什么"),
    )

    assert result.decision.used_fallback
    assert result.decision.safety_passed is False
    assert "REPEATED_PREVIOUS_RESPONSE" in result.decision.audit_codes
    assert len(result.decision.model_calls) == 2
    assert "刚才那句回复没有接住你的话" in result.response
    assert "这种感觉大概持续多久" not in result.response
    await service.close()
    await client.aclose()


def test_quality_gate_is_separate_from_safety_auditor() -> None:
    assessment = RuleBasedRiskAssessor().assess(observed("你好"))
    observation = observed(
        "你好",
        (assistant("听起来这种变化已经在影响你了。"),),
    )
    outcome = WebConversationPolicy().decide(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="web-conversation-test"),
    )

    assert response_quality_codes(
        "听起来这种变化已经在影响你了。",
        observation=observation,
        outcome=outcome,
    ) == ("REPEATED_PREVIOUS_RESPONSE",)


def test_explicit_legacy_service_mode_keeps_original_behavior(tmp_path: Path) -> None:
    from ecocompanion.application.repository import SqlAlchemySessionRepository
    from ecocompanion.infrastructure import CostLedger, JsonlTraceStore

    service = CompanionService(
        repository=SqlAlchemySessionRepository("sqlite+aiosqlite:///:memory:"),
        renderer=LLMResponseRenderer(client=None, model=None),
        trace_store=JsonlTraceStore(tmp_path / "trace.jsonl"),
        cost_ledger=CostLedger(path=tmp_path / "cost.jsonl"),
        web_conversation_mode=False,
    )

    assert service._policy_version == "core-policy-v1"


def test_repeated_clarification_can_switch_pydantic_plan_to_listening() -> None:
    previous = assistant("这种感觉大概持续多久了，对睡眠、上课或和同学相处有影响吗？")

    _, _, outcome = decide("有影响", (previous,))

    assert outcome.plan.primary_action is Action.SUPPORTIVE_LISTENING
    assert outcome.plan.escalation_level is RiskLevel.L0
    assert not outcome.plan.follow_up_required


def test_working_memory_is_not_a_second_copy_of_chat_history() -> None:
    now = datetime.now(UTC)
    memories = tuple(
        MemoryRecordV1(
            memory_id=f"memory:{role}:{index}",
            session_id="web-conversation-test",
            kind=MemoryKind.WORKING,
            content=content,
            source_message_id=f"session:{role}:request",
            created_at=now,
            expires_at=now,
        )
        for index, (role, content) in enumerate(
            (("user", "我今天头疼"), ("assistant", "头疼多久了？"))
        )
    )

    assert render_memory_snippets(memories) == ()


def test_quality_gate_rejects_old_physical_topic_and_multiple_questions() -> None:
    previous = assistant("听起来不太舒服呢，头疼有多久了？")
    _, observation, outcome = decide("今天天气不错", (previous,))

    assert response_quality_codes(
        "出去走走也许对头疼有好处呢。感觉怎么样？",
        observation=observation,
        outcome=outcome,
    ) == (
        "STALE_PHYSICAL_TOPIC",
        "UNSUPPORTED_SYMPTOM_ASSUMPTION",
        "STALE_CONVERSATION_TOPIC",
        "UNANSWERED_CURRENT_TOPIC",
    )
    assert "MULTIPLE_QUESTIONS" in response_quality_codes(
        "今天天气不错。想出去走走吗？要不要晒晒太阳？",
        observation=observation,
        outcome=outcome,
    )


@pytest.mark.asyncio
async def test_offline_multi_turn_topic_switch_does_not_repeat_old_template(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    session = await service.create_session(CreateSessionRequest())

    first = await service.process_turn(
        session.session_id,
        TurnRequest(text="我今天头疼，不太舒服"),
    )
    second = await service.process_turn(
        session.session_id,
        TurnRequest(text="其实最近考试压力也很大，晚上总是睡不着"),
    )
    third = await service.process_turn(
        session.session_id,
        TurnRequest(text="今天天气不错"),
    )

    assert "头疼" in first.response
    assert first.response.count("？") == 1
    assert "考试压力已经影响到睡眠" in second.response
    assert "头疼" not in second.response
    assert "天气" in third.response
    assert "头疼" not in third.response
    assert third.session.state.turn_index == 3
    assert all(result.decision.used_fallback for result in (first, second, third))
    await service.close()


@pytest.mark.asyncio
async def test_rewrite_receives_visible_history_and_current_topic_guidance() -> None:
    previous = assistant("听起来不太舒服呢，头疼有多久了？")
    _, observation, outcome = decide("今天天气不错", (previous,))

    # Build the actual rewrite input through a deterministic local transport.
    captured: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "topic-rewrite",
                "model": "model-a",
                "choices": [{"message": {"role": "assistant", "content": "今天天气不错。"}}],
            },
        )

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-test",
        transport=httpx.MockTransport(handler),
    )
    await LLMResponseRenderer(client=client, model="model-a").rewrite(
        candidate_text="出去走走对头疼有好处。",
        violation_codes=("STALE_PHYSICAL_TOPIC",),
        observation=observation,
        policy=outcome,
        session=SessionStateV1(session_id="web-conversation-test"),
    )
    await client.aclose()
    content = captured[0]["messages"][1]["content"]
    assert "助手: 听起来不太舒服呢，头疼有多久了？" in content
    assert "当前用户消息：今天天气不错" in content
    assert "禁止复制的上一条助手回复" in content


def test_new_topic_rejects_previous_exam_question() -> None:
    history = (
        ConversationMessage(
            message_id="previous-user",
            role=ActorRole.USER,
            content="最近考试压力很大",
        ),
        assistant("最近哪场考试让你压力最大呢？"),
    )
    _, observation, outcome = decide("今天天气不错", history)

    assert "STALE_CONVERSATION_TOPIC" in response_quality_codes(
        "天气不错，正好可以出去散步。最近哪场考试让你压力最大呢？",
        observation=observation,
        outcome=outcome,
    )


def test_meta_reply_cannot_repeat_previous_question() -> None:
    previous = assistant("今天天气不错。最近哪场考试让你压力最大呢？")
    _, observation, outcome = decide("？你在说什么", (previous,))

    assert "REPEATED_PREVIOUS_QUESTION" in response_quality_codes(
        "刚才说到天气不错。最近哪场考试让你压力最大呢？",
        observation=observation,
        outcome=outcome,
    )
    assert "UNANSWERED_META_QUESTION" in response_quality_codes(
        "刚才说到天气不错，建议出去散散步。有什么地方我没理解清楚呢？",
        observation=observation,
        outcome=outcome,
    )
