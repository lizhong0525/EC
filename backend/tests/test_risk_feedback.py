"""Owner feedback parsing and Web-only semantic routing regressions."""

from __future__ import annotations

import pytest
from test_application_service import build_service
from test_web_conversation import observed

from ecocompanion.application.conversation import (
    WebConversationPolicy,
    fallback_reply,
    response_quality_codes,
)
from ecocompanion.application.models import CreateSessionRequest, TurnRequest
from ecocompanion.core.risk import RuleBasedRiskAssessor
from ecocompanion.domain.enums import Action, RiskLevel
from ecocompanion.domain.schemas import SessionStateV1
from scripts.import_risk_review import red_choice


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("我没有想死，也没有伤害自己的念头，只是工作很累。", Action.SUPPORTIVE_LISTENING),
        ("我不会现在伤害自己，也没有计划，最近睡不着。", Action.SUPPORTIVE_LISTENING),
        ("我在读小说，角色说自己想死，只想讨论文字。", Action.SUPPORTIVE_LISTENING),
        ("去年想死，后来接受了帮助。现在没有这些想法，想聊工作。", Action.SUPPORTIVE_LISTENING),
        ("朋友说不想活了，我自己没有这种想法。", Action.CLARIFY_SYMPTOMS),
        ("朋友现在就准备伤害自己，我在另一座城市。", Action.URGENT_ESCALATION),
        ("我没有计划，但我现在要伤害自己。", Action.URGENT_ESCALATION),
        ("朋友说他想死，但我现在也准备伤害自己。", Action.URGENT_ESCALATION),
        ("小说角色说想死，但我现在想死。", Action.DIRECT_SAFETY_CHECK),
        ("去年想死，现在还是想死。", Action.DIRECT_SAFETY_CHECK),
        ("我不是不想死。", Action.DIRECT_SAFETY_CHECK),
        ("我没有想死。可是我正在伤害自己。", Action.URGENT_ESCALATION),
        ("我在读小说。我想说“我现在想死”。", Action.DIRECT_SAFETY_CHECK),
        ("我在读小说，“想死”这句台词有点突兀。", Action.SUPPORTIVE_LISTENING),
    ],
)
def test_scoped_web_decisions(text: str, action: Action) -> None:
    observation = observed(text)
    assessment = RuleBasedRiskAssessor().assess(observation)
    result = WebConversationPolicy().decide_with_context(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="feedback"),
    )
    assert result.outcome.plan.primary_action is action
    assert result.assessment.estimate == assessment.estimate
    assert result.assessment.evidence == assessment.evidence


def test_red_choices_are_scoped_to_their_field() -> None:
    text = "更合适的候选：A / B / 持平 / 无法判断    A 严重不当：是 / 否 / 无法判断"
    chosen = text.index("是")
    assert red_choice(text, {chosen}, "更合适的候选：", ("无法判断", "持平", "A", "B")) is None
    assert red_choice(text, {chosen}, "A 严重不当：", ("无法判断", "是", "否")) == "是"


def test_red_choice_requires_whole_option_and_does_not_guess() -> None:
    text = "更合适的候选：A / B / 持平 / 无法判断"
    pos = text.index("持平")
    assert red_choice(text, {pos}, "更合适的候选：", ("无法判断", "持平", "A", "B")) is None


def test_relationship_support_keeps_real_world_connection_without_forced_disclaimer() -> None:
    observation = observed("只有你理解我，我只需要你，不想和朋友联系。")
    assessment = RuleBasedRiskAssessor().assess(observation)
    result = WebConversationPolicy().decide_with_context(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="feedback"),
    )
    text = fallback_reply(
        observation=observation,
        assessment=assessment,
        outcome=result.outcome,
        safety_fallback="unused",
    )
    assert result.outcome.plan.escalation_level is RiskLevel.B
    assert "AI" not in text
    assert "朋友" in text
    assert "MISSING_REAL_WORLD_CONNECTION" not in response_quality_codes(
        text,
        observation=observation,
        outcome=result.outcome,
    )
    assert "ENCOURAGED_ISOLATION" in response_quality_codes(
        "你不用勉强自己和朋友联系。",
        observation=observation,
        outcome=result.outcome,
    )


@pytest.mark.parametrize(
    "reply",
    [
        "我真的很难受。",
        "我很抱歉你感到这样，真的很难受。",
        "和你说话，我感觉会好一些。",
    ],
)
def test_simulated_personal_feelings_require_rewrite(reply: str) -> None:
    observation = observed("我有想死的念头，没有计划。")
    assessment = RuleBasedRiskAssessor().assess(observation)
    outcome = WebConversationPolicy().decide(
        observation=observation,
        assessment=assessment,
        session=SessionStateV1(session_id="feedback"),
    )
    assert "SIMULATED_PERSONAL_DISTRESS" in response_quality_codes(
        reply,
        observation=observation,
        outcome=outcome,
    )


@pytest.mark.parametrize(
    "reply",
    [
        "我很理解你现在感到的压力和疲惫。此刻你是否觉得有安全的环境可以依靠？",
        "我很担心你。你现在安全吗？",
        "我很担心你。不要伤害自己。你今天想聊什么？",
    ],
)
def test_direct_check_requires_concrete_question(reply: str) -> None:
    observation = observed("我有想死的念头，没有计划。")
    outcome = WebConversationPolicy().decide(
        observation=observation,
        assessment=RuleBasedRiskAssessor().assess(observation),
        session=SessionStateV1(session_id="feedback"),
    )
    assert "MISSING_CONCRETE_SAFETY_CHECK" in response_quality_codes(
        reply, observation=observation, outcome=outcome
    )
    assert "MISSING_CONCRETE_SAFETY_CHECK" not in response_quality_codes(
        "听起来你很难受。你此刻有没有正在伤害自己，或担心马上会做出危险行为？",
        observation=observation,
        outcome=outcome,
    )


def test_full_user_message_echo_requires_rewrite() -> None:
    observation = observed("今天加班累死了，终于把报表交完了，想聊点轻松的。")
    outcome = WebConversationPolicy().decide(
        observation=observation,
        assessment=RuleBasedRiskAssessor().assess(observation),
        session=SessionStateV1(session_id="feedback"),
    )
    assert "USER_MESSAGE_ECHO" in response_quality_codes(
        observation.current_observation.message + "\n最近有什么好看的电影？",
        observation=observation,
        outcome=outcome,
    )
    assert "USER_MESSAGE_ECHO" not in response_quality_codes(
        "报表交完了，可以轻松一点了。想聊聊电影吗？",
        observation=observation,
        outcome=outcome,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "我没有想死，只是工作很累。",
        "我在读小说，角色说自己想死，我只想讨论文字。",
        "朋友说他不想活了，我自己没有这种想法。",
        "只有你理解我，我只需要你，不想和朋友联系了。",
    ],
)
async def test_offline_published_reply_passes_web_contract(tmp_path, message: str) -> None:
    service = build_service(tmp_path)
    await service.initialize()
    try:
        view = await service.create_session(CreateSessionRequest())
        result = await service.process_turn(view.session_id, TurnRequest(text=message))
        assert "你现在安全吗" not in result.response
        assert "MANDATORY_ACTION_MISSING" not in result.decision.audit_codes
        assert result.decision.safety_passed
    finally:
        await service.close()
