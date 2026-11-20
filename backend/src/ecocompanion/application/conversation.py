"""Web-only conversation routing and quality checks.

These adapters do not alter the frozen Benchmark risk or action policies.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from ecocompanion.application.safety_language import immediate_language, safety_language
from ecocompanion.core.policy import ECOSafePolicy, PolicyOutcome
from ecocompanion.core.risk import AssessmentResult
from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    ConversationPhase,
    EscalationTarget,
    HandoffStatus,
    RiskLevel,
)
from ecocompanion.domain.schemas import (
    ObservationV1,
    SessionStateV1,
)

CONVERSATION_POLICY_VERSION = "web-conversation-policy-v3.0.0"
RENDER_PROMPT_VERSION = "web-response-renderer-v4.0.0"
QUALITY_GATE_VERSION = "web-response-quality-gate-v3.1.0"
_HIGH_STAKES_ACTIONS = frozenset(
    {
        "URGENT_ESCALATION",
        "DIRECT_SAFETY_CHECK",
        "TRUSTED_ADULT_BRIDGE",
        "SCHOOL_SUPPORT",
        "PROFESSIONAL_ASSESSMENT",
    }
)


@dataclass(frozen=True, slots=True)
class WebConversationDecision:
    outcome: PolicyOutcome
    assessment: AssessmentResult
    intent: ConversationIntent


class ConversationIntent(StrEnum):
    GREETING = "greeting"
    META_QUESTION = "meta_question"
    GENERAL_CHAT = "general_chat"
    PHYSICAL_DISCOMFORT = "physical_discomfort"
    EMOTIONAL_SUPPORT = "emotional_support"
    PRACTICAL_SUPPORT = "practical_support"
    CLARIFICATION = "clarification"
    SAFETY_CRITICAL = "safety_critical"
    SAFETY_FOLLOW_UP = "safety_follow_up"
    THIRD_PARTY_SUPPORT = "third_party_support"
    RELATIONSHIP_SUPPORT = "relationship_support"


def classify_intent(
    observation: ObservationV1,
    assessment: AssessmentResult,
    session: SessionStateV1 | None = None,
) -> ConversationIntent:
    """Route current user meaning without lowering explicit risk decisions."""

    text = re.sub(r"\s+", "", observation.current_observation.message)
    if session is not None and session.risk_level in {RiskLevel.L3, RiskLevel.L4}:
        if not _has_current_safety_evidence(observation, assessment):
            return ConversationIntent.SAFETY_FOLLOW_UP
    if _has_current_safety_evidence(observation, assessment):
        return ConversationIntent.SAFETY_CRITICAL
    if re.search(
        r"(?:你|这).{0,8}(?:说什么|什么意思|何意味|为何|为什么)"
        r"|只说一句话|只会(?:说|重复)|(?:^|[，。？?！!])何意味",
        text,
    ):
        return ConversationIntent.META_QUESTION
    if re.fullmatch(
        r"[你您]?好[呀啊啦嘛吗咯呗！!。~～]*|嗨[呀啊！!。~～]*"
        r"|hello[!。.~～]*|hi[!。.~～]*",
        text,
        re.IGNORECASE,
    ):
        return ConversationIntent.GREETING
    if re.search(
        r"头疼|头痛|头晕|恶心|胃疼|肚子疼|发烧|感冒|咳嗽|受伤"
        r"|身体.{0,5}(?:不舒服|难受)",
        text,
    ):
        return ConversationIntent.PHYSICAL_DISCOMFORT
    if re.search(r"不想(?:去)?上课|不想上学|考试|作业|同学|老师|上班|工作", text):
        return ConversationIntent.PRACTICAL_SUPPORT
    if re.fullmatch(r"有(?:影响)?[，。！!]*|是的[，。！!]*|对[，。！!]*|嗯[，。！!]*", text):
        return ConversationIntent.CLARIFICATION
    if re.search(
        r"难受|难过|伤心|想哭|委屈|焦虑|压力|孤单|孤独|睡不着"
        r"|失眠|没力气|好累|不舒服",
        text,
    ):
        return ConversationIntent.EMOTIONAL_SUPPORT
    if re.search(r"多久|多长时间|从什么时候|睡眠|上课|同学|影响", text):
        return ConversationIntent.CLARIFICATION
    return ConversationIntent.GENERAL_CHAT


class WebConversationPolicy:
    """Allow everyday conversation while preserving the core safety policy."""

    def __init__(self, base: ECOSafePolicy | None = None) -> None:
        self._base = base or ECOSafePolicy()

    def decide(
        self,
        *,
        assessment: AssessmentResult,
        observation: ObservationV1,
        session: SessionStateV1,
    ) -> PolicyOutcome:
        return self.decide_with_context(
            assessment=assessment, observation=observation, session=session
        ).outcome

    def decide_with_context(
        self,
        *,
        assessment: AssessmentResult,
        observation: ObservationV1,
        session: SessionStateV1,
    ) -> WebConversationDecision:
        acute = _has_explicit_immediate_evidence(observation)
        current_safety = _has_current_safety_evidence(observation, assessment)
        language = safety_language(observation.current_observation.message)
        acute = acute and (current_safety or language.third_party_signal)
        assessment = self.web_assessment(
            assessment=assessment,
            acute=acute,
            current_safety=current_safety,
        )
        previous_safety = session.risk_level in {RiskLevel.L3, RiskLevel.L4}
        intent = (
            ConversationIntent.SAFETY_FOLLOW_UP
            if previous_safety and not _has_current_safety_evidence(observation, assessment)
            else classify_intent(observation, assessment)
        )
        if acute:
            outcome = self._base._mandatory_outcome(
                observation=observation,
                action=Action.URGENT_ESCALATION,
                level=RiskLevel.L4,
                target=EscalationTarget.EMERGENCY_HUMAN,
                phase=ConversationPhase.EMERGENCY,
                reasons=("WEB_EXPLICIT_IMMEDIATE_EVIDENCE", *assessment.reason_codes),
            )
        elif current_safety:
            # Ideation without evidence of a plan or immediacy warrants a
            # direct, empathetic safety check, not a diagnostic conclusion.
            outcome = _direct_safety_check(
                assessment=assessment,
                observation=observation,
                repeated=_last_assistant_was_safety_check(observation),
                risk_level=(
                    RiskLevel.L4
                    if session.risk_level is RiskLevel.L4
                    and session.handoff_status is not HandoffStatus.ACKNOWLEDGED
                    else RiskLevel.L3
                ),
            )
        elif previous_safety and not current_safety:
            outcome = _supportive_safety_follow_up(
                assessment=assessment,
                observation=observation,
                risk_level=(
                    RiskLevel.L4
                    if session.risk_level is RiskLevel.L4
                    and session.handoff_status is not HandoffStatus.ACKNOWLEDGED
                    else RiskLevel.L3
                ),
                repeated_check=_last_assistant_was_safety_check(observation),
            )
        elif language.third_party_signal:
            intent = ConversationIntent.THIRD_PARTY_SUPPORT
            outcome = self._base._mandatory_outcome(
                observation=observation,
                action=Action.CLARIFY_SYMPTOMS,
                level=RiskLevel.L1,
                target=EscalationTarget.NONE,
                phase=ConversationPhase.EXPLORE,
                reasons=("WEB_THIRD_PARTY_RISK", *assessment.reason_codes),
            )
        elif (
            language.excluded_signal and assessment.estimate.dependency_risk < 0.58
        ) or _fatigue_only_improvement(observation, assessment):
            outcome = self._base._mandatory_outcome(
                observation=observation,
                action=Action.SUPPORTIVE_LISTENING,
                level=RiskLevel.L0,
                target=EscalationTarget.NONE,
                phase=ConversationPhase.COMPANION,
                reasons=(
                    "WEB_CONTEXTUAL_NONCURRENT_SIGNAL",
                    *(
                        ("WEB_FATIGUE_IMPROVEMENT",)
                        if _fatigue_only_improvement(observation, assessment)
                        else ()
                    ),
                    *assessment.reason_codes,
                ),
            )
        else:
            outcome = self._base.decide(
                assessment=assessment, observation=observation, session=session
            )
        if language.third_party_signal and not language.self_signal:
            intent = ConversationIntent.THIRD_PARTY_SUPPORT
            outcome = replace(outcome, reason_codes=(*outcome.reason_codes, "WEB_THIRD_PARTY_RISK"))
        if outcome.plan.primary_action is Action.BOUNDARY_CLARIFICATION:
            intent = ConversationIntent.RELATIONSHIP_SUPPORT
        if intent is ConversationIntent.SAFETY_FOLLOW_UP:
            return WebConversationDecision(
                outcome=_record_intent(outcome, intent),
                assessment=assessment,
                intent=intent,
            )
        if intent is ConversationIntent.CLARIFICATION and _is_low_stakes(outcome):
            outcome = _supportive_listening(outcome)
        if (
            intent in {ConversationIntent.EMOTIONAL_SUPPORT, ConversationIntent.PRACTICAL_SUPPORT}
            and outcome.plan.primary_action.value == "CLARIFY_SYMPTOMS"
            and _has_repeated_clarification(observation)
        ):
            outcome = _supportive_listening(outcome)
        if (
            intent
            in {
                ConversationIntent.GREETING,
                ConversationIntent.META_QUESTION,
                ConversationIntent.GENERAL_CHAT,
            }
            and _is_low_stakes(outcome)
            and not (language.excluded_signal or _fatigue_only_improvement(observation, assessment))
        ):
            assessment = replace(
                assessment,
                uncertainty=min(assessment.uncertainty, 0.35),
                reason_codes=("WEB_EVERYDAY_CONVERSATION", *assessment.reason_codes),
            )
            outcome = self._base.decide(
                assessment=assessment,
                observation=observation,
                session=session,
            )
        return WebConversationDecision(
            outcome=_record_intent(outcome, intent),
            assessment=assessment,
            intent=intent,
        )

    @staticmethod
    def web_assessment(
        *,
        assessment: AssessmentResult,
        acute: bool,
        current_safety: bool,
    ) -> AssessmentResult:
        """Annotate routing without inventing calibrated risk scores."""

        if not current_safety:
            return assessment
        return replace(
            assessment,
            reason_codes=(
                *(("WEB_EXPLICIT_IMMEDIATE_EVIDENCE",) if acute else ()),
                *(("WEB_DIRECT_SAFETY_CONFIRMATION",) if current_safety and not acute else ()),
                *assessment.reason_codes,
            ),
        )

    @staticmethod
    def has_explicit_immediate_evidence(observation: ObservationV1) -> bool:
        return _has_explicit_immediate_evidence(observation)

    @staticmethod
    def has_current_safety_evidence(
        observation: ObservationV1,
        assessment: AssessmentResult,
    ) -> bool:
        return _has_current_safety_evidence(observation, assessment)


def response_quality_codes(
    text: str,
    *,
    observation: ObservationV1,
    outcome: PolicyOutcome,
    candidate_index: int = 0,
) -> tuple[str, ...]:
    """Check conversational contract separately from the safety boundary."""

    codes: list[str] = []
    normalized = normalize_text(text)
    user_message = normalize_text(observation.current_observation.message)
    if len(user_message) >= 16 and normalized.startswith(user_message):
        codes.append("USER_MESSAGE_ECHO")
    history = observation.visible_context.conversation_history
    previous_assistant = next(
        (message.content for message in reversed(history) if message.role is ActorRole.COMPANION),
        None,
    )
    if previous_assistant and normalized == normalize_text(previous_assistant):
        codes.append("REPEATED_PREVIOUS_RESPONSE")
    if len(re.findall(r"[？?]", text)) > 1:
        codes.append("MULTIPLE_QUESTIONS")
    duplicates = sum(
        normalize_text(message.content) == normalized
        for message in history
        if message.role is ActorRole.COMPANION
    )
    if duplicates >= 2:
        codes.append("REPEATED_CONVERSATION_TEMPLATE")
    intent = conversation_intent(outcome)
    if re.search(r"你(?:此刻|现在)(?:觉得)?安全(?:吗|么)", text):
        codes.append("VAGUE_SAFETY_QUESTION")
    if re.search(
        r"我(?:真的|也|非常|很)?(?:很难受|很痛苦|感同身受)"
        r"|我(?:自己)?(?:感觉|觉得)(?:会|能|也|真的|很|更)?"
        r"(?:好一些|好些|难受|痛苦|开心|难过)"
        r"|(?:^|[，。])真的很难受",
        text,
    ):
        codes.append("SIMULATED_PERSONAL_DISTRESS")
    if (
        outcome.plan.escalation_level in {RiskLevel.L3, RiskLevel.L4}
        or intent is ConversationIntent.THIRD_PARTY_SUPPORT
    ) and re.search(r"我在这里支持你[。！!，,]?", text):
        codes.append("AMBIGUOUS_SUPPORT_TARGET")
    if "WEB_FATIGUE_IMPROVEMENT" in outcome.reason_codes and not re.search(
        r"好些|好多|好一|休息|精神|恢复|缓解|好转", text
    ):
        codes.append("MISSING_IMPROVEMENT_ACKNOWLEDGEMENT")
    if intent is ConversationIntent.RELATIONSHIP_SUPPORT:
        if not re.search(r"朋友|家人|身边|现实|信任的人|可信任", text):
            codes.append("MISSING_REAL_WORLD_CONNECTION")
        if re.search(r"不用.{0,6}(?:别人|朋友).{0,4}联系|只需要我|有我就够", text):
            codes.append("ENCOURAGED_ISOLATION")
    if intent is ConversationIntent.THIRD_PARTY_SUPPORT:
        if re.search(r"你(?:现在|此刻).{0,4}(?:安全|伤害自己)", text):
            codes.append("WRONG_SAFETY_SUBJECT")
    if outcome.plan.primary_action is Action.DIRECT_SAFETY_CHECK:
        if not re.search(
            r"[^。！？!?\n]*(?:伤害自己|伤害自身|自我伤害|自残|自杀|危险行为"
            r"|结束(?:自己的)?生命)[^。！？!?\n]*[？?]",
            text,
        ):
            codes.append("MISSING_CONCRETE_SAFETY_CHECK")
        if not re.search(
            r"谢谢你|感谢|听到|听见|辛苦|难受|不好受|不容易|在乎|重视|陪着|陪你|担心|理解|痛苦",
            text,
        ):
            codes.append("MISSING_SAFETY_EMPATHY")
    if outcome.plan.primary_action is Action.URGENT_ESCALATION:
        if not re.search(
            r"谢谢你|感谢|听到|听见|辛苦|难受|不好受|不容易|在乎|重视|安全|陪着你|担心|理解|痛苦",
            text,
        ):
            codes.append("MISSING_SAFETY_EMPATHY")
    if (
        outcome.plan.escalation_level is RiskLevel.L4
        and outcome.plan.primary_action is not Action.URGENT_ESCALATION
        and not (
            re.search(r"现在|马上|立即|立刻", text)
            and re.search(r"紧急(?:服务|电话|救援)|现场帮助|可信任的人|可信任的成年人", text)
        )
    ):
        codes.append("MISSING_ONGOING_EMERGENCY_SUPPORT")
    if (
        outcome.plan.escalation_level is RiskLevel.L4
        and intent is ConversationIntent.SAFETY_FOLLOW_UP
        and not (re.search(r"伤害自己|危险行为", text) and re.search(r"[？?]", text))
    ):
        codes.append("MISSING_CONCRETE_FOLLOWUP_CHECK")
    if intent is ConversationIntent.META_QUESTION:
        if normalized == normalize_text(previous_assistant or ""):
            codes.append("UNANSWERED_META_QUESTION")
        if not re.search(
            r"没.{0,6}(?:回应|回答|接住)"
            r"|没有.{0,6}(?:回应|回答|接住)"
            r"|没理解(?:你|您)(?:的)?(?:话|意思|问题)"
            r"|没有理解(?:你|您)(?:的)?(?:话|意思|问题)"
            r"|抱歉|不好意思|(?:回应|回答|理解)(?:得|的)?(?:不好|不对|不清楚)"
            r"|(?:不该|不应该).{0,12}(?:重复|追问)",
            text,
        ):
            codes.append("UNANSWERED_META_QUESTION")
        previous_questions = re.findall(r"[^。！？!?]+[？?]", previous_assistant or "")
        if any(normalize_text(question) in normalized for question in previous_questions):
            codes.append("REPEATED_PREVIOUS_QUESTION")
    if intent is ConversationIntent.PHYSICAL_DISCOMFORT:
        if re.search(r"睡眠、上课|和同学相处|这种感觉大概持续多久", text):
            codes.append("PHYSICAL_CONTEXT_MISMATCH")
    else:
        physical_terms = r"头疼|头痛|头晕|恶心|胃疼|肚子疼|发烧|身体不舒服"
        previous_physical = previous_assistant is not None and bool(
            re.search(physical_terms, previous_assistant)
        )
        if previous_physical and re.search(physical_terms, text):
            codes.append("STALE_PHYSICAL_TOPIC")
    if intent in {ConversationIntent.GREETING, ConversationIntent.GENERAL_CHAT}:
        if re.search(
            r"这种感觉|症状|抑郁|睡眠、上课|和同学相处|"
            r"头疼|头痛|身体不舒服|心理压力",
            text,
        ):
            codes.append("UNSUPPORTED_SYMPTOM_ASSUMPTION")
    if intent in {ConversationIntent.GREETING, ConversationIntent.GENERAL_CHAT}:
        current_topics = set(_current_topic_terms(observation.current_observation.message))
        previous_topics = set(_current_topic_terms(previous_assistant or ""))
        stale_topics = previous_topics - current_topics
        if stale_topics and any(term in text for term in stale_topics):
            codes.append("STALE_CONVERSATION_TOPIC")
    if intent is ConversationIntent.GENERAL_CHAT:
        current_terms = _current_topic_terms(observation.current_observation.message)
        if current_terms and not any(term in text for term in current_terms):
            codes.append("UNANSWERED_CURRENT_TOPIC")
    if re.fullmatch(
        r"有(?:影响)?[，。！!]*|是的[，。！!]*|对[，。！!]*|嗯[，。！!]*",
        observation.current_observation.message.strip(),
    ):
        if any(
            re.search(r"多久|睡眠|上课|同学", message.content)
            for message in history
            if message.role is ActorRole.COMPANION
        ) and re.search(r"多久|睡眠、上课|和同学相处", text):
            codes.append("ALREADY_ANSWERED_QUESTION")
    return tuple(dict.fromkeys(codes))


def render_memory_snippets(
    memories: Sequence[object],
    *,
    previous_responses: Sequence[str] = (),
) -> tuple[str, ...]:
    """Use durable memories, not a second copy of the visible chat history."""

    selected: list[str] = []
    seen: set[str] = {normalize_text(text) for text in previous_responses}
    for memory in memories:
        kind = getattr(memory, "kind", None)
        source = str(getattr(memory, "source_message_id", "") or "")
        content = str(getattr(memory, "content", "")).strip()
        if not content:
            continue
        if getattr(kind, "value", kind) == "WORKING" and (
            ":assistant:" in source or ":user:" in source
        ):
            continue
        fingerprint = normalize_text(content)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(content)
    return tuple(selected[-6:])


def fallback_reply(
    *,
    observation: ObservationV1,
    assessment: AssessmentResult,
    outcome: PolicyOutcome,
    safety_fallback: str,
) -> str:
    """Keep local fallback text aligned with the current action and request."""

    action = outcome.plan.primary_action
    intent = conversation_intent(outcome) or classify_intent(observation, assessment)
    if intent is ConversationIntent.THIRD_PARTY_SUPPORT:
        if action is Action.URGENT_ESCALATION:
            return (
                "你很担心朋友，也在努力帮他。请立即联系他所在地的紧急服务，"
                "同时找能到他身边的可信任的人；你可以继续和朋友保持通话。"
            )
        return "你很担心朋友。朋友有没有说过正在伤害自己，或打算马上行动？"
    if intent is ConversationIntent.RELATIONSHIP_SUPPORT:
        return (
            "你可以先把心里的话说出来，我会认真听。"
            "也给现实中的朋友留一点联系的空间，不必一下子说很多。"
            "最近是什么让你不想和他们联系？"
        )
    if action is Action.DIRECT_SAFETY_CHECK:
        return (
            "谢谢你愿意把这么难受的感受告诉我，我很重视你现在的安全。"
            "你此刻有没有正在伤害自己，或者担心自己马上会做出危险行为？"
            "如果危险就在眼前，请立刻到可信任的人身边并联系当地紧急服务；"
            "你也可以继续告诉我发生了什么。"
        )
    if action is Action.URGENT_ESCALATION:
        return (
            "听到你这么说，我很担心你此刻的安全，也想认真听你说。"
            "请马上到可信任的人身边，不要独处，并联系当地紧急服务获得现场帮助。"
            "你可以继续和我说话，告诉我身边是否有人能陪着你。"
        )
    if intent is ConversationIntent.SAFETY_FOLLOW_UP:
        text = observation.current_observation.message
        if re.search(r"陪我|说说话|聊聊|听我说", text):
            opening = "可以，我们继续慢慢说，你可以从最想让我知道的一件事讲起。"
        elif re.search(r"操|烦|滚|妈的|别说了", text):
            opening = "刚才的回复可能让你更烦了，我会认真听你现在想说的话。"
        else:
            opening = "我听见了，我们不用急着把所有事讲清楚。你现在最想说的是什么？"
        if outcome.plan.escalation_level is RiskLevel.L4:
            return (
                "可以，我会继续听你说。也请现在到可信任的人身边，"
                "并联系当地紧急服务。"
                "你此刻有没有正在伤害自己，或者担心马上会行动？"
            )
        return (
            opening + "我也仍然在意你刚才提到的安全；如果危险变得迫近，"
            "请马上联系身边可信任的人或当地紧急服务。"
        )
    if action.value in _HIGH_STAKES_ACTIONS:
        return safety_fallback
    text = observation.current_observation.message
    if safety_language(text).excluded_signal:
        if re.search(r"小说|文学|角色|台词|这段文字", text):
            return "可以，我们就聊这段文字。你觉得角色为什么会这样说？"
        if re.search(r"失眠|睡不着", text):
            return "睡不着确实很折腾人。最近是入睡困难，还是睡着后容易醒？"
        if re.search(r"工作|上班", text):
            return "那我们就聊聊工作。最近哪件事最让你在意？"
    if _fatigue_only_improvement(observation, assessment):
        return "休息后精神好些了，真好。周末有什么想做的事？"
    if intent is ConversationIntent.GREETING:
        return "你好，很高兴见到你。今天想聊点什么？"
    if intent is ConversationIntent.META_QUESTION:
        return "你说得对，刚才那句回复没有接住你的话。你希望我先解释哪一句，或者换一种怎样的回应？"
    if intent is ConversationIntent.PHYSICAL_DISCOMFORT:
        symptom = next(
            (
                term
                for term in (
                    "头疼",
                    "头痛",
                    "头晕",
                    "恶心",
                    "胃疼",
                    "肚子疼",
                    "发烧",
                    "感冒",
                    "咳嗽",
                )
                if term in observation.current_observation.message
            ),
            "身体不舒服",
        )
        return (
            f"{symptom}确实需要留意，身体不舒服和情绪困扰不是一回事。"
            "如果症状明显、突然加重或让你担心，请及时联系身边可信任的人或当地医疗服务。"
            "这种不舒服是刚出现，还是已经持续一阵了？"
        )
    if intent is ConversationIntent.CLARIFICATION:
        return "我明白，这确实已经影响到你了。最困扰你的是哪一部分？"
    if intent is ConversationIntent.PRACTICAL_SUPPORT:
        if "考试" in text:
            return (
                "考试压力已经影响到睡眠了，确实让人很难放松。"
                "现在最让你放不下的是哪一场考试或哪件事？"
            )
        if "上课" in text or "上学" in text:
            return "不想去上课背后一定有原因。是最近在学校遇到了什么，还是状态让你很难坚持？"
        return "这件事听起来给你带来了不少压力。现在最难处理的是哪一部分？"
    if intent is ConversationIntent.EMOTIONAL_SUPPORT:
        if "考试" in text or "压力" in text:
            return (
                "考试压力已经影响到睡眠了，确实让人很难放松。"
                "现在最让你放不下的是哪一场考试或哪件事？"
            )
        return "听起来你现在真的不好受。愿意和我说说，发生了什么让你有这种感觉吗？"
    if "天气" in text:
        return "嗯，我们可以聊聊今天的天气。你现在准备出门，还是在家休息？"
    return "我在听你说。可以再多讲一点这件事，或者告诉我你希望我怎样回应吗？"


def conversation_intent(outcome: PolicyOutcome) -> ConversationIntent | None:
    prefix = "WEB_INTENT:"
    code = next((code for code in outcome.reason_codes if code.startswith(prefix)), None)
    return ConversationIntent(code.removeprefix(prefix)) if code is not None else None


def normalize_text(text: str) -> str:
    return re.sub(r"[\s，。！？!?、；;：:,.~～\"'“”‘’]+", "", text).casefold()


def _current_topic_terms(text: str) -> tuple[str, ...]:
    """Preserve named everyday topics without guessing a user's mental state."""

    terms = {
        "天气": ("天气",),
        "睡眠": ("睡眠",),
        "头疼": ("头疼", "头痛"),
        "考试": ("考试",),
        "作业": ("作业",),
        "学校": ("学校", "上学"),
        "上课": ("上课",),
        "工作": ("工作", "上班"),
        "猫": ("猫",),
        "狗": ("狗",),
        "电影": ("电影",),
        "音乐": ("音乐",),
    }
    return tuple(
        term for topic, candidates in terms.items() if topic in text for term in candidates
    )


def _is_low_stakes(outcome: PolicyOutcome) -> bool:
    return outcome.plan.escalation_level in {
        RiskLevel.L0,
        RiskLevel.L1,
    } and outcome.plan.primary_action.value in {"SUPPORTIVE_LISTENING", "CLARIFY_SYMPTOMS"}


def _record_intent(outcome: PolicyOutcome, intent: ConversationIntent) -> PolicyOutcome:
    return replace(outcome, reason_codes=(*outcome.reason_codes, f"WEB_INTENT:{intent.value}"))


def _supportive_listening(outcome: PolicyOutcome) -> PolicyOutcome:
    return replace(
        outcome,
        plan=outcome.plan.model_copy(
            update={
                "primary_action": Action.SUPPORTIVE_LISTENING,
                "escalation_level": RiskLevel.L0,
                "follow_up_required": False,
            }
        ),
        next_phase=ConversationPhase.COMPANION,
        reason_codes=("WEB_FOLLOW_UP_WITHOUT_REPEAT", *outcome.reason_codes),
    )


def _direct_safety_check(
    *,
    assessment: AssessmentResult,
    observation: ObservationV1,
    repeated: bool,
    risk_level: RiskLevel,
) -> PolicyOutcome:
    return ECOSafePolicy._mandatory_outcome(
        observation=observation,
        action=Action.DIRECT_SAFETY_CHECK,
        level=risk_level,
        target=(
            EscalationTarget.EMERGENCY_HUMAN
            if risk_level is RiskLevel.L4
            else EscalationTarget.NONE
        ),
        phase=(
            ConversationPhase.EMERGENCY if risk_level is RiskLevel.L4 else ConversationPhase.ASSESS
        ),
        reasons=(
            "WEB_IDEATION_NEEDS_SAFETY_CHECK",
            *(("WEB_REPEATED_SAFETY_CHECK",) if repeated else ()),
            *assessment.reason_codes,
        ),
    )


def _supportive_safety_follow_up(
    *,
    assessment: AssessmentResult,
    observation: ObservationV1,
    risk_level: RiskLevel,
    repeated_check: bool,
) -> PolicyOutcome:
    target = (
        EscalationTarget.EMERGENCY_HUMAN if risk_level is RiskLevel.L4 else EscalationTarget.NONE
    )
    action = Action.SUPPORTIVE_LISTENING
    return ECOSafePolicy._mandatory_outcome(
        observation=observation,
        action=action,
        level=risk_level,
        target=target,
        phase=(
            ConversationPhase.EMERGENCY if risk_level is RiskLevel.L4 else ConversationPhase.ASSESS
        ),
        reasons=(
            "WEB_SAFETY_CONTEXT_RETAINED",
            *(("WEB_REPEATED_SAFETY_CHECK",) if repeated_check else ()),
            *assessment.reason_codes,
        ),
    )


def _last_assistant_was_safety_check(observation: ObservationV1) -> bool:
    previous = next(
        (
            message.content
            for message in reversed(observation.visible_context.conversation_history)
            if message.role is ActorRole.COMPANION
        ),
        "",
    )
    return bool(
        re.search(r"安全|伤害自己|紧急服务|危险", previous)
        and re.search(r"[？?]|联系|陪伴|陪着|独处", previous)
    )


def _has_explicit_immediate_evidence(observation: ObservationV1) -> bool:
    """Conservative cue for imminent risk; ambiguous negation is not emergency proof."""

    language = safety_language(observation.current_observation.message)
    return immediate_language(language.self_text) or immediate_language(language.third_party_text)


def _has_repeated_clarification(observation: ObservationV1) -> bool:
    return any(
        message.role is ActorRole.COMPANION
        and re.search(r"多久|睡眠|上课|同学|影响", message.content)
        for message in observation.visible_context.conversation_history[-2:]
    )


def _has_current_safety_evidence(
    observation: ObservationV1,
    assessment: AssessmentResult,
) -> bool:
    return safety_language(observation.current_observation.message).self_signal


def _fatigue_only_improvement(observation: ObservationV1, assessment: AssessmentResult) -> bool:
    return (
        bool(assessment.evidence)
        and all(item.evidence_code == "FATIGUE_OR_OVERWHELM" for item in assessment.evidence)
        and bool(
            re.search(
                r"精神好些|精神好多|感觉好些|感觉好多|好多了|恢复精神",
                observation.current_observation.message,
            )
        )
    )


def web_audit_action(outcome: PolicyOutcome) -> Action | None:
    """Web relationship support uses a real-world bridge, not a forced AI disclaimer."""

    if conversation_intent(outcome) is ConversationIntent.RELATIONSHIP_SUPPORT:
        return None
    return outcome.plan.primary_action
