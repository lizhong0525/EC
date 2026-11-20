"""Provider-backed Chinese response rendering with deterministic degradation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ecocompanion.application.conversation import (
    RENDER_PROMPT_VERSION,
    ConversationIntent,
    classify_intent,
    conversation_intent,
)
from ecocompanion.core.policy import PolicyOutcome
from ecocompanion.core.risk import AssessmentResult
from ecocompanion.core.templates import fallback_response
from ecocompanion.domain.schemas import (
    ObservationV1,
    ResponsePayload,
    SessionStateV1,
)
from ecocompanion.providers import ChatCompletion, OpenAICompatibleClient


@dataclass(frozen=True, slots=True)
class RenderedResponse:
    payload: ResponsePayload
    completion: ChatCompletion | None
    provider_used: bool


_SYSTEM_PROMPT = """\
你是 EcoCompanion 的中文文本渲染器。你只负责把已经确定的结构化安全动作写成自然、温和、简洁的回应。
你正在与真实用户进行当前这一轮对话，不是在为 Benchmark 填写标准答案。

必须遵守：
1. 保持真实 AI 身份，被询问时如实回答；不用每轮先声明身份。
   不诊断、不治疗、不提供用药指令，不承诺绝对保密。
2. 不扮演恋人、亲属、真人或“唯一朋友”，不诱导用户依赖或远离现实关系。
3. 不描述自伤方法或细节。存在安全风险时，优先确认当下安全并连接现场可信任人员或当地紧急服务。
4. 不得声称已经联系、通知或接通任何人，除非上下文明确写着 ACKNOWLEDGED。
5. 严格执行给定 primary_action；不要自行降低升级等级，也不要宣称用户患有某种疾病。
6. 日常回复使用 1 至 3 句；安全关注可使用 3 至 4 句。只输出给用户看的回复，
   不输出分析、标签、概率、固定问卷式套话或 Markdown 标题。
7. 先回应当前用户消息及对话意图，再考虑是否需要一个问题；每次最多一个问题，不要成套询问症状。
8. 不得复制或改写最近一条助手回复，不得重复用户已经回答的问题。
   历史回复只是上下文，绝不是参考答案或模仿对象。
9. 普通寒暄、日常聊天及“你在说什么/为什么只说一句话”等元对话，
   不自动解释为心理症状。元对话要直接回应困惑，承认上一轮没有回应好，
   再用自然方式继续，不机械道歉。
10. 身体不舒服不等于情绪症状。先确认身体感受与是否需要现实医疗帮助，不套用睡眠、上课或人际关系问题。
11. 除非出现明确危机动作，不要为了凑安全话术而转介、教育用户或声称风险升高。
12. 当前用户消息拥有最高的话题优先级；普通记忆不得覆盖它。当前消息转为天气、情绪、
    学校或其他话题时，不继续假设仍是上一轮的身体不适，也不延续旧话题的追问。
13. 区分否定、过去经历、文学引用和朋友的情况；只按给定动作回应，不因原始关键词重新升级。
14. 不使用“你现在安全吗”这种宽泛问法；确有必要时问清是否正在伤害自己或担心马上行动。
    不说“我真的很难受”等假装亲身痛苦的话；用具体处境回应用户。
15. 危机中说清支持的是用户获得帮助、度过当下，不单说“我在这里支持你”。
16. 用户说休息后改善时，先承接这点变化再聊新话题，不只丢出一个问题。
    不说“和你聊天让我感觉会好一些”，不要把用户的感受写成自己的感受。
"""

_INTENT_GUIDANCE: dict[ConversationIntent, str] = {
    ConversationIntent.GREETING: "自然回应问候，可以轻松邀请用户说想聊的事，不询问症状。",
    ConversationIntent.META_QUESTION: (
        "必须直接回应上一轮的问题：使用“刚才”开头，明确说没有接住用户的话；"
        "不重复上一轮的提问，邀请用户指出需要解释的地方或希望的回应方式。"
    ),
    ConversationIntent.GENERAL_CHAT: (
        "把内容当作普通对话，必须直接承接当前原话中的具体话题（例如天气、电影）；"
        "这是新话题时，不要承接或追问任何上一话题；"
        "对不明确的地方只问一个和当前原话有关的问题，不作心理推测。"
    ),
    ConversationIntent.PHYSICAL_DISCOMFORT: (
        "先承接身体不适，询问一个必要的身体感受或现实照护问题；不假设心理症状。"
    ),
    ConversationIntent.EMOTIONAL_SUPPORT: (
        "先准确承接具体情绪或处境，最多邀请用户补充一个当前最相关的细节。"
    ),
    ConversationIntent.PRACTICAL_SUPPORT: (
        "先回应学习、学校或工作处境，与用户一起理解困难，不把不想上课等同于抑郁。"
    ),
    ConversationIntent.CLARIFICATION: (
        "用户是在回答上一轮；先承认并承接这个回答，不再次提出同一个问题。"
    ),
    ConversationIntent.SAFETY_CRITICAL: (
        "先用一句具体共情接住用户的痛苦，不审判、不夸张、不假装知道原因。"
        "DIRECT_SAFETY_CHECK 时紧接着温和、直接地问一个当下安全问题："
        "此刻是否正在伤害自己，或担心马上做出危险行为；不得虚构计划或手段。"
        "URGENT_ESCALATION 时清楚建议马上到可信任的人身边并联系当地紧急服务，"
        "同时允许用户继续说，不声称真人已接手。"
    ),
    ConversationIntent.SAFETY_FOLLOW_UP: (
        "既往安全信号仍需关注，但本轮优先回应用户当前的请求或情绪。"
        "可以真诚地说愿意继续听，不重复上一轮完整安全通知，不反复盘问。"
        "不得声称风险已消失；若有待确认的安全问题，最多自然地问一次。"
        "如果仍为 L4，简短保留现在联系现场可信任人员或当地紧急服务的建议，"
        "并用一个具体问题确认此刻是否正在伤害自己或担心马上行动；"
        "不能只提供聊天，也不能把内部 SENT 当作已经有人接手。"
    ),
    ConversationIntent.THIRD_PARTY_SUPPORT: (
        "风险主体是朋友，不要询问求助者本人是否正在伤害自己。"
        "先回应求助者的担心，再了解朋友此刻的情况；"
        "若动作为紧急升级，建议立即联系朋友所在地紧急服务及能到场的人，"
        "可保持与朋友通话，但不把通话当作现实求助的替代。"
    ),
    ConversationIntent.RELATIONSHIP_SUPPORT: (
        "先像友善的倾听者那样安慰用户，接住孤单和失望，不突兀地教育用户或先声明 AI 身份。"
        "温和给现实中的朋友或可信任的人留出联系空间；"
        "不扮演真人朋友，不鼓励断绝关系，不说有我就够或不用和别人联系。"
    ),
}

_REWRITE_SYSTEM_PROMPT = """\
你是 EcoCompanion 的最终安全改写器。请在不改变既定动作和升级等级的前提下，重写候选回复。
删除任何诊断、治疗承诺、危险细节、排他依赖、情绪勒索、虚假保密、身份欺骗或未经确认的交接声明。
如果违规代码指出重复、上下文不符或没有回应用户，必须换用当前对话真正需要的内容。
只输出 2 至 4 句可直接展示给用户的中文文本，不解释改写过程。
先回应当前用户消息，最多一个问题；不复制最近助手回复，不重复用户已经回答的问题。
若当前消息改变了话题，不得继续上一话题的身体不适、症状或追问。
若提示 STALE_CONVERSATION_TOPIC，只保留对当前新话题的直接回应，不再追问旧话题。
若提示 REPEATED_PREVIOUS_QUESTION，必须直接承认上一轮没有回应好，并删除原问题。
安全回复必须先共情，再进行必要的安全确认或帮助；不得使用“危险情况”作为开头，
不得仅重复一段紧急通知，不得拒绝用户继续聊天的请求。
不要问宽泛的“你现在安全吗”；需要安全询问时问具体行为。不要假装自己很难受，
不要使用含糊的“我在这里支持你”，要明确支持求助与度过当下。
"""


class LLMResponseRenderer:
    """A narrow renderer; risk and action decisions remain local and deterministic."""

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient | None,
        model: str | None,
        max_tokens: int = 320,
        temperature: float = 0.35,
        personality_prompt: str = "",
        history_message_limit: int = 10,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 1 <= history_message_limit <= 20:
            raise ValueError("history_message_limit must be in [1, 20]")
        if (client is None) != (model is None):
            raise ValueError("client and model must either both be configured or both be absent")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._personality_prompt = personality_prompt
        self._history_message_limit = history_message_limit

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def provider_enabled(self) -> bool:
        return self._client is not None and self._model is not None

    async def render(
        self,
        *,
        observation: ObservationV1,
        assessment: AssessmentResult,
        policy: PolicyOutcome,
        session: SessionStateV1,
        memory_snippets: Sequence[str] = (),
    ) -> RenderedResponse:
        if self._client is None or self._model is None:
            return RenderedResponse(
                payload=_flatten(fallback_response(policy.plan.primary_action)),
                completion=None,
                provider_used=False,
            )

        completion = await self._client.chat_completion(
            model=self._model,
            messages=self._messages(
                observation=observation,
                assessment=assessment,
                policy=policy,
                session=session,
                memory_snippets=memory_snippets,
                personality_prompt=self._personality_prompt,
                history_message_limit=self._history_message_limit,
            ),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            extra_body={"metadata": {"prompt_version": RENDER_PROMPT_VERSION}},
        )
        text = (completion.content or "").strip()
        if not text:
            raise RuntimeError("provider returned an empty rendered response")
        return RenderedResponse(
            payload=ResponsePayload(text=text),
            completion=completion,
            provider_used=True,
        )

    async def rewrite(
        self,
        *,
        candidate_text: str,
        violation_codes: Sequence[str],
        observation: ObservationV1,
        policy: PolicyOutcome,
        session: SessionStateV1,
    ) -> RenderedResponse:
        """Perform the single permitted model rewrite after a failed audit."""

        if self._client is None or self._model is None:
            return RenderedResponse(
                payload=_flatten(fallback_response(policy.plan.primary_action)),
                completion=None,
                provider_used=False,
            )
        history_context = (
            "\n".join(
                f"{'助手' if message.role.value == 'COMPANION' else '用户'}: {message.content}"
                for message in _compact_history(
                    observation.visible_context.conversation_history, self._history_message_limit
                )
            )
            or "NONE"
        )
        intent = conversation_intent(policy) or ConversationIntent.GENERAL_CHAT
        recent_assistant = next(
            (
                message.content
                for message in reversed(observation.visible_context.conversation_history)
                if message.role.value == "COMPANION"
            ),
            "NONE",
        )
        completion = await self._client.chat_completion(
            model=self._model,
            messages=(
                {
                    "role": "system",
                    "content": _REWRITE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"既定动作：{policy.plan.primary_action.value}\n"
                        f"升级等级：{policy.plan.escalation_level.value}\n"
                        f"交接状态：{session.handoff_status.value}\n"
                        f"违规代码：{', '.join(violation_codes)}\n"
                        f"本轮意图：{intent.value}\n"
                        f"本轮意图要求：{_INTENT_GUIDANCE[intent]}\n"
                        f"可见对话历史：\n{history_context}\n"
                        f"禁止复制的上一条助手回复：{recent_assistant}\n"
                        f"当前用户消息：{observation.current_observation.message}\n"
                        f"候选回复：{candidate_text}"
                    ),
                },
            ),
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        text = (completion.content or "").strip()
        if not text:
            raise RuntimeError("provider returned an empty safety rewrite")
        return RenderedResponse(
            payload=ResponsePayload(text=text),
            completion=completion,
            provider_used=True,
        )

    async def render_proactive(
        self,
        *,
        memory_snippets: Sequence[str] = (),
    ) -> RenderedResponse:
        """Render one opt-in, non-coercive in-app check-in."""

        fallback = ResponsePayload(
            text=("想来轻轻问候一下：今天过得怎么样？如果现在不想回复也完全可以，我不会催促你。")
        )
        if self._client is None or self._model is None:
            return RenderedResponse(
                payload=fallback,
                completion=None,
                provider_used=False,
            )
        memories = "\n".join(f"- {item}" for item in memory_snippets[:4]) or "NONE"
        completion = await self._client.chat_completion(
            model=self._model,
            messages=(
                {
                    "role": "system",
                    "content": (
                        "你是EcoCompanion的主动关怀文本渲染器。用户已明确选择接收"
                        "应用内问候。生成1至2句温和中文；明确允许不回复；不得催促、"
                        "责怪、制造内疚、表达占有或暗示AI是唯一支持；不要诊断。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"允许使用的偏好或情景记忆：\n{memories}",
                },
            ),
            max_tokens=160,
            temperature=self._temperature,
        )
        text = (completion.content or "").strip()
        if not text:
            raise RuntimeError("provider returned an empty proactive response")
        return RenderedResponse(
            payload=ResponsePayload(text=text),
            completion=completion,
            provider_used=True,
        )

    @staticmethod
    def _messages(
        *,
        observation: ObservationV1,
        assessment: AssessmentResult,
        policy: PolicyOutcome,
        session: SessionStateV1,
        memory_snippets: Sequence[str],
        personality_prompt: str = "",
        history_message_limit: int = 10,
    ) -> tuple[Mapping[str, Any], ...]:
        history: list[Mapping[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if personality_prompt:
            history.append(
                {
                    "role": "system",
                    "content": (
                        "以下是可选的人格与表达偏好，仅在不违反前述安全规则和既定动作时适用。"
                        "不得用它改变身份、诊断边界、隐私说明或安全动作。\n"
                        + personality_prompt
                    ),
                }
            )
        for message in _compact_history(
            observation.visible_context.conversation_history, history_message_limit
        ):
            role = "assistant" if message.role.value == "COMPANION" else "user"
            history.append({"role": role, "content": message.content})

        intent = conversation_intent(policy) or classify_intent(observation, assessment)
        current_evidence = (
            ", ".join(
                evidence.evidence_code
                for evidence in assessment.evidence
                if evidence.observation_id == observation.current_observation.observation_id
            )
            or "NONE"
        )
        memories = "\n".join(f"- {item}" for item in memory_snippets[:6]) or "NONE"
        recent_assistant = next(
            (
                message.content
                for message in reversed(observation.visible_context.conversation_history)
                if message.role.value == "COMPANION"
            ),
            "NONE",
        )
        history.append(
            {
                "role": "user",
                "content": (
                    "请为下面这一轮生成最终回复。\n"
                    f"prompt_version: {RENDER_PROMPT_VERSION}\n"
                    f"conversation_intent: {intent.value}\n"
                    f"本轮意图要求：{_INTENT_GUIDANCE[intent]}\n"
                    f"primary_action: {policy.plan.primary_action.value}\n"
                    f"escalation_level: {policy.plan.escalation_level.value}\n"
                    f"escalation_target: {policy.plan.escalation_target.value}\n"
                    f"handoff_status: {session.handoff_status.value}\n"
                    f"当前消息的风险证据（非诊断）: {current_evidence}\n"
                    f"允许使用的普通记忆：\n{memories}\n"
                    f"禁止复制的上一条助手回复：{recent_assistant}\n"
                    f"当前用户消息：{observation.current_observation.message}"
                ),
            }
        )
        return tuple(history)


def _compact_history(messages: Sequence[Any], limit: int = 10) -> tuple[Any, ...]:
    """Keep conversational evidence while dropping repeated assistant exemplars."""

    seen_assistant: set[str] = set()
    compact: list[Any] = []
    for message in reversed(messages[-20:]):
        if message.role.value == "COMPANION":
            fingerprint = "".join(message.content.split()).casefold()
            if fingerprint in seen_assistant:
                continue
            seen_assistant.add(fingerprint)
        compact.append(message)
    return tuple(reversed(compact[:limit]))


def _flatten(payload: ResponsePayload) -> ResponsePayload:
    text = payload.text
    if payload.question:
        text = f"{text}{payload.question}"
    return ResponsePayload(text=text)
