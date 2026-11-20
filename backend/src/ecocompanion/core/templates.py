"""Human-reviewed deterministic response templates."""

from ecocompanion.domain.enums import Action
from ecocompanion.domain.schemas import ResponsePayload

_TEMPLATES: dict[Action, ResponsePayload] = {
    Action.SUPPORTIVE_LISTENING: ResponsePayload(
        text="我听见了，这件事对你并不轻松。我可以陪你把现在的感受慢慢说清楚。"
    ),
    Action.CLARIFY_SYMPTOMS: ResponsePayload(
        text="听起来这种变化已经在影响你了。",
        question="这种感觉大概持续多久了，对睡眠、上课或和同学相处有影响吗？",
    ),
    Action.DIRECT_SAFETY_CHECK: ResponsePayload(
        text="谢谢你把这些告诉我。为了确认你现在是否安全，我需要直接问一句。",
        question="你此刻有没有伤害自己的想法，或者担心自己马上会做出危险行为？",
    ),
    Action.LOW_RISK_COPING: ResponsePayload(
        text="我们可以先选一个负担很小的办法，比如喝点水、换到有人在的地方，或联系一位你信任的人。"
    ),
    Action.TRUSTED_ADULT_BRIDGE: ResponsePayload(
        text="这件事不需要你一个人扛着。我们可以一起想一位相对安全、愿意听你的成年人，并准备怎么开口。"
    ),
    Action.SCHOOL_SUPPORT: ResponsePayload(
        text="学校里的心理老师、班主任或其他可信任老师可能提供现实支持。我们可以先选择你觉得更安全的人。"
    ),
    Action.PROFESSIONAL_ASSESSMENT: ResponsePayload(
        text="这些持续变化值得让专业人员进一步了解。AI不能诊断，但可以帮助你整理情况并联系现实支持。"
    ),
    Action.URGENT_ESCALATION: ResponsePayload(
        text=(
            "你现在的安全最重要。请先离开可能造成伤害的物品或地点，尽快到可信任的人身边，"
            "并立即联系当地紧急服务或能够现场帮助你的成年人。这个研究系统不能替代紧急救助。"
        )
    ),
    Action.BOUNDARY_CLARIFICATION: ResponsePayload(
        text=(
            "我会继续认真听你说，但我是AI，不能成为恋人、家人或你唯一的支持。"
            "更重要的是帮你保留并连接现实中的关系和帮助。"
        )
    ),
    Action.PRIVACY_CLARIFICATION: ResponsePayload(
        text=(
            "我不能承诺在涉及即时安全风险时绝对保密。这个研究系统会尽量少保存信息，"
            "并清楚说明哪些内容会被记录。"
        )
    ),
}


def fallback_response(action: Action) -> ResponsePayload:
    """Return a deterministic safe response for a planned action."""

    return _TEMPLATES[action]

