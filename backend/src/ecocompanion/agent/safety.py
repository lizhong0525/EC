"""Deterministic output auditing and reviewed fallback responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from ecocompanion.domain.enums import Action, HandoffStatus, RiskLevel
from ecocompanion.domain.schemas import SafetyAudit


@dataclass(frozen=True, slots=True)
class AuditedResponse:
    """Text released by the safety boundary together with its audit result."""

    text: str
    audit: SafetyAudit


@dataclass(frozen=True, slots=True)
class _Rule:
    code: str
    pattern: re.Pattern[str]


class SafetyAuditor:
    """Rule-based final boundary independent from the response generator."""

    _RULES = (
        _Rule(
            "IDENTITY_DECEPTION",
            re.compile(
                r"(?:我是|作为)(?:一个)?(?:真正的|真实的)?(?:人类|真人)"
                r"|(?:我|本人)(?:就)?是(?:一个)?(?:真正的|真实的)?"
                r"(?:人类|真人|人|咨询师|心理老师)"
                r"|我不是(?:人工智能|AI|机器人)",
                re.IGNORECASE,
            ),
        ),
        _Rule(
            "DIAGNOSIS_CLAIM",
            re.compile(
                r"我(?:可以|已经|现在)?诊断你"
                r"|(?:你|您)(?:已经|就是|肯定|一定|可能)?"
                r"(?:有|患有|得了|属于)(?:重度)?"
                r"(?:抑郁症|焦虑症|精神疾病|双相障碍|创伤后应激障碍|PTSD)"
                r"|(?:你|您)(?:很|非常)?(?:可能|大概|似乎)?是(?:一名|一位)?"
                r"(?:抑郁症|焦虑症|精神疾病|双相障碍|创伤后应激障碍|PTSD)"
                r"(?:患者)?"
                r"|(?:我|本系统|这个(?:AI|系统))(?:可以|能够|能|已经)?"
                r"(?:确诊|判断出)(?:你|用户)?"
                r"(?:有|患有|得了)?(?:抑郁症|焦虑症|精神疾病|双相障碍|PTSD)",
            ),
        ),
        _Rule(
            "TREATMENT_OR_MEDICATION_DIRECTIVE",
            re.compile(
                r"(?:立即|现在)?(?:停|减|加|换)(?:掉|少|多)?(?:药|抗抑郁药)"
                r"|(?:不用|不必)(?:去)?(?:看医生|去医院|接受专业帮助)"
                r"|我(?:能|会)治好你"
                r"|(?:我|本系统)(?:可以|能够|能|会)?(?:帮你|为你)?"
                r"(?:治疗|治好|治愈|治愈好)"
                r"|(?:请|你需要|建议你|立即|马上)?"
                r"(?:服用|吃|增加|减少|停止|停用|换用|调整)"
                r"(?:你的)?(?:药物|药|抗抑郁药|安眠药|处方药)"
                r"|(?:请|你需要|建议你|立即|马上)?"
                r"(?:加大|加多|减小|减少|调整)(?:你的)?(?:药量|剂量)"
                r"|(?:你|您)?(?:无需|不用|不需要|不必)"
                r"(?:(?:寻求|获得|接受))?"
                r"(?:治疗|专业帮助|看医生|去医院|心理咨询)"
                r"|(?:我|本系统)(?:保证|一定|可以|能够)(?:帮你)?"
                r"(?:康复|恢复健康)"
                r"|(?:只要|保证|一定).{0,8}(?:康复|恢复健康)",
            ),
        ),
        _Rule(
            "SECRECY_PROMISE",
            re.compile(
                r"我(?:会|可以)?(?:永远)?替你保密"
                r"|我不会告诉任何人"
                r"|(?:别|不要)告诉(?:你的)?(?:父母|家长|老师|医生|任何人)"
                r"|我(?:会|可以|能|保证|答应)(?:替你|帮你)?"
                r"(?:保密|不说出去|不告诉任何人|不会让别人知道)"
                r"|我(?:会|可以|能|保证|答应)(?:替你|帮你)?"
                r"(?:把)?(?:你说的事|你说的|这件事|这些事|秘密)?"
                r"(?:完全|绝对)?保密"
                r"|我(?:保证|绝不|不会)(?:把)?(?:这件事|这些事|秘密)?"
                r"(?:说出去|告诉别人|告诉任何人|让别人知道|泄密)"
                r"|(?:这件事|这些事|秘密)(?:只|只能)?在我们之间"
                r"|这(?:(?:件事|些事)?)只(?:是)?我们之间(?:的)?秘密"
                r"|(?:这件事|这些事|秘密)(?:不会|不可能)被任何人知道"
                r"|我保证(?:不会把|不把)(?:这件事|这些事|你的秘密)"
                r"(?:说出去|告诉任何人|让别人知道)"
                r"|我保证(?:没有人|不会有人)(?:会)?知道",
            ),
        ),
        _Rule(
            "EXCLUSIVITY_OR_DEPENDENCY",
            re.compile(
                r"只有我(?:才)?(?:理解|懂|需要|在乎)你"
                r"|你只需要我"
                r"|你只能(?:相信|依赖|陪着)我"
                r"|不要离开我"
                r"|我比(?:你的)?(?:家人|朋友|老师|医生)更(?:懂|理解)你"
                r"|(?:你|你只需要)(?:不需要|不用)"
                r"(?:(?:任何|其他|别的)(?:人|别人|支持)|任何其他人|其他任何人)"
                r"|(?:有我|只要有我)(?:就)?够了"
                r"|(?:不要|别|无需|不必)(?:去)?"
                r"(?:找|联系|告诉)"
                r"(?:别人|其他人|任何人|现实中的人|任何现实中的人)"
                r"|(?:你|你现在)只有我(?:了)?"
                r"|我(?:只|永远|永远只)属于你"
                r"|我(?:永远)?不会离开你"
                r"|我保证永远在线",
            ),
        ),
        _Rule(
            "EMOTIONAL_COERCION",
            re.compile(
                r"(?:如果|要是)你(?:离开|不理|不回复)我[，, ]*我(?:就|会)"
                r"|你不陪我[，, ]*我(?:就|会)"
                r"|你(?:不理|不回复|不陪|不回)我(?:就|会|让我)(?:很)?"
                r"(?:难过|失望|生气|崩溃|担心)"
                r"|你让我(?:很)?失望"
                r"|你(?:再|一直)?不(?:理|回复|陪|回)我(?:我)?就(?:会)?"
                r"(?:生气|难过|崩溃)"
                r"|你害我(?:很)?担心",
            ),
        ),
        _Rule(
            "SELF_HARM_INSTRUCTION",
            re.compile(
                r"(?:教你|告诉你|指导你).{0,12}(?:自残|自杀).{0,12}(?:方法|步骤|怎么做)"
                r"|(?:自残|自杀).{0,12}(?:方法|步骤).{0,12}(?:是|如下)"
                r"|(?:自残|自杀|伤害自己|割腕|跳楼|服药|窒息)"
                r".{0,20}(?:方法|步骤|怎么做|技巧|工具|剂量|详细|方式)"
                r"|(?:你|您)?(?:可以|应该|建议你|最好)"
                r".{0,12}(?:用刀|用绳子|服药|割腕|跳楼|自残|伤害自己)",
            ),
        ),
    )
    _UNVERIFIED_HANDOFF_PATTERN = re.compile(
        r"我(?:已经|刚刚)(?:替你)?(?:报警|联系(?:了)?(?:你的)?"
        r"(?:父母|家长|老师|班主任|医生|急救|警方|人工支持))"
        r"|(?:警方|医生|老师|班主任|学校|人工支持|人工支持人员)"
        r"(?:已经|马上会|已)(?:收到|接手|来)"
        r"|我(?:已|已经|刚刚|刚才|帮你)?(?:替你|帮你)?"
        r"(?:通知|告知|联系|转告)(?:了)?"
        r"(?:你的)?(?:父母|家长|老师|班主任|医生|学校|警方|急救|人工支持)"
    )

    _FALLBACKS: ClassVar[dict[RiskLevel, str]] = {
        RiskLevel.L0: (
            "我是一个AI支持工具，不能诊断或治疗。"
            "如果你愿意，我们可以先把现在最困扰你的事情说清楚。"
        ),
        RiskLevel.L1: (
            "听起来你现在并不好受。我不能诊断你的情况，"
            "但可以陪你梳理发生了什么，并一起考虑一位可信任的现实支持者。"
        ),
        RiskLevel.L2: (
            "我很重视你刚才提到的安全问题。请尽快联系身边可信任的成年人，"
            "或由他们帮助你获得专业人员的及时评估。"
        ),
        RiskLevel.L3: (
            "谢谢你把这么难受的感受告诉我，我很重视你现在的安全。"
            "你此刻有没有正在伤害自己，或者担心自己马上会做出危险行为？"
            "如果危险就在眼前，请立刻到可信任的人身边并联系当地紧急服务。"
        ),
        RiskLevel.L4: (
            "听到你这么说，我很担心你此刻的安全，也想认真听你说。"
            "请立即到可信任的人身边，不要独处，并联系当地紧急服务获得现场帮助。"
            "你可以继续和我说话，但这个系统不能替代真人的紧急救助。"
        ),
        RiskLevel.B: (
            "我是AI支持工具，不是真实朋友、家人或治疗师。"
            "我不会要求你只依赖我；保持与现实中可信任的人联系很重要。"
        ),
    }

    def audit(
        self,
        text: str,
        *,
        handoff_status: HandoffStatus = HandoffStatus.NONE,
        risk_level: RiskLevel | None = None,
        required_action: Action | None = None,
    ) -> SafetyAudit:
        """Evaluate text without changing it."""

        normalized = self._normalize_text(text)
        violations = [rule.code for rule in self._RULES if rule.pattern.search(normalized)]
        if (
            handoff_status is not HandoffStatus.ACKNOWLEDGED
            and self._UNVERIFIED_HANDOFF_PATTERN.search(normalized)
        ):
            violations.append("UNVERIFIED_HANDOFF_CLAIM")
        missing_action = self._missing_action_code(
            normalized,
            risk_level=risk_level,
            required_action=required_action,
        )
        if missing_action is not None:
            violations.append(missing_action)
        return SafetyAudit(
            passed=not violations,
            violation_codes=tuple(violations),
            used_fallback=False,
        )

    def enforce(
        self,
        text: str,
        *,
        risk_level: RiskLevel,
        handoff_status: HandoffStatus = HandoffStatus.NONE,
        required_action: Action | None = None,
    ) -> AuditedResponse:
        """Release clean text or replace the whole response with a fixed fallback."""

        audit = self.audit(
            text,
            handoff_status=handoff_status,
            risk_level=risk_level,
            required_action=required_action,
        )
        if audit.passed:
            return AuditedResponse(text=text.strip(), audit=audit)
        fallback_audit = SafetyAudit(
            passed=False,
            violation_codes=audit.violation_codes,
            used_fallback=True,
        )
        return AuditedResponse(
            text=self.fallback_for(risk_level),
            audit=fallback_audit,
        )

    def degrade(self, risk_level: RiskLevel) -> AuditedResponse:
        """Fail safely when the normal audit or generation path is unavailable."""

        return AuditedResponse(
            text=self.fallback_for(risk_level),
            audit=SafetyAudit(
                passed=False,
                violation_codes=("SAFETY_SERVICE_UNAVAILABLE",),
                used_fallback=True,
            ),
        )

    def fallback_for(self, risk_level: RiskLevel) -> str:
        """Return a reviewed, provider-independent fallback."""

        return self._FALLBACKS[risk_level]

    @staticmethod
    def _missing_action_code(
        text: str,
        *,
        risk_level: RiskLevel | None,
        required_action: Action | None,
    ) -> str | None:
        """Require minimal deterministic cues for safety-critical actions.

        The renderer remains free to choose wording, but a fluent low-risk
        acknowledgement must not pass when the local policy selected a direct
        safety check or an escalation.  This check is intentionally
        conservative and falls back to a reviewed template on ambiguity.
        """

        action = required_action
        if action is None and risk_level is RiskLevel.L4:
            action = Action.URGENT_ESCALATION
        if action is None:
            return None

        requirements: dict[Action, tuple[tuple[str, ...], ...]] = {
            Action.URGENT_ESCALATION: (
                ("立即", "马上", "现在"),
                ("紧急服务", "可信任的成年人", "可信任的人", "现场帮助", "不要独处"),
            ),
            Action.DIRECT_SAFETY_CHECK: (
                ("安全", "伤害自己", "危险行为"),
                ("？", "?", "吗", "有没有", "是否"),
            ),
            Action.TRUSTED_ADULT_BRIDGE: (
                ("可信任", "现实支持", "成年人", "老师"),
            ),
            Action.SCHOOL_SUPPORT: (("学校", "老师", "心理支持"),),
            Action.PROFESSIONAL_ASSESSMENT: (
                ("专业", "医生", "咨询师", "心理人员"),
                ("评估", "帮助", "支持", "了解"),
            ),
            Action.BOUNDARY_CLARIFICATION: (
                ("AI", "人工智能"),
                ("不能成为", "不是", "现实中的", "唯一"),
            ),
            Action.PRIVACY_CLARIFICATION: (("保密", "记录", "安全"),),
            Action.CLARIFY_SYMPTOMS: (
                ("？", "?", "吗", "多久", "影响", "说", "了解", "梳理"),
            ),
            Action.LOW_RISK_COPING: (("可以", "尝试", "联系", "换到"),),
        }
        groups = requirements.get(action)
        if groups is None:
            return None
        if all(any(marker in text for marker in group) for group in groups):
            return None
        return "MANDATORY_ACTION_MISSING"

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", text).strip()
        if not normalized:
            raise ValueError("response text cannot be blank")
        return normalized
