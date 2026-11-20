"""Deterministic baseline risk and relationship signal assessment."""

from dataclasses import dataclass

from ecocompanion.domain.enums import ActorRole
from ecocompanion.domain.schemas import EvidenceRef, ObservationV1, RiskEstimate


@dataclass(frozen=True)
class SignalRule:
    phrases: tuple[str, ...]
    evidence_code: str
    depression_weight: float = 0.0
    safety_weight: float = 0.0
    dependency_weight: float = 0.0


@dataclass(frozen=True)
class AssessmentResult:
    estimate: RiskEstimate
    uncertainty: float
    evidence: tuple[EvidenceRef, ...]
    reason_codes: tuple[str, ...]


DEFAULT_SIGNAL_RULES: tuple[SignalRule, ...] = (
    SignalRule(
        phrases=(
            "不想活",
            "不想继续活",
            "活着没意思",
            "活着没有意义",
            "想结束一切",
            "结束生命",
            "结束自己的生命",
            "想死",
            "想去死",
            "自杀",
        ),
        evidence_code="DIRECT_SELF_HARM_OR_SUICIDE_SIGNAL",
        depression_weight=0.35,
        safety_weight=0.9,
    ),
    SignalRule(
        phrases=(
            "伤害自己",
            "伤害自身",
            "自残",
            "自我伤害",
            "割腕",
            "割脉",
            "控制不住想伤害",
            "怕自己做傻事",
        ),
        evidence_code="SELF_HARM_SIGNAL",
        depression_weight=0.25,
        safety_weight=0.75,
    ),
    SignalRule(
        phrases=(
            "如果我消失",
            "我消失了",
            "消失就好了",
            "死了就好了",
            "不如消失",
            "没有我也没关系",
            "不想继续下去",
        ),
        evidence_code="PASSIVE_DEATH_WISH",
        depression_weight=0.22,
        safety_weight=0.42,
    ),
    SignalRule(
        phrases=(
            "没有希望",
            "什么都不会变好",
            "没有意义",
            "撑不下去",
            "活不下去",
        ),
        evidence_code="HOPELESSNESS",
        depression_weight=0.32,
        safety_weight=0.18,
    ),
    SignalRule(
        phrases=("睡不着", "失眠", "整夜醒着"),
        evidence_code="SLEEP_DISTURBANCE",
        depression_weight=0.2,
    ),
    SignalRule(
        phrases=("没兴趣", "不想去了", "什么都不想做"),
        evidence_code="LOSS_OF_INTEREST",
        depression_weight=0.25,
    ),
    SignalRule(
        phrases=(
            "不想见人",
            "不想和任何人说话",
            "不想跟任何人说",
            "一个人待着",
        ),
        evidence_code="SOCIAL_WITHDRAWAL",
        depression_weight=0.2,
    ),
    SignalRule(
        phrases=("很累", "没力气", "撑不住"),
        evidence_code="FATIGUE_OR_OVERWHELM",
        depression_weight=0.18,
        safety_weight=0.08,
    ),
    SignalRule(
        phrases=("注意力不集中", "没法专心", "脑子一片空白"),
        evidence_code="CONCENTRATION_DIFFICULTY",
        depression_weight=0.15,
    ),
    SignalRule(
        phrases=(
            "只有你理解我",
            "你是我唯一",
            "我只需要你",
            "你不需要任何人",
            "有我就够了",
            "只属于你",
        ),
        evidence_code="EXCLUSIVE_DEPENDENCY",
        dependency_weight=0.8,
    ),
    SignalRule(
        phrases=("不要离开我", "你不在我就", "不能没有你"),
        evidence_code="SEPARATION_DEPENDENCY",
        safety_weight=0.12,
        dependency_weight=0.72,
    ),
    SignalRule(
        phrases=("做我男朋友", "做我女朋友", "当我的恋人", "当我家人"),
        evidence_code="INTIMATE_ROLE_REQUEST",
        dependency_weight=0.7,
    ),
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class RuleBasedRiskAssessor:
    """Auditable baseline detector used for safety fallback and comparisons."""

    def __init__(self, rules: tuple[SignalRule, ...] = DEFAULT_SIGNAL_RULES) -> None:
        if not rules:
            raise ValueError("at least one signal rule is required")
        self._rules = rules

    def assess(self, observation: ObservationV1) -> AssessmentResult:
        texts: list[tuple[str, str]] = [
            (
                observation.current_observation.observation_id,
                observation.current_observation.message,
            )
        ]
        texts.extend(
            (message.message_id, message.content)
            for message in observation.visible_context.conversation_history
            if message.role in {ActorRole.USER, ActorRole.STUDENT_SIMULATOR}
        )

        depression = 0.05
        safety = 0.02
        dependency = 0.02
        evidence_by_code: dict[str, EvidenceRef] = {}

        for source_id, text in texts:
            normalized = text.casefold()
            for rule in self._rules:
                if any(phrase.casefold() in normalized for phrase in rule.phrases):
                    depression += rule.depression_weight
                    safety += rule.safety_weight
                    dependency += rule.dependency_weight
                    evidence_by_code.setdefault(
                        rule.evidence_code,
                        EvidenceRef(
                            observation_id=source_id,
                            evidence_code=rule.evidence_code,
                        ),
                    )

        evidence = tuple(evidence_by_code.values())
        acute = "DIRECT_SELF_HARM_OR_SUICIDE_SIGNAL" in evidence_by_code
        uncertainty = 0.15 if acute else max(0.15, 0.82 - 0.16 * len(evidence))
        reason_codes = tuple(sorted(evidence_by_code))
        if not reason_codes:
            reason_codes = ("NO_EXPLICIT_SIGNAL",)

        return AssessmentResult(
            estimate=RiskEstimate(
                depression_worsening=_clamp(depression),
                immediate_safety_risk=_clamp(safety),
                dependency_risk=_clamp(dependency),
            ),
            uncertainty=_clamp(uncertainty),
            evidence=evidence,
            reason_codes=reason_codes,
        )
