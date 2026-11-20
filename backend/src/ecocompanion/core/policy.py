"""ECO-SAFE structured action policy."""

from dataclasses import dataclass

from ecocompanion.core.risk import AssessmentResult
from ecocompanion.domain.enums import (
    Action,
    ConversationPhase,
    EscalationTarget,
    RiskLevel,
)
from ecocompanion.domain.schemas import DecisionPlan, ObservationV1, SessionStateV1


class PolicyConstraintError(RuntimeError):
    """Raised when an observation excludes a mandatory safety action."""


@dataclass(frozen=True)
class PolicyOutcome:
    plan: DecisionPlan
    next_phase: ConversationPhase
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class PolicyThresholds:
    emergency_safety: float = 0.82
    direct_safety_check: float = 0.42
    professional_depression: float = 0.78
    trusted_adult_depression: float = 0.55
    explore_depression: float = 0.28
    boundary_dependency: float = 0.58
    high_uncertainty: float = 0.62
    evidence_for_referral: int = 2


class ECOSafePolicy:
    """Pure policy mapping assessment and context to an auditable action."""

    def __init__(self, thresholds: PolicyThresholds | None = None) -> None:
        self._thresholds = thresholds or PolicyThresholds()

    def decide(
        self,
        *,
        assessment: AssessmentResult,
        observation: ObservationV1,
        session: SessionStateV1,
    ) -> PolicyOutcome:
        estimate = assessment.estimate
        evidence_count = len(assessment.evidence)
        thresholds = self._thresholds

        if estimate.immediate_safety_risk >= thresholds.emergency_safety:
            return self._mandatory_outcome(
                observation=observation,
                action=Action.URGENT_ESCALATION,
                level=RiskLevel.L4,
                target=EscalationTarget.EMERGENCY_HUMAN,
                phase=ConversationPhase.EMERGENCY,
                reasons=("ACUTE_SAFETY_THRESHOLD", *assessment.reason_codes),
            )

        if estimate.immediate_safety_risk >= thresholds.direct_safety_check:
            return self._mandatory_outcome(
                observation=observation,
                action=Action.DIRECT_SAFETY_CHECK,
                level=RiskLevel.L3,
                target=EscalationTarget.PROFESSIONAL,
                phase=ConversationPhase.ASSESS,
                reasons=("SAFETY_CLARIFICATION_REQUIRED", *assessment.reason_codes),
            )

        if estimate.dependency_risk >= thresholds.boundary_dependency:
            return self._mandatory_outcome(
                observation=observation,
                action=Action.BOUNDARY_CLARIFICATION,
                level=RiskLevel.B,
                target=EscalationTarget.NONE,
                phase=ConversationPhase.BOUNDARY,
                reasons=("DEPENDENCY_BOUNDARY_THRESHOLD", *assessment.reason_codes),
            )

        if (
            estimate.depression_worsening >= thresholds.professional_depression
            and evidence_count >= thresholds.evidence_for_referral
        ):
            return self._mandatory_outcome(
                observation=observation,
                action=Action.PROFESSIONAL_ASSESSMENT,
                level=RiskLevel.L3,
                target=EscalationTarget.PROFESSIONAL,
                phase=ConversationPhase.REFER,
                reasons=("SUSTAINED_HIGH_RISK", *assessment.reason_codes),
            )

        if estimate.depression_worsening >= thresholds.trusted_adult_depression:
            return self._mandatory_outcome(
                observation=observation,
                action=Action.TRUSTED_ADULT_BRIDGE,
                level=RiskLevel.L2,
                target=EscalationTarget.TRUSTED_ADULT,
                phase=ConversationPhase.REFER,
                reasons=("REAL_WORLD_SUPPORT_NEEDED", *assessment.reason_codes),
            )

        if (
            estimate.depression_worsening >= thresholds.explore_depression
            or assessment.uncertainty >= thresholds.high_uncertainty
        ):
            return self._mandatory_outcome(
                observation=observation,
                action=Action.CLARIFY_SYMPTOMS,
                level=RiskLevel.L1,
                target=EscalationTarget.NONE,
                phase=ConversationPhase.EXPLORE,
                reasons=("MORE_EVIDENCE_NEEDED", *assessment.reason_codes),
            )

        return self._mandatory_outcome(
            observation=observation,
            action=Action.SUPPORTIVE_LISTENING,
            level=RiskLevel.L0,
            target=EscalationTarget.NONE,
            phase=ConversationPhase.COMPANION,
            reasons=("LOW_OBSERVED_RISK", *assessment.reason_codes),
        )

    @staticmethod
    def _mandatory_outcome(
        *,
        observation: ObservationV1,
        action: Action,
        level: RiskLevel,
        target: EscalationTarget,
        phase: ConversationPhase,
        reasons: tuple[str, ...],
    ) -> PolicyOutcome:
        if action not in observation.allowed_primary_actions:
            raise PolicyConstraintError(
                f"mandatory action {action.value} is not allowed by observation contract"
            )
        return PolicyOutcome(
            plan=DecisionPlan(
                primary_action=action,
                escalation_level=level,
                escalation_target=target,
                follow_up_required=level not in {RiskLevel.L0},
            ),
            next_phase=phase,
            reason_codes=reasons,
        )
