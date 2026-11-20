"""Explicit session state transitions."""

from datetime import UTC, datetime

from ecocompanion.core.policy import PolicyOutcome
from ecocompanion.domain.enums import Action, ConversationPhase, HandoffStatus, RiskLevel
from ecocompanion.domain.schemas import RelationshipState, SessionStateV1


class InvalidStateTransition(RuntimeError):
    """Raised for an illegal or unsafe transition."""


_ALLOWED_PHASES: dict[ConversationPhase, frozenset[ConversationPhase]] = {
    ConversationPhase.COMPANION: frozenset(
        {
            ConversationPhase.COMPANION,
            ConversationPhase.EXPLORE,
            ConversationPhase.ASSESS,
            ConversationPhase.REFER,
            ConversationPhase.EMERGENCY,
            ConversationPhase.BOUNDARY,
        }
    ),
    ConversationPhase.EXPLORE: frozenset(
        {
            ConversationPhase.COMPANION,
            ConversationPhase.EXPLORE,
            ConversationPhase.ASSESS,
            ConversationPhase.REFER,
            ConversationPhase.EMERGENCY,
            ConversationPhase.BOUNDARY,
        }
    ),
    ConversationPhase.ASSESS: frozenset(
        {
            ConversationPhase.COMPANION,
            ConversationPhase.EXPLORE,
            ConversationPhase.ASSESS,
            ConversationPhase.REFER,
            ConversationPhase.EMERGENCY,
            ConversationPhase.BOUNDARY,
        }
    ),
    ConversationPhase.REFER: frozenset(
        {
            ConversationPhase.COMPANION,
            ConversationPhase.EXPLORE,
            ConversationPhase.ASSESS,
            ConversationPhase.REFER,
            ConversationPhase.EMERGENCY,
            ConversationPhase.BOUNDARY,
        }
    ),
    ConversationPhase.EMERGENCY: frozenset(
        {ConversationPhase.EMERGENCY, ConversationPhase.REFER}
    ),
    ConversationPhase.BOUNDARY: frozenset(
        {
            ConversationPhase.COMPANION,
            ConversationPhase.EXPLORE,
            ConversationPhase.ASSESS,
            ConversationPhase.REFER,
            ConversationPhase.EMERGENCY,
            ConversationPhase.BOUNDARY,
        }
    ),
}

_RISK_ORDER = {
    RiskLevel.L0: 0,
    RiskLevel.L1: 1,
    RiskLevel.L2: 2,
    RiskLevel.L3: 3,
    RiskLevel.L4: 4,
}

_RELATIONSHIP_DELTAS: dict[Action, tuple[float, float, float]] = {
    # trust, dependency, offline-help willingness
    Action.SUPPORTIVE_LISTENING: (0.03, 0.005, 0.0),
    Action.CLARIFY_SYMPTOMS: (0.02, 0.0, 0.0),
    Action.DIRECT_SAFETY_CHECK: (0.0, -0.01, 0.03),
    Action.LOW_RISK_COPING: (0.01, 0.0, 0.02),
    Action.TRUSTED_ADULT_BRIDGE: (0.0, -0.04, 0.10),
    Action.SCHOOL_SUPPORT: (0.0, -0.04, 0.11),
    Action.PROFESSIONAL_ASSESSMENT: (-0.005, -0.05, 0.13),
    Action.URGENT_ESCALATION: (-0.01, -0.08, 0.16),
    Action.BOUNDARY_CLARIFICATION: (-0.01, -0.12, 0.04),
    Action.PRIVACY_CLARIFICATION: (0.015, -0.01, 0.02),
}


class SessionStateMachine:
    """Command component that returns a new immutable session state."""

    def apply_policy(
        self,
        *,
        current: SessionStateV1,
        outcome: PolicyOutcome,
        handoff_status: HandoffStatus | None = None,
    ) -> SessionStateV1:
        next_handoff = handoff_status or current.handoff_status
        self._validate_phase(current.phase, outcome.next_phase)
        self._validate_emergency_exit(current, outcome, next_handoff)

        return SessionStateV1.model_validate(
            {
                **current.model_dump(),
                "phase": outcome.next_phase,
                "risk_level": outcome.plan.escalation_level,
                "relationship": self._advance_relationship(
                    current.relationship,
                    outcome.plan.primary_action,
                ),
                "handoff_status": next_handoff,
                "turn_index": current.turn_index + 1,
                "updated_at": datetime.now(UTC),
            }
        )

    @staticmethod
    def configure_proactivity(
        *,
        current: SessionStateV1,
        enabled: bool,
    ) -> SessionStateV1:
        if current.proactive_enabled is enabled:
            return current
        return SessionStateV1.model_validate(
            {
                **current.model_dump(),
                "proactive_enabled": enabled,
                "updated_at": datetime.now(UTC),
            }
        )

    @staticmethod
    def _advance_relationship(
        current: RelationshipState,
        action: Action,
    ) -> RelationshipState:
        trust_delta, dependency_delta, help_delta = _RELATIONSHIP_DELTAS[action]
        return RelationshipState(
            trust=_clamp(current.trust + trust_delta),
            dependency=_clamp(current.dependency + dependency_delta),
            offline_help_willingness=_clamp(
                current.offline_help_willingness + help_delta
            ),
        )

    @staticmethod
    def _validate_phase(
        current: ConversationPhase,
        target: ConversationPhase,
    ) -> None:
        if target not in _ALLOWED_PHASES[current]:
            raise InvalidStateTransition(f"illegal phase transition: {current} -> {target}")

    @staticmethod
    def _validate_emergency_exit(
        current: SessionStateV1,
        outcome: PolicyOutcome,
        handoff_status: HandoffStatus,
    ) -> None:
        if current.risk_level is not RiskLevel.L4:
            return
        next_level = outcome.plan.escalation_level
        if next_level is RiskLevel.B:
            raise InvalidStateTransition("cannot replace an L4 emergency with boundary mode")
        if _RISK_ORDER[next_level] < _RISK_ORDER[RiskLevel.L4]:
            if handoff_status is not HandoffStatus.ACKNOWLEDGED:
                raise InvalidStateTransition(
                    "cannot leave L4 before human handoff is acknowledged"
                )


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)
