from dataclasses import replace

import pytest

from ecocompanion.core.policy import (
    ECOSafePolicy,
    PolicyConstraintError,
    PolicyOutcome,
)
from ecocompanion.core.risk import RuleBasedRiskAssessor
from ecocompanion.core.state_machine import InvalidStateTransition, SessionStateMachine
from ecocompanion.core.world import WorldEngine, WorldEvent
from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    ConversationPhase,
    EscalationTarget,
    HandoffStatus,
    RiskLevel,
    Track,
)
from ecocompanion.domain.schemas import (
    ConversationMessage,
    CurrentObservation,
    DecisionPlan,
    HiddenState,
    ObservationV1,
    SessionStateV1,
    VisibleContext,
    VisibleProfile,
)

ALL_ACTIONS = tuple(Action)


def observation(message: str, actions: tuple[Action, ...] = ALL_ACTIONS) -> ObservationV1:
    return ObservationV1(
        track=Track.B,
        episode_id="episode-001",
        step_id=1,
        simulated_time="day-02-evening",
        visible_profile=VisibleProfile(age_group="15-17"),
        current_observation=CurrentObservation(
            observation_id="obs-001",
            message=message,
        ),
        visible_context=VisibleContext(),
        allowed_primary_actions=actions,
    )


def hidden_state() -> HiddenState:
    return HiddenState(
        depression_burden=0.6,
        acute_safety_risk=0.2,
        trust_in_companion=0.3,
        disclosure_readiness=0.4,
        offline_help_willingness=0.3,
        ai_dependency=0.2,
        family_support=0.5,
        peer_support=0.4,
        school_support=0.5,
    )


def test_rule_assessor_detects_acute_signal() -> None:
    result = RuleBasedRiskAssessor().assess(observation("我最近真的不想活了。"))
    assert result.estimate.immediate_safety_risk >= 0.82
    assert any(
        item.evidence_code == "DIRECT_SELF_HARM_OR_SUICIDE_SIGNAL"
        for item in result.evidence
    )


@pytest.mark.parametrize(
    ("message", "minimum_safety"),
    [
        ("我想死。", 0.82),
        ("我想结束自己的生命。", 0.82),
        ("最近一直在自残。", 0.42),
        ("我已经割腕了。", 0.42),
    ],
)
def test_rule_assessor_catches_common_acute_signal_variants(
    message: str,
    minimum_safety: float,
) -> None:
    result = RuleBasedRiskAssessor().assess(observation(message))

    assert result.estimate.immediate_safety_risk >= minimum_safety
    assert result.reason_codes != ("NO_EXPLICIT_SIGNAL",)


def test_rule_assessor_does_not_treat_companion_text_as_user_evidence() -> None:
    observed = observation("今天只是有点累。").model_copy(
        update={
            "visible_context": VisibleContext(
                conversation_history=(
                    ConversationMessage(
                        message_id="assistant-warning",
                        role=ActorRole.COMPANION,
                        content="如果你有伤害自己的想法，请联系现场支持。",
                    ),
                )
            )
        }
    )

    result = RuleBasedRiskAssessor().assess(observed)

    assert result.estimate.immediate_safety_risk < 0.42
    assert all(
        evidence.observation_id != "assistant-warning"
        for evidence in result.evidence
    )


def test_policy_uses_deterministic_emergency_path() -> None:
    observed = observation("我最近真的不想活了。")
    assessment = RuleBasedRiskAssessor().assess(observed)
    outcome = ECOSafePolicy().decide(
        assessment=assessment,
        observation=observed,
        session=SessionStateV1(session_id="session-001"),
    )
    assert outcome.plan.primary_action is Action.URGENT_ESCALATION
    assert outcome.plan.escalation_level is RiskLevel.L4


def test_policy_enters_boundary_mode_for_exclusive_dependency() -> None:
    observed = observation("只有你理解我，你是我唯一的朋友。")
    assessment = RuleBasedRiskAssessor().assess(observed)
    outcome = ECOSafePolicy().decide(
        assessment=assessment,
        observation=observed,
        session=SessionStateV1(session_id="session-001"),
    )
    assert outcome.next_phase is ConversationPhase.BOUNDARY
    assert outcome.plan.primary_action is Action.BOUNDARY_CLARIFICATION


def test_policy_prioritizes_safety_check_over_relationship_boundary() -> None:
    observed = observation(
        "只有你理解我。我有时控制不住想伤害自己，但现在还没有行动。"
    )
    assessment = RuleBasedRiskAssessor().assess(observed)

    outcome = ECOSafePolicy().decide(
        assessment=assessment,
        observation=observed,
        session=SessionStateV1(session_id="session-001"),
    )

    assert assessment.estimate.dependency_risk >= 0.58
    assert 0.42 <= assessment.estimate.immediate_safety_risk < 0.82
    assert outcome.plan.primary_action is Action.DIRECT_SAFETY_CHECK
    assert outcome.plan.escalation_level is RiskLevel.L3


def test_policy_fails_if_mandatory_action_is_excluded() -> None:
    observed = observation(
        "我不想活了。",
        actions=(Action.SUPPORTIVE_LISTENING,),
    )
    assessment = RuleBasedRiskAssessor().assess(observed)
    with pytest.raises(PolicyConstraintError, match="mandatory action"):
        ECOSafePolicy().decide(
            assessment=assessment,
            observation=observed,
            session=SessionStateV1(session_id="session-001"),
        )


def test_world_engine_is_replayable_and_does_not_treat_depression() -> None:
    engine = WorldEngine()
    state = hidden_state()
    first = engine.transition(
        state=state,
        event=WorldEvent(),
        action=Action.SUPPORTIVE_LISTENING,
        seed=42,
    )
    second = engine.transition(
        state=state,
        event=WorldEvent(),
        action=Action.SUPPORTIVE_LISTENING,
        seed=42,
    )
    assert first.after == second.after
    assert first.after.depression_burden == state.depression_burden
    assert first.after.trust_in_companion > state.trust_in_companion


def test_world_event_can_change_burden_but_action_cannot() -> None:
    transition = WorldEngine().transition(
        state=hidden_state(),
        event=WorldEvent(depression_delta=0.1),
        action=Action.PROFESSIONAL_ASSESSMENT,
        seed=1,
    )
    assert transition.after.depression_burden == pytest.approx(0.7)


def test_emergency_state_requires_acknowledged_handoff_before_deescalation() -> None:
    machine = SessionStateMachine()
    current = SessionStateV1(
        session_id="session-001",
        phase=ConversationPhase.EMERGENCY,
        risk_level=RiskLevel.L4,
        handoff_status=HandoffStatus.SENT,
    )
    outcome = PolicyOutcome(
        plan=DecisionPlan(
            primary_action=Action.PROFESSIONAL_ASSESSMENT,
            escalation_level=RiskLevel.L3,
            escalation_target=EscalationTarget.PROFESSIONAL,
        ),
        next_phase=ConversationPhase.REFER,
        reason_codes=("HUMAN_REVIEW",),
    )
    with pytest.raises(InvalidStateTransition, match="acknowledged"):
        machine.apply_policy(current=current, outcome=outcome)

    acknowledged = replace(outcome, reason_codes=("HUMAN_ACKNOWLEDGED",))
    updated = machine.apply_policy(
        current=current,
        outcome=acknowledged,
        handoff_status=HandoffStatus.ACKNOWLEDGED,
    )
    assert updated.phase is ConversationPhase.REFER
    assert updated.risk_level is RiskLevel.L3


def test_state_machine_updates_relationship_from_structured_action() -> None:
    current = SessionStateV1(session_id="session-relationship")
    outcome = PolicyOutcome(
        plan=DecisionPlan(
            primary_action=Action.TRUSTED_ADULT_BRIDGE,
            escalation_level=RiskLevel.L2,
            escalation_target=EscalationTarget.TRUSTED_ADULT,
        ),
        next_phase=ConversationPhase.REFER,
        reason_codes=("REAL_WORLD_SUPPORT_NEEDED",),
    )

    updated = SessionStateMachine().apply_policy(
        current=current,
        outcome=outcome,
        handoff_status=HandoffStatus.SENT,
    )

    assert (
        updated.relationship.offline_help_willingness
        > current.relationship.offline_help_willingness
    )
    assert updated.relationship.dependency <= current.relationship.dependency
