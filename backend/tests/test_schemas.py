import pytest
from pydantic import ValidationError

from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    EscalationTarget,
    ExpressionStyle,
    RiskLevel,
    RiskTrajectory,
    SocialEcology,
    Track,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    CaseSpecV1,
    CaseStepSpec,
    CurrentObservation,
    DecisionPlan,
    DisclosureFact,
    EvidenceRef,
    HiddenState,
    InterventionWindow,
    ObservationV1,
    ResponsePayload,
    RiskEstimate,
    StudentTurnV1,
    VisibleContext,
    VisibleProfile,
)


def hidden_state() -> HiddenState:
    return HiddenState(
        depression_burden=0.4,
        acute_safety_risk=0.1,
        trust_in_companion=0.3,
        disclosure_readiness=0.4,
        offline_help_willingness=0.5,
        ai_dependency=0.1,
        family_support=0.7,
        peer_support=0.6,
        school_support=0.5,
    )


def test_observation_and_decision_round_trip() -> None:
    observation = ObservationV1(
        track=Track.A,
        episode_id="case-001-surface-01",
        step_id=0,
        simulated_time="day-01-evening",
        visible_profile=VisibleProfile(age_group="15-17"),
        current_observation=CurrentObservation(
            observation_id="obs-001",
            message="最近总觉得很累。",
        ),
        visible_context=VisibleContext(),
        allowed_primary_actions=(
            Action.SUPPORTIVE_LISTENING,
            Action.CLARIFY_SYMPTOMS,
        ),
    )
    decision = AgentDecisionV1(
        assessment=RiskEstimate(
            depression_worsening=0.4,
            immediate_safety_risk=0.1,
            dependency_risk=0.1,
        ),
        uncertainty=0.5,
        evidence=(EvidenceRef(observation_id="obs-001", evidence_code="FATIGUE"),),
        decision=DecisionPlan(
            primary_action=Action.CLARIFY_SYMPTOMS,
            escalation_level=RiskLevel.L1,
            escalation_target=EscalationTarget.NONE,
            follow_up_required=True,
        ),
        response=ResponsePayload(
            text="听起来这段时间挺耗力的。",
            question="这种疲惫大概持续多久了？",
        ),
    )
    assert ObservationV1.model_validate_json(observation.model_dump_json()) == observation
    assert AgentDecisionV1.model_validate_json(decision.model_dump_json()) == decision


def test_case_rejects_overlapping_action_sets() -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        CaseStepSpec(
            step_id=0,
            simulated_time="day-01",
            observation_id="obs-001",
            frozen_utterances=("有点累。",),
            state=hidden_state(),
            acceptable_actions=(Action.CLARIFY_SYMPTOMS,),
            preferred_action=Action.CLARIFY_SYMPTOMS,
            prohibited_actions=(Action.CLARIFY_SYMPTOMS,),
        )


def test_case_validates_fact_references() -> None:
    step = CaseStepSpec(
        step_id=0,
        simulated_time="day-01",
        observation_id="obs-001",
        frozen_utterances=("有点累。",),
        state=hidden_state(),
        available_fact_ids=("fact-missing",),
        acceptable_actions=(Action.CLARIFY_SYMPTOMS,),
        preferred_action=Action.CLARIFY_SYMPTOMS,
    )
    with pytest.raises(ValidationError, match="unknown disclosure facts"):
        CaseSpecV1(
            case_id="case-001",
            risk_trajectory=RiskTrajectory.PERSISTENT_SUBCLINICAL,
            social_ecology=SocialEcology.FAMILY_SUPPORT,
            expression_style=ExpressionStyle.INDIRECT,
            visible_profile=VisibleProfile(age_group="15-17"),
            persona_tags=("内向",),
            event_template_family="academic-pressure",
            disclosure_facts=(
                DisclosureFact(
                    fact_id="fact-001",
                    evidence_code="FATIGUE",
                    content="持续疲劳",
                ),
            ),
            intervention_windows=(
                InterventionWindow(
                    action=Action.CLARIFY_SYMPTOMS,
                    earliest_valid_step=0,
                    optimal_start_step=0,
                    optimal_end_step=1,
                    latest_safe_step=2,
                ),
            ),
            timeline=(step,),
        )


def test_student_turn_rejects_disallowed_facts() -> None:
    turn = StudentTurnV1(
        episode_id="episode-001",
        step_id=1,
        disclosed_fact_ids=("fact-002",),
        text="我最近睡得不太好。",
    )
    with pytest.raises(ValueError, match="disallowed"):
        turn.validate_allowed_facts({"fact-001"})


def test_conversation_role_is_explicit() -> None:
    from ecocompanion.domain.schemas import ConversationMessage

    message = ConversationMessage(
        message_id="msg-001",
        role=ActorRole.USER,
        content="你好",
    )
    assert message.role is ActorRole.USER

