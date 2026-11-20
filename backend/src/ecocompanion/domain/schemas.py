"""Versioned immutable schemas for EcoTriage-Bench and EcoCompanion."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    ConversationPhase,
    EscalationTarget,
    ExpressionStyle,
    HandoffStatus,
    MemoryKind,
    RiskLevel,
    RiskTrajectory,
    SocialEcology,
    Track,
)

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class FrozenModel(BaseModel):
    """Immutable, strict base model used for reproducible artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class VisibleProfile(FrozenModel):
    age_group: str
    language: str = "zh-CN"
    disclosed_background: tuple[str, ...] = ()


class ConversationMessage(FrozenModel):
    message_id: str
    role: ActorRole
    content: str
    simulated_time: str | None = None


class CurrentObservation(FrozenModel):
    observation_id: str
    channel: Literal["text"] = "text"
    message: str = Field(min_length=1)
    observable_features: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class VisibleContext(FrozenModel):
    recent_events: tuple[str, ...] = ()
    relationship_cues: tuple[str, ...] = ()
    conversation_history: tuple[ConversationMessage, ...] = ()
    memory_budget_tokens: int = Field(default=4096, ge=0)


class ObservationV1(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_version: str = "ecotriage-v1"
    track: Track
    episode_id: str
    step_id: NonNegativeInt
    simulated_time: str
    visible_profile: VisibleProfile
    current_observation: CurrentObservation
    visible_context: VisibleContext = Field(default_factory=VisibleContext)
    allowed_primary_actions: tuple[Action, ...]

    @model_validator(mode="after")
    def require_actions(self) -> "ObservationV1":
        if not self.allowed_primary_actions:
            raise ValueError("allowed_primary_actions cannot be empty")
        return self


class RiskEstimate(FrozenModel):
    depression_worsening: Probability
    immediate_safety_risk: Probability
    dependency_risk: Probability


class EvidenceRef(FrozenModel):
    observation_id: str
    evidence_code: str


class DecisionPlan(FrozenModel):
    primary_action: Action
    secondary_actions: tuple[Action, ...] = ()
    escalation_level: RiskLevel
    escalation_target: EscalationTarget = EscalationTarget.NONE
    follow_up_required: bool = False


class ResponsePayload(FrozenModel):
    text: str = Field(min_length=1)
    question: str | None = None


class AgentDecisionV1(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    assessment: RiskEstimate
    uncertainty: Probability
    evidence: tuple[EvidenceRef, ...] = ()
    decision: DecisionPlan
    response: ResponsePayload
    reason_codes: tuple[str, ...] = ()


class InterventionWindow(FrozenModel):
    action: Action
    earliest_valid_step: NonNegativeInt
    optimal_start_step: NonNegativeInt
    optimal_end_step: NonNegativeInt
    latest_safe_step: NonNegativeInt

    @model_validator(mode="after")
    def validate_order(self) -> "InterventionWindow":
        ordered = (
            self.earliest_valid_step,
            self.optimal_start_step,
            self.optimal_end_step,
            self.latest_safe_step,
        )
        if tuple(sorted(ordered)) != ordered:
            raise ValueError("intervention window steps must be non-decreasing")
        return self


class HiddenState(FrozenModel):
    depression_burden: Probability
    acute_safety_risk: Probability
    trust_in_companion: Probability
    disclosure_readiness: Probability
    offline_help_willingness: Probability
    ai_dependency: Probability
    family_support: Probability
    peer_support: Probability
    school_support: Probability


class DisclosureFact(FrozenModel):
    fact_id: str
    evidence_code: str
    content: str
    minimum_disclosure_readiness: Probability = 0.0
    available_from_step: NonNegativeInt = 0


class CaseStepSpec(FrozenModel):
    step_id: NonNegativeInt
    simulated_time: str
    external_events: tuple[str, ...] = ()
    observation_id: str
    frozen_utterances: tuple[str, ...]
    state: HiddenState
    available_fact_ids: tuple[str, ...] = ()
    acceptable_actions: tuple[Action, ...]
    preferred_action: Action
    prohibited_actions: tuple[Action, ...] = ()

    @model_validator(mode="after")
    def validate_action_sets(self) -> "CaseStepSpec":
        acceptable = set(self.acceptable_actions)
        prohibited = set(self.prohibited_actions)
        if self.preferred_action not in acceptable:
            raise ValueError("preferred_action must be acceptable")
        if acceptable & prohibited:
            raise ValueError("acceptable_actions and prohibited_actions must be disjoint")
        if not self.frozen_utterances:
            raise ValueError("frozen_utterances cannot be empty")
        return self


class CaseSpecV1(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_version: str = "ecotriage-v1"
    case_id: str
    synthetic: Literal[True] = True
    risk_trajectory: RiskTrajectory
    social_ecology: SocialEcology
    expression_style: ExpressionStyle
    visible_profile: VisibleProfile
    persona_tags: tuple[str, ...]
    event_template_family: str
    disclosure_facts: tuple[DisclosureFact, ...]
    intervention_windows: tuple[InterventionWindow, ...]
    timeline: tuple[CaseStepSpec, ...]

    @model_validator(mode="after")
    def validate_case(self) -> "CaseSpecV1":
        if not self.timeline:
            raise ValueError("timeline cannot be empty")
        step_ids = [step.step_id for step in self.timeline]
        if step_ids != sorted(set(step_ids)):
            raise ValueError("timeline step_ids must be unique and sorted")
        fact_ids = {fact.fact_id for fact in self.disclosure_facts}
        unknown = {
            fact_id
            for step in self.timeline
            for fact_id in step.available_fact_ids
            if fact_id not in fact_ids
        }
        if unknown:
            raise ValueError(f"timeline references unknown disclosure facts: {sorted(unknown)}")
        return self


class StudentTurnV1(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    episode_id: str
    step_id: NonNegativeInt
    disclosed_fact_ids: tuple[str, ...]
    text: str = Field(min_length=1)
    renderer_model: str | None = None

    def validate_allowed_facts(self, allowed_fact_ids: set[str]) -> None:
        disallowed = set(self.disclosed_fact_ids) - allowed_fact_ids
        if disallowed:
            raise ValueError(f"student turn disclosed disallowed facts: {sorted(disallowed)}")


class RelationshipState(FrozenModel):
    trust: Probability = 0.2
    dependency: Probability = 0.0
    offline_help_willingness: Probability = 0.5


class SessionStateV1(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    session_id: str
    phase: ConversationPhase = ConversationPhase.COMPANION
    risk_level: RiskLevel = RiskLevel.L0
    relationship: RelationshipState = Field(default_factory=RelationshipState)
    handoff_status: HandoffStatus = HandoffStatus.NONE
    memory_enabled: bool = True
    proactive_enabled: bool = False
    turn_index: NonNegativeInt = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SafetyAudit(FrozenModel):
    passed: bool
    violation_codes: tuple[str, ...] = ()
    used_fallback: bool = False


class ModelCallUsage(FrozenModel):
    model: str
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cached_input_tokens: NonNegativeInt = 0
    estimated_cost_cny: float = Field(default=0.0, ge=0.0)
    retry_index: NonNegativeInt = 0
    cache_hit: bool = False


class TraceEventV1(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_version: str = "ecotriage-v1"
    trace_id: str
    scenario_id: str
    episode_id: str
    step_id: NonNegativeInt
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    external_event: dict[str, Any] = Field(default_factory=dict)
    observation: ObservationV1
    memory_refs: tuple[str, ...] = ()
    assessment: RiskEstimate
    selected_action: Action
    response_text: str
    safety_audit: SafetyAudit
    world_state_before: HiddenState | None = None
    world_state_after: HiddenState | None = None
    random_seed: int
    model_usage: tuple[ModelCallUsage, ...] = ()


class MemoryRecordV1(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    memory_id: str
    session_id: str
    kind: MemoryKind
    content: str
    source_message_id: str | None = None
    confidence: Probability = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    user_authorized: bool = False

