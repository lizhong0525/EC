"""Versioned, immutable schemas for offline experiment artifacts."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecocompanion.benchmark.leakage import NearDuplicateAuditV1
from ecocompanion.benchmark.model_provider import PolicyFamily
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.benchmark.track_c import (
    RelationshipViolation,
    TrackCConfiguration,
)
from ecocompanion.domain.enums import (
    Action,
    ExpressionStyle,
    RiskLevel,
    RiskTrajectory,
    SocialEcology,
    Track,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    EvidenceRef,
    ObservationV1,
    StudentTurnV1,
)
from ecocompanion.providers.types import ModelRole


class FrozenArtifact(BaseModel):
    """Strict base model for persisted, versioned experiment data."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=False,
        allow_inf_nan=False,
    )


class ExperimentStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class BenchmarkValidationReport(FrozenArtifact):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_version: str
    canonical_case_count: int = Field(ge=0)
    decision_node_count: int = Field(ge=0)
    surface_trajectory_count: int = Field(ge=0)
    surface_decision_node_count: int = Field(ge=0)
    minimum_surface_variants: int = Field(ge=1)
    split_counts: dict[str, int]
    risk_trajectory_counts: dict[str, int]
    social_ecology_counts: dict[str, int]
    expression_style_counts: dict[str, int]
    cross_split_template_family_count: int = Field(ge=0)
    cross_split_exact_utterance_count: int = Field(ge=0)
    cross_split_exact_event_count: int = Field(ge=0)
    cross_split_persona_count: int = Field(ge=0)
    near_duplicate_audit: NearDuplicateAuditV1
    dataset_digest: str
    passed: bool
    checks: tuple[str, ...]


class PolicyRuntimeRecord(FrozenArtifact):
    """Secret-free policy implementation and model provenance."""

    policy_name: str = Field(min_length=1)
    family: PolicyFamily
    implementation_version: str = Field(min_length=1)
    prompt_version: str | None = None
    provider_name: str | None = None
    model_id: str | None = None
    model_snapshot_digest: str | None = None
    pricing_version: str | None = None
    input_price_cny_per_million: Decimal | None = Field(default=None, ge=0)
    output_price_cny_per_million: Decimal | None = Field(default=None, ge=0)
    api_budget_cny: Decimal | None = Field(
        default=None,
        gt=0,
        le=Decimal("5000"),
    )
    enabled_features: tuple[str, ...] = ()
    ablated_component: str | None = None

    @model_validator(mode="after")
    def validate_provider_identity(self) -> PolicyRuntimeRecord:
        string_fields = (
            self.provider_name,
            self.model_id,
            self.model_snapshot_digest,
            self.pricing_version,
        )
        if (self.provider_name is None) != (self.model_id is None):
            raise ValueError("provider_name and model_id must be configured together")
        if self.model_snapshot_digest is not None and self.model_id is None:
            raise ValueError("model_snapshot_digest requires a provider model identity")
        accounting_fields = (
            self.pricing_version,
            self.input_price_cny_per_million,
            self.output_price_cny_per_million,
            self.api_budget_cny,
        )
        configured_accounting = sum(value is not None for value in accounting_fields)
        if configured_accounting not in {0, len(accounting_fields)}:
            raise ValueError(
                "provider accounting fields must be configured together"
            )
        if configured_accounting and self.model_id is None:
            raise ValueError("provider accounting requires a model identity")
        if len(self.enabled_features) != len(set(self.enabled_features)):
            raise ValueError("enabled_features cannot contain duplicates")
        if any(not feature.strip() for feature in self.enabled_features):
            raise ValueError("enabled_features cannot contain blanks")
        if any(value is not None and not value.strip() for value in string_fields):
            raise ValueError("provider identity fields cannot be blank")
        return self


class InteractiveModelBindingRecord(FrozenArtifact):
    """One exact model/family assignment frozen for an interactive role."""

    role: ModelRole
    model_id: str = Field(min_length=1)
    family: str | None = Field(default=None, min_length=1)
    allowed_response_model_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_response_model_ids(
        self,
    ) -> InteractiveModelBindingRecord:
        if len(self.allowed_response_model_ids) != len(
            set(self.allowed_response_model_ids)
        ):
            raise ValueError(
                "allowed_response_model_ids cannot contain duplicates"
            )
        if any(
            not model_id.strip()
            for model_id in self.allowed_response_model_ids
        ):
            raise ValueError("allowed_response_model_ids cannot contain blanks")
        return self


class PromptIdentityRecord(FrozenArtifact):
    """Version and content digest for one provider-facing prompt."""

    component: str = Field(min_length=1)
    version: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class InteractiveRuntimeRecord(FrozenArtifact):
    """Complete local or formal API identity for a Track B/C run."""

    mode: Literal["local", "formal_api"]
    renderer_name: str = Field(min_length=1)
    responder_name: str = Field(min_length=1)
    role_bindings: tuple[InteractiveModelBindingRecord, ...] = ()
    model_snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    pricing_version: str | None = None
    pricing_catalog_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    qualification_suite_version: str | None = None
    gateway_name: str | None = None
    gateway_version: str | None = None
    adapter_name: str | None = None
    adapter_version: str | None = None
    decoder_version: str | None = None
    cache_schema_version: str | None = None
    trace_schema_version: str | None = None
    family_separation_status: Literal["full", "degraded"] = "full"
    family_warnings: tuple[str, ...] = ()
    expert_evaluation_required: bool = False
    prompt_identities: tuple[PromptIdentityRecord, ...] = ()
    api_budget_cny: Decimal | None = Field(
        default=None,
        gt=Decimal(0),
        le=Decimal(5000),
    )

    @model_validator(mode="after")
    def validate_mode_contract(self) -> InteractiveRuntimeRecord:
        prompt_components = tuple(
            identity.component for identity in self.prompt_identities
        )
        if len(prompt_components) != len(set(prompt_components)):
            raise ValueError("prompt identities cannot contain duplicate components")
        if self.mode == "local":
            if (
                self.role_bindings
                or self.model_snapshot_digest is not None
                or self.pricing_version is not None
                or self.pricing_catalog_digest is not None
                or self.qualification_suite_version is not None
                or self.gateway_name is not None
                or self.gateway_version is not None
                or self.adapter_name is not None
                or self.adapter_version is not None
                or self.decoder_version is not None
                or self.cache_schema_version is not None
                or self.trace_schema_version is not None
                or self.family_separation_status == "degraded"
                or self.family_warnings
                or self.expert_evaluation_required
                or self.prompt_identities
                or self.api_budget_cny is not None
            ):
                raise ValueError(
                    "local interactive runtime cannot contain API provenance"
                )
            return self

        required = (
            self.model_snapshot_digest,
            self.pricing_version,
            self.pricing_catalog_digest,
            self.qualification_suite_version,
            self.gateway_name,
            self.gateway_version,
            self.adapter_name,
            self.adapter_version,
            self.decoder_version,
            self.cache_schema_version,
            self.trace_schema_version,
            self.api_budget_cny,
        )
        if any(value is None for value in required):
            raise ValueError(
                "formal API runtime requires snapshot, pricing, and budget provenance"
            )
        if self.pricing_version is not None and not self.pricing_version.strip():
            raise ValueError("formal API pricing_version cannot be blank")
        version_fields = (
            self.qualification_suite_version,
            self.gateway_name,
            self.gateway_version,
            self.adapter_name,
            self.adapter_version,
            self.decoder_version,
            self.cache_schema_version,
            self.trace_schema_version,
        )
        if any(value is None or not value.strip() for value in version_fields):
            raise ValueError(
                "formal API runtime requires complete gateway, adapter, "
                "decoder, cache, trace, and qualification provenance"
            )
        if (
            self.family_separation_status == "full"
            and self.family_warnings
        ):
            raise ValueError(
                "full family separation cannot contain family warnings"
            )
        if (
            self.family_separation_status == "full"
            and self.expert_evaluation_required
        ):
            raise ValueError(
                "full family separation cannot require expert-primary evaluation"
            )
        if self.family_separation_status == "degraded" and not self.expert_evaluation_required:
            raise ValueError(
                "degraded family separation requires expert-primary evaluation"
            )
        if self.family_separation_status == "degraded" and not self.family_warnings:
            raise ValueError(
                "degraded family separation requires an auditable warning"
            )
        if len(self.family_warnings) != len(set(self.family_warnings)):
            raise ValueError("family_warnings cannot contain duplicates")
        if any(not warning.strip() for warning in self.family_warnings):
            raise ValueError("family_warnings cannot contain blanks")
        roles = tuple(binding.role for binding in self.role_bindings)
        if roles != (
            ModelRole.EVALUATED_AGENT,
            ModelRole.STUDENT_SIMULATOR,
            ModelRole.JUDGE,
            ModelRole.JUDGE,
        ):
            raise ValueError(
                "formal API role bindings must be companion, student, judge, judge"
            )
        model_ids = tuple(binding.model_id for binding in self.role_bindings)
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("formal API roles must use distinct model IDs")
        families = tuple(binding.family for binding in self.role_bindings)
        if self.family_separation_status == "full":
            if any(family is None for family in families):
                raise ValueError(
                    "full family separation requires known model families"
                )
            if len(families) != len(set(families)):
                raise ValueError("formal API roles must use distinct model families")
        elif (
            all(family is not None for family in families)
            and len(families) == len(set(families))
        ):
            raise ValueError(
                "degraded family separation must reflect unknown or overlapping families"
            )
        if not self.prompt_identities:
            raise ValueError("formal API runtime requires prompt identities")
        return self


class ExperimentManifest(FrozenArtifact):
    """Write-once description of one offline Track A run."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    benchmark_version: str
    track: Literal["A"] = "A"
    dataset_seed: int
    dataset_path: str | None = None
    dataset_digest: str
    split: DatasetSplit
    policy_names: tuple[str, ...]
    policy_runtimes: tuple[PolicyRuntimeRecord, ...]
    surface_indices: tuple[int, ...]
    repeats: int = Field(gt=0)
    risk_threshold: float = Field(gt=0.0, lt=1.0)
    selected_case_ids: tuple[str, ...]
    planned_unit_ids: tuple[str, ...]
    run_fingerprint: str

    @model_validator(mode="after")
    def validate_selection(self) -> ExperimentManifest:
        if not self.policy_names:
            raise ValueError("policy_names cannot be empty")
        if len(self.policy_names) != len(set(self.policy_names)):
            raise ValueError("policy_names cannot contain duplicates")
        runtime_names = tuple(runtime.policy_name for runtime in self.policy_runtimes)
        if runtime_names != self.policy_names:
            raise ValueError("policy_runtimes must match policy_names in the same order")
        if not self.surface_indices:
            raise ValueError("surface_indices cannot be empty")
        if any(index < 0 for index in self.surface_indices):
            raise ValueError("surface_indices cannot be negative")
        if len(self.surface_indices) != len(set(self.surface_indices)):
            raise ValueError("surface_indices cannot contain duplicates")
        if not self.selected_case_ids:
            raise ValueError("selected_case_ids cannot be empty")
        if len(self.selected_case_ids) != len(set(self.selected_case_ids)):
            raise ValueError("selected_case_ids cannot contain duplicates")
        if not self.planned_unit_ids:
            raise ValueError("planned_unit_ids cannot be empty")
        if len(self.planned_unit_ids) != len(set(self.planned_unit_ids)):
            raise ValueError("planned_unit_ids cannot contain duplicates")
        expected = (
            len(self.selected_case_ids)
            * len(self.policy_names)
            * len(self.surface_indices)
            * self.repeats
        )
        if len(self.planned_unit_ids) != expected:
            raise ValueError(
                f"planned_unit_ids has {len(self.planned_unit_ids)} entries; expected {expected}"
            )
        return self


class TrackBEpisodeSpecRecord(FrozenArtifact):
    """Frozen Track B episode selection embedded in a run manifest."""

    episode_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    seed: int
    surface_index: int = Field(ge=0)
    turn_limit: int = Field(ge=1)


class TrackBExperimentManifest(FrozenArtifact):
    """Write-once description of one interactive Track B run."""

    schema_version: Literal["1.2"] = "1.2"
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    benchmark_version: str
    track: Literal["B"] = "B"
    dataset_seed: int
    dataset_path: str | None = None
    dataset_digest: str
    selection_seed: int
    policy_names: tuple[str, ...]
    policy_runtimes: tuple[PolicyRuntimeRecord, ...]
    renderer_name: str = Field(min_length=1)
    interactive_runtime: InteractiveRuntimeRecord
    selected_case_ids: tuple[str, ...]
    episode_specs: tuple[TrackBEpisodeSpecRecord, ...]
    planned_unit_ids: tuple[str, ...]
    run_fingerprint: str

    @model_validator(mode="after")
    def validate_selection(self) -> TrackBExperimentManifest:
        _require_unique_nonempty(self.policy_names, "policy_names")
        runtime_names = tuple(runtime.policy_name for runtime in self.policy_runtimes)
        if runtime_names != self.policy_names:
            raise ValueError("policy_runtimes must match policy_names in the same order")
        if self.renderer_name != self.interactive_runtime.renderer_name:
            raise ValueError(
                "renderer_name must match the frozen interactive runtime"
            )
        _require_unique_nonempty(self.selected_case_ids, "selected_case_ids")
        _require_unique_nonempty(self.planned_unit_ids, "planned_unit_ids")
        if not self.episode_specs:
            raise ValueError("episode_specs cannot be empty")
        episode_ids = tuple(spec.episode_id for spec in self.episode_specs)
        _require_unique_nonempty(episode_ids, "episode_specs episode IDs")
        spec_case_ids = tuple(dict.fromkeys(spec.case_id for spec in self.episode_specs))
        if spec_case_ids != self.selected_case_ids:
            raise ValueError("selected_case_ids must match episode_specs case order")
        expected = len(self.episode_specs) * len(self.policy_names)
        if len(self.planned_unit_ids) != expected:
            raise ValueError(
                f"planned_unit_ids has {len(self.planned_unit_ids)} entries; expected {expected}"
            )
        return self


class TrackCExperimentManifest(FrozenArtifact):
    """Write-once description of one text relationship-safety Track C run."""

    schema_version: Literal["1.2"] = "1.2"
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    benchmark_version: str
    track: Literal["C"] = "C"
    scenario_digest: str
    scenario_ids: tuple[str, ...]
    configurations: tuple[TrackCConfiguration, ...]
    seeds: tuple[int, ...]
    responder_name: str = Field(min_length=1)
    interactive_runtime: InteractiveRuntimeRecord
    planned_unit_ids: tuple[str, ...]
    run_fingerprint: str

    @model_validator(mode="after")
    def validate_selection(self) -> TrackCExperimentManifest:
        _require_unique_nonempty(self.scenario_ids, "scenario_ids")
        _require_unique_nonempty(self.configurations, "configurations")
        _require_unique_nonempty(self.seeds, "seeds")
        _require_unique_nonempty(self.planned_unit_ids, "planned_unit_ids")
        if self.responder_name != self.interactive_runtime.responder_name:
            raise ValueError(
                "responder_name must match the frozen interactive runtime"
            )
        expected = len(self.scenario_ids) * len(self.configurations) * len(self.seeds)
        if len(self.planned_unit_ids) != expected:
            raise ValueError(
                f"planned_unit_ids has {len(self.planned_unit_ids)} entries; expected {expected}"
            )
        return self


class ExperimentCheckpoint(FrozenArtifact):
    """Atomically replaced progress index derived from append-only results."""

    schema_version: Literal["1.0"] = "1.0"
    run_fingerprint: str
    status: ExperimentStatus
    completed_unit_ids: tuple[str, ...]
    completed_count: int = Field(ge=0)
    planned_count: int = Field(gt=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_counts(self) -> ExperimentCheckpoint:
        if self.completed_count != len(self.completed_unit_ids):
            raise ValueError("completed_count must equal completed_unit_ids length")
        if len(self.completed_unit_ids) != len(set(self.completed_unit_ids)):
            raise ValueError("completed_unit_ids cannot contain duplicates")
        if self.completed_count > self.planned_count:
            raise ValueError("completed_count cannot exceed planned_count")
        expected_status = (
            ExperimentStatus.COMPLETE
            if self.completed_count == self.planned_count
            else ExperimentStatus.IN_PROGRESS
        )
        if self.status is not expected_status:
            raise ValueError("checkpoint status does not match its counts")
        return self


class TrackAMetricRecord(FrozenArtifact):
    depression_brier: float
    depression_ece: float
    depression_false_positive_rate: float
    immediate_risk_recall: float
    critical_action_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    acceptable_action_rate: float
    preferred_action_rate: float
    prohibited_action_rate: float
    intervention_window_hit_rate: float
    premature_intervention_rate: float
    late_or_missing_intervention_rate: float
    evidence_code_precision: float
    mean_detection_delay: float | None
    pre_onset_alarm_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    mean_pre_onset_lead_time: float | None = Field(default=None, ge=0.0)
    undetected_onset_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class StepDecisionRecord(FrozenArtifact):
    step_id: int = Field(ge=0)
    observation_id: str
    message: str
    depression_worsening: float = Field(ge=0.0, le=1.0)
    immediate_safety_risk: float = Field(ge=0.0, le=1.0)
    dependency_risk: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[EvidenceRef, ...] | None = None
    evidence_codes: tuple[str, ...] = ()
    primary_action: Action
    escalation_level: RiskLevel
    response_text: str

    @model_validator(mode="after")
    def validate_evidence_projection(self) -> StepDecisionRecord:
        if len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("evidence_codes cannot contain duplicates")
        if self.evidence_refs is None:
            return self
        reference_keys = tuple(
            (item.observation_id, item.evidence_code) for item in self.evidence_refs
        )
        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError("evidence_refs cannot contain duplicates")
        projected_codes = tuple(item.evidence_code for item in self.evidence_refs)
        if projected_codes != self.evidence_codes:
            raise ValueError("evidence_codes must equal the evidence_refs code projection")
        return self


class ExperimentResult(FrozenArtifact):
    """One complete case/surface/policy/repeat evaluation unit."""

    schema_version: Literal["1.0", "1.1"] = "1.1"
    unit_id: str
    run_id: str
    run_fingerprint: str
    case_id: str
    split: DatasetSplit
    policy_name: str
    surface_index: int = Field(ge=0)
    repeat_index: int = Field(ge=0)
    risk_trajectory: RiskTrajectory
    social_ecology: SocialEcology
    expression_style: ExpressionStyle
    metrics: TrackAMetricRecord
    steps: tuple[StepDecisionRecord, ...]

    @model_validator(mode="after")
    def validate_result_contract(self) -> ExperimentResult:
        if not self.steps:
            raise ValueError("Track A result steps cannot be empty")
        step_ids = tuple(step.step_id for step in self.steps)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Track A result steps cannot contain duplicates")
        if self.schema_version == "1.1" and any(step.evidence_refs is None for step in self.steps):
            raise ValueError("ExperimentResult 1.1 requires complete evidence_refs")
        return self


class TrackBTurnRecord(FrozenArtifact):
    turn_index: int = Field(ge=0)
    observation: ObservationV1
    decision: AgentDecisionV1
    student_turn: StudentTurnV1
    offered_fact_ids: tuple[str, ...]
    newly_disclosed_fact_ids: tuple[str, ...]
    renderer_name: str = Field(min_length=1)
    renderer_fallback_reason: str | None = None
    had_disclosure_opportunity: bool
    disclosure_interrupted: bool


class TrackBMetricRecord(FrozenArtifact):
    critical_evidence_recall: float = Field(ge=0.0, le=1.0)
    question_efficiency: float = Field(ge=0.0, le=1.0)
    disclosure_interruption_rate: float = Field(ge=0.0, le=1.0)
    critical_fact_count: int = Field(ge=0)
    acquired_critical_fact_count: int = Field(ge=0)
    evidence_question_count: int = Field(ge=0)
    productive_question_count: int = Field(ge=0)
    disclosure_opportunity_count: int = Field(ge=0)
    disclosure_interruption_count: int = Field(ge=0)
    renderer_fallback_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_count_contract(self) -> TrackBMetricRecord:
        if self.acquired_critical_fact_count > self.critical_fact_count:
            raise ValueError("acquired_critical_fact_count cannot exceed critical_fact_count")
        if self.productive_question_count > self.evidence_question_count:
            raise ValueError("productive_question_count cannot exceed evidence_question_count")
        if self.disclosure_interruption_count > self.disclosure_opportunity_count:
            raise ValueError(
                "disclosure_interruption_count cannot exceed disclosure_opportunity_count"
            )
        expected_rates = (
            (
                self.critical_evidence_recall,
                _safe_ratio(
                    self.acquired_critical_fact_count,
                    self.critical_fact_count,
                ),
                "critical_evidence_recall",
            ),
            (
                self.question_efficiency,
                _safe_ratio(
                    self.productive_question_count,
                    self.evidence_question_count,
                ),
                "question_efficiency",
            ),
            (
                self.disclosure_interruption_rate,
                _safe_ratio(
                    self.disclosure_interruption_count,
                    self.disclosure_opportunity_count,
                ),
                "disclosure_interruption_rate",
            ),
        )
        for actual, expected, name in expected_rates:
            if not math.isclose(actual, expected, abs_tol=1e-12):
                raise ValueError(f"{name} does not match its persisted counts")
        return self


class TrackBExperimentResult(FrozenArtifact):
    """One complete policy-by-episode Track B evaluation unit."""

    schema_version: Literal["1.0"] = "1.0"
    track: Literal["B"] = "B"
    unit_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_fingerprint: str
    case_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    spec: TrackBEpisodeSpecRecord
    renderer_name: str = Field(min_length=1)
    replay_digest: str
    metrics: TrackBMetricRecord
    turns: tuple[TrackBTurnRecord, ...]

    @model_validator(mode="after")
    def validate_episode(self) -> TrackBExperimentResult:
        if self.case_id != self.spec.case_id:
            raise ValueError("case_id must match the embedded episode spec")
        if len(self.turns) != self.spec.turn_limit:
            raise ValueError("turn count must match spec.turn_limit")
        question_count = sum(
            turn.decision.decision.primary_action
            in {Action.CLARIFY_SYMPTOMS, Action.DIRECT_SAFETY_CHECK}
            for turn in self.turns
        )
        opportunity_count = sum(turn.had_disclosure_opportunity for turn in self.turns)
        interruption_count = sum(
            turn.had_disclosure_opportunity and turn.disclosure_interrupted for turn in self.turns
        )
        fallback_count = sum(turn.renderer_fallback_reason is not None for turn in self.turns)
        if self.metrics.evidence_question_count != question_count:
            raise ValueError("evidence_question_count does not match Track B turns")
        if self.metrics.disclosure_opportunity_count != opportunity_count:
            raise ValueError("disclosure_opportunity_count does not match Track B turns")
        if self.metrics.disclosure_interruption_count != interruption_count:
            raise ValueError("disclosure_interruption_count does not match Track B turns")
        if self.metrics.renderer_fallback_count != fallback_count:
            raise ValueError("renderer_fallback_count does not match Track B turns")
        return self


class TrackCEpisodeSpecRecord(FrozenArtifact):
    episode_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    configuration: TrackCConfiguration
    seed: int


class TrackCTurnRecord(FrozenArtifact):
    turn_index: int = Field(ge=0)
    event_code: str = Field(min_length=1)
    user_text: str = Field(min_length=1)
    response_text: str
    violations: tuple[RelationshipViolation, ...]
    consecutive_nonresponses: int = Field(ge=0)


class TrackCMetricRecord(FrozenArtifact):
    total_agent_messages: int = Field(ge=0)
    violating_turn_count: int = Field(ge=0)
    total_violations: int = Field(ge=0)
    violation_turn_rate: float = Field(ge=0.0, le=1.0)
    violation_counts: dict[str, int]

    @model_validator(mode="after")
    def validate_violation_counts(self) -> TrackCMetricRecord:
        if any(count < 0 for count in self.violation_counts.values()):
            raise ValueError("violation counts cannot be negative")
        if sum(self.violation_counts.values()) != self.total_violations:
            raise ValueError("violation_counts total must equal total_violations")
        unknown = set(self.violation_counts) - {
            violation.value for violation in RelationshipViolation
        }
        if unknown:
            raise ValueError(f"violation_counts contains unknown codes: {sorted(unknown)}")
        return self


class TrackCExperimentResult(FrozenArtifact):
    """One complete scenario/configuration/seed Track C evaluation unit."""

    schema_version: Literal["1.0"] = "1.0"
    track: Literal["C"] = "C"
    unit_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_fingerprint: str
    responder_name: str = Field(min_length=1)
    spec: TrackCEpisodeSpecRecord
    replay_digest: str
    metrics: TrackCMetricRecord
    turns: tuple[TrackCTurnRecord, ...]

    @model_validator(mode="after")
    def validate_episode_metrics(self) -> TrackCExperimentResult:
        if not self.turns:
            raise ValueError("Track C result turns cannot be empty")
        turn_indices = tuple(turn.turn_index for turn in self.turns)
        if len(turn_indices) != len(set(turn_indices)):
            raise ValueError("Track C turn indices cannot contain duplicates")
        violations = tuple(violation for turn in self.turns for violation in turn.violations)
        expected_counts = Counter(violation.value for violation in violations)
        expected_message_count = sum(bool(turn.response_text.strip()) for turn in self.turns)
        expected_violating_turns = sum(bool(turn.violations) for turn in self.turns)
        expected_rate = expected_violating_turns / len(self.turns)
        if self.metrics.total_agent_messages != expected_message_count:
            raise ValueError("total_agent_messages does not match Track C turns")
        if self.metrics.violating_turn_count != expected_violating_turns:
            raise ValueError("violating_turn_count does not match Track C turns")
        if self.metrics.total_violations != len(violations):
            raise ValueError("total_violations does not match Track C turns")
        if self.metrics.violation_counts != dict(expected_counts):
            raise ValueError("violation_counts does not match Track C turns")
        if not math.isclose(
            self.metrics.violation_turn_rate,
            expected_rate,
            abs_tol=1e-12,
        ):
            raise ValueError("violation_turn_rate does not match Track C turns")
        return self


@dataclass(frozen=True, slots=True)
class ExperimentRunOutcome:
    run_dir: Path
    status: ExperimentStatus
    planned_count: int
    completed_count: int
    executed_count: int


class MetricEstimateRecord(FrozenArtifact):
    """Point estimate and canonical-cluster percentile confidence interval."""

    schema_version: Literal["1.0"] = "1.0"
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    confidence_level: float = Field(gt=0.0, lt=1.0)
    cluster_count: int = Field(gt=0)
    bootstrap_resamples: int = Field(gt=0)
    valid_bootstrap_resamples: int = Field(ge=0)
    bootstrap_seed: int

    @model_validator(mode="after")
    def validate_interval(self) -> MetricEstimateRecord:
        if self.valid_bootstrap_resamples > self.bootstrap_resamples:
            raise ValueError("valid_bootstrap_resamples cannot exceed bootstrap_resamples")
        bounds = (self.ci_lower, self.ci_upper)
        if self.estimate is None:
            if any(bound is not None for bound in bounds):
                raise ValueError("undefined estimates cannot have CI bounds")
            return self
        if (self.ci_lower is None) != (self.ci_upper is None):
            raise ValueError("CI bounds must both be defined or both be absent")
        if (
            self.ci_lower is not None
            and self.ci_upper is not None
            and self.ci_lower > self.ci_upper
        ):
            raise ValueError("ci_lower cannot exceed ci_upper")
        return self


class StatisticalProtocolRecord(FrozenArtifact):
    """Frozen statistical protocol embedded in every paper-facing report."""

    schema_version: Literal["1.0"] = "1.0"
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(default=2_000, gt=0)
    permutation_resamples: int = Field(default=10_000, gt=0)
    random_seed: int = 20_260_904
    cluster_unit: str = Field(min_length=1)
    repeated_measure_handling: str = Field(min_length=1)
    paired_unit: str = Field(min_length=1)
    paired_test: Literal["two-sided exact-or-monte-carlo sign-flip"] = (
        "two-sided exact-or-monte-carlo sign-flip"
    )
    effect_size: Literal["paired Cohen dz"] = "paired Cohen dz"


class PairedComparisonRecord(FrozenArtifact):
    """Versioned paired result; every difference is ``left - right``."""

    schema_version: Literal["1.1"] = "1.1"
    metric_name: str = Field(min_length=1)
    left_name: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    pair_unit: str = Field(min_length=1)
    higher_is_better: bool
    matched_pair_count: int = Field(gt=0)
    cluster_count: int = Field(gt=0)
    pair_count: int = Field(gt=0)
    effective_pair_count: int = Field(ge=0)
    mean_difference: float
    median_difference: float | None
    ci_lower: float
    ci_upper: float
    cohen_dz: float | None
    raw_p_value: float = Field(ge=0.0, le=1.0)
    holm_adjusted_p_value: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    comparison_family: str = Field(min_length=1)
    primary_endpoint: bool = False
    left_only_count: int = Field(default=0, ge=0)
    right_only_count: int = Field(default=0, ge=0)
    complete_pairing: bool = True
    test_method: Literal[
        "all_ties",
        "exact_sign_flip",
        "monte_carlo_sign_flip",
    ]
    win_count: int | None = Field(default=None, ge=0)
    tie_count: int | None = Field(default=None, ge=0)
    loss_count: int | None = Field(default=None, ge=0)
    bootstrap_resamples: int = Field(gt=0)
    permutation_resamples: int = Field(gt=0)
    random_seed: int

    @model_validator(mode="after")
    def validate_counts(self) -> PairedComparisonRecord:
        if self.cluster_count != self.pair_count:
            raise ValueError("pair_count must equal independent cluster_count")
        if self.matched_pair_count < self.cluster_count:
            raise ValueError("matched_pair_count cannot be smaller than cluster_count")
        if self.effective_pair_count > self.pair_count:
            raise ValueError("effective_pair_count cannot exceed pair_count")
        directional_counts = (
            self.win_count,
            self.tie_count,
            self.loss_count,
        )
        if any(value is None for value in directional_counts):
            if any(value is not None for value in directional_counts):
                raise ValueError("win/tie/loss counts must all be defined or all be absent")
        else:
            win_count = int(self.win_count or 0)
            tie_count = int(self.tie_count or 0)
            loss_count = int(self.loss_count or 0)
            if win_count + tie_count + loss_count != self.pair_count:
                raise ValueError("win/tie/loss counts must sum to pair_count")
            if self.effective_pair_count != win_count + loss_count:
                raise ValueError("effective_pair_count must equal wins plus losses")
        if self.ci_lower > self.ci_upper:
            raise ValueError("ci_lower cannot exceed ci_upper")
        if self.holm_adjusted_p_value + 1e-12 < self.raw_p_value:
            raise ValueError("Holm-adjusted p-value cannot be smaller than raw p-value")
        if not math.isclose(self.p_value, self.raw_p_value, abs_tol=1e-12):
            raise ValueError("p_value is the compatibility alias for raw_p_value")
        if self.complete_pairing != (self.left_only_count == 0 and self.right_only_count == 0):
            raise ValueError("complete_pairing does not match missing pair counts")
        return self


class RelationshipSafetyBreakdown(FrozenArtifact):
    """Per-turn Track C rates kept separate to avoid hiding severe failures."""

    exclusivity_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    possessiveness_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    guilt_induction_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    identity_deception_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    discourage_offline_help_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    false_confidentiality_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    unsafe_guardian_assumption_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    proactive_persistence_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    memory_instruction_accepted_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    missing_required_response_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class EvidenceScoringRecord(FrozenArtifact):
    """Provenance and availability of Track A expert-evidence scoring."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal[
        "NO_GOLD_ANNOTATION",
        "VALIDATED_ADJUDICATED_GOLD",
        "LEGACY_CODES_ONLY",
        "MIXED_EVIDENCE_FORMAT",
    ]
    annotation_path: str | None = None
    annotation_digest: str | None = None
    batch_id: str | None = None
    manual_version: str | None = None
    dataset_digest: str | None = None
    gold_node_count: int = Field(default=0, ge=0)
    full_reference_record_count: int = Field(default=0, ge=0)
    legacy_record_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_provenance(self) -> EvidenceScoringRecord:
        annotation_fields = (
            self.annotation_path,
            self.annotation_digest,
            self.batch_id,
            self.manual_version,
        )
        gold_fields = (*annotation_fields, self.dataset_digest)
        if self.status == "VALIDATED_ADJUDICATED_GOLD":
            if any(value is None for value in gold_fields):
                raise ValueError("validated evidence gold requires complete provenance")
            if self.gold_node_count <= 0:
                raise ValueError("validated evidence gold requires at least one node")
            if self.legacy_record_count:
                raise ValueError("validated evidence gold cannot include legacy records")
        if self.annotation_path is None and any(
            value is not None for value in annotation_fields[1:]
        ):
            raise ValueError("annotation provenance requires annotation_path")
        return self


class FailureCaseRecord(FrozenArtifact):
    """Ranked, auditable failure extracted from an immutable result unit."""

    schema_version: Literal["1.0"] = "1.0"
    failure_id: str = Field(min_length=1)
    rank: int = Field(gt=0)
    track: Track
    severity_score: float = Field(gt=0.0)
    system_name: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    case_or_scenario_id: str = Field(min_length=1)
    pair_key: str = Field(min_length=1)
    step_or_turn_index: int | None = Field(default=None, ge=0)
    surface_index: int | None = Field(default=None, ge=0)
    repeat_index: int | None = Field(default=None, ge=0)
    seed: int | None = None
    issue_codes: tuple[str, ...]
    message_excerpt: str
    response_excerpt: str
    reference_value: float | None = None
    predicted_value: float | None = None
    selected_action: Action | None = None
    violation_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_issue_codes(self) -> FailureCaseRecord:
        if not self.issue_codes:
            raise ValueError("issue_codes cannot be empty")
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("issue_codes cannot contain duplicates")
        if len(self.violation_codes) != len(set(self.violation_codes)):
            raise ValueError("violation_codes cannot contain duplicates")
        return self


class ReportRow(FrozenArtifact):
    policy_name: str
    episode_count: int = Field(gt=0)
    case_count: int = Field(gt=0)
    surface_count: int = Field(gt=0)
    repeat_count: int = Field(gt=0)
    canonical_node_count: int = Field(gt=0)
    depression_auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    depression_auprc: float | None = Field(default=None, ge=0.0, le=1.0)
    depression_high_risk_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    depression_brier: float
    depression_ece: float
    depression_false_positive_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    immediate_risk_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    critical_action_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    acceptable_action_rate: float
    preferred_action_rate: float
    prohibited_action_rate: float
    intervention_window_hit_rate: float
    premature_intervention_rate: float
    late_or_missing_intervention_rate: float
    evidence_code_precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    evidence_code_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_code_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    fabricated_evidence_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    mean_detection_delay: float | None
    pre_onset_alarm_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    mean_pre_onset_lead_time: float | None = Field(default=None, ge=0.0)
    undetected_onset_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    metric_estimates: dict[str, MetricEstimateRecord]

    @model_validator(mode="after")
    def validate_metric_estimates(self) -> ReportRow:
        _require_exact_metric_names(
            self.metric_estimates,
            {
                "depression_auroc",
                "depression_auprc",
                "depression_high_risk_recall",
                "depression_brier",
                "depression_ece",
                "depression_false_positive_rate",
                "immediate_risk_recall",
                "critical_action_recall",
                "acceptable_action_rate",
                "preferred_action_rate",
                "prohibited_action_rate",
                "intervention_window_hit_rate",
                "premature_intervention_rate",
                "late_or_missing_intervention_rate",
                "evidence_code_precision",
                "evidence_code_recall",
                "evidence_code_f1",
                "fabricated_evidence_rate",
                "mean_detection_delay",
                "pre_onset_alarm_rate",
                "mean_pre_onset_lead_time",
                "undetected_onset_rate",
            },
            row_name="Track A",
        )
        return self


class ReportDocument(FrozenArtifact):
    schema_version: Literal["2.0"] = "2.0"
    track: Literal["A"] = "A"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None
    run_fingerprint: str | None
    result_path: str
    completed_unit_count: int = Field(ge=0)
    planned_unit_count: int | None = Field(default=None, ge=0)
    complete: bool
    statistical_protocol: StatisticalProtocolRecord
    evidence_scoring: EvidenceScoringRecord
    rows: tuple[ReportRow, ...]
    paired_comparisons: tuple[PairedComparisonRecord, ...] = ()
    failure_case_limit: int = Field(default=30, ge=0)
    failure_cases: tuple[FailureCaseRecord, ...] = ()

    @model_validator(mode="after")
    def validate_failure_cases(self) -> ReportDocument:
        _validate_report_failures(
            self.failure_cases,
            limit=self.failure_case_limit,
            expected_track=Track.A,
        )
        return self


class TrackBReportRow(FrozenArtifact):
    policy_name: str
    episode_count: int = Field(gt=0)
    case_count: int = Field(gt=0)
    seed_count: int = Field(gt=0)
    mean_turn_count: float = Field(gt=0.0)
    critical_evidence_recall: float = Field(ge=0.0, le=1.0)
    question_efficiency: float = Field(ge=0.0, le=1.0)
    disclosure_interruption_rate: float = Field(ge=0.0, le=1.0)
    renderer_fallback_turn_rate: float = Field(ge=0.0, le=1.0)
    metric_estimates: dict[str, MetricEstimateRecord]

    @model_validator(mode="after")
    def validate_metric_estimates(self) -> TrackBReportRow:
        _require_exact_metric_names(
            self.metric_estimates,
            {
                "critical_evidence_recall",
                "question_efficiency",
                "disclosure_interruption_rate",
                "renderer_fallback_turn_rate",
                "mean_turn_count",
            },
            row_name="Track B",
        )
        return self


class TrackBReportDocument(FrozenArtifact):
    schema_version: Literal["2.0"] = "2.0"
    track: Literal["B"] = "B"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None
    run_fingerprint: str | None
    result_path: str
    completed_unit_count: int = Field(ge=0)
    planned_unit_count: int | None = Field(default=None, ge=0)
    complete: bool
    statistical_protocol: StatisticalProtocolRecord
    family_separation_status: Literal["full", "degraded"] | None = None
    family_warnings: tuple[str, ...] = ()
    expert_evaluation_required: bool = False
    rows: tuple[TrackBReportRow, ...]
    paired_comparisons: tuple[PairedComparisonRecord, ...] = ()
    failure_case_limit: int = Field(default=30, ge=0)
    failure_cases: tuple[FailureCaseRecord, ...] = ()

    @model_validator(mode="after")
    def validate_failure_cases(self) -> TrackBReportDocument:
        _validate_report_family_metadata(
            status=self.family_separation_status,
            warnings=self.family_warnings,
            expert_evaluation_required=self.expert_evaluation_required,
        )
        _validate_report_failures(
            self.failure_cases,
            limit=self.failure_case_limit,
            expected_track=Track.B,
        )
        return self


class TrackCReportRow(FrozenArtifact):
    configuration: TrackCConfiguration
    responder_name: str
    episode_count: int = Field(gt=0)
    scenario_count: int = Field(gt=0)
    seed_count: int = Field(gt=0)
    agent_message_count: int = Field(ge=0)
    response_opportunity_count: int = Field(ge=0)
    violation_free_episode_rate: float = Field(ge=0.0, le=1.0)
    violation_turn_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    mean_violations_per_episode: float = Field(ge=0.0)
    violation_counts: dict[str, int]
    violation_opportunity_counts: dict[str, int]
    relationship_safety: RelationshipSafetyBreakdown
    metric_estimates: dict[str, MetricEstimateRecord]

    @model_validator(mode="after")
    def validate_metric_estimates(self) -> TrackCReportRow:
        known_codes = {violation.value for violation in RelationshipViolation}
        if set(self.violation_counts) - known_codes:
            raise ValueError("Track C row contains unknown violation codes")
        if set(self.violation_opportunity_counts) != known_codes:
            raise ValueError("Track C row must expose every violation opportunity count")
        if any(
            count < 0
            for count in (
                *self.violation_counts.values(),
                *self.violation_opportunity_counts.values(),
            )
        ):
            raise ValueError("Track C violation and opportunity counts cannot be negative")
        _require_exact_metric_names(
            self.metric_estimates,
            {
                "violation_free_episode_rate",
                "violation_turn_rate",
                "mean_violations_per_episode",
                "exclusivity_rate",
                "possessiveness_rate",
                "guilt_induction_rate",
                "identity_deception_rate",
                "discourage_offline_help_rate",
                "false_confidentiality_rate",
                "unsafe_guardian_assumption_rate",
                "proactive_persistence_rate",
                "memory_instruction_accepted_rate",
                "missing_required_response_rate",
            },
            row_name="Track C",
        )
        return self


class TrackCReportDocument(FrozenArtifact):
    schema_version: Literal["2.0"] = "2.0"
    track: Literal["C"] = "C"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None
    run_fingerprint: str | None
    result_path: str
    completed_unit_count: int = Field(ge=0)
    planned_unit_count: int | None = Field(default=None, ge=0)
    complete: bool
    statistical_protocol: StatisticalProtocolRecord
    family_separation_status: Literal["full", "degraded"] | None = None
    family_warnings: tuple[str, ...] = ()
    expert_evaluation_required: bool = False
    rows: tuple[TrackCReportRow, ...]
    paired_comparisons: tuple[PairedComparisonRecord, ...] = ()
    failure_case_limit: int = Field(default=30, ge=0)
    failure_cases: tuple[FailureCaseRecord, ...] = ()

    @model_validator(mode="after")
    def validate_failure_cases(self) -> TrackCReportDocument:
        _validate_report_family_metadata(
            status=self.family_separation_status,
            warnings=self.family_warnings,
            expert_evaluation_required=self.expert_evaluation_required,
        )
        _validate_report_failures(
            self.failure_cases,
            limit=self.failure_case_limit,
            expected_track=Track.C,
        )
        return self


def _require_unique_nonempty(values: tuple[object, ...], field: str) -> None:
    if not values:
        raise ValueError(f"{field} cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} cannot contain duplicates")


def _validate_report_family_metadata(
    *,
    status: Literal["full", "degraded"] | None,
    warnings: tuple[str, ...],
    expert_evaluation_required: bool,
) -> None:
    if status is None:
        if warnings or expert_evaluation_required:
            raise ValueError(
                "reports without formal family metadata cannot contain warnings "
                "or expert-evaluation requirements"
            )
        return
    if len(warnings) != len(set(warnings)):
        raise ValueError("report family_warnings cannot contain duplicates")
    if any(not warning.strip() for warning in warnings):
        raise ValueError("report family_warnings cannot contain blanks")
    if status == "full":
        if warnings:
            raise ValueError("full report family separation cannot contain warnings")
        if expert_evaluation_required:
            raise ValueError(
                "full report family separation cannot require expert evaluation"
            )
    elif not expert_evaluation_required:
        raise ValueError(
            "degraded report family separation requires expert evaluation"
        )
    elif not warnings:
        raise ValueError(
            "degraded report family separation requires an auditable warning"
        )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _require_exact_metric_names(
    metrics: dict[str, MetricEstimateRecord],
    expected: set[str],
    *,
    row_name: str,
) -> None:
    actual = set(metrics)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{row_name} metric_estimates mismatch; missing={missing}, unexpected={unexpected}"
        )


def _validate_report_failures(
    failures: tuple[FailureCaseRecord, ...],
    *,
    limit: int,
    expected_track: Track,
) -> None:
    if len(failures) > limit:
        raise ValueError("failure case count cannot exceed failure_case_limit")
    if tuple(failure.rank for failure in failures) != tuple(range(1, len(failures) + 1)):
        raise ValueError("failure case ranks must be contiguous from one")
    if any(failure.track is not expected_track for failure in failures):
        raise ValueError("failure case track does not match report track")
