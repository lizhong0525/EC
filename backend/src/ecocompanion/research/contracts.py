"""Separate observable inputs, provenance, human judgments and simulation truth."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecocompanion.domain.enums import Action, ActorRole, EscalationTarget, RiskLevel
from ecocompanion.domain.schemas import AgentDecisionV1, EvidenceRef, ObservationV1
from ecocompanion.research import VERSION


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Provenance(Record):
    origin: Literal["public_generator", "independent_author", "automatic_draft"]
    source_id: str = Field(min_length=1)
    author_id: str = Field(min_length=1)
    person_group: str = Field(min_length=1)
    plot_group: str = Field(min_length=1)
    template_group: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    review_status: Literal["draft", "reviewed"] = "draft"
    review_reference: str | None = None
    synthetic: Literal[True] = True

    @model_validator(mode="after")
    def validate_review(self) -> Provenance:
        if self.review_status == "reviewed" and not self.review_reference:
            raise ValueError("reviewed provenance needs an external review reference")
        return self


class ResearchCase(Record):
    case_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    observations: tuple[ObservationV1, ...]
    provenance: Provenance
    split: Literal["development", "validation", "public_test", "independent_blind"]
    contrast_group: str | None = None
    contrast_kind: Literal["same_word", "same_meaning", "minimal_counterfactual"] | None = None
    changed_factor: str | None = None

    @model_validator(mode="after")
    def validate_sequence(self) -> ResearchCase:
        if not self.observations:
            raise ValueError("a case needs observations")
        ids = [o.current_observation.observation_id for o in self.observations]
        if len(ids) != len(set(ids)) or any(re.fullmatch(r"[0-9a-f]{32}", i) is None for i in ids):
            raise ValueError("duplicate or nonopaque observation IDs")
        for index, observation in enumerate(self.observations):
            if observation.step_id != index or observation.episode_id != self.case_id:
                raise ValueError("case/observation identity or sequence mismatch")
            if observation.benchmark_version != VERSION:
                raise ValueError("v2 observations must declare v2")
            allowed = set(ids[:index])
            if any(
                m.message_id not in allowed
                for m in observation.visible_context.conversation_history
            ):
                raise ValueError("history references future or unknown input")
            for message in observation.visible_context.conversation_history:
                earlier = self.observations[ids.index(message.message_id)]
                if (
                    message.role is not ActorRole.STUDENT_SIMULATOR
                    or message.content != earlier.current_observation.message
                    or message.simulated_time != earlier.simulated_time
                ):
                    raise ValueError("history differs from frozen student input")
        if self.split == "independent_blind" and (
            self.provenance.origin != "independent_author"
            or self.provenance.review_status != "reviewed"
        ):
            raise ValueError("blind input needs independently authored, reviewed provenance")
        if self.contrast_kind is not None and not (self.contrast_group and self.changed_factor):
            raise ValueError("contrast cases need a pair group and changed factor")
        return self


class Judgment(Record):
    """Human input only. Identity, timing and lineage are stamped by the service."""

    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[EvidenceRef, ...] = ()
    missing_information: tuple[str, ...] = ()
    acceptable_actions: tuple[Action, ...] = Field(min_length=1)
    preferred_action: Action
    prohibited_actions: tuple[Action, ...] = ()
    direct_safety_check_required: bool
    referral_target: EscalationTarget = EscalationTarget.NONE
    deadline_now: bool = False
    reason: str = Field(min_length=1, max_length=600)
    resolution: Literal["resolved", "unresolved"] = "resolved"

    @model_validator(mode="after")
    def validate_sets(self) -> Judgment:
        acceptable, prohibited = set(self.acceptable_actions), set(self.prohibited_actions)
        if len(acceptable) != len(self.acceptable_actions) or len(prohibited) != len(
            self.prohibited_actions
        ):
            raise ValueError("duplicate actions")
        if self.preferred_action not in acceptable or acceptable & prohibited:
            raise ValueError("invalid acceptable/preferred/prohibited actions")
        refs = [(e.observation_id, e.evidence_code) for e in self.evidence]
        if len(refs) != len(set(refs)) or any(not e.evidence_code.strip() for e in self.evidence):
            raise ValueError("duplicate or empty evidence")
        return self


class Prediction(Record):
    policy: str
    case_id: str
    step_id: int = Field(ge=0)
    repeat: int = Field(default=0, ge=0)
    status: Literal["ok", "error", "refusal", "fallback", "format_error", "timeout"]
    decision: AgentDecisionV1 | None = None
    error_code: str | None = None
    calls: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_cny: float | None = Field(default=None, ge=0)
    latency_seconds: float = Field(default=0, ge=0)
    protocol_digest: str | None = None
    model_snapshot_digest: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> Prediction:
        if self.status in {"ok", "fallback"} and self.decision is None:
            raise ValueError("successful/fallback records require a decision")
        if self.status not in {"ok", "fallback"} and self.decision is not None:
            raise ValueError("failed predictions cannot carry a scored decision")
        return self


class Protocol(Record):
    version: Literal["ecotriage-v2.0.0"] = "ecotriage-v2.0.0"
    status: Literal["development", "preregistered"] = "development"
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_names: tuple[str, ...] = Field(min_length=1)
    model_identity: str = Field(min_length=1)
    model_snapshot_digest: str | None = None
    prompt_digest: str = Field(min_length=1)
    source_digest: str | None = None
    runtime_digest: str | None = None
    context_contract: Literal["frozen_student_history_only"] = "frozen_student_history_only"
    execution: Literal["local_rules", "qualified_api"] = "local_rules"
    repeats: int = Field(default=1, ge=1)
    seeds: tuple[int, ...] = (20261020,)
    simulation_threshold: float = Field(default=0.5, gt=0, lt=1)
    bootstrap_resamples: int = Field(default=2000, ge=100)
    max_calls_per_node: int = Field(default=1, ge=1)
    max_same_model_retries: int = Field(default=0, ge=0)
    max_cost_cny: float = Field(default=0, ge=0, le=5000)
    primary_metrics: tuple[str, ...] = (
        "required_safety_action_recall",
        "prohibited_action_rate",
        "acceptable_action_rate",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_identity(self) -> Protocol:
        if not self.primary_metrics or not set(self.primary_metrics) <= {
            "required_safety_action_recall",
            "prohibited_action_rate",
            "acceptable_action_rate",
        }:
            raise ValueError("primary metrics must use the versioned preregistered allowlist")
        if len(set(self.policy_names)) != len(self.policy_names):
            raise ValueError("duplicate policies")
        if len(self.seeds) != self.repeats or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("one distinct frozen seed is required per repeat")
        if self.execution == "qualified_api" and not self.model_snapshot_digest:
            raise ValueError("API experiments need a qualified model snapshot digest")
        if self.execution == "local_rules" and self.model_identity != "no-model":
            raise ValueError("local rule runs must not claim a model identity")
        return self
