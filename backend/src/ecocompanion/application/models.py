"""Immutable HTTP/application contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    EscalationTarget,
    MemoryKind,
    RiskLevel,
)
from ecocompanion.domain.schemas import ModelCallUsage, RiskEstimate, SessionStateV1


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParticipantType(StrEnum):
    SYNTHETIC_CASE = "synthetic_case"
    ADULT_RESEARCH_TESTER = "adult_research_tester"


class CreateSessionRequest(ImmutableModel):
    participant_type: ParticipantType = ParticipantType.ADULT_RESEARCH_TESTER
    synthetic_case_id: str | None = None
    memory_enabled: bool = True
    proactive_enabled: bool = False
    research_mode: bool = True
    model: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_participant_contract(self) -> CreateSessionRequest:
        if (
            self.participant_type is ParticipantType.SYNTHETIC_CASE
            and not self.synthetic_case_id
        ):
            raise ValueError("synthetic_case_id is required for synthetic sessions")
        if (
            self.participant_type is ParticipantType.ADULT_RESEARCH_TESTER
            and self.synthetic_case_id is not None
        ):
            raise ValueError("synthetic_case_id is only valid for synthetic sessions")
        if (
            self.participant_type is ParticipantType.SYNTHETIC_CASE
            and self.proactive_enabled
        ):
            raise ValueError("proactivity is only available to adult research testers")
        return self


class TurnRequest(ImmutableModel):
    text: str = Field(min_length=1, max_length=8_000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    simulated_time: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def normalize_text_contract(self) -> TurnRequest:
        if not self.text.strip():
            raise ValueError("text cannot be blank")
        return self


class AuthorizedMemoryRequest(ImmutableModel):
    kind: Literal[MemoryKind.EPISODIC, MemoryKind.PREFERENCE]
    content: str = Field(min_length=1, max_length=4_000)
    user_authorized: Literal[True]
    source_message_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def reject_blank_content(self) -> AuthorizedMemoryRequest:
        if not self.content.strip():
            raise ValueError("memory content cannot be blank")
        return self


class MemoryCorrectionRequest(ImmutableModel):
    content: str = Field(min_length=1, max_length=4_000)
    user_authorized: Literal[True]

    @model_validator(mode="after")
    def reject_blank_content(self) -> MemoryCorrectionRequest:
        if not self.content.strip():
            raise ValueError("memory content cannot be blank")
        return self


class ProactivityUpdateRequest(ImmutableModel):
    enabled: bool


class ProactiveMessageView(ImmutableModel):
    prompt_id: str
    text: str
    sent_at: datetime
    used_fallback: bool


class SessionView(ImmutableModel):
    session_id: str
    state: SessionStateV1
    model: str | None
    research_mode: bool
    participant_type: ParticipantType
    synthetic_case_id: str | None = None


class MessageView(ImmutableModel):
    message_id: str
    ordinal: int = Field(ge=0)
    role: ActorRole
    content: str
    simulated_time: str
    created_at: datetime


class DecisionView(ImmutableModel):
    primary_action: Action
    escalation_level: RiskLevel
    escalation_target: EscalationTarget
    risk_estimate: RiskEstimate
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence_codes: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    safety_passed: bool
    used_fallback: bool
    audit_codes: tuple[str, ...] = ()
    model_calls: tuple[ModelCallUsage, ...] = ()
    policy_version: str = "core-policy-v1"
    prompt_version: str | None = None

    @model_validator(mode="after")
    def distinguish_model_generation_from_fallback(self) -> DecisionView:
        if not self.used_fallback and not self.model_calls:
            raise ValueError(
                "a non-fallback Web turn must contain at least one recorded model call"
            )
        return self


class TurnResult(ImmutableModel):
    session: SessionView
    response: str
    decision: DecisionView


class ModelSnapshotView(ImmutableModel):
    source: Literal["snapshot", "runtime", "unavailable"]
    active_model: str | None
    discovered_at: str | None = None
    models: tuple[dict[str, object], ...] = ()
    capabilities: tuple[dict[str, object], ...] = ()
    authenticated_discovery: bool | None = None
    detail: str | None = None
    provider_location: Literal["local", "remote", "unavailable"] = "unavailable"


class BenchmarkRunView(ImmutableModel):
    """Read-only summary of one persisted benchmark run."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    directory_name: str = Field(min_length=1)
    track: Literal["A", "B", "C"]
    created_at: datetime
    status: Literal["in_progress", "complete"]
    completed_unit_count: int = Field(ge=0)
    planned_unit_count: int = Field(gt=0)
    methods: tuple[str, ...]
    report_available: bool

    @model_validator(mode="after")
    def validate_progress(self) -> BenchmarkRunView:
        if self.completed_unit_count > self.planned_unit_count:
            raise ValueError("completed units cannot exceed planned units")
        expected_status = (
            "complete"
            if self.completed_unit_count == self.planned_unit_count
            else "in_progress"
        )
        if self.status != expected_status:
            raise ValueError("benchmark run status does not match unit counts")
        if not self.methods:
            raise ValueError("methods cannot be empty")
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("methods cannot contain duplicates")
        return self


class BenchmarkRunCatalogView(ImmutableModel):
    """Versioned list response for the local research console."""

    schema_version: Literal["1.0"] = "1.0"
    runs: tuple[BenchmarkRunView, ...]
    ignored_directory_names: tuple[str, ...] = ()


class BenchmarkReportView(ImmutableModel):
    """A validated run summary plus its versioned report payload."""

    schema_version: Literal["1.0"] = "1.0"
    run: BenchmarkRunView
    report: dict[str, Any]


class BenchmarkStartRequest(ImmutableModel):
    track: Literal["A", "B", "C"] = "A"
    case_limit: int = Field(default=4, ge=1, le=24)
    policies: tuple[
        Literal["history-aware-rule", "keyword-triage", "always-listen", "always-escalate"], ...
    ] = ("history-aware-rule", "keyword-triage", "always-listen")
    split: Literal["dev", "public_test"] = "dev"

    @model_validator(mode="after")
    def validate_selection(self) -> BenchmarkStartRequest:
        if (self.track != "C" and not self.policies) or len(self.policies) != len(
            set(self.policies)
        ):
            raise ValueError("select at least one policy, without duplicates")
        if self.track == "C" and self.case_limit > 12:
            raise ValueError("Track C has at most 12 scenarios")
        return self


class BenchmarkJobView(ImmutableModel):
    job_id: str
    run_id: str
    status: Literal["queued", "running", "reporting", "complete", "failed", "interrupted"]
    created_at: datetime
    configuration: BenchmarkStartRequest
    execution: Literal["local_rules"] = "local_rules"
    model_calls: Literal[0] = 0
    error: str | None = None
    run: BenchmarkRunView | None = None
