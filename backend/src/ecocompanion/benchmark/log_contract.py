"""Versioned neutral contracts for importing synthetic external event logs.

The import contract intentionally contains only observable information.  Its
output is an expert-relabel draft and cannot be parsed as ``CaseSpecV1`` until
the project-specific hidden state and annotation workflow have been completed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ecocompanion.domain.schemas import VisibleProfile

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_MAX_SHORT_TEXT_LENGTH = 256
_MAX_TEXT_LENGTH = 16_384

RequiredExpertAnnotation = Literal[
    "risk_trajectory",
    "social_ecology",
    "expression_style",
    "disclosure_facts_and_evidence_codes",
    "risk_level_and_confidence_per_step",
    "acceptable_preferred_and_prohibited_actions_per_step",
    "direct_safety_question_requirement_per_step",
    "intervention_windows",
    "escalation_level_and_target",
]

REQUIRED_EXPERT_ANNOTATIONS: tuple[RequiredExpertAnnotation, ...] = (
    "risk_trajectory",
    "social_ecology",
    "expression_style",
    "disclosure_facts_and_evidence_codes",
    "risk_level_and_confidence_per_step",
    "acceptable_preferred_and_prohibited_actions_per_step",
    "direct_safety_question_requirement_per_step",
    "intervention_windows",
    "escalation_level_and_target",
)


class _FrozenContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _strip_required_text(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be blank")
        return stripped
    return value


def _strip_text_tuple(value: object) -> object:
    if isinstance(value, (list, tuple)):
        normalized: list[object] = []
        for item in value:
            normalized.append(_strip_required_text(item))
        return tuple(normalized)
    return value


class ExternalPersonV1(_FrozenContractModel):
    """Observable synthetic-person attributes carried by each source record."""

    source_person_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
        description="Opaque synthetic identifier; never use a real name or contact detail.",
    )
    age_group: str = Field(min_length=1, max_length=64)
    language: str = Field(default="zh-CN", min_length=1, max_length=32)
    disclosed_background: tuple[str, ...] = Field(default=(), max_length=64)
    persona_tags: tuple[str, ...] = Field(default=(), max_length=64)

    _normalize_required = field_validator(
        "source_person_id",
        "age_group",
        "language",
        mode="before",
    )(_strip_required_text)
    _normalize_collections = field_validator(
        "disclosed_background",
        "persona_tags",
        mode="before",
    )(_strip_text_tuple)

    @model_validator(mode="after")
    def require_unique_profile_items(self) -> ExternalPersonV1:
        if len(self.disclosed_background) != len(set(self.disclosed_background)):
            raise ValueError("disclosed_background cannot contain duplicates")
        if len(self.persona_tags) != len(set(self.persona_tags)):
            raise ValueError("persona_tags cannot contain duplicates")
        return self


class ExternalEventLogRecordV1(_FrozenContractModel):
    """One visible synthetic utterance and its external context.

    JSONL inputs contain one instance per non-empty line.  JSON inputs may
    contain one instance, an array of instances, or an
    :class:`ExternalEventLogDocumentV1` envelope.
    """

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["external_visible_event"] = "external_visible_event"
    synthetic: Literal[True] = True
    source_system: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    source_episode_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    source_record_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    sequence_index: int = Field(ge=0)
    simulated_time: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT_LENGTH)
    person: ExternalPersonV1
    visible_events: tuple[str, ...] = Field(default=(), max_length=128)
    text: str = Field(min_length=1, max_length=_MAX_TEXT_LENGTH)

    _normalize_required = field_validator(
        "source_system",
        "source_episode_id",
        "source_record_id",
        "simulated_time",
        "text",
        mode="before",
    )(_strip_required_text)
    _normalize_events = field_validator("visible_events", mode="before")(
        _strip_text_tuple
    )

    @field_validator("synthetic", mode="before")
    @classmethod
    def require_json_boolean_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("synthetic must be the JSON boolean true")
        return value

    @field_validator("sequence_index", mode="before")
    @classmethod
    def require_json_integer(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("sequence_index must be a JSON integer")
        return value

    @model_validator(mode="after")
    def require_unique_events(self) -> ExternalEventLogRecordV1:
        if len(self.visible_events) != len(set(self.visible_events)):
            raise ValueError("visible_events cannot contain duplicates")
        return self


class ExternalEventLogDocumentV1(_FrozenContractModel):
    """JSON envelope for a collection of neutral external-log records."""

    schema_version: Literal["1.0"] = "1.0"
    document_type: Literal[
        "ecocompanion_external_visible_event_log"
    ] = "ecocompanion_external_visible_event_log"
    records: tuple[ExternalEventLogRecordV1, ...] = Field(min_length=1)


class ImportedObservationDraftV1(_FrozenContractModel):
    """One sanitized visible step awaiting expert annotation."""

    step_id: int = Field(ge=0)
    observation_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    source_event_ref: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    simulated_time: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT_LENGTH)
    visible_events: tuple[str, ...] = Field(default=(), max_length=128)
    utterance: str = Field(min_length=1, max_length=_MAX_TEXT_LENGTH)


class ImportProvenanceV1(_FrozenContractModel):
    """Non-identifying provenance for reproducibility and conflict checks."""

    source_system: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    source_episode_ref: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    source_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_record_count: int = Field(ge=1)


class ImportedCaseDraftV1(_FrozenContractModel):
    """Sanitized observable-only draft that must be relabeled by experts."""

    schema_version: Literal["1.0"] = "1.0"
    draft_type: Literal[
        "ecocompanion_expert_relabel_draft"
    ] = "ecocompanion_expert_relabel_draft"
    importer_version: Literal["external-log-import-v1"] = "external-log-import-v1"
    draft_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    synthetic: Literal[True] = True
    annotation_status: Literal[
        "PENDING_EXPERT_RELABEL"
    ] = "PENDING_EXPERT_RELABEL"
    benchmark_eligible: Literal[False] = False
    provenance: ImportProvenanceV1
    visible_profile: VisibleProfile
    persona_tags: tuple[str, ...] = Field(default=(), max_length=64)
    timeline: tuple[ImportedObservationDraftV1, ...] = Field(min_length=1)
    required_expert_annotations: tuple[
        RequiredExpertAnnotation, ...
    ] = REQUIRED_EXPERT_ANNOTATIONS

    @model_validator(mode="after")
    def enforce_draft_contract(self) -> ImportedCaseDraftV1:
        expected_steps = tuple(range(len(self.timeline)))
        actual_steps = tuple(step.step_id for step in self.timeline)
        if actual_steps != expected_steps:
            raise ValueError("draft timeline step_ids must be consecutive from zero")
        observation_ids = tuple(step.observation_id for step in self.timeline)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("draft observation_ids must be unique")
        if self.required_expert_annotations != REQUIRED_EXPERT_ANNOTATIONS:
            raise ValueError("required_expert_annotations cannot be changed")
        return self
