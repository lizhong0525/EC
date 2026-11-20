"""Versioned expert-annotation contracts and agreement utilities.

The utilities in this module validate human annotations against observable
case data. They never derive "expert" labels from hidden simulator state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from ecocompanion.benchmark.cases import validate_cases
from ecocompanion.domain.enums import Action, EscalationTarget, RiskLevel
from ecocompanion.domain.schemas import CaseSpecV1, EvidenceRef


class _FrozenAnnotationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnnotationRole(StrEnum):
    INDEPENDENT = "independent"
    ADJUDICATOR = "adjudicator"


class NodeAnnotationV1(_FrozenAnnotationModel):
    """One human judgment for one case-decision node."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_version: str = "ecotriage-v1"
    case_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    annotator_id: str = Field(min_length=1)
    role: AnnotationRole = AnnotationRole.INDEPENDENT
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[EvidenceRef, ...] = ()
    acceptable_actions: tuple[Action, ...]
    preferred_action: Action
    prohibited_actions: tuple[Action, ...] = ()
    direct_safety_check_required: bool
    referral_target: EscalationTarget = EscalationTarget.NONE
    earliest_referral_step: int | None = Field(default=None, ge=0)
    optimal_referral_start_step: int | None = Field(default=None, ge=0)
    optimal_referral_end_step: int | None = Field(default=None, ge=0)
    latest_safe_referral_step: int | None = Field(default=None, ge=0)
    reason_codes: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_annotation_contract(self) -> NodeAnnotationV1:
        if len(self.acceptable_actions) != len(set(self.acceptable_actions)):
            raise ValueError("acceptable_actions cannot contain duplicates")
        if len(self.prohibited_actions) != len(set(self.prohibited_actions)):
            raise ValueError("prohibited_actions cannot contain duplicates")
        if self.preferred_action not in self.acceptable_actions:
            raise ValueError("preferred_action must be acceptable")
        overlap = set(self.acceptable_actions) & set(self.prohibited_actions)
        if overlap:
            raise ValueError("acceptable_actions and prohibited_actions cannot overlap")
        evidence_keys = tuple((item.observation_id, item.evidence_code) for item in self.evidence)
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("evidence references cannot contain duplicates")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes cannot contain duplicates")

        window = (
            self.earliest_referral_step,
            self.optimal_referral_start_step,
            self.optimal_referral_end_step,
            self.latest_safe_referral_step,
        )
        populated = tuple(value is not None for value in window)
        if any(populated) and not all(populated):
            raise ValueError("referral window must provide all four boundaries")
        if all(populated):
            numeric = tuple(int(value) for value in window if value is not None)
            if numeric != tuple(sorted(numeric)):
                raise ValueError("referral window boundaries must be non-decreasing")
        return self


class AnnotationBatchV1(_FrozenAnnotationModel):
    """Write-once annotation task result from one human role."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_version: str = "ecotriage-v1"
    manual_version: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    annotations: tuple[NodeAnnotationV1, ...]

    @model_validator(mode="after")
    def require_unique_nodes(self) -> AnnotationBatchV1:
        if not self.annotations:
            raise ValueError("annotation batch cannot be empty")
        keys = tuple((item.case_id, item.step_id, item.annotator_id) for item in self.annotations)
        if len(keys) != len(set(keys)):
            raise ValueError("annotation batch contains duplicate node judgments")
        if any(item.benchmark_version != self.benchmark_version for item in self.annotations):
            raise ValueError("annotation benchmark versions must match the batch")
        return self


class AnnotationAgreement(_FrozenAnnotationModel):
    """Paired agreement summary for two independent annotation batches."""

    paired_node_count: int = Field(gt=0)
    crisis_kappa: float | None
    ordered_risk_weighted_kappa: float | None
    boundary_raw_agreement: float | None
    action_set_mean_jaccard: float
    preferred_action_kappa: float | None
    preferred_action_raw_agreement: float
    adjudication_node_count: int = Field(ge=0)


def validate_annotation_batch(
    batch: AnnotationBatchV1,
    cases: Sequence[CaseSpecV1],
) -> AnnotationBatchV1:
    """Validate references using only information observable by each node."""

    validated_cases = validate_cases(cases)
    case_by_id = {case.case_id: case for case in validated_cases}
    benchmark_versions = {case.benchmark_version for case in validated_cases}
    if benchmark_versions != {batch.benchmark_version}:
        raise ValueError("annotation batch does not match the case benchmark version")

    for annotation in batch.annotations:
        case = case_by_id.get(annotation.case_id)
        if case is None:
            raise ValueError(f"unknown annotation case_id: {annotation.case_id}")
        if annotation.step_id >= len(case.timeline):
            raise ValueError(f"{annotation.case_id}: step_id {annotation.step_id} is out of range")
        visible_steps = case.timeline[: annotation.step_id + 1]
        known_observation_ids = {step.observation_id for step in visible_steps}
        unknown_observations = {
            evidence.observation_id
            for evidence in annotation.evidence
            if evidence.observation_id not in known_observation_ids
        }
        if unknown_observations:
            raise ValueError(
                f"{annotation.case_id}: evidence cites future or unknown observations: "
                f"{sorted(unknown_observations)}"
            )
        facts = {fact.fact_id: fact for fact in case.disclosure_facts}
        observable_pairs = {
            (step.observation_id, facts[fact_id].evidence_code)
            for step in visible_steps
            for fact_id in step.available_fact_ids
        }
        unknown_pairs = {
            (evidence.observation_id, evidence.evidence_code)
            for evidence in annotation.evidence
            if (
                evidence.observation_id,
                evidence.evidence_code,
            )
            not in observable_pairs
        }
        if unknown_pairs:
            raise ValueError(
                f"{annotation.case_id}: evidence observation/code pairs are "
                f"not observable: {sorted(unknown_pairs)}"
            )
        window_values = (
            annotation.earliest_referral_step,
            annotation.optimal_referral_start_step,
            annotation.optimal_referral_end_step,
            annotation.latest_safe_referral_step,
        )
        if any(value is not None and value >= len(case.timeline) for value in window_values):
            raise ValueError(f"{annotation.case_id}: referral window exceeds the case timeline")
    return batch


def annotation_agreement(
    left: AnnotationBatchV1,
    right: AnnotationBatchV1,
) -> AnnotationAgreement:
    """Compute paired agreement without treating surface variants as samples."""

    left_by_node = _independent_node_map(left)
    right_by_node = _independent_node_map(right)
    if set(left_by_node) != set(right_by_node):
        missing_left = sorted(set(right_by_node) - set(left_by_node))
        missing_right = sorted(set(left_by_node) - set(right_by_node))
        raise ValueError(
            "annotation batches must cover identical nodes; "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )
    keys = sorted(left_by_node)
    left_items = [left_by_node[key] for key in keys]
    right_items = [right_by_node[key] for key in keys]

    crisis_left = [item.risk_level is RiskLevel.L4 for item in left_items]
    crisis_right = [item.risk_level is RiskLevel.L4 for item in right_items]
    ordered_pairs = [
        (left_item.risk_level, right_item.risk_level)
        for left_item, right_item in zip(left_items, right_items, strict=True)
        if left_item.risk_level is not RiskLevel.B and right_item.risk_level is not RiskLevel.B
    ]
    risk_order = {
        RiskLevel.L0: 0,
        RiskLevel.L1: 1,
        RiskLevel.L2: 2,
        RiskLevel.L3: 3,
        RiskLevel.L4: 4,
    }
    boundary_pairs = [
        (
            left_item.risk_level is RiskLevel.B,
            right_item.risk_level is RiskLevel.B,
        )
        for left_item, right_item in zip(left_items, right_items, strict=True)
        if left_item.risk_level is RiskLevel.B or right_item.risk_level is RiskLevel.B
    ]
    action_jaccards = [
        _jaccard(
            set(left_item.acceptable_actions),
            set(right_item.acceptable_actions),
        )
        for left_item, right_item in zip(left_items, right_items, strict=True)
    ]
    preferred_left = [item.preferred_action.value for item in left_items]
    preferred_right = [item.preferred_action.value for item in right_items]
    adjudication_nodes = find_adjudication_nodes(left, right)
    return AnnotationAgreement(
        paired_node_count=len(keys),
        crisis_kappa=_safe_kappa(crisis_left, crisis_right),
        ordered_risk_weighted_kappa=(
            _safe_kappa(
                [risk_order[left_level] for left_level, _ in ordered_pairs],
                [risk_order[right_level] for _, right_level in ordered_pairs],
                weights="quadratic",
            )
            if ordered_pairs
            else None
        ),
        boundary_raw_agreement=(
            sum(left_value == right_value for left_value, right_value in boundary_pairs)
            / len(boundary_pairs)
            if boundary_pairs
            else None
        ),
        action_set_mean_jaccard=sum(action_jaccards) / len(action_jaccards),
        preferred_action_kappa=_safe_kappa(preferred_left, preferred_right),
        preferred_action_raw_agreement=(
            sum(
                left_value == right_value
                for left_value, right_value in zip(
                    preferred_left,
                    preferred_right,
                    strict=True,
                )
            )
            / len(preferred_left)
        ),
        adjudication_node_count=len(adjudication_nodes),
    )


def find_adjudication_nodes(
    left: AnnotationBatchV1,
    right: AnnotationBatchV1,
) -> tuple[tuple[str, int], ...]:
    """Return every high-risk node and every node with a material disagreement."""

    left_by_node = _independent_node_map(left)
    right_by_node = _independent_node_map(right)
    if set(left_by_node) != set(right_by_node):
        raise ValueError("annotation batches must cover identical nodes")
    selected: list[tuple[str, int]] = []
    for key in sorted(left_by_node):
        left_item = left_by_node[key]
        right_item = right_by_node[key]
        high_risk = any(
            item.risk_level in {RiskLevel.L3, RiskLevel.L4} for item in (left_item, right_item)
        )
        disagreement = any(
            (
                left_item.risk_level is not right_item.risk_level,
                set(left_item.acceptable_actions) != set(right_item.acceptable_actions),
                left_item.preferred_action is not right_item.preferred_action,
                set(left_item.prohibited_actions) != set(right_item.prohibited_actions),
                left_item.direct_safety_check_required != right_item.direct_safety_check_required,
                left_item.referral_target is not right_item.referral_target,
            )
        )
        if high_risk or disagreement:
            selected.append(key)
    return tuple(selected)


def _independent_node_map(
    batch: AnnotationBatchV1,
) -> dict[tuple[str, int], NodeAnnotationV1]:
    if any(item.role is not AnnotationRole.INDEPENDENT for item in batch.annotations):
        raise ValueError("agreement requires independent annotation batches")
    by_node = {(item.case_id, item.step_id): item for item in batch.annotations}
    if len(by_node) != len(batch.annotations):
        raise ValueError("each batch must contain one annotation per node")
    return by_node


def _safe_kappa(
    left: Sequence[object],
    right: Sequence[object],
    *,
    weights: Literal["linear", "quadratic"] | None = None,
) -> float | None:
    if not left:
        return None
    value = float(cohen_kappa_score(left, right, weights=weights))
    return value if math.isfinite(value) else None


def _jaccard(left: set[Action], right: set[Action]) -> float:
    union = left | right
    if not union:
        raise ValueError("acceptable action sets cannot both be empty")
    return len(left & right) / len(union)
