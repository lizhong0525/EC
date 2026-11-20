"""Guideline-derived partial references, never a substitute for expert Gold."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from ecocompanion.domain.enums import Action, EscalationTarget, RiskLevel
from ecocompanion.research.contracts import Record

REFERENCE_VERSION: Literal["guideline-reference-v1.0.0"] = "guideline-reference-v1.0.0"
NormField = Literal[
    "risk_level",
    "acceptable_actions",
    "preferred_action",
    "prohibited_actions",
    "direct_safety_check_required",
    "referral_target",
]


class ReferenceRecord(Record):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )


class PartialNorm(ReferenceRecord):
    """None means unsupported, not negative; no diagnostic probabilities or deadlines."""

    risk_level: RiskLevel | None = None
    acceptable_actions: tuple[Action, ...] | None = None
    preferred_action: Action | None = None
    prohibited_actions: tuple[Action, ...] | None = None
    direct_safety_check_required: bool | None = None
    referral_target: EscalationTarget | None = None

    @model_validator(mode="after")
    def sets_are_consistent(self) -> PartialNorm:
        for actions in (self.acceptable_actions, self.prohibited_actions):
            if actions is not None and (not actions or len(actions) != len(set(actions))):
                raise ValueError("provided action sets must be nonempty and unique")
        if set(self.acceptable_actions or ()) & set(self.prohibited_actions or ()):
            raise ValueError("acceptable and prohibited actions overlap")
        if self.preferred_action is not None and (
            self.acceptable_actions is None or self.preferred_action not in self.acceptable_actions
        ):
            raise ValueError("preferred action requires a supporting acceptable set")
        return self


class GuidelineSource(ReferenceRecord):
    source_id: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    edition: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    jurisdiction: str = Field(min_length=1)
    population_scope: str = Field(min_length=1)
    limitations: str = Field(min_length=1)


class GuidelineRule(ReferenceRecord):
    rule_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    source_anchor: str = Field(min_length=1, max_length=180)
    paraphrase: str = Field(min_length=1)
    conditions: tuple[str, ...] = Field(min_length=1)
    exclusions: tuple[str, ...] = Field(min_length=1)
    conclusion: PartialNorm
    transfer_limitations: str = Field(min_length=1)


class GuidelineCatalog(ReferenceRecord):
    version: Literal["guideline-reference-v1.0.0"] = REFERENCE_VERSION
    sources: tuple[GuidelineSource, ...] = Field(min_length=1)
    rules: tuple[GuidelineRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def links_are_unique(self) -> GuidelineCatalog:
        ids = [s.source_id for s in self.sources]
        rules = [r.rule_id for r in self.rules]
        if len(ids) != len(set(ids)) or len(rules) != len(set(rules)):
            raise ValueError("duplicate source or rule ID")
        if any(r.source_id not in ids for r in self.rules):
            raise ValueError("rule references an unknown source")
        return self


class RuleReview(ReferenceRecord):
    rule_id: str
    reviewer_id: str = Field(min_length=1)
    verification_reference: str = Field(min_length=1)
    source_and_scope_checked: Literal[True]
    disposition: Literal["allow_silver_mapping", "block"]
    reason: str = Field(min_length=1, max_length=1000)


class EvidenceSpan(ReferenceRecord):
    observation_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=1000)


class RuleApplication(ReferenceRecord):
    rule_id: str
    # Every catalog condition and exclusion must be addressed explicitly.
    conditions: dict[str, Literal["met", "not_met", "unknown"]]
    exclusions: dict[str, Literal["present", "absent", "unknown"]]
    fields: tuple[NormField, ...] = Field(min_length=1)
    applicability_reason: str = Field(min_length=1, max_length=1000)


class GuidelineMapping(ReferenceRecord):
    author_id: str = Field(min_length=1)
    origin: Literal["human_curator", "ai_draft"]
    case_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    step_id: int = Field(ge=0)
    observation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolution: Literal["mapped", "unresolved"]
    norm: PartialNorm = Field(default_factory=PartialNorm)
    applications: tuple[RuleApplication, ...] = ()
    evidence: tuple[EvidenceSpan, ...] = ()
    missing_information: tuple[str, ...] = ()
    reason: str = Field(min_length=1, max_length=1000)
    supersedes: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def no_invented_labels(self) -> GuidelineMapping:
        fields = self.norm.model_dump(exclude_none=True)
        if self.resolution == "mapped" and not (fields and self.applications and self.evidence):
            raise ValueError("mapped reference requires partial labels, rules and visible evidence")
        if self.resolution == "unresolved" and (fields or self.applications):
            raise ValueError("unresolved records must not carry scored normative labels")
        if len({a.rule_id for a in self.applications}) != len(self.applications):
            raise ValueError("duplicate rule applications")
        return self


class MappingReview(ReferenceRecord):
    mapping_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1)
    verification_reference: str = Field(min_length=1)
    disposition: Literal["confirm", "disagree"]
    reason: str = Field(min_length=1, max_length=1000)
