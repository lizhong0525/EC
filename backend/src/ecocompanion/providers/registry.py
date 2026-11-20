"""Runtime model discovery, smoke testing, and role-separation checks."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ecocompanion.providers.client import OpenAICompatibleClient
from ecocompanion.providers.errors import ProviderError
from ecocompanion.providers.qualification import QualifiedModelSnapshot
from ecocompanion.providers.types import (
    CapabilityStatus,
    ModelCapabilityReport,
    ModelDescriptor,
    ModelRegistrySnapshot,
    ModelRole,
    RoleBinding,
    RoleSeparationReport,
)


class _SmokePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class ModelCapabilityProbe:
    """Low-cost eligibility probe; it is not a quality benchmark."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._client = client
        self._max_concurrency = max_concurrency

    async def probe_model(self, model_id: str) -> ModelCapabilityReport:
        failures: list[str] = []
        try:
            basic = await self._client.chat_completion(
                model=model_id,
                messages=(
                    {
                        "role": "system",
                        "content": "这是接口资格测试。请遵循后续消息。",
                    },
                    {"role": "user", "content": "第一轮，请回复任意简短中文。"},
                    {"role": "assistant", "content": "收到。"},
                    {"role": "user", "content": "第二轮，请回复“可用”。"},
                ),
                max_tokens=64,
                temperature=0.0,
            )
        except ProviderError as exc:
            status = (
                CapabilityStatus.UNKNOWN if exc.retryable else CapabilityStatus.INELIGIBLE
            )
            return ModelCapabilityReport(
                model_id=model_id,
                status=status,
                basic_chat=False,
                system_and_multiturn=False,
                chinese_text=False,
                usage_reported=False,
                preferred_structured_mode=None,
                streaming=False,
                failures=(f"basic_chat:{exc.category}",),
            )

        basic_content = (basic.content or "").strip()
        if not basic_content:
            return ModelCapabilityReport(
                model_id=model_id,
                status=CapabilityStatus.INELIGIBLE,
                basic_chat=False,
                system_and_multiturn=False,
                chinese_text=False,
                usage_reported=False,
                preferred_structured_mode=None,
                streaming=False,
                failures=("basic_chat:empty_content",),
            )
        chinese_text = any(
            "\u3400" <= character <= "\u9fff"
            for character in basic_content
        )
        system_and_multiturn = "可用" in basic_content
        if not chinese_text or not system_and_multiturn:
            if not chinese_text:
                failures.append("chinese_text:missing_cjk")
            if not system_and_multiturn:
                failures.append("system_and_multiturn:instruction_not_followed")
            return ModelCapabilityReport(
                model_id=model_id,
                status=CapabilityStatus.INELIGIBLE,
                basic_chat=True,
                system_and_multiturn=system_and_multiturn,
                chinese_text=chinese_text,
                usage_reported=(
                    basic.usage.input_tokens > 0
                    or basic.usage.output_tokens > 0
                ),
                preferred_structured_mode=None,
                streaming=False,
                failures=tuple(failures),
            )

        usage_reported = basic.usage.input_tokens > 0 or basic.usage.output_tokens > 0
        streaming = False
        try:
            streamed = await self._client.stream_chat_completion(
                model=model_id,
                messages=(
                    {
                        "role": "system",
                        "content": "这是流式接口资格测试。",
                    },
                    {"role": "user", "content": "请回复“流式可用”。"},
                ),
                max_tokens=128,
                temperature=0.0,
            )
            streamed_content = (streamed.content or "").strip()
            streaming = bool(streamed_content) and any(
                "\u3400" <= character <= "\u9fff"
                for character in streamed_content
            )
            usage_reported = usage_reported or (
                streamed.usage.input_tokens > 0
                or streamed.usage.output_tokens > 0
            )
        except ProviderError as exc:
            failures.append(f"streaming:{exc.category}")

        try:
            structured = await self._client.complete_structured(
                model=model_id,
                messages=(
                    {
                        "role": "system",
                        "content": "Return the required object and no other content.",
                    },
                    {"role": "user", "content": 'Return {"status":"ok"}.'},
                ),
                schema=_SmokePayload,
                max_tokens=256,
                temperature=0.0,
            )
        except ProviderError as exc:
            failures.append(f"structured_output:{exc.category}")
            status = (
                CapabilityStatus.UNKNOWN if exc.retryable else CapabilityStatus.DEGRADED
            )
            return ModelCapabilityReport(
                model_id=model_id,
                status=status,
                basic_chat=True,
                system_and_multiturn=True,
                chinese_text=True,
                usage_reported=usage_reported,
                preferred_structured_mode=None,
                streaming=streaming,
                eligible_roles=frozenset(),
                failures=tuple(failures),
            )

        usage_reported = usage_reported or (
            structured.completion.usage.input_tokens > 0
            or structured.completion.usage.output_tokens > 0
        )
        status = (
            CapabilityStatus.READY
            if streaming
            else CapabilityStatus.DEGRADED
        )
        if not streaming:
            failures.append("streaming:unavailable")
        return ModelCapabilityReport(
            model_id=model_id,
            status=status,
            basic_chat=True,
            system_and_multiturn=True,
            chinese_text=True,
            usage_reported=usage_reported,
            preferred_structured_mode=structured.mode,
            streaming=streaming,
            eligible_roles=(
                frozenset(ModelRole) if streaming else frozenset()
            ),
            failures=tuple(failures),
        )

    async def probe_all(
        self,
        models: Sequence[ModelDescriptor],
    ) -> tuple[ModelCapabilityReport, ...]:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def bounded_probe(model: ModelDescriptor) -> ModelCapabilityReport:
            async with semaphore:
                return await self.probe_model(model.model_id)

        reports = await asyncio.gather(*(bounded_probe(model) for model in models))
        return tuple(sorted(reports, key=lambda report: report.model_id))


class ModelRegistry:
    """Query object that returns one immutable discovery/capability snapshot."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        probe: ModelCapabilityProbe | None = None,
    ) -> None:
        self._client = client
        self._probe = probe or ModelCapabilityProbe(client)

    async def refresh(self, *, run_smoke_tests: bool = True) -> ModelRegistrySnapshot:
        models, authenticated = await self._client.discover_models()
        capabilities = (
            await self._probe.probe_all(models) if run_smoke_tests else ()
        )
        return ModelRegistrySnapshot(
            models=models,
            capabilities=capabilities,
            authenticated_discovery=authenticated,
        )


@dataclass(frozen=True, slots=True)
class ModelRoleRecommendation:
    """Deterministic role suggestions derived only from a frozen snapshot."""

    companion_model: str | None
    comparison_models: tuple[str, ...]
    student_renderer_model: str | None
    judge_models: tuple[str, ...]
    distinct_family_requirement_met: bool
    family_separation_status: FamilySeparationStatus = "degraded"
    expert_evaluation_required: bool = True
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "companion_model": self.companion_model,
            "comparison_models": list(self.comparison_models),
            "student_renderer_model": self.student_renderer_model,
            "judge_models": list(self.judge_models),
            "distinct_family_requirement_met": self.distinct_family_requirement_met,
            "family_separation_status": self.family_separation_status,
            "expert_evaluation_required": self.expert_evaluation_required,
            "warnings": list(self.warnings),
        }


FamilySeparationStatus = Literal["full", "degraded"]


@dataclass(frozen=True, slots=True)
class FamilySeparationAssessment:
    """Auditable family-independence assessment for one role assignment.

    A ``degraded`` assignment is still usable when exact model IDs remain
    distinct, but automated judge independence cannot be claimed.  Callers
    should surface ``warnings`` and make expert evaluation primary.
    """

    status: FamilySeparationStatus
    warnings: tuple[str, ...] = ()
    expert_evaluation_required: bool = False

    def __post_init__(self) -> None:
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("family separation warnings cannot contain duplicates")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("family separation warnings cannot contain blanks")
        expected = self.status == "degraded"
        if self.expert_evaluation_required != expected:
            raise ValueError(
                "expert_evaluation_required must match family separation status"
            )


def assess_family_separation(
    *,
    evaluated_agent: RoleBinding,
    student_simulator: RoleBinding,
    judges: Sequence[RoleBinding],
) -> FamilySeparationAssessment:
    """Classify family separation without rejecting a degraded assignment."""

    bindings = (evaluated_agent, student_simulator, *judges)
    warnings: list[str] = []
    families = tuple(binding.family for binding in bindings)
    known_families = tuple(family for family in families if family)

    if len(known_families) != len(families):
        warnings.append("one or more model families are unknown")

    evaluated_family = evaluated_agent.family
    student_family = student_simulator.family
    if (
        evaluated_family
        and student_family
        and evaluated_family == student_family
    ):
        warnings.append("student shares companion family")

    known_judge_families = tuple(judge.family for judge in judges if judge.family)
    if len(known_judge_families) != len(set(known_judge_families)):
        warnings.append("judge panel shares a family")

    for judge in judges:
        if evaluated_family and judge.family == evaluated_family:
            warnings.append("judge shares companion/student family")
            break
    for judge in judges:
        if student_family and judge.family == student_family:
            warnings.append("judge shares companion/student family")
            break

    if len(set(known_families)) < 2:
        warnings.append(
            "fewer than two independently verified families; "
            "expert ratings are primary"
        )

    full = (
        len(bindings) == 4
        and all(family is not None for family in families)
        and len(set(families)) == len(families)
    )
    status: FamilySeparationStatus = "full" if full else "degraded"
    return FamilySeparationAssessment(
        status=status,
        warnings=tuple(dict.fromkeys(warnings)),
        expert_evaluation_required=status == "degraded",
    )


def recommend_model_roles(
    snapshot: QualifiedModelSnapshot,
    *,
    comparison_limit: int = 4,
) -> ModelRoleRecommendation:
    """Recommend four distinct model IDs with explicit family degradation.

    Family separation is preferred, but it is not allowed to make a usable
    four-model snapshot impossible.  The returned warning is the audit record
    that tells downstream callers when expert ratings must be primary.
    """

    if comparison_limit <= 0:
        raise ValueError("comparison_limit must be positive")
    ready_ids = {
        report.model_id
        for report in snapshot.capabilities
        if report.status is CapabilityStatus.READY
    }
    qualifications = {
        result.model_id: result
        for result in snapshot.qualifications
        if result.model_id in ready_ids
    }
    descriptors = {
        model.model_id: model
        for model in snapshot.models
        if model.model_id in qualifications
    }
    ranked = sorted(
        descriptors,
        key=lambda model_id: (
            -qualifications[model_id].qualification_score,
            model_id,
        ),
    )
    if not ranked:
        return ModelRoleRecommendation(
            companion_model=None,
            comparison_models=(),
            student_renderer_model=None,
            judge_models=(),
            distinct_family_requirement_met=False,
            family_separation_status="degraded",
            expert_evaluation_required=True,
            warnings=("no ready model completed the qualification suite",),
        )

    companion = ranked[0]
    comparisons = _prefer_distinct_families(
        ranked,
        descriptors=descriptors,
        limit=comparison_limit,
    )
    companion_family = descriptors[companion].declared_owner
    simulator_candidates = [
        model_id
        for model_id in ranked
        if model_id != companion
        and descriptors[model_id].declared_owner
        and descriptors[model_id].declared_owner != companion_family
    ]
    if not simulator_candidates:
        # Family undersupply is a supported, explicit fallback.  Keep the
        # lower-cost preference used by the fully separated path.
        simulator_candidates = [
            model_id for model_id in ranked if model_id != companion
        ]
    simulator = min(
        simulator_candidates,
        key=lambda model_id: (
            qualifications[model_id].mean_cost_cny is None,
            qualifications[model_id].mean_cost_cny
            if qualifications[model_id].mean_cost_cny is not None
            else float("inf"),
            -qualifications[model_id].qualification_score,
            model_id,
        ),
        default=None,
    )

    excluded_ids = {companion}
    excluded_families = {companion_family} if companion_family else set()
    if simulator is not None:
        excluded_ids.add(simulator)
        simulator_family = descriptors[simulator].declared_owner
        if simulator_family:
            excluded_families.add(simulator_family)
    judges: list[str] = []
    judge_families: set[str] = set()

    # First spend distinct, known families that do not overlap the two
    # evaluated roles.  The second pass below fills the panel when the
    # snapshot does not contain enough independent families.
    for model_id in ranked:
        family = descriptors[model_id].declared_owner
        if (
            model_id in excluded_ids
            or not family
            or family in excluded_families
            or family in judge_families
        ):
            continue
        judges.append(model_id)
        judge_families.add(family)
        if len(judges) == 2:
            break

    if len(judges) < 2:
        for model_id in ranked:
            if model_id in excluded_ids or model_id in judges:
                continue
            judges.append(model_id)
            if len(judges) == 2:
                break

    role_ids = (companion, simulator, *judges) if simulator is not None else ()
    role_bindings = tuple(
        RoleBinding(
            role=(
                ModelRole.EVALUATED_AGENT
                if index == 0
                else (
                    ModelRole.STUDENT_SIMULATOR
                    if index == 1
                    else ModelRole.JUDGE
                )
            ),
            model_id=model_id,
            family=descriptors[model_id].declared_owner,
        )
        for index, model_id in enumerate(role_ids)
    )
    assessment = (
        assess_family_separation(
            evaluated_agent=role_bindings[0],
            student_simulator=role_bindings[1],
            judges=role_bindings[2:],
        )
        if len(role_bindings) == 4
        else FamilySeparationAssessment(
            status="degraded",
            warnings=(
                "fewer than four distinct qualified model IDs are available",
            ),
            expert_evaluation_required=True,
        )
    )
    warnings = list(assessment.warnings)
    if simulator is None:
        warnings.append("no qualified student renderer is available")
    if len(judges) < 2:
        warnings.append("fewer than two qualified judge models are available")
    requirement_met = assessment.status == "full"
    return ModelRoleRecommendation(
        companion_model=companion,
        comparison_models=comparisons,
        student_renderer_model=simulator,
        judge_models=tuple(judges),
        distinct_family_requirement_met=requirement_met,
        family_separation_status=assessment.status,
        expert_evaluation_required=assessment.expert_evaluation_required,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _prefer_distinct_families(
    ranked_model_ids: Sequence[str],
    *,
    descriptors: dict[str, ModelDescriptor],
    limit: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    selected_families: set[str] = set()
    for model_id in ranked_model_ids:
        family = descriptors[model_id].declared_owner
        if family and family not in selected_families:
            selected.append(model_id)
            selected_families.add(family)
        if len(selected) == limit:
            return tuple(selected)
    for model_id in ranked_model_ids:
        if model_id not in selected:
            selected.append(model_id)
        if len(selected) == limit:
            break
    return tuple(selected)


def validate_role_separation(
    *,
    evaluated_agent: RoleBinding,
    student_simulator: RoleBinding,
    judges: Sequence[RoleBinding],
    require_known_families: bool = True,
    require_two_judges: bool = True,
    allow_family_overlap: bool = False,
) -> RoleSeparationReport:
    """Validate model-source separation for one evaluated-agent condition."""

    bindings = (evaluated_agent, student_simulator, *judges)
    violations: list[str] = []
    warnings: list[str] = []

    if evaluated_agent.role is not ModelRole.EVALUATED_AGENT:
        violations.append("evaluated_agent binding has the wrong role")
    if student_simulator.role is not ModelRole.STUDENT_SIMULATOR:
        violations.append("student_simulator binding has the wrong role")
    if any(judge.role is not ModelRole.JUDGE for judge in judges):
        violations.append("all judge bindings must use the judge role")
    if require_two_judges and len(judges) < 2:
        violations.append("research evaluation requires at least two judges")

    ids = [binding.model_id for binding in bindings]
    if len(ids) != len(set(ids)):
        violations.append("the same exact model cannot occupy multiple roles")

    unknown_roles = [binding.role.value for binding in bindings if not binding.family]
    if unknown_roles:
        message = f"unknown model family for roles: {sorted(unknown_roles)}"
        if require_known_families:
            violations.append(message)
        else:
            warnings.append(message)

    known_judge_families = [judge.family for judge in judges if judge.family]
    if len(known_judge_families) != len(set(known_judge_families)):
        message = "judge panel must use distinct known model families"
        (warnings if allow_family_overlap else violations).append(message)

    evaluated_family = evaluated_agent.family
    simulator_family = student_simulator.family
    if evaluated_family and simulator_family and evaluated_family == simulator_family:
        message = "student simulator shares the evaluated-agent model family"
        (warnings if allow_family_overlap else violations).append(message)
    for judge in judges:
        if evaluated_family and judge.family == evaluated_family:
            message = (
                f"judge {judge.model_id} shares the evaluated-agent model family"
            )
            (warnings if allow_family_overlap else violations).append(message)
        if simulator_family and judge.family == simulator_family:
            message = (
                f"judge {judge.model_id} shares the student-simulator model family"
            )
            (warnings if allow_family_overlap else violations).append(message)

    return RoleSeparationReport(
        valid=not violations,
        violations=tuple(dict.fromkeys(violations)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
