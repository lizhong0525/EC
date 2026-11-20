"""Formal API runtime selection and lifecycle for interactive benchmarks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ecocompanion.benchmark.api_factory import (
    InteractiveAPIComponents,
    InteractiveAPIConfig,
    build_interactive_api_components,
)
from ecocompanion.benchmark.api_renderers import (
    api_renderer_prompt_identities,
)
from ecocompanion.benchmark.judging import judge_prompt_identity
from ecocompanion.benchmark.prompt_identity import PromptIdentity
from ecocompanion.benchmark.track_b import TemplateStudentRenderer
from ecocompanion.benchmark.track_c import (
    DeterministicTrackCBaselineResponder,
)
from ecocompanion.config import Settings
from ecocompanion.experiments.io import read_json_object
from ecocompanion.infrastructure import ModelPricing
from ecocompanion.providers.client import OpenAICompatibleClient
from ecocompanion.providers.recorded_gateway import (
    InteractiveModelBinding,
    OpenAICompatibleSyncGateway,
    SyncCompletionGateway,
)
from ecocompanion.providers.registry import (
    assess_family_separation,
    validate_role_separation,
)
from ecocompanion.providers.types import ModelRole, RoleBinding

if TYPE_CHECKING:
    from ecocompanion.benchmark.model_provider import PolicyRuntime

INTERACTIVE_CACHE_NAME = "api_cache.sqlite3"
INTERACTIVE_COST_LEDGER_NAME = "api_cost_ledger.jsonl"
INTERACTIVE_TRACE_NAME = "api_trace.jsonl"
LOCAL_INTERACTIVE_MODE = "local"
FORMAL_API_INTERACTIVE_MODE = "formal_api"
MAX_API_BUDGET_CNY = Decimal("5000")
QUALIFICATION_SUITE_VERSION = "zh-safety-micro-v1"
INTERACTIVE_GATEWAY_NAME = "openai-compatible"
INTERACTIVE_GATEWAY_VERSION = "sync-bridge-v1"
INTERACTIVE_ADAPTER_NAME = "ecocompanion-interactive-api"
INTERACTIVE_ADAPTER_VERSION = "1.1"
INTERACTIVE_DECODER_VERSION = "recorded-model-invoker-v1"
INTERACTIVE_CACHE_SCHEMA_VERSION = "exact-cache-v1"
INTERACTIVE_TRACE_SCHEMA_VERSION = "trace-event-v1"


@dataclass(frozen=True, slots=True)
class FormalInteractiveSelection:
    """Validated, secret-free inputs selected from immutable artifacts."""

    model_snapshot_path: Path
    model_snapshot_digest: str
    pricing_catalog_path: Path
    pricing_catalog_digest: str
    pricing_version: str
    companion: InteractiveModelBinding
    student_renderer: InteractiveModelBinding
    judges: tuple[InteractiveModelBinding, InteractiveModelBinding]
    prompt_identities: tuple[PromptIdentity, ...]
    api_budget_cny: Decimal = MAX_API_BUDGET_CNY
    qualification_suite_version: str = QUALIFICATION_SUITE_VERSION
    gateway_name: str = INTERACTIVE_GATEWAY_NAME
    gateway_version: str = INTERACTIVE_GATEWAY_VERSION
    adapter_name: str = INTERACTIVE_ADAPTER_NAME
    adapter_version: str = INTERACTIVE_ADAPTER_VERSION
    decoder_version: str = INTERACTIVE_DECODER_VERSION
    cache_schema_version: str = INTERACTIVE_CACHE_SCHEMA_VERSION
    trace_schema_version: str = INTERACTIVE_TRACE_SCHEMA_VERSION
    family_separation_status: Literal["full", "degraded"] = "full"
    family_warnings: tuple[str, ...] = ()
    expert_evaluation_required: bool = False

    def __post_init__(self) -> None:
        if not self.model_snapshot_path.is_file():
            raise FileNotFoundError(self.model_snapshot_path)
        if not self.pricing_catalog_path.is_file():
            raise FileNotFoundError(self.pricing_catalog_path)
        _validate_sha256(self.model_snapshot_digest, "model_snapshot_digest")
        _validate_sha256(self.pricing_catalog_digest, "pricing_catalog_digest")
        if not self.pricing_version.strip():
            raise ValueError("pricing_version cannot be blank")
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
        if any(not value.strip() for value in version_fields):
            raise ValueError("formal interactive version fields cannot be blank")
        if self.api_budget_cny <= 0 or self.api_budget_cny > MAX_API_BUDGET_CNY:
            raise ValueError("api_budget_cny must be in (0, 5000]")
        bindings = (self.companion, self.student_renderer, *self.judges)
        missing_prices = sorted(
            binding.model_id for binding in bindings if binding.pricing is None
        )
        if missing_prices:
            raise ValueError(
                f"formal API mode requires explicit prices for models: {missing_prices}"
            )
        if len({binding.model_id for binding in bindings}) != len(bindings):
            raise ValueError("formal API roles must use distinct model IDs")
        report = validate_role_separation(
            evaluated_agent=_role_binding(self.companion),
            student_simulator=_role_binding(self.student_renderer),
            judges=tuple(_role_binding(judge) for judge in self.judges),
            require_known_families=self.family_separation_status == "full",
            require_two_judges=True,
            allow_family_overlap=self.family_separation_status == "degraded",
        )
        if not report.valid:
            raise ValueError(
                "formal interactive role separation failed: "
                + "; ".join(report.violations)
            )
        assessment = assess_family_separation(
            evaluated_agent=_role_binding(self.companion),
            student_simulator=_role_binding(self.student_renderer),
            judges=tuple(_role_binding(judge) for judge in self.judges),
        )
        if assessment.status != self.family_separation_status:
            raise ValueError(
                "family_separation_status does not match selected role families"
            )
        if tuple(self.family_warnings) != assessment.warnings:
            raise ValueError(
                "family_warnings do not match selected role family assessment"
            )
        if self.expert_evaluation_required != assessment.expert_evaluation_required:
            raise ValueError(
                "expert_evaluation_required does not match selected role families"
            )
        if len(self.family_warnings) != len(set(self.family_warnings)):
            raise ValueError("family_warnings cannot contain duplicates")
        if any(not warning.strip() for warning in self.family_warnings):
            raise ValueError("family_warnings cannot contain blanks")
        components = tuple(identity.component for identity in self.prompt_identities)
        if len(components) != len(set(components)):
            raise ValueError("prompt identities cannot contain duplicate components")

    @property
    def bindings(self) -> tuple[InteractiveModelBinding, ...]:
        return (self.companion, self.student_renderer, *self.judges)

    def identity_payload(self) -> dict[str, object]:
        """Return the exact manifest payload used for start/resume validation."""

        return {
            "mode": FORMAL_API_INTERACTIVE_MODE,
            "renderer_name": (
                f"api-student-renderer:{self.student_renderer.model_id}"
            ),
            "responder_name": f"api-track-c-responder:{self.companion.model_id}",
            "role_bindings": [
                {
                    "role": binding.role.value,
                    "model_id": binding.model_id,
                    "family": binding.family,
                    "allowed_response_model_ids": sorted(
                        binding.allowed_response_model_ids
                    ),
                }
                for binding in self.bindings
            ],
            "model_snapshot_digest": self.model_snapshot_digest,
            "pricing_version": self.pricing_version,
            "pricing_catalog_digest": self.pricing_catalog_digest,
            "qualification_suite_version": self.qualification_suite_version,
            "gateway_name": self.gateway_name,
            "gateway_version": self.gateway_version,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "decoder_version": self.decoder_version,
            "cache_schema_version": self.cache_schema_version,
            "trace_schema_version": self.trace_schema_version,
            "family_separation_status": self.family_separation_status,
            "family_warnings": list(self.family_warnings),
            "expert_evaluation_required": self.expert_evaluation_required,
            "prompt_identities": [
                identity.to_dict() for identity in self.prompt_identities
            ],
            "api_budget_cny": str(self.api_budget_cny),
        }


@dataclass(frozen=True, slots=True)
class _QualifiedCandidate:
    model_id: str
    family: str | None
    qualification_score: float
    mean_cost_cny: float | None
    eligible_roles: frozenset[ModelRole]
    allowed_response_model_ids: frozenset[str]


class FormalInteractiveRuntime:
    """Lifecycle owner for one validated formal API experiment runtime."""

    def __init__(
        self,
        *,
        selection: FormalInteractiveSelection,
        gateway: SyncCompletionGateway,
        close_gateway: Callable[[], None] | None = None,
    ) -> None:
        self._selection = selection
        self._gateway = gateway
        self._close_gateway = close_gateway
        self._components: InteractiveAPIComponents | None = None
        self._component_key: tuple[Path, str] | None = None
        self._closed = False

    @property
    def selection(self) -> FormalInteractiveSelection:
        return self._selection

    @property
    def identity_payload(self) -> dict[str, object]:
        return self._selection.identity_payload()

    def components_for(
        self,
        *,
        run_dir: Path,
        run_id: str,
    ) -> InteractiveAPIComponents:
        """Build once after the runner has atomically created its run directory."""

        if self._closed:
            raise RuntimeError("formal interactive runtime is closed")
        if not run_id.strip():
            raise ValueError("run_id cannot be blank")
        destination = run_dir.resolve()
        if not destination.is_dir():
            raise FileNotFoundError(
                f"interactive run directory has not been initialized: {destination}"
            )
        key = (destination, run_id)
        if self._components is not None:
            if self._component_key != key:
                raise RuntimeError(
                    "one formal interactive runtime cannot serve multiple runs"
                )
            return self._components
        config = InteractiveAPIConfig(
            run_id=run_id,
            cache_path=destination / INTERACTIVE_CACHE_NAME,
            cost_ledger_path=destination / INTERACTIVE_COST_LEDGER_NAME,
            trace_path=destination / INTERACTIVE_TRACE_NAME,
            pricing_version=self._selection.pricing_version,
            companion=self._selection.companion,
            student_renderer=self._selection.student_renderer,
            judges=self._selection.judges,
            formal_mode=True,
            api_budget_cny=self._selection.api_budget_cny,
            family_separation_status=self._selection.family_separation_status,
            family_warnings=self._selection.family_warnings,
            expert_evaluation_required=self._selection.expert_evaluation_required,
        )
        self._components = build_interactive_api_components(
            gateway=self._gateway,
            config=config,
        )
        self._component_key = key
        return self._components

    def deferred_policy_runtime(
        self,
        *,
        run_dir: Path,
        run_id: str,
    ) -> PolicyRuntime:
        """Expose the companion policy identity before lazily sharing services."""

        if self._closed:
            raise RuntimeError("formal interactive runtime is closed")
        if not run_id.strip():
            raise ValueError("run_id cannot be blank")

        from ecocompanion.benchmark.model_provider import PolicyRuntime
        from ecocompanion.providers.benchmark_adapter import (
            BenchmarkProviderConfig,
            OpenAICompatibleBenchmarkProvider,
        )

        destination = run_dir.resolve()
        selection = self._selection
        pricing = selection.companion.pricing
        if pricing is None:
            raise RuntimeError(
                "formal interactive companion is missing frozen pricing"
            )

        def shared_services() -> tuple[Any, Any]:
            components = self.components_for(
                run_dir=destination,
                run_id=run_id,
            )
            return components.invoker, components.trace_recorder

        provider = OpenAICompatibleBenchmarkProvider.from_shared_service_factory(
            service_factory=shared_services,
            config=BenchmarkProviderConfig(
                model_id=selection.companion.model_id,
                run_id=run_id,
                cache_path=destination / INTERACTIVE_CACHE_NAME,
                cost_ledger_path=destination / INTERACTIVE_COST_LEDGER_NAME,
                trace_path=destination / INTERACTIVE_TRACE_NAME,
                model_snapshot_digest=selection.model_snapshot_digest,
                model_family=selection.companion.family,
                pricing=pricing,
                pricing_version=selection.pricing_version,
                formal_mode=True,
                api_budget_cny=selection.api_budget_cny,
                allowed_response_model_ids=(
                    selection.companion.allowed_response_model_ids
                ),
            ),
        )
        return PolicyRuntime(
            provider=provider,
            scope_factory=provider.for_scope,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close_gateway is not None:
            self._close_gateway()


def local_interactive_identity_payload() -> dict[str, object]:
    """Return the deterministic CI identity stored by local Track B/C runs."""

    return {
        "mode": LOCAL_INTERACTIVE_MODE,
        "renderer_name": TemplateStudentRenderer().name,
        "responder_name": DeterministicTrackCBaselineResponder().name,
        "role_bindings": [],
        "model_snapshot_digest": None,
        "pricing_version": None,
        "pricing_catalog_digest": None,
        "qualification_suite_version": None,
        "gateway_name": None,
        "gateway_version": None,
        "adapter_name": None,
        "adapter_version": None,
        "decoder_version": None,
        "cache_schema_version": None,
        "trace_schema_version": None,
        "prompt_identities": [],
        "api_budget_cny": None,
    }


def load_formal_interactive_selection(
    *,
    model_snapshot_path: Path,
    pricing_catalog_path: Path,
    companion_model_id: str | None = None,
    student_model_id: str | None = None,
    judge_model_ids: Sequence[str] | None = None,
    api_budget_cny: Decimal = MAX_API_BUDGET_CNY,
) -> FormalInteractiveSelection:
    """Select four qualified, explicitly priced model roles.

    Exact model IDs remain distinct.  Family separation is preferred and
    recorded; when the frozen snapshot cannot provide four independent
    families, the run is explicitly marked degraded and expert evaluation is
    required.
    """

    snapshot_path = model_snapshot_path.resolve()
    catalog_path = pricing_catalog_path.resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"qualified model snapshot not found: {snapshot_path}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"pricing catalog not found: {catalog_path}")
    snapshot = read_json_object(snapshot_path)
    candidates = _qualified_candidates(snapshot)
    recommended = _recommended_roles(snapshot)
    companion_id = _select_companion(
        candidates,
        requested=companion_model_id,
        recommended=recommended.get("companion_model"),
    )
    student_id = _select_student(
        candidates,
        companion_id=companion_id,
        requested=student_model_id,
        recommended=recommended.get("student_renderer_model"),
    )
    judge_ids = _select_judges(
        candidates,
        companion_id=companion_id,
        student_id=student_id,
        requested=judge_model_ids,
        recommended=recommended.get("judge_models"),
    )
    pricing_version, prices = _load_pricing_catalog(catalog_path)
    selected_ids = (companion_id, student_id, *judge_ids)
    missing_prices = sorted(model_id for model_id in selected_ids if model_id not in prices)
    if missing_prices:
        raise ValueError(
            f"formal API mode has unknown model prices: {missing_prices}"
        )

    def binding(model_id: str, role: ModelRole) -> InteractiveModelBinding:
        candidate = candidates[model_id]
        if role not in candidate.eligible_roles:
            raise ValueError(
                f"qualified model {model_id!r} is not eligible for role {role.value!r}"
            )
        return InteractiveModelBinding(
            model_id=model_id,
            family=candidate.family,
            role=role,
            pricing=prices[model_id],
            allowed_response_model_ids=candidate.allowed_response_model_ids,
        )

    role_bindings = (
        RoleBinding(
            role=ModelRole.EVALUATED_AGENT,
            model_id=companion_id,
            family=candidates[companion_id].family,
        ),
        RoleBinding(
            role=ModelRole.STUDENT_SIMULATOR,
            model_id=student_id,
            family=candidates[student_id].family,
        ),
        *(
            RoleBinding(
                role=ModelRole.JUDGE,
                model_id=judge_id,
                family=candidates[judge_id].family,
            )
            for judge_id in judge_ids
        ),
    )
    family_assessment = assess_family_separation(
        evaluated_agent=role_bindings[0],
        student_simulator=role_bindings[1],
        judges=role_bindings[2:],
    )
    selection = FormalInteractiveSelection(
        model_snapshot_path=snapshot_path,
        model_snapshot_digest=_file_digest(snapshot_path),
        pricing_catalog_path=catalog_path,
        pricing_catalog_digest=_file_digest(catalog_path),
        pricing_version=pricing_version,
        companion=binding(companion_id, ModelRole.EVALUATED_AGENT),
        student_renderer=binding(student_id, ModelRole.STUDENT_SIMULATOR),
        judges=(
            binding(judge_ids[0], ModelRole.JUDGE),
            binding(judge_ids[1], ModelRole.JUDGE),
        ),
        prompt_identities=(
            *api_renderer_prompt_identities(),
            judge_prompt_identity(),
        ),
        api_budget_cny=api_budget_cny,
        qualification_suite_version=_qualification_suite_version(snapshot),
        family_separation_status=family_assessment.status,
        family_warnings=family_assessment.warnings,
        expert_evaluation_required=family_assessment.expert_evaluation_required,
    )
    return selection


def build_openai_formal_interactive_runtime(
    *,
    settings: Settings,
    model_snapshot_path: Path,
    pricing_catalog_path: Path,
    companion_model_id: str | None = None,
    student_model_id: str | None = None,
    judge_model_ids: Sequence[str] | None = None,
    api_budget_cny: Decimal | None = None,
) -> FormalInteractiveRuntime:
    """Validate all artifacts before creating an authenticated API client."""

    budget = (
        Decimal(str(settings.api_budget_cny))
        if api_budget_cny is None
        else api_budget_cny
    )
    selection = load_formal_interactive_selection(
        model_snapshot_path=model_snapshot_path,
        pricing_catalog_path=pricing_catalog_path,
        companion_model_id=companion_model_id,
        student_model_id=student_model_id,
        judge_model_ids=judge_model_ids,
        api_budget_cny=budget,
    )
    api_key = settings.require_api_key()
    allowed_response_ids = {
        binding.model_id: binding.allowed_response_model_ids
        for binding in selection.bindings
        if binding.allowed_response_model_ids
    }
    client = OpenAICompatibleClient(
        base_url=settings.litellm_base_url,
        api_key=api_key,
        connect_timeout_seconds=settings.connect_timeout_seconds,
        read_timeout_seconds=settings.read_timeout_seconds,
        max_same_model_retries=settings.max_same_model_retries,
        allowed_response_model_ids=allowed_response_ids,
        default_extra_body=settings.provider_extra_body,
    )
    gateway = OpenAICompatibleSyncGateway(client, close_client=True)
    return FormalInteractiveRuntime(
        selection=selection,
        gateway=gateway,
        close_gateway=gateway.close,
    )


def _qualified_candidates(
    snapshot: Mapping[str, Any],
) -> dict[str, _QualifiedCandidate]:
    if snapshot.get("authenticated_discovery") is not True:
        raise ValueError(
            "formal API mode requires an authenticated qualified model snapshot"
        )
    _qualification_suite_version(snapshot)
    raw_models = _object_sequence(snapshot, "models")
    raw_capabilities = _object_sequence(snapshot, "capabilities")
    raw_qualifications = _object_sequence(snapshot, "qualifications")
    descriptors = _unique_by_model_id(raw_models, "models")
    capabilities = _unique_by_model_id(raw_capabilities, "capabilities")
    qualifications = _unique_by_model_id(raw_qualifications, "qualifications")

    candidates: dict[str, _QualifiedCandidate] = {}
    for model_id, qualification in qualifications.items():
        descriptor = descriptors.get(model_id)
        capability = capabilities.get(model_id)
        if descriptor is None or capability is None:
            continue
        if not _capability_is_ready(capability):
            continue
        raw_family = descriptor.get("declared_owner")
        family = (
            raw_family.strip()
            if isinstance(raw_family, str) and raw_family.strip()
            else None
        )
        sample_count = qualification.get("sample_count")
        score = _finite_float(qualification.get("qualification_score"))
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count != 20
            or score is None
        ):
            continue
        raw_roles = capability.get("eligible_roles")
        if not isinstance(raw_roles, list):
            continue
        try:
            roles = frozenset(ModelRole(str(role)) for role in raw_roles)
        except ValueError:
            continue
        if not roles:
            continue
        response_ids = _response_model_ids(capability, model_id=model_id)
        mean_cost = _finite_float(qualification.get("mean_cost_cny"))
        candidates[model_id] = _QualifiedCandidate(
            model_id=model_id,
            family=family,
            qualification_score=score,
            mean_cost_cny=mean_cost,
            eligible_roles=roles,
            allowed_response_model_ids=response_ids,
        )
    if not candidates:
        raise ValueError(
            "qualified snapshot contains no authenticated, ready, "
            "twenty-probe models with eligible roles"
        )
    return candidates


def _qualification_suite_version(snapshot: Mapping[str, Any]) -> str:
    value = snapshot.get("qualification_suite_version")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model snapshot is missing qualification_suite_version")
    return value


def _recommended_roles(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    recommended = snapshot.get("recommended_roles")
    if recommended is None:
        return {}
    if not isinstance(recommended, dict):
        raise ValueError("recommended_roles must be an object")
    return recommended


def _select_companion(
    candidates: Mapping[str, _QualifiedCandidate],
    *,
    requested: str | None,
    recommended: object,
) -> str:
    model_id = requested or _optional_model_id(recommended, "companion_model")
    if model_id is None:
        model_id = _ranked(candidates, role=ModelRole.EVALUATED_AGENT)[0]
    _require_candidate(candidates, model_id, ModelRole.EVALUATED_AGENT)
    return model_id


def _select_student(
    candidates: Mapping[str, _QualifiedCandidate],
    *,
    companion_id: str,
    requested: str | None,
    recommended: object,
) -> str:
    companion_family = candidates[companion_id].family
    recommended_id = _optional_model_id(
        recommended,
        "student_renderer_model",
    )
    requested_or_recommended = requested is not None or recommended_id is not None
    model_id = requested or recommended_id
    eligible = [
        candidate
        for candidate in candidates.values()
        if ModelRole.STUDENT_SIMULATOR in candidate.eligible_roles
        and candidate.model_id != companion_id
    ]
    if not eligible:
        raise ValueError(
            "qualified snapshot has no distinct student renderer model"
        )
    preferred = [
        candidate
        for candidate in eligible
        if companion_family is not None
        and candidate.family is not None
        and candidate.family != companion_family
    ]
    if model_id is None:
        model_id = min(
            preferred or eligible,
            key=_student_sort_key,
        ).model_id
    candidate = _require_candidate(
        candidates,
        model_id,
        ModelRole.STUDENT_SIMULATOR,
    )
    if candidate.model_id == companion_id:
        raise ValueError("student renderer must use a distinct model from companion")
    same_known_family = (
        companion_family is not None
        and candidate.family is not None
        and candidate.family == companion_family
    )
    if same_known_family and preferred and requested_or_recommended:
        # An explicit/recommended override may degrade only when the snapshot
        # genuinely has no cross-family student option.  This retains a
        # deterministic safety preference while still supporting undersupply.
        raise ValueError(
            "student renderer must use a different model and family from companion "
            "when a cross-family candidate is available"
        )
    return model_id


def _select_judges(
    candidates: Mapping[str, _QualifiedCandidate],
    *,
    companion_id: str,
    student_id: str,
    requested: Sequence[str] | None,
    recommended: object,
) -> tuple[str, str]:
    if requested is not None:
        selected = tuple(requested)
    elif recommended is not None:
        if not isinstance(recommended, list | tuple):
            raise ValueError("recommended judge_models must be a list")
        selected = tuple(str(model_id) for model_id in recommended)
    else:
        excluded_ids = {companion_id, student_id}
        excluded_families = {
            candidates[companion_id].family,
            candidates[student_id].family,
        }
        chosen: list[str] = []
        chosen_families: set[str] = set()
        ranked_judges = _ranked(candidates, role=ModelRole.JUDGE)
        for model_id in _ranked(candidates, role=ModelRole.JUDGE):
            candidate = candidates[model_id]
            if (
                model_id in excluded_ids
                or candidate.family in excluded_families
                or candidate.family in chosen_families
            ):
                continue
            chosen.append(model_id)
            if candidate.family is not None:
                chosen_families.add(candidate.family)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            for model_id in ranked_judges:
                if model_id in excluded_ids or model_id in chosen:
                    continue
                chosen.append(model_id)
                if len(chosen) == 2:
                    break
        selected = tuple(chosen)
    if len(selected) != 2:
        raise ValueError("formal API mode requires exactly two judge models")
    first = _require_candidate(candidates, selected[0], ModelRole.JUDGE)
    second = _require_candidate(candidates, selected[1], ModelRole.JUDGE)
    all_ids = {companion_id, student_id, first.model_id, second.model_id}
    if len(all_ids) != 4:
        raise ValueError(
            "companion, student renderer, and both judges must use four distinct models"
        )
    return first.model_id, second.model_id


def _student_sort_key(candidate: _QualifiedCandidate) -> tuple[bool, float, float, str]:
    return (
        candidate.mean_cost_cny is None,
        (
            candidate.mean_cost_cny
            if candidate.mean_cost_cny is not None
            else math.inf
        ),
        -candidate.qualification_score,
        candidate.model_id,
    )


def _ranked(
    candidates: Mapping[str, _QualifiedCandidate],
    *,
    role: ModelRole,
) -> list[str]:
    ranked = sorted(
        (
            candidate
            for candidate in candidates.values()
            if role in candidate.eligible_roles
        ),
        key=lambda candidate: (
            -candidate.qualification_score,
            candidate.model_id,
        ),
    )
    if not ranked:
        raise ValueError(f"qualified snapshot has no model eligible for {role.value}")
    return [candidate.model_id for candidate in ranked]


def _require_candidate(
    candidates: Mapping[str, _QualifiedCandidate],
    model_id: str,
    role: ModelRole,
) -> _QualifiedCandidate:
    candidate = candidates.get(model_id)
    if candidate is None:
        raise ValueError(
            f"model {model_id!r} is not eligible in the qualified snapshot"
        )
    if role not in candidate.eligible_roles:
        raise ValueError(
            f"qualified model {model_id!r} is not eligible for role {role.value!r}"
        )
    return candidate


def _load_pricing_catalog(
    path: Path,
) -> tuple[str, dict[str, ModelPricing]]:
    payload = read_json_object(path)
    version = payload.get("pricing_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pricing catalog requires a non-blank pricing_version")
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("pricing catalog requires a non-empty models object")
    prices: dict[str, ModelPricing] = {}
    for raw_model_id, raw_price in raw_models.items():
        model_id = str(raw_model_id)
        if not model_id.strip() or not isinstance(raw_price, dict):
            raise ValueError("pricing catalog model entries must be named objects")
        input_price = _decimal_price(
            raw_price,
            "input_per_million",
            required=True,
        )
        output_price = _decimal_price(
            raw_price,
            "output_per_million",
            required=True,
        )
        if input_price is None or output_price is None:
            raise AssertionError("required pricing fields were validated")
        prices[model_id] = ModelPricing(
            input_per_million=input_price,
            output_per_million=output_price,
            cached_input_per_million=_decimal_price(
                raw_price,
                "cached_input_per_million",
                required=False,
            ),
            reasoning_per_million=_decimal_price(
                raw_price,
                "reasoning_per_million",
                required=False,
            ),
        )
    return version, prices


def _decimal_price(
    payload: Mapping[str, Any],
    key: str,
    *,
    required: bool,
) -> Decimal | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise ValueError(f"pricing entry requires {key}")
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite non-negative decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"{key} must be a finite non-negative decimal"
        ) from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{key} must be a finite non-negative decimal")
    return parsed


def _object_sequence(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[dict[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"qualified snapshot requires a non-empty {key} list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"qualified snapshot {key} entries must be objects")
    return tuple(item for item in value if isinstance(item, dict))


def _unique_by_model_id(
    items: Sequence[Mapping[str, Any]],
    field_name: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in items:
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"{field_name} entry has no valid model_id")
        if model_id in indexed:
            raise ValueError(f"{field_name} contains duplicate model_id {model_id!r}")
        indexed[model_id] = item
    return indexed


def _capability_is_ready(capability: Mapping[str, Any]) -> bool:
    return (
        capability.get("status") == "ready"
        and capability.get("basic_chat") is True
        and capability.get("system_and_multiturn") is True
        and capability.get("chinese_text") is True
        and capability.get("usage_reported") is True
        and capability.get("preferred_structured_mode") is not None
    )


def _response_model_ids(
    capability: Mapping[str, Any],
    *,
    model_id: str,
) -> frozenset[str]:
    raw_ids = capability.get("response_model_ids", ())
    if not isinstance(raw_ids, list | tuple):
        raise ValueError(f"model {model_id!r} response_model_ids must be a list")
    normalized = frozenset(str(item) for item in raw_ids)
    if any(not item.strip() for item in normalized):
        raise ValueError(f"model {model_id!r} has a blank response model ID")
    return frozenset(item for item in normalized if item != model_id)


def _optional_model_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recommended {field_name} must be a non-blank string")
    return value


def _finite_float(value: object) -> float | None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _role_binding(binding: InteractiveModelBinding) -> RoleBinding:
    return RoleBinding(
        role=binding.role,
        model_id=binding.model_id,
        family=binding.family,
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal") from exc
