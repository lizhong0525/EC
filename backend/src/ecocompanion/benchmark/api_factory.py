"""Composition root for API-backed interactive benchmark components."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from ecocompanion.benchmark.api_renderers import (
    APIStudentLanguageRenderer,
    APITrackCResponder,
)
from ecocompanion.benchmark.judging import AnonymousDualJudgePanel
from ecocompanion.benchmark.model_trace import ModelTraceRecorder
from ecocompanion.infrastructure import ExactCache, JsonlTraceStore
from ecocompanion.providers.recorded_gateway import (
    AsyncCompletionClient,
    InteractiveModelBinding,
    OpenAICompatibleSyncGateway,
    RecordedModelInvoker,
    SyncCompletionGateway,
)
from ecocompanion.providers.registry import (
    assess_family_separation,
    validate_role_separation,
)
from ecocompanion.providers.types import ModelRole, RoleBinding


@dataclass(frozen=True, slots=True)
class InteractiveAPIConfig:
    """Frozen inputs required to reproduce one API-backed interactive run."""

    run_id: str
    cache_path: Path
    cost_ledger_path: Path
    trace_path: Path
    pricing_version: str | None
    companion: InteractiveModelBinding
    student_renderer: InteractiveModelBinding
    judges: tuple[InteractiveModelBinding, InteractiveModelBinding]
    formal_mode: bool = True
    api_budget_cny: Decimal = Decimal("5000")
    family_separation_status: Literal["full", "degraded"] = "full"
    family_warnings: tuple[str, ...] = ()
    expert_evaluation_required: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be blank")
        if self.companion.role is not ModelRole.EVALUATED_AGENT:
            raise ValueError("companion binding must use EVALUATED_AGENT")
        if self.student_renderer.role is not ModelRole.STUDENT_SIMULATOR:
            raise ValueError("student renderer binding must use STUDENT_SIMULATOR")
        if any(judge.role is not ModelRole.JUDGE for judge in self.judges):
            raise ValueError("judge bindings must use JUDGE")
        if self.api_budget_cny <= 0 or self.api_budget_cny > Decimal("5000"):
            raise ValueError("api_budget_cny must be in (0, 5000]")
        report = validate_role_separation(
            evaluated_agent=_role_binding(self.companion),
            student_simulator=_role_binding(self.student_renderer),
            judges=tuple(_role_binding(judge) for judge in self.judges),
            require_known_families=(
                self.formal_mode and self.family_separation_status == "full"
            ),
            require_two_judges=True,
            allow_family_overlap=(
                self.formal_mode and self.family_separation_status == "degraded"
            ),
        )
        if not report.valid:
            raise ValueError(
                "interactive model-role separation failed: "
                + "; ".join(report.violations)
            )
        assessment = assess_family_separation(
            evaluated_agent=_role_binding(self.companion),
            student_simulator=_role_binding(self.student_renderer),
            judges=tuple(_role_binding(judge) for judge in self.judges),
        )
        if assessment.status != self.family_separation_status:
            raise ValueError(
                "family_separation_status does not match configured model families"
            )
        if tuple(self.family_warnings) != assessment.warnings:
            raise ValueError(
                "family_warnings do not match configured model families"
            )
        if self.expert_evaluation_required != assessment.expert_evaluation_required:
            raise ValueError(
                "expert_evaluation_required does not match configured model families"
            )


@dataclass(frozen=True, slots=True)
class InteractiveAPIComponents:
    """Ready-to-inject benchmark ports and their accounting service."""

    student_renderer: APIStudentLanguageRenderer
    track_c_responder: APITrackCResponder
    judge_panel: AnonymousDualJudgePanel
    invoker: RecordedModelInvoker
    trace_store: JsonlTraceStore
    trace_recorder: ModelTraceRecorder
    companion_binding: InteractiveModelBinding
    student_renderer_binding: InteractiveModelBinding
    judge_bindings: tuple[InteractiveModelBinding, InteractiveModelBinding]

    def student_renderer_for(self, *, run_id: str) -> APIStudentLanguageRenderer:
        """Create a scoped renderer while sharing one cache and cost ledger."""

        return APIStudentLanguageRenderer(
            invoker=self.invoker,
            binding=self.student_renderer_binding,
            run_id=run_id,
            trace_recorder=self.trace_recorder,
        )

    def track_c_responder_for(self, *, run_id: str) -> APITrackCResponder:
        """Create a scoped responder while sharing one cache and cost ledger."""

        return APITrackCResponder(
            invoker=self.invoker,
            binding=self.companion_binding,
            run_id=run_id,
            trace_recorder=self.trace_recorder,
        )


def build_interactive_api_components(
    *,
    gateway: SyncCompletionGateway,
    config: InteractiveAPIConfig,
) -> InteractiveAPIComponents:
    """Build components around any sync gateway, including deterministic fakes."""

    bindings = (
        config.companion,
        config.student_renderer,
        *config.judges,
    )
    trace_store = JsonlTraceStore(config.trace_path)
    recorder = ModelTraceRecorder(trace_store)
    invoker = RecordedModelInvoker(
        gateway=gateway,
        cache=ExactCache(config.cache_path),
        bindings=bindings,
        pricing_version=config.pricing_version,
        cost_ledger_path=config.cost_ledger_path,
        formal_mode=config.formal_mode,
        api_budget_cny=config.api_budget_cny,
    )
    excluded_families = frozenset(
        family
        for family in (
            config.companion.family,
            config.student_renderer.family,
        )
        if family is not None
    )
    degraded_family_separation = config.family_separation_status == "degraded"
    return InteractiveAPIComponents(
        student_renderer=APIStudentLanguageRenderer(
            invoker=invoker,
            binding=config.student_renderer,
            run_id=config.run_id,
            trace_recorder=recorder,
        ),
        track_c_responder=APITrackCResponder(
            invoker=invoker,
            binding=config.companion,
            run_id=config.run_id,
            trace_recorder=recorder,
        ),
        judge_panel=AnonymousDualJudgePanel(
            invoker=invoker,
            judge_bindings=config.judges,
            run_id=config.run_id,
            trace_recorder=recorder,
            excluded_families=excluded_families,
            require_candidate_family=(
                config.formal_mode and not degraded_family_separation
            ),
            allow_family_overlap=degraded_family_separation,
        ),
        invoker=invoker,
        trace_store=trace_store,
        trace_recorder=recorder,
        companion_binding=config.companion,
        student_renderer_binding=config.student_renderer,
        judge_bindings=config.judges,
    )


class OpenAIInteractiveRuntime:
    """Lifecycle owner for a real OpenAI-compatible async client bridge."""

    def __init__(
        self,
        *,
        client: AsyncCompletionClient,
        config: InteractiveAPIConfig,
        close_client: bool = True,
    ) -> None:
        self._gateway = OpenAICompatibleSyncGateway(
            client,
            close_client=close_client,
        )
        try:
            self.components = build_interactive_api_components(
                gateway=self._gateway,
                config=config,
            )
        except Exception:
            self._gateway.close()
            raise

    def __enter__(self) -> OpenAIInteractiveRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._gateway.close()


def _role_binding(binding: InteractiveModelBinding) -> RoleBinding:
    return RoleBinding(
        role=binding.role,
        model_id=binding.model_id,
        family=binding.family,
    )
