"""Recorded, model-pinned provider adapter for benchmark policies."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from ecocompanion.agent.safety import SafetyAuditor
from ecocompanion.benchmark.model_provider import (
    BenchmarkInvocationScope,
    BenchmarkModelProvider,
    BenchmarkProviderFatalError,
    BenchmarkProviderIdentity,
    DecisionProviderRequest,
    ResponseProviderRequest,
)
from ecocompanion.benchmark.model_trace import (
    ModelTraceContext,
    ModelTraceRecorder,
)
from ecocompanion.domain.enums import Action
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    ResponsePayload,
    RiskEstimate,
    SafetyAudit,
)
from ecocompanion.infrastructure import ExactCache, JsonlTraceStore, ModelPricing
from ecocompanion.providers.recorded_gateway import (
    InteractiveModelBinding,
    InteractiveRunIntegrityError,
    InvocationContext,
    RecordedCompletion,
    RecordedModelInvoker,
    SyncCompletionGateway,
)
from ecocompanion.providers.types import ModelRole, StructuredMode

_MAX_API_BUDGET_CNY = Decimal("5000")
_STRUCTURED_MODES = (
    StructuredMode.JSON_SCHEMA,
    StructuredMode.TOOL_CALL,
    StructuredMode.JSON_OBJECT,
    StructuredMode.PLAIN_JSON,
)


@dataclass(frozen=True, slots=True)
class BenchmarkProviderConfig:
    """Immutable decoding, accounting, and artifact configuration."""

    model_id: str
    run_id: str
    cache_path: Path
    cost_ledger_path: Path
    trace_path: Path
    provider_name: str = "litellm-openai-compatible"
    model_snapshot_digest: str | None = None
    model_family: str | None = None
    pricing: ModelPricing | None = None
    pricing_version: str | None = None
    formal_mode: bool = True
    api_budget_cny: Decimal = _MAX_API_BUDGET_CNY
    allowed_response_model_ids: frozenset[str] = frozenset()
    decision_max_tokens: int = 900
    response_max_tokens: int = 320
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be blank")
        if not self.run_id.strip():
            raise ValueError("run_id cannot be blank")
        if not self.provider_name.strip():
            raise ValueError("provider_name cannot be blank")
        if self.model_snapshot_digest is not None and not (
            self.model_snapshot_digest.strip()
        ):
            raise ValueError("model_snapshot_digest cannot be blank")
        if self.model_family is not None and not self.model_family.strip():
            raise ValueError("model_family cannot be blank")
        if self.decision_max_tokens <= 0:
            raise ValueError("decision_max_tokens must be positive")
        if self.response_max_tokens <= 0:
            raise ValueError("response_max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not Decimal(0) < self.api_budget_cny <= _MAX_API_BUDGET_CNY:
            raise ValueError("api_budget_cny must be in (0, 5000]")
        paths = (self.cache_path, self.cost_ledger_path, self.trace_path)
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("cache, cost ledger, and trace paths must differ")
        if any(not model_id.strip() for model_id in self.allowed_response_model_ids):
            raise ValueError("allowed response model IDs cannot be blank")
        if self.formal_mode:
            if self.pricing is None:
                raise ValueError("formal benchmark calls require explicit pricing")
            if self.pricing_version is None or not self.pricing_version.strip():
                raise ValueError(
                    "formal benchmark calls require a pricing_version"
                )


class _RecordedProviderBackend:
    """Lifecycle owner that creates file-backed services only on first call."""

    def __init__(
        self,
        *,
        gateway: SyncCompletionGateway | None,
        config: BenchmarkProviderConfig,
        close_callback: Callable[[], None] | None,
        invoker: RecordedModelInvoker | None = None,
        trace: ModelTraceRecorder | None = None,
        service_factory: (
            Callable[[], tuple[RecordedModelInvoker, ModelTraceRecorder]]
            | None
        ) = None,
    ) -> None:
        if (invoker is None) != (trace is None):
            raise ValueError("shared invoker and trace must be provided together")
        service_sources = sum(
            (
                gateway is not None,
                invoker is not None,
                service_factory is not None,
            )
        )
        if service_sources != 1:
            raise ValueError(
                "configure exactly one gateway, shared services, or service factory"
            )
        self._gateway = gateway
        self._config = config
        self._close_callback = close_callback
        self._service_factory = service_factory
        self._lock = threading.RLock()
        self._invoker = invoker
        self._trace = trace
        self._closed = False

    def services(self) -> tuple[RecordedModelInvoker, ModelTraceRecorder]:
        with self._lock:
            if self._closed:
                raise RuntimeError("benchmark provider is closed")
            if self._invoker is None or self._trace is None:
                if self._service_factory is not None:
                    self._invoker, self._trace = self._service_factory()
                else:
                    if self._gateway is None:
                        raise RuntimeError(
                            "benchmark provider gateway is unavailable"
                        )
                    binding = _binding(self._config)
                    self._invoker = RecordedModelInvoker(
                        gateway=self._gateway,
                        cache=ExactCache(self._config.cache_path),
                        bindings=(binding,),
                        pricing_version=self._config.pricing_version,
                        cost_ledger_path=self._config.cost_ledger_path,
                        formal_mode=self._config.formal_mode,
                        api_budget_cny=self._config.api_budget_cny,
                    )
                    self._trace = ModelTraceRecorder(
                        JsonlTraceStore(self._config.trace_path)
                    )
            return self._invoker, self._trace

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            callback = self._close_callback
        if callback is not None:
            callback()


class OpenAICompatibleBenchmarkProvider(BenchmarkModelProvider):
    """Benchmark adapter with exact cache, accounting, traces, and unit scopes."""

    def __init__(
        self,
        *,
        gateway: SyncCompletionGateway | None,
        config: BenchmarkProviderConfig,
        close_callback: Callable[[], None] | None = None,
        _backend: _RecordedProviderBackend | None = None,
        _scope: BenchmarkInvocationScope | None = None,
        _owns_backend: bool = True,
    ) -> None:
        self._config = config
        self._backend = _backend or _RecordedProviderBackend(
            gateway=gateway,
            config=config,
            close_callback=close_callback,
        )
        self._scope = _scope
        self._owns_backend = _owns_backend
        self._closed = False
        self._identity = BenchmarkProviderIdentity(
            provider_name=config.provider_name,
            model_id=config.model_id,
            model_snapshot_digest=config.model_snapshot_digest,
            pricing_version=config.pricing_version,
            input_price_cny_per_million=(
                config.pricing.input_per_million
                if config.pricing is not None
                else None
            ),
            output_price_cny_per_million=(
                config.pricing.output_per_million
                if config.pricing is not None
                else None
            ),
            api_budget_cny=(
                config.api_budget_cny
                if config.pricing is not None
                and config.pricing_version is not None
                else None
            ),
        )

    @classmethod
    def from_shared_services(
        cls,
        *,
        invoker: RecordedModelInvoker,
        trace_recorder: ModelTraceRecorder,
        config: BenchmarkProviderConfig,
    ) -> OpenAICompatibleBenchmarkProvider:
        """Reuse one interactive run's cache, ledger, budget, and trace."""

        backend = _RecordedProviderBackend(
            gateway=None,
            config=config,
            close_callback=None,
            invoker=invoker,
            trace=trace_recorder,
        )
        return cls(
            gateway=None,
            config=config,
            _backend=backend,
            _owns_backend=False,
        )

    @classmethod
    def from_shared_service_factory(
        cls,
        *,
        service_factory: Callable[
            [], tuple[RecordedModelInvoker, ModelTraceRecorder]
        ],
        config: BenchmarkProviderConfig,
    ) -> OpenAICompatibleBenchmarkProvider:
        """Bind identity now and acquire shared services on the first call."""

        backend = _RecordedProviderBackend(
            gateway=None,
            config=config,
            close_callback=None,
            service_factory=service_factory,
        )
        return cls(
            gateway=None,
            config=config,
            _backend=backend,
            _owns_backend=False,
        )

    @property
    def identity(self) -> BenchmarkProviderIdentity:
        return self._identity

    def for_scope(
        self,
        scope: BenchmarkInvocationScope,
    ) -> OpenAICompatibleBenchmarkProvider:
        """Return a lightweight immutable unit view over shared services."""

        self._ensure_open()
        if scope.run_id != self._config.run_id:
            raise ValueError("provider scope run_id does not match configuration")
        return OpenAICompatibleBenchmarkProvider(
            gateway=None,
            config=self._config,
            _backend=self._backend,
            _scope=scope,
            _owns_backend=False,
        )

    def decide(self, request: DecisionProviderRequest) -> AgentDecisionV1:
        self._ensure_open()
        scope = self._effective_scope(request.episode_id)
        invoker, trace = self._backend.services()
        try:
            invocation = invoker.complete_structured(
                binding=_binding(self._config),
                context=_invocation_context(
                    scope,
                    request=request,
                    kind="decision",
                ),
                messages=tuple(
                    message.to_dict() for message in request.messages
                ),
                schema=AgentDecisionV1,
                prompt_version=request.prompt_version,
                max_tokens=self._config.decision_max_tokens,
                temperature=self._config.temperature,
                modes=_STRUCTURED_MODES,
            )
        except InteractiveRunIntegrityError as exc:
            raise BenchmarkProviderFatalError(str(exc)) from exc

        decision = invocation.value
        _validate_decision_contract(request, decision)
        audit = _audit_response(decision.response)
        self._record_decision_trace(
            trace=trace,
            scope=scope,
            request=request,
            invocation=invocation,
            audit=audit,
        )
        return decision

    def render(self, request: ResponseProviderRequest) -> ResponsePayload:
        self._ensure_open()
        scope = self._effective_scope(request.episode_id)
        invoker, trace = self._backend.services()
        try:
            invocation = invoker.complete_structured(
                binding=_binding(self._config),
                context=_invocation_context(
                    scope,
                    request=request,
                    kind="response",
                ),
                messages=tuple(
                    message.to_dict() for message in request.messages
                ),
                schema=ResponsePayload,
                prompt_version=request.prompt_version,
                max_tokens=self._config.response_max_tokens,
                temperature=self._config.temperature,
                modes=_STRUCTURED_MODES,
            )
        except InteractiveRunIntegrityError as exc:
            raise BenchmarkProviderFatalError(str(exc)) from exc

        response = invocation.value
        audit = _audit_response(response)
        self._record_response_trace(
            trace=trace,
            scope=scope,
            request=request,
            invocation=invocation,
            audit=audit,
        )
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_backend:
            self._backend.close()

    def _record_decision_trace(
        self,
        *,
        trace: ModelTraceRecorder,
        scope: BenchmarkInvocationScope,
        request: DecisionProviderRequest,
        invocation: RecordedCompletion[AgentDecisionV1],
        audit: SafetyAudit,
    ) -> None:
        decision = invocation.value
        trace.record(
            context=_trace_context(
                scope=scope,
                request=request,
                kind="decision",
                selected_action=decision.decision.primary_action,
                assessment=decision.assessment,
                evidence_ids=tuple(
                    evidence.observation_id for evidence in decision.evidence
                ),
                invocation=invocation,
            ),
            response_text=_flatten_response(decision.response),
            safety_audit=audit,
            invocations=(invocation,),
        )

    def _record_response_trace(
        self,
        *,
        trace: ModelTraceRecorder,
        scope: BenchmarkInvocationScope,
        request: ResponseProviderRequest,
        invocation: RecordedCompletion[ResponsePayload],
        audit: SafetyAudit,
    ) -> None:
        trace.record(
            context=_trace_context(
                scope=scope,
                request=request,
                kind="response",
                selected_action=request.decision.primary_action,
                assessment=request.assessment,
                evidence_ids=(),
                invocation=invocation,
            ),
            response_text=_flatten_response(invocation.value),
            safety_audit=audit,
            invocations=(invocation,),
        )

    def _effective_scope(self, episode_id: str) -> BenchmarkInvocationScope:
        if self._scope is not None:
            return self._scope
        if self._config.formal_mode:
            raise BenchmarkProviderFatalError(
                "formal benchmark provider requires an experiment-unit scope"
            )
        return BenchmarkInvocationScope(
            run_id=self._config.run_id,
            unit_id=f"direct:{episode_id}",
            seed=_stable_seed(self._config.run_id, episode_id),
            replicate_id=0,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("benchmark provider is closed")


def _binding(config: BenchmarkProviderConfig) -> InteractiveModelBinding:
    return InteractiveModelBinding(
        model_id=config.model_id,
        family=config.model_family,
        role=ModelRole.EVALUATED_AGENT,
        pricing=config.pricing,
        allowed_response_model_ids=config.allowed_response_model_ids,
    )


def _invocation_context(
    scope: BenchmarkInvocationScope,
    *,
    request: DecisionProviderRequest | ResponseProviderRequest,
    kind: str,
) -> InvocationContext:
    track_slug = request.observation.track.value.lower()
    return InvocationContext(
        run_id=scope.run_id,
        invocation_id=(
            f"track-{track_slug}:{scope.unit_id}:{request.policy_name}:"
            f"step-{request.step_id}:{kind}"
        ),
        seed=scope.seed,
        replicate_id=scope.replicate_id,
    )


def _trace_context(
    *,
    scope: BenchmarkInvocationScope,
    request: DecisionProviderRequest | ResponseProviderRequest,
    kind: str,
    selected_action: Action,
    assessment: RiskEstimate,
    evidence_ids: tuple[str, ...],
    invocation: RecordedCompletion[Any],
) -> ModelTraceContext:
    observation = request.observation
    track_slug = observation.track.value.lower()
    return ModelTraceContext(
        trace_id=(
            f"trace:{scope.run_id}:{scope.unit_id}:"
            f"step-{request.step_id}:{kind}"
        ),
        scenario_id=scope.unit_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        track=observation.track,
        simulated_time=observation.simulated_time,
        observation_id=observation.current_observation.observation_id,
        message=observation.current_observation.message,
        selected_action=selected_action,
        allowed_actions=observation.allowed_primary_actions,
        random_seed=scope.seed,
        assessment=assessment,
        conversation_history=observation.visible_context.conversation_history,
        memory_refs=tuple(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id != observation.current_observation.observation_id
        ),
        external_event={
            "component": f"track_{track_slug}_model_{kind}",
            "policy_name": request.policy_name,
            "prompt_version": request.prompt_version,
            "unit_id": scope.unit_id,
            "replicate_id": scope.replicate_id,
            "call_id": invocation.call_id,
            "cache_key": invocation.cache_key,
            "structured_mode": (
                invocation.structured_mode.value
                if invocation.structured_mode is not None
                else None
            ),
            "evidence_ids": list(evidence_ids),
        },
    )


def _validate_decision_contract(
    request: DecisionProviderRequest,
    decision: AgentDecisionV1,
) -> None:
    if decision.decision.primary_action not in request.allowed_actions:
        raise ValueError("provider selected an action outside allowed_actions")
    unknown_evidence = {
        evidence.observation_id
        for evidence in decision.evidence
        if evidence.observation_id not in request.allowed_evidence_ids
    }
    if unknown_evidence:
        raise ValueError(
            "provider cited evidence outside allowed_evidence_ids: "
            f"{sorted(unknown_evidence)}"
        )


def _audit_response(response: ResponsePayload) -> SafetyAudit:
    return SafetyAuditor().audit(_flatten_response(response))


def _flatten_response(response: ResponsePayload) -> str:
    return (
        response.text
        if response.question is None
        else f"{response.text} {response.question}"
    ).strip()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
