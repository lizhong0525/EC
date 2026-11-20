from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ecocompanion.benchmark.model_provider import (
    BenchmarkInvocationScope,
    BenchmarkPromptMessage,
    DecisionProviderRequest,
    PolicyRuntime,
    ResponseProviderRequest,
)
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.config import Settings
from ecocompanion.domain.enums import (
    Action,
    EscalationTarget,
    RiskLevel,
    Track,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    CurrentObservation,
    DecisionPlan,
    ObservationV1,
    ResponsePayload,
    RiskEstimate,
    VisibleProfile,
)
from ecocompanion.experiments.provider_runtime import (
    build_new_policy_runtime,
)
from ecocompanion.experiments.runner import (
    resume_experiment,
    start_experiment,
)
from ecocompanion.infrastructure import (
    CostLedger,
    JsonlTraceStore,
    ModelPricing,
)
from ecocompanion.providers.benchmark_adapter import (
    BenchmarkProviderConfig,
    OpenAICompatibleBenchmarkProvider,
)
from ecocompanion.providers.types import (
    ChatCompletion,
    CompletionUsage,
    StructuredAttempt,
    StructuredCompletion,
)


class _FakeStructuredGateway:
    def __init__(self) -> None:
        self.schemas: list[type[Any]] = []
        self.closed = False

    def chat_completion(self, **_kwargs: Any) -> ChatCompletion:
        raise AssertionError("Track A adapter unexpectedly requested plain text")

    def complete_structured(
        self,
        *,
        model: str,
        schema: type[Any],
        modes: tuple[Any, ...],
        **_kwargs: Any,
    ) -> StructuredCompletion[Any]:
        self.schemas.append(schema)
        if schema is AgentDecisionV1:
            value = AgentDecisionV1(
                assessment=RiskEstimate(
                    depression_worsening=0.1,
                    immediate_safety_risk=0.1,
                    dependency_risk=0.1,
                ),
                uncertainty=0.2,
                decision=DecisionPlan(
                    primary_action=Action.SUPPORTIVE_LISTENING,
                    escalation_level=RiskLevel.L0,
                    escalation_target=EscalationTarget.NONE,
                ),
                response=ResponsePayload(text="我在听。"),
            )
        else:
            value = ResponsePayload(text="渲染后的回复。")
        mode = modes[0]
        return StructuredCompletion(
            value=value,
            completion=ChatCompletion(
                requested_model=model,
                response_model=model,
                content=value.model_dump_json(),
                finish_reason="stop",
                usage=CompletionUsage(
                    input_tokens=100,
                    output_tokens=20,
                ),
                retry_count=0,
                response_id=f"response:{len(self.schemas)}",
            ),
            mode=mode,
            attempts=(StructuredAttempt(mode=mode, succeeded=True),),
        )

    def close(self) -> None:
        self.closed = True


def _observation() -> ObservationV1:
    return ObservationV1(
        track=Track.A,
        episode_id="episode",
        step_id=0,
        simulated_time="day-01",
        visible_profile=VisibleProfile(age_group="13-17 synthetic"),
        current_observation=CurrentObservation(
            observation_id="obs",
            message="最近有点累。",
        ),
        allowed_primary_actions=(Action.SUPPORTIVE_LISTENING,),
    )


def _qualified_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "model-a",
                        "declared_owner": "family-a",
                        "created": None,
                    }
                ],
                "capabilities": [
                    {
                        "model_id": "model-a",
                        "status": "ready",
                        "basic_chat": True,
                        "chinese_text": True,
                    }
                ],
                "qualifications": [
                    {
                        "model_id": "model-a",
                        "qualification_score": 91.0,
                    }
                ],
                "recommended_roles": {"companion_model": "model-a"},
            }
        ),
        encoding="utf-8",
    )


def test_sync_adapter_records_cache_cost_and_trace_without_network(
    tmp_path: Path,
) -> None:
    gateway = _FakeStructuredGateway()
    root_provider = OpenAICompatibleBenchmarkProvider(
        gateway=gateway,
        config=BenchmarkProviderConfig(
            model_id="model-a",
            run_id="run-a",
            cache_path=tmp_path / "api_cache.sqlite3",
            cost_ledger_path=tmp_path / "api_cost_ledger.jsonl",
            trace_path=tmp_path / "api_trace.jsonl",
            model_snapshot_digest="a" * 64,
            model_family="family-a",
            pricing=ModelPricing(
                input_per_million=Decimal("2"),
                output_per_million=Decimal("4"),
            ),
            pricing_version="test-pricing-v1",
        ),
        close_callback=gateway.close,
    )
    provider = root_provider.for_scope(
        BenchmarkInvocationScope(
            run_id="run-a",
            unit_id="case-a__policy=test__surface=00__repeat=000",
            seed=7,
            replicate_id=0,
        )
    )
    messages = (
        BenchmarkPromptMessage(role="system", content="system"),
        BenchmarkPromptMessage(role="user", content="user"),
    )
    observation = _observation()

    decision = provider.decide(
        DecisionProviderRequest(
            policy_name="current-message-llm",
            prompt_version="v1",
            episode_id="episode",
            step_id=0,
            observation=observation,
            messages=messages,
            allowed_actions=(Action.SUPPORTIVE_LISTENING,),
            allowed_evidence_ids=("obs",),
        )
    )
    response = provider.render(
        ResponseProviderRequest(
            policy_name="eco-safe",
            prompt_version="v1",
            episode_id="episode",
            step_id=0,
            observation=observation,
            assessment=decision.assessment,
            decision=decision.decision,
            messages=messages,
        )
    )
    assert provider.decide(
        DecisionProviderRequest(
            policy_name="current-message-llm",
            prompt_version="v1",
            episode_id="episode",
            step_id=0,
            observation=observation,
            messages=messages,
            allowed_actions=(Action.SUPPORTIVE_LISTENING,),
            allowed_evidence_ids=("obs",),
        )
    ) == decision
    provider.close()
    root_provider.close()
    root_provider.close()

    assert decision.decision.primary_action is Action.SUPPORTIVE_LISTENING
    assert response.text == "渲染后的回复。"
    assert gateway.schemas == [AgentDecisionV1, ResponsePayload]
    assert gateway.closed
    costs = CostLedger(path=tmp_path / "api_cost_ledger.jsonl").summary()
    assert costs.call_count == 2
    assert costs.known_cost_cny > 0
    traces = JsonlTraceStore(tmp_path / "api_trace.jsonl").read_all()
    assert len(traces) == 2
    assert {trace.external_event["component"] for trace in traces} == {
        "track_a_model_decision",
        "track_a_model_response",
    }


def test_local_policy_runtime_does_not_require_snapshot_or_key(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        ECO_ARTIFACTS_DIR=tmp_path,
        LITELLM_API_KEY=None,
    )

    runtime = build_new_policy_runtime(
        settings=settings,
        policy_names=("history-aware-rule",),
        model_id=None,
        snapshot_path=None,
    )

    assert runtime.provider is None


def test_track_a_runner_scopes_repeats_and_resumes_recorded_calls(
    tmp_path: Path,
) -> None:
    gateway = _FakeStructuredGateway()
    run_dir = tmp_path / "track-a-run"
    root_provider = OpenAICompatibleBenchmarkProvider(
        gateway=gateway,
        config=BenchmarkProviderConfig(
            model_id="model-a",
            run_id="track-a-run",
            cache_path=run_dir / "api_cache.sqlite3",
            cost_ledger_path=run_dir / "api_cost_ledger.jsonl",
            trace_path=run_dir / "api_trace.jsonl",
            model_snapshot_digest="b" * 64,
            model_family="family-a",
            pricing=ModelPricing(
                input_per_million=Decimal("2"),
                output_per_million=Decimal("4"),
            ),
            pricing_version="test-pricing-v1",
        ),
        close_callback=gateway.close,
    )
    runtime = PolicyRuntime(
        provider=root_provider,
        scope_factory=root_provider.for_scope,
    )
    assert not run_dir.exists()

    partial = start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=("current-message-llm",),
        surface_indices=(0,),
        repeats=2,
        case_limit=1,
        run_id="track-a-run",
        max_units=1,
        policy_runtime=runtime,
    )
    first_call_count = len(gateway.schemas)
    completed = resume_experiment(run_dir, policy_runtime=runtime)
    final_call_count = len(gateway.schemas)
    repeated = resume_experiment(run_dir, policy_runtime=runtime)
    runtime.close()

    assert partial.completed_count == 1
    assert completed.completed_count == completed.planned_count == 2
    assert repeated.executed_count == 0
    assert first_call_count == 14
    assert final_call_count == 28
    assert len(gateway.schemas) == final_call_count
    assert gateway.closed
    assert (
        CostLedger(path=run_dir / "api_cost_ledger.jsonl").summary().call_count
        == 28
    )
    assert len(JsonlTraceStore(run_dir / "api_trace.jsonl").read_all()) == 28


def test_model_policy_runtime_uses_frozen_qualified_snapshot_without_calling_api(
    tmp_path,
) -> None:
    snapshot = tmp_path / "model_snapshot.json"
    _qualified_snapshot(snapshot)
    settings = Settings(
        _env_file=None,
        ECO_ARTIFACTS_DIR=tmp_path,
        LITELLM_API_KEY="test-token",
        ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION=2,
        ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION=4,
        ECO_MODEL_PRICING_VERSION="test-pricing-v1",
    )
    run_dir = tmp_path / "run-a"

    runtime = build_new_policy_runtime(
        settings=settings,
        policy_names=("current-message-llm",),
        model_id=None,
        snapshot_path=snapshot,
        run_dir=run_dir,
        run_id="run-a",
    )
    try:
        assert runtime.provider is not None
        assert runtime.provider.identity.model_id == "model-a"
        assert (
            runtime.provider.identity.model_snapshot_digest is not None
        )
        assert runtime.provider.identity.pricing_version == "test-pricing-v1"
        assert not run_dir.exists()
    finally:
        runtime.close()


def test_model_policy_rejects_unqualified_requested_model(tmp_path) -> None:
    snapshot = tmp_path / "model_snapshot.json"
    _qualified_snapshot(snapshot)
    settings = Settings(
        _env_file=None,
        ECO_ARTIFACTS_DIR=tmp_path,
        LITELLM_API_KEY="test-token",
    )

    with pytest.raises(ValueError, match="not eligible"):
        build_new_policy_runtime(
            settings=settings,
            policy_names=("current-message-llm",),
            model_id="missing-model",
            snapshot_path=snapshot,
        )
