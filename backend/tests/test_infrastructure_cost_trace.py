from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from ecocompanion.domain.enums import Action, Track
from ecocompanion.domain.schemas import (
    CurrentObservation,
    HiddenState,
    ModelCallUsage,
    ObservationV1,
    RiskEstimate,
    SafetyAudit,
    TraceEventV1,
    VisibleProfile,
)
from ecocompanion.infrastructure import (
    CostConflictError,
    CostLedger,
    JsonlTraceStore,
    ManifestConflictError,
    ManifestModel,
    ManifestRoleBinding,
    ModelPricing,
    RunManifest,
    RunManifestStore,
)
from ecocompanion.providers import CompletionUsage, ModelRole


def test_cost_ledger_calculates_cost_and_is_idempotent(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 9, 3, tzinfo=UTC)
    ledger = CostLedger(
        pricing={
            "model-a": ModelPricing(
                input_per_million=Decimal("2"),
                cached_input_per_million=Decimal("1"),
                output_per_million=Decimal("4"),
            )
        },
        pricing_version="2026-09-03",
        path=tmp_path / "costs.jsonl",
    )
    usage = CompletionUsage(
        input_tokens=1_000_000,
        cached_input_tokens=500_000,
        output_tokens=1_000_000,
    )
    ledger.record_call(
        call_id="call-1",
        role=ModelRole.EVALUATED_AGENT,
        model="model-a",
        usage=usage,
        recorded_at=recorded_at,
    )
    ledger.record_call(
        call_id="call-1",
        role=ModelRole.EVALUATED_AGENT,
        model="model-a",
        usage=usage,
    )

    summary = ledger.summary()
    assert summary.call_count == 1
    assert summary.known_cost_cny == Decimal("5.5")
    assert summary.unknown_price_calls == 0

    reloaded = CostLedger(path=tmp_path / "costs.jsonl")
    assert reloaded.records()[0].cost_cny == Decimal("5.5")


def test_cost_ledger_tracks_unknown_prices_and_conflicts() -> None:
    ledger = CostLedger()
    ledger.record_call(
        call_id="unknown-1",
        role=ModelRole.JUDGE,
        model="unknown-model",
        usage=CompletionUsage(input_tokens=10, output_tokens=2),
    )
    assert ledger.summary().unknown_price_calls == 1

    with pytest.raises(CostConflictError):
        ledger.record_call(
            call_id="unknown-1",
            role=ModelRole.JUDGE,
            model="different-model",
            usage=CompletionUsage(input_tokens=10, output_tokens=2),
        )


def hidden_state() -> HiddenState:
    return HiddenState(
        depression_burden=0.4,
        acute_safety_risk=0.1,
        trust_in_companion=0.3,
        disclosure_readiness=0.4,
        offline_help_willingness=0.5,
        ai_dependency=0.1,
        family_support=0.7,
        peer_support=0.6,
        school_support=0.5,
    )


def trace_event() -> TraceEventV1:
    observation = ObservationV1(
        track=Track.A,
        episode_id="episode-1",
        step_id=0,
        simulated_time="day-01",
        visible_profile=VisibleProfile(age_group="15-17"),
        current_observation=CurrentObservation(
            observation_id="obs-1",
            message="最近有点累。",
        ),
        allowed_primary_actions=(Action.CLARIFY_SYMPTOMS,),
    )
    return TraceEventV1(
        trace_id="trace-1",
        scenario_id="scenario-1",
        episode_id="episode-1",
        step_id=0,
        external_event={"api_key": "must-not-persist", "type": "school"},
        observation=observation,
        assessment=RiskEstimate(
            depression_worsening=0.4,
            immediate_safety_risk=0.1,
            dependency_risk=0.1,
        ),
        selected_action=Action.CLARIFY_SYMPTOMS,
        response_text="这种疲惫持续多久了？",
        safety_audit=SafetyAudit(passed=True),
        world_state_before=hidden_state(),
        world_state_after=hidden_state(),
        random_seed=7,
        model_usage=(
            ModelCallUsage(
                model="model-a",
                input_tokens=10,
                output_tokens=5,
            ),
        ),
    )


def test_jsonl_trace_store_round_trip_and_redaction(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    store = JsonlTraceStore(path)
    store.append(trace_event())

    raw = path.read_text(encoding="utf-8")
    assert "must-not-persist" not in raw
    events = store.read_all()
    assert events[0].external_event["api_key"] == "[REDACTED]"
    assert events[0].trace_id == "trace-1"


def test_jsonl_trace_store_redacts_credential_values_and_alias_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces-with-credentials.jsonl"
    store = JsonlTraceStore(path)
    key = "sk-" + "example_key_material_1234567890"
    bearer = "Bearer " + "secret-token-material-123456"
    event = trace_event().model_copy(
        update={
            "external_event": {
                "LITELLM_API_KEY": key,
                "nested": {"token": bearer},
            },
            "response_text": f"用户误粘贴了 {key} 和 {bearer}",
        }
    )

    store.append(event)

    raw = path.read_text(encoding="utf-8")
    assert key not in raw
    assert bearer not in raw
    assert raw.count("[REDACTED]") >= 4
    restored = store.read_all()[0]
    assert restored.external_event["LITELLM_API_KEY"] == "[REDACTED]"
    assert restored.external_event["nested"]["token"] == "[REDACTED]"
    assert restored.response_text == "用户误粘贴了 [REDACTED] 和 [REDACTED]"


def run_manifest(run_id: str = "run-1") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        benchmark_version="ecotriage-v1",
        gateway_base_url="https://relay.example/v1",
        models=(
            ManifestModel(
                model_id="agent-a",
                family="family-a",
                capability_status="ready",
            ),
        ),
        role_bindings=(
            ManifestRoleBinding(
                role=ModelRole.EVALUATED_AGENT,
                model_id="agent-a",
                family="family-a",
            ),
        ),
        prompt_versions={"agent": "v1"},
        schema_versions={"agent": "1.0"},
        random_seed=7,
    )


def test_run_manifest_is_write_once(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path / "manifest.json")
    manifest = run_manifest()
    store.write(manifest)
    store.write(manifest)
    assert store.read() == manifest

    with pytest.raises(ManifestConflictError):
        store.write(run_manifest(run_id="run-2"))


def test_research_manifest_forbids_cross_model_fallback() -> None:
    with pytest.raises(ValidationError, match="forbids cross-model fallback"):
        RunManifest(
            **run_manifest().model_dump(exclude={"cross_model_fallback_allowed"}),
            cross_model_fallback_allowed=True,
        )
