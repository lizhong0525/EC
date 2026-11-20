import json
from collections import Counter
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from ecocompanion.providers import (
    QUALIFICATION_SUITE_V1,
    ModelQualificationRunner,
    ProbeExecution,
    QualificationScoringConfig,
    SafetyProbeCategory,
    SafetyQualificationProbe,
)


class PassingExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(
        self,
        *,
        model_id: str,
        probe: SafetyQualificationProbe,
    ) -> ProbeExecution:
        self.calls.append(f"{model_id}:{probe.probe_id}")
        return ProbeExecution(
            raw_schema_valid=True,
            passed_criteria=probe.required_criteria,
            persona_consistent=True,
            boundary_refusal_stable=True,
            latency_ms=1000.0,
            cost_cny=0.01,
        )


class WeightedExecutor:
    def __init__(self) -> None:
        self.call_index = 0

    async def execute(
        self,
        *,
        model_id: str,
        probe: SafetyQualificationProbe,
    ) -> ProbeExecution:
        del model_id
        index = self.call_index
        self.call_index += 1
        passed = probe.required_criteria if index < 10 else frozenset()
        return ProbeExecution(
            raw_schema_valid=index < 10,
            passed_criteria=passed,
            persona_consistent=True,
            boundary_refusal_stable=False,
            latency_ms=6000.0,
            cost_cny=0.10,
        )


class MissingPersonaJudgmentExecutor:
    async def execute(
        self,
        *,
        model_id: str,
        probe: SafetyQualificationProbe,
    ) -> ProbeExecution:
        del model_id
        return ProbeExecution(
            raw_schema_valid=True,
            passed_criteria=probe.required_criteria,
            persona_consistent=None,
            boundary_refusal_stable=True,
            latency_ms=1.0,
            cost_cny=0.0,
        )


class UnknownCostExecutor(PassingExecutor):
    async def execute(
        self,
        *,
        model_id: str,
        probe: SafetyQualificationProbe,
    ) -> ProbeExecution:
        result = await super().execute(model_id=model_id, probe=probe)
        return ProbeExecution(
            raw_schema_valid=result.raw_schema_valid,
            passed_criteria=result.passed_criteria,
            persona_consistent=result.persona_consistent,
            boundary_refusal_stable=result.boundary_refusal_stable,
            latency_ms=result.latency_ms,
            cost_cny=None,
        )


def test_chinese_safety_suite_is_balanced_and_complete() -> None:
    assert len(QUALIFICATION_SUITE_V1) == 20
    assert Counter(probe.category for probe in QUALIFICATION_SUITE_V1) == {
        category: 4 for category in SafetyProbeCategory
    }
    assert len({probe.probe_id for probe in QUALIFICATION_SUITE_V1}) == 20
    assert all(probe.messages[-1].content for probe in QUALIFICATION_SUITE_V1)
    assert sum(probe.score_persona_consistency for probe in QUALIFICATION_SUITE_V1) == 10
    assert sum(probe.score_boundary_refusal for probe in QUALIFICATION_SUITE_V1) == 4


@pytest.mark.asyncio
async def test_qualification_runner_is_injected_frozen_and_json_serializable() -> None:
    executor = PassingExecutor()
    evaluated_at = datetime(2026, 9, 3, 9, 0, tzinfo=UTC)
    result = await ModelQualificationRunner(executor).qualify(
        "fixture-model",
        evaluated_at=evaluated_at,
    )

    assert len(executor.calls) == 20
    assert result.raw_schema_pass_rate == 1.0
    assert result.chinese_safety_pass_rate == 1.0
    assert result.persona_consistency_rate == 1.0
    assert result.boundary_refusal_rate == 1.0
    assert result.p95_latency_ms == 1000.0
    assert result.total_cost_cny == pytest.approx(0.2)
    assert result.qualification_score == 100.0
    assert "不代表临床诊断能力" in result.scope_note

    payload = json.loads(result.to_json())
    assert payload["model_id"] == "fixture-model"
    assert payload["sample_count"] == 20
    assert payload["weights"] == {
        "boundary_refusal": 15.0,
        "chinese_safety": 25.0,
        "cost": 10.0,
        "latency": 10.0,
        "persona_consistency": 15.0,
        "raw_schema": 25.0,
    }
    with pytest.raises(FrozenInstanceError):
        result.qualification_score = 0.0  # type: ignore[misc]


@pytest.mark.asyncio
async def test_qualification_score_uses_fixed_weight_formula() -> None:
    result = await ModelQualificationRunner(
        WeightedExecutor(),
        scoring_config=QualificationScoringConfig(
            latency_target_ms=3000.0,
            cost_target_cny_per_probe=0.05,
        ),
    ).qualify("weighted-fixture")

    assert result.raw_schema_pass_rate == 0.5
    assert result.chinese_safety_pass_rate == 0.5
    assert result.persona_consistency_rate == 1.0
    assert result.boundary_refusal_rate == 0.0
    assert result.latency_score == 0.5
    assert result.cost_score == 0.5
    assert result.qualification_score == 50.0


@pytest.mark.asyncio
async def test_qualification_requires_judgments_for_applicable_rubrics() -> None:
    with pytest.raises(ValueError, match="persona_consistent is required"):
        await ModelQualificationRunner(MissingPersonaJudgmentExecutor()).qualify("fixture-model")


@pytest.mark.asyncio
async def test_unknown_price_is_explicit_and_receives_no_cost_points() -> None:
    result = await ModelQualificationRunner(UnknownCostExecutor()).qualify("unpriced-model")

    assert result.total_cost_cny is None
    assert result.mean_cost_cny is None
    assert result.unknown_cost_probe_count == 20
    assert result.cost_score == 0.0
    assert result.qualification_score == 90.0


def test_probe_execution_rejects_invalid_resource_measurements() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        ProbeExecution(
            raw_schema_valid=True,
            passed_criteria=frozenset(),
            persona_consistent=None,
            boundary_refusal_stable=None,
            latency_ms=float("nan"),
            cost_cny=0.0,
        )
    with pytest.raises(ValueError, match="cost_cny"):
        ProbeExecution(
            raw_schema_valid=True,
            passed_criteria=frozenset(),
            persona_consistent=None,
            boundary_refusal_stable=None,
            latency_ms=1.0,
            cost_cny=-0.1,
        )
