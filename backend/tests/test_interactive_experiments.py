import json

import pytest

from ecocompanion.benchmark.model_provider import (
    BenchmarkProviderIdentity,
    DecisionProviderRequest,
    PolicyRuntime,
    ResponseProviderRequest,
)
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.benchmark.track_c import TrackCConfiguration
from ecocompanion.domain.enums import Action, EscalationTarget, RiskLevel
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    DecisionPlan,
    ResponsePayload,
    RiskEstimate,
)
from ecocompanion.experiments.interactive_runner import (
    resume_benchmark_experiment,
    resume_track_b_experiment,
    resume_track_c_experiment,
    start_track_b_experiment,
    start_track_c_experiment,
)
from ecocompanion.experiments.io import ArtifactConflictError
from ecocompanion.experiments.models import (
    ExperimentCheckpoint,
    ExperimentStatus,
    TrackBExperimentManifest,
    TrackBExperimentResult,
    TrackCExperimentManifest,
    TrackCExperimentResult,
)
from ecocompanion.experiments.policy_registry import available_policy_names
from ecocompanion.experiments.runner import (
    CHECKPOINT_NAME,
    MANIFEST_NAME,
    RESULTS_NAME,
    start_experiment,
)


class _FakeBenchmarkProvider:
    identity = BenchmarkProviderIdentity(
        provider_name="fake",
        model_id="fake-model",
        model_snapshot_digest="a" * 64,
    )

    def decide(
        self,
        request: DecisionProviderRequest,
    ) -> AgentDecisionV1:
        action = (
            Action.SUPPORTIVE_LISTENING
            if Action.SUPPORTIVE_LISTENING in request.allowed_actions
            else request.allowed_actions[0]
        )
        return AgentDecisionV1(
            assessment=RiskEstimate(
                depression_worsening=0.0,
                immediate_safety_risk=0.0,
                dependency_risk=0.0,
            ),
            uncertainty=0.5,
            decision=DecisionPlan(
                primary_action=action,
                escalation_level=RiskLevel.L0,
                escalation_target=EscalationTarget.NONE,
            ),
            response=ResponsePayload(text="我在听。"),
        )

    def render(
        self,
        request: ResponseProviderRequest,
    ) -> ResponsePayload:
        del request
        return ResponsePayload(text="我在听。")

    def close(self) -> None:
        return None


def test_track_b_run_is_resumable_idempotent_and_hides_world_state(
    tmp_path,
) -> None:
    run_dir = tmp_path / "track-b"
    partial = start_track_b_experiment(
        run_dir=run_dir,
        policy_names=("history-aware-rule",),
        case_limit=1,
        run_id="track-b",
        max_units=2,
    )

    assert partial.status is ExperimentStatus.IN_PROGRESS
    assert partial.completed_count == 2
    assert partial.planned_count == 5

    completed = resume_track_b_experiment(run_dir)
    assert completed.status is ExperimentStatus.COMPLETE
    assert completed.completed_count == completed.planned_count == 5
    assert completed.executed_count == 3

    artifacts_before = _artifact_bytes(run_dir)
    repeated = resume_benchmark_experiment(run_dir)
    assert repeated.executed_count == 0
    assert _artifact_bytes(run_dir) == artifacts_before

    manifest = TrackBExperimentManifest.model_validate_json(
        (run_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    checkpoint = ExperimentCheckpoint.model_validate_json(
        (run_dir / CHECKPOINT_NAME).read_text(encoding="utf-8")
    )
    results = tuple(
        TrackBExperimentResult.model_validate_json(line)
        for line in _nonempty_lines(run_dir / RESULTS_NAME)
    )
    assert manifest.track == "B"
    assert checkpoint.completed_unit_ids == manifest.planned_unit_ids
    assert tuple(result.unit_id for result in results) == manifest.planned_unit_ids
    serialized = (run_dir / RESULTS_NAME).read_text(encoding="utf-8")
    assert '"world_state"' not in serialized
    assert '"depression_burden"' not in serialized


def test_track_c_run_is_resumable_and_preserves_fixed_configuration(
    tmp_path,
) -> None:
    run_dir = tmp_path / "track-c"
    partial = start_track_c_experiment(
        run_dir=run_dir,
        configurations=(
            TrackCConfiguration.PLAIN_ASSISTANT,
            TrackCConfiguration.MEMORY_PROACTIVE_ECO_SAFE,
        ),
        scenario_limit=1,
        run_id="track-c",
        max_units=3,
    )

    assert partial.status is ExperimentStatus.IN_PROGRESS
    assert partial.completed_count == 3
    assert partial.planned_count == 10

    completed = resume_track_c_experiment(run_dir)
    assert completed.status is ExperimentStatus.COMPLETE
    assert completed.completed_count == completed.planned_count == 10
    assert completed.executed_count == 7

    artifacts_before = _artifact_bytes(run_dir)
    repeated = resume_benchmark_experiment(run_dir)
    assert repeated.executed_count == 0
    assert _artifact_bytes(run_dir) == artifacts_before

    manifest = TrackCExperimentManifest.model_validate_json(
        (run_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    results = tuple(
        TrackCExperimentResult.model_validate_json(line)
        for line in _nonempty_lines(run_dir / RESULTS_NAME)
    )
    assert manifest.track == "C"
    assert {result.spec.configuration for result in results} == {
        TrackCConfiguration.PLAIN_ASSISTANT,
        TrackCConfiguration.MEMORY_PROACTIVE_ECO_SAFE,
    }


def test_generic_resume_keeps_existing_track_a_manifest_compatible(
    tmp_path,
) -> None:
    run_dir = tmp_path / "track-a"
    started = start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=("always-listen",),
        surface_indices=(0,),
        repeats=1,
        case_limit=1,
        run_id="track-a",
    )

    assert started.status is ExperimentStatus.COMPLETE
    manifest_before = (run_dir / MANIFEST_NAME).read_bytes()
    resumed = resume_benchmark_experiment(run_dir)

    assert resumed.status is ExperimentStatus.COMPLETE
    assert resumed.executed_count == 0
    assert (run_dir / MANIFEST_NAME).read_bytes() == manifest_before


def test_track_b_resume_rejects_a_tampered_replay_digest(tmp_path) -> None:
    run_dir = tmp_path / "track-b"
    start_track_b_experiment(
        run_dir=run_dir,
        policy_names=("always-listen",),
        case_limit=1,
        run_id="track-b",
    )
    lines = _nonempty_lines(run_dir / RESULTS_NAME)
    payload = json.loads(lines[0])
    payload["replay_digest"] = "0" * 64
    lines[0] = json.dumps(payload, ensure_ascii=False)
    (run_dir / RESULTS_NAME).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactConflictError, match="do not match"):
        resume_track_b_experiment(run_dir)


def test_every_registered_policy_can_execute_a_track_b_unit(tmp_path) -> None:
    policies = available_policy_names()
    run_dir = tmp_path / "all-policies"

    outcome = start_track_b_experiment(
        run_dir=run_dir,
        policy_names=policies,
        case_limit=1,
        run_id="all-policies",
        max_units=len(policies),
        policy_runtime=PolicyRuntime(provider=_FakeBenchmarkProvider()),
    )

    assert outcome.executed_count == len(policies)
    results = tuple(
        TrackBExperimentResult.model_validate_json(line)
        for line in _nonempty_lines(run_dir / RESULTS_NAME)
    )
    assert {result.policy_name for result in results} == set(policies)


def _artifact_bytes(run_dir) -> dict[str, bytes]:
    return {
        name: (run_dir / name).read_bytes()
        for name in (MANIFEST_NAME, CHECKPOINT_NAME, RESULTS_NAME)
    }


def _nonempty_lines(path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
