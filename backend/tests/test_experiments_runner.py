import json

import pytest

from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.experiments.io import ArtifactConflictError
from ecocompanion.experiments.models import (
    ExperimentCheckpoint,
    ExperimentManifest,
    ExperimentResult,
    ExperimentStatus,
)
from ecocompanion.experiments.runner import (
    CHECKPOINT_NAME,
    MANIFEST_NAME,
    RESULTS_NAME,
    resume_experiment,
    start_experiment,
)


def test_offline_run_is_resumable_and_idempotent(tmp_path) -> None:
    run_dir = tmp_path / "partial-run"
    partial = start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.PUBLIC_TEST,
        policy_names=("always-listen", "keyword-triage"),
        surface_indices=(0,),
        repeats=2,
        case_limit=1,
        run_id="partial-run",
        max_units=2,
    )

    assert partial.status is ExperimentStatus.IN_PROGRESS
    assert partial.completed_count == 2
    assert partial.planned_count == 4
    assert partial.executed_count == 2
    assert len(_read_nonempty_lines(run_dir / RESULTS_NAME)) == 2

    completed = resume_experiment(run_dir)
    assert completed.status is ExperimentStatus.COMPLETE
    assert completed.completed_count == completed.planned_count == 4
    assert completed.executed_count == 2

    artifacts_before = {
        name: (run_dir / name).read_bytes()
        for name in (MANIFEST_NAME, CHECKPOINT_NAME, RESULTS_NAME)
    }
    repeated = resume_experiment(run_dir)
    artifacts_after = {
        name: (run_dir / name).read_bytes()
        for name in (MANIFEST_NAME, CHECKPOINT_NAME, RESULTS_NAME)
    }

    assert repeated.status is ExperimentStatus.COMPLETE
    assert repeated.executed_count == 0
    assert artifacts_after == artifacts_before

    manifest = ExperimentManifest.model_validate_json(
        (run_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    checkpoint = ExperimentCheckpoint.model_validate_json(
        (run_dir / CHECKPOINT_NAME).read_text(encoding="utf-8")
    )
    results = tuple(
        ExperimentResult.model_validate_json(line)
        for line in _read_nonempty_lines(run_dir / RESULTS_NAME)
    )
    assert checkpoint.completed_unit_ids == manifest.planned_unit_ids
    assert tuple(result.unit_id for result in results) == manifest.planned_unit_ids
    assert all(len(result.steps) == 14 for result in results)
    assert "depression_burden" not in (run_dir / RESULTS_NAME).read_text(
        encoding="utf-8"
    )


def test_new_run_refuses_to_replace_an_existing_directory(tmp_path) -> None:
    run_dir = tmp_path / "run"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=("always-listen",),
        surface_indices=(0,),
        repeats=1,
        case_limit=1,
        run_id="run",
    )

    with pytest.raises(ArtifactConflictError, match="resume"):
        start_experiment(
            run_dir=run_dir,
            split=DatasetSplit.DEV,
            policy_names=("always-listen",),
            surface_indices=(0,),
            repeats=1,
            case_limit=1,
            run_id="run",
        )


def test_resume_rejects_checkpoint_ahead_of_results(tmp_path) -> None:
    run_dir = tmp_path / "run"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=("always-listen",),
        surface_indices=(0,),
        repeats=2,
        case_limit=1,
        run_id="run",
    )
    result_lines = _read_nonempty_lines(run_dir / RESULTS_NAME)
    (run_dir / RESULTS_NAME).write_text(
        result_lines[0] + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactConflictError, match="ahead"):
        resume_experiment(run_dir)


def test_manifest_contains_no_api_credentials(tmp_path) -> None:
    run_dir = tmp_path / "run"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.TRAIN,
        policy_names=("history-aware-rule",),
        surface_indices=(1,),
        repeats=1,
        case_limit=1,
        run_id="run",
    )

    payload = json.loads((run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    serialized = json.dumps(payload).lower()
    assert "api_key" not in serialized
    assert "authorization" not in serialized


def _read_nonempty_lines(path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
