"""Resumable, deterministic offline Track A experiment runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ecocompanion.benchmark.generation import DEFAULT_CASE_SEED
from ecocompanion.benchmark.metrics import evaluate_track_a
from ecocompanion.benchmark.model_provider import (
    BenchmarkInvocationScope,
    PolicyRuntime,
)
from ecocompanion.benchmark.splitting import DatasetSplit, canonical_split
from ecocompanion.benchmark.tracks import TrackARun, TrackARunner
from ecocompanion.domain.schemas import CaseSpecV1
from ecocompanion.experiments.io import (
    ArtifactConflictError,
    append_jsonl_model,
    read_json_object,
    read_jsonl_models,
    unique_records_by_id,
    write_json_atomic,
)
from ecocompanion.experiments.models import (
    ExperimentCheckpoint,
    ExperimentManifest,
    ExperimentResult,
    ExperimentRunOutcome,
    ExperimentStatus,
    StepDecisionRecord,
    TrackAMetricRecord,
)
from ecocompanion.experiments.policy_registry import (
    available_policy_names as available_policy_names,
)
from ecocompanion.experiments.policy_registry import (
    create_policy,
    describe_policy_runtimes,
    normalize_policy_names,
)
from ecocompanion.experiments.resumable import (
    build_checkpoint,
    validate_checkpoint,
    validate_max_units,
)
from ecocompanion.experiments.validation import (
    dataset_digest,
    load_benchmark_cases,
)

MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = "checkpoint.json"
RESULTS_NAME = "results.jsonl"


def start_experiment(
    *,
    run_dir: Path,
    split: DatasetSplit,
    policy_names: Sequence[str],
    surface_indices: Sequence[int],
    repeats: int,
    seed: int = DEFAULT_CASE_SEED,
    cases_path: Path | None = None,
    risk_threshold: float = 0.5,
    case_limit: int | None = None,
    run_id: str | None = None,
    max_units: int | None = None,
    policy_runtime: PolicyRuntime | None = None,
) -> ExperimentRunOutcome:
    """Create a write-once run and execute its pending work units."""

    destination = run_dir.resolve()
    if destination.exists():
        raise ArtifactConflictError(f"{destination} already exists; use `eco benchmark resume`")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not 0.0 < risk_threshold < 1.0:
        raise ValueError("risk_threshold must be between zero and one")
    if case_limit is not None and case_limit <= 0:
        raise ValueError("case_limit must be positive")
    validate_max_units(max_units)

    normalized_policies = normalize_policy_names(policy_names)
    effective_runtime = policy_runtime or PolicyRuntime()
    normalized_surfaces = _normalize_surface_indices(surface_indices)
    all_cases = load_benchmark_cases(cases_path=cases_path, seed=seed)
    selected_cases = canonical_split(all_cases).for_split(split)
    if case_limit is not None:
        selected_cases = selected_cases[:case_limit]
    _validate_surfaces(selected_cases, normalized_surfaces)

    selected_run_id = run_id or _default_run_id(split)
    source_path = str(cases_path.resolve()) if cases_path is not None else None
    manifest = _build_manifest(
        run_id=selected_run_id,
        all_cases=all_cases,
        selected_cases=selected_cases,
        dataset_path=source_path,
        seed=seed,
        split=split,
        policy_names=normalized_policies,
        surface_indices=normalized_surfaces,
        repeats=repeats,
        risk_threshold=risk_threshold,
        policy_runtime=effective_runtime,
    )

    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / MANIFEST_NAME,
        manifest,
        overwrite=False,
    )
    initial = build_checkpoint(manifest, ())
    write_json_atomic(
        destination / CHECKPOINT_NAME,
        initial,
        overwrite=False,
    )
    return _execute_pending(
        run_dir=destination,
        manifest=manifest,
        cases=selected_cases,
        completed={},
        max_units=max_units,
        policy_runtime=effective_runtime,
    )


def resume_experiment(
    run_dir: Path,
    *,
    max_units: int | None = None,
    policy_runtime: PolicyRuntime | None = None,
) -> ExperimentRunOutcome:
    """Resume a run without duplicating already persisted result units."""

    validate_max_units(max_units)
    source = run_dir.resolve()
    effective_runtime = policy_runtime or PolicyRuntime()
    manifest = ExperimentManifest.model_validate(read_json_object(source / MANIFEST_NAME))
    all_cases = load_benchmark_cases(
        cases_path=(Path(manifest.dataset_path) if manifest.dataset_path is not None else None),
        seed=manifest.dataset_seed,
    )
    if dataset_digest(all_cases) != manifest.dataset_digest:
        raise ArtifactConflictError("dataset content no longer matches the immutable run manifest")
    cases_by_id = {case.case_id: case for case in all_cases}
    try:
        selected_cases = tuple(cases_by_id[case_id] for case_id in manifest.selected_case_ids)
    except KeyError as exc:
        raise ArtifactConflictError(f"manifest references a missing case: {exc.args[0]}") from exc
    rebuilt = _build_manifest(
        run_id=manifest.run_id,
        all_cases=all_cases,
        selected_cases=selected_cases,
        dataset_path=manifest.dataset_path,
        seed=manifest.dataset_seed,
        split=manifest.split,
        policy_names=manifest.policy_names,
        surface_indices=manifest.surface_indices,
        repeats=manifest.repeats,
        risk_threshold=manifest.risk_threshold,
        policy_runtime=effective_runtime,
        created_at=manifest.created_at,
    )
    if rebuilt != manifest:
        raise ArtifactConflictError(
            "run manifest does not match the reconstructed experiment selection"
        )

    results = unique_records_by_id(
        read_jsonl_models(source / RESULTS_NAME, ExperimentResult),
        id_attribute="unit_id",
    )
    _validate_results_against_manifest(results, manifest)
    checkpoint = ExperimentCheckpoint.model_validate(read_json_object(source / CHECKPOINT_NAME))
    validate_checkpoint(checkpoint, manifest, set(results))

    if set(checkpoint.completed_unit_ids) != set(results):
        checkpoint = build_checkpoint(manifest, tuple(results))
        write_json_atomic(
            source / CHECKPOINT_NAME,
            checkpoint,
            overwrite=True,
        )
    if checkpoint.status is ExperimentStatus.COMPLETE:
        return ExperimentRunOutcome(
            run_dir=source,
            status=checkpoint.status,
            planned_count=checkpoint.planned_count,
            completed_count=checkpoint.completed_count,
            executed_count=0,
        )
    return _execute_pending(
        run_dir=source,
        manifest=manifest,
        cases=selected_cases,
        completed=results,
        max_units=max_units,
        policy_runtime=effective_runtime,
    )


def _build_manifest(
    *,
    run_id: str,
    all_cases: tuple[CaseSpecV1, ...],
    selected_cases: tuple[CaseSpecV1, ...],
    dataset_path: str | None,
    seed: int,
    split: DatasetSplit,
    policy_names: tuple[str, ...],
    surface_indices: tuple[int, ...],
    repeats: int,
    risk_threshold: float,
    policy_runtime: PolicyRuntime,
    created_at: datetime | None = None,
) -> ExperimentManifest:
    if not selected_cases:
        raise ValueError("selected split contains no cases")
    benchmark_versions = {case.benchmark_version for case in all_cases}
    if len(benchmark_versions) != 1:
        raise ValueError(f"dataset mixes benchmark versions: {sorted(benchmark_versions)}")
    unit_ids = tuple(
        _unit_id(
            case_id=case.case_id,
            policy_name=policy_name,
            surface_index=surface_index,
            repeat_index=repeat_index,
        )
        for case in selected_cases
        for policy_name in policy_names
        for surface_index in surface_indices
        for repeat_index in range(repeats)
    )
    policy_runtimes = describe_policy_runtimes(
        policy_names,
        policy_runtime,
    )
    fingerprint_payload = {
        "benchmark_version": next(iter(benchmark_versions)),
        "dataset_seed": seed,
        "dataset_path": dataset_path,
        "dataset_digest": dataset_digest(all_cases),
        "split": split.value,
        "policy_names": policy_names,
        "policy_runtimes": tuple(record.model_dump(mode="json") for record in policy_runtimes),
        "surface_indices": surface_indices,
        "repeats": repeats,
        "risk_threshold": risk_threshold,
        "selected_case_ids": tuple(case.case_id for case in selected_cases),
        "planned_unit_ids": unit_ids,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_payload: dict[str, object] = {
        "run_id": run_id,
        "benchmark_version": next(iter(benchmark_versions)),
        "dataset_seed": seed,
        "dataset_path": dataset_path,
        "dataset_digest": fingerprint_payload["dataset_digest"],
        "split": split,
        "policy_names": policy_names,
        "policy_runtimes": policy_runtimes,
        "surface_indices": surface_indices,
        "repeats": repeats,
        "risk_threshold": risk_threshold,
        "selected_case_ids": fingerprint_payload["selected_case_ids"],
        "planned_unit_ids": unit_ids,
        "run_fingerprint": fingerprint,
    }
    if created_at is not None:
        manifest_payload["created_at"] = created_at
    return ExperimentManifest.model_validate(manifest_payload)


def _execute_pending(
    *,
    run_dir: Path,
    manifest: ExperimentManifest,
    cases: tuple[CaseSpecV1, ...],
    completed: dict[str, ExperimentResult],
    max_units: int | None,
    policy_runtime: PolicyRuntime,
) -> ExperimentRunOutcome:
    cases_by_id = {case.case_id: case for case in cases}
    executed = 0
    for unit_id in manifest.planned_unit_ids:
        if unit_id in completed:
            continue
        if max_units is not None and executed >= max_units:
            break
        case_id, policy_name, surface_index, repeat_index = _parse_unit_id(unit_id)
        case = cases_by_id[case_id]
        unit_runtime = policy_runtime.for_unit(
            BenchmarkInvocationScope(
                run_id=manifest.run_id,
                unit_id=unit_id,
                seed=manifest.dataset_seed,
                replicate_id=repeat_index,
            )
        )
        policy = create_policy(policy_name, unit_runtime)
        track_run = TrackARunner().run(
            case,
            policy,
            surface_index=surface_index,
        )
        result = _result_from_run(
            manifest=manifest,
            unit_id=unit_id,
            repeat_index=repeat_index,
            run=track_run,
        )
        append_jsonl_model(run_dir / RESULTS_NAME, result)
        completed[unit_id] = result
        executed += 1
        write_json_atomic(
            run_dir / CHECKPOINT_NAME,
            build_checkpoint(manifest, tuple(completed)),
            overwrite=True,
        )

    final_checkpoint = build_checkpoint(manifest, tuple(completed))
    existing_checkpoint = ExperimentCheckpoint.model_validate(
        read_json_object(run_dir / CHECKPOINT_NAME)
    )
    if final_checkpoint.status is not existing_checkpoint.status:
        write_json_atomic(
            run_dir / CHECKPOINT_NAME,
            final_checkpoint,
            overwrite=True,
        )
    return ExperimentRunOutcome(
        run_dir=run_dir,
        status=final_checkpoint.status,
        planned_count=final_checkpoint.planned_count,
        completed_count=final_checkpoint.completed_count,
        executed_count=executed,
    )


def _result_from_run(
    *,
    manifest: ExperimentManifest,
    unit_id: str,
    repeat_index: int,
    run: TrackARun,
) -> ExperimentResult:
    metrics = evaluate_track_a(run, risk_threshold=manifest.risk_threshold)
    return ExperimentResult(
        unit_id=unit_id,
        run_id=manifest.run_id,
        run_fingerprint=manifest.run_fingerprint,
        case_id=run.case.case_id,
        split=manifest.split,
        policy_name=run.policy_name,
        surface_index=run.surface_index,
        repeat_index=repeat_index,
        risk_trajectory=run.case.risk_trajectory,
        social_ecology=run.case.social_ecology,
        expression_style=run.case.expression_style,
        metrics=TrackAMetricRecord.model_validate(asdict(metrics)),
        steps=tuple(
            StepDecisionRecord(
                step_id=step.reference.step_id,
                observation_id=step.observation.current_observation.observation_id,
                message=step.observation.current_observation.message,
                depression_worsening=step.decision.assessment.depression_worsening,
                immediate_safety_risk=step.decision.assessment.immediate_safety_risk,
                dependency_risk=step.decision.assessment.dependency_risk,
                uncertainty=step.decision.uncertainty,
                evidence_refs=step.decision.evidence,
                evidence_codes=tuple(evidence.evidence_code for evidence in step.decision.evidence),
                primary_action=step.decision.decision.primary_action,
                escalation_level=step.decision.decision.escalation_level,
                response_text=step.decision.response.text,
            )
            for step in run.steps
        ),
    )


def _validate_results_against_manifest(
    results: dict[str, ExperimentResult],
    manifest: ExperimentManifest,
) -> None:
    planned = set(manifest.planned_unit_ids)
    unexpected = set(results) - planned
    if unexpected:
        raise ArtifactConflictError(f"results contain unplanned unit ids: {sorted(unexpected)}")
    mismatched = [
        result.unit_id
        for result in results.values()
        if result.run_id != manifest.run_id
        or result.run_fingerprint != manifest.run_fingerprint
        or result.split is not manifest.split
    ]
    if mismatched:
        raise ArtifactConflictError(f"results do not match the run manifest: {sorted(mismatched)}")


def _normalize_surface_indices(indices: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(indices))
    if not normalized:
        raise ValueError("at least one surface index is required")
    if any(index < 0 for index in normalized):
        raise ValueError("surface indices cannot be negative")
    return normalized


def _validate_surfaces(
    cases: Sequence[CaseSpecV1],
    surface_indices: tuple[int, ...],
) -> None:
    for case in cases:
        for surface_index in surface_indices:
            if any(surface_index >= len(step.frozen_utterances) for step in case.timeline):
                raise ValueError(f"{case.case_id} does not provide surface {surface_index}")


def _unit_id(
    *,
    case_id: str,
    policy_name: str,
    surface_index: int,
    repeat_index: int,
) -> str:
    return (
        f"{case_id}__policy={policy_name}__surface={surface_index:02d}__repeat={repeat_index:03d}"
    )


def _parse_unit_id(unit_id: str) -> tuple[str, str, int, int]:
    try:
        case_id, policy_part, surface_part, repeat_part = unit_id.split("__")
        policy_name = policy_part.removeprefix("policy=")
        surface_index = int(surface_part.removeprefix("surface="))
        repeat_index = int(repeat_part.removeprefix("repeat="))
    except (ValueError, TypeError) as exc:
        raise ArtifactConflictError(f"invalid planned unit id: {unit_id}") from exc
    if (
        not policy_part.startswith("policy=")
        or not surface_part.startswith("surface=")
        or not repeat_part.startswith("repeat=")
    ):
        raise ArtifactConflictError(f"invalid planned unit id: {unit_id}")
    return case_id, policy_name, surface_index, repeat_index


def _default_run_id(split: DatasetSplit) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"track-a-{split.value}-{timestamp}"
