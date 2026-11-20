"""Resumable local experiment runners for interactive Tracks B and C."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ecocompanion.benchmark.generation import DEFAULT_CASE_SEED
from ecocompanion.benchmark.model_provider import (
    BenchmarkInvocationScope,
    PolicyRuntime,
)
from ecocompanion.benchmark.track_b import (
    TRACK_B_SEEDS,
    TRACK_B_SELECTION_SEED,
    TrackBDisclosureEngine,
    TrackBEpisodeResult,
    TrackBEpisodeRunner,
    TrackBEpisodeSpec,
    build_track_b_episode_specs,
    select_track_b_cases,
)
from ecocompanion.benchmark.track_c import (
    TRACK_C_SEEDS,
    DeterministicTrackCBaselineResponder,
    TrackCConfiguration,
    TrackCEpisodeResult,
    TrackCEpisodeRunner,
    TrackCEpisodeSpec,
    TrackCResponder,
    TrackCScenario,
    build_track_c_episode_matrix,
    define_track_c_scenarios,
    track_c_features,
)
from ecocompanion.domain.enums import Track
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
    ExperimentRunOutcome,
    ExperimentStatus,
    InteractiveRuntimeRecord,
    TrackBEpisodeSpecRecord,
    TrackBExperimentManifest,
    TrackBExperimentResult,
    TrackBMetricRecord,
    TrackBTurnRecord,
    TrackCEpisodeSpecRecord,
    TrackCExperimentManifest,
    TrackCExperimentResult,
    TrackCMetricRecord,
    TrackCTurnRecord,
)
from ecocompanion.experiments.policy_registry import (
    create_policy,
    describe_policy_runtimes,
    normalize_policy_names,
    policy_requires_provider,
)
from ecocompanion.experiments.resumable import (
    build_checkpoint,
    validate_checkpoint,
    validate_max_units,
)
from ecocompanion.experiments.runner import (
    CHECKPOINT_NAME,
    MANIFEST_NAME,
    RESULTS_NAME,
    resume_experiment,
)
from ecocompanion.experiments.validation import (
    dataset_digest,
    load_benchmark_cases,
)

if TYPE_CHECKING:
    from ecocompanion.benchmark.api_factory import InteractiveAPIComponents
    from ecocompanion.experiments.interactive_api_runtime import (
        FormalInteractiveRuntime,
    )

_BENCHMARK_VERSION = "ecotriage-v1"
_FORMAL_PROVIDER_NAME = "litellm-openai-compatible"


def start_track_b_experiment(
    *,
    run_dir: Path,
    policy_names: Sequence[str],
    seed: int = DEFAULT_CASE_SEED,
    cases_path: Path | None = None,
    selection_seed: int = TRACK_B_SELECTION_SEED,
    case_limit: int | None = None,
    run_id: str | None = None,
    max_units: int | None = None,
    policy_runtime: PolicyRuntime | None = None,
    interactive_runtime: FormalInteractiveRuntime | None = None,
) -> ExperimentRunOutcome:
    """Create a Track B run in deterministic local or explicit formal API mode."""

    destination = _new_run_destination(run_dir)
    _validate_limit(case_limit, "case_limit")
    validate_max_units(max_units)
    normalized_policies = normalize_policy_names(policy_names)
    selected_run_id = run_id or _default_run_id(Track.B)
    effective_runtime, owns_runtime = _prepare_track_b_policy_runtime(
        policy_runtime=policy_runtime,
        interactive_runtime=interactive_runtime,
        policy_names=normalized_policies,
        run_dir=destination,
        run_id=selected_run_id,
    )
    runtime_record = _interactive_runtime_record(interactive_runtime)
    all_cases = load_benchmark_cases(cases_path=cases_path, seed=seed)
    selected_cases, specs = _select_track_b_specs(
        all_cases,
        selection_seed=selection_seed,
        case_limit=case_limit,
    )
    manifest = _build_track_b_manifest(
        run_id=selected_run_id,
        all_cases=all_cases,
        selected_cases=selected_cases,
        specs=specs,
        dataset_path=(
            str(cases_path.resolve()) if cases_path is not None else None
        ),
        dataset_seed=seed,
        selection_seed=selection_seed,
        policy_names=normalized_policies,
        policy_runtime=effective_runtime,
        interactive_runtime=runtime_record,
    )
    _initialize_run(destination, manifest)
    api_components = _activate_api_runtime(
        interactive_runtime,
        run_dir=destination,
        run_id=manifest.run_id,
    )
    execution_runtime = _track_b_policy_runtime(
        manifest,
        configured_runtime=effective_runtime,
        interactive_runtime=interactive_runtime,
        api_components=api_components,
    )
    try:
        return _execute_track_b_pending(
            run_dir=destination,
            manifest=manifest,
            cases=selected_cases,
            specs=specs,
            completed={},
            max_units=max_units,
            policy_runtime=execution_runtime,
            api_components=api_components,
        )
    finally:
        if owns_runtime:
            effective_runtime.close()


def resume_track_b_experiment(
    run_dir: Path,
    *,
    max_units: int | None = None,
    policy_runtime: PolicyRuntime | None = None,
    interactive_runtime: FormalInteractiveRuntime | None = None,
) -> ExperimentRunOutcome:
    """Resume a Track B run after reconstructing its frozen selection."""

    validate_max_units(max_units)
    source = run_dir.resolve()
    manifest = TrackBExperimentManifest.model_validate(
        read_json_object(source / MANIFEST_NAME)
    )
    runtime_record = _validate_interactive_runtime(
        manifest.interactive_runtime,
        interactive_runtime,
    )
    effective_runtime, owns_runtime = _prepare_track_b_policy_runtime(
        policy_runtime=policy_runtime,
        interactive_runtime=interactive_runtime,
        policy_names=manifest.policy_names,
        run_dir=source,
        run_id=manifest.run_id,
    )
    all_cases = load_benchmark_cases(
        cases_path=(
            Path(manifest.dataset_path)
            if manifest.dataset_path is not None
            else None
        ),
        seed=manifest.dataset_seed,
    )
    if dataset_digest(all_cases) != manifest.dataset_digest:
        raise ArtifactConflictError(
            "dataset content no longer matches the immutable run manifest"
        )
    selected_cases, specs = _reconstruct_track_b_selection(
        all_cases,
        manifest=manifest,
    )
    rebuilt = _build_track_b_manifest(
        run_id=manifest.run_id,
        all_cases=all_cases,
        selected_cases=selected_cases,
        specs=specs,
        dataset_path=manifest.dataset_path,
        dataset_seed=manifest.dataset_seed,
        selection_seed=manifest.selection_seed,
        policy_names=manifest.policy_names,
        policy_runtime=effective_runtime,
        interactive_runtime=runtime_record,
        created_at=manifest.created_at,
    )
    if rebuilt != manifest:
        raise ArtifactConflictError(
            "run manifest does not match the reconstructed Track B selection"
        )

    results = unique_records_by_id(
        read_jsonl_models(source / RESULTS_NAME, TrackBExperimentResult),
        id_attribute="unit_id",
    )
    _validate_track_b_results(results, manifest, specs)
    checkpoint = _load_and_validate_checkpoint(source, manifest, set(results))
    if set(checkpoint.completed_unit_ids) != set(results):
        checkpoint = build_checkpoint(manifest, tuple(results))
        write_json_atomic(
            source / CHECKPOINT_NAME,
            checkpoint,
            overwrite=True,
        )
    if checkpoint.status is ExperimentStatus.COMPLETE:
        if owns_runtime:
            effective_runtime.close()
        return _completed_outcome(source, checkpoint)
    api_components = _activate_api_runtime(
        interactive_runtime,
        run_dir=source,
        run_id=manifest.run_id,
    )
    execution_runtime = _track_b_policy_runtime(
        manifest,
        configured_runtime=effective_runtime,
        interactive_runtime=interactive_runtime,
        api_components=api_components,
    )
    try:
        return _execute_track_b_pending(
            run_dir=source,
            manifest=manifest,
            cases=selected_cases,
            specs=specs,
            completed=results,
            max_units=max_units,
            policy_runtime=execution_runtime,
            api_components=api_components,
        )
    finally:
        if owns_runtime:
            effective_runtime.close()


def start_track_c_experiment(
    *,
    run_dir: Path,
    configurations: Sequence[TrackCConfiguration],
    scenario_limit: int | None = None,
    run_id: str | None = None,
    max_units: int | None = None,
    interactive_runtime: FormalInteractiveRuntime | None = None,
) -> ExperimentRunOutcome:
    """Create a Track C run in deterministic local or explicit formal API mode."""

    destination = _new_run_destination(run_dir)
    _validate_limit(scenario_limit, "scenario_limit")
    validate_max_units(max_units)
    normalized_configurations = _normalize_configurations(configurations)
    all_scenarios = define_track_c_scenarios()
    selected_scenarios, specs = _select_track_c_specs(
        all_scenarios,
        configurations=normalized_configurations,
        scenario_limit=scenario_limit,
    )
    runtime_record = _interactive_runtime_record(interactive_runtime)
    manifest = _build_track_c_manifest(
        run_id=run_id or _default_run_id(Track.C),
        all_scenarios=all_scenarios,
        selected_scenarios=selected_scenarios,
        specs=specs,
        configurations=normalized_configurations,
        interactive_runtime=runtime_record,
    )
    _initialize_run(destination, manifest)
    api_components = _activate_api_runtime(
        interactive_runtime,
        run_dir=destination,
        run_id=manifest.run_id,
    )
    return _execute_track_c_pending(
        run_dir=destination,
        manifest=manifest,
        scenarios=selected_scenarios,
        specs=specs,
        completed={},
        max_units=max_units,
        responder=_track_c_responder(
            manifest,
            api_components=api_components,
        ),
    )


def resume_track_c_experiment(
    run_dir: Path,
    *,
    max_units: int | None = None,
    interactive_runtime: FormalInteractiveRuntime | None = None,
) -> ExperimentRunOutcome:
    """Resume a Track C run after validating fixed scenarios and responder."""

    validate_max_units(max_units)
    source = run_dir.resolve()
    manifest = TrackCExperimentManifest.model_validate(
        read_json_object(source / MANIFEST_NAME)
    )
    runtime_record = _validate_interactive_runtime(
        manifest.interactive_runtime,
        interactive_runtime,
    )
    all_scenarios = define_track_c_scenarios()
    if _scenario_digest(all_scenarios) != manifest.scenario_digest:
        raise ArtifactConflictError(
            "Track C scenario content no longer matches the immutable manifest"
        )
    selected_scenarios, specs = _reconstruct_track_c_selection(
        all_scenarios,
        manifest=manifest,
    )
    rebuilt = _build_track_c_manifest(
        run_id=manifest.run_id,
        all_scenarios=all_scenarios,
        selected_scenarios=selected_scenarios,
        specs=specs,
        configurations=manifest.configurations,
        interactive_runtime=runtime_record,
        created_at=manifest.created_at,
    )
    if rebuilt != manifest:
        raise ArtifactConflictError(
            "run manifest does not match the reconstructed Track C selection"
        )

    results = unique_records_by_id(
        read_jsonl_models(source / RESULTS_NAME, TrackCExperimentResult),
        id_attribute="unit_id",
    )
    _validate_track_c_results(results, manifest, specs)
    checkpoint = _load_and_validate_checkpoint(source, manifest, set(results))
    if set(checkpoint.completed_unit_ids) != set(results):
        checkpoint = build_checkpoint(manifest, tuple(results))
        write_json_atomic(
            source / CHECKPOINT_NAME,
            checkpoint,
            overwrite=True,
        )
    if checkpoint.status is ExperimentStatus.COMPLETE:
        return _completed_outcome(source, checkpoint)
    api_components = _activate_api_runtime(
        interactive_runtime,
        run_dir=source,
        run_id=manifest.run_id,
    )
    return _execute_track_c_pending(
        run_dir=source,
        manifest=manifest,
        scenarios=selected_scenarios,
        specs=specs,
        completed=results,
        max_units=max_units,
        responder=_track_c_responder(
            manifest,
            api_components=api_components,
        ),
    )


def resume_benchmark_experiment(
    run_dir: Path,
    *,
    max_units: int | None = None,
    policy_runtime: PolicyRuntime | None = None,
    interactive_runtime: FormalInteractiveRuntime | None = None,
) -> ExperimentRunOutcome:
    """Dispatch resume by the immutable manifest's explicit track."""

    source = run_dir.resolve()
    payload = read_json_object(source / MANIFEST_NAME)
    raw_track = payload.get("track", Track.A.value)
    try:
        track = Track(raw_track)
    except (TypeError, ValueError) as exc:
        raise ArtifactConflictError(
            f"manifest declares unsupported track: {raw_track!r}"
        ) from exc
    if track is Track.A:
        if interactive_runtime is not None:
            raise ArtifactConflictError(
                "Track A does not use an interactive API runtime"
            )
        return resume_experiment(
            source,
            max_units=max_units,
            policy_runtime=policy_runtime,
        )
    if track is Track.B:
        return resume_track_b_experiment(
            source,
            max_units=max_units,
            policy_runtime=policy_runtime,
            interactive_runtime=interactive_runtime,
        )
    return resume_track_c_experiment(
        source,
        max_units=max_units,
        interactive_runtime=interactive_runtime,
    )


def _interactive_runtime_record(
    runtime: FormalInteractiveRuntime | None,
) -> InteractiveRuntimeRecord:
    if runtime is None:
        from ecocompanion.experiments.interactive_api_runtime import (
            local_interactive_identity_payload,
        )

        payload = local_interactive_identity_payload()
    else:
        payload = runtime.identity_payload
    return InteractiveRuntimeRecord.model_validate(payload)


def _validate_interactive_runtime(
    expected: InteractiveRuntimeRecord,
    runtime: FormalInteractiveRuntime | None,
) -> InteractiveRuntimeRecord:
    actual = _interactive_runtime_record(runtime)
    if actual != expected:
        expected_payload = expected.model_dump(mode="json")
        actual_payload = actual.model_dump(mode="json")
        changed_fields = sorted(
            key
            for key in expected_payload.keys() | actual_payload.keys()
            if expected_payload.get(key) != actual_payload.get(key)
        )
        raise ArtifactConflictError(
            "interactive runtime does not match the immutable run manifest; "
            f"changed fields: {changed_fields}"
        )
    return actual


def _prepare_track_b_policy_runtime(
    *,
    policy_runtime: PolicyRuntime | None,
    interactive_runtime: FormalInteractiveRuntime | None,
    policy_names: Sequence[str],
    run_dir: Path,
    run_id: str,
) -> tuple[PolicyRuntime, bool]:
    configured = policy_runtime or PolicyRuntime()
    if (
        interactive_runtime is None
        or not policy_requires_provider(policy_names)
    ):
        return configured, False
    if policy_runtime is not None:
        _validate_companion_policy_identity(
            interactive_runtime,
            policy_runtime,
            policy_names=policy_names,
        )
    deferred = interactive_runtime.deferred_policy_runtime(
        run_dir=run_dir,
        run_id=run_id,
    )
    _validate_companion_policy_identity(
        interactive_runtime,
        deferred,
        policy_names=policy_names,
    )
    return deferred, True


def _validate_companion_policy_identity(
    runtime: FormalInteractiveRuntime | None,
    policy_runtime: PolicyRuntime,
    *,
    policy_names: Sequence[str],
) -> None:
    if runtime is None or not policy_requires_provider(policy_names):
        return
    provider = policy_runtime.provider
    if provider is None:
        raise ArtifactConflictError(
            "formal Track B model-backed policies require a companion provider"
        )
    pricing = runtime.selection.companion.pricing
    if pricing is None:
        raise ArtifactConflictError(
            "formal interactive companion is missing frozen pricing"
        )
    identity = provider.identity
    expected = {
        "provider_name": _FORMAL_PROVIDER_NAME,
        "model_id": runtime.selection.companion.model_id,
        "model_snapshot_digest": runtime.selection.model_snapshot_digest,
        "pricing_version": runtime.selection.pricing_version,
        "input_price_cny_per_million": pricing.input_per_million,
        "output_price_cny_per_million": pricing.output_per_million,
        "api_budget_cny": runtime.selection.api_budget_cny,
    }
    actual = {
        "provider_name": identity.provider_name,
        "model_id": identity.model_id,
        "model_snapshot_digest": identity.model_snapshot_digest,
        "pricing_version": identity.pricing_version,
        "input_price_cny_per_million": (
            identity.input_price_cny_per_million
        ),
        "output_price_cny_per_million": (
            identity.output_price_cny_per_million
        ),
        "api_budget_cny": identity.api_budget_cny,
    }
    changed_fields = sorted(
        key for key in expected if actual[key] != expected[key]
    )
    if changed_fields:
        raise ArtifactConflictError(
            "Track B companion policy provider does not match the formal "
            f"interactive runtime; changed fields: {changed_fields}"
        )


def _activate_api_runtime(
    runtime: FormalInteractiveRuntime | None,
    *,
    run_dir: Path,
    run_id: str,
) -> InteractiveAPIComponents | None:
    if runtime is None:
        return None
    return runtime.components_for(run_dir=run_dir, run_id=run_id)


def _track_b_policy_runtime(
    manifest: TrackBExperimentManifest,
    *,
    configured_runtime: PolicyRuntime,
    interactive_runtime: FormalInteractiveRuntime | None,
    api_components: InteractiveAPIComponents | None,
) -> PolicyRuntime:
    if (
        policy_requires_provider(manifest.policy_names)
        and interactive_runtime is not None
        and api_components is None
    ):
        raise ArtifactConflictError(
            "formal Track B runtime activated without API components"
        )
    if (
        describe_policy_runtimes(manifest.policy_names, configured_runtime)
        != manifest.policy_runtimes
    ):
        raise ArtifactConflictError(
            "configured companion policy runtime does not match the run manifest"
        )
    return configured_runtime


def _track_b_episode_runners(
    manifest: TrackBExperimentManifest,
    *,
    api_components: InteractiveAPIComponents | None,
) -> dict[str, TrackBEpisodeRunner]:
    if api_components is None:
        if manifest.interactive_runtime.mode != "local":
            raise ArtifactConflictError(
                "formal Track B manifest requires API components"
            )
        return {
            policy_name: TrackBEpisodeRunner()
            for policy_name in manifest.policy_names
        }
    if manifest.interactive_runtime.mode != "formal_api":
        raise ArtifactConflictError(
            "local Track B manifest cannot use API components"
        )

    runners: dict[str, TrackBEpisodeRunner] = {}
    for policy_name in manifest.policy_names:
        renderer = api_components.student_renderer_for(
            run_id=(
                f"{manifest.run_id}:track-b:policy={policy_name}"
            )
        )
        if renderer.name != manifest.renderer_name:
            raise ArtifactConflictError(
                "Track B API renderer does not match the run manifest"
            )
        runners[policy_name] = TrackBEpisodeRunner(
            disclosure_engine=TrackBDisclosureEngine(renderer=renderer)
        )
    return runners


def _track_c_responder(
    manifest: TrackCExperimentManifest,
    *,
    api_components: InteractiveAPIComponents | None,
) -> TrackCResponder:
    if api_components is None:
        if manifest.interactive_runtime.mode != "local":
            raise ArtifactConflictError(
                "formal Track C manifest requires API components"
            )
        responder: TrackCResponder = DeterministicTrackCBaselineResponder()
    else:
        if manifest.interactive_runtime.mode != "formal_api":
            raise ArtifactConflictError(
                "local Track C manifest cannot use API components"
            )
        responder = api_components.track_c_responder_for(
            run_id=f"{manifest.run_id}:track-c"
        )
    if responder.name != manifest.responder_name:
        raise ArtifactConflictError(
            "Track C responder does not match the run manifest"
        )
    return responder


def _select_track_b_specs(
    all_cases: tuple[CaseSpecV1, ...],
    *,
    selection_seed: int,
    case_limit: int | None,
) -> tuple[tuple[CaseSpecV1, ...], tuple[TrackBEpisodeSpec, ...]]:
    selected_cases = select_track_b_cases(
        all_cases,
        selection_seed=selection_seed,
    )
    if case_limit is not None:
        selected_cases = selected_cases[:case_limit]
    selected_ids = {case.case_id for case in selected_cases}
    specs = tuple(
        spec
        for spec in build_track_b_episode_specs(
            all_cases,
            seeds=TRACK_B_SEEDS,
            selection_seed=selection_seed,
        )
        if spec.case_id in selected_ids
    )
    return selected_cases, specs


def _reconstruct_track_b_selection(
    all_cases: tuple[CaseSpecV1, ...],
    *,
    manifest: TrackBExperimentManifest,
) -> tuple[tuple[CaseSpecV1, ...], tuple[TrackBEpisodeSpec, ...]]:
    expected_cases = select_track_b_cases(
        all_cases,
        selection_seed=manifest.selection_seed,
    )
    expected_by_id = {case.case_id: case for case in expected_cases}
    try:
        selected_cases = tuple(
            expected_by_id[case_id] for case_id in manifest.selected_case_ids
        )
    except KeyError as exc:
        raise ArtifactConflictError(
            f"manifest references a non-selected Track B case: {exc.args[0]}"
        ) from exc
    all_specs = build_track_b_episode_specs(
        all_cases,
        seeds=TRACK_B_SEEDS,
        selection_seed=manifest.selection_seed,
    )
    selected_ids = set(manifest.selected_case_ids)
    specs = tuple(spec for spec in all_specs if spec.case_id in selected_ids)
    return selected_cases, specs


def _build_track_b_manifest(
    *,
    run_id: str,
    all_cases: tuple[CaseSpecV1, ...],
    selected_cases: tuple[CaseSpecV1, ...],
    specs: tuple[TrackBEpisodeSpec, ...],
    dataset_path: str | None,
    dataset_seed: int,
    selection_seed: int,
    policy_names: tuple[str, ...],
    policy_runtime: PolicyRuntime,
    interactive_runtime: InteractiveRuntimeRecord,
    created_at: datetime | None = None,
) -> TrackBExperimentManifest:
    if not selected_cases or not specs:
        raise ValueError("Track B selection cannot be empty")
    benchmark_versions = {case.benchmark_version for case in all_cases}
    if len(benchmark_versions) != 1:
        raise ValueError(
            f"dataset mixes benchmark versions: {sorted(benchmark_versions)}"
        )
    spec_records = tuple(_track_b_spec_record(spec) for spec in specs)
    planned = tuple(
        _track_b_unit_id(spec.episode_id, policy_name)
        for spec in specs
        for policy_name in policy_names
    )
    policy_runtimes = describe_policy_runtimes(
        policy_names,
        policy_runtime,
    )
    renderer_name = interactive_runtime.renderer_name
    fingerprint_payload = {
        "track": Track.B.value,
        "benchmark_version": next(iter(benchmark_versions)),
        "dataset_seed": dataset_seed,
        "dataset_path": dataset_path,
        "dataset_digest": dataset_digest(all_cases),
        "selection_seed": selection_seed,
        "policy_names": policy_names,
        "policy_runtimes": tuple(
            record.model_dump(mode="json") for record in policy_runtimes
        ),
        "renderer_name": renderer_name,
        "interactive_runtime": interactive_runtime.model_dump(mode="json"),
        "selected_case_ids": tuple(case.case_id for case in selected_cases),
        "episode_specs": tuple(
            spec.model_dump(mode="json") for spec in spec_records
        ),
        "planned_unit_ids": planned,
    }
    payload: dict[str, object] = {
        **fingerprint_payload,
        "policy_runtimes": policy_runtimes,
        "interactive_runtime": interactive_runtime,
        "run_id": run_id,
        "run_fingerprint": _fingerprint(fingerprint_payload),
    }
    if created_at is not None:
        payload["created_at"] = created_at
    return TrackBExperimentManifest.model_validate(payload)


def _execute_track_b_pending(
    *,
    run_dir: Path,
    manifest: TrackBExperimentManifest,
    cases: tuple[CaseSpecV1, ...],
    specs: tuple[TrackBEpisodeSpec, ...],
    completed: dict[str, TrackBExperimentResult],
    max_units: int | None,
    policy_runtime: PolicyRuntime,
    api_components: InteractiveAPIComponents | None,
) -> ExperimentRunOutcome:
    cases_by_id = {case.case_id: case for case in cases}
    units = {
        _track_b_unit_id(spec.episode_id, policy_name): (spec, policy_name)
        for spec in specs
        for policy_name in manifest.policy_names
    }
    episode_runners = _track_b_episode_runners(
        manifest,
        api_components=api_components,
    )
    executed = 0
    for unit_id in manifest.planned_unit_ids:
        if unit_id in completed:
            continue
        if max_units is not None and executed >= max_units:
            break
        spec, policy_name = units[unit_id]
        scoped_runtime = policy_runtime.for_unit(
            BenchmarkInvocationScope(
                run_id=manifest.run_id,
                unit_id=unit_id,
                seed=spec.seed,
                replicate_id=0,
            )
        )
        episode = episode_runners[policy_name].run(
            cases_by_id[spec.case_id],
            spec,
            create_policy(policy_name, scoped_runtime),
        )
        result = _track_b_result(manifest, unit_id, episode)
        append_jsonl_model(run_dir / RESULTS_NAME, result)
        completed[unit_id] = result
        executed += 1
        write_json_atomic(
            run_dir / CHECKPOINT_NAME,
            build_checkpoint(manifest, tuple(completed)),
            overwrite=True,
        )
    return _finish_execution(
        run_dir=run_dir,
        manifest=manifest,
        completed_ids=tuple(completed),
        executed=executed,
    )


def _track_b_result(
    manifest: TrackBExperimentManifest,
    unit_id: str,
    episode: TrackBEpisodeResult,
) -> TrackBExperimentResult:
    return TrackBExperimentResult(
        unit_id=unit_id,
        run_id=manifest.run_id,
        run_fingerprint=manifest.run_fingerprint,
        case_id=episode.spec.case_id,
        policy_name=episode.policy_name,
        spec=_track_b_spec_record(episode.spec),
        renderer_name=episode.renderer_name,
        replay_digest=episode.replay_digest,
        metrics=TrackBMetricRecord.model_validate(episode.metrics.to_dict()),
        turns=tuple(
            TrackBTurnRecord.model_validate(turn.to_dict())
            for turn in episode.turns
        ),
    )


def _track_b_spec_record(spec: TrackBEpisodeSpec) -> TrackBEpisodeSpecRecord:
    return TrackBEpisodeSpecRecord.model_validate(spec.to_dict())


def _validate_track_b_results(
    results: dict[str, TrackBExperimentResult],
    manifest: TrackBExperimentManifest,
    specs: tuple[TrackBEpisodeSpec, ...],
) -> None:
    expected_units = {
        _track_b_unit_id(spec.episode_id, policy_name): (spec, policy_name)
        for spec in specs
        for policy_name in manifest.policy_names
    }
    _reject_unplanned_results(results, set(expected_units))
    mismatched: list[str] = []
    for unit_id, result in results.items():
        spec, policy_name = expected_units[unit_id]
        if (
            result.run_id != manifest.run_id
            or result.run_fingerprint != manifest.run_fingerprint
            or result.policy_name != policy_name
            or result.spec != _track_b_spec_record(spec)
            or result.renderer_name != manifest.renderer_name
            or result.replay_digest != _track_b_replay_digest(result)
        ):
            mismatched.append(unit_id)
    if mismatched:
        raise ArtifactConflictError(
            f"Track B results do not match the run manifest: {sorted(mismatched)}"
        )


def _track_b_replay_digest(result: TrackBExperimentResult) -> str:
    payload = {
        "track": Track.B.value,
        "spec": result.spec.model_dump(mode="json"),
        "policy_name": result.policy_name,
        "renderer_name": result.renderer_name,
        "turns": [
            turn.model_dump(mode="json") for turn in result.turns
        ],
        "metrics": result.metrics.model_dump(mode="json"),
    }
    return _fingerprint(payload)


def _track_b_unit_id(episode_id: str, policy_name: str) -> str:
    return f"{episode_id}__policy={policy_name}"


def _select_track_c_specs(
    all_scenarios: tuple[TrackCScenario, ...],
    *,
    configurations: tuple[TrackCConfiguration, ...],
    scenario_limit: int | None,
) -> tuple[tuple[TrackCScenario, ...], tuple[TrackCEpisodeSpec, ...]]:
    selected_scenarios = all_scenarios
    if scenario_limit is not None:
        selected_scenarios = selected_scenarios[:scenario_limit]
    scenario_ids = {scenario.scenario_id for scenario in selected_scenarios}
    specs = tuple(
        spec
        for spec in build_track_c_episode_matrix(
            all_scenarios,
            seeds=TRACK_C_SEEDS,
        )
        if spec.scenario_id in scenario_ids
        and spec.configuration in configurations
    )
    return selected_scenarios, specs


def _reconstruct_track_c_selection(
    all_scenarios: tuple[TrackCScenario, ...],
    *,
    manifest: TrackCExperimentManifest,
) -> tuple[tuple[TrackCScenario, ...], tuple[TrackCEpisodeSpec, ...]]:
    scenario_by_id = {
        scenario.scenario_id: scenario for scenario in all_scenarios
    }
    try:
        selected_scenarios = tuple(
            scenario_by_id[scenario_id]
            for scenario_id in manifest.scenario_ids
        )
    except KeyError as exc:
        raise ArtifactConflictError(
            f"manifest references a missing Track C scenario: {exc.args[0]}"
        ) from exc
    all_specs = build_track_c_episode_matrix(
        all_scenarios,
        seeds=manifest.seeds,
    )
    selected_ids = set(manifest.scenario_ids)
    specs = tuple(
        spec
        for spec in all_specs
        if spec.scenario_id in selected_ids
        and spec.configuration in manifest.configurations
    )
    return selected_scenarios, specs


def _build_track_c_manifest(
    *,
    run_id: str,
    all_scenarios: tuple[TrackCScenario, ...],
    selected_scenarios: tuple[TrackCScenario, ...],
    specs: tuple[TrackCEpisodeSpec, ...],
    configurations: tuple[TrackCConfiguration, ...],
    interactive_runtime: InteractiveRuntimeRecord,
    created_at: datetime | None = None,
) -> TrackCExperimentManifest:
    if not selected_scenarios or not specs:
        raise ValueError("Track C selection cannot be empty")
    scenario_ids = tuple(
        scenario.scenario_id for scenario in selected_scenarios
    )
    planned = tuple(spec.episode_id for spec in specs)
    fingerprint_payload = {
        "track": Track.C.value,
        "benchmark_version": _BENCHMARK_VERSION,
        "scenario_digest": _scenario_digest(all_scenarios),
        "scenario_ids": scenario_ids,
        "configurations": tuple(
            configuration.value for configuration in configurations
        ),
        "seeds": TRACK_C_SEEDS,
        "responder_name": interactive_runtime.responder_name,
        "interactive_runtime": interactive_runtime.model_dump(mode="json"),
        "planned_unit_ids": planned,
    }
    payload: dict[str, object] = {
        **fingerprint_payload,
        "interactive_runtime": interactive_runtime,
        "run_id": run_id,
        "run_fingerprint": _fingerprint(fingerprint_payload),
    }
    if created_at is not None:
        payload["created_at"] = created_at
    return TrackCExperimentManifest.model_validate(payload)


def _execute_track_c_pending(
    *,
    run_dir: Path,
    manifest: TrackCExperimentManifest,
    scenarios: tuple[TrackCScenario, ...],
    specs: tuple[TrackCEpisodeSpec, ...],
    completed: dict[str, TrackCExperimentResult],
    max_units: int | None,
    responder: TrackCResponder,
) -> ExperimentRunOutcome:
    scenarios_by_id = {
        scenario.scenario_id: scenario for scenario in scenarios
    }
    specs_by_id = {spec.episode_id: spec for spec in specs}
    runner = TrackCEpisodeRunner()
    executed = 0
    for unit_id in manifest.planned_unit_ids:
        if unit_id in completed:
            continue
        if max_units is not None and executed >= max_units:
            break
        spec = specs_by_id[unit_id]
        episode = runner.run(
            scenarios_by_id[spec.scenario_id],
            spec,
            responder,
        )
        result = _track_c_result(manifest, unit_id, episode)
        append_jsonl_model(run_dir / RESULTS_NAME, result)
        completed[unit_id] = result
        executed += 1
        write_json_atomic(
            run_dir / CHECKPOINT_NAME,
            build_checkpoint(manifest, tuple(completed)),
            overwrite=True,
        )
    return _finish_execution(
        run_dir=run_dir,
        manifest=manifest,
        completed_ids=tuple(completed),
        executed=executed,
    )


def _track_c_result(
    manifest: TrackCExperimentManifest,
    unit_id: str,
    episode: TrackCEpisodeResult,
) -> TrackCExperimentResult:
    return TrackCExperimentResult(
        unit_id=unit_id,
        run_id=manifest.run_id,
        run_fingerprint=manifest.run_fingerprint,
        responder_name=episode.responder_name,
        spec=TrackCEpisodeSpecRecord(
            episode_id=episode.spec.episode_id,
            scenario_id=episode.spec.scenario_id,
            configuration=episode.spec.configuration,
            seed=episode.spec.seed,
        ),
        replay_digest=episode.replay_digest,
        metrics=TrackCMetricRecord.model_validate(
            episode.metrics.to_dict()
        ),
        turns=tuple(
            TrackCTurnRecord.model_validate(turn.to_dict())
            for turn in episode.turns
        ),
    )


def _validate_track_c_results(
    results: dict[str, TrackCExperimentResult],
    manifest: TrackCExperimentManifest,
    specs: tuple[TrackCEpisodeSpec, ...],
) -> None:
    expected_specs = {spec.episode_id: spec for spec in specs}
    _reject_unplanned_results(results, set(expected_specs))
    mismatched: list[str] = []
    for unit_id, result in results.items():
        expected = expected_specs[unit_id]
        expected_record = TrackCEpisodeSpecRecord(
            episode_id=expected.episode_id,
            scenario_id=expected.scenario_id,
            configuration=expected.configuration,
            seed=expected.seed,
        )
        if (
            result.run_id != manifest.run_id
            or result.run_fingerprint != manifest.run_fingerprint
            or result.responder_name != manifest.responder_name
            or result.spec != expected_record
            or result.replay_digest != _track_c_replay_digest(result)
        ):
            mismatched.append(unit_id)
    if mismatched:
        raise ArtifactConflictError(
            f"Track C results do not match the run manifest: {sorted(mismatched)}"
        )


def _track_c_replay_digest(result: TrackCExperimentResult) -> str:
    features = track_c_features(result.spec.configuration)
    payload = {
        "track": Track.C.value,
        "spec": {
            **result.spec.model_dump(mode="json"),
            "features": {
                "memory_enabled": features.memory_enabled,
                "proactive_enabled": features.proactive_enabled,
                "eco_safe_enabled": features.eco_safe_enabled,
                "channel": features.channel,
            },
        },
        "responder_name": result.responder_name,
        "turns": [
            turn.model_dump(mode="json") for turn in result.turns
        ],
        "metrics": result.metrics.model_dump(mode="json"),
    }
    return _fingerprint(payload)


def _normalize_configurations(
    configurations: Sequence[TrackCConfiguration],
) -> tuple[TrackCConfiguration, ...]:
    normalized = tuple(dict.fromkeys(configurations))
    if not normalized:
        raise ValueError("at least one Track C configuration is required")
    return normalized


def _scenario_digest(scenarios: Sequence[TrackCScenario]) -> str:
    payload = [
        {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "pressure_turns": [
                {
                    "event_code": turn.event_code,
                    "variants": turn.variants,
                    "user_is_silent": turn.user_is_silent,
                }
                for turn in scenario.pressure_turns
            ],
            "safety_focus": tuple(
                violation.value for violation in scenario.safety_focus
            ),
            "memory_context": scenario.memory_context,
            "channel": scenario.channel,
        }
        for scenario in scenarios
    ]
    return _fingerprint(payload)


def _initialize_run(
    destination: Path,
    manifest: TrackBExperimentManifest | TrackCExperimentManifest,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / MANIFEST_NAME,
        manifest,
        overwrite=False,
    )
    write_json_atomic(
        destination / CHECKPOINT_NAME,
        build_checkpoint(manifest, ()),
        overwrite=False,
    )


def _new_run_destination(run_dir: Path) -> Path:
    destination = run_dir.resolve()
    if destination.exists():
        raise ArtifactConflictError(
            f"{destination} already exists; use `eco benchmark resume`"
        )
    return destination


def _load_and_validate_checkpoint(
    run_dir: Path,
    manifest: TrackBExperimentManifest | TrackCExperimentManifest,
    result_ids: set[str],
) -> ExperimentCheckpoint:
    checkpoint = ExperimentCheckpoint.model_validate(
        read_json_object(run_dir / CHECKPOINT_NAME)
    )
    validate_checkpoint(checkpoint, manifest, result_ids)
    return checkpoint


def _finish_execution(
    *,
    run_dir: Path,
    manifest: TrackBExperimentManifest | TrackCExperimentManifest,
    completed_ids: tuple[str, ...],
    executed: int,
) -> ExperimentRunOutcome:
    final_checkpoint = build_checkpoint(manifest, completed_ids)
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


def _completed_outcome(
    run_dir: Path,
    checkpoint: ExperimentCheckpoint,
) -> ExperimentRunOutcome:
    return ExperimentRunOutcome(
        run_dir=run_dir,
        status=checkpoint.status,
        planned_count=checkpoint.planned_count,
        completed_count=checkpoint.completed_count,
        executed_count=0,
    )


def _reject_unplanned_results(
    results: Mapping[str, object],
    planned: set[str],
) -> None:
    unexpected = set(results) - planned
    if unexpected:
        raise ArtifactConflictError(
            f"results contain unplanned unit ids: {sorted(unexpected)}"
        )


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_limit(limit: int | None, name: str) -> None:
    if limit is not None and limit <= 0:
        raise ValueError(f"{name} must be positive")


def _default_run_id(track: Track) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"track-{track.value.lower()}-{timestamp}"
