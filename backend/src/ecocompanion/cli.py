"""Unified ``eco`` command-line interface."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer
import uvicorn

from ecocompanion.benchmark.generation import DEFAULT_CASE_SEED
from ecocompanion.benchmark.log_importer import import_external_log
from ecocompanion.benchmark.model_provider import PolicyRuntime
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.benchmark.track_c import TrackCConfiguration
from ecocompanion.config import Settings, get_settings
from ecocompanion.domain.enums import Track
from ecocompanion.experiments import (
    ExperimentRunOutcome,
    available_policy_names,
    build_release_package,
    build_report,
    resume_benchmark_experiment,
    start_experiment,
    start_track_b_experiment,
    start_track_c_experiment,
    validate_benchmark,
)
from ecocompanion.experiments.interactive_api_runtime import (
    FormalInteractiveRuntime,
    build_openai_formal_interactive_runtime,
)
from ecocompanion.experiments.io import (
    ArtifactConflictError,
    read_json_object,
    write_json_atomic,
)
from ecocompanion.experiments.models import PolicyRuntimeRecord
from ecocompanion.experiments.provider_runtime import (
    build_new_policy_runtime,
    build_resume_policy_runtime,
)
from ecocompanion.providers import (
    ConservativeChineseSafetyEvaluator,
    FixedTokenPriceEstimator,
    ModelCapabilityProbe,
    ModelDescriptor,
    ModelQualificationResult,
    ModelQualificationRunner,
    ModelRegistry,
    ModelRegistrySnapshot,
    ModelRole,
    OpenAICompatibleClient,
    OpenAIQualificationExecutor,
    ProviderError,
    QualifiedModelSnapshot,
    UnknownCostEstimator,
    recommend_model_roles,
)
from ecocompanion.research.cli import app as research_app

app = typer.Typer(
    name="eco",
    no_args_is_help=True,
    help="EcoCompanion research, benchmark, and local demo commands.",
)
models_app = typer.Typer(no_args_is_help=True, help="Model discovery and qualification.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Offline benchmark workflows.")
report_app = typer.Typer(no_args_is_help=True, help="Experiment report generation.")
web_app = typer.Typer(no_args_is_help=True, help="Local FastAPI service commands.")

app.add_typer(models_app, name="models")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(report_app, name="report")
app.add_typer(web_app, name="web")
app.add_typer(research_app, name="research")


@models_app.command("discover")
def models_discover(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Immutable discovery snapshot path.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Explicitly replace an existing snapshot.",
        ),
    ] = False,
) -> None:
    """Discover gateway model identifiers without running paid probes."""

    settings = get_settings()
    destination = output or settings.artifacts_dir / "models" / "model_discovery.json"
    try:
        snapshot = asyncio.run(discover_model_snapshot(settings))
        write_model_snapshot(destination, snapshot, overwrite=overwrite)
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    typer.echo(
        f"discovered={len(snapshot.models)} "
        f"authenticated={str(snapshot.authenticated_discovery).lower()} "
        f"snapshot={destination.resolve()}"
    )


@models_app.command("qualify")
def models_qualify(
    input_snapshot: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="Optional discovery snapshot; otherwise discover live.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Immutable qualified model snapshot path.",
        ),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="Comma-separated model IDs to probe; default is all discovered models.",
        ),
    ] = None,
    max_concurrency: Annotated[
        int,
        typer.Option("--max-concurrency", min=1, help="Concurrent capability probes."),
    ] = 4,
    input_price_cny_per_million: Annotated[
        float | None,
        typer.Option(
            "--input-price-cny-per-million",
            min=0.0,
            help="Optional explicit input-token price used for qualification cost.",
        ),
    ] = None,
    output_price_cny_per_million: Annotated[
        float | None,
        typer.Option(
            "--output-price-cny-per-million",
            min=0.0,
            help="Optional explicit output-token price used for qualification cost.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Explicitly replace an existing snapshot.",
        ),
    ] = False,
) -> None:
    """Run capability probes and the fixed twenty-case safety qualification."""

    settings = get_settings()
    destination = output or settings.artifacts_dir / "models" / "model_snapshot.json"
    requested_models = _parse_csv(models) if models is not None else ()
    try:
        snapshot = asyncio.run(
            qualify_model_snapshot(
                settings,
                input_snapshot=input_snapshot,
                requested_model_ids=requested_models,
                max_concurrency=max_concurrency,
                input_price_cny_per_million=input_price_cny_per_million,
                output_price_cny_per_million=output_price_cny_per_million,
            )
        )
        write_model_snapshot(destination, snapshot, overwrite=overwrite)
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    ready = sum(report.status.value == "ready" for report in snapshot.capabilities)
    typer.echo(
        f"qualified={len(snapshot.capabilities)} ready={ready} "
        f"safety_suite={len(snapshot.qualifications)} "
        f"snapshot={destination.resolve()}"
    )


@benchmark_app.command("validate")
def benchmark_validate(
    cases: Annotated[
        Path | None,
        typer.Option(
            "--cases",
            help="Canonical JSON/JSONL dataset; omit to regenerate it.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Canonical generation seed."),
    ] = DEFAULT_CASE_SEED,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON validation report."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing validation report."),
    ] = False,
) -> None:
    """Validate canonical cases, surfaces, factorial coverage, and splits."""

    try:
        report = validate_benchmark(cases_path=cases, seed=seed)
        if output is not None:
            write_json_atomic(output, report, overwrite=overwrite)
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    typer.echo(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@benchmark_app.command("import-log")
def benchmark_import_log(
    source: Annotated[
        Path,
        typer.Argument(
            help="Synthetic external event log in the neutral JSON/JSONL contract.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Expert-relabel draft JSONL destination.",
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Explicitly replace a different existing draft artifact.",
        ),
    ] = False,
) -> None:
    """Import observable events into sanitized drafts requiring expert relabeling."""

    try:
        outcome = import_external_log(
            source,
            output,
            overwrite=overwrite,
        )
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    typer.echo(
        f"status={outcome.status} drafts={outcome.draft_count} "
        f"records={outcome.imported_record_count}/{outcome.input_record_count} "
        f"duplicates={outcome.duplicate_record_count} "
        f"redactions={outcome.redaction_count} output={outcome.output_path}"
    )


@benchmark_app.command("run")
def benchmark_run(
    track: Annotated[
        Track,
        typer.Option(
            "--track",
            help=("Benchmark track: A (frozen), B (interactive), or C (relationship safety)."),
        ),
    ] = Track.A,
    split: Annotated[
        DatasetSplit,
        typer.Option("--split", help="Canonical dataset split (Track A only)."),
    ] = DatasetSplit.PUBLIC_TEST,
    policies: Annotated[
        str | None,
        typer.Option(
            "--policy",
            help=(
                "Comma-separated policies for Tracks A/B. Available: "
                + ", ".join(available_policy_names())
            ),
        ),
    ] = None,
    surfaces: Annotated[
        str,
        typer.Option(
            "--surface",
            help="Track A surface index, comma-separated indices, or 'all'.",
        ),
    ] = "all",
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            min=1,
            help="Track A repeats; Tracks B/C use five fixed seeds.",
        ),
    ] = 1,
    configurations: Annotated[
        str,
        typer.Option(
            "--configuration",
            help="Track C configuration, comma-separated values, or 'all'.",
        ),
    ] = "all",
    cases: Annotated[
        Path | None,
        typer.Option(
            "--cases",
            help="Canonical JSON/JSONL dataset for Tracks A/B.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Canonical generation seed."),
    ] = DEFAULT_CASE_SEED,
    risk_threshold: Annotated[
        float,
        typer.Option(
            "--risk-threshold",
            min=0.000001,
            max=0.999999,
            help="Binary risk threshold for Track A metrics.",
        ),
    ] = 0.5,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Optional case limit (A/B) or scenario limit (C).",
        ),
    ] = None,
    max_units: Annotated[
        int | None,
        typer.Option(
            "--max-units",
            min=1,
            help="Stop after N units; resume later from durable results.",
        ),
    ] = None,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            help="New run directory; default is a timestamped artifact directory.",
        ),
    ] = None,
    agent_model: Annotated[
        str | None,
        typer.Option(
            "--agent-model",
            help=(
                "Qualified model ID for model-backed policies; defaults to the "
                "snapshot recommendation or top score."
            ),
        ),
    ] = None,
    model_snapshot: Annotated[
        Path | None,
        typer.Option(
            "--model-snapshot",
            help=(
                "Frozen qualified-model snapshot used by model-backed policies "
                "or formal interactive Tracks B/C."
            ),
        ),
    ] = None,
    formal_api: Annotated[
        bool,
        typer.Option(
            "--formal-api",
            help=(
                "Run Track B/C with the frozen API companion, student renderer, "
                "and dual judges instead of deterministic local components."
            ),
        ),
    ] = False,
    pricing_catalog: Annotated[
        Path | None,
        typer.Option(
            "--pricing-catalog",
            help=(
                "Versioned per-model CNY pricing catalog for formal Track B/C; "
                "defaults to artifacts/models/pricing_catalog.json."
            ),
        ),
    ] = None,
    student_model: Annotated[
        str | None,
        typer.Option(
            "--student-model",
            help="Qualified student-language renderer model for formal Track B/C.",
        ),
    ] = None,
    judge_models: Annotated[
        str | None,
        typer.Option(
            "--judge-models",
            help="Exactly two comma-separated qualified judge model IDs.",
        ),
    ] = None,
) -> None:
    """Run Track A, B, or C locally and persist resumable artifacts."""

    settings = get_settings()
    family_warning: str | None = None
    try:
        destination = run_dir or _default_run_dir(settings, track, split)
        _validate_interactive_cli_options(
            track=track,
            formal_api=formal_api,
            pricing_catalog=pricing_catalog,
            student_model=student_model,
            judge_models=judge_models,
        )
        if track is Track.A:
            _reject_track_c_configuration(configurations)
            selected_policies = _parse_csv(policies or "history-aware-rule")
            runtime = build_new_policy_runtime(
                settings=settings,
                policy_names=selected_policies,
                model_id=agent_model,
                snapshot_path=model_snapshot,
                run_dir=destination,
                run_id=destination.name,
            )
            try:
                outcome = start_experiment(
                    run_dir=destination,
                    split=split,
                    policy_names=selected_policies,
                    surface_indices=_parse_surfaces(surfaces),
                    repeats=repeats,
                    seed=seed,
                    cases_path=cases,
                    risk_threshold=risk_threshold,
                    case_limit=limit,
                    run_id=destination.name,
                    max_units=max_units,
                    policy_runtime=runtime,
                )
            finally:
                runtime.close()
        elif track is Track.B:
            _reject_track_b_unused_options(
                surfaces=surfaces,
                repeats=repeats,
                configurations=configurations,
            )
            selected_policies = _parse_csv(policies or "history-aware-rule")
            interactive_runtime = _build_formal_interactive_runtime(
                settings=settings,
                enabled=formal_api,
                model_snapshot=model_snapshot,
                pricing_catalog=pricing_catalog,
                companion_model=agent_model,
                student_model=student_model,
                judge_models=judge_models,
            )
            family_warning = _interactive_family_warning(interactive_runtime)
            track_b_runtime: PolicyRuntime | None = (
                None
                if interactive_runtime is not None
                else build_new_policy_runtime(
                    settings=settings,
                    policy_names=selected_policies,
                    model_id=agent_model,
                    snapshot_path=model_snapshot,
                    run_dir=destination,
                    run_id=destination.name,
                )
            )
            try:
                outcome = start_track_b_experiment(
                    run_dir=destination,
                    policy_names=selected_policies,
                    seed=seed,
                    cases_path=cases,
                    case_limit=limit,
                    run_id=destination.name,
                    max_units=max_units,
                    policy_runtime=track_b_runtime,
                    interactive_runtime=interactive_runtime,
                )
            finally:
                if track_b_runtime is not None:
                    track_b_runtime.close()
                if interactive_runtime is not None:
                    interactive_runtime.close()
        else:
            _reject_track_c_unused_options(
                policies=policies,
                surfaces=surfaces,
                repeats=repeats,
                cases=cases,
                agent_model=agent_model,
                model_snapshot=model_snapshot,
                formal_api=formal_api,
            )
            interactive_runtime = _build_formal_interactive_runtime(
                settings=settings,
                enabled=formal_api,
                model_snapshot=model_snapshot,
                pricing_catalog=pricing_catalog,
                companion_model=agent_model,
                student_model=student_model,
                judge_models=judge_models,
            )
            family_warning = _interactive_family_warning(interactive_runtime)
            try:
                outcome = start_track_c_experiment(
                    run_dir=destination,
                    configurations=_parse_track_c_configurations(configurations),
                    scenario_limit=limit,
                    run_id=destination.name,
                    max_units=max_units,
                    interactive_runtime=interactive_runtime,
                )
            finally:
                if interactive_runtime is not None:
                    interactive_runtime.close()
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    if family_warning is not None:
        typer.echo(f"warning: {family_warning}")
    typer.echo(_format_outcome(outcome))


@benchmark_app.command("resume")
def benchmark_resume(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Existing run directory containing manifest/checkpoint."),
    ],
    max_units: Annotated[
        int | None,
        typer.Option(
            "--max-units",
            min=1,
            help="Execute at most N currently pending units.",
        ),
    ] = None,
    agent_model: Annotated[
        str | None,
        typer.Option(
            "--agent-model",
            help="Must match the model frozen in a model-backed run manifest.",
        ),
    ] = None,
    model_snapshot: Annotated[
        Path | None,
        typer.Option(
            "--model-snapshot",
            help="Frozen qualified-model snapshot used by the original run.",
        ),
    ] = None,
    formal_api: Annotated[
        bool,
        typer.Option(
            "--formal-api",
            help="Assert that the existing Track B/C run uses formal API mode.",
        ),
    ] = False,
    pricing_catalog: Annotated[
        Path | None,
        typer.Option(
            "--pricing-catalog",
            help=(
                "Pricing catalog used by the original formal run; defaults to "
                "artifacts/models/pricing_catalog.json."
            ),
        ),
    ] = None,
    student_model: Annotated[
        str | None,
        typer.Option(
            "--student-model",
            help="Optional assertion for the frozen student renderer model.",
        ),
    ] = None,
    judge_models: Annotated[
        str | None,
        typer.Option(
            "--judge-models",
            help="Optional assertion for the two frozen judge model IDs.",
        ),
    ] = None,
) -> None:
    """Resume any track using only units missing from durable JSONL results."""

    settings = get_settings()
    family_warning: str | None = None
    try:
        manifest = read_json_object(run_dir.resolve() / "manifest.json")
        track = _manifest_track(manifest)
        manifest_uses_formal_api = _manifest_uses_formal_api(manifest)
        _validate_resume_interactive_options(
            track=track,
            manifest_uses_formal_api=manifest_uses_formal_api,
            formal_api=formal_api,
            pricing_catalog=pricing_catalog,
            agent_model=agent_model,
            model_snapshot=model_snapshot,
            student_model=student_model,
            judge_models=judge_models,
        )
        frozen_roles = (
            _formal_role_ids_from_manifest(manifest)
            if manifest_uses_formal_api
            else None
        )
        if frozen_roles is not None:
            companion_id, student_id, judge_ids = frozen_roles
            _assert_optional_model(agent_model, companion_id, "--agent-model")
            _assert_optional_model(student_model, student_id, "--student-model")
            if judge_models is not None:
                requested_judges = _parse_csv(judge_models)
                if requested_judges != judge_ids:
                    raise ValueError(
                        "--judge-models does not match the immutable run manifest"
                    )
            interactive_runtime = _build_formal_interactive_runtime(
                settings=settings,
                enabled=True,
                model_snapshot=model_snapshot,
                pricing_catalog=pricing_catalog,
                companion_model=companion_id,
                student_model=student_id,
                judge_models=",".join(judge_ids),
            )
            family_warning = _interactive_family_warning(interactive_runtime)
        else:
            interactive_runtime = None

        policy_runtimes = _manifest_policy_runtimes(run_dir)
        runtime = (
            None
            if interactive_runtime is not None or track is Track.C
            else build_resume_policy_runtime(
                settings=settings,
                policy_runtimes=policy_runtimes,
                model_id=agent_model,
                snapshot_path=model_snapshot,
                run_dir=run_dir,
                run_id=_manifest_run_id(manifest),
            )
        )
        try:
            outcome = resume_benchmark_experiment(
                run_dir,
                max_units=max_units,
                policy_runtime=runtime,
                interactive_runtime=interactive_runtime,
            )
        finally:
            if runtime is not None:
                runtime.close()
            if interactive_runtime is not None:
                interactive_runtime.close()
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    if family_warning is not None:
        typer.echo(f"warning: {family_warning}")
    typer.echo(_format_outcome(outcome))


@benchmark_app.command("package")
def benchmark_package(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="New release-package directory.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Canonical generation seed."),
    ] = DEFAULT_CASE_SEED,
    run_dirs: Annotated[
        list[Path] | None,
        typer.Option(
            "--run-dir",
            help=(
                "Optional audited experiment run directory to include. "
                "Repeat the option to include multiple redacted runs."
            ),
        ),
    ] = None,
    model_snapshot: Annotated[
        Path | None,
        typer.Option(
            "--model-snapshot",
            help=(
                "Optional authenticated, qualified model snapshot to include. "
                "No placeholder is replaced unless this file passes sanitization."
            ),
        ),
    ] = None,
    pricing_catalog: Annotated[
        Path | None,
        typer.Option(
            "--pricing-catalog",
            help=(
                "Optional versioned per-model pricing catalog to include "
                "alongside the model snapshot."
            ),
        ),
    ] = None,
) -> None:
    """Build a sanitized research-preview package with checksums."""

    settings = get_settings()
    destination = output or (settings.artifacts_dir / "release" / "ecotriage-v1-preview")
    project_root = Path(__file__).resolve().parents[3]
    try:
        built = build_release_package(
            destination,
            project_root=project_root,
            seed=seed,
            run_dirs=tuple(run_dirs or ()),
            model_snapshot_path=model_snapshot,
            pricing_catalog_path=pricing_catalog,
        )
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    typer.echo(f"release={built}")


@report_app.command("build")
def report_build(
    source: Annotated[
        Path,
        typer.Argument(help="Run directory or ExperimentResult JSONL file."),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Report destination directory."),
    ] = None,
    annotations: Annotated[
        Path | None,
        typer.Option(
            "--annotations",
            help=("Track A adjudicator AnnotationBatchV1 used as expert evidence gold."),
        ),
    ] = None,
    failure_case_limit: Annotated[
        int,
        typer.Option(
            "--failure-case-limit",
            min=0,
            help="Keep at most N ranked failure cases in JSON and Markdown.",
        ),
    ] = 30,
    bootstrap_resamples: Annotated[
        int,
        typer.Option(
            "--bootstrap-resamples",
            min=1,
            help="Canonical-cluster bootstrap replicates for 95% intervals.",
        ),
    ] = 2_000,
    permutation_resamples: Annotated[
        int,
        typer.Option(
            "--permutation-resamples",
            min=1,
            help="Monte Carlo sign-flip replicates for larger paired samples.",
        ),
    ] = 10_000,
    statistics_seed: Annotated[
        int,
        typer.Option(
            "--statistics-seed",
            help="Deterministic seed for bootstrap and paired tests.",
        ),
    ] = 20_260_904,
) -> None:
    """Create JSON and Markdown summaries for Tracks A, B, or C."""

    try:
        json_path, markdown_path, report = build_report(
            source,
            output_dir=output_dir,
            annotation_path=annotations,
            failure_case_limit=failure_case_limit,
            bootstrap_resamples=bootstrap_resamples,
            permutation_resamples=permutation_resamples,
            random_seed=statistics_seed,
        )
    except _EXPECTED_CLI_ERRORS as exc:
        _abort(str(exc))
    typer.echo(
        f"complete={str(report.complete).lower()} "
        f"units={report.completed_unit_count} "
        f"json={json_path} markdown={markdown_path}"
    )


@web_app.command("serve")
def web_serve(
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="Bind port."),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable Uvicorn development reload."),
    ] = False,
) -> None:
    """Serve the FastAPI application through Uvicorn."""

    uvicorn.run(
        "ecocompanion.api:app",
        host=host,
        port=port,
        reload=reload,
    )


async def discover_model_snapshot(settings: Settings) -> ModelRegistrySnapshot:
    """Discover candidates using the configured OpenAI-compatible gateway."""

    api_key = (
        settings.litellm_api_key.get_secret_value()
        if settings.litellm_api_key is not None
        else None
    )
    async with _provider_client(settings, api_key=api_key) as client:
        return await ModelRegistry(client).refresh(run_smoke_tests=False)


async def qualify_model_snapshot(
    settings: Settings,
    *,
    input_snapshot: Path | None,
    requested_model_ids: Sequence[str],
    max_concurrency: int,
    input_price_cny_per_million: float | None = None,
    output_price_cny_per_million: float | None = None,
) -> QualifiedModelSnapshot:
    """Probe capabilities, then run the fixed safety suite for ready models."""

    api_key = settings.require_api_key()
    if (input_price_cny_per_million is None) != (output_price_cny_per_million is None):
        raise ValueError("input and output token prices must be provided together")
    if input_snapshot is not None:
        models, authenticated = _load_model_descriptors(input_snapshot)
    else:
        models = ()
        authenticated = False

    async with _provider_client(settings, api_key=api_key) as client:
        if input_snapshot is None:
            models, authenticated = await client.discover_models()
        selected = _select_models(models, requested_model_ids)
        capabilities = await ModelCapabilityProbe(
            client,
            max_concurrency=max_concurrency,
        ).probe_all(selected)
        cost_estimator = (
            FixedTokenPriceEstimator(
                input_cny_per_million=input_price_cny_per_million,
                output_cny_per_million=output_price_cny_per_million,
            )
            if input_price_cny_per_million is not None and output_price_cny_per_million is not None
            else UnknownCostEstimator()
        )
        executor = OpenAIQualificationExecutor(
            client=client,
            evaluator=ConservativeChineseSafetyEvaluator(),
            cost_estimator=cost_estimator,
        )
        semaphore = asyncio.Semaphore(max_concurrency)

        async def qualify_ready_model(
            model_id: str,
        ) -> ModelQualificationResult:
            async with semaphore:
                return await ModelQualificationRunner(executor).qualify(model_id)

        qualifications = await asyncio.gather(
            *(
                qualify_ready_model(report.model_id)
                for report in capabilities
                if report.status.value == "ready"
            )
        )
    registry = ModelRegistrySnapshot(
        models=selected,
        capabilities=capabilities,
        authenticated_discovery=authenticated,
    )
    return QualifiedModelSnapshot(
        registry=registry,
        qualifications=tuple(qualifications),
    )


def write_model_snapshot(
    path: Path,
    snapshot: ModelRegistrySnapshot | QualifiedModelSnapshot,
    *,
    overwrite: bool,
) -> Path:
    """Persist a key-free model snapshot with explicit replacement semantics."""

    payload = snapshot.to_dict()
    if isinstance(snapshot, QualifiedModelSnapshot):
        payload["recommended_roles"] = recommend_model_roles(snapshot).to_dict()
    return write_json_atomic(path, payload, overwrite=overwrite)


def _provider_client(
    settings: Settings,
    *,
    api_key: str | None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=settings.litellm_base_url,
        api_key=api_key,
        connect_timeout_seconds=settings.connect_timeout_seconds,
        read_timeout_seconds=settings.read_timeout_seconds,
        max_same_model_retries=settings.max_same_model_retries,
        default_extra_body=settings.provider_extra_body,
    )


def _load_model_descriptors(
    path: Path,
) -> tuple[tuple[ModelDescriptor, ...], bool]:
    payload = read_json_object(path)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"{path}: snapshot has no models")
    models: list[ModelDescriptor] = []
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ValueError(f"{path}: model {index} is not an object")
        model_id = raw_model.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"{path}: model {index} has no valid model_id")
        owner = raw_model.get("declared_owner")
        created = raw_model.get("created")
        if owner is not None and not isinstance(owner, str):
            raise ValueError(f"{path}: model {model_id} has invalid declared_owner")
        if created is not None and (not isinstance(created, int) or isinstance(created, bool)):
            raise ValueError(f"{path}: model {model_id} has invalid created value")
        models.append(
            ModelDescriptor(
                model_id=model_id,
                declared_owner=owner,
                created=created,
            )
        )
    authenticated = payload.get("authenticated_discovery", False)
    if not isinstance(authenticated, bool):
        raise ValueError(f"{path}: authenticated_discovery must be boolean")
    return tuple(models), authenticated


def _select_models(
    models: Sequence[ModelDescriptor],
    requested_model_ids: Sequence[str],
) -> tuple[ModelDescriptor, ...]:
    by_id = {model.model_id: model for model in models}
    if len(by_id) != len(models):
        raise ValueError("discovery snapshot contains duplicate model IDs")
    if not requested_model_ids:
        return tuple(sorted(models, key=lambda model: model.model_id))
    requested = tuple(dict.fromkeys(requested_model_ids))
    missing = set(requested) - set(by_id)
    if missing:
        raise ValueError(f"requested models were not discovered: {sorted(missing)}")
    return tuple(by_id[model_id] for model_id in requested)


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(dict.fromkeys(item.strip() for item in value.split(",")))
    if not items or any(not item for item in items):
        raise ValueError("comma-separated values cannot contain empty items")
    return items


def _parse_surfaces(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return (0, 1)
    try:
        surfaces = tuple(int(item) for item in _parse_csv(value))
    except ValueError as exc:
        raise ValueError("surface must be 'all' or comma-separated integers") from exc
    if any(surface < 0 for surface in surfaces):
        raise ValueError("surface indices cannot be negative")
    return surfaces


def _parse_track_c_configurations(
    value: str,
) -> tuple[TrackCConfiguration, ...]:
    if value.strip().lower() == "all":
        return tuple(TrackCConfiguration)
    try:
        return tuple(TrackCConfiguration(item.upper()) for item in _parse_csv(value))
    except ValueError as exc:
        available = ", ".join(configuration.value for configuration in TrackCConfiguration)
        raise ValueError(f"unknown Track C configuration; available: {available}") from exc


def _reject_track_c_configuration(configurations: str) -> None:
    if configurations.strip().lower() != "all":
        raise ValueError("--configuration is only valid for Track C")


def _reject_track_b_unused_options(
    *,
    surfaces: str,
    repeats: int,
    configurations: str,
) -> None:
    if surfaces.strip().lower() != "all":
        raise ValueError("Track B chooses paired surfaces from its fixed episode specs")
    if repeats != 1:
        raise ValueError("Track B uses five fixed seeds; --repeats must be 1")
    _reject_track_c_configuration(configurations)


def _reject_track_c_unused_options(
    *,
    policies: str | None,
    surfaces: str,
    repeats: int,
    cases: Path | None,
    agent_model: str | None,
    model_snapshot: Path | None,
    formal_api: bool,
) -> None:
    if policies is not None:
        raise ValueError("Track C uses its configuration matrix; omit --policy")
    if surfaces.strip().lower() != "all":
        raise ValueError("Track C is text-only and does not use --surface")
    if repeats != 1:
        raise ValueError("Track C uses five fixed seeds; --repeats must be 1")
    if cases is not None:
        raise ValueError("Track C uses fixed pressure scenarios, not --cases")
    if not formal_api and (agent_model is not None or model_snapshot is not None):
        raise ValueError(
            "Track C model-provider options require --formal-api"
        )


def _validate_interactive_cli_options(
    *,
    track: Track,
    formal_api: bool,
    pricing_catalog: Path | None,
    student_model: str | None,
    judge_models: str | None,
) -> None:
    supplied_interactive_options = (
        pricing_catalog is not None
        or student_model is not None
        or judge_models is not None
    )
    if track is Track.A and (formal_api or supplied_interactive_options):
        raise ValueError(
            "--formal-api, --pricing-catalog, --student-model, and "
            "--judge-models are only valid for Tracks B/C"
        )
    if track is not Track.A and not formal_api and supplied_interactive_options:
        raise ValueError(
            "interactive model-role options require --formal-api"
        )
    if judge_models is not None and len(_parse_csv(judge_models)) != 2:
        raise ValueError("--judge-models requires exactly two model IDs")


def _build_formal_interactive_runtime(
    *,
    settings: Settings,
    enabled: bool,
    model_snapshot: Path | None,
    pricing_catalog: Path | None,
    companion_model: str | None,
    student_model: str | None,
    judge_models: str | None,
) -> FormalInteractiveRuntime | None:
    if not enabled:
        return None
    snapshot_path = (
        model_snapshot
        if model_snapshot is not None
        else settings.artifacts_dir / "models" / "model_snapshot.json"
    )
    catalog_path = (
        pricing_catalog
        if pricing_catalog is not None
        else settings.artifacts_dir / "models" / "pricing_catalog.json"
    )
    parsed_judges = (
        _parse_csv(judge_models) if judge_models is not None else None
    )
    if parsed_judges is not None and len(parsed_judges) != 2:
        raise ValueError("--judge-models requires exactly two model IDs")
    return build_openai_formal_interactive_runtime(
        settings=settings,
        model_snapshot_path=snapshot_path,
        pricing_catalog_path=catalog_path,
        companion_model_id=companion_model,
        student_model_id=student_model,
        judge_model_ids=parsed_judges,
    )


def _manifest_track(manifest: dict[str, object]) -> Track:
    raw_track = manifest.get("track", Track.A.value)
    try:
        return Track(str(raw_track))
    except ValueError as exc:
        raise ValueError(
            f"manifest declares unsupported track: {raw_track!r}"
        ) from exc


def _manifest_run_id(manifest: dict[str, object]) -> str:
    raw_run_id = manifest.get("run_id")
    if not isinstance(raw_run_id, str) or not raw_run_id.strip():
        raise ValueError("manifest requires a non-blank run_id")
    return raw_run_id


def _manifest_uses_formal_api(manifest: dict[str, object]) -> bool:
    track = _manifest_track(manifest)
    runtime = manifest.get("interactive_runtime")
    if track is Track.A:
        if runtime is not None:
            raise ValueError("Track A manifest cannot contain interactive_runtime")
        return False
    if not isinstance(runtime, dict):
        raise ValueError("Track B/C manifest requires interactive_runtime")
    mode = runtime.get("mode")
    if not isinstance(mode, str) or mode not in {"local", "formal_api"}:
        raise ValueError(
            f"manifest declares unsupported interactive runtime mode: {mode!r}"
        )
    return mode == "formal_api"


def _formal_role_ids_from_manifest(
    manifest: dict[str, object],
) -> tuple[str, str, tuple[str, str]]:
    runtime = manifest.get("interactive_runtime")
    if not isinstance(runtime, dict) or runtime.get("mode") != "formal_api":
        raise ValueError("manifest does not describe a formal API runtime")
    raw_bindings = runtime.get("role_bindings")
    if not isinstance(raw_bindings, list | tuple) or len(raw_bindings) != 4:
        raise ValueError(
            "formal API manifest requires companion, student, and two judge bindings"
        )
    bindings: list[tuple[str, str]] = []
    for index, raw_binding in enumerate(raw_bindings):
        if not isinstance(raw_binding, dict):
            raise ValueError(
                f"formal API role binding {index} must be an object"
            )
        role = raw_binding.get("role")
        model_id = raw_binding.get("model_id")
        if not isinstance(role, str) or not isinstance(model_id, str):
            raise ValueError(
                f"formal API role binding {index} is missing role/model_id"
            )
        if not model_id.strip():
            raise ValueError(
                f"formal API role binding {index} has a blank model_id"
            )
        bindings.append((role, model_id))
    expected_roles = (
        ModelRole.EVALUATED_AGENT.value,
        ModelRole.STUDENT_SIMULATOR.value,
        ModelRole.JUDGE.value,
        ModelRole.JUDGE.value,
    )
    if tuple(role for role, _ in bindings) != expected_roles:
        raise ValueError(
            "formal API role bindings are not in the required immutable order"
        )
    return (
        bindings[0][1],
        bindings[1][1],
        (bindings[2][1], bindings[3][1]),
    )


def _assert_optional_model(
    requested: str | None,
    expected: str,
    option_name: str,
) -> None:
    if requested is not None and requested != expected:
        raise ValueError(
            f"{option_name} does not match the immutable run manifest"
        )


def _validate_resume_interactive_options(
    *,
    track: Track,
    manifest_uses_formal_api: bool,
    formal_api: bool,
    pricing_catalog: Path | None,
    agent_model: str | None,
    model_snapshot: Path | None,
    student_model: str | None,
    judge_models: str | None,
) -> None:
    supplied_interactive_options = (
        pricing_catalog is not None
        or student_model is not None
        or judge_models is not None
    )
    if track is Track.A and (formal_api or supplied_interactive_options):
        raise ValueError(
            "Track A resume does not accept interactive API options"
        )
    if track is not Track.A and formal_api and not manifest_uses_formal_api:
        raise ValueError(
            "--formal-api does not match the local interactive run manifest"
        )
    if (
        track is not Track.A
        and not manifest_uses_formal_api
        and supplied_interactive_options
    ):
        raise ValueError(
            "interactive model-role options are unused by this local run"
        )
    if (
        track is Track.C
        and not manifest_uses_formal_api
        and (agent_model is not None or model_snapshot is not None)
    ):
        raise ValueError(
            "Track C model-provider options are unused by this local run"
        )
    if judge_models is not None and len(_parse_csv(judge_models)) != 2:
        raise ValueError("--judge-models requires exactly two model IDs")


def _manifest_policy_runtimes(
    run_dir: Path,
) -> tuple[PolicyRuntimeRecord, ...]:
    payload = read_json_object(run_dir.resolve() / "manifest.json")
    raw_records = payload.get("policy_runtimes", ())
    if not isinstance(raw_records, list | tuple):
        raise ValueError("manifest policy_runtimes must be a list")
    return tuple(PolicyRuntimeRecord.model_validate(record) for record in raw_records)


def _default_run_dir(
    settings: Settings,
    track: Track,
    split: DatasetSplit,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = (
        f"track-a-{split.value}-{timestamp}"
        if track is Track.A
        else f"track-{track.value.lower()}-{timestamp}"
    )
    return settings.artifacts_dir / "runs" / suffix


def _format_outcome(outcome: ExperimentRunOutcome) -> str:
    return (
        f"status={outcome.status.value} "
        f"completed={outcome.completed_count}/{outcome.planned_count} "
        f"executed={outcome.executed_count} run_dir={outcome.run_dir}"
    )


def _interactive_family_warning(
    runtime: FormalInteractiveRuntime | None,
) -> str | None:
    """Return the frozen family caveat for a formal API CLI run."""

    if runtime is None:
        return None
    selection = runtime.selection
    if selection.family_separation_status != "degraded":
        return None
    warning_text = "; ".join(selection.family_warnings)
    return (
        "formal API model-family separation is degraded; "
        "expert evaluation is required"
        + (f" ({warning_text})" if warning_text else "")
    )


def _abort(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


_EXPECTED_CLI_ERRORS = (
    ArtifactConflictError,
    FileNotFoundError,
    ProviderError,
    RuntimeError,
    ValueError,
)
