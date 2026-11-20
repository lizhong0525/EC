import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from typer.testing import CliRunner

from ecocompanion import cli
from ecocompanion.config import Settings
from ecocompanion.providers import (
    CapabilityStatus,
    ChatCompletion,
    CompletionUsage,
    ModelCapabilityReport,
    ModelDescriptor,
    ModelQualificationRunner,
    ModelRegistrySnapshot,
    ProbeExecution,
    QualifiedModelSnapshot,
    SafetyQualificationProbe,
    StructuredMode,
)

runner = CliRunner()


class StaticQualificationExecutor:
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
            persona_consistent=True,
            boundary_refusal_stable=True,
            latency_ms=10.0,
            cost_cny=0.001,
        )


class FakeQualifiedClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    async def discover_models(self):
        return (ModelDescriptor(model_id="model-a"),), True

    async def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        temperature: float | None = 0.0,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatCompletion:
        del messages, max_tokens, temperature, response_format
        return ChatCompletion(
            requested_model=model,
            response_model=model,
            content='{"response_text":"谢谢你告诉我，我在听。"}',
            finish_reason="stop",
            usage=CompletionUsage(input_tokens=10, output_tokens=5),
            retry_count=0,
        )


class FakeCapabilityProbe:
    def __init__(self, client, *, max_concurrency: int) -> None:
        del client, max_concurrency

    async def probe_all(self, models):
        return tuple(
            ModelCapabilityReport(
                model_id=model.model_id,
                status=CapabilityStatus.READY,
                basic_chat=True,
                system_and_multiturn=True,
                chinese_text=True,
                usage_reported=True,
                preferred_structured_mode=StructuredMode.JSON_SCHEMA,
            )
            for model in models
        )


def test_root_cli_exposes_all_planned_command_groups() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    for command in ("models", "benchmark", "report", "web"):
        assert command in result.stdout

    expected = {
        "models": ("discover", "qualify"),
        "benchmark": ("validate", "run", "resume", "package"),
        "report": ("build",),
        "web": ("serve",),
    }
    for group, commands in expected.items():
        help_result = runner.invoke(cli.app, [group, "--help"])
        assert help_result.exit_code == 0
        for command in commands:
            assert command in help_result.stdout


def test_cli_runs_resumes_and_reports_one_offline_unit(tmp_path) -> None:
    run_dir = tmp_path / "cli-run"
    run_result = runner.invoke(
        cli.app,
        [
            "benchmark",
            "run",
            "--split",
            "dev",
            "--policy",
            "keyword-triage",
            "--surface",
            "0",
            "--repeats",
            "1",
            "--limit",
            "1",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    assert "status=complete" in run_result.stdout

    resume_result = runner.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir)],
    )
    assert resume_result.exit_code == 0, resume_result.output
    assert "executed=0" in resume_result.stdout

    report_result = runner.invoke(
        cli.app,
        [
            "report",
            "build",
            str(run_dir),
            "--failure-case-limit",
            "2",
            "--bootstrap-resamples",
            "40",
            "--permutation-resamples",
            "100",
            "--statistics-seed",
            "19",
        ],
    )
    assert report_result.exit_code == 0, report_result.output
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["failure_case_limit"] == 2
    assert report["statistical_protocol"]["bootstrap_resamples"] == 40
    assert report["statistical_protocol"]["random_seed"] == 19


def test_cli_runs_resumes_and_reports_track_b(tmp_path) -> None:
    run_dir = tmp_path / "track-b-cli"
    started = runner.invoke(
        cli.app,
        [
            "benchmark",
            "run",
            "--track",
            "B",
            "--policy",
            "history-aware-rule",
            "--limit",
            "1",
            "--max-units",
            "1",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert started.exit_code == 0, started.output
    assert "status=in_progress" in started.stdout
    assert "completed=1/5" in started.stdout

    resumed = runner.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir)],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "status=complete" in resumed.stdout

    reported = runner.invoke(
        cli.app,
        ["report", "build", str(run_dir)],
    )
    assert reported.exit_code == 0, reported.output
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["track"] == "B"


def test_cli_runs_resumes_and_reports_track_c(tmp_path) -> None:
    run_dir = tmp_path / "track-c-cli"
    started = runner.invoke(
        cli.app,
        [
            "benchmark",
            "run",
            "--track",
            "C",
            "--configuration",
            "MEMORY_PROACTIVE_ECO_SAFE",
            "--limit",
            "1",
            "--max-units",
            "1",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert started.exit_code == 0, started.output
    assert "status=in_progress" in started.stdout
    assert "completed=1/5" in started.stdout

    resumed = runner.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir)],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "status=complete" in resumed.stdout

    reported = runner.invoke(
        cli.app,
        ["report", "build", str(run_dir)],
    )
    assert reported.exit_code == 0, reported.output
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["track"] == "C"


@pytest.mark.parametrize(
    "option",
    [
        "--agent-model",
        "--model-snapshot",
    ],
)
def test_cli_rejects_track_c_provider_options_on_local_resume(
    tmp_path,
    option: str,
) -> None:
    run_dir = tmp_path / "track-c-local"
    started = runner.invoke(
        cli.app,
        [
            "benchmark",
            "run",
            "--track",
            "C",
            "--configuration",
            "PLAIN_ASSISTANT",
            "--limit",
            "1",
            "--max-units",
            "1",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert started.exit_code == 0, started.output

    value = "candidate-model" if option == "--agent-model" else str(
        tmp_path / "snapshot.json"
    )
    resumed = runner.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir), option, value],
    )

    assert resumed.exit_code == 1
    assert "unused by this local run" in resumed.stderr


def test_cli_packages_optional_snapshot_pricing_and_run_artifacts(tmp_path) -> None:
    snapshot = tmp_path / "model_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "authenticated_discovery": True,
                "models": [],
                "capabilities": [],
                "qualifications": [],
            }
        ),
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps({"pricing_version": "test", "models": {}}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "result.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    destination = tmp_path / "release"

    result = runner.invoke(
        cli.app,
        [
            "benchmark",
            "package",
            "--output",
            str(destination),
            "--run-dir",
            str(run_dir),
            "--model-snapshot",
            str(snapshot),
            "--pricing-catalog",
            str(pricing),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads(
        (destination / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["model_snapshot_status"] == "included"
    assert manifest["run_artifact_status"] == "included"
    assert (destination / "models" / "model_snapshot.json").exists()
    assert (destination / "models" / "pricing_catalog.json").exists()
    assert (destination / "runs" / "run" / "metadata" / "result.json").exists()


def test_cli_rejects_track_c_policy_option(tmp_path) -> None:
    result = runner.invoke(
        cli.app,
        [
            "benchmark",
            "run",
            "--track",
            "C",
            "--policy",
            "always-listen",
            "--run-dir",
            str(tmp_path / "invalid"),
        ],
    )

    assert result.exit_code == 1
    assert "omit --policy" in result.stderr


def test_model_discovery_snapshot_is_write_once_and_key_free(
    tmp_path,
    monkeypatch,
) -> None:
    snapshot = ModelRegistrySnapshot(
        models=(ModelDescriptor(model_id="model-a", declared_owner="family-a"),),
        capabilities=(),
        authenticated_discovery=True,
    )

    async def fake_discovery(_settings):
        return snapshot

    monkeypatch.setattr(cli, "discover_model_snapshot", fake_discovery)
    destination = tmp_path / "models.json"

    first = runner.invoke(
        cli.app,
        ["models", "discover", "--output", str(destination)],
    )
    assert first.exit_code == 0, first.output
    document = json.loads(destination.read_text(encoding="utf-8"))
    serialized = json.dumps(document).lower()
    assert document["models"][0]["model_id"] == "model-a"
    assert "api_key" not in serialized
    assert "authorization" not in serialized

    refused = runner.invoke(
        cli.app,
        ["models", "discover", "--output", str(destination)],
    )
    assert refused.exit_code == 1
    assert "explicitly overwrite" in refused.stderr

    replaced = runner.invoke(
        cli.app,
        [
            "models",
            "discover",
            "--output",
            str(destination),
            "--overwrite",
        ],
    )
    assert replaced.exit_code == 0, replaced.output


def test_model_qualify_writes_twenty_probe_results_without_live_api(
    tmp_path,
    monkeypatch,
) -> None:
    registry = ModelRegistrySnapshot(
        models=(ModelDescriptor(model_id="model-a", declared_owner="family-a"),),
        capabilities=(
            ModelCapabilityReport(
                model_id="model-a",
                status=CapabilityStatus.READY,
                basic_chat=True,
                system_and_multiturn=True,
                chinese_text=True,
                usage_reported=True,
                preferred_structured_mode=StructuredMode.JSON_SCHEMA,
            ),
        ),
        authenticated_discovery=True,
    )

    async def fake_qualification(
        _settings,
        **_kwargs,
    ) -> QualifiedModelSnapshot:
        qualification = await ModelQualificationRunner(StaticQualificationExecutor()).qualify(
            "model-a"
        )
        return QualifiedModelSnapshot(
            registry=registry,
            qualifications=(qualification,),
        )

    monkeypatch.setattr(cli, "qualify_model_snapshot", fake_qualification)
    destination = tmp_path / "qualified-models.json"
    result = runner.invoke(
        cli.app,
        ["models", "qualify", "--output", str(destination)],
    )

    assert result.exit_code == 0, result.output
    assert "safety_suite=1" in result.stdout
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["qualification_suite_version"] == "zh-safety-micro-v1"
    assert document["qualifications"][0]["sample_count"] == 20
    assert document["qualifications"][0]["qualification_score"] == 100.0
    assert document["recommended_roles"]["companion_model"] == "model-a"
    assert document["recommended_roles"]["distinct_family_requirement_met"] is False


@pytest.mark.asyncio
async def test_qualify_pipeline_runs_safety_suite_with_fake_client(
    monkeypatch,
) -> None:
    fake_client = FakeQualifiedClient()
    monkeypatch.setattr(
        cli,
        "_provider_client",
        lambda settings, api_key: fake_client,
    )
    monkeypatch.setattr(cli, "ModelCapabilityProbe", FakeCapabilityProbe)

    snapshot = await cli.qualify_model_snapshot(
        Settings(LITELLM_API_KEY="test-key"),
        input_snapshot=None,
        requested_model_ids=(),
        max_concurrency=1,
    )

    assert len(snapshot.capabilities) == 1
    assert len(snapshot.qualifications) == 1
    assert snapshot.qualifications[0].sample_count == 20
    assert snapshot.qualifications[0].unknown_cost_probe_count == 20


def test_web_serve_delegates_to_uvicorn(monkeypatch) -> None:
    calls: list[tuple[str, str, int, bool]] = []

    def fake_run(
        target: str,
        *,
        host: str,
        port: int,
        reload: bool,
    ) -> None:
        calls.append((target, host, port, reload))

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(
        cli.app,
        ["web", "serve", "--host", "0.0.0.0", "--port", "9001", "--reload"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("ecocompanion.api:app", "0.0.0.0", 9001, True)]


def test_validation_reports_zero_cross_split_exact_leakage() -> None:
    report = cli.validate_benchmark()

    assert report.passed
    assert report.cross_split_template_family_count == 0
    assert report.cross_split_exact_utterance_count == 0
    assert report.cross_split_exact_event_count == 0
    assert report.cross_split_persona_count == 0
    assert report.near_duplicate_audit.passed
    assert report.near_duplicate_audit.ngram_size == 3
    assert report.near_duplicate_audit.threshold == 0.85
    assert sum(
        comparison.pairs_at_or_above_threshold
        for comparison in report.near_duplicate_audit.comparisons
    ) == 0
