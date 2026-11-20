"""CLI-level contract tests for formal Track B/C runs.

These tests keep the provider boundary fake while exercising the real CLI,
formal-runtime selection, resumable runners, and report builder.  They are
intentionally small (one selected case/scenario) so they can run in CI without
spending external API quota.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from typer.testing import CliRunner

from ecocompanion import cli
from ecocompanion.domain.schemas import StudentTurnV1, TraceEventV1
from ecocompanion.experiments.interactive_api_runtime import (
    INTERACTIVE_CACHE_NAME,
    INTERACTIVE_COST_LEDGER_NAME,
    INTERACTIVE_TRACE_NAME,
    FormalInteractiveRuntime,
    load_formal_interactive_selection,
)
from ecocompanion.infrastructure import CostLedger
from ecocompanion.providers.types import (
    ChatCompletion,
    CompletionUsage,
    ModelRole,
    StructuredAttempt,
    StructuredCompletion,
    StructuredMode,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

_RUNNER = CliRunner()
_MODEL_IDS = ("companion-a", "student-b", "judge-c", "judge-d")


class _FormalFakeGateway:
    """Deterministic provider fake that records every sampled model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float | None,
    ) -> ChatCompletion:
        del messages, max_tokens, temperature
        self.calls.append(("text", model))
        return _completion(
            model=model,
            content="我愿意听你说，也可以一起寻找现实中的支持。",
        )

    def complete_structured(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        schema: type[ModelT],
        max_tokens: int,
        temperature: float | None,
        modes: Sequence[StructuredMode],
    ) -> StructuredCompletion[ModelT]:
        del max_tokens, temperature
        self.calls.append(("structured", model))
        if schema is not StudentTurnV1:
            raise AssertionError(f"unexpected formal CLI schema: {schema.__name__}")
        raw_payload = messages[-1].get("content")
        if not isinstance(raw_payload, str):
            raise AssertionError("student renderer prompt must be JSON text")
        request = json.loads(raw_payload)
        value = schema.model_validate(
            {
                "episode_id": request["episode_id"],
                "step_id": request["step_id"],
                "disclosed_fact_ids": [],
                "text": request["base_utterance"],
            }
        )
        mode = next(iter(modes))
        return StructuredCompletion(
            value=value,
            completion=_completion(
                model=model,
                content=value.model_dump_json(),
            ),
            mode=mode,
            attempts=(StructuredAttempt(mode=mode, succeeded=True),),
        )


def _completion(*, model: str, content: str) -> ChatCompletion:
    return ChatCompletion(
        requested_model=model,
        response_model=model,
        content=content,
        finish_reason="stop",
        usage=CompletionUsage(input_tokens=10, output_tokens=5),
        retry_count=0,
        response_id=f"response:{model}",
    )


def _write_formal_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    families = {model_id: f"family-{index}" for index, model_id in enumerate(_MODEL_IDS)}
    snapshot = tmp_path / "model_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "authenticated_discovery": True,
                "qualification_suite_version": "zh-safety-micro-v1",
                "models": [
                    {
                        "model_id": model_id,
                        "declared_owner": families[model_id],
                        "created": 1,
                    }
                    for model_id in _MODEL_IDS
                ],
                "capabilities": [
                    {
                        "model_id": model_id,
                        "status": "ready",
                        "basic_chat": True,
                        "system_and_multiturn": True,
                        "chinese_text": True,
                        "usage_reported": True,
                        "streaming": True,
                        "preferred_structured_mode": "json_schema",
                        "eligible_roles": [role.value for role in ModelRole],
                        "response_model_ids": [],
                        "failures": [],
                    }
                    for model_id in _MODEL_IDS
                ],
                "qualifications": [
                    {
                        "model_id": model_id,
                        "sample_count": 20,
                        "qualification_score": 98.0 - index,
                        "mean_cost_cny": 0.01,
                    }
                    for index, model_id in enumerate(_MODEL_IDS)
                ],
                "recommended_roles": {
                    "companion_model": "companion-a",
                    "student_renderer_model": "student-b",
                    "judge_models": ["judge-c", "judge-d"],
                },
            }
        ),
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing_catalog.json"
    pricing.write_text(
        json.dumps(
            {
                "pricing_version": "formal-cli-test-v1",
                "models": {
                    model_id: {
                        "input_per_million": "2",
                        "output_per_million": "4",
                    }
                    for model_id in _MODEL_IDS
                },
            }
        ),
        encoding="utf-8",
    )
    return snapshot, pricing


def _patch_formal_builder(
    monkeypatch,
    *,
    gateways: list[_FormalFakeGateway],
) -> None:
    def build_fake_runtime(
        *,
        settings: Any,
        model_snapshot_path: Path,
        pricing_catalog_path: Path,
        companion_model_id: str | None = None,
        student_model_id: str | None = None,
        judge_model_ids: Sequence[str] | None = None,
    ) -> FormalInteractiveRuntime:
        del settings
        selection = load_formal_interactive_selection(
            model_snapshot_path=model_snapshot_path,
            pricing_catalog_path=pricing_catalog_path,
            companion_model_id=companion_model_id,
            student_model_id=student_model_id,
            judge_model_ids=judge_model_ids,
        )
        gateway = _FormalFakeGateway()
        gateways.append(gateway)
        return FormalInteractiveRuntime(
            selection=selection,
            gateway=gateway,
        )

    monkeypatch.setattr(cli, "build_openai_formal_interactive_runtime", build_fake_runtime)


def _nonempty_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sqlite_count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _assert_formal_artifacts_reconcile(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime = manifest["interactive_runtime"]
    assert runtime["mode"] == "formal_api"
    assert [item["model_id"] for item in runtime["role_bindings"]] == list(_MODEL_IDS)
    assert len({item["family"] for item in runtime["role_bindings"]}) == 4

    ledger = CostLedger(path=run_dir / INTERACTIVE_COST_LEDGER_NAME)
    ledger_records = ledger.records()
    trace_events = [
        TraceEventV1.model_validate(item)
        for item in _nonempty_jsonl(run_dir / INTERACTIVE_TRACE_NAME)
    ]
    cache_count = _sqlite_count(run_dir / INTERACTIVE_CACHE_NAME, "exact_cache")
    journal_count = _sqlite_count(
        run_dir / f"{Path(INTERACTIVE_COST_LEDGER_NAME).stem}.calls.sqlite3",
        "interactive_call_journal",
    )
    journal_states = _journal_states(
        run_dir / f"{Path(INTERACTIVE_COST_LEDGER_NAME).stem}.calls.sqlite3"
    )

    assert len({record.call_id for record in ledger_records}) == len(ledger_records)
    assert len({event.trace_id for event in trace_events}) == len(trace_events)
    assert cache_count == len(ledger_records)
    assert journal_count == len(ledger_records)
    assert journal_states == {"committed": len(ledger_records)}
    trace_models = Counter(usage.model for event in trace_events for usage in event.model_usage)
    ledger_models = Counter(record.model for record in ledger_records)
    assert trace_models == ledger_models
    assert all(record.cost_cny is not None for record in ledger_records)


def _journal_states(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT state, COUNT(*) FROM interactive_call_journal GROUP BY state"
        ).fetchall()
    return {str(state): int(count) for state, count in rows}


def test_formal_track_b_cli_run_resume_report_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot, pricing = _write_formal_artifacts(tmp_path)
    gateways: list[_FormalFakeGateway] = []
    _patch_formal_builder(monkeypatch, gateways=gateways)
    run_dir = tmp_path / "track-b-formal"
    common = [
        "--formal-api",
        "--model-snapshot",
        str(snapshot),
        "--pricing-catalog",
        str(pricing),
    ]

    started = _RUNNER.invoke(
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
            *common,
        ],
    )
    assert started.exit_code == 0, started.output
    assert "status=in_progress" in started.stdout
    first_call_count = len(gateways[0].calls)
    assert first_call_count > 0

    resumed = _RUNNER.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir), *common],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "status=complete" in resumed.stdout
    calls_after_completion = sum(len(gateway.calls) for gateway in gateways)

    repeated = _RUNNER.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir), *common],
    )
    assert repeated.exit_code == 0, repeated.output
    assert "executed=0" in repeated.stdout
    assert gateways[-1].calls == []
    assert sum(len(gateway.calls) for gateway in gateways) == calls_after_completion

    reported = _RUNNER.invoke(cli.app, ["report", "build", str(run_dir)])
    assert reported.exit_code == 0, reported.output
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["track"] == "B"
    _assert_formal_artifacts_reconcile(run_dir)


def test_formal_track_c_cli_run_resume_report_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot, pricing = _write_formal_artifacts(tmp_path)
    gateways: list[_FormalFakeGateway] = []
    _patch_formal_builder(monkeypatch, gateways=gateways)
    run_dir = tmp_path / "track-c-formal"
    common = [
        "--formal-api",
        "--model-snapshot",
        str(snapshot),
        "--pricing-catalog",
        str(pricing),
    ]

    started = _RUNNER.invoke(
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
            *common,
        ],
    )
    assert started.exit_code == 0, started.output
    assert "status=in_progress" in started.stdout
    assert len(gateways[0].calls) > 0

    resumed = _RUNNER.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir), *common],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "status=complete" in resumed.stdout
    calls_after_completion = sum(len(gateway.calls) for gateway in gateways)

    repeated = _RUNNER.invoke(
        cli.app,
        ["benchmark", "resume", str(run_dir), *common],
    )
    assert repeated.exit_code == 0, repeated.output
    assert "executed=0" in repeated.stdout
    assert gateways[-1].calls == []
    assert sum(len(gateway.calls) for gateway in gateways) == calls_after_completion

    reported = _RUNNER.invoke(cli.app, ["report", "build", str(run_dir)])
    assert reported.exit_code == 0, reported.output
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["track"] == "C"
    _assert_formal_artifacts_reconcile(run_dir)


def test_formal_resume_rejects_model_snapshot_digest_drift_before_api_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot, pricing = _write_formal_artifacts(tmp_path)
    drifted_snapshot = tmp_path / "drifted_snapshot.json"
    drifted_snapshot.write_text(
        snapshot.read_text(encoding="utf-8").replace('"created": 1', '"created": 2'),
        encoding="utf-8",
    )
    gateways: list[_FormalFakeGateway] = []
    _patch_formal_builder(monkeypatch, gateways=gateways)
    run_dir = tmp_path / "track-c-drift"
    common = [
        "--formal-api",
        "--model-snapshot",
        str(snapshot),
        "--pricing-catalog",
        str(pricing),
    ]

    started = _RUNNER.invoke(
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
            *common,
        ],
    )
    assert started.exit_code == 0, started.output
    previous_gateway_count = len(gateways)

    drifted = _RUNNER.invoke(
        cli.app,
        [
            "benchmark",
            "resume",
            str(run_dir),
            "--formal-api",
            "--model-snapshot",
            str(drifted_snapshot),
            "--pricing-catalog",
            str(pricing),
        ],
    )
    assert drifted.exit_code == 1
    assert "interactive runtime does not match" in drifted.stderr
    assert len(gateways) == previous_gateway_count + 1
    assert gateways[-1].calls == []
