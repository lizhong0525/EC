import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ecocompanion.api import (
    _qualified_model_from_snapshot,
    build_runtime,
    create_app,
)
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.config import Settings
from ecocompanion.experiments.report import build_report
from ecocompanion.experiments.runner import start_experiment


def test_api_session_sse_memory_and_trace_round_trip(tmp_path: Path) -> None:
    settings = Settings(
        LITELLM_API_KEY=None,
        ECO_DEFAULT_MODEL=None,
        ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
        ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
    )
    application = create_app(settings=settings)

    with TestClient(application) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["provider_configured"] is False

        created = client.post(
            "/api/v1/sessions",
            json={
                "participant_type": "adult_research_tester",
                "memory_enabled": True,
                "proactive_enabled": False,
            },
        )
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        streamed = client.post(
            f"/api/v1/sessions/{session_id}/turn",
            headers={"Accept": "text/event-stream"},
            json={"text": "最近总是睡不着，也没什么兴趣。"},
        )
        assert streamed.status_code == 200
        assert "event: token" in streamed.text
        done_line = next(
            line
            for frame in streamed.text.split("\n\n")
            if frame.startswith("event: done")
            for line in frame.splitlines()
            if line.startswith("data:")
        )
        result = json.loads(done_line.removeprefix("data: "))
        assert result["result"]["session"]["state"]["turn_index"] == 1

        restored_session = client.get(f"/api/v1/sessions/{session_id}")
        assert restored_session.status_code == 200
        assert restored_session.json()["state"]["turn_index"] == 1
        restored_messages = client.get(
            f"/api/v1/sessions/{session_id}/messages"
        )
        assert restored_messages.status_code == 200
        assert [item["role"] for item in restored_messages.json()] == [
            "USER",
            "COMPANION",
        ]
        assert restored_messages.json()[0]["content"].startswith("最近")

        memories = client.get(f"/api/v1/sessions/{session_id}/memory")
        assert memories.status_code == 200
        assert len(memories.json()) == 2
        memory_id = memories.json()[0]["memory_id"]
        deleted = client.delete(
            f"/api/v1/sessions/{session_id}/memory/{memory_id}"
        )
        assert deleted.status_code == 204

        added = client.post(
            f"/api/v1/sessions/{session_id}/memory",
            json={
                "kind": "PREFERENCE",
                "content": "请叫我小林",
                "user_authorized": True,
            },
        )
        assert added.status_code == 201
        corrected = client.put(
            f"/api/v1/sessions/{session_id}/memory/{added.json()['memory_id']}",
            json={
                "content": "请叫我林同学",
                "user_authorized": True,
            },
        )
        assert corrected.status_code == 200
        assert corrected.json()["content"] == "请叫我林同学"

        traces = client.get(f"/api/v1/research/sessions/{session_id}/trace")
        assert traces.status_code == 200
        assert len(traces.json()) == 1
        assert traces.json()[0]["selected_action"]

        models = client.get("/api/v1/models")
        assert models.status_code == 200
        assert models.json()["source"] == "unavailable"


def test_api_keeps_safety_ledger_out_of_user_memory_controls(tmp_path: Path) -> None:
    application = create_app(
        settings=Settings(
            LITELLM_API_KEY=None,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    with TestClient(application) as client:
        created = client.post(
            "/api/v1/sessions",
            json={"participant_type": "adult_research_tester"},
        )
        session_id = created.json()["session_id"]
        turn = client.post(
            f"/api/v1/sessions/{session_id}/turn",
            json={"text": "我现在不想活了，怕马上伤害自己。"},
        )
        assert turn.status_code == 200

        memories = client.get(f"/api/v1/sessions/{session_id}/memory")
        assert memories.status_code == 200
        assert all(item["kind"] != "SAFETY_LEDGER" for item in memories.json())
        ledger_path = (
            tmp_path / "artifacts" / "memory" / "events.jsonl"
        )
        ledger_ids = [
            json.loads(line)["record"]["memory_id"]
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if '"SAFETY_LEDGER"' in line
        ]
        assert len(ledger_ids) == 1
        ledger_id = ledger_ids[0]

        deleted = client.delete(
            f"/api/v1/sessions/{session_id}/memory/{ledger_id}"
        )
        corrected = client.put(
            f"/api/v1/sessions/{session_id}/memory/{ledger_id}",
            json={"content": "不应修改", "user_authorized": True},
        )
        assert deleted.status_code == 403
        assert corrected.status_code == 403


def test_api_rejects_unknown_synthetic_case(tmp_path: Path) -> None:
    application = create_app(
        settings=Settings(
            LITELLM_API_KEY=None,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/sessions",
            json={
                "participant_type": "synthetic_case",
                "synthetic_case_id": "does-not-exist",
            },
        )
    assert response.status_code == 400


def test_api_runtime_can_disable_research_trace_access(tmp_path: Path) -> None:
    application = create_app(
        settings=Settings(
            LITELLM_API_KEY=None,
            ECO_RESEARCH_MODE=False,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    with TestClient(application) as client:
        denied_default = client.post(
            "/api/v1/sessions",
            json={"participant_type": "adult_research_tester"},
        )
        assert denied_default.status_code == 403
        assert "disabled" in denied_default.json()["detail"]

        created = client.post(
            "/api/v1/sessions",
            json={
                "participant_type": "adult_research_tester",
                "research_mode": False,
            },
        )
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        trace = client.get(f"/api/v1/research/sessions/{session_id}/trace")
        assert trace.status_code == 403


def test_api_rejects_turn_replay_before_advancing_session(tmp_path: Path) -> None:
    application = create_app(
        settings=Settings(
            LITELLM_API_KEY=None,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    with TestClient(application) as client:
        created = client.post(
            "/api/v1/sessions",
            json={"participant_type": "adult_research_tester"},
        )
        session_id = created.json()["session_id"]
        first = client.post(
            f"/api/v1/sessions/{session_id}/turn",
            json={
                "text": "我现在不想活了。",
                "request_id": "api-replay-001",
            },
        )
        assert first.status_code == 200

        replay = client.post(
            f"/api/v1/sessions/{session_id}/turn",
            json={
                "text": "我现在不想活了。",
                "request_id": "api-replay-001",
            },
        )
        assert replay.status_code == 409
        assert "already committed" in replay.json()["detail"]

        session = client.get(f"/api/v1/sessions/{session_id}")
        assert session.status_code == 200
        assert session.json()["state"]["turn_index"] == 1
        messages = client.get(f"/api/v1/sessions/{session_id}/messages")
        assert len(messages.json()) == 2


def test_api_proactivity_requires_opt_in_and_enforces_daily_limit(
    tmp_path: Path,
) -> None:
    application = create_app(
        settings=Settings(
            LITELLM_API_KEY=None,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    with TestClient(application) as client:
        created = client.post(
            "/api/v1/sessions",
            json={
                "participant_type": "adult_research_tester",
                "proactive_enabled": False,
            },
        )
        session_id = created.json()["session_id"]

        disabled_claim = client.post(
            f"/api/v1/sessions/{session_id}/proactivity/claim"
        )
        assert disabled_claim.status_code == 403

        enabled = client.put(
            f"/api/v1/sessions/{session_id}/proactivity",
            json={"enabled": True},
        )
        assert enabled.status_code == 200
        assert enabled.json()["state"]["proactive_enabled"] is True

        first_claim = client.post(
            f"/api/v1/sessions/{session_id}/proactivity/claim"
        )
        assert first_claim.status_code == 200
        assert first_claim.json()["text"]
        assert first_claim.json()["used_fallback"] is True

        outstanding_claim = client.post(
            f"/api/v1/sessions/{session_id}/proactivity/claim"
        )
        assert outstanding_claim.status_code == 409
        assert "AWAITING_RESPONSE" in outstanding_claim.json()["detail"]

        response = client.post(
            f"/api/v1/sessions/{session_id}/turn",
            json={"text": "我看到了，谢谢。"},
        )
        assert response.status_code == 200

        daily_limit = client.post(
            f"/api/v1/sessions/{session_id}/proactivity/claim"
        )
        assert daily_limit.status_code == 409
        assert "DAILY_LIMIT" in daily_limit.json()["detail"]

        disabled = client.put(
            f"/api/v1/sessions/{session_id}/proactivity",
            json={"enabled": False},
        )
        assert disabled.status_code == 200
        assert disabled.json()["state"]["proactive_enabled"] is False


def test_api_can_explicitly_resume_paused_proactivity(tmp_path: Path) -> None:
    application = create_app(
        settings=Settings(
            LITELLM_API_KEY=None,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    with TestClient(application) as client:
        created = client.post(
            "/api/v1/sessions",
            json={"participant_type": "adult_research_tester"},
        )
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        enabled = client.put(
            f"/api/v1/sessions/{session_id}/proactivity",
            json={"enabled": True},
        )
        assert enabled.status_code == 200

        scheduler = application.state.runtime.service._proactivity
        scheduler.record_sent(
            prompt_id="api-resume-prompt-001",
            session_id=session_id,
            sent_at=datetime(2026, 9, 3, 9, 0, tzinfo=UTC),
        )
        scheduler.record_unanswered(
            "api-resume-prompt-001",
            resolved_at=datetime(2026, 9, 3, 10, 0, tzinfo=UTC),
        )
        scheduler.record_sent(
            prompt_id="api-resume-prompt-002",
            session_id=session_id,
            sent_at=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
        )
        scheduler.record_unanswered(
            "api-resume-prompt-002",
            resolved_at=datetime(2026, 9, 4, 10, 0, tzinfo=UTC),
        )
        assert scheduler.get_state(session_id).paused

        resumed = client.post(
            f"/api/v1/sessions/{session_id}/proactivity/resume"
        )
        assert resumed.status_code == 200
        assert resumed.json()["state"]["proactive_enabled"] is True
        assert not scheduler.get_state(session_id).paused


def test_runtime_selects_highest_scoring_ready_qualified_model(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "model_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "model_id": "model-a",
                        "status": "ready",
                        "basic_chat": True,
                        "chinese_text": True,
                    },
                    {
                        "model_id": "model-b",
                        "status": "ready",
                        "basic_chat": True,
                        "chinese_text": True,
                    },
                    {
                        "model_id": "model-c",
                        "status": "degraded",
                        "basic_chat": True,
                        "chinese_text": True,
                    },
                ],
                "qualifications": [
                    {"model_id": "model-a", "qualification_score": 81.0},
                    {"model_id": "model-b", "qualification_score": 92.0},
                    {"model_id": "model-c", "qualification_score": 99.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _qualified_model_from_snapshot(snapshot_path) == "model-b"


def test_provider_backed_runtime_requires_explicit_token_prices(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "model_snapshot.json").write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "model_id": "model-a",
                        "status": "ready",
                        "basic_chat": True,
                        "chinese_text": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PRICE"):
        build_runtime(
            Settings(
                LITELLM_API_KEY="test-token",
                ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
                ECO_DATABASE_URL=(
                    f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}"
                ),
            )
        )


def test_explicit_web_model_can_start_without_qualification_snapshot(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(
        Settings(
            LITELLM_API_KEY="test-token",
            ECO_DEFAULT_MODEL="web-model",
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=tmp_path / "artifacts",
        )
    )
    try:
        assert runtime.client is not None
        assert runtime.service.model == "web-model"
    finally:
        asyncio.run(runtime.close())


def test_api_lists_benchmark_runs_and_returns_validated_report(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "runs" / "web-visible-run"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.PUBLIC_TEST,
        policy_names=("history-aware-rule",),
        surface_indices=(0,),
        repeats=1,
        case_limit=1,
        run_id="web-visible-run",
    )
    build_report(run_dir)
    application = create_app(
        settings=Settings(
            LITELLM_API_KEY=None,
            ECO_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
            ECO_ARTIFACTS_DIR=artifacts,
        )
    )

    with TestClient(application) as client:
        catalog = client.get("/api/v1/research/benchmark/runs")
        assert catalog.status_code == 200
        assert catalog.json()["runs"][0]["run_id"] == "web-visible-run"
        assert catalog.json()["runs"][0]["report_available"] is True

        report = client.get(
            "/api/v1/research/benchmark/runs/web-visible-run/report"
        )
        assert report.status_code == 200
        assert report.json()["run"]["track"] == "A"
        assert report.json()["report"]["complete"] is True
        assert report.json()["report"]["rows"][0]["policy_name"] == (
            "history-aware-rule"
        )

        missing = client.get(
            "/api/v1/research/benchmark/runs/not-found/report"
        )
        assert missing.status_code == 404
