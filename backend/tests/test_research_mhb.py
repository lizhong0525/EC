"""Synthetic service fixtures for the external MentalHealthBench workflow."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research import mhb, mhb_protocol
from ecocompanion.research.cli import app
from scripts import external_datasets as intake


def _record(index: int) -> dict[str, Any]:
    return {
        "id": f"{index:032x}",
        "conversation": {
            "messages": [
                {"role": "user", "content": "SYNTHETIC engineering input only"},
                {"role": "assistant", "content": "SYNTHETIC prior reply"},
                {"role": "user", "content": "SYNTHETIC current question"},
            ]
        },
        "rubric_items": [
            {"criterion_text": "SYNTHETIC positive", "points": 5, "behavior_axis": "axis-a"},
            {"criterion_text": "SYNTHETIC negative", "points": -5, "behavior_axis": "axis-b"},
            {"criterion_text": "SYNTHETIC other", "points": 5, "behavior_axis": "axis-a"},
        ],
        "acuity": "non_acute",
        "user_profile": "adult",
        "language": "en",
        "has_prior_context": True,
        "canary_string": "DO_NOT_SEND_CANARY_827194",
    }


def _snapshot(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    records = [_record(0), _record(1)]
    content = ("\n".join(json.dumps(record) for record in records) + "\n").encode()
    counts = {
        "records": 2,
        "unique_ids": 2,
        "rubric_items": 6,
        "source_digest": hashlib.sha256(content).hexdigest(),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mentalhealthbench/mentalhealthbench_eval.jsonl", content)
    archive_bytes = buffer.getvalue()
    spec = dataclasses.replace(
        intake.source("mentalhealthbench"),
        identity=hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(intake, "SOURCE_ORDER", (spec,))
    monkeypatch.setattr(mhb, "verify_snapshot", intake.verify)
    monkeypatch.setattr(intake, "inventory_mentalhealthbench", lambda _: counts)
    intake.write_snapshot(
        root / "snapshot",
        spec,
        {"archive.zip": archive_bytes, "mentalhealthbench_eval.jsonl": content},
        counts,
        "mentalhealthbench_eval.jsonl",
        "2026-10-25T00:00:00+00:00",
        {},
    )
    return root / "snapshot"


@pytest.fixture
def packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    snapshot = _snapshot(tmp_path, monkeypatch)
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "paper.pdf").write_bytes(b"SYNTHETIC PDF BYTES")
    paper_hash = hashlib.sha256(b"SYNTHETIC PDF BYTES").hexdigest()
    write_json_atomic(
        paper / "paper-manifest.json",
        {
            "version": mhb.PAPER_SNAPSHOT_VERSION,
            "paper_url": mhb.PAPER_URL,
            "paper_sha256": paper_hash,
            "bytes": 19,
        },
        overwrite=False,
    )
    monkeypatch.setattr(mhb, "verify_paper", lambda path: read_json_object(
        path / "paper-manifest.json"
    ))
    monkeypatch.setattr(mhb, "PAPER_SHA256", paper_hash)
    result = mhb.prepare(snapshot, paper, tmp_path / "packet", scope="smoke")
    assert result["tasks"] == 2
    return tmp_path / "packet"


def test_prompt_literal_braces_and_complete_request_have_no_labels() -> None:
    record = _record(0)
    messages = mhb_protocol.visible_conversation(record)
    payload = mhb_protocol.grader_request(
        "fixture-model", messages, "reply", "criterion", max_tokens=100
    )
    text = payload["messages"][0]["content"]
    assert '"satisfaction": 0' in text
    assert "{conversation_prefix}" not in text
    generation = mhb_protocol.generation_request(
        "fixture-model", messages, max_tokens=100, temperature=0, seed=1
    )
    serialized = json.dumps(generation)
    assert "canary" not in serialized.lower()
    assert record["id"] not in serialized
    assert generation["messages"] == messages


def test_prepare_separates_references_and_refuses_mutation(packet: Path) -> None:
    loaded = mhb.load_packet(packet)
    task = loaded["tasks"][0]
    assert set(task) == {"task_id", "messages"}
    assert "rubric" not in json.dumps(task)
    assert "source_id" not in json.dumps(task)
    assert len(loaded["references"][task["task_id"]]["rubric_items"]) == 3
    (packet / "tasks.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        mhb.load_packet(packet)


def test_native_signed_score_and_axis_denominator() -> None:
    items = [
        {"criterion_id": "a", "points": 5, "behavior_axis": "axis-a"},
        {"criterion_id": "b", "points": 5, "behavior_axis": "axis-a"},
        {"criterion_id": "c", "points": -5, "behavior_axis": "axis-b"},
    ]
    result = mhb.score_completion(items, {"a": 1, "b": 0, "c": 1})
    assert result["signed_score"] == 0
    assert result["positive_contribution"] == 0.5
    assert result["penalty_burden"] == 0.5
    assert result["behavior_axes"]["axis-b"]["signed_contribution"] == -0.5
    changed = mhb.score_completion(items, {"a": 0, "b": 0, "c": 1})
    assert changed["clipped_score"] == 0
    assert changed["signed_score"] == -0.5


def test_incomplete_judgments_are_not_zeros() -> None:
    items = [{"criterion_id": "a", "points": 5, "behavior_axis": "axis-a"}]
    assert mhb.score_completion(items, None)["signed_score"] is None
    with pytest.raises(ValueError, match="missing or unknown"):
        mhb.score_completion(items, {})


def test_bootstrap_absent_axis_is_zero_and_small_sample_is_unavailable() -> None:
    tasks = []
    for index in range(30):
        axes = {"axis-a": {
            "positive_contribution": 1.0, "penalty_burden": 0.0,
            "signed_contribution": 1.0,
        }}
        if index % 2 == 0:
            axes["axis-b"] = {
                "positive_contribution": 0.0, "penalty_burden": 0.5,
                "signed_contribution": -0.5,
            }
        tasks.append({
            "task_id": f"fixture-{index}",
            "status": "COMPLETE",
            "mean_signed_score": 0.5,
            "mean_clipped_score": 0.5,
            "mean_positive_contribution": 0.5,
            "mean_penalty_burden": 0.0,
            "behavior_axes": axes,
        })
    result = mhb._bootstrap(tasks)
    assert result["clusters"] == 30
    assert len(result["intervals"]["axis:axis-b:penalty_burden"]) == 2
    assert mhb._bootstrap(tasks[:2])["intervals"] is None


def _raw(content: str, model: str = "fixture-model") -> dict[str, Any]:
    return {
        "model": model,
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10,
        "eval_count": 5,
        "thinking": "DO_NOT_STORE_PRIVATE_92741",
        "arbitrary_secret": "DO_NOT_STORE_SERVER_FIELD_92741",
    }


def _mock_local(
    monkeypatch: pytest.MonkeyPatch, *, fail_judgment: bool = False,
    empty_generation: bool = False,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "fixture-model", "digest": "fixture-digest"}
            ]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "fixture-runtime"})
        assert request.url.host == "127.0.0.1"
        payload = json.loads(request.content)
        requests.append(payload)
        if "rubric_criterion" in payload["messages"][0]["content"]:
            content = "INVALID" if fail_judgment else json.dumps({
                "explanation": "SYNTHETIC JUDGMENT", "satisfaction": 1
            })
        else:
            assert "SYNTHETIC positive" not in json.dumps(payload)
            assert "source_id" not in json.dumps(payload)
            content = "" if empty_generation else "SYNTHETIC generated reply"
        return httpx.Response(200, json=_raw(content))

    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(mhb.httpx, "Client", client)
    return requests


def test_mock_generation_grading_replay_and_no_thinking_persistence(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _mock_local(monkeypatch)
    config = tmp_path / "config.json"
    mhb.bind_local(packet, "fixture-model", config)
    output = tmp_path / "run"
    result = mhb.run_local(packet, config, output)
    assert result["status"] == "COMPLETE_LOCAL_UNCALIBRATED_DIAGNOSTIC"
    assert result["complete_tasks"] == 2
    assert len(requests) == 8
    assert result["primary_macro_clipped_score_overall"] == pytest.approx(0.5)
    assert result["new_project_action_gold"] == 0
    assert result["official_judge_used"] is False
    assert result["simulation_state_recovery"].startswith("NOT_APPLICABLE")
    assert result["bootstrap"]["intervals"] is None
    assert mhb.score_run(packet, config, output) == result
    report_path = tmp_path / "independent-report.json"
    mhb.report(packet, config, output, report_path)
    assert read_json_object(report_path) == result
    trace = next((output / "records/generations").glob("*.json")).read_text()
    assert "PRIVATE_92741" not in trace
    assert mhb.run_local(packet, config, output, resume=True) == result


def test_failed_judgment_keeps_incomplete_and_blocks_resume(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_local(monkeypatch, fail_judgment=True)
    config = tmp_path / "config.json"
    mhb.bind_local(packet, "fixture-model", config)
    output = tmp_path / "failed"
    result = mhb.run_local(packet, config, output)
    assert result["status"] == "INCOMPLETE_LOCAL_DIAGNOSTIC_NOT_BENCHMARK_SCORE"
    assert result["primary_macro_clipped_score_overall"] is None
    assert result["generation"]["recorded_calls"] == 1
    assert result["grading"]["failed"] == 1
    assert result["complete_tasks"] == 0
    assert result["total_model_calls"] == 2
    with pytest.raises(ValueError, match="cannot be retried"):
        mhb.run_local(packet, config, output, resume=True)


def test_empty_completion_zero_without_judge_calls(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _mock_local(monkeypatch, empty_generation=True)
    config = tmp_path / "config.json"
    mhb.bind_local(packet, "fixture-model", config)
    result = mhb.run_local(packet, config, tmp_path / "empty")
    assert len(requests) == 2
    assert result["primary_macro_clipped_score_overall"] == 0
    assert result["grading"]["empty_completions_scored_zero"] == 2
    assert result["grading"]["missing_records_excluding_empty_completions"] == 0


def test_changed_model_digest_blocks_before_inference(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_local(monkeypatch)
    config = tmp_path / "config.json"
    mhb.bind_local(packet, "fixture-model", config)
    monkeypatch.setattr(mhb, "_model_digest", lambda model: "CHANGED")
    with pytest.raises(ValueError, match="identity changed"):
        mhb.run_local(packet, config, tmp_path / "never")
    assert not (tmp_path / "never/records").exists()


def test_explicit_resume_skips_only_unattempted_work(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _mock_local(monkeypatch)
    config = tmp_path / "config.json"
    mhb.bind_local(packet, "fixture-model", config)
    loaded = mhb.load_packet(packet)
    task = loaded["tasks"][0]
    output = tmp_path / "interrupted"
    started = {
        "version": mhb.VERSION,
        "created_at_utc": "2026-10-25T00:00:00+00:00",
        "config_sha256": mhb_protocol.sha256(config.read_bytes()),
        "packet_protocol_sha256": mhb_protocol.sha256(
            (packet / "protocol.json").read_bytes()
        ),
        "task_set_digest": mhb.digest(loaded["tasks"]),
        "stage": "all",
        "max_generation_calls": 2,
        "max_grading_calls": 6,
        "maximum_inference_calls_before_empty_reply_exemption": 8,
        "max_retries": 0,
        "read_timeout_seconds": mhb_protocol.READ_TIMEOUT_SECONDS,
        "runtime_version": "fixture-runtime",
    }
    write_json_atomic(output / "started.json", started, overwrite=False)
    payload = mhb._generation_payload(loaded["protocol"], "fixture-model", task, 0)
    generation = {
        "version": mhb.VERSION,
        "kind": "generation",
        "task_id": task["task_id"],
        "sample_index": 0,
        "task_sha256": mhb.digest(task),
        "config_sha256": started["config_sha256"],
        "request": payload,
        "request_sha256": mhb.digest(payload),
        "calls": 1,
        "attempt": 1,
        "model_digest_verified_before_call": "fixture-digest",
        "status": "ok",
        "raw_response": mhb.visible_response(_raw("SYNTHETIC generated reply")),
    }
    write_json_atomic(mhb._generation_path(output, task["task_id"], 0), generation,
                      overwrite=False)
    result = mhb.run_local(packet, config, output, resume=True)
    assert result["complete_tasks"] == 2
    assert len(requests) == 7
    assert result["generation"]["recorded_calls"] == 2


def test_tampered_verdict_is_rejected_by_replay(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_local(monkeypatch)
    config = tmp_path / "config.json"
    mhb.bind_local(packet, "fixture-model", config)
    output = tmp_path / "tampered"
    mhb.run_local(packet, config, output)
    path = next((output / "records/judgments").glob("*.json"))
    record = read_json_object(path)
    record["verdict"]["satisfaction"] = 0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="valid identity or verdict"):
        mhb.score_run(packet, config, output)


def test_mhb_cli_prepare_is_registered() -> None:
    result = CliRunner().invoke(app, ["mhb", "--help"])
    assert result.exit_code == 0
    assert "bind-local" in result.stdout
    assert "report" in result.stdout
