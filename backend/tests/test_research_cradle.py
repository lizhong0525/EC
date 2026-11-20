"""Engineering fixtures only; never substitute for clinician study data."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from ecocompanion.research import cradle
from ecocompanion.research.cli import app


def rows() -> list[dict[str, Any]]:
    return [
        {"turn_id": 0, "text": "User: ordinary message", "labels": ""},
        {"turn_id": 1, "text": "Listener: listener-only text", "labels": "alert_past"},
        {
            "turn_id": 2,
            "text": "User: I plan to kill myself",
            "labels": "confirm_SI_active_ongoing",
        },
        {"turn_id": 0, "text": "User: ordinary second dialogue", "labels": ""},
    ]


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    (root / "data").mkdir(parents=True)
    (root / "README.md").write_text("Fixture, not expert data.", encoding="utf-8")
    for split in cradle.SPLITS:
        pq.write_table(pa.Table.from_pylist(rows()), root / f"data/{split}-00000-of-00001.parquet")
    for filename in ("upstream-metadata.json", "upstream-tree.json"):
        (root / filename).write_text("{}", encoding="utf-8")
    files = {}
    for path in [root / "README.md", *sorted((root / "data").glob("*.parquet"))]:
        content = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    manifest = {
        "dataset": cradle.DATASET,
        "revision": cradle.DEFAULT_REVISION,
        "files": files,
        "license": "UNSPECIFIED",
        "metadata_sha256": hashlib.sha256(
            (root / "upstream-metadata.json").read_bytes()
        ).hexdigest(),
        "tree_sha256": hashlib.sha256((root / "upstream-tree.json").read_bytes()).hexdigest(),
    }
    (root / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_parser_recovers_reset_boundaries_and_preserves_listener_labels() -> None:
    result = cradle.parse_dialogues(rows())
    assert len(result) == 2
    assert result[0][1]["labels"] == ["alert_past"]
    assert result[1][0]["row"] == 3


@pytest.mark.parametrize("change", ["gap", "role", "column", "label", "duplicate", "null", "first"])
def test_invalid_source_contract_is_rejected(change: str) -> None:
    source = rows()
    if change == "gap":
        source[2]["turn_id"] = 5
    elif change == "role":
        source[1]["text"] = "Unprefixed bullet"
    elif change == "column":
        source[0]["diagnosis"] = "do not expose"
    elif change == "label":
        source[0]["labels"] = "invented"
    elif change == "duplicate":
        source[0]["labels"] = "alert_past; alert_past"
    elif change == "null":
        source[0]["text"] = None
    else:
        source[0]["turn_id"] = 1
    with pytest.raises(ValueError):
        cradle.parse_dialogues(source)


def test_audit_unknown_role_does_not_invent_speaker() -> None:
    source = rows()
    source[1]["text"] = "- continuation, actual speaker unknown"
    parsed = cradle.parse_dialogues(source, strict_roles=False)
    assert parsed[0][1]["role"] == "Unknown"
    assert parsed[0][1]["text"] == source[1]["text"]


def test_visible_prefix_has_no_future_labels_source_ids_or_split() -> None:
    nodes = list(cradle.visible_nodes(cradle.parse_dialogues(rows())))
    first, second = nodes[0][0], nodes[1][0]
    assert first["history"] == []
    assert "kill myself" not in json.dumps(first)
    assert "listener-only text" in json.dumps(second)
    assert "confirm_SI_active_ongoing" not in json.dumps(second)
    assert not ({"labels", "source_row", "source_split", "dialogue_ordinal"} & set(second))
    assert len(first["dialogue_id"]) == len(first["observation_id"]) == 32
    assert nodes[0][1]["labels"] == []
    assert nodes[1][1]["labels"] == ["confirm_SI_active_ongoing"]
    observation = cradle.project_observation(second)
    assert observation.visible_profile.age_group == "unspecified"
    assert observation.visible_profile.language == "en"
    assert len(observation.visible_context.conversation_history) == 1
    assert "listener-only" not in observation.model_dump_json()


def test_future_change_cannot_change_earlier_view() -> None:
    original = cradle.parse_dialogues(rows())
    changed = copy.deepcopy(original)
    changed[0][2]["text"] = "future changed"
    changed[0][2]["labels"] = ["alert_ongoing"]
    a = next(cradle.visible_nodes(original))[0]
    b = next(cradle.visible_nodes(changed))[0]
    for key in ("history", "current_message", "turn_index", "language"):
        assert a[key] == b[key]


def test_first_mention_baseline_suppresses_repeat_without_using_future() -> None:
    node = {"current_message": "I plan to kill myself"}
    seen: set[str] = set()
    assert cradle.native_labels("english-lexicon-first-mention", node, seen) == (
        "confirm_SI_active_ongoing",
    )
    assert cradle.native_labels("english-lexicon-first-mention", node, seen) == ()
    assert cradle.native_labels("no-signal", node, set()) == ()


def metric_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [{"observation_id": "a", "dialogue_id": "d"}]
    gold = [{"observation_id": "a", "dialogue_id": "d", "labels": ["alert_past"]}]
    pred = [{"observation_id": "a", "dialogue_id": "d", "labels": ["alert_past"], "status": "ok"}]
    return nodes, gold, pred


def test_native_metric_changes_with_reference_not_project_execution() -> None:
    nodes, gold, pred = metric_fixture()
    assert cradle.native_metrics(nodes, gold, pred, resamples=20)["micro"]["f1"] == 1
    gold[0]["labels"] = ["alert_ongoing"]
    assert cradle.native_metrics(nodes, gold, pred, resamples=20)["micro"]["f1"] == 0
    visible = [x[0] for x in cradle.visible_nodes(cradle.parse_dialogues(rows()))]
    before = list(cradle.execute_records(visible, "p"))
    gold[0]["labels"] = []
    assert list(cradle.execute_records(visible, "p")) == before


@pytest.mark.parametrize(
    "invalid", ["missing", "unknown_label", "duplicate_labels", "wrong_dialogue"]
)
def test_failed_predictions_remain_in_denominator(invalid: str) -> None:
    nodes, gold, pred = metric_fixture()
    if invalid == "missing":
        pred = []
    elif invalid == "unknown_label":
        pred[0]["labels"] = ["fake"]
    elif invalid == "duplicate_labels":
        pred[0]["labels"] = ["alert_past", "alert_past"]
    else:
        pred[0]["dialogue_id"] = "wrong"
    result = cradle.native_metrics(nodes, gold, pred, resamples=20)
    assert result["retained_nodes"] == 1
    assert result["contract_failures_including_missing"] == 1
    assert result["micro"]["fn"] == 1
    assert result["micro"]["f1"] == 0


def test_empty_denominator_is_not_perfect_accuracy() -> None:
    nodes, gold, pred = metric_fixture()
    gold[0]["labels"] = pred[0]["labels"] = []
    result = cradle.native_metrics(nodes, gold, pred, resamples=20)
    assert result["micro"]["f1"] is None
    assert result["micro_f1_dialogue_cluster_ci95"] is None


def test_metric_rejects_missing_references_and_duplicate_predictions() -> None:
    nodes, gold, pred = metric_fixture()
    with pytest.raises(ValueError):
        cradle.native_metrics(nodes, [], pred)
    with pytest.raises(ValueError):
        cradle.native_metrics(nodes, gold, pred * 2)


def test_snapshot_tamper_is_detected(snapshot: Path) -> None:
    cradle.verify_snapshot(snapshot)
    (snapshot / "README.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        cradle.verify_snapshot(snapshot)


def test_source_audit_preserves_listener_anomaly_and_exact_overlap(snapshot: Path) -> None:
    audit = cradle.audit_snapshot(snapshot)
    assert audit["splits"]["test"]["listener_labeled_rows"][0]["labels"] == ["alert_past"]
    assert audit["splits"]["test"]["label_instances"] == 2
    assert audit["cross_split_exact_dialogue_overlap"]["train__test"] == 2


def test_plan_run_report_end_to_end_offline_and_no_gold(
    snapshot: Path, tmp_path: Path, monkeypatch
) -> None:
    def deny_network(*args, **kwargs):
        raise AssertionError("offline evaluation must not create a network client")

    monkeypatch.setattr(cradle.httpx, "Client", deny_network)
    packet, run = tmp_path / "packet", tmp_path / "run"
    plan = cradle.freeze_external_plan(snapshot, packet)
    assert plan["user_nodes"] == 3 and plan["dialogues"] == 2
    assert plan["expert_action_norms"] == "UNAVAILABLE"
    result = cradle.run_external(packet, run)
    assert result["stage_replay"] == {"status": "PASS", "verified_records": 21}
    assert result["native_crisis_events"]["english-lexicon-first-mention"]["micro"]["f1"] == 1
    assert result["native_crisis_events"]["no-signal"]["micro"]["f1"] == 0
    assert result["expert_action_gold"] == "NOT_CREATED"
    assert result["model_calls"] == result["cost_cny"] == 0
    assert not (packet / "private" / "annotations.sqlite3").exists()
    assert not (packet / "private" / "simulation-truth.json").exists()
    for entry in result["project_action_diagnostics"].values():
        assert entry["expert_action_norms"].startswith("UNAVAILABLE")
    with pytest.raises(FileExistsError):
        cradle.run_external(packet, run)
    with pytest.raises(FileExistsError):
        cradle.freeze_external_plan(snapshot, packet)


@pytest.mark.parametrize("target", ["inputs.jsonl", "private/references.jsonl", "runtime.json"])
def test_frozen_artifact_tampering_rejected(snapshot: Path, tmp_path: Path, target: str) -> None:
    packet = tmp_path / "packet"
    cradle.freeze_external_plan(snapshot, packet)
    (packet / target).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cradle.run_external(packet, tmp_path / "run")


def test_rehashed_fake_predictions_are_rejected(snapshot: Path, tmp_path: Path) -> None:
    packet, run = tmp_path / "packet", tmp_path / "run"
    cradle.freeze_external_plan(snapshot, packet)
    cradle.run_external(packet, run)
    path = run / "predictions.jsonl"
    predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    predictions[0]["labels"] = ["alert_ongoing"]
    path.write_text("\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reproducible"):
        cradle.report_external(packet, run, tmp_path / "fake-report.json")


def test_source_drift_requires_new_plan(snapshot: Path, tmp_path: Path, monkeypatch) -> None:
    packet = tmp_path / "packet"
    cradle.freeze_external_plan(snapshot, packet)
    monkeypatch.setattr(cradle, "source_fingerprint", lambda: "drift")
    with pytest.raises(ValueError, match="drift"):
        cradle.run_external(packet, tmp_path / "run")


def test_truncated_run_is_not_reported_complete(snapshot: Path, tmp_path: Path) -> None:
    packet, run = tmp_path / "packet", tmp_path / "run"
    cradle.freeze_external_plan(snapshot, packet)
    cradle.run_external(packet, run)
    path = run / "predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        cradle.report_external(packet, run, tmp_path / "incomplete-report.json")
    assert not (tmp_path / "incomplete-report.json").exists()


def test_listener_label_never_becomes_user_reference() -> None:
    parsed = cradle.parse_dialogues(rows())
    nodes = list(cradle.visible_nodes(parsed))
    assert sum(len(n[1]["labels"]) for n in nodes) == 1
    assert sum(len(r["labels"]) for d in parsed for r in d) == 2
    assert all("alert_past" not in n[1]["labels"] for n in nodes)


def test_listener_opening_is_preserved_without_dropping_dialogue() -> None:
    source = [
        {"turn_id": 0, "text": "Listener: How are things?", "labels": ""},
        {"turn_id": 1, "text": "User: It has been difficult", "labels": "alert_ongoing"},
    ]
    nodes = list(cradle.visible_nodes(cradle.parse_dialogues(source)))
    assert len(nodes) == 1
    assert nodes[0][0]["history"][0]["role"] == "Listener"
    assert nodes[0][0]["history"][0]["text"] == "How are things?"
    assert nodes[0][0]["turn_index"] == 1
    assert nodes[0][1]["labels"] == ["alert_ongoing"]
    assert cradle.project_observation(nodes[0][0]).visible_context.conversation_history == ()


def test_cli_offline_end_to_end(snapshot: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    packet, run = tmp_path / "cli-packet", tmp_path / "cli-run"
    for args in (
        ["cradle-plan", "--snapshot", str(snapshot), "--output", str(packet)],
        ["cradle-run", "--packet", str(packet), "--output", str(run)],
        [
            "cradle-report",
            "--packet",
            str(packet),
            "--run",
            str(run),
            "--output",
            str(tmp_path / "r.json"),
        ],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output


def test_download_is_pinned_unauthenticated_and_checks_lfs(tmp_path: Path, monkeypatch) -> None:
    payload = b"fixture parquet bytes"
    urls: list[str] = []
    client_class = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        urls.append(str(request.url))
        if "/revision/" in request.url.path:
            return httpx.Response(200, json={"sha": cradle.DEFAULT_REVISION, "cardData": {}})
        if "/tree/" in request.url.path:
            return httpx.Response(
                200,
                json=[
                    {
                        "path": f"data/{s}-00000-of-00001.parquet",
                        "lfs": {"oid": hashlib.sha256(payload).hexdigest()},
                    }
                    for s in cradle.SPLITS
                ],
            )
        return httpx.Response(200, content=payload)

    monkeypatch.setattr(
        cradle.httpx,
        "Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = cradle.download_snapshot(tmp_path / "download")
    assert result["license"] == "UNSPECIFIED"
    assert all(cradle.DEFAULT_REVISION in url for url in urls)
    cradle.verify_snapshot(tmp_path / "download")
    with pytest.raises(ValueError, match="immutable"):
        cradle.download_snapshot(tmp_path / "bad", revision="main")
