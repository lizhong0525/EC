"""Synthetic engineering fixtures only; no real clinicians or model calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research.annotation import digest
from ecocompanion.research.chinese_sources import DIMENSIONS, blob_sha, sha256, write_jsonl
from scripts import external_reference as workflow
from scripts import local_reference_judge as judge
from scripts import reference_sources as sources


def synthetic_rows() -> list[dict[str, str]]:
    rows = []
    for conversation, score in (("000001", "Best Practice"), ("000002", "Not Relevant")):
        for i in range(3):
            rows.append(
                {
                    "conversation_id": conversation,
                    "clinician_id": f"ENGINEERING_FIXTURE_{i}",
                    "user_profile": f"FICTIONAL_PROFILE_{conversation}",
                    **dict.fromkeys(sources.VERA_DIMENSIONS, score),
                }
            )
    return rows


def test_nominal_alpha_known_examples_and_undefined_constant() -> None:
    assert workflow.nominal_alpha([["a", "a", "a"], ["b", "b", "b"]]) == 1
    # Six ratings, category totals (3,3): De=0.6, Do=2/3.
    assert workflow.nominal_alpha([["a", "a", "b"], ["b", "b", "a"]]) == pytest.approx(-1 / 9)
    assert workflow.nominal_alpha([["a", "a"]]) is None
    assert workflow.nominal_alpha([]) is None


def test_clinician_majority_retains_raw_disagreements_and_does_not_invent_tiebreak() -> None:
    rows = synthetic_rows()
    dimension = sources.VERA_DIMENSIONS[0]
    rows[0][dimension] = "High Potential for Harm"
    rows[1][dimension] = "Suboptimal but Low Potential for Harm"
    original = json.dumps(rows, sort_keys=True)
    refs, audit = workflow.vera_reference(rows)
    assert json.dumps(rows, sort_keys=True) == original
    assert refs[0]["scores"][dimension] is None
    assert len(audit["unresolved_dimensions"]) == 1
    assert audit["resolved_majority_dimensions"] == 9
    assert audit["new_local_judge_vs_clinician_agreement"] == "NOT_RUN"
    assert "NOT_PROJECT_ACTION_GOLD" in refs[0]["origin"]
    assert audit["clinician_reliability_by_conversation"]["clusters"] == 2


def test_missing_clinician_is_not_silently_removed() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        workflow.vera_reference(synthetic_rows()[:-1])


def test_bootstrap_undefined_alpha_not_perfect() -> None:
    value = workflow.clustered_alpha({"a": [["same", "same", "same"]]}, resamples=20)
    assert value["nominal_alpha"] is None
    assert value["ci95"] is None
    assert value["defined_resamples"] == 0


def test_extract_literal_prompt_without_executing_upstream() -> None:
    text = b"""
raise RuntimeError("THIS FILE MUST NEVER EXECUTE")
data = {"role": "system", "content": "Literal source rubric"}
"""
    assert sources.psy_system_prompt(text) == "Literal source rubric"
    with pytest.raises(ValueError):
        sources.psy_system_prompt(b'{"role": "system", "content": make_prompt()}')


def test_source_identity_never_accepts_moving_ref() -> None:
    assert len(sources.source_identity("vera")[1]) == 40
    with pytest.raises(ValueError):
        sources.source_identity("main")


def test_only_explicit_reviewed_source_id_aliases_are_applied() -> None:
    import csv
    import io

    row = synthetic_rows()[0]
    row["conversation_id"] = "#402e12"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    parsed = sources.clinician_rows(stream.getvalue().encode())
    assert parsed[0]["conversation_id"] == "402e12"
    assert parsed[0]["source_conversation_id"] == "#402e12"
    with pytest.raises(ValueError):
        sources.clinician_rows(stream.getvalue().replace("#402e12", "#000001").encode())


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    files = {}
    entries = []
    for path in sources.PSY_PATHS:
        content = ("ENGINEERING_FIXTURE_" + path).encode()
        git = blob_sha(content)
        relative = f"objects/{git}.bin"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        files[path] = {
            "local": relative, "git_blob": git, "sha256": sha256(content), "bytes": len(content)
        }
        entries.append({"path": path, "type": "blob", "sha": git, "size": len(content)})
    write_json_atomic(
        root / "commit.json",
        {"sha": sources.PSY_REVISION, "commit": {"tree": {"sha": "fixturetree"}}},
        overwrite=False,
    )
    write_json_atomic(
        root / "tree.json",
        {"sha": "fixturetree", "tree": entries, "truncated": False},
        overwrite=False,
    )
    write_json_atomic(
        root / "snapshot.json",
        {
            "version": sources.VERSION,
            "source": "psycrisis",
            "repository": sources.PSY_REPO,
            "revision": sources.PSY_REVISION,
            "files": files,
            "tree_sha256": sha256((root / "tree.json").read_bytes()),
            "commit_sha256": sha256((root / "commit.json").read_bytes()),
        },
        overwrite=False,
    )
    return root


def test_snapshot_checks_git_and_sha256_and_never_executes(snapshot: Path) -> None:
    result = sources.verify(snapshot)
    assert result["revision"] == sources.PSY_REVISION
    item = result["files"]["README.md"]
    (snapshot / item["local"]).write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="object changed"):
        sources.verify(snapshot)


def test_snapshot_rejects_path_traversal_even_if_inventory_was_rehashed(snapshot: Path) -> None:
    path = snapshot / "snapshot.json"
    data = read_json_object(path)
    data["files"]["README.md"]["local"] = "../outside"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="location"):
        sources.verify(snapshot)


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    root = tmp_path / "packet"
    tasks, refs = [], []
    for i in range(8):
        group = str(i // 2)
        tasks.append(
            {
                "candidate_id": f"fixture-{i}",
                "input_group": group,
                "user": "FICTIONAL_INPUT_" + group,
                "reply": "FICTIONAL_REPLY_" + str(i),
            }
        )
        refs.append(
            {
                "candidate_id": f"fixture-{i}",
                "input_group": group,
                "partition": "calibration" if i < 4 else "public_holdout",
                "scores": dict.fromkeys(DIMENSIONS, i % 2),
            }
        )
    write_jsonl(root / "tasks.jsonl", tasks)
    write_jsonl(root / "private/references.jsonl", refs)
    write_jsonl(root / "project-tasks.jsonl", tasks[:1])
    workflow.seal(
        root, "psycrisis", ["tasks.jsonl", "private/references.jsonl", "project-tasks.jsonl"]
    )
    return root


def test_complete_model_request_has_no_semantic_metadata_or_reference_fields() -> None:
    task = {
        "candidate_id": "MUST_NOT_LEAK_ID",
        "input_group": "MUST_NOT_LEAK_GROUP",
        "policy": "MUST_NOT_LEAK_POLICY",
        "partition": "MUST_NOT_LEAK_SPLIT",
        "scores": {"risk_inquiry": 1},
        "future": "MUST_NOT_LEAK_FUTURE",
        "hidden_state": "MUST_NOT_LEAK_STATE",
        "source_path": "MUST_NOT_LEAK_FILENAME",
        "user": "FICTIONAL USER",
        "reply": "FICTIONAL REPLY",
    }
    payload = workflow.request_payload(task)
    encoded = json.dumps(payload)
    assert "MUST_NOT_LEAK" not in encoded
    assert json.loads(payload["messages"][1]["content"]) == {
        "user": task["user"], "assistant": task["reply"]
    }
    assert "untrusted data" in payload["messages"][0]["content"]


def test_phase_projection_preserves_groups_and_protects_holdout(packet: Path) -> None:
    calibration = judge.phase_tasks(packet, "calibration")
    heldout = judge.phase_tasks(packet, "public_holdout")
    assert len(calibration) == len(heldout) == 4
    assert not ({t["input_group"] for t in calibration} & {t["input_group"] for t in heldout})
    with pytest.raises(ValueError):
        judge.phase_tasks(packet, "arbitrary")


def test_group_integrity_requires_source_content_and_labels(packet: Path) -> None:
    from ecocompanion.research.chinese_sources import read_jsonl

    tasks = read_jsonl(packet / "tasks.jsonl")
    refs = read_jsonl(packet / "private/references.jsonl")
    indexed = {r["candidate_id"]: r for r in refs}
    ratings = [
        {"user": t["user"], "reply": t["reply"], "scores": indexed[t["candidate_id"]]["scores"]}
        for t in tasks
    ]
    workflow.validate_psy_material(tasks, refs, ratings)
    refs[0]["scores"]["empathy"] = 1
    # Deep-copy before mutation is required because fixture data above shares score objects.
    ratings[0]["scores"] = dict.fromkeys(DIMENSIONS, 0)
    with pytest.raises(ValueError, match="source workbook"):
        workflow.validate_psy_material(tasks, refs, ratings)


def test_group_leakage_rejected(packet: Path) -> None:
    from ecocompanion.research.chinese_sources import read_jsonl

    tasks = read_jsonl(packet / "tasks.jsonl")
    refs = read_jsonl(packet / "private/references.jsonl")
    refs[1]["partition"] = "public_holdout"
    with pytest.raises(ValueError, match="leakage"):
        workflow.validate_psy_material(tasks, refs, [])


def test_binary_reference_changes_only_matching_dimension_and_missing_retained() -> None:
    refs = [
        {"candidate_id": "x", "input_group": "one", "scores": dict.fromkeys(DIMENSIONS, 1)},
        {"candidate_id": "y", "input_group": "two", "scores": dict.fromkeys(DIMENSIONS, 0)},
    ]
    predictions = {"x": {"scores": dict.fromkeys(DIMENSIONS, 1)}}
    first = workflow.binary_metrics(refs, predictions)
    assert first["empathy"]["missing_or_invalid"] == 1
    assert first["empathy"]["accuracy_failures_in_denominator"] == 0.5
    refs[0]["scores"]["empathy"] = 0
    second = workflow.binary_metrics(refs, predictions)
    assert second["empathy"]["accuracy_failures_in_denominator"] == 0
    assert first["regulation"] == second["regulation"]
    assert not workflow.phase_gate(second)["passed"]


def test_low_support_perfect_f1_never_qualifies() -> None:
    refs = [
        {"candidate_id": "x", "input_group": "one", "scores": dict.fromkeys(DIMENSIONS, 1)}
    ]
    result = workflow.binary_metrics(refs, {"x": {"scores": dict.fromkeys(DIMENSIONS, 1)}})
    assert result["risk_inquiry"]["positive_f1"] == 1
    assert not workflow.phase_gate(result)["passed"]


def test_packet_drift_requires_new_version(packet: Path) -> None:
    (packet / "project-tasks.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        workflow.validate_packet(packet)


def verdict() -> dict[str, Any]:
    return {
        "scores": dict.fromkeys(DIMENSIONS, 1),
        "reasons": dict.fromkeys(DIMENSIONS, "ENGINEERING FIXTURE ONLY."),
        "human_review_flags": [],
    }


@pytest.mark.parametrize("bad", ["boolean", "expert", "missing", "empty_reason", "bad_flags"])
def test_strict_judge_output_never_imports_experts(bad: str) -> None:
    data = verdict()
    if bad == "boolean":
        data["scores"]["empathy"] = True
    elif bad == "expert":
        data["kind"] = "expert"
    elif bad == "missing":
        del data["scores"]["risk_inquiry"]
    elif bad == "empty_reason":
        data["reasons"]["empathy"] = ""
    else:
        data["human_review_flags"] = [1]
    with pytest.raises(ValueError):
        judge.parse_verdict(json.dumps(data))


def test_never_collects_private_thinking_or_arbitrary_server_metadata() -> None:
    raw = {
        "model": "fixture:tag", "done": True,
        "message": {"content": json.dumps(verdict()), "thinking": "PRIVATE_DO_NOT_STORE"},
        "secret": "MUST_NOT_STORE", "prompt_eval_count": 1, "eval_count": 2,
    }
    response = judge.visible_response(raw)
    assert "PRIVATE_DO_NOT_STORE" not in json.dumps(response)
    assert "MUST_NOT_STORE" not in json.dumps(response)


def test_malformed_local_response_fails_instead_of_silent_repair() -> None:
    with pytest.raises(ValueError):
        judge.visible_response({"message": "NOT_A_MESSAGE"})


def test_probe_unavailable_does_not_fabricate_model_or_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> list[Any]:
        raise httpx.ConnectError("ENGINEERING FIXTURE")

    monkeypatch.setattr(judge, "local_models", unavailable)
    result = judge.probe(tmp_path / "probe.json")
    assert result["status"] == "LOCAL_SERVICE_UNAVAILABLE"
    assert result["inference_calls"] == 0
    assert result["models"] == []


@pytest.fixture
def config(packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        judge, "local_models", lambda: [{"name": "fixture:tag", "digest": "FIXTURE_DIGEST"}]
    )
    path = tmp_path / "judge.json"
    judge.freeze(packet, "fixture:tag", path)
    return path


def test_no_silent_remote_endpoint_or_fallback(packet: Path, config: Path) -> None:
    data = read_json_object(config)
    data["endpoint"] = "https://example.invalid"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="budget changed"):
        judge.validate_config(packet, config)


def test_holdout_and_project_require_prior_real_runs(
    packet: Path, config: Path, tmp_path: Path
) -> None:
    for phase in ("public_holdout", "project"):
        with pytest.raises(ValueError, match="run is required"):
            judge.run_local(packet, config, phase, tmp_path / phase)
        assert not (tmp_path / phase).exists()


def test_missing_outputs_retained_and_cis_are_grouped(
    packet: Path, config: Path, tmp_path: Path
) -> None:
    root = tmp_path / "partial"
    tasks = judge.phase_tasks(packet, "calibration")
    write_json_atomic(
        root / "started.json",
        {
            "config_sha256": sha256(config.read_bytes()),
            "phase": "calibration", "task_set_digest": digest(tasks),
        },
        overwrite=False,
    )
    report = judge.score_run(packet, config, root)
    assert report["expected"] == report["failed_or_missing"] == 4
    assert report["valid"] == 0
    assert report["descriptive_group_bootstrap"]["clusters"] == 2
    assert report["expert_action_metrics"] is None
    assert not report["gate"]["passed"]


def test_changed_model_digest_blocks_before_any_inference(
    packet: Path, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        judge, "local_models", lambda: [{"name": "fixture:tag", "digest": "CHANGED_DIGEST"}]
    )
    output = tmp_path / "should-not-start"
    with pytest.raises(ValueError, match="identity changed"):
        judge.run_local(packet, config, "calibration", output)
    assert not output.exists()


def test_end_to_end_local_mock_never_calls_outside_loopback_or_retries(
    packet: Path, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = []

    def respond(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        payload = json.loads(request.content)
        assert payload["model"] == "fixture:tag"
        assert "FICTIONAL_INPUT" in payload["messages"][1]["content"]
        assert "fixture-0" not in json.dumps(payload)
        return httpx.Response(
            200,
            json={
                "model": "fixture:tag", "done": True,
                "message": {
                    "content": json.dumps(verdict()), "thinking": "PRIVATE_NOT_TO_COLLECT",
                },
                "prompt_eval_count": 10, "eval_count": 20,
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        judge.httpx, "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    output = tmp_path / "mock-run"
    result = judge.run_local(packet, config, "calibration", output)
    assert result["valid"] == result["calls"] == 4
    assert result["new_project_action_gold"] == 0
    assert all(u == "http://127.0.0.1:11434/api/chat" for u in urls)
    assert result == judge.score_run(packet, config, output)
    assert "PRIVATE_NOT_TO_COLLECT" not in (
        output / "records/fixture-0.json"
    ).read_text()
    with pytest.raises(FileExistsError):
        judge.run_local(packet, config, "calibration", output)


def test_bad_local_output_stops_and_keeps_expected_denominator(
    packet: Path, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "WRONG_MODEL", "done": True, "message": {"content": "{}"},
        })

    real_client = httpx.Client
    monkeypatch.setattr(
        judge.httpx, "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    result = judge.run_local(packet, config, "calibration", tmp_path / "failed")
    assert result["calls"] == 1
    assert result["expected"] == result["failed_or_missing"] == 4
    assert result["valid"] == 0
    assert not result["gate"]["passed"]
