"""Offline engineering fixtures only, never actual expert or model judgments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpx
import openpyxl
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ecocompanion.research import chinese_sources as sources
from ecocompanion.research import chinese_study as study
from ecocompanion.research.annotation import AnnotationStore
from ecocompanion.research.api import annotation_app
from ecocompanion.research.cli import app
from ecocompanion.research.contracts import Judgment
from ecocompanion.research.workflows import run_local


def workbook(path: Path, columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(columns)
    for row in rows:
        sheet.append(row)
    book.save(path)
    book.close()


def snapshot_at(root: Path, source: str) -> Path:
    root.mkdir()
    (root / "README.md").write_text("Engineering fixture only.", encoding="utf-8")
    if source == "psycrisis":
        (root / "LICENSE").write_text("Fixture, not a license grant.", encoding="utf-8")
        workbook(
            root / sources.PSY_FILE,
            sources.PSY_COLUMNS,
            [(str(i % 3), f"opening user {i}") for i in range(4)],
        )
        workbook(
            root / sources.RATING_FILE,
            sources.RATING_COLUMNS,
            [
                (f"opening user {i}", f"source counselor reply {i}/{j}", *([j] * 5))
                for i in range(4)
                for j in (0, 1)
            ],
        )
        paths = ["README.md", "LICENSE", sources.PSY_FILE, sources.RATING_FILE]
    else:
        cases = [
            {
                "instruction": f"last user {i}",
                "input": "",
                "output": f"source counselor final {i}",
                "history": [
                    [f"opening user {i}", f"source counselor opening {i}"],
                    *[
                        [f"user middle {i}/{j}", f"source counselor middle {i}/{j}"]
                        for j in range(i % 7)
                    ],
                ],
            }
            for i in range(24)
        ]
        (root / sources.CP_FILE).write_text(json.dumps(cases), encoding="utf-8")
        paths = ["README.md", sources.CP_FILE]
    revision = sources.PSY_REVISION if source == "psycrisis" else sources.CP_REVISION
    (root / "upstream.json").write_text(json.dumps({"sha": revision}), encoding="utf-8")
    (root / "tree.json").write_text(
        json.dumps(
            {
                "tree": [
                    {
                        "path": name,
                        "sha": sources.blob_sha((root / name).read_bytes()),
                        "size": (root / name).stat().st_size,
                    }
                    for name in paths
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "source": source,
        "revision": revision,
        "download_verification": "PINNED_UPSTREAM_OBJECT_AND_LOCAL_SHA256",
        "license": "ENGINEERING_FIXTURE_ONLY",
        "license_conflict": None,
        "files": {
            name: {
                "sha256": sources.sha256((root / name).read_bytes()),
                "bytes": (root / name).stat().st_size,
            }
            for name in paths
        },
        "upstream_sha256": sources.sha256((root / "upstream.json").read_bytes()),
        "tree_sha256": sources.sha256((root / "tree.json").read_bytes()),
    }
    (root / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture
def psy(tmp_path: Path) -> Path:
    return snapshot_at(tmp_path / "psy", "psycrisis")


@pytest.fixture
def cp(tmp_path: Path) -> Path:
    return snapshot_at(tmp_path / "cp", "cpsycoun")


def test_audit_counts_input_groups_not_response_rows(psy: Path, cp: Path) -> None:
    audit = sources.audit_chinese(psy)
    assert audit["prompts"] == audit["unique_rated_inputs"] == 4
    assert audit["upstream_rated_responses"] == 8
    assert audit["prompt_rating_input_overlap"] == 4
    assert audit["expert_action_gold"] == "UNAVAILABLE"
    assert audit["negative_noncrisis_controls"] == "NOT_PROVIDED"
    other = sources.audit_chinese(cp)
    assert other["dialogues"] == 24
    assert other["age_group"] == "UNSPECIFIED_NOT_ASSUMED_ADOLESCENT"
    assert other["topic_or_school_labels"].startswith("NOT_PRESENT")


def test_rewriting_local_hashes_does_not_bypass_saved_upstream_objects(cp: Path) -> None:
    path = cp / "README.md"
    path.write_text("changed")
    manifest_path = cp / "snapshot.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["README.md"] = {
        "sha256": sources.sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="upstream object"):
        sources.verify_chinese(cp)


@pytest.mark.parametrize("target", ["README.md", "tree.json", "upstream.json", sources.PSY_FILE])
def test_snapshot_changes_are_detected(psy: Path, target: str) -> None:
    (psy / target).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        sources.verify_chinese(psy)


@pytest.mark.parametrize("bad", ["formula", "header", "sheet", "external", "macro"])
def test_workbook_rejects_nondata_content(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "bad.xlsx"
    workbook(path, sources.PSY_COLUMNS, [("0", "text")])
    if bad in {"external", "macro"}:
        with ZipFile(path, "a") as archive:
            name = (
                "xl/externalLinks/externalLink1.xml" if bad == "external" else "xl/vbaProject.bin"
            )
            archive.writestr(name, b"fixture")
    else:
        book = openpyxl.load_workbook(path)
        if bad == "formula":
            book.active["B2"] = "=1+1"
        elif bad == "header":
            book.active["A1"] = "diagnosis"
        else:
            book.create_sheet("unexpected")
        book.save(path)
        book.close()
    with pytest.raises(ValueError):
        sources.workbook_rows(path, sources.PSY_COLUMNS)


@pytest.mark.parametrize("bad", ["input", "history", "empty", "extra"])
def test_cp_schema_changes_need_review(cp: Path, bad: str) -> None:
    records = json.loads((cp / sources.CP_FILE).read_text())
    if bad == "input":
        records[0]["input"] = "additional context cannot be silently lost"
    elif bad == "history":
        records[0]["history"][0] = ["missing assistant"]
    elif bad == "empty":
        records[0]["instruction"] = ""
    else:
        records[0]["source_topic"] = "do not silently forward"
    (cp / sources.CP_FILE).write_text(json.dumps(records))
    with pytest.raises(ValueError):
        sources.load_cp(cp)


def test_selection_is_deterministic_and_groups_duplicate_openings(cp: Path) -> None:
    cases = sources.load_cp(cp)
    duplicate = {**cases[0], "row": 1000}
    selected = study.select_cp([*cases, duplicate], 20)
    assert selected == study.select_cp(list(reversed([*cases, duplicate])), 20)
    assert len({sources.normalized(c["pairs"][0][0]) for c in selected}) == 20
    with pytest.raises(ValueError):
        study.select_cp(cases, 25)


def test_prefix_never_contains_future_source_replies_or_source_labels(
    cp: Path, tmp_path: Path
) -> None:
    packet = tmp_path / "packet"
    plan = study.freeze_chinese(cp, packet)
    assert plan["cases"] == 20
    inputs = sources.read_jsonl(packet / "inputs.jsonl")
    first = inputs[0]
    serialized = json.dumps(inputs)
    for hidden in ("source counselor", "category", "source_row", "source_pairs", "CPsyCounD"):
        assert hidden not in serialized
    assert first["visible_context"]["conversation_history"] == []
    assert "last user" not in json.dumps(first)
    assert len(first["episode_id"]) == len(first["current_observation"]["observation_id"]) == 32
    assert first["visible_profile"]["age_group"] == "unspecified"
    later = inputs[1]
    assert (
        later["visible_context"]["conversation_history"][0]["content"]
        == first["current_observation"]["message"]
    )
    assert sources.read_jsonl(packet / "private/source-pairs.jsonl")


def test_future_rewrite_leaves_earlier_visible_content_unchanged() -> None:
    before = study.observations_for("1" * 32, ["current", "future original"])[0]
    after = study.observations_for("1" * 32, ["current", "future changed"])[0]
    a, b = before.model_dump(), after.model_dump()
    a["current_observation"].pop("observation_id")
    b["current_observation"].pop("observation_id")
    assert a == b


def test_end_to_end_is_offline_replayable_without_action_gold(
    psy: Path, tmp_path: Path, monkeypatch
) -> None:
    def deny(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no network in evaluation")

    monkeypatch.setattr(sources.httpx, "Client", deny)
    packet, run = tmp_path / "packet", tmp_path / "run"
    plan = study.freeze_chinese(psy, packet)
    result = study.run_chinese(packet, run)
    assert plan["cases"] == plan["nodes"] == 4
    assert result["stage_replay"] == {"status": "PASS", "verified_records": 20}
    assert result["model_calls"] == result["cost_cny"] == 0
    assert result["clinical_or_action_correctness"] == "NOT_ESTABLISHED"
    assert all(p["expert_action_norms"] == "UNAVAILABLE" for p in result["policies"].values())
    assert not list(packet.rglob("*truth*"))
    review = study.chinese_reply_packet(packet, run, tmp_path / "reply-review")
    assert review["tasks"] == 20
    tasks = json.loads((tmp_path / "reply-review/reply-tasks.json").read_text(encoding="utf-8"))[
        "tasks"
    ]
    assert all(not ({"policy", "category", "source_row"} & t.keys()) for t in tasks)
    with pytest.raises(FileExistsError):
        study.run_chinese(packet, run)
    with pytest.raises(FileExistsError):
        study.freeze_chinese(psy, packet)
    with pytest.raises(ValueError, match="synthetic-only"):
        study.annotation_packet(packet, tmp_path / "invalid", Path(__file__))


@pytest.mark.parametrize(
    "target",
    ["inputs.jsonl", "private/metadata.jsonl", "private/source-pairs.jsonl", "runtime.json"],
)
def test_plan_detects_tampering(psy: Path, tmp_path: Path, target: str) -> None:
    packet = tmp_path / "packet"
    study.freeze_chinese(psy, packet)
    (packet / target).write_text("{}\n")
    with pytest.raises(ValueError):
        study.run_chinese(packet, tmp_path / "run")


def test_source_drift_requires_new_plan(psy: Path, tmp_path: Path, monkeypatch) -> None:
    packet = tmp_path / "packet"
    study.freeze_chinese(psy, packet)
    monkeypatch.setattr(study, "source_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="drift"):
        study.validate_chinese_plan(packet)


@pytest.mark.parametrize("change", ["missing", "tampered"])
def test_incomplete_outputs_cannot_be_reported_complete(
    psy: Path, tmp_path: Path, change: str
) -> None:
    packet, run = tmp_path / "packet", tmp_path / "run"
    study.freeze_chinese(psy, packet)
    study.run_chinese(packet, run)
    path = run / "predictions.jsonl"
    records = sources.read_jsonl(path)
    if change == "missing":
        records.pop()
    else:
        records[0]["observation_digest"] = "changed"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="missing/tampered"):
        study.chinese_report(packet, run, tmp_path / "false-report.json")


def test_annotation_store_has_no_experts_or_gold_and_sequential_api(
    cp: Path, tmp_path: Path
) -> None:
    packet, output = tmp_path / "packet", tmp_path / "annotation"
    study.freeze_chinese(cp, packet)
    manual = tmp_path / "manual.md"
    manual.write_text("ENGINEERING FIXTURE ONLY. Not expert advice.")
    result = study.annotation_packet(packet, output, manual)
    assert result["cases"] == 20
    assert result["gold_status"] == "NOT_FROZEN"
    store = AnnotationStore(output / "private/annotations.sqlite3")
    assert all(c.provenance.review_status == "draft" for c in store.cases())
    assert all(c.provenance.person_group == "UNKNOWN_ALL_CPSYCOUN" for c in store.cases())
    with store.connection() as connection:
        assert connection.execute("SELECT count(*) FROM judgments").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM attestations").fetchone()[0] == 0
    with pytest.raises(ValueError):
        store.freeze(output / "private/should-not-exist.json")
    # These credentials and judgments exist only inside the temporary engineering fixture.
    tokens = json.loads((output / "private/access.json").read_text())
    client = TestClient(annotation_app(store))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/task").status_code == 401
    assert client.get("/private/access.json").status_code == 404
    headers = {"Authorization": f"Bearer {tokens['expert-a']}"}
    task = client.get("/task", headers=headers).json()
    assert task["step_id"] == 0 and "source counselor" not in json.dumps(task)
    assert "independent_labels" not in task
    judgment = Judgment(
        risk_level="L0",
        confidence=0.1,
        acceptable_actions=["SUPPORTIVE_LISTENING"],
        preferred_action="SUPPORTIVE_LISTENING",
        direct_safety_check_required=False,
        reason="Engineering fixture, not an expert judgment.",
    )
    body = {"task_id": task["task_id"], "judgment": judgment.model_dump(mode="json")}
    assert client.post("/judgments", headers=headers, json=body).json() == {"status": "locked"}
    assert client.post("/judgments", headers=headers, json=body).status_code == 409
    assert client.get("/task", headers=headers).json()["step_id"] == 1
    other = client.get("/task", headers={"Authorization": f"Bearer {tokens['expert-b']}"}).json()
    assert other["step_id"] == 0 and "independent_labels" not in other
    assert store.audit()["status"] == "verified"
    compatible = run_local(output, tmp_path / "v2-compatible-run")
    assert len(compatible["policies"]) == 4
    assert all(
        p["expert_action_norms"]["status"] == "UNAVAILABLE_NO_APPLICABLE_GOLD"
        for p in compatible["policies"].values()
    )


def test_calibration_groups_keep_both_responses_together_and_hide_ratings(
    psy: Path, tmp_path: Path
) -> None:
    output = tmp_path / "calibration"
    result = sources.calibration_packet(psy, output, holdout_groups=1)
    assert result["tasks"] == 8 and result["input_groups"] == 4
    assert result["calibration_groups"] == 3 and result["public_holdout_groups"] == 1
    tasks = sources.read_jsonl(output / "tasks.jsonl")
    assert all("scores" not in t and "partition" not in t and "source_row" not in t for t in tasks)
    refs = sources.read_jsonl(output / "private/references.jsonl")
    for group in {r["input_group"] for r in refs}:
        assert len({r["partition"] for r in refs if r["input_group"] == group}) == 1
        assert sum(r["input_group"] == group for r in refs) == 2
    with pytest.raises(FileExistsError):
        sources.calibration_packet(psy, output, holdout_groups=1)


def fixture_predictions(packet: Path) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": r["candidate_id"],
            "status": "ok",
            "scores": r["scores"],
            "judge_id": "ENGINEERING_FIXTURE_NOT_REAL_JUDGE",
            "verification_reference": "pytest-only-fixture",
        }
        for r in sources.read_jsonl(packet / "private/references.jsonl")
    ]


def test_reference_changes_affect_agreement_not_project_diagnostics(
    psy: Path, tmp_path: Path
) -> None:
    calibration, packet = tmp_path / "calibration", tmp_path / "packet"
    sources.calibration_packet(psy, calibration, holdout_groups=1)
    study.freeze_chinese(psy, packet)
    before = list(study.chinese_records(packet))
    predictions = tmp_path / "predictions.jsonl"
    sources.write_jsonl(predictions, fixture_predictions(calibration))
    baseline = study.score_calibration(calibration, predictions, tmp_path / "baseline.json")
    assert (
        baseline["dimensions"]["empathy"]["calibration"]["accuracy_failures_counted_in_denominator"]
        == 1
    )
    refs_path = calibration / "private/references.jsonl"
    refs = sources.read_jsonl(refs_path)
    for ref in refs:
        ref["scores"]["empathy"] = 1 - ref["scores"]["empathy"]
    refs_path.write_text("".join(json.dumps(r) + "\n" for r in refs))
    manifest_path = calibration / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["references_sha256"] = sources.sha256(refs_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    changed = study.score_calibration(calibration, predictions, tmp_path / "changed.json")
    assert (
        changed["dimensions"]["empathy"]["calibration"]["accuracy_failures_counted_in_denominator"]
        == 0
    )
    assert changed["dimensions"]["risk_inquiry"] == baseline["dimensions"]["risk_inquiry"]
    assert list(study.chinese_records(packet)) == before


@pytest.mark.parametrize("bad", ["missing", "boolean", "no_lineage", "format_error"])
def test_invalid_or_missing_judgments_remain_in_denominator(
    psy: Path, tmp_path: Path, bad: str
) -> None:
    packet = tmp_path / "calibration"
    sources.calibration_packet(psy, packet, holdout_groups=1)
    predictions = fixture_predictions(packet)
    if bad == "missing":
        predictions = []
    else:
        for prediction in predictions:
            if bad == "boolean":
                prediction["scores"]["empathy"] = True
            elif bad == "no_lineage":
                prediction.pop("verification_reference")
            else:
                prediction["status"] = "format_error"
    path = tmp_path / "predictions.jsonl"
    sources.write_jsonl(path, predictions)
    report = study.score_calibration(packet, path, tmp_path / "report.json")
    for partition in ("calibration", "public_holdout"):
        result = report["dimensions"]["empathy"][partition]
        assert result["valid"] == 0 and result["failed_or_missing"] == result["expected"]
        assert result["accuracy_failures_counted_in_denominator"] == 0
        assert result["kappa_valid_pairs_only_NOT_FULL_COVERAGE"] is None


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "tampered_reference"])
def test_calibration_rejects_ambiguous_ids_and_tampering(
    psy: Path, tmp_path: Path, bad: str
) -> None:
    packet = tmp_path / "calibration"
    sources.calibration_packet(psy, packet, holdout_groups=1)
    predictions = fixture_predictions(packet)
    if bad == "duplicate":
        predictions.append(predictions[0])
    elif bad == "unknown":
        predictions[0]["candidate_id"] = "unknown"
    else:
        (packet / "private/references.jsonl").write_text("{}\n")
    path = tmp_path / "predictions.jsonl"
    sources.write_jsonl(path, predictions)
    with pytest.raises(ValueError):
        study.score_calibration(packet, path, tmp_path / "report.json")


def test_cli_plan_run_and_no_credential_output(psy: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    packet, run = tmp_path / "packet", tmp_path / "run"
    planned = runner.invoke(app, ["zh", "plan", "--snapshot", str(psy), "--output", str(packet)])
    assert planned.exit_code == 0, planned.output
    executed = runner.invoke(app, ["zh", "run", "--packet", str(packet), "--output", str(run)])
    assert executed.exit_code == 0, executed.output
    assert "model_calls=0" in executed.output
    assert "access.json" not in executed.output


@pytest.mark.parametrize("api_status", [403, 429, 404, 200])
def test_download_rate_limit_limitation_is_explicit_and_other_errors_fail(
    tmp_path: Path, monkeypatch, api_status: int
) -> None:
    real_client = httpx.Client
    payload = b"ENGINEERING FIXTURE"
    paths = ["README.md", "LICENSE", sources.PSY_FILE, sources.RATING_FILE]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            if api_status != 200:
                return httpx.Response(api_status, json={"message": "fixture"})
            if "/commits/" in request.url.path:
                return httpx.Response(200, json={"sha": sources.PSY_REVISION})
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {
                            "path": path,
                            "type": "blob",
                            "sha": sources.blob_sha(payload),
                            "size": len(payload),
                        }
                        for path in paths
                    ]
                },
            )
        assert sources.PSY_REVISION in str(request.url)
        return httpx.Response(200, content=payload)

    monkeypatch.setattr(
        sources.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    output = tmp_path / "download"
    if api_status == 404:
        with pytest.raises(httpx.HTTPStatusError):
            sources.download_chinese("psycrisis", output)
        assert not output.exists()
    else:
        result = sources.download_chinese("psycrisis", output)
        if api_status == 200:
            assert result["download_verification"] == "PINNED_UPSTREAM_OBJECT_AND_LOCAL_SHA256"
        else:
            assert result["download_verification"].endswith("NOT_UPSTREAM_TREE_VERIFIED")
        assert sources.verify_chinese(output) == result
