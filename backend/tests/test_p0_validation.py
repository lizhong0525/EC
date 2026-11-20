"""P0 engineering fixtures only; no new human ratings or actual inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ecocompanion.experiments.io import read_json_object
from ecocompanion.research.chinese_sources import DIMENSIONS, write_jsonl
from scripts import external_reference as workflow
from scripts import p0_validation as p0


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    root = tmp_path / "packet"
    tasks, refs = [], []
    for phase in ("calibration", "public_holdout"):
        for i in range(40):
            candidate = f"fixture-{phase}-{i}"
            group = f"group-{phase}-{i // 2}"
            tasks.append({
                "candidate_id": candidate, "input_group": group,
                "user": "ENGINEERING_INPUT", "reply": "ENGINEERING_REPLY",
            })
            refs.append({
                "candidate_id": candidate, "input_group": group, "partition": phase,
                "scores": dict.fromkeys(DIMENSIONS, i % 2),
            })
    write_jsonl(root / "tasks.jsonl", tasks)
    write_jsonl(root / "private/references.jsonl", refs)
    workflow.seal(root, "psycrisis", ["tasks.jsonl", "private/references.jsonl"])
    return root


@pytest.fixture
def local_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        p0.judge, "local_models",
        lambda: [{"name": "FIXTURE:exact", "digest": "FIXTURE_DIGEST"}],
    )


def test_support_counts_original_units_not_inference(packet: Path) -> None:
    report = p0.support_audit(packet)
    assert report["all_dimension_support_sufficient"]
    assert report["phases"]["calibration"]["responses"] == 40
    assert report["phases"]["calibration"]["input_groups_NOT_VERIFIED_PERSONS"] == 20
    assert report["phases"]["public_holdout"]["dimensions"]["risk_inquiry"] == {
        "reference_positive": 20, "reference_negative": 20,
        "positive_shortfall_under_existing_gate": 0,
        "negative_shortfall_under_existing_gate": 0,
        "support_sufficient": True, "reasons": [],
        "judge_agreement": "NOT_MEASURED_BY_PREFLIGHT",
    }


def test_existing_support_shortfall_blocks_without_changing_gate(
    packet: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = p0.read_jsonl

    def sparse(path: Path) -> list[dict[str, Any]]:
        rows = original_read(path)
        if path.name == "references.jsonl":
            for row in rows:
                if row["partition"] == "public_holdout":
                    row["scores"]["risk_inquiry"] = 0
        return rows

    monkeypatch.setattr(p0, "read_jsonl", sparse)
    result = p0.support_audit(packet)
    assert result["blockers"] == [
        "public_holdout:risk_inquiry:INSUFFICIENT_POSITIVE_SUPPORT"
    ]
    assert result["phases"]["public_holdout"]["dimensions"]["risk_inquiry"][
        "positive_shortfall_under_existing_gate"
    ] == 20
    assert workflow.GATES["min_positive"] == 20


def test_doctor_never_selects_or_invokes_available_model(
    packet: Path, tmp_path: Path, local_service: None,
) -> None:
    result = p0.execute(packet, tmp_path / "doctor")
    saved = read_json_object(tmp_path / "doctor/preflight.json")
    assert saved["selected_model"] is None
    assert not saved["calibration_execution_ready"]
    assert result["judge_inference_calls"] == 0
    assert result["historical_packet_unchanged"]


def test_missing_selected_model_blocks_run_without_freezing(
    packet: Path, tmp_path: Path, local_service: None,
) -> None:
    result = p0.execute(packet, tmp_path / "run", model="NOT_INSTALLED", run=True)
    assert result["status"] == "BLOCKED_LOCAL_RUNTIME_OR_MODEL"
    assert result["judge_inference_calls"] == 0
    assert not (tmp_path / "run/judge-config.json").exists()


def test_no_implicit_execution_or_overwrite(
    packet: Path, tmp_path: Path, local_service: None,
) -> None:
    with pytest.raises(ValueError, match="explicit exact"):
        p0.execute(packet, tmp_path / "unselected", run=True)
    assert not (tmp_path / "unselected").exists()
    p0.execute(packet, tmp_path / "doctor")
    with pytest.raises(FileExistsError):
        p0.execute(packet, tmp_path / "doctor")


def test_changed_packet_rejected_before_probe(
    packet: Path, tmp_path: Path, local_service: None,
) -> None:
    (packet / "tasks.jsonl").write_text("CHANGED", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        p0.execute(packet, tmp_path / "bad")
    assert not (tmp_path / "bad/probe.json").exists()


@pytest.mark.parametrize("complete,gate_passed", [(False, False), (True, False), (True, True)])
def test_explicit_run_orders_phases_and_never_bypasses_failed_gate(
    packet: Path, tmp_path: Path, local_service: None,
    monkeypatch: pytest.MonkeyPatch, complete: bool, gate_passed: bool,
) -> None:
    phases = []

    def fixture_run(
        packet: Path, config: Path, phase: str, output: Path, **kwargs: Any,
    ) -> dict[str, Any]:
        phases.append(phase)
        if phase != "calibration":
            assert kwargs["calibration_run"].name == "calibration"
        return {
            "phase": phase, "valid": 40 if complete else 1, "expected": 40,
            "calls": 40 if complete else 1, "gate": {"passed": gate_passed},
            "status": "ENGINEERING_FIXTURE_NOT_REAL_MODEL",
        }

    monkeypatch.setattr(p0.judge, "run_local", fixture_run)
    result = p0.execute(
        packet, tmp_path / "run", model="FIXTURE:exact", run=True, include_project=True,
    )
    if not complete:
        assert phases == ["calibration"]
        assert result["status"] == "STOPPED_INCOMPLETE_PHASE_NO_RETRY"
        assert result["project_review"] == "BLOCKED_INCOMPLETE_REFERENCE_PHASE"
    elif not gate_passed:
        assert phases == ["calibration", "public_holdout"]
        assert result["project_review"] == "BLOCKED_REFERENCE_AGREEMENT_GATE"
    else:
        assert phases == ["calibration", "public_holdout", "project"]
    assert result["judge_inference_calls"] == sum(r["calls"] for r in result["phases"].values())
    assert result["new_project_action_gold"] == 0
    assert result["target_model_experiment"] == "NOT_RUN"


def test_project_opt_in_cannot_start_from_doctor(packet: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit run"):
        p0.execute(packet, tmp_path / "bad", include_project=True)
    assert not (tmp_path / "bad").exists()


def test_model_digest_drift_before_freeze_stops_all_inference(
    packet: Path, tmp_path: Path, local_service: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p0.judge, "freeze", lambda *a: {"model_digest": "DRIFTED"})
    with pytest.raises(ValueError, match="model changed"):
        p0.execute(packet, tmp_path / "drift", model="FIXTURE:exact", run=True)
    assert not (tmp_path / "drift/calibration").exists()


def test_offline_service_retains_blocker_without_inference(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    def offline() -> list[dict[str, Any]]:
        raise httpx.ConnectError("ENGINEERING_OFFLINE_FIXTURE")

    monkeypatch.setattr(p0.judge, "local_models", offline)
    result = p0.execute(
        packet, tmp_path / "offline", model="FIXTURE:exact", run=True, include_project=True,
    )
    assert result["status"] == "BLOCKED_LOCAL_RUNTIME_OR_MODEL"
    assert "LOCAL_SERVICE_UNAVAILABLE" in result["execution_blockers"]
    assert result["judge_inference_calls"] == 0
    assert result["project_review"] == "BLOCKED_NO_REFERENCE_RUNS"


@pytest.mark.parametrize("error", ["duplicate", "group", "score"])
def test_preflight_rejects_inconsistent_reference_contract(
    packet: Path, monkeypatch: pytest.MonkeyPatch, error: str,
) -> None:
    original_read = p0.read_jsonl

    def inconsistent(path: Path) -> list[dict[str, Any]]:
        rows = original_read(path)
        if path.name == "references.jsonl":
            if error == "duplicate":
                rows.append(rows[0])
            elif error == "group":
                rows[40]["input_group"] = rows[0]["input_group"]
            else:
                rows[0]["scores"]["empathy"] = True
        return rows

    monkeypatch.setattr(p0, "read_jsonl", inconsistent)
    with pytest.raises(ValueError, match=r"inventory|references"):
        p0.support_audit(packet)
