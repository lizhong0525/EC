"""This module uses deliberately tiny engineering fixtures, not real AI/expert data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ecocompanion.research.annotation import digest
from ecocompanion.research.chinese_sources import DIMENSIONS, read_jsonl, sha256, write_jsonl
from scripts import review_chinese as review


def test_scan_is_only_a_queue_and_does_not_echo_values() -> None:
    text = "Fixture email fixture@example.invalid and mobile 13800000000"
    flagged = review.scan_text(text, source="ENGINEERING_FIXTURE", row=1)
    assert {f["kind"] for f in flagged} == {"email", "possible_mobile"}
    assert "13800000000" not in json.dumps(flagged)
    assert "fixture@example.invalid" not in json.dumps(flagged)
    assert review.scan_text("普通的研究夹具，不是真实材料。") == []


def test_calibration_selects_complete_groups_without_holdout_or_references() -> None:
    tasks = [
        {"candidate_id": str(i), "input_group": str(i // 2), "user": "fixture", "reply": "fixture"}
        for i in range(8)
    ]
    refs = [
        {
            "candidate_id": str(i),
            "partition": "calibration" if i < 6 else "public_holdout",
            "scores": {d: 1 for d in DIMENSIONS},
        }
        for i in range(8)
    ]
    selected = review.select_calibration(tasks, refs, groups=2)
    assert len(selected) == 4
    assert not ({t["candidate_id"] for t in selected} & {"6", "7"})
    assert all("scores" not in t and "partition" not in t for t in selected)
    for group in {t["input_group"] for t in selected}:
        assert sum(t["input_group"] == group for t in selected) == 2
    assert selected == review.select_calibration(list(reversed(tasks)), refs, groups=2)
    with pytest.raises(ValueError):
        review.select_calibration(tasks, refs, groups=4)


def test_reply_sample_covers_payloads_without_policy_identity() -> None:
    tasks = [
        {
            "candidate_id": str(i),
            "reply": {"text": str(i % 3), "question": None},
            "observation": {"current_observation": {"message": "fixture"}},
            "policy": "must-not-be-projected",
        }
        for i in range(15)
    ]
    selected = review.select_replies(tasks, count=10)
    assert len(selected) == 10
    assert len({t["candidate_id"] for t in selected}) == 10
    assert all("policy" not in t for t in selected)
    assert {t["reply"]["text"] for t in selected} == {"0", "1", "2"}
    with pytest.raises(ValueError):
        review.select_replies(tasks, count=2)


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    root, original = tmp_path / "packet", tmp_path / "original"
    root.mkdir()
    calibration = original / "psycrisis-calibration"
    calibration.mkdir(parents=True)
    refs = [
        {
            "candidate_id": "cal-1",
            "partition": "calibration",
            "scores": {d: 1 for d in DIMENSIONS},
        },
        {
            "candidate_id": "holdout",
            "partition": "public_holdout",
            "scores": {d: 0 for d in DIMENSIONS},
        },
    ]
    write_jsonl(calibration / "private/references.jsonl", refs)
    write_jsonl(
        calibration / "tasks.jsonl", [{"candidate_id": "cal-1"}, {"candidate_id": "holdout"}]
    )
    tasks = [
        {
            "candidate_id": "cal-1",
            "review_task_id": "cal-001",
            "review_track": "source_calibration",
            "user": "fixture user",
            "reply": "fixture reply",
        },
        {
            "candidate_id": "reply-1",
            "review_task_id": "reply-001",
            "review_track": "project_reply",
            "user": "fixture user",
            "reply": {"text": "fixture reply", "question": None},
        },
    ]
    write_jsonl(root / "calibration-batch.jsonl", tasks[:1])
    write_jsonl(root / "reply-batch.jsonl", tasks[1:])
    (root / "prereview.json").write_text("{}", encoding="utf-8")
    rubric = "ENGINEERING FIXTURE ONLY"
    (root / "rubric-snapshot.json").write_text(json.dumps({"text": rubric}), encoding="utf-8")
    plan = {
        "version": review.VERSION,
        "review_script_sha256": review.source_digest(),
        "judge_id": "codex-in-session-assistant",
        "judge_kind": "AI_ASSISTED_IN_SESSION_NOT_HUMAN_NOT_INDEPENDENT_EXPERT",
        "rubric_sha256": sha256(rubric.encode()),
        "original_root": str(original),
        "original_calibration_task_sha256": sha256((calibration / "tasks.jsonl").read_bytes()),
        "original_calibration_ref_sha256": sha256(
            (calibration / "private/references.jsonl").read_bytes()
        ),
        "task_hashes": {
            name: sha256((root / name).read_bytes())
            for name in ("calibration-batch.jsonl", "reply-batch.jsonl", "prereview.json")
        },
        "full_source_task_count": 2,
    }
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return root


def ratings(packet: Path) -> list[dict[str, Any]]:
    plan = json.loads((packet / "plan.json").read_text(encoding="utf-8"))
    return [
        {
            "candidate_id": t["candidate_id"],
            "kind": "llm",
            "judge_id": plan["judge_id"],
            "status": "ok",
            "task_sha256": digest(t),
            "rubric_sha256": plan["rubric_sha256"],
            "verification_reference": "PYTEST_ENGINEERING_FIXTURE_NOT_ACTUAL_JUDGMENT",
            "scores": {d: 1 for d in DIMENSIONS},
            "reasons": {d: "Engineering fixture only." for d in DIMENSIONS},
            "human_review_flags": [],
        }
        for t in [
            *read_jsonl(packet / "calibration-batch.jsonl"),
            *read_jsonl(packet / "reply-batch.jsonl"),
        ]
    ]


def test_actual_aggregation_without_experts_or_holdout_and_ref_changes_are_separate(
    packet: Path, tmp_path: Path
) -> None:
    path = tmp_path / "ratings.jsonl"
    write_jsonl(path, ratings(packet))
    first = review.score_batch(packet, path, tmp_path / "first.json")
    assert first["status"] == "COMPLETE_AI_PILOT_NOT_EXPERT_CALIBRATED"
    assert first["new_expert_gold"] == first["public_holdout_rated"] == 0
    assert first["source_full_task_count"] == 2
    dim = first["tracks"]["source_calibration"]["dimensions"]["empathy"]
    assert dim["upstream_agreement_accuracy_failures_in_denominator"] == 1
    plan_path = packet / "plan.json"
    plan = json.loads(plan_path.read_text())
    ref_path = Path(plan["original_root"]) / "psycrisis-calibration/private/references.jsonl"
    refs = read_jsonl(ref_path)
    refs[0]["scores"]["empathy"] = 0
    ref_path.write_text("".join(json.dumps(r) + "\n" for r in refs))
    plan["original_calibration_ref_sha256"] = sha256(ref_path.read_bytes())
    plan_path.write_text(json.dumps(plan))
    changed = review.score_batch(packet, path, tmp_path / "changed.json")
    dim = changed["tracks"]["source_calibration"]["dimensions"]["empathy"]
    assert dim["upstream_agreement_accuracy_failures_in_denominator"] == 0
    assert changed["tracks"]["project_reply"] == first["tracks"]["project_reply"]
    assert (
        changed["tracks"]["source_calibration"]["dimensions"]["regulation"]
        == (first["tracks"]["source_calibration"]["dimensions"]["regulation"])
    )
    with pytest.raises(FileExistsError):
        review.score_batch(packet, path, tmp_path / "first.json")


def test_missing_ratings_are_not_excluded(packet: Path, tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    write_jsonl(path, [])
    result = review.score_batch(packet, path, tmp_path / "partial.json")
    dim = result["tracks"]["source_calibration"]["dimensions"]["empathy"]
    assert result["status"] == "PARTIAL_AI_PILOT"
    assert dim["expected"] == dim["not_yet_rated"] == 1 and dim["rated"] == 0
    assert dim["upstream_agreement_accuracy_failures_in_denominator"] == 0
    assert dim["positive_recall_failures_retained"] == 0


def test_transcribing_explicit_decisions_does_not_create_experts(
    packet: Path, tmp_path: Path
) -> None:
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            {
                "dimension_order": list(DIMENSIONS),
                "decisions": [
                    {
                        "review_task_id": "cal-001",
                        "values": [1, 0, 0, 0, 0],
                        "reasons": ["ENGINEERING FIXTURE ONLY"] * 5,
                        "human_review_flags": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "transcribed.jsonl"
    result = review.record_ai_decisions(packet, path, output)
    assert result["in_session_ai_judgments"] == 1
    loaded = read_jsonl(output)
    assert loaded[0]["kind"] == "llm"
    assert loaded[0]["scores"]["empathy"] == 1
    assert loaded[0]["scores"]["risk_inquiry"] == 0


@pytest.mark.parametrize(
    "bad", ["expert", "boolean", "task_hash", "rubric", "duplicate", "unknown"]
)
def test_rejects_false_expert_or_invalid_ratings(packet: Path, tmp_path: Path, bad: str) -> None:
    records = ratings(packet)
    if bad == "expert":
        records[0]["kind"] = "expert"
    elif bad == "boolean":
        records[0]["scores"]["empathy"] = True
    elif bad == "task_hash":
        records[0]["task_sha256"] = "wrong"
    elif bad == "rubric":
        records[0]["rubric_sha256"] = "wrong"
    elif bad == "duplicate":
        records.append(records[0])
    else:
        records[0]["candidate_id"] = "unknown"
    path = tmp_path / "invalid.jsonl"
    write_jsonl(path, records)
    with pytest.raises(ValueError):
        review.score_batch(packet, path, tmp_path / "invalid-report.json")


@pytest.mark.parametrize(
    "name", ["calibration-batch.jsonl", "prereview.json", "rubric-snapshot.json"]
)
def test_frozen_review_tampering_is_rejected(packet: Path, tmp_path: Path, name: str) -> None:
    path = tmp_path / "ratings.jsonl"
    write_jsonl(path, ratings(packet))
    (packet / name).write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        review.score_batch(packet, path, tmp_path / "invalid-report.json")
