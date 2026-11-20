"""Offline evaluation-governance fixtures; no real reviewers, labels or model calls."""

from __future__ import annotations

import csv
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from scripts import distillation as d
from scripts import renderer_evaluation as e


@pytest.fixture
def setup(tmp_path: Path) -> Path:
    root = tmp_path / "TEST_FIXTURE_DATA"
    files = {
        "docs/DISTILLATION_PILOT_SEEDS_V1.json": [
            {
                "id": "old-fixture",
                "group": "old-fixture",
                "split": "heldout",
                "text": "这是过去开发阶段已经使用过的原文。",
            }
        ],
        "artifacts/research/judge-diagnostic-v2/cases.json": {
            "cases": [{"user": "旧题的用户输入。", "assistant": "旧题的助手回答。"}]
        },
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        d.write_json(path, value)
    calibration = root / "artifacts/research/human-calibration-20261023-v1/items.jsonl"
    calibration.parent.mkdir(parents=True)
    e.write_jsonl(calibration, [{"user": "历史校准输入。", "assistant": "历史校准回答。"}])
    inventory = root / "docs/DATA_SOURCE_INVENTORY_20261025.md"
    inventory.write_text("TEST_FIXTURE inventory only\n", encoding="utf-8", newline="\n")
    output = tmp_path / "setup"
    e.prepare(output, data_root=root)
    return output


def batch() -> list[dict[str, object]]:
    categories = d.read_json(e.SPEC)["sample_plan"]["categories"]
    return [
        {
            "case_id": f"fixture-{index}",
            "category": category,
            "text": f"TEST_FIXTURE {uuid5(NAMESPACE_URL, str(index)).hex} 今天想聊聊日常。",
            "history": [],
            "source_id": "TEST_FIXTURE_independent_author_batch",
            "source_revision": "TEST_FIXTURE_revision",
            "source_record_id": f"TEST_FIXTURE_record_{index}",
            "author_id": "TEST_FIXTURE_AUTHOR_NOT_A_REAL_PERSON",
            "origin": "independent_author",
            "adult_fictional": True,
            "rights_reference": "TEST_FIXTURE_RIGHTS_NOT_REAL_AUTHORIZATION",
            "privacy_reference": "TEST_FIXTURE_PRIVACY_NOT_REAL_REVIEW",
            **{field: f"TEST_FIXTURE_{field}_{index}" for field in e.GROUPS},
        }
        for index, category in enumerate(category for category in categories for _ in range(5))
    ]


def approvals(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for case in cases:
        candidate = e.Candidate.model_validate(case)
        _, _, outcome = d.context(candidate.model_dump(mode="json"))
        for reviewer in ("TEST_FIXTURE_REVIEWER_A", "TEST_FIXTURE_REVIEWER_B"):
            result.append(
                {
                    "case_id": candidate.case_id,
                    "candidate_sha256": e.canonical(candidate.model_dump(mode="json")),
                    "reviewer_id": reviewer,
                    "review_reference": "TEST_FIXTURE_NOT_A_REAL_REVIEW",
                    "decision": "APPROVE",
                    "checks": sorted(e.CHECKS),
                    "approved_action": outcome.plan.primary_action.value,
                }
            )
    return result


def audit_batch(
    setup: Path,
    tmp_path: Path,
    cases: list[dict[str, object]],
    reviews: list[dict[str, object]],
) -> Path:
    source = tmp_path / "candidates.jsonl"
    votes = tmp_path / "reviews.jsonl"
    e.write_jsonl(source, cases)
    e.write_jsonl(votes, reviews)
    output = tmp_path / "audit"
    e.audit(setup, source, votes, output)
    return output


def codes(audited: Path) -> set[str]:
    return {error["code"] for error in d.read_json(audited / "audit.json")["errors"]}


def test_design_is_frozen_but_does_not_claim_an_evaluation_set(setup: Path) -> None:
    protocol, exposure = e.load_setup(setup)
    spec = protocol["specification"]
    assert spec["sample_plan"]["cases"] == 40
    assert spec["primary_metric"]["name"] == "paired_human_preference_distilled_vs_prompt_only"
    assert spec["human_review"]["no_aggregate_safety_score"] is True
    assert spec["release_policy"]["automatic_release"] is False
    readiness = d.read_json(setup / "readiness.json")
    assert readiness["independent_evaluation_cases"] == 0
    assert readiness["ready_for_evaluation"] is False
    assert exposure["complete_global_exposure_history"] is False
    assert (setup / "intake/candidates.jsonl").read_bytes() == b""
    with (setup / "case_groups.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["review_status"] == "DEVELOPMENT_ONLY_NOT_INDEPENDENT"
    assert rows[0]["person_group"] == "UNKNOWN"


def test_empty_input_is_audited_and_refuses_freeze(setup: Path, tmp_path: Path) -> None:
    audited = audit_batch(setup, tmp_path, [], [])
    assert "CASE_COUNT_MUST_BE_40" in codes(audited)
    output = tmp_path / "private/evaluation"
    with pytest.raises(ValueError, match="source audit blocked"):
        e.freeze(setup, audited, output)
    assert not output.exists()


def test_complete_test_fixture_can_freeze_but_never_run_or_be_gold(
    setup: Path, tmp_path: Path
) -> None:
    cases = batch()
    audited = audit_batch(setup, tmp_path, cases, approvals(cases))
    assert not codes(audited)
    output = tmp_path / "private/fixture"
    summary = e.freeze(setup, audited, output)
    assert summary["cases"] == 40
    assert summary["ready_for_model_evaluation"] is False
    assert summary["human_response_ratings"] == summary["model_runs"] == 0
    assert summary["expert_gold"] is False
    for task in d.jsonl(output / "tasks.jsonl"):
        assert set(task) == {"task_id", "messages"}
        payload = str(task["messages"])
        assert "TEST_FIXTURE_record_" not in payload
        assert "TEST_FIXTURE_RIGHTS" not in payload
        assert "source_group" not in payload
        assert "category" not in payload
    with pytest.raises(FileExistsError):
        e.freeze(setup, audited, output)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("person_group", "UNKNOWN", "UNKNOWN_PROVENANCE_OR_GROUP"),
        ("template_group", "PENDING_TEMPLATE", "UNKNOWN_PROVENANCE_OR_GROUP"),
        ("source_id", "psycrisis", "SOURCE_ALREADY_USED"),
        ("source_id", "distillation_pilot_v1", "SOURCE_ALREADY_USED"),
        ("rights_reference", "UNKNOWN", "UNKNOWN_PROVENANCE_OR_GROUP"),
        ("text", "这是过去开发阶段已经使用过的原文。", "EXACT_OR_NEAR_EXPOSURE_MATCH"),
        ("text", "我现在想死", "CRISIS_EXCLUDED_FROM_RENDERER_PILOT"),
    ],
)
def test_source_and_scope_failures_are_not_auto_approved(
    setup: Path, tmp_path: Path, field: str, value: str, expected: str
) -> None:
    cases = batch()
    cases[0][field] = value
    audited = audit_batch(setup, tmp_path, cases, approvals(cases))
    assert expected in codes(audited)
    with pytest.raises(ValueError, match="source audit blocked"):
        e.freeze(setup, audited, tmp_path / "private/fixture")


def test_related_cases_are_counted_as_one_group(setup: Path, tmp_path: Path) -> None:
    cases = batch()
    for case in cases:
        case["template_group"] = "TEST_FIXTURE_SHARED_TEMPLATE"
    audited = audit_batch(setup, tmp_path, cases, approvals(cases))
    report = d.read_json(audited / "audit.json")
    assert report["connected_source_groups"] == 1
    assert "INSUFFICIENT_CONNECTED_SOURCE_GROUPS" in codes(audited)


def test_duplicate_and_shared_history_are_not_independent(setup: Path, tmp_path: Path) -> None:
    cases = batch()
    cases[1]["text"] = cases[0]["text"]
    common = [{"role": "user", "content": "TEST_FIXTURE 这段历史已经在另一条案例中出现过了。"}]
    cases[2]["history"] = common
    cases[3]["history"] = common
    audited = audit_batch(setup, tmp_path, cases, approvals(cases))
    assert "WITHIN_BATCH_EXACT_OR_NEAR_DUPLICATE" in codes(audited)
    assert "SHARED_VISIBLE_CONTEXT_ACROSS_SOURCE_GROUPS" in codes(audited)


@pytest.mark.parametrize("kind", ["missing", "digest", "author", "action", "pending"])
def test_two_human_approvals_must_be_bound_and_complete(
    setup: Path, tmp_path: Path, kind: str
) -> None:
    cases = batch()
    reviews = approvals(cases)
    if kind == "missing":
        reviews.pop()
    elif kind == "digest":
        reviews[0]["candidate_sha256"] = "0" * 64
    elif kind == "author":
        reviews[0]["reviewer_id"] = cases[0]["author_id"]
    elif kind == "action":
        reviews[0]["approved_action"] = "NOT_THE_FROZEN_ACTION"
    else:
        reviews[0]["review_reference"] = "PENDING"
    audited = audit_batch(setup, tmp_path, cases, reviews)
    assert "NEED_TWO_BOUND_INDEPENDENT_HUMAN_SOURCE_APPROVALS" in codes(audited)


def test_duplicate_reviewer_identity_is_rejected(setup: Path, tmp_path: Path) -> None:
    cases = batch()
    reviews = approvals(cases)
    reviews[1]["reviewer_id"] = str(reviews[0]["reviewer_id"]).lower()
    with pytest.raises(ValueError, match="duplicate reviewer"):
        audit_batch(setup, tmp_path, cases, reviews)
    assert not (tmp_path / "audit").exists()


def test_setup_tamper_and_code_drift_are_rejected(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(e, "code_identity", lambda: {})
    with pytest.raises(ValueError, match="code changed"):
        e.load_setup(setup)
    (setup / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen packet changed"):
        e.load_setup(setup)


def test_audit_tamper_and_nonprivate_export_are_rejected(setup: Path, tmp_path: Path) -> None:
    cases = batch()
    audited = audit_batch(setup, tmp_path, cases, approvals(cases))
    with pytest.raises(ValueError, match="private directory"):
        e.freeze(setup, audited, tmp_path / "public")
    (audited / "audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen packet changed"):
        e.freeze(setup, audited, tmp_path / "private/fixture")


def test_credentials_in_metadata_are_rejected_before_output(setup: Path, tmp_path: Path) -> None:
    cases = batch()
    cases[0]["privacy_reference"] = "sk-" + "TEST_FIXTURE_ONLY_NOT_A_REAL_KEY"
    with pytest.raises(ValueError, match="credentials"):
        audit_batch(setup, tmp_path, cases, approvals(cases))
    assert not (tmp_path / "audit").exists()


def test_unicode_duplicate_screen_and_immutability(setup: Path, tmp_path: Path) -> None:
    assert e.similar("Ａ B Ｃ", "abc")
    assert not e.similar("", "")
    with pytest.raises(FileExistsError):
        e.prepare(setup, data_root=tmp_path / "TEST_FIXTURE_DATA")
