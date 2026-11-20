"""Routing fixtures only; no clinical answer labels and no model generation."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest

from scripts import vera_flow_audit as flow
from scripts.reference_sources import VERA_CATEGORIES, VERA_DIMENSIONS


def rubric(rows: list[dict[str, str]]) -> str:
    headers = [
        "Question ID", "Dimension", "Severity", "Question", "Examples", "Answer", "GOTO",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, headers, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def q(qid: str, dimension: str, answer: str = "", target: str = "") -> dict[str, str]:
    return {
        "Question ID": qid, "Dimension": dimension,
        "Severity": "Red" if qid != "1" else "", "Question": "FIXTURE",
        "Examples": "", "Answer": answer, "GOTO": target,
    }


@pytest.fixture
def questions() -> list[dict[str, Any]]:
    dimensions = list(flow.DIMENSION_FIELDS)
    return flow.parse_rubric(rubric([
        q("1", dimensions[0], "Yes", "2"),
        {"Answer": "No", "GOTO": "END"},
        q("2", dimensions[0]),
        q("3", dimensions[0]),
        q("4", dimensions[1]),
        q("5", dimensions[2]),
        q("6", dimensions[3]),
        q("7", dimensions[4]),
    ]))


def test_implicit_yes_skips_dimension_no_advances(questions: list[dict[str, Any]]) -> None:
    assert flow.next_step(questions, "2", "Yes") == ("4", None)
    assert flow.next_step(questions, "2", "No") == ("3", None)
    assert flow.next_step(questions, "7", "Yes") == (None, None)


def test_explicit_end_is_not_a_successful_safe_score(questions: list[dict[str, Any]]) -> None:
    result = flow.replay(questions, [("1", "No")])
    assert set(result["scores"].values()) == {"Not Relevant"}
    assert "NOT_MODEL_OR_EXPERT" in result["status"]


def test_partial_path_has_no_fabricated_scores(questions: list[dict[str, Any]]) -> None:
    result = flow.replay(questions, [("1", "Yes")])
    assert result["status"] == "INCOMPLETE"
    assert result["scores"] is None
    assert result["next_question"] == "2"


@pytest.mark.parametrize("answer", ["", "yes", "Yes indeed", "No, actually Yes", "unknown"])
def test_invalid_answer_is_not_substring_matched_or_defaulted(
    questions: list[dict[str, Any]], answer: str,
) -> None:
    with pytest.raises(ValueError, match="exact"):
        flow.next_step(questions, "1", answer)


def test_skipped_future_answer_rejected(questions: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="skipped"):
        flow.replay(questions, [("1", "Yes"), ("4", "No")])


@pytest.mark.parametrize("mutation", ["duplicate", "dangling", "dimension", "severity"])
def test_unreviewed_rubric_contracts_rejected(mutation: str) -> None:
    rows = [q("1", "Detects Risk", "Yes", "2"), q("2", "Clarifies Risk")]
    if mutation == "duplicate":
        rows[1]["Question ID"] = "1"
    elif mutation == "dangling":
        rows[0]["GOTO"] = "999"
    elif mutation == "dimension":
        rows[0]["Dimension"] = "NEW CLINICAL MEANING"
    else:
        rows[0]["Severity"] = "Purple"
    with pytest.raises(ValueError):
        flow.parse_rubric(rubric(rows))


def test_cycle_and_unreachable_nodes_are_not_silently_accepted() -> None:
    cyclic = flow.parse_rubric(rubric([q("1", "Detects Risk", "Yes", "1")]))
    with pytest.raises(ValueError, match="cycle"):
        flow.graph_audit(cyclic)
    disconnected = flow.parse_rubric(rubric([
        q("1", "Detects Risk", "Yes", "END"), q("2", "Clarifies Risk"),
    ]))
    with pytest.raises(ValueError, match="unreachable"):
        flow.graph_audit(disconnected)


def test_assign_end_retains_severity_and_other_dimensions_na() -> None:
    questions = flow.parse_rubric(rubric([
        q("1", "Detects Risk", "No", "2"),
        q("2", "Detects Risk", "Yes", "ASSIGN_END"),
    ]))
    result = flow.replay(questions, [("1", "No"), ("2", "Yes")])
    assert result["scores"]["clinician_detects"] == flow.HIGH
    assert all(
        score == flow.NA for d, score in result["scores"].items() if d != "clinician_detects"
    )


def test_not_relevant_jump_affects_only_its_dimension() -> None:
    questions = flow.parse_rubric(rubric([
        q("1", "Detects Risk", "No", "2"),
        q("2", "Clarifies Risk", "Not Relevant", "NOT_RELEVANT>>3"),
        q("3", "Guides to Human Support", "No", "4"),
        q("4", "Collaborates and Validates Appropriately", "No", "5"),
        q("5", "Maintains Safe Boundaries"),
    ]))
    result = flow.replay(questions, [
        ("1", "No"), ("2", "Not Relevant"), ("3", "No"), ("4", "No"), ("5", "No"),
    ])
    assert result["scores"]["clinician_confirms"] == flow.NA
    assert result["scores"]["clinician_detects"] == flow.BEST
    assert result["scores"]["clinician_guides"] == flow.BEST


def test_upstream_ast_is_inspected_not_executed() -> None:
    upstream = """
raise RuntimeError("DO NOT EXECUTE")
class Fixture:
    def _match_answer_to_options(self, a, options):
        return a in options
    def _ask_single_question(self, answer_options):
        matched_answer = answer_options[0]
    def _create_evaluator(self):
        raise ValueError("Not supported: Ollama models.")
"""
    report = flow.ast_findings(upstream)
    assert not report["upstream_python_executed"]
    assert {f["kind"] for f in report["findings"]} == {
        "SUBSTRING_ANSWER_MATCH", "INVALID_ANSWER_DEFAULTS_TO_FIRST_OPTION",
        "PINNED_UPSTREAM_ADAPTER_REJECTS_OLLAMA",
    }


def test_rater_disagreement_remains_in_per_dimension_statistics() -> None:
    rows = []
    for i in range(3):
        rows.append({
            "conversation_id": "fixture-a", "clinician_id": f"FIXTURE_{i}",
            "user_profile": "SYNTHETIC_FIXTURE_PERSONA",
            **dict.fromkeys(VERA_DIMENSIONS, VERA_CATEGORIES[i]),
        })
    report = flow.clinician_diagnostics(rows)
    for dim in report["dimensions"].values():
        assert dim["unresolved"] == 1
        assert dim["pairwise_agreement"] == 0
        assert sum(dim["original_rating_counts"].values()) == 3
        assert sum(dim["majority_counts_excluding_unresolved"].values()) == 0
        assert dim["conversation_clustered_alpha"]["clusters"] == 1
        assert dim["persona_clustered_alpha_sensitivity"]["clusters"] == 1
        assert dim["majority_reference_categories_without_support"] == list(VERA_CATEGORIES)
    assert "NOT_JUDGE" in report["scope"]


def test_repeated_rater_identity_rejected() -> None:
    row = {
        "conversation_id": "fixture", "clinician_id": "SAME_FIXTURE",
        "user_profile": "SYNTHETIC_FIXTURE_PERSONA",
        **dict.fromkeys(VERA_DIMENSIONS, VERA_CATEGORIES[0]),
    }
    with pytest.raises(ValueError, match="distinct"):
        flow.clinician_diagnostics([row, row, row])


@pytest.mark.parametrize("mutation", ["empty", "category", "persona", "blank_identity"])
def test_invalid_original_ratings_fail_closed(mutation: str) -> None:
    rows = [
        {
            "conversation_id": "fixture", "clinician_id": f"FIXTURE_{i}",
            "user_profile": "SYNTHETIC_FIXTURE_PERSONA",
            **dict.fromkeys(VERA_DIMENSIONS, VERA_CATEGORIES[0]),
        }
        for i in range(3)
    ]
    if mutation == "empty":
        rows = []
    elif mutation == "category":
        rows[0][VERA_DIMENSIONS[0]] = "UNKNOWN"
    elif mutation == "persona":
        rows[0]["user_profile"] = "DIFFERENT_FIXTURE"
    else:
        rows[0]["clinician_id"] = ""
    with pytest.raises(ValueError):
        flow.clinician_diagnostics(rows)


def test_undefined_reliability_is_not_reported_as_perfect() -> None:
    rows = [
        {
            "conversation_id": "fixture", "clinician_id": f"FIXTURE_{i}",
            "user_profile": "SYNTHETIC_FIXTURE_PERSONA",
            **dict.fromkeys(VERA_DIMENSIONS, VERA_CATEGORIES[0]),
        }
        for i in range(3)
    ]
    report = flow.clinician_diagnostics(rows)
    for dim in report["dimensions"].values():
        assert dim["nominal_alpha"] is None
        assert dim["conversation_clustered_alpha"]["ci95"] is None
        assert dim["conversation_clustered_alpha"]["defined_resamples"] == 0
        assert dim["pairwise_agreement"] == 1


def test_audit_refuses_overwrite_without_touching_snapshot(tmp_path: Path) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        flow.audit(tmp_path / "no-snapshot-read-needed", out)
