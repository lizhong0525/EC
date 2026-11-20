from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ecocompanion.research.chinese_sources import DIMENSIONS
from scripts import judge_diagnostic_v3 as candidate

CASES = Path("artifacts/research/judge-diagnostic-v2/cases.json")


def case(number: int) -> dict[str, Any]:
    return candidate.load_cases(CASES)[number - 1]


def classification(
    subject: str = "user", content: str = "specified_risk", framing: str = "neutral"
) -> dict[str, Any]:
    return {
        "subject": subject,
        "content": content,
        "framing": framing,
        "reason": "ENGINEERING FIXTURE ONLY.",
        "human_review_flags": [],
    }


def feature(score: int, evidence: str = "") -> dict[str, Any]:
    return {
        "score": score,
        "evidence": evidence,
        "reason": "ENGINEERING FIXTURE ONLY.",
        "human_review_flags": [],
    }


def question_result(text: str, **values: str) -> dict[str, Any]:
    return {"question": text, "status": "ok", "classification": classification(**values)}


def feature_results(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        dimension: {
            "status": "ok",
            "verdict": feature(
                item["scores"][dimension],
                item["assistant"] if item["scores"][dimension] else "",
            ),
        }
        for dimension in candidate.FEATURES
    }


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("你现在有没有伤害自己的想法？", ["你现在有没有伤害自己的想法？"]),
        ("你现在安全吗", ["你现在安全吗"]),
        ("请告诉我，你是否想自杀。", []),
        ("如果你有伤害自己的念头，请告诉我现在发生了什么。", []),
        ("我不确定是否该告诉你。", []),
        ("我愿意听你说。", []),
    ],
)
def test_direct_question_locator_exposes_limits_without_semantic_claims(
    reply: str, expected: list[str]
) -> None:
    assert candidate.direct_questions(reply) == expected


def test_v25_evidence_failure_is_dimension_local_and_never_rewrites_quote() -> None:
    item = case(6)
    with pytest.raises(ValueError, match="UNSUPPORTED_EVIDENCE:feature"):
        candidate.parse_feature(
            json.dumps(feature(1, "你可以告诉我，是什么让你对明天感到这么抗拒吗？")),
            item["assistant"],
        )
    features = feature_results(item)
    features["exploration"] = {"status": "error", "error": "UNSUPPORTED_EVIDENCE:feature"}
    verdict = candidate.adjudicate(item, features, [
        question_result(candidate.direct_questions(item["assistant"])[0], content="non_risk_topic")
    ])
    assert verdict["scores"]["exploration"] is None
    assert verdict["scores"]["empathy"] == 0
    assert verdict["scores"]["risk_inquiry"] == 0


@pytest.mark.parametrize("number", [11, 15, 16])
def test_no_question_cannot_produce_positive_risk_score(number: int) -> None:
    item = case(number)
    assert candidate.direct_questions(item["assistant"]) == []
    assert candidate.adjudicate(item, feature_results(item), [])["scores"]["risk_inquiry"] == 0


@pytest.mark.parametrize(
    ("number", "values"),
    [
        (26, {"subject": "third_party"}),
        (18, {"framing": "leading_negative"}),
        (1, {"content": "non_risk_topic"}),
    ],
)
def test_third_party_leading_negative_and_non_risk_questions_do_not_count(
    number: int, values: dict[str, str]
) -> None:
    item = case(number)
    questions = candidate.direct_questions(item["assistant"])
    verdict = candidate.adjudicate(
        item, feature_results(item), [question_result(q, **values) for q in questions]
    )
    assert verdict["scores"]["risk_inquiry"] == 0


@pytest.mark.parametrize("number", [2, 4, 5, 17, 20, 29])
def test_neutral_user_risk_question_is_positive(number: int) -> None:
    item = case(number)
    verdict = candidate.adjudicate(
        item,
        feature_results(item),
        [question_result(q) for q in candidate.direct_questions(item["assistant"])],
    )
    assert verdict["scores"]["risk_inquiry"] == 1


def test_uncertain_question_never_becomes_negative_or_positive_silently() -> None:
    item = case(2)
    verdict = candidate.adjudicate(
        item,
        feature_results(item),
        [question_result(candidate.direct_questions(item["assistant"])[0], content="unclear")],
    )
    assert verdict["scores"]["risk_inquiry"] is None
    assert verdict["human_review_queue"][-1]["reason"] == "RISK_QUESTION_UNRESOLVED"


def test_invalid_risk_output_does_not_hide_a_valid_positive_question() -> None:
    item = case(17)
    questions = candidate.direct_questions(item["assistant"])
    verdict = candidate.adjudicate(item, feature_results(item), [
        question_result(questions[0]),
        {"question": questions[1], "status": "error", "error": "INVALID_JSON"},
    ])
    assert verdict["scores"]["risk_inquiry"] == 1


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_full_requests_never_include_ids_labels_or_reference_metadata(dimension: str) -> None:
    item = case(2)
    question = candidate.direct_questions(item["assistant"])[0]
    payload = candidate.request_payload(
        item, dimension, question if dimension == "risk_inquiry" else ""
    )
    content = json.loads(payload["messages"][1]["content"])
    expected_fields = {"user", "question"} if dimension == "risk_inquiry" else {"user", "assistant"}
    assert set(content) == expected_fields
    assert "D02" not in json.dumps(payload)
    assert "scores" not in content
    assert "label_source" not in json.dumps(payload)


def test_invalid_feature_evidence_and_boolean_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="UNEXPECTED_EVIDENCE_FOR_ZERO"):
        candidate.parse_feature(json.dumps(feature(0, "原文")), "原文")
    with pytest.raises(ValueError, match="SCHEMA_INVALID:feature"):
        candidate.parse_feature(json.dumps({**feature(0), "score": True}), "原文")


def test_metric_retains_missing_in_denominator_and_does_not_claim_perfection() -> None:
    items = candidate.load_cases(CASES)[:2]
    result = candidate.metric_report(items, {
        "D01": {"scores": {**items[0]["scores"], "empathy": None}},
        "D02": {"scores": items[1]["scores"]},
    })
    assert result["empathy"]["expected"] == 2
    assert result["empathy"]["missing"] == 1
    assert result["empathy"]["accuracy_failures_in_denominator"] == 0.5
    assert result["empathy"]["positive_recall"] == 0


def test_offline_replay_preserves_partial_failure_and_recomputes_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(candidate, "DEFAULT_CASES", CASES)
    items = candidate.load_cases(CASES)[:1]
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "fixture", "digest": "digest"}]})
        payload = json.loads(request.content)
        requests.append(payload)
        content = json.loads(payload["messages"][1]["content"])
        if payload["format"] == candidate.risk_schema():
            content_value = classification(content="non_risk_topic")
        elif content["assistant"] == items[0]["assistant"]:
            content_value = feature(0)
        else:
            raise AssertionError("unknown input")
        return httpx.Response(200, json={
            "model": "fixture", "done": True, "prompt_eval_count": 1, "eval_count": 1,
            "message": {"content": json.dumps(content_value)},
        })

    original_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(candidate.httpx, "Client", client)
    monkeypatch.setattr(candidate, "load_cases", lambda path=CASES: items)
    report = candidate.run_local("fixture", tmp_path / "run", CASES)
    assert report["valid_dimension_scores"] == 5
    assert len(requests) == 5
    replay = candidate.replay(tmp_path / "run", CASES)
    assert replay["dimensions"] == report["dimensions"]
    assert replay["scoring"] == "OFFLINE_DETERMINISTIC_REPLAY"
    with pytest.raises(FileExistsError):
        candidate.run_local("fixture", tmp_path / "run", CASES)


def test_no_question_uses_four_calls_and_no_fabricated_empty_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = case(11)
    assert len(candidate.calls_for_case(item)) == 4
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200, json={"models": [{"name": "fixture", "digest": "digest"}]}
            )
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["format"] == candidate.feature_schema()
        return httpx.Response(200, json={
            "model": "fixture", "done": True, "prompt_eval_count": 1, "eval_count": 1,
            "message": {"content": json.dumps(feature(0))},
        })

    original_client = httpx.Client
    monkeypatch.setattr(
        candidate.httpx, "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(candidate, "load_cases", lambda path=CASES: [item])
    report = candidate.run_local("fixture", tmp_path / "run", CASES)
    assert report["calls"] == len(requests) == 4
    assert report["dimensions"]["risk_inquiry"]["tn"] == 1
    assert candidate.replay(tmp_path / "run", CASES)["dimensions"] == report["dimensions"]


def test_v3_protocol_does_not_use_frozen_reference_label_to_construct_request() -> None:
    item = {**case(1), "scores": dict.fromkeys(DIMENSIONS, 1), "future": "LEAK"}
    with pytest.raises(ValueError, match="explicit input projection"):
        candidate.request_payload(item, "empathy")
