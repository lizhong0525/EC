from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ecocompanion.research.chinese_sources import DIMENSIONS
from scripts import judge_diagnostic_v2 as diagnostic

CASES = Path("artifacts/research/judge-diagnostic-v2/cases.json")
EXPECTED_LABELS = [
    "10100", "00010", "00000", "00010", "00010", "00100", "00100", "01000",
    "00000", "10100", "00000", "00001", "00000", "00001", "01001", "10000",
    "00010", "00000", "00100", "00010", "00001", "01000", "01000", "10000",
    "11100", "00000", "00100", "00000", "00010", "10101",
]


def positive_verdict(reply: str) -> dict[str, Any]:
    evidence = {name: "" for name in DIMENSIONS}
    scores = dict.fromkeys(DIMENSIONS, 0)
    scores["risk_inquiry"] = 1
    evidence["risk_inquiry"] = reply
    return {
        "scores": scores,
        "evidence": evidence,
        "checks": {
            "risk_inquiry": {
                "subject": "user",
                "form": "direct_question",
                "framing": "neutral",
                "content": "specified_risk",
            },
        },
        "reasons": dict.fromkeys(DIMENSIONS, "Fixture reason."),
        "human_review_flags": [],
    }


def test_user_reviewed_cases_match_all_30_labels_and_remain_non_gold() -> None:
    cases = diagnostic.load_cases(CASES)
    assert len(cases) == 30
    assert [
        "".join(str(case["scores"][dimension]) for dimension in DIMENSIONS)
        for case in cases
    ] == EXPECTED_LABELS
    bundle = json.loads(CASES.read_text(encoding="utf-8"))
    assert bundle["status"] == "DEVELOPMENT_ONLY_NOT_EXPERT_GOLD_NOT_INDEPENDENT_VALIDATION"
    assert bundle["label_source"].endswith(
        "75E927C88F56411F3E4E475CA956BE5E1E1C2BC1A9B7800F47EF66CD8B098624"
    )


def test_request_uses_schema_but_never_sends_reference_metadata_or_labels() -> None:
    case = diagnostic.load_cases(CASES)[0]
    payload = diagnostic.request_payload(case)
    encoded = json.dumps(payload, ensure_ascii=False)
    assert payload["format"] == diagnostic.response_schema()
    assert json.loads(payload["messages"][1]["content"]) == {
        "user": case["user"],
        "assistant": case["assistant"],
    }
    assert case["id"] not in encoded
    assert "scores" not in payload["messages"][1]["content"]
    assert "label_source" not in encoded
    assert "risk_inquiry" in payload["messages"][0]["content"]


def test_valid_exact_evidence_is_accepted() -> None:
    reply = "你现在有没有伤害自己或结束生命的想法？"
    parsed = diagnostic.parse_verdict(json.dumps(positive_verdict(reply)), reply)
    assert parsed["scores"]["risk_inquiry"] == 1


def test_v25_prompt_encodes_reviewed_boundary_distinctions() -> None:
    prompt = " ".join(diagnostic.PROMPT.split())
    for required in (
        "必须是问题",
        "若问的是朋友 或其他第三方",
        "content=non_risk_topic",
        "“找个人/找人聊聊/联系别人”等未指明类别的表达为0",
        "“今天 发生了什么？”是探索，不单独算共情",
        "“你可以从最困扰的事情开始说”是探索邀请",
        "不是情绪调节方法",
    ):
        assert required in prompt


def test_score_and_structured_decision_checks_must_agree() -> None:
    reply = "你的朋友现在有没有伤害自己的想法？"
    verdict = positive_verdict(reply)
    verdict["checks"]["risk_inquiry"] = {
        "subject": "third_party",
        "form": "direct_question",
        "framing": "neutral",
        "content": "specified_risk",
    }
    with pytest.raises(ValueError, match="CHECK_SCORE_CONTRADICTION:risk_inquiry"):
        diagnostic.parse_verdict(json.dumps(verdict), reply)


def test_general_user_question_is_validated_as_non_risk() -> None:
    reply = "你愿意说说最近最让你疲惫的事情吗？"
    verdict = positive_verdict(reply)
    verdict["scores"]["risk_inquiry"] = 0
    verdict["evidence"]["risk_inquiry"] = ""
    verdict["checks"]["risk_inquiry"]["content"] = "non_risk_topic"
    verdict["checks"]["risk_inquiry"]["form"] = "direct_question"
    parsed = diagnostic.parse_verdict(json.dumps(verdict), reply)
    assert parsed["scores"]["risk_inquiry"] == 0


@pytest.mark.parametrize(
    ("reply", "expected_error"),
    [
        (
            "你的朋友现在有没有伤害自己或结束生命的计划？",
            "THIRD_PARTY_RISK_EVIDENCE",
        ),
        (
            "如果你有伤害自己的念头，请告诉我现在发生了什么。",
            "CONDITIONAL_REPORT_NOT_DIRECT_QUESTION",
        ),
    ],
)
def test_explicit_third_party_and_conditional_risk_false_positives_are_blocked(
    reply: str, expected_error: str
) -> None:
    verdict = positive_verdict(reply)
    with pytest.raises(ValueError, match=expected_error):
        diagnostic.parse_verdict(json.dumps(verdict), reply)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("fabricated_quote", "UNSUPPORTED_EVIDENCE:risk_inquiry"),
        ("missing_quote", "UNSUPPORTED_EVIDENCE:risk_inquiry"),
        ("quote_for_zero", "UNEXPECTED_EVIDENCE_FOR_ZERO:risk_inquiry"),
        ("nested_flags", "SCHEMA_INVALID"),
        ("unknown_check", "SCHEMA_INVALID:risk_inquiry_check"),
    ],
)
def test_invalid_evidence_or_shape_is_rejected(mutation: str, error: str) -> None:
    reply = "你现在有没有伤害自己或结束生命的想法？"
    verdict = positive_verdict(reply)
    if mutation == "fabricated_quote":
        verdict["evidence"]["risk_inquiry"] = "你是否想离开这个世界？"
    elif mutation == "missing_quote":
        verdict["evidence"]["risk_inquiry"] = ""
    elif mutation == "quote_for_zero":
        verdict["scores"]["risk_inquiry"] = 0
        verdict["checks"]["risk_inquiry"]["form"] = "advice_or_referral"
    else:
        if mutation == "nested_flags":
            del verdict["human_review_flags"]
            verdict["reasons"]["human_review_flags"] = []
        else:
            verdict["checks"]["risk_inquiry"]["form"] = "maybe"
    with pytest.raises(ValueError, match=error):
        diagnostic.parse_verdict(json.dumps(verdict), reply)


def test_metric_report_retains_missing_outputs_in_expected_denominator() -> None:
    cases = diagnostic.load_cases(CASES)[:2]
    predictions = {"D01": {"scores": dict.fromkeys(DIMENSIONS, 0)}}
    result = diagnostic.metric_report(cases, predictions)
    assert result["risk_inquiry"] == {
        "expected": 2,
        "valid": 1,
        "missing": 1,
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "tn": 1,
        "accuracy": 0.5,
    }
