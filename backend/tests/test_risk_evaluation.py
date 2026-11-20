"""Mock-only diagnostics tests; never human labels, paid calls or model validation."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts import risk_evaluation as r


def test_baseline_parser_requires_a_closed_valid_contract() -> None:
    valid = {"action": "SUPPORTIVE_LISTENING", "risk_level": "L0", "reply": "TEST_FIXTURE"}
    assert r.parse_baseline(json.dumps(valid)) == valid
    for invalid in (
        {**valid, "action": "UNKNOWN"},
        {**valid, "risk_level": "UNKNOWN"},
        {**valid, "reply": ""},
        {**valid, "extra": True},
    ):
        with pytest.raises(ValueError):
            r.parse_baseline(json.dumps(invalid))


def test_requests_have_equal_limits_and_no_reference_answer() -> None:
    probe = r.d.read_json(r.PROBES)[8]
    simple, renderer = r.requests_for(probe)
    assert simple["options"] == renderer["options"]
    assert simple["model"] == renderer["model"] == r.MODEL
    shared = json.loads(simple["messages"][-1]["content"])
    assert shared["current_message"] == probe["text"]
    assert shared["visible_history"] == probe["history"]
    assert shared["previous_system_state"]["risk_level"] == "L3"
    for request in (simple, renderer):
        text = json.dumps(request, ensure_ascii=False)
        assert "post_ideation_companionship" not in text
        assert "expert_gold" not in text
        assert "P09" not in text


@pytest.mark.parametrize("failure", ["none", "timeout", "truncated", "contract"])
def test_local_call_accounting_never_becomes_expert_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(
        r, "historical_comparison", lambda _: {"scope": "TEST_FIXTURE_NOT_HISTORICAL_RESULTS"}
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        assert str(request.url).startswith(r.BASE)
        assert "authorization" not in request.headers
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": r.MODEL, "digest": "TEST_FIXTURE_DIGEST"}]}
            )
        assert request.url.path == "/api/chat"
        calls += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("TEST_FIXTURE_TIMEOUT")
        payload = json.loads(request.content)
        content = "TEST_FIXTURE_NOT_A_REAL_REPLY"
        if "format" in payload:
            content = json.dumps(
                {"action": "SUPPORTIVE_LISTENING", "risk_level": "L0", "reply": content}
            )
            if failure == "contract":
                content = "{}"
        return httpx.Response(
            200,
            json={
                "message": {"content": content},
                "done_reason": "length" if failure == "truncated" else "stop",
                "prompt_eval_count": 100,
                "eval_count": 30,
            },
        )

    output = tmp_path / "run"
    summary = r.run(output, transport=httpx.MockTransport(handler))
    assert summary["attempted_calls"] == calls
    assert calls == (1 if failure == "timeout" else 32)
    assert sum(summary["statuses"].values()) == 32
    assert summary["expert_accuracy"] is None
    assert summary["safety_recall"] is None
    assert summary["mis_escalation_rate"] is None
    assert summary["human_verdicts"] == summary["paid_calls"] == 0
    if failure == "timeout":
        assert summary["statuses"]["NOT_ATTEMPTED_STOPPED"] == 31
    elif failure == "truncated":
        assert summary["statuses"]["TRUNCATED_OR_INCOMPLETE"] == 32
    elif failure == "contract":
        assert summary["statuses"]["CALL_OR_CONTRACT_ERROR"] == 16


def test_historical_recompute_detects_coverage_or_identity_errors(tmp_path: Path) -> None:
    r.d.write_json(tmp_path / "manifest.json", {"planned_unit_ids": ["EXPECTED"]})
    r.d.write_json(tmp_path / "aggregate.json", {})
    (tmp_path / "results.jsonl").write_text('{"unit_id":"OTHER"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="coverage incomplete"):
        r.historical_comparison(tmp_path)
    (tmp_path / "results.jsonl").write_text(
        '{"unit_id":"OTHER"}\n{"unit_id":"OTHER"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate historical"):
        r.historical_comparison(tmp_path)
