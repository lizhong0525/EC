"""One-call loading-check fixtures, not actual model validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ecocompanion.research.chinese_sources import DIMENSIONS
from scripts import local_judge_smoke as smoke


@pytest.mark.parametrize("mode", ["ok", "http_error", "invalid_json", "digest_drift"])
def test_smoke_records_one_call_and_preserves_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    model_checks = 0
    requests: list[dict[str, Any]] = []

    def models() -> list[dict[str, str]]:
        nonlocal model_checks
        model_checks += 1
        return [{
            "name": "FIXTURE:exact",
            "digest": "DRIFT" if mode == "digest_drift" and model_checks > 1 else "FIXTURE",
        }]

    monkeypatch.setattr(smoke, "local_models", models)
    verdict = {
        "scores": dict.fromkeys(DIMENSIONS, 0),
        "reasons": dict.fromkeys(DIMENSIONS, "ENGINEERING_FIXTURE_NOT_HUMAN_RATING"),
        "human_review_flags": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        if request.method == "GET":
            return httpx.Response(200, json={"models": []})
        payload = json.loads(request.content)
        requests.append(payload)
        assert "candidate_id" not in payload and "reference" not in payload
        if mode == "http_error":
            return httpx.Response(500)
        return httpx.Response(200, json={
            "model": "FIXTURE:exact", "done": True,
            "prompt_eval_count": 12, "eval_count": 8,
            "message": {"content": "INVALID" if mode == "invalid_json" else json.dumps(verdict)},
        })

    original_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(smoke.httpx, "Client", client)
    output = tmp_path / "new-run"
    result = smoke.smoke("FIXTURE:exact", output)
    assert result["calls"] == len(requests) == 1
    assert result["retry_budget"] == 0
    assert result["source_reference_used"] is False
    assert result["expert_action_gold"] == 0
    assert result["status"] == (
        "VALID_GENERIC_LOADING_CHECK_NOT_JUDGE_QUALITY_VALIDATION"
        if mode == "ok" else "FAILED_NO_RETRY"
    )
    assert json.loads((output / "result.json").read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        smoke.smoke("FIXTURE:exact", output)
