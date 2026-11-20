"""One separately-accounted generic loading check; no source reference or Gold."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ecocompanion.experiments.io import write_json_atomic
from scripts.external_reference import request_payload
from scripts.local_reference_judge import BASE, local_models, parse_verdict, visible_response


def smoke(model: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("use a new loading-check directory, no hidden retries")
    models = [m for m in local_models() if m["name"] == model]
    if len(models) != 1:
        raise ValueError("exact installed model required")
    task = {
        "user": "I am a fictional adult testing a text interface. Today was tiring.",
        "reply": "It sounds like today left you feeling tired. What felt most tiring?",
    }
    payload = {**request_payload(task), "model": model}
    record: dict[str, Any] = {
        "version": "local-judge-smoke-v1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "model_digest": models[0]["digest"],
        "request": payload,
        "calls": 0,
        "retry_budget": 0,
        "purpose": "GENERIC_LOADING_CHECK_NOT_CALIBRATION_OR_HOLDOUT",
        "source_reference_used": False,
        "expert_action_gold": 0,
        "remote_model_calls": 0,
    }
    write_json_atomic(output / "started.json", record, overwrite=False)
    start = time.monotonic()
    try:
        with httpx.Client(timeout=180, trust_env=False, follow_redirects=False) as client:
            record["calls"] = 1
            response = client.post(BASE + "/api/chat", json=payload)
            response.raise_for_status()
            raw = visible_response(response.json())
            record["response"] = raw
            if raw.get("model") != model or raw.get("done") is not True:
                raise ValueError("model identity/completeness mismatch")
            if any(
                type(raw.get(k)) is not int or raw[k] < 0
                for k in ("prompt_eval_count", "eval_count")
            ):
                raise ValueError("token accounting unavailable")
            record["verdict"] = parse_verdict(raw["message"]["content"])
            state = client.get(BASE + "/api/ps")
            state.raise_for_status()
            record["loaded_models"] = state.json()
        after = [m for m in local_models() if m["name"] == model]
        if len(after) != 1 or after[0]["digest"] != record["model_digest"]:
            raise ValueError("model digest changed during loading check")
        record["status"] = "VALID_GENERIC_LOADING_CHECK_NOT_JUDGE_QUALITY_VALIDATION"
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        record.update(status="FAILED_NO_RETRY", error_type=type(exc).__name__)
    record["latency_ms"] = (time.monotonic() - start) * 1000
    write_json_atomic(output / "result.json", record, overwrite=False)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = smoke(args.model, args.output)
    print(json.dumps({
        "status": record["status"], "calls": record["calls"],
        "latency_ms": record["latency_ms"],
    }))
    if record["status"] == "FAILED_NO_RETRY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
