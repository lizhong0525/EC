"""Run the frozen evidence-extraction fixture against an explicit local Qwen model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from pydantic import ValidationError

from ecocompanion.pet.evidence import (
    QWEN_EVIDENCE_PROMPT,
    QwenAssessment,
    qwen_messages,
)
from ecocompanion.pet.training import load_examples


async def evaluate(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    examples = load_examples(
        args.data or root / "training/data/reviewed_examples.jsonl",
        args.registry or root / "training/data/source_registry.json",
    )
    if args.split:
        groups = json.loads((args.split).read_text(encoding="utf-8"))
        allowed = set(groups.get("test", []))
        examples = [example for example in examples if example.group_id in allowed]
    results = []
    async with httpx.AsyncClient(trust_env=False) as http:
        for example in examples:
            record = {"example_id": example.example_id, "scenario": example.scenario}
            try:
                response = await http.post(
                    f"{args.base_url}/api/chat",
                    json={
                        "model": args.model,
                        "stream": False,
                        "format": QwenAssessment.model_json_schema(),
                        "messages": qwen_messages(example.text),
                        "options": {"temperature": 0, "num_predict": 400, "num_ctx": 4096},
                    },
                    timeout=httpx.Timeout(args.timeout, connect=3),
                )
                response.raise_for_status()
                actual = QwenAssessment.model_validate_json(response.json()["message"]["content"])
                grounded = all(quote and quote in example.text for quote in actual.evidence_quotes)
                record.update(
                    valid=True,
                    grounded=grounded,
                    actual=actual.model_dump(),
                    signal_match=actual.safety_signal == example.target.safety_signal,
                )
            except (httpx.HTTPError, ValidationError, ValueError, KeyError, TypeError) as exc:
                record.update(
                    valid=False, grounded=False, signal_match=False, error=type(exc).__name__
                )
            results.append(record)
    counts = Counter(result["scenario"] for result in results)
    return {
        "schema": "ecocompanion-qwen-baseline-v1",
        "run_id": str(uuid4()),
        "created_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "base_url": "local" if "127.0.0.1" in args.base_url else args.base_url,
        "system_prompt": QWEN_EVIDENCE_PROMPT,
        "scope": "small synthetic engineering fixture; not clinical validation",
        "data": "training/data/reviewed_examples.jsonl",
        "total": len(results),
        "valid_rate": sum(result["valid"] for result in results) / max(1, len(results)),
        "grounded_rate": sum(result["grounded"] for result in results) / max(1, len(results)),
        "signal_accuracy": sum(result["signal_match"] for result in results) / max(1, len(results)),
        "scenario_counts": dict(counts),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--split", type=Path, help="groups.json; only its held-out test groups are used"
    )
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    data = (
        args.data or Path(__file__).resolve().parents[1] / "training/data/reviewed_examples.jsonl"
    )
    report["input_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
