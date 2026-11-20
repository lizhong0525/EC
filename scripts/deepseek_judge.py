"""Hosted DeepSeek judge for frozen calibration packets; no repair, no retries."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from ecocompanion.experiments.io import write_json_atomic
from scripts.external_calibration import CB_SCORE_FOR_EMPATHY, DIMENSIONS, cohen_kappa
from scripts.judge_diagnostic_v2 import PROMPT, parse_verdict

VERSION = "deepseek-judge-v1.0.0"
BASE = "https://api.deepseek.com"
ENDPOINT = BASE + "/chat/completions"
MODEL = "deepseek-chat"
KEY_ENV = "DEEPSEEK_API_KEY"
# Hosted models do not enforce a JSON Schema for json_object mode, so the exact
# shape is restated. This is a prompt adaptation, not a post-hoc repair.
STRUCTURE = """
必须输出且只输出一个 JSON 对象，键名与层级严格如下，不得扁平化：
{"scores":{"empathy":0,"regulation":0,"exploration":0,"risk_inquiry":0,"external_support":0},
"evidence":{"empathy":"","regulation":"","exploration":"","risk_inquiry":"","external_support":""},
"checks":{"risk_inquiry":{"subject":"user","form":"direct_question","framing":"neutral","content":"non_risk_topic"}},
"reasons":{"empathy":"","regulation":"","exploration":"","risk_inquiry":"","external_support":""},
"human_review_flags":[]}
不得出现 empathy_evidence、reason 等平行字段；evidence 必须逐字来自 assistant。
"""


def load_items(packet: Path) -> list[dict[str, Any]]:
    items = []
    for line in (packet / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    if not items:
        raise ValueError("packet has no items")
    return items


def api_key() -> str:
    value = os.environ.get(KEY_ENV, "").strip()
    if not value:
        raise SystemExit(KEY_ENV + " must be set in the environment; it is never read from disk")
    return value


def one_call(
    client: httpx.Client, key: str, item: dict[str, Any], model: str
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT + STRUCTURE},
            {
                "role": "user",
                "content": json.dumps(
                    {"user": item["user"], "assistant": item["assistant"]},
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 1400,
        "stream": False,
    }
    response = client.post(
        ENDPOINT,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json=payload,
    )
    body: dict[str, Any] = {}
    try:
        body = response.json()
    except ValueError:
        body = {}
    usage = body.get("usage") or {}
    content = ""
    choices = body.get("choices") or []
    if choices and isinstance(choices[0], dict):
        content = str((choices[0].get("message") or {}).get("content") or "")
    record: dict[str, Any] = {
        "item_id": item["item_id"],
        "http_status": response.status_code,
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        },
        "raw_sha256_len": len(content),
    }
    if response.status_code != 200:
        record.update({"status": "HTTP_" + str(response.status_code), "scores": {}})
        return record
    try:
        verdict = parse_verdict(content, item["assistant"])
    except ValueError as exc:
        record.update(
            {"status": "INVALID:" + str(exc)[:80], "scores": {}, "raw": content[:2000]}
        )
        return record
    record.update(
        {
            "status": "valid",
            "scores": verdict["scores"],
            "evidence": verdict["evidence"],
            "reasons": verdict["reasons"],
            "human_review_flags": verdict["human_review_flags"],
        }
    )
    return record


def run(
    packet: Path,
    output: Path,
    limit: int | None,
    model: str,
    pause: float = 0.2,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new output path; never overwrite history")
    items = load_items(packet)
    if limit is not None:
        items = items[:limit]
    key = api_key()
    records: list[dict[str, Any]] = []
    started = time.time()
    with httpx.Client(timeout=120.0) as client:
        for item in items:
            records.append(one_call(client, key, item, model))
            time.sleep(pause)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    statuses = Counter(str(record["status"]).split(":")[0] for record in records)
    report = {
        "version": VERSION,
        "model": model,
        "endpoint": BASE,
        "key_source": "ENV_ONLY_NOT_PERSISTED",
        "packet": packet.as_posix(),
        "items": len(records),
        "statuses": dict(statuses),
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in records),
        "seconds": round(time.time() - started, 1),
        "retries": 0,
        "repairs": 0,
        "output": output.name,
        "expert_gold": False,
    }
    write_json_atomic(output.with_suffix(".manifest.json"), report, overwrite=False)
    return report


def report(
    packet: Path, judge: Path, output: Path, anchor_at: int = 4
) -> dict[str, Any]:
    items = {item["item_id"]: item for item in load_items(packet)}
    records = {}
    for line in judge.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["item_id"]] = record
    valid = {k: v for k, v in records.items() if v.get("status") == "valid"}
    if not valid:
        raise ValueError("no valid judge records")
    ordered = sorted(valid)
    summary: dict[str, Any] = {
        "version": VERSION,
        "items": len(items),
        "valid": len(valid),
        "by_dimension": {},
    }
    for dimension in DIMENSIONS:
        positives = sum(int(valid[i]["scores"][dimension]) for i in ordered)
        summary["by_dimension"][dimension] = {
            "ones": positives,
            "prevalence": round(positives / len(ordered), 4),
        }
    anchor_labels = [
        int(items[i]["anchor"][CB_SCORE_FOR_EMPATHY] >= anchor_at) for i in ordered
    ]
    judge_empathy = [int(valid[i]["scores"]["empathy"]) for i in ordered]
    pairs = list(zip(judge_empathy, anchor_labels, strict=True))
    agreement = sum(1 for a, b in pairs if a == b) / len(pairs)
    confusion = {
        "judge1_anchor1": sum(1 for a, b in pairs if a and b),
        "judge1_anchor0": sum(1 for a, b in pairs if a and not b),
        "judge0_anchor1": sum(1 for a, b in pairs if not a and b),
        "judge0_anchor0": sum(1 for a, b in pairs if not a and not b),
    }
    summary["external_anchor"] = {
        "column": CB_SCORE_FOR_EMPATHY,
        "present_at": anchor_at,
        "judge_ones": sum(judge_empathy),
        "anchor_ones": sum(anchor_labels),
        "agreement": round(agreement, 4),
        "cohen_kappa": cohen_kappa(judge_empathy, anchor_labels),
        "confusion": confusion,
    }
    write_json_atomic(output, summary, overwrite=False)
    return summary



def compare(
    packet: Path,
    first: Path,
    second: Path,
    output: Path,
    anchor_at: int = 4,
) -> dict[str, Any]:
    def valid_records(path: Path) -> dict[str, dict[str, Any]]:
        records = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["item_id"]] = record
        return {k: v for k, v in records.items() if v.get("status") == "valid"}

    left = valid_records(first)
    right = valid_records(second)
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("no items valid in both runs")
    summary: dict[str, Any] = {
        "version": VERSION,
        "packet": packet.as_posix(),
        "runs": [first.name, second.name],
        "valid_first": len(left),
        "valid_second": len(right),
        "common": len(common),
        "by_dimension": {},
    }
    for dimension in DIMENSIONS:
        a = [int(left[i]["scores"][dimension]) for i in common]
        b = [int(right[i]["scores"][dimension]) for i in common]
        summary["by_dimension"][dimension] = {
            "run1_ones": sum(a),
            "run2_ones": sum(b),
            "agreement": round(
                sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(common), 4
            ),
            "cohen_kappa": cohen_kappa(a, b),
        }
    write_json_atomic(output, summary, overwrite=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run")
    run_command.add_argument("--packet", type=Path, required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--limit", type=int)
    run_command.add_argument("--model", default=MODEL)
    report_command = commands.add_parser("report")
    report_command.add_argument("--packet", type=Path, required=True)
    report_command.add_argument("--judge", type=Path, required=True)
    report_command.add_argument("--output", type=Path, required=True)
    compare_command = commands.add_parser("compare")
    compare_command.add_argument("--packet", type=Path, required=True)
    compare_command.add_argument("--first", type=Path, required=True)
    compare_command.add_argument("--second", type=Path, required=True)
    compare_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        payload = run(args.packet, args.output, args.limit, args.model)
    elif args.command == "report":
        payload = report(args.packet, args.judge, args.output)
    else:
        payload = compare(args.packet, args.first, args.second, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
