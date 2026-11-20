"""Versioned renderer-distillation pilot; never auto-approve or train model weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

from ecocompanion.agent import SafetyAuditor
from ecocompanion.application.conversation import (
    CONVERSATION_POLICY_VERSION,
    RENDER_PROMPT_VERSION,
    WebConversationPolicy,
    response_quality_codes,
)
from ecocompanion.application.renderer import LLMResponseRenderer
from ecocompanion.core.policy import PolicyOutcome
from ecocompanion.core.risk import AssessmentResult, RuleBasedRiskAssessor
from ecocompanion.domain.enums import Action, ActorRole, RiskLevel, Track
from ecocompanion.domain.schemas import (
    ConversationMessage,
    CurrentObservation,
    ObservationV1,
    SessionStateV1,
    VisibleContext,
    VisibleProfile,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "renderer-distillation-pilot-v1.0.0"
ENDPOINT = "https://api.deepseek.com/chat/completions"
SOURCES = {
    "terms": "https://cdn.deepseek.com/policies/zh-CN/deepseek-terms-of-use.html",
    "pricing": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    "thinking": "https://api-docs.deepseek.com/guides/thinking_mode/",
}
KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
UNSUPPORTED = re.compile(r"自杀|自残|伤害自己|不想活|想死|结束生命|结束自己的生命")


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def freeze_sources(output: Path, *, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    with httpx.Client(
        timeout=30, trust_env=False, follow_redirects=False, transport=transport
    ) as client:
        for name, url in SOURCES.items():
            response = client.get(url)
            response.raise_for_status()
            raw = response.content
            (output / f"{name}.html").write_bytes(raw)
            parser = TextParser()
            parser.feed(response.text)
            text = "\n".join(parser.parts)
            (output / f"{name}.txt").write_text(text, encoding="utf-8", newline="\n")
            records[name] = {
                "url": url,
                "sha256": digest(raw),
                "text_sha256": digest(text.encode()),
            }
    manifest = {
        "version": VERSION,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "records": records,
        "review": "SOURCE_CAPTURE_NOT_LEGAL_OR_CLINICAL_APPROVAL",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_sources(source: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = read_json(source / "manifest.json")
    for name, url in SOURCES.items():
        record = manifest["records"][name]
        if (
            record["url"] != url
            or digest((source / f"{name}.html").read_bytes()) != record["sha256"]
            or digest((source / f"{name}.txt").read_bytes()) != record["text_sha256"]
        ):
            raise ValueError("official source snapshot does not match its manifest")
    if "训练其他模型（如模型蒸馏）" not in (source / "terms.txt").read_text(encoding="utf-8"):
        raise ValueError("terms snapshot does not contain the reviewed distillation clause")
    if "deepseek-flash" not in (source / "pricing.txt").read_text(encoding="utf-8"):
        raise ValueError("pricing snapshot does not contain the selected model")
    return manifest


def context(item: dict[str, Any]) -> tuple[ObservationV1, AssessmentResult, PolicyOutcome]:
    history = tuple(
        ConversationMessage(
            message_id=f"visible-{index}",
            role=ActorRole.USER if entry["role"] == "user" else ActorRole.COMPANION,
            content=entry["content"],
        )
        for index, entry in enumerate(item.get("history", []))
    )
    observation = ObservationV1(
        track=Track.C,
        episode_id="synthetic-renderer-development",
        step_id=len(history) // 2,
        simulated_time="2026-10-25T00:00:00Z",
        visible_profile=VisibleProfile(age_group="fictional adult research scenario"),
        current_observation=CurrentObservation(
            observation_id="current-visible-message", message=item["text"]
        ),
        visible_context=VisibleContext(conversation_history=history),
        allowed_primary_actions=tuple(Action),
    )
    assessment = RuleBasedRiskAssessor().assess(observation)
    result = WebConversationPolicy().decide_with_context(
        assessment=assessment,
        observation=observation,
        session=SessionStateV1(session_id="synthetic-renderer-development"),
    )
    return observation, result.assessment, result.outcome


def prepare(output: Path, sources: Path, seeds: Path) -> dict[str, Any]:
    verify_sources(sources)
    items = read_json(seeds)
    seen: set[str] = set()
    groups: dict[str, str] = {}
    for item in items:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", item["id"]):
            raise ValueError("sample ID must be an opaque safe filename")
        if item["id"] in seen or item["split"] not in {"train", "dev", "heldout"}:
            raise ValueError("duplicate sample ID or invalid split")
        seen.add(item["id"])
        if groups.setdefault(item["group"], item["split"]) != item["split"]:
            raise ValueError("a source group crosses partitions")
        visible = item["text"] + json.dumps(item.get("history", []), ensure_ascii=False)
        if UNSUPPORTED.search(visible) or KEY_PATTERN.search(visible):
            raise ValueError("pilot excludes crisis content and credentials")
        observation, assessment, outcome = context(item)
        if outcome.plan.escalation_level in {RiskLevel.L2, RiskLevel.L3, RiskLevel.L4}:
            raise ValueError("first pilot excludes elevated safety decisions")
        messages = LLMResponseRenderer._messages(
            observation=observation,
            assessment=assessment,
            policy=outcome,
            session=SessionStateV1(session_id="synthetic-renderer-development"),
            memory_snippets=(),
        )
        item["messages"] = [dict(message) for message in messages]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "version": VERSION,
        "origin": "NEW_AI_AUTHORED_FICTIONAL_ADULT_DEVELOPMENT",
        "independent_blind_test": False,
        "expert_gold": False,
        "human_approved_samples": 0,
        "input_sha256": digest((output / "inputs.jsonl").read_bytes()),
        "sources": str(sources.resolve()),
        "sources_manifest_sha256": digest((sources / "manifest.json").read_bytes()),
        "seeds_sha256": digest(seeds.read_bytes()),
        "counts": dict(Counter(item["split"] for item in items)),
        "policy_version": CONVERSATION_POLICY_VERSION,
        "prompt_version": RENDER_PROMPT_VERSION,
        "model": "deepseek-flash",
        "documented_model_version": "DeepSeek-V4.1-Flash-1017",
        "immutable_provider_checkpoint": False,
        "endpoint": ENDPOINT,
        "max_calls": 8,
        "max_tokens": 384,
        "input_token_reservation": 16384,
        "input_cny_per_million_upper": "2",
        "output_cny_per_million_upper": "8",
        "pricing_basis": "reviewed_peak_cache_miss_rates_not_billing_guarantee",
        "budget_cny": "1",
        "retries": 0,
        "scope": "NON_CRISIS_RENDERER_DEVELOPMENT_ONLY",
    }
    write_json(output / "plan.json", manifest)
    return manifest


def load_packet(packet: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = read_json(packet / "plan.json")
    if (
        plan["version"] != VERSION
        or digest((packet / "inputs.jsonl").read_bytes()) != plan["input_sha256"]
    ):
        raise ValueError("packet inputs changed")
    sources = Path(plan["sources"])
    verify_sources(sources)
    if digest((sources / "manifest.json").read_bytes()) != plan["sources_manifest_sha256"]:
        raise ValueError("source manifest changed")
    if plan["endpoint"] != ENDPOINT or plan["model"] != "deepseek-flash":
        raise ValueError("pilot endpoint/model is not allowlisted")
    limits = {
        "max_calls": 8,
        "max_tokens": 384,
        "input_token_reservation": 16384,
        "input_cny_per_million_upper": "2",
        "output_cny_per_million_upper": "8",
        "budget_cny": "1",
        "retries": 0,
    }
    if any(plan.get(key) != value for key, value in limits.items()):
        raise ValueError("pilot limits changed; prepare a separately reviewed version")
    if (
        plan["policy_version"] != CONVERSATION_POLICY_VERSION
        or plan["prompt_version"] != RENDER_PROMPT_VERSION
    ):
        raise ValueError("renderer version changed; prepare a new packet")
    items = jsonl(packet / "inputs.jsonl")
    for item in items:
        observation, assessment, outcome = context(item)
        expected = LLMResponseRenderer._messages(
            observation=observation,
            assessment=assessment,
            policy=outcome,
            session=SessionStateV1(session_id="synthetic-renderer-development"),
            memory_snippets=(),
        )
        if item["messages"] != [dict(message) for message in expected]:
            raise ValueError("renderer messages changed; prepare a new packet")
    return plan, items


def candidate_codes(item: dict[str, Any], text: str) -> tuple[str, ...]:
    observation, _, outcome = context(item)
    audit = SafetyAuditor().audit(
        text, risk_level=outcome.plan.escalation_level, required_action=outcome.plan.primary_action
    )
    return tuple(
        dict.fromkeys(
            (
                *audit.violation_codes,
                *response_quality_codes(text, observation=observation, outcome=outcome),
            )
        )
    )


def generate(
    packet: Path,
    output: Path,
    *,
    limit: int = 8,
    allow_paid: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    plan, all_items = load_packet(packet)
    if not allow_paid:
        raise ValueError("paid calls require --allow-paid")
    if not 1 <= limit <= min(8, plan["max_calls"]):
        raise ValueError("first pilot permits at most 8 requests")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is not configured; no request sent")
    items = [item for item in all_items if item["split"] == "train"][:limit]
    if not items:
        raise ValueError("no training-partition pilot items")
    input_rate = Decimal(plan["input_cny_per_million_upper"])
    output_rate = Decimal(plan["output_cny_per_million_upper"])
    budget = min(Decimal(plan["budget_cny"]), Decimal("1"))
    reservation = (
        Decimal(plan["input_token_reservation"]) * input_rate
        + Decimal(plan["max_tokens"]) * output_rate
    ) / 1_000_000
    if reservation <= 0 or reservation * len(items) > budget:
        raise ValueError("pilot reservation exceeds budget")
    for item in items:
        if (
            len(json.dumps(item["messages"], ensure_ascii=False).encode()) + 1024
            > plan["input_token_reservation"]
        ):
            raise ValueError("prompt exceeds conservative input reservation")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    with httpx.Client(timeout=90, trust_env=False, transport=transport) as client:
        for item in items:
            record: dict[str, Any] = {"sample_id": item["id"], "status": "INVALID"}
            payload = {
                "model": plan["model"],
                "messages": item["messages"],
                "temperature": 0.5,
                "max_tokens": plan["max_tokens"],
                "thinking": {"type": "disabled"},
                "stream": False,
            }
            write_json(output / f"{item['id']}.request.json", payload)
            try:
                response = client.post(
                    ENDPOINT, headers={"Authorization": "Bearer " + key}, json=payload
                )
                record["http_status"] = response.status_code
                if response.status_code != 200:
                    record["status"] = "HTTP_ERROR_STOPPED"
                else:
                    body = response.json()
                    choice = body["choices"][0]
                    text = choice["message"].get("content") or ""
                    for field, value in (
                        ("returned_model", body.get("model")),
                        ("system_fingerprint", body.get("system_fingerprint")),
                        ("finish_reason", choice.get("finish_reason")),
                    ):
                        if (
                            isinstance(value, str)
                            and key not in value
                            and not KEY_PATTERN.search(value)
                        ):
                            record[field] = value
                    usage = body.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens")
                    output_tokens = usage.get("completion_tokens")
                    if (
                        type(prompt_tokens) is not int
                        or type(output_tokens) is not int
                        or prompt_tokens < 0
                        or output_tokens < 0
                    ):
                        record["status"] = "MISSING_USAGE_STOPPED"
                    elif (
                        prompt_tokens > plan["input_token_reservation"]
                        or output_tokens > plan["max_tokens"]
                    ):
                        record["status"] = "RESERVATION_EXCEEDED_STOPPED"
                    elif key in text or KEY_PATTERN.search(text):
                        record["status"] = "SECRET_PATTERN_STOPPED"
                    else:
                        record["usage"] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": output_tokens,
                        }
                        record["cost_cny_upper"] = str(
                            (prompt_tokens * input_rate + output_tokens * output_rate) / 1_000_000
                        )
                        record["text"] = text
                        record["text_sha256"] = digest(text.encode())
                        if not text.strip() or choice.get("finish_reason") != "stop":
                            record["status"] = "EMPTY_REFUSED_OR_TRUNCATED"
                        elif choice["message"].get("refusal"):
                            record["status"] = "EMPTY_REFUSED_OR_TRUNCATED"
                        else:
                            record["gate_codes"] = candidate_codes(item, text)
                            record["status"] = (
                                "PENDING_HUMAN_REVIEW"
                                if not record["gate_codes"]
                                else "AUTO_REJECTED"
                            )
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                record["status"] = "CALL_OR_CONTRACT_ERROR_STOPPED"
                record["error_type"] = type(exc).__name__
            write_json(output / f"{item['id']}.response.json", record)
            records.append(record)
            if record["status"].endswith("STOPPED") or record["status"] == "HTTP_ERROR_STOPPED":
                break
    fields = ["sample_id", "candidate_sha256", "decision", "reviewer_id", "review_reference"]
    with (output / "review.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "candidate_sha256": record.get("text_sha256", ""),
                    "decision": "PENDING",
                }
            )
    manifest = {
        "version": VERSION,
        "packet_plan_sha256": digest((packet / "plan.json").read_bytes()),
        "requested": len(items),
        "attempted": len(records),
        "statuses": dict(Counter(record["status"] for record in records)),
        "cost_cny_upper_known": str(
            sum((Decimal(record.get("cost_cny_upper", "0")) for record in records), Decimal("0"))
        ),
        "cost_unknown_calls": sum("cost_cny_upper" not in record for record in records),
        "reserved_cny": str(reservation * len(records)),
        "budget_is_estimate_not_provider_spending_limit": True,
        "responses": {
            record["sample_id"]: digest(
                (output / f"{record['sample_id']}.response.json").read_bytes()
            )
            for record in records
        },
        "key_source": "ENV_ONLY_NOT_PERSISTED",
        "expert_gold": False,
        "training_approved": 0,
        "weights_trained": False,
        "retries": 0,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def export_reviewed(packet: Path, run: Path, reviews: Path, output: Path) -> dict[str, Any]:
    _, items = load_packet(packet)
    by_id = {item["id"]: item for item in items}
    manifest = read_json(run / "manifest.json")
    if manifest["packet_plan_sha256"] != digest((packet / "plan.json").read_bytes()):
        raise ValueError("teacher run belongs to a different packet")
    with reviews.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    seen = set()
    for row in rows:
        identifier = row["sample_id"]
        if identifier in seen or identifier not in manifest["responses"]:
            raise ValueError("duplicate or unknown review sample")
        seen.add(identifier)
        if row["decision"] not in {"PENDING", "APPROVE", "REJECT"}:
            raise ValueError("unknown review decision")
        if row["decision"] != "APPROVE":
            continue
        record_path = run / f"{identifier}.response.json"
        if digest(record_path.read_bytes()) != manifest["responses"][identifier]:
            raise ValueError("teacher record changed")
        record = read_json(record_path)
        if (
            record["status"] != "PENDING_HUMAN_REVIEW"
            or row["candidate_sha256"] != record["text_sha256"]
            or not row["reviewer_id"].strip()
            or not row["review_reference"].strip()
            or by_id[identifier]["split"] != "train"
            or digest(record["text"].encode()) != record["text_sha256"]
            or candidate_codes(by_id[identifier], record["text"])
        ):
            raise ValueError("sample lacks bound, traceable human approval or failed its gate")
        selected.append(
            {
                "messages": [
                    *by_id[identifier]["messages"],
                    {"role": "assistant", "content": record["text"]},
                ]
            }
        )
    if not selected:
        raise ValueError("no human-approved training samples; nothing exported")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "train.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "version": VERSION,
        "samples": len(selected),
        "train_sha256": digest((output / "train.jsonl").read_bytes()),
        "review_sha256": digest(reviews.read_bytes()),
        "expert_gold": False,
        "weights_trained": False,
        "reviewer_identity": "CURATOR_ATTESTATION_NOT_AUTOMATICALLY_VERIFIED",
    }
    write_json(output / "manifest.json", summary)
    return summary


def doctor() -> dict[str, Any]:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu_description = gpu.stdout.strip() if gpu.returncode == 0 else "UNAVAILABLE"
    except FileNotFoundError:
        gpu_description = "UNAVAILABLE"
    return {
        "version": VERSION,
        "api_key_configured": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "gpu": gpu_description,
        "packages": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes")
        },
        "training_status": "NOT_STARTED_REQUIRES_REVIEWED_DATA_AND_SEPARATE_ENVIRONMENT",
        "weights_downloaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    sources = commands.add_parser("freeze-sources")
    sources.add_argument("--output", type=Path, required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--sources", type=Path, required=True)
    prep.add_argument("--seeds", type=Path, default=ROOT / "docs/DISTILLATION_PILOT_SEEDS_V1.json")
    gen = commands.add_parser("generate")
    gen.add_argument("--packet", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--limit", type=int, default=8)
    gen.add_argument("--allow-paid", action="store_true")
    export = commands.add_parser("export")
    for name in ("packet", "run", "reviews", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "freeze-sources":
            result = freeze_sources(args.output)
        elif args.command == "prepare":
            result = prepare(args.output, args.sources, args.seeds)
        elif args.command == "generate":
            result = generate(
                args.packet, args.output, limit=args.limit, allow_paid=args.allow_paid
            )
        else:
            result = export_reviewed(args.packet, args.run, args.reviews, args.output)
    except (ValueError, FileExistsError, httpx.HTTPError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
