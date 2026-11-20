"""Explicit Ollama-only Judge. No keys, remote endpoint, fallback, or model download."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research.annotation import digest
from ecocompanion.research.chinese_sources import DIMENSIONS, read_jsonl, sha256, write_jsonl
from scripts.external_reference import (
    DEFAULT_ROOT,
    GATES,
    VERSION,
    binary_metrics,
    phase_gate,
    request_payload,
    validate_packet,
)

BASE = "http://127.0.0.1:11434"
PHASES = ("calibration", "public_holdout", "project")


def local_models() -> list[dict[str, Any]]:
    with httpx.Client(timeout=5, trust_env=False, follow_redirects=False) as client:
        response = client.get(BASE + "/api/tags")
        response.raise_for_status()
        return list(response.json()["models"])


def probe(output: Path) -> dict[str, Any]:
    result: dict[str, Any]
    try:
        models = local_models()
        result = {
            "status": "LOCAL_SERVICE_AVAILABLE" if models else "NO_INSTALLED_LOCAL_MODEL",
            "models": [{"name": m["name"], "digest": m["digest"]} for m in models],
        }
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        result = {
            "status": "LOCAL_SERVICE_UNAVAILABLE",
            "error_type": type(exc).__name__,
            "models": [],
        }
    result.update(
        {"endpoint": BASE, "inference_calls": 0, "remote_uploads": 0, "keys_read": False}
    )
    write_json_atomic(output, result, overwrite=False)
    return result


def freeze(packet: Path, model: str, output: Path) -> dict[str, Any]:
    plan = validate_packet(packet)
    if plan["source"] != "psycrisis":
        raise ValueError("VERA branching/version equivalence must be audited before live judging")
    available = [m for m in local_models() if m["name"] == model]
    if len(available) != 1:
        raise ValueError("explicit exact installed model name required; never pull/fallback")
    config = {
        "version": VERSION,
        "packet_plan_sha256": sha256((packet / "plan.json").read_bytes()),
        "provider": "OLLAMA_LOCAL_ONLY",
        "endpoint": BASE,
        "model": model,
        "model_digest": available[0]["digest"],
        "retry_budget": 0,
        "max_calls_per_task": 1,
        "fallback": "FORBIDDEN",
        "options": request_payload({"user": "", "reply": ""})["options"],
        "gates": GATES,
        "request_projection": "ONLY_USER_AND_REPLY_TEXT_NO_LABELS_IDS_GROUPS_OR_POLICY",
        "privacy": "LOOPBACK_ONLY_NO_EXTERNAL_UPLOAD",
        "reproducibility": "FROZEN_INPUT_AND_TRACE_NOT_DETERMINISTIC_REGENERATION_PROMISE",
        "independent_model_family": "NOT_VERIFIED_LOCAL_MODEL_MAY_SHARE_ORIGINS",
    }
    write_json_atomic(output, config, overwrite=False)
    return config


def validate_config(packet: Path, path: Path) -> dict[str, Any]:
    plan = validate_packet(packet)
    config = read_json_object(path)
    if (
        plan["source"] != "psycrisis"
        or config["version"] != VERSION
        or config["packet_plan_sha256"] != sha256((packet / "plan.json").read_bytes())
        or config["endpoint"] != BASE
        or config["provider"] != "OLLAMA_LOCAL_ONLY"
        or config["retry_budget"] != 0
        or config["max_calls_per_task"] != 1
        or config["fallback"] != "FORBIDDEN"
        or config["gates"] != GATES
        or config["options"] != request_payload({"user": "", "reply": ""})["options"]
    ):
        raise ValueError("Judge identity/prompt/budget changed; create a new version")
    return config


def phase_tasks(packet: Path, phase: str) -> list[dict[str, Any]]:
    validate_packet(packet)
    if phase not in PHASES:
        raise ValueError("unsupported phase")
    if phase == "project":
        return read_jsonl(packet / "project-tasks.jsonl")
    refs = read_jsonl(packet / "private/references.jsonl")
    ids = {r["candidate_id"] for r in refs if r["partition"] == phase}
    return [t for t in read_jsonl(packet / "tasks.jsonl") if t["candidate_id"] in ids]


def parse_verdict(text: str) -> dict[str, Any]:
    verdict = json.loads(text)
    if (
        not isinstance(verdict, dict)
        or set(verdict) != {"scores", "reasons", "human_review_flags"}
        or not isinstance(verdict["scores"], dict)
        or set(verdict["scores"]) != set(DIMENSIONS)
        or any(type(v) is not int or v not in {0, 1} for v in verdict["scores"].values())
        or not isinstance(verdict["reasons"], dict)
        or set(verdict["reasons"]) != set(DIMENSIONS)
        or any(not isinstance(v, str) or not v.strip() for v in verdict["reasons"].values())
        or not isinstance(verdict["human_review_flags"], list)
        or any(
            not isinstance(v, str) or not v.strip() for v in verdict["human_review_flags"]
        )
    ):
        raise ValueError("Judge output violates five-dimensional nonexpert contract")
    return dict(verdict)


def visible_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Never persist optional thinking fields or arbitrary server metadata."""
    if not isinstance(raw, dict) or not isinstance(raw.get("message"), dict):
        raise ValueError("unreviewed local response structure")
    return {
        **{
            k: raw[k] for k in (
                "model", "done", "done_reason", "prompt_eval_count", "eval_count",
            ) if k in raw
        },
        "message": {"content": raw.get("message", {}).get("content", "")},
    }


def agreement_ci(
    refs: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], *, resamples: int = 1000
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        groups.setdefault(ref["input_group"], []).append(ref)
    values = list(groups.values())
    rng = np.random.default_rng(20261021)
    ci: dict[str, Any] = {}
    # Bootstrap entire input groups, retaining both source responses and failures.
    for dimension in DIMENSIONS:
        counts = []
        for group in values:
            m = binary_metrics(group, predictions)[dimension]
            counts.append(
                [
                    m["tp"], m["fp"], m["fn_including_missing_positive"],
                    m["tn"], m["expected"],
                ]
            )
        matrix = np.array(counts, dtype=np.int64)
        accuracy, f1 = [], []
        for _ in range(resamples):
            tp, fp, fn, tn, n = matrix[
                rng.integers(0, len(values), len(values))
            ].sum(axis=0).tolist()
            accuracy.append((tp + tn) / n)
            if 2 * tp + fp + fn:
                f1.append(2 * tp / (2 * tp + fp + fn))
        ci[dimension] = {
            "accuracy_ci95": [float(x) for x in np.quantile(accuracy, [0.025, 0.975])],
            "positive_f1_ci95": [float(x) for x in np.quantile(f1, [0.025, 0.975])]
            if f1 else None,
            "f1_defined_resamples": len(f1),
        }
    return {
        "dimensions": ci,
        "clusters": len(groups),
        "unit": "NORMALIZED_INPUT_GROUP_NOT_VERIFIED_PERSON",
        "resamples": resamples,
        "seed": 20261021,
        "limitation": "DESCRIPTIVE_PUBLIC_SOURCE_PERSON_TEMPLATE_DEPENDENCE_UNKNOWN",
    }


def score_run(packet: Path, config_path: Path, run: Path) -> dict[str, Any]:
    config = validate_config(packet, config_path)
    started = read_json_object(run / "started.json")
    if started["config_sha256"] != sha256(config_path.read_bytes()):
        raise ValueError("run/Judge config differs")
    phase = started["phase"]
    tasks = phase_tasks(packet, phase)
    indexed: dict[str, dict[str, Any]] = {}
    flags = []
    actual_calls = token_in = token_out = 0
    for task in tasks:
        path = run / "records" / f"{task['candidate_id']}.json"
        if not path.exists():
            continue
        record = read_json_object(path)
        payload = {**request_payload(task), "model": config["model"]}
        if (
            record["task_sha256"] != digest(task)
            or record["request"] != payload
            or record["config_sha256"] != started["config_sha256"]
            or record["kind"] != "llm"
            or record["candidate_id"] != task["candidate_id"]
            or type(record["calls"]) is not int
            or record["calls"] not in {0, 1}
            or record["status"] not in {"ok", "error"}
        ):
            raise ValueError("record identity or full request changed")
        actual_calls += record["calls"]
        if record["status"] == "ok":
            raw = record["raw_response"]
            if (
                raw.get("model") != config["model"]
                or raw.get("done") is not True
                or record.get("model_digest_verified_before_call") != config["model_digest"]
                or record["calls"] != 1
                or any(
                    type(raw.get(k)) is not int or raw[k] < 0
                    for k in ("prompt_eval_count", "eval_count")
                )
            ):
                raise ValueError("model switched or incomplete generation")
            parsed = parse_verdict(raw["message"]["content"])
            indexed[task["candidate_id"]] = parsed
            token_in += raw["prompt_eval_count"]
            token_out += raw["eval_count"]
            if parsed["human_review_flags"]:
                flags.append(
                    {
                        "candidate_id": task["candidate_id"],
                        "flags": parsed["human_review_flags"],
                        "status": "PENDING_QUALIFIED_REVIEW_NOT_CONFIRMED_VIOLATION",
                    }
                )
    expected_paths = {f"{t['candidate_id']}.json" for t in tasks}
    if {p.name for p in (run / "records").glob("*.json")} - expected_paths:
        raise ValueError("unknown/extra Judge outputs")
    if started["task_set_digest"] != digest(tasks):
        raise ValueError("phase task inventory changed")
    result: dict[str, Any] = {
        "version": VERSION,
        "phase": phase,
        "status": "COMPLETE_AUTOMATED_REVIEW_NOT_EXPERT_GOLD"
        if len(indexed) == len(tasks) else "PARTIAL_OR_FAILED_AUTOMATED_REVIEW",
        "config_sha256": started["config_sha256"],
        "expected": len(tasks),
        "valid": len(indexed),
        "failed_or_missing": len(tasks) - len(indexed),
        "calls": actual_calls,
        "prompt_tokens_successful_calls": token_in,
        "completion_tokens_successful_calls": token_out,
        "failed_call_tokens": "UNKNOWN_NOT_ASSUMED_ZERO",
        "api_cost_cny": 0,
        "local_compute_cost": "NOT_MEASURED",
        "human_review_queue": flags,
        "new_project_action_gold": 0,
        "expert_action_metrics": None,
        "safety_total": "NOT_COMPUTED",
        "policy_ranking": "NOT_COMPUTED",
        "records_digest": digest(
            {
                p.name: sha256(p.read_bytes())
                for p in sorted((run / "records").glob("*.json"))
            }
        ),
    }
    if phase == "project":
        result["feature_counts_NOT_CORRECTNESS"] = {
            d: sum(r["scores"][d] for r in indexed.values()) for d in DIMENSIONS
        }
    else:
        refs = [
            r for r in read_jsonl(packet / "private/references.jsonl")
            if r["partition"] == phase
        ]
        metrics = binary_metrics(refs, indexed)
        result.update(
            {
                "dimensions": metrics,
                "descriptive_group_bootstrap": agreement_ci(refs, indexed),
                "gate": phase_gate(metrics),
                "scope": "UPSTREAM_HUMAN_RATING_AGREEMENT_NOT_CLINICAL_CORRECTNESS",
            }
        )
    return result


def prerequisite(
    packet: Path, config: Path, prior_run: Path | None, phase: str
) -> dict[str, Any]:
    if prior_run is None:
        raise ValueError(f"{phase} run is required")
    report = score_run(packet, config, prior_run)
    saved = read_json_object(prior_run / "report.json")
    if report != saved or report["phase"] != phase or report["valid"] != report["expected"]:
        raise ValueError("prior phase must be complete, frozen and recomputable")
    return report


def run_local(
    packet: Path, config_path: Path, phase: str, output: Path,
    calibration_run: Path | None = None, holdout_run: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new run; no implicit retries or overwrite")
    config = validate_config(packet, config_path)
    if phase in {"public_holdout", "project"}:
        calibration = prerequisite(packet, config_path, calibration_run, "calibration")
        if phase == "project":
            heldout = prerequisite(packet, config_path, holdout_run, "public_holdout")
            if not calibration["gate"]["passed"] or not heldout["gate"]["passed"]:
                raise ValueError("project review blocked by upstream agreement gates")
    installed = [m for m in local_models() if m["name"] == config["model"]]
    if len(installed) != 1 or installed[0]["digest"] != config["model_digest"]:
        raise ValueError("local model identity changed or missing; no fallback")
    tasks = phase_tasks(packet, phase)
    config_hash = sha256(config_path.read_bytes())
    write_json_atomic(
        output / "started.json",
        {
            "config_sha256": config_hash,
            "phase": phase,
            "task_set_digest": digest(tasks),
            "max_calls": len(tasks),
            "max_retries": 0,
            "cost_policy": "LOCAL_ONLY_ZERO_REMOTE_API_CALLS",
        },
        overwrite=False,
    )
    for task in tasks:
        payload = {**request_payload(task), "model": config["model"]}
        record: dict[str, Any] = {
            "candidate_id": task["candidate_id"],
            "task_sha256": digest(task),
            "config_sha256": config_hash,
            "kind": "llm",
            "request": payload,
            "calls": 0,
        }
        start = time.monotonic()
        try:
            models = [m for m in local_models() if m["name"] == config["model"]]
            if len(models) != 1 or models[0]["digest"] != config["model_digest"]:
                raise ValueError("MODEL_DIGEST_DRIFT_NO_INFERENCE")
            record["model_digest_verified_before_call"] = models[0]["digest"]
            with httpx.Client(
                timeout=180, trust_env=False, follow_redirects=False
            ) as client:
                record["calls"] = 1
                response = client.post(BASE + "/api/chat", json=payload)
                response.raise_for_status()
                raw = visible_response(response.json())
            record["raw_response"] = raw
            if raw.get("model") != config["model"] or raw.get("done") is not True:
                raise ValueError("MODEL_IDENTITY_OR_COMPLETENESS")
            if any(
                type(raw.get(k)) is not int or raw[k] < 0
                for k in ("prompt_eval_count", "eval_count")
            ):
                raise ValueError("TOKEN_ACCOUNTING_UNAVAILABLE")
            parse_verdict(raw["message"]["content"])
            record["status"] = "ok"
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            record.update({"status": "error", "error_type": type(exc).__name__})
        record["latency_ms"] = (time.monotonic() - start) * 1000
        write_json_atomic(
            output / "records" / f"{task['candidate_id']}.json", record, overwrite=False
        )
        if record["status"] == "error":
            # Stop on failure; unattempted tasks remain in the expected denominator.
            break
    report = score_run(packet, config_path, output)
    write_json_atomic(output / "report.json", report, overwrite=False)
    return report


def export_requests(packet: Path, output: Path) -> dict[str, Any]:
    """Calibration only. No holdout or project release merely by preparing a packet."""
    tasks = phase_tasks(packet, "calibration")
    write_jsonl(
        output,
        [
            {
                "candidate_id": t["candidate_id"],
                "task_sha256": digest(t),
                "request_template": request_payload(t),
                "execution": "NOT_RUN_MODEL_UNBOUND_NO_UPLOAD_AUTHORIZED",
            }
            for t in tasks
        ],
    )
    return {"status": "CALIBRATION_REQUESTS_FROZEN_NOT_EXECUTED", "requests": len(tasks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("probe")
    check.add_argument("--output", type=Path, default=DEFAULT_ROOT / "local-judge-probe.json")
    bind = commands.add_parser("freeze")
    bind.add_argument("--packet", type=Path, default=DEFAULT_ROOT / "psycrisis")
    bind.add_argument("--model", required=True)
    bind.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export-calibration")
    export.add_argument("--packet", type=Path, default=DEFAULT_ROOT / "psycrisis")
    export.add_argument("--output", type=Path, default=DEFAULT_ROOT / "calibration-requests.jsonl")
    execute = commands.add_parser("run")
    execute.add_argument("--packet", type=Path, default=DEFAULT_ROOT / "psycrisis")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--phase", choices=PHASES, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--calibration-run", type=Path)
    execute.add_argument("--holdout-run", type=Path)
    score = commands.add_parser("score")
    score.add_argument("--packet", type=Path, default=DEFAULT_ROOT / "psycrisis")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        result = probe(args.output)
    elif args.command == "freeze":
        result = freeze(args.packet, args.model, args.output)
    elif args.command == "export-calibration":
        result = export_requests(args.packet, args.output)
    elif args.command == "score":
        result = score_run(args.packet, args.config, args.run)
        write_json_atomic(args.output, result, overwrite=False)
    else:
        result = run_local(
            args.packet, args.config, args.phase, args.output,
            args.calibration_run, args.holdout_run,
        )
    print(json.dumps({k: v for k, v in result.items() if k in {"status", "phase", "requests"}}))


if __name__ == "__main__":
    main()
