"""P0 readiness audit and explicit local Judge orchestration; never install models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ecocompanion.experiments.io import write_json_atomic
from ecocompanion.research.annotation import digest
from ecocompanion.research.chinese_sources import DIMENSIONS, read_jsonl, sha256
from scripts import local_reference_judge as judge
from scripts.external_reference import DEFAULT_ROOT, GATES, validate_packet

VERSION = "p0-readiness-v1.0.0"


def support_audit(packet: Path) -> dict[str, Any]:
    """Count existing references, not synthetic model predictions or future labels."""
    plan = validate_packet(packet)
    if plan["source"] != "psycrisis":
        raise ValueError("P0 Judge route requires the original PsyCrisis task")
    tasks = read_jsonl(packet / "tasks.jsonl")
    refs = read_jsonl(packet / "private/references.jsonl")
    indexed = {t["candidate_id"]: t for t in tasks}
    if (
        len(indexed) != len(tasks)
        or len({r["candidate_id"] for r in refs}) != len(refs)
        or set(indexed) != {r["candidate_id"] for r in refs}
        or not refs
    ):
        raise ValueError("reference/task inventory is not exact")
    group_phases: dict[str, str] = {}
    for ref in refs:
        task = indexed[ref["candidate_id"]]
        if (
            ref["partition"] not in {"calibration", "public_holdout"}
            or ref["input_group"] != task["input_group"]
            or group_phases.setdefault(ref["input_group"], ref["partition"])
            != ref["partition"]
            or set(ref["scores"]) != set(DIMENSIONS)
            or any(type(v) is not int or v not in {0, 1} for v in ref["scores"].values())
        ):
            raise ValueError("invalid references or input group leakage")
    phases = {}
    blockers: list[str] = []
    for phase in ("calibration", "public_holdout"):
        selected = [r for r in refs if r["partition"] == phase]
        dimensions = {}
        for dimension in DIMENSIONS:
            counts = Counter(r["scores"][dimension] for r in selected)
            positive_gap = max(0, GATES["min_positive"] - counts[1])
            negative_gap = max(0, GATES["min_negative"] - counts[0])
            reasons = []
            if positive_gap:
                reasons.append("INSUFFICIENT_POSITIVE_SUPPORT")
            if negative_gap:
                reasons.append("INSUFFICIENT_NEGATIVE_SUPPORT")
            dimensions[dimension] = {
                "reference_positive": counts[1],
                "reference_negative": counts[0],
                "positive_shortfall_under_existing_gate": positive_gap,
                "negative_shortfall_under_existing_gate": negative_gap,
                "support_sufficient": not reasons,
                "reasons": reasons,
                "judge_agreement": "NOT_MEASURED_BY_PREFLIGHT",
            }
            blockers.extend(f"{phase}:{dimension}:{reason}" for reason in reasons)
        phases[phase] = {
            "responses": len(selected),
            "input_groups_NOT_VERIFIED_PERSONS": len({r["input_group"] for r in selected}),
            "dimensions": dimensions,
        }
    return {
        "phases": phases,
        "thresholds": dict(GATES),
        "all_dimension_support_sufficient": not blockers,
        "blockers": blockers,
        "interpretation": "EXISTING_EXPLORATORY_ENGINEERING_GATE_NOT_CLINICAL_SAMPLE_SIZE",
        "shortfall_is_not_a_collection_plan": True,
        "forbidden_repairs": [
            "LOWER_THRESHOLD_AFTER_OBSERVING_SUPPORT",
            "MOVE_OR_DUPLICATE_LABELS_TO_PASS",
            "COUNT_PARAPHRASES_AS_NEW_INDEPENDENT_CASES",
            "PROMOTE_LLM_LABELS_TO_EXPERT_GOLD",
        ],
    }


def readiness(packet: Path, model: str | None, probe_path: Path) -> dict[str, Any]:
    support = support_audit(packet)
    service = judge.probe(probe_path)
    available = [
        m for m in service["models"] if model is not None and m["name"] == model
    ]
    blockers: list[str] = []
    if service["status"] != "LOCAL_SERVICE_AVAILABLE":
        blockers.append(service["status"])
    if model is None:
        blockers.append("EXPLICIT_INSTALLED_MODEL_SELECTION_REQUIRED")
    elif len(available) != 1:
        blockers.append("EXACT_SELECTED_MODEL_NOT_INSTALLED")
    return {
        "version": VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "packet_plan_sha256": sha256((packet / "plan.json").read_bytes()),
        "orchestrator_sha256": sha256(Path(__file__).read_bytes()),
        "local_service": service,
        "selected_model": model,
        "selected_model_digest": available[0]["digest"] if len(available) == 1 else None,
        "calibration_execution_ready": not blockers,
        "execution_blockers": blockers,
        "project_support_blockers": support["blockers"],
        "support": support,
        "judge_inference_calls_in_preflight": 0,
        "target_model_experiment": "NOT_RUN_BY_THIS_TOOL",
        "expert_action_gold": "UNAVAILABLE_NOT_CREATED_BY_EXTERNAL_REFERENCE",
        "project_review": "NOT_RUN",
        "paid_calls": 0,
        "remote_uploads": 0,
        "automatic_downloads": 0,
        "clinical_safety_claim": False,
    }


def execute(
    packet: Path, output: Path, *, model: str | None = None,
    run: bool = False, include_project: bool = False,
) -> dict[str, Any]:
    if run and not model:
        raise ValueError("execution requires an explicit exact installed model")
    if include_project and not run:
        raise ValueError("project review requires an explicit run request")
    # Reserve a new experiment directory even when the audit finds a blocker.
    output.mkdir(parents=True, exist_ok=False)
    preflight = readiness(packet, model, output / "probe.json")
    write_json_atomic(output / "preflight.json", preflight, overwrite=False)
    result: dict[str, Any] = {
        "version": VERSION,
        "preflight_digest": digest(preflight),
        "status": (
            "PREFLIGHT_COMPLETE_BLOCKERS_FOUND"
            if preflight["execution_blockers"] or preflight["project_support_blockers"]
            else "PREFLIGHT_COMPLETE_EXPERIMENT_NOT_RUN"
        ),
        "phases": {},
        "judge_inference_calls": 0,
        "target_model_experiment": "NOT_RUN",
        "project_review": "NOT_REQUESTED",
        "new_project_action_gold": 0,
    }
    if run and not preflight["calibration_execution_ready"]:
        result["status"] = "BLOCKED_LOCAL_RUNTIME_OR_MODEL"
        result["execution_blockers"] = preflight["execution_blockers"]
    elif run:
        config_path = output / "judge-config.json"
        config = judge.freeze(packet, str(model), config_path)
        if config["model_digest"] != preflight["selected_model_digest"]:
            raise ValueError("model changed between preflight and freeze; no inference")
        for phase in ("calibration", "public_holdout"):
            report = judge.run_local(
                packet, config_path, phase, output / phase,
                calibration_run=output / "calibration" if phase == "public_holdout" else None,
            )
            result["phases"][phase] = report
            result["judge_inference_calls"] += report["calls"]
            if report["valid"] != report["expected"]:
                result["status"] = "STOPPED_INCOMPLETE_PHASE_NO_RETRY"
                if include_project:
                    result["project_review"] = "BLOCKED_INCOMPLETE_REFERENCE_PHASE"
                break
        else:
            passed = all(r["gate"]["passed"] for r in result["phases"].values())
            result["status"] = (
                "REFERENCE_PHASES_COMPLETE_GATE_PASSED"
                if passed else "REFERENCE_PHASES_COMPLETE_GATE_FAILED"
            )
            if include_project and passed:
                report = judge.run_local(
                    packet, config_path, "project", output / "project",
                    calibration_run=output / "calibration",
                    holdout_run=output / "public_holdout",
                )
                result["phases"]["project"] = report
                result["judge_inference_calls"] += report["calls"]
                result["project_review"] = report["status"]
            elif include_project:
                result["project_review"] = "BLOCKED_REFERENCE_AGREEMENT_GATE"
    # Validate packet integrity again; never repair historical hashes.
    validate_packet(packet)
    if sha256((packet / "plan.json").read_bytes()) != preflight["packet_plan_sha256"]:
        raise ValueError("packet plan changed during P0 execution")
    if sha256(Path(__file__).read_bytes()) != preflight["orchestrator_sha256"]:
        raise ValueError("P0 orchestrator changed during execution; preserve partial records")
    result["historical_packet_unchanged"] = True
    if include_project and not result["phases"]:
        result["project_review"] = "BLOCKED_NO_REFERENCE_RUNS"
    write_json_atomic(output / "summary.json", result, overwrite=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "run"))
    parser.add_argument("--packet", type=Path, default=DEFAULT_ROOT / "psycrisis")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--include-project", action="store_true")
    args = parser.parse_args()
    if args.command == "run" and not args.model:
        parser.error("run requires --model EXACT_INSTALLED_MODEL:TAG; never auto-select/download")
    result = execute(
        args.packet, args.output, model=args.model,
        run=args.command == "run", include_project=args.include_project,
    )
    print(json.dumps({
        "status": result["status"],
        "judge_inference_calls": result["judge_inference_calls"],
        "report": str(args.output / "summary.json"),
    }))


if __name__ == "__main__":
    main()
