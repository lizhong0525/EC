"""Input-only Chinese diagnostics and a pending prospective expert annotation packet."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ecocompanion.domain.enums import Action, ActorRole, Track
from ecocompanion.domain.schemas import (
    ConversationMessage,
    CurrentObservation,
    ObservationV1,
    VisibleContext,
    VisibleProfile,
)
from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research import VERSION as RESEARCH_VERSION
from ecocompanion.research.annotation import AnnotationStore, dataset_digest, digest
from ecocompanion.research.chinese_sources import (
    DIMENSIONS,
    VERSION,
    audit_chinese,
    load_cp,
    load_psy,
    normalized,
    read_jsonl,
    sha256,
    verify_chinese,
    write_jsonl,
)
from ecocompanion.research.contracts import Prediction, Protocol, Provenance, ResearchCase
from ecocompanion.research.cradle import PROJECT_POLICIES
from ecocompanion.research.fingerprint import runtime_manifest, source_fingerprint
from ecocompanion.research.identity import audit_visible_payload
from ecocompanion.research.measurement import contract_issues


def select_cp(cases: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Length-quartile pilot, grouped by exact normalized opening, without risk inference."""
    representatives: dict[str, dict[str, Any]] = {}
    for case in sorted(cases, key=lambda c: (digest(c["pairs"]), c["row"])):
        representatives.setdefault(normalized(case["pairs"][0][0]), case)
    pool = sorted(representatives.values(), key=lambda c: (len(c["pairs"]), digest(c["pairs"])))
    if count < 1 or count > len(pool):
        raise ValueError("pilot count exceeds available unique input groups")
    quartiles = [
        sorted(
            pool[i * len(pool) // 4 : (i + 1) * len(pool) // 4], key=lambda c: digest(c["pairs"])
        )
        for i in range(4)
    ]
    chosen = []
    for index in range(count):
        bucket = quartiles[index % 4]
        if not bucket:
            bucket = next(b for b in quartiles if b)
        chosen.append(bucket.pop(0))
    return chosen


def observations_for(case_id: str, utterances: list[str]) -> tuple[ObservationV1, ...]:
    observations = []
    history: list[ConversationMessage] = []
    for index, text in enumerate(utterances):
        observation_id = uuid4().hex
        observation = ObservationV1(
            benchmark_version=RESEARCH_VERSION,
            track=Track.A,
            episode_id=case_id,
            step_id=index,
            simulated_time=f"turn:{index}",
            visible_profile=VisibleProfile(age_group="unspecified", language="zh-CN"),
            current_observation=CurrentObservation(observation_id=observation_id, message=text),
            visible_context=VisibleContext(conversation_history=tuple(history)),
            allowed_primary_actions=tuple(Action),
        )
        audit_visible_payload(observation.model_dump_json())
        observations.append(observation)
        history.append(
            ConversationMessage(
                message_id=observation_id,
                role=ActorRole.STUDENT_SIMULATOR,
                content=text,
                simulated_time=f"turn:{index}",
            )
        )
    return tuple(observations)


def freeze_chinese(snapshot: Path, output: Path, *, pilot_count: int = 20) -> dict[str, Any]:
    manifest = verify_chinese(snapshot)
    audit = audit_chinese(snapshot)
    if output.exists():
        raise FileExistsError("choose a new study directory")
    inputs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    replies: list[dict[str, Any]] = []
    if manifest["source"] == "psycrisis":
        prompts, _ = load_psy(snapshot)
        records = [
            {"row": p["row"], "utterances": [p["text"]], "category": p["category"]} for p in prompts
        ]
    else:
        records = [
            {"row": c["row"], "utterances": [p[0] for p in c["pairs"]], "pairs": c["pairs"]}
            for c in select_cp(load_cp(snapshot), pilot_count)
        ]
    for record in records:
        case_id = uuid4().hex
        observations = observations_for(case_id, record["utterances"])
        inputs.extend(o.model_dump(mode="json") for o in observations)
        metadata.append(
            {
                "case_id": case_id,
                "source_row": record["row"],
                "category": record.get("category"),
                "review_status": "PENDING_HUMAN_REVIEW",
            }
        )
        if "pairs" in record:
            replies.append({"case_id": case_id, "source_pairs": record["pairs"]})
    write_jsonl(output / "inputs.jsonl", inputs)
    write_jsonl(output / "private" / "metadata.jsonl", metadata)
    write_jsonl(output / "private" / "source-pairs.jsonl", replies)
    write_json_atomic(output / "data-audit.json", audit, overwrite=False)
    runtime = runtime_manifest()
    write_json_atomic(output / "runtime.json", runtime, overwrite=False)
    plan = {
        "version": VERSION,
        "source": manifest["source"],
        "revision": manifest["revision"],
        "created_at": datetime.now(UTC).isoformat(),
        "study_status": "EXPLORATORY_LOCAL_RULE_DIAGNOSTIC_NOT_PREREGISTERED",
        "cases": len(metadata),
        "nodes": len(inputs),
        "source_snapshot_sha256": sha256((snapshot / "snapshot.json").read_bytes()),
        "artifacts": {
            name: sha256((output / name).read_bytes())
            for name in (
                "inputs.jsonl",
                "private/metadata.jsonl",
                "private/source-pairs.jsonl",
                "data-audit.json",
                "runtime.json",
            )
        },
        "source_digest": source_fingerprint(),
        "runtime_digest": digest(runtime),
        "policies": list(PROJECT_POLICIES),
        "context": "CURRENT_AND_PRIOR_USER_MESSAGES_ONLY_NOT_FULL_COUNSELING_DIALOGUE",
        "source_assistant_replies": "HIDDEN_NOT_ACTION_GOLD_NOT_GENERATED_BY_CURRENT_POLICY",
        "node_time": "TURN_ORDER_NOT_ELAPSED_DAYS",
        "split": "PUBLIC_DEVELOPMENT_NOT_BLIND",
        "sampling": "ALL_SOURCE_PROMPTS"
        if manifest["source"] == "psycrisis"
        else f"{pilot_count}_LENGTH_QUARTILE_PILOT_NO_RISK_OR_TOPIC_STRATIFICATION",
        "person_post_template_independence": "UNKNOWN_NO_INDEPENDENT_CASE_CI",
        "expert_action_norms": "UNAVAILABLE_NO_INPUT_MATCHED_GOLD",
        "reply_safety_scores": "UNAVAILABLE_REQUIRES_HUMAN_REVIEW",
        "category_detection_accuracy": "NOT_SCORED_NO_VALIDATED_ACTION_CATEGORY_MAPPING",
        "simulation_recovery": "UNAVAILABLE_NO_SIMULATION_TRUTH",
        "model_identity": "NONE_LOCAL_RULES_ONLY",
        "max_calls": 0,
        "max_retries": 0,
        "max_cost_cny": 0,
        "failure_policy": "RETAIN_ALL_EXPECTED_NODES_RAW_OUTPUTS_AND_CONTRACT_FAILURES",
        "statistics": "ACTION_EVIDENCE_CONTRACT_COUNTS_ONLY_NOT_CLINICAL_ACCURACY",
        "independent_blind": False,
        "gold_created": False,
    }
    write_json_atomic(output / "plan.json", plan, overwrite=False)
    return plan


def validate_chinese_plan(packet: Path) -> dict[str, Any]:
    plan = read_json_object(packet / "plan.json")
    if plan.get("version") != VERSION or plan.get("policies") != list(PROJECT_POLICIES):
        raise ValueError("unsupported frozen protocol")
    if set(plan["artifacts"]) != {
        "inputs.jsonl",
        "private/metadata.jsonl",
        "private/source-pairs.jsonl",
        "data-audit.json",
        "runtime.json",
    }:
        raise ValueError("frozen artifact inventory changed")
    for name, expected in plan["artifacts"].items():
        if (
            name
            not in {
                "inputs.jsonl",
                "private/metadata.jsonl",
                "private/source-pairs.jsonl",
                "data-audit.json",
                "runtime.json",
            }
            or sha256((packet / name).read_bytes()) != expected
        ):
            raise ValueError("frozen input/reference artifact changed")
    if plan["source_digest"] != source_fingerprint() or plan["runtime_digest"] != digest(
        runtime_manifest()
    ):
        raise ValueError("source/runtime drift; create a new plan, never update old hashes")
    return plan


def chinese_records(packet: Path) -> Iterator[dict[str, Any]]:
    plan_hash = sha256((packet / "plan.json").read_bytes())
    policies = {name: factory() for name, factory in PROJECT_POLICIES.items()}
    for raw in read_jsonl(packet / "inputs.jsonl"):
        observation = ObservationV1.model_validate(raw)
        audit_visible_payload(observation.model_dump_json())
        for name, policy in policies.items():
            try:
                decision = policy.decide(observation)
                prediction = Prediction(
                    policy=name,
                    case_id=observation.episode_id,
                    step_id=observation.step_id,
                    status="ok",
                    decision=decision,
                    cost_cny=0,
                    protocol_digest=plan_hash,
                )
                issues = contract_issues(prediction, observation)
            except Exception as exc:
                prediction = Prediction(
                    policy=name,
                    case_id=observation.episode_id,
                    step_id=observation.step_id,
                    status="error",
                    error_code=type(exc).__name__,
                    cost_cny=0,
                    protocol_digest=plan_hash,
                )
                issues = ("EXECUTION_ERROR",)
            yield {
                "prediction": prediction.model_dump(mode="json"),
                "observation_digest": digest(raw),
                "contract_issues": list(issues),
            }


def chinese_report(packet: Path, run: Path, output: Path) -> dict[str, Any]:
    plan = validate_chinese_plan(packet)
    records = read_jsonl(run / "predictions.jsonl")
    replayed = list(chinese_records(packet))
    if records != replayed or len(records) != plan["nodes"] * len(plan["policies"]):
        raise ValueError("missing/tampered/nonreproducible output; do not report COMPLETE")
    metadata = {r["case_id"]: r for r in read_jsonl(packet / "private" / "metadata.jsonl")}
    policies = {}
    for name in plan["policies"]:
        selected = [r for r in records if r["prediction"]["policy"] == name]
        predictions = [r["prediction"] for r in selected]
        ok = [p for p in predictions if p["status"] == "ok"]
        categories = {}
        for category in sorted({m["category"] for m in metadata.values() if m["category"]}):
            subset = [p for p in ok if metadata[p["case_id"]]["category"] == category]
            categories[category] = {
                "nodes": len(subset),
                "primary_actions": dict(
                    Counter(p["decision"]["decision"]["primary_action"] for p in subset)
                ),
            }
        policies[name] = {
            "expected_nodes": plan["nodes"],
            "recorded_nodes": len(predictions),
            "successful_outputs": len(ok),
            "execution_errors": len(predictions) - len(ok),
            "contract_failure_nodes": sum(bool(r["contract_issues"]) for r in selected),
            "contract_issues": dict(
                Counter(issue for r in selected for issue in r["contract_issues"])
            ),
            "primary_actions": dict(
                Counter(p["decision"]["decision"]["primary_action"] for p in ok)
            ),
            "nodes_with_evidence": sum(bool(p["decision"]["evidence"]) for p in ok),
            "per_source_category_action_counts_NOT_ACCURACY": categories,
            "expert_action_norms": "UNAVAILABLE",
            "reply_safety": "PENDING_ANONYMOUS_HUMAN_REVIEW",
            "simulation_recovery": "UNAVAILABLE",
        }
    report = {
        "version": VERSION,
        "source": plan["source"],
        "revision": plan["revision"],
        "status": "COMPLETE_LOCAL_RULE_DIAGNOSTIC",
        "cases": plan["cases"],
        "nodes": plan["nodes"],
        "stage_replay": {"status": "PASS", "verified_records": len(records)},
        "policies": policies,
        "plan_sha256": sha256((packet / "plan.json").read_bytes()),
        "predictions_sha256": sha256((run / "predictions.jsonl").read_bytes()),
        "model_calls": 0,
        "tokens": 0,
        "cost_cny": 0,
        "real_model_experiment": "NOT_RUN",
        "upstream_five_dimension_scoring": "NOT_RUN_NO_AUTOMATIC_SUBJECTIVE_SCORES",
        "clinical_or_action_correctness": "NOT_ESTABLISHED",
        "source_person_independence": "UNKNOWN",
    }
    write_json_atomic(output, report, overwrite=False)
    return report


def run_chinese(packet: Path, output: Path) -> dict[str, Any]:
    validate_chinese_plan(packet)
    if output.exists():
        raise FileExistsError("choose a new run; history is immutable")
    write_json_atomic(
        output / "started.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256((packet / "plan.json").read_bytes()),
        },
        overwrite=False,
    )
    write_jsonl(output / "predictions.jsonl", list(chinese_records(packet)))
    report = chinese_report(packet, output, output / "report.json")
    write_json_atomic(
        output / "completed.json",
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "status": report["status"],
            "report_sha256": sha256((output / "report.json").read_bytes()),
        },
        overwrite=False,
    )
    return report


def annotation_packet(packet: Path, output: Path, manual: Path) -> dict[str, Any]:
    plan = validate_chinese_plan(packet)
    if plan["source"] != "cpsycoun":
        raise ValueError("real-source PsyCrisis prompts cannot enter synthetic-only ResearchCase")
    if output.exists():
        raise FileExistsError("choose a new annotation packet")
    groups: dict[str, list[ObservationV1]] = {}
    for raw in read_jsonl(packet / "inputs.jsonl"):
        observation = ObservationV1.model_validate(raw)
        groups.setdefault(observation.episode_id, []).append(observation)
    metadata = {r["case_id"]: r for r in read_jsonl(packet / "private" / "metadata.jsonl")}
    cases = tuple(
        ResearchCase(
            case_id=case_id,
            observations=tuple(observations),
            provenance=Provenance(
                origin="automatic_draft",
                source_id=f"CPsyCounD:{plan['revision']}:row:{metadata[case_id]['source_row']}",
                author_id="upstream-report-reconstruction-not-independent-expert-author",
                person_group="UNKNOWN_ALL_CPSYCOUN",
                plot_group="UNKNOWN_ALL_CPSYCOUN",
                template_group="UNKNOWN_ALL_CPSYCOUN",
                source_group=f"CPsyCounD:{plan['revision']}",
            ),
            split="development",
        )
        for case_id, observations in groups.items()
    )
    manual_bytes = manual.read_bytes()
    store, tokens = AnnotationStore.create(
        output / "private" / "annotations.sqlite3",
        cases,
        manual_digest=sha256(manual_bytes),
    )
    write_json_atomic(output / "private" / "access.json", tokens, overwrite=False)
    write_json_atomic(
        output / "private" / "manual-snapshot.json",
        {"sha256": sha256(manual_bytes), "text": manual_bytes.decode("utf-8")},
        overwrite=False,
    )
    # Keep the existing Gold scorer usable after real annotations, without inventing truth.
    from ecocompanion.research.workflows import LOCAL_POLICIES

    protocol = Protocol(
        dataset_digest=dataset_digest(cases),
        policy_names=tuple(name for name in PROJECT_POLICIES if name in LOCAL_POLICIES),
        model_identity="no-model",
        prompt_digest=digest({"context": plan["context"], "source_plan": plan}),
        source_digest=plan["source_digest"],
        runtime_digest=plan["runtime_digest"],
    )
    write_json_atomic(output / "protocol.json", protocol, overwrite=False)
    result = {
        "version": VERSION,
        "status": "PENDING_SOURCE_AND_EXPERT_REVIEW",
        "cases": len(cases),
        "nodes": sum(len(c.observations) for c in cases),
        "source_plan_sha256": sha256((packet / "plan.json").read_bytes()),
        "dataset_context": plan["context"],
        "input_redaction_semantics": "PRIOR_COUNSELOR_TEXT_EXCLUDED_MAY_REDUCE_COHERENCE",
        "clinical_originality": "RECONSTRUCTED_NOT_ACTUAL_CLINICAL_RECORD",
        "source_person_or_template_groups": "UNKNOWN_CONSERVATIVE_SINGLE_GROUP",
        "age_group": "UNSPECIFIED_NOT_ASSUMED_MINOR",
        "label_status": "UNLABELED",
        "gold_status": "NOT_FROZEN",
        "manual_sha256": sha256(manual_bytes),
        "annotation_audit": store.audit(),
        "source_privacy_review": "PENDING_BEFORE_DISTRIBUTING_TO_EXPERTS",
        "model_replies_visible_to_norm_annotators": False,
        "no_simulation_truth_created": True,
        "compatible_v2_local_policies": list(protocol.policy_names),
        "statistics": "NO_VERIFIED_PERSON_OR_TEMPLATE_INDEPENDENCE_NO_POPULATION_CI_CLAIMS",
    }
    write_json_atomic(output / "annotation-manifest.json", result, overwrite=False)
    return result


def chinese_reply_packet(packet: Path, run: Path, output: Path) -> dict[str, Any]:
    """Reuse the existing blind reply schema without mislabelling real inputs as synthetic."""
    plan = validate_chinese_plan(packet)
    records = read_jsonl(run / "predictions.jsonl")
    if records != list(chinese_records(packet)):
        raise ValueError("reply review requires genuine, replayable frozen outputs")
    if plan["source"] == "cpsycoun":
        raise ValueError("norm annotators must not be mixed with model reply reviewers")
    # This path intentionally does not instantiate ResearchCase for real-source prompts.
    import random

    if output.exists():
        raise FileExistsError("choose a new reply review directory")
    observations = {(o["episode_id"], o["step_id"]): o for o in read_jsonl(packet / "inputs.jsonl")}
    tasks, mapping = [], []
    for record in records:
        prediction = Prediction.model_validate(record["prediction"])
        if not prediction.decision:
            continue
        candidate_id = uuid4().hex
        tasks.append(
            {
                "candidate_id": candidate_id,
                "rubric_version": "reply-v2",
                "observation": observations[(prediction.case_id, prediction.step_id)],
                "reply": prediction.decision.response.model_dump(mode="json"),
            }
        )
        mapping.append(
            {
                "candidate_id": candidate_id,
                "policy": prediction.policy,
                "case_id": prediction.case_id,
                "step_id": prediction.step_id,
                "repeat": prediction.repeat,
                "status": prediction.status,
            }
        )
    random.Random(20261021).shuffle(tasks)
    write_json_atomic(output / "private" / "mapping.json", {"mapping": mapping}, overwrite=False)
    write_json_atomic(
        output / "reply-tasks.json",
        {
            "rubric_version": "reply-v2",
            "status": "PENDING_EXPERT_REVIEW",
            "tasks": tasks,
            "task_digest": digest(tasks),
        },
        overwrite=False,
    )
    return {"tasks": len(tasks), "status": "PENDING_EXPERT_REVIEW_NO_AUTOMATIC_SCORES"}


def score_calibration(packet: Path, predictions: Path, output: Path) -> dict[str, Any]:
    """Compare external binary judgments to upstream ratings, not clinical safety Gold."""
    manifest = read_json_object(packet / "manifest.json")
    if (
        sha256((packet / "tasks.jsonl").read_bytes()) != manifest["tasks_sha256"]
        or sha256((packet / "private/references.jsonl").read_bytes())
        != manifest["references_sha256"]
    ):
        raise ValueError("frozen calibration artifacts changed")
    tasks = read_jsonl(packet / "tasks.jsonl")
    references = read_jsonl(packet / "private/references.jsonl")
    ids = {t["candidate_id"] for t in tasks}
    if (
        len(ids) != len(tasks)
        or len(references) != len(tasks)
        or ids != {r["candidate_id"] for r in references}
    ):
        raise ValueError("duplicate or missing calibration identities")
    task_groups = {t["candidate_id"]: t["input_group"] for t in tasks}
    text_groups: dict[str, str] = {}
    for task in tasks:
        key = normalized(task["user"])
        if key in text_groups and text_groups[key] != task["input_group"]:
            raise ValueError("same input assigned multiple groups")
        text_groups[key] = task["input_group"]
    for ref in references:
        if (
            ref["input_group"] != task_groups[ref["candidate_id"]]
            or ref["partition"] not in {"calibration", "public_holdout"}
            or set(ref["scores"]) != set(DIMENSIONS)
            or any(type(v) is not int or v not in {0, 1} for v in ref["scores"].values())
        ):
            raise ValueError("invalid upstream reference/group schema")
    groups = {
        part: {r["input_group"] for r in references if r["partition"] == part}
        for part in ("calibration", "public_holdout")
    }
    if not all(groups.values()) or groups["calibration"] & groups["public_holdout"]:
        raise ValueError("calibration/holdout input-group leakage")
    loaded = read_jsonl(predictions)
    indexed = {p["candidate_id"]: p for p in loaded}
    if len(indexed) != len(loaded) or set(indexed) - ids:
        raise ValueError("duplicate or unknown candidate prediction")
    dimensions = {}
    from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

    for dimension in DIMENSIONS:
        partitions = {}
        for partition in groups:
            selected = [r for r in references if r["partition"] == partition]
            correct = valid = tp = fp = fn = 0
            a, b = [], []
            for ref in selected:
                gold = ref["scores"][dimension]
                prediction = indexed.get(ref["candidate_id"], {})
                scores = prediction.get("scores")
                success = (
                    prediction.get("status") == "ok"
                    and isinstance(scores, dict)
                    and set(scores) == set(DIMENSIONS)
                    and all(type(x) is int and x in {0, 1} for x in scores.values())
                    and isinstance(prediction.get("judge_id"), str)
                    and bool(prediction["judge_id"].strip())
                    and isinstance(prediction.get("verification_reference"), str)
                    and bool(prediction["verification_reference"].strip())
                )
                if not success:
                    fn += int(gold == 1)
                    continue
                assert isinstance(scores, dict)
                value = scores[dimension]
                valid += 1
                correct += int(value == gold)
                tp += int(value == gold == 1)
                fp += int(value == 1 and gold == 0)
                fn += int(value == 0 and gold == 1)
                a.append(gold)
                b.append(value)
            kappa = float(cohen_kappa_score(a, b)) if a and len(set(a + b)) > 1 else None
            partitions[partition] = {
                "expected": len(selected),
                "valid": valid,
                "failed_or_missing": len(selected) - valid,
                "accuracy_failures_counted_in_denominator": correct / len(selected),
                "positive_precision": tp / (tp + fp) if tp + fp else None,
                "positive_recall_failures_retained": tp / (tp + fn) if tp + fn else None,
                "positive_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "kappa_valid_pairs_only_NOT_FULL_COVERAGE": (
                    kappa if kappa is not None and np.isfinite(kappa) else None
                ),
                "kappa_applicability": "REQUIRES_LABEL_VARIATION_AND_COMPLETE_COVERAGE_FOR_CLAIMS",
            }
        dimensions[dimension] = partitions
    report = {
        "version": VERSION,
        "status": "UPSTREAM_RATING_AGREEMENT_ONLY_NOT_CLINICAL_CERTIFICATION",
        "dimensions": dimensions,
        "prediction_sha256": sha256(predictions.read_bytes()),
        "manifest_sha256": sha256((packet / "manifest.json").read_bytes()),
        "source_rater_lineage": "NOT_INDEPENDENTLY_VERIFIED",
        "imported_judge_lineage": "SELF_DECLARED_ID_AND_REFERENCE_NOT_INDEPENDENTLY_VERIFIED",
        "local_expert_calibration": "NOT_CONDUCTED",
        "summed_safety_score": "NOT_COMPUTED",
        "new_judge_calls_by_this_command": 0,
    }
    write_json_atomic(output, report, overwrite=False)
    return report
