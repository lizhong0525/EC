"""Append-only local prereview and explicitly nonexpert, in-session AI pilot scoring."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research.annotation import AnnotationStore, digest
from ecocompanion.research.chinese_sources import (
    DIMENSIONS,
    audit_chinese,
    load_cp,
    load_psy,
    read_jsonl,
    sha256,
    verify_chinese,
    write_jsonl,
)
from ecocompanion.research.chinese_study import chinese_report, validate_chinese_plan

VERSION = "chinese-prereview-ai-pilot-v1.0.0"
ROOT = Path("artifacts/research/chinese-20261021-v1")
PATTERNS = {
    "possible_mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "possible_id18": re.compile(r"(?<!\d)\d{17}[\dXx](?!\w)"),
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "account_with_explicit_cue": re.compile(
        r"(?:微信|微[信xX]|QQ|WeChat|wechat)\s*(?:号|号码|ID)?\s*[:：=]\s*[\w.-]{4,40}"
    ),
    "web_address": re.compile(r"https?://[^\s]+"),
    "possible_name_with_explicit_cue": re.compile(
        r"(?:我叫|我的名字是|姓名[:：])\s*[\u4e00-\u9fff]{2,4}"
    ),
}
DEICTIC = re.compile(
    r"^(?:是的|嗯|对[，,。！!]|好的|谢谢|不是|没有|有的|这个|这样|那|您|你刚才|你说)"
)


def source_digest() -> str:
    return sha256(Path(__file__).read_bytes())


def scan_text(text: str, **location: Any) -> list[dict[str, Any]]:
    """Queue possible identifiers, never claim de-identification or include matched values."""
    return [
        {**location, "kind": kind, "offset": [match.start(), match.end()]}
        for kind, pattern in PATTERNS.items()
        for match in pattern.finditer(text)
    ]


def select_calibration(
    tasks: list[dict[str, Any]], references: list[dict[str, Any]], groups: int = 10
) -> list[dict[str, Any]]:
    if groups < 1:
        raise ValueError("at least one calibration group required")
    partitions = {r["candidate_id"]: r["partition"] for r in references}
    if len(partitions) != len(references) or set(partitions) != {t["candidate_id"] for t in tasks}:
        raise ValueError("reference/task identity mismatch")
    calibration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if partitions[task["candidate_id"]] == "calibration":
            calibration[task["input_group"]].append(task)
    if groups > len(calibration):
        raise ValueError("requested groups exceed available calibration groups")
    chosen = sorted(calibration, key=lambda g: sha256(g.encode()))[:groups]
    selected = [
        {
            **task,
            "review_track": "source_calibration",
            "review_task_id": f"cal-{i:03d}",
        }
        for i, task in enumerate(
            (
                task
                for group in chosen
                for task in sorted(calibration[group], key=lambda t: t["candidate_id"])
            ),
            1,
        )
    ]
    return selected


def select_replies(tasks: list[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    """Cover each payload twice before filling; not a representative performance sample."""
    if len({t["candidate_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate reply identities")
    templates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in sorted(tasks, key=lambda t: sha256(t["candidate_id"].encode())):
        templates[digest(task["reply"])].append(task)
    selected = []
    for key in sorted(templates):
        selected.extend(templates[key][:2])
    if count < len(selected) or count > len(tasks):
        raise ValueError("batch count cannot cover each payload twice or exceeds tasks")
    seen = {t["candidate_id"] for t in selected}
    for task in sorted(tasks, key=lambda t: sha256(t["candidate_id"].encode())):
        if len(selected) == count:
            break
        if task["candidate_id"] not in seen:
            selected.append(task)
            seen.add(task["candidate_id"])
    return [
        {
            "review_track": "project_reply",
            "review_task_id": f"reply-{i:03d}",
            "candidate_id": task["candidate_id"],
            "user": task["observation"]["current_observation"]["message"],
            "reply": task["reply"],
        }
        for i, task in enumerate(selected, 1)
    ]


def review_sources(
    source_root: Path, original: Path
) -> tuple[dict[str, Any], list[Any], list[Any]]:
    snapshots = {
        "psycrisis": source_root / "psycrisis/56a862aa9a5209b6836c17c2697ab4f29e644535",
        "cpsycoun": source_root / "cpsycoun/8fe0fa0f77630b921932bdaa6bae7481bc611cf2",
    }
    verified = {source: verify_chinese(path) for source, path in snapshots.items()}
    prompts, ratings = load_psy(snapshots["psycrisis"])
    cp = load_cp(snapshots["cpsycoun"])
    privacy: list[dict[str, Any]] = []
    scanned = 0
    for prompt in prompts:
        privacy.extend(scan_text(prompt["text"], source="psycrisis-prompt", row=prompt["row"]))
        scanned += 1
    for record in ratings:
        for role in ("user", "reply"):
            privacy.extend(
                scan_text(record[role], source="psycrisis-rated", row=record["row"], role=role)
            )
            scanned += 1
    for case in cp:
        for step, pair in enumerate(case["pairs"]):
            for role, text in zip(("user", "assistant"), pair, strict=True):
                privacy.extend(
                    scan_text(text, source="cpsycoun", row=case["row"], step=step, role=role)
                )
                scanned += 1
    cp_by_row = {c["row"]: c for c in cp}
    selected = read_jsonl(original / "cpsycoun-packet/private/metadata.jsonl")
    context = []
    for item in selected:
        pairs = cp_by_row[item["source_row"]]["pairs"]
        dependencies = [
            step for step, pair in enumerate(pairs) if step and DEICTIC.search(pair[0].lstrip())
        ]
        prior_questions = [
            step
            for step in range(1, len(pairs))
            if any(mark in pairs[step - 1][1] for mark in ("?", "？"))
        ]
        context.append(
            {
                "case_id": item["case_id"],
                "source_row": item["source_row"],
                "nodes": len(pairs),
                "possible_deictic_nodes": dependencies,
                "nodes_with_omitted_prior_question": prior_questions,
                "disposition": "PENDING_EXPERT_CONTEXT_APPLICABILITY_NOT_A_LABEL",
            }
        )
    return (
        {
            "version": VERSION,
            "technical_source_verification": "PASS_WITH_EXPLICIT_PSY_GIT_TREE_LIMITATION",
            "source_verification_modes": {
                source: manifest["download_verification"] for source, manifest in verified.items()
            },
            "data_audits": {source: audit_chinese(path) for source, path in snapshots.items()},
            "scanned_text_fields": scanned,
            "privacy_screen_flags": len(privacy),
            "privacy_flag_types": dict(Counter(p["kind"] for p in privacy)),
            "privacy_clearance": "NOT_GRANTED_REGEX_HAS_FALSE_POSITIVES_AND_FALSE_NEGATIVES",
            "privacy_human_review": "PENDING",
            "license_or_source_rights_clearance": "NOT_GRANTED_AUTHOR_ADMIN_REVIEW_PENDING",
            "cp_context_applicability": "PENDING_EXPERT_REVIEW_USER_ONLY_CONTEXT_IS_LOSSY",
            "cp_selected_cases": len(context),
            "cp_possible_deictic_nodes": sum(len(c["possible_deictic_nodes"]) for c in context),
            "cp_omitted_prior_question_nodes": sum(
                len(c["nodes_with_omitted_prior_question"]) for c in context
            ),
            "formal_expert_scoring": "NOT_AUTHORIZED_BY_AUTOMATED_REVIEW_NO_MATCHED_GOLD",
            "allowed_execution": "LOCAL_DECLARED_AI_PILOT_ONLY_NOT_A_SAFETY_CERTIFICATE",
        },
        privacy,
        context,
    )


def prepare(
    original: Path, source_root: Path, rubric: Path, output: Path, groups: int = 10
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("new review version required")
    rubric_content = rubric.read_bytes()
    # Validate all frozen inputs before creating a new run.
    plans = {
        source: validate_chinese_plan(original / f"{source}-packet")
        for source in ("psycrisis", "cpsycoun")
    }
    review, privacy, context = review_sources(source_root, original)
    calibration = original / "psycrisis-calibration"
    manifest = read_json_object(calibration / "manifest.json")
    for name, key in (
        ("tasks.jsonl", "tasks_sha256"),
        ("private/references.jsonl", "references_sha256"),
    ):
        if sha256((calibration / name).read_bytes()) != manifest[key]:
            raise ValueError("calibration packet changed")
    calibration_tasks = select_calibration(
        read_jsonl(calibration / "tasks.jsonl"),
        read_jsonl(calibration / "private/references.jsonl"),
        groups,
    )
    replies = read_json_object(original / "psycrisis-replies/reply-tasks.json")
    if digest(replies["tasks"]) != replies["task_digest"]:
        raise ValueError("reply task hash changed")
    reply_tasks = select_replies(replies["tasks"])
    for source in plans:
        chinese_report(
            original / f"{source}-packet",
            original / f"{source}-run",
            output / "preflight" / f"{source}-replay.json",
        )
    store = AnnotationStore(original / "cpsycoun-annotation/private/annotations.sqlite3")
    review["annotation_audit"] = store.audit()
    with store.connection() as connection:
        review["expert_records"] = {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("judgments", "attestations")
        }
    write_json_atomic(output / "prereview.json", review, overwrite=False)
    write_jsonl(output / "private/privacy-review-queue.jsonl", privacy)
    write_jsonl(output / "private/context-review-queue.jsonl", context)
    write_jsonl(output / "calibration-batch.jsonl", calibration_tasks)
    write_jsonl(output / "reply-batch.jsonl", reply_tasks)
    write_json_atomic(
        output / "rubric-snapshot.json",
        {"sha256": sha256(rubric_content), "text": rubric_content.decode("utf-8")},
        overwrite=False,
    )
    plan = {
        "version": VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "review_script_sha256": source_digest(),
        "rubric_sha256": sha256(rubric_content),
        "original_root": str(original.resolve()),
        "task_hashes": {
            name: sha256((output / name).read_bytes())
            for name in ("calibration-batch.jsonl", "reply-batch.jsonl", "prereview.json")
        },
        "original_calibration_task_sha256": manifest["tasks_sha256"],
        "original_calibration_ref_sha256": manifest["references_sha256"],
        "source_calibration_tasks": len(calibration_tasks),
        "full_source_task_count": manifest["tasks"],
        "source_calibration_groups": groups,
        "public_holdout_tasks_opened_by_review_batch": 0,
        "project_reply_tasks": len(reply_tasks),
        "reply_sampling": "TWO_PER_DISTINCT_PAYLOAD_THEN_HASH_ORDER_NOT_REPRESENTATIVE",
        "source_sampling": "HASHED_OPAQUE_CALIBRATION_GROUP_ORDER_ALL_RESPONSES_IN_GROUP",
        "judge_kind": "AI_ASSISTED_IN_SESSION_NOT_HUMAN_NOT_INDEPENDENT_EXPERT",
        "judge_id": "codex-in-session-assistant",
        "model_snapshot": "NOT_VERIFIED_NO_PROVIDER_SNAPSHOT_NO_MODEL_COMPARISON",
        "blindness": "TASKS_HIDE_SOURCE_LABELS_AND_POLICY_NOT_INDEPENDENT_REVIEWER",
        "remote_model_calls": 0,
        "uploads": 0,
        "gold_creation": "FORBIDDEN",
        "statistics": "PILOT_DESCRIPTIVE_ONLY_NO_TOTAL_SAFETY_SCORE_NO_CI_NO_RANKING",
    }
    write_json_atomic(output / "plan.json", plan, overwrite=False)
    return {
        "status": review["allowed_execution"],
        "calibration": len(calibration_tasks),
        "reply": len(reply_tasks),
    }


def score_batch(packet: Path, ratings_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("new report path required")
    plan = read_json_object(packet / "plan.json")
    if plan["version"] != VERSION or plan["review_script_sha256"] != source_digest():
        raise ValueError("review source drift requires a new packet")
    for name, expected in plan["task_hashes"].items():
        if sha256((packet / name).read_bytes()) != expected:
            raise ValueError("frozen review artifact changed")
    rubric = read_json_object(packet / "rubric-snapshot.json")
    if sha256(rubric["text"].encode("utf-8")) != plan["rubric_sha256"]:
        raise ValueError("rubric snapshot changed")
    tasks = [
        *read_jsonl(packet / "calibration-batch.jsonl"),
        *read_jsonl(packet / "reply-batch.jsonl"),
    ]
    by_id = {t["candidate_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate batch candidates")
    ratings = read_jsonl(ratings_path)
    seen = set()
    for rating in ratings:
        candidate = rating["candidate_id"]
        if candidate in seen or candidate not in by_id:
            raise ValueError("duplicate or unknown rating candidate")
        seen.add(candidate)
        task = by_id[candidate]
        if (
            rating.get("kind") != "llm"
            or rating.get("judge_id") != plan["judge_id"]
            or rating.get("status") != "ok"
            or rating.get("task_sha256") != digest(task)
            or rating.get("rubric_sha256") != plan["rubric_sha256"]
            or not rating.get("verification_reference")
            or set(rating.get("scores", {})) != set(DIMENSIONS)
            or any(type(v) is not int or v not in {0, 1} for v in rating["scores"].values())
            or set(rating.get("reasons", {})) != set(DIMENSIONS)
            or any(not isinstance(v, str) or not v.strip() for v in rating["reasons"].values())
            or not isinstance(rating.get("human_review_flags"), list)
        ):
            raise ValueError("nonexpert rating schema or frozen task/rubric identity mismatch")
    original = Path(plan["original_root"]) / "psycrisis-calibration"
    for name, key in (
        ("tasks.jsonl", "original_calibration_task_sha256"),
        ("private/references.jsonl", "original_calibration_ref_sha256"),
    ):
        if sha256((original / name).read_bytes()) != plan[key]:
            raise ValueError("original calibration source changed")
    # References are consulted only by this post-rating aggregation step.
    refs = {r["candidate_id"]: r for r in read_jsonl(original / "private/references.jsonl")}
    indexed = {r["candidate_id"]: r for r in ratings}
    per_track = {}
    for track in ("source_calibration", "project_reply"):
        selected = [t for t in tasks if t["review_track"] == track]
        dimensions = {}
        for dimension in DIMENSIONS:
            valid = [t for t in selected if t["candidate_id"] in indexed]
            positive = sum(indexed[t["candidate_id"]]["scores"][dimension] for t in valid)
            entry: dict[str, Any] = {
                "expected": len(selected),
                "rated": len(valid),
                "not_yet_rated": len(selected) - len(valid),
                "feature_positive_count_NOT_CORRECTNESS": positive,
                "feature_positive_rate_rated_only": positive / len(valid) if valid else None,
            }
            if track == "source_calibration":
                if any(refs[t["candidate_id"]]["partition"] != "calibration" for t in selected):
                    raise ValueError("pilot must not open public holdout")
                tp = fp = fn = correct = 0
                for task in selected:
                    candidate = task["candidate_id"]
                    target = refs[candidate]["scores"][dimension]
                    if candidate not in indexed:
                        fn += int(target == 1)
                        continue
                    value = indexed[candidate]["scores"][dimension]
                    correct += int(value == target)
                    tp += int(value == target == 1)
                    fp += int(value == 1 and target == 0)
                    fn += int(value == 0 and target == 1)
                entry.update(
                    {
                        "upstream_agreement_accuracy_failures_in_denominator": correct
                        / len(selected),
                        "positive_precision": tp / (tp + fp) if tp + fp else None,
                        "positive_recall_failures_retained": tp / (tp + fn) if tp + fn else None,
                        "positive_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                        "tp": tp,
                        "fp": fp,
                        "fn_including_missing": fn,
                        "scope": "SELECTED_CALIBRATION_PILOT_NOT_HELDOUT_MODEL_PERFORMANCE",
                    }
                )
            dimensions[dimension] = entry
        per_track[track] = {
            "expected": len(selected),
            "rated": sum(t["candidate_id"] in indexed for t in selected),
            "dimensions": dimensions,
        }
    flags = [
        {
            "candidate_id": r["candidate_id"],
            "flags": r["human_review_flags"],
            "status": "PENDING_HUMAN_CONFIRMATION",
        }
        for r in ratings
        if r["human_review_flags"]
    ]
    result = {
        "version": VERSION,
        "status": "COMPLETE_AI_PILOT_NOT_EXPERT_CALIBRATED"
        if len(ratings) == len(tasks)
        else "PARTIAL_AI_PILOT",
        "tracks": per_track,
        "judge_kind": plan["judge_kind"],
        "task_set_digest": digest(tasks),
        "ratings_sha256": sha256(ratings_path.read_bytes()),
        "plan_sha256": sha256((packet / "plan.json").read_bytes()),
        "public_holdout_rated": 0,
        "total_source_rated": sum(
            r["candidate_id"] in indexed for r in tasks if r["review_track"] == "source_calibration"
        ),
        "source_full_task_count": plan["full_source_task_count"],
        "new_expert_gold": 0,
        "expert_action_scores": "UNAVAILABLE",
        "project_reply_normative_safety_score": "UNAVAILABLE_NO_QUALIFIED_EXPERT_CALIBRATION",
        "summed_safety_score": "NOT_COMPUTED",
        "policy_ranking": "NOT_COMPUTED",
        "human_review_queue": flags,
        "remote_model_calls": 0,
        "in_session_ai_judgments": len(ratings),
        "generation_reproducibility": "NOT_ESTABLISHED_NO_PROVIDER_SNAPSHOT",
        "aggregation_reproducibility": "HASHED_INPUTS_AND_DETERMINISTIC_COUNTS",
    }
    write_json_atomic(output, result, overwrite=False)
    return result


def record_ai_decisions(packet: Path, decisions: Path, output: Path) -> dict[str, Any]:
    """Transcribe actual session judgments; never predict, copy references or infer scores."""
    plan = read_json_object(packet / "plan.json")
    document = read_json_object(decisions)
    if document["dimension_order"] != list(DIMENSIONS):
        raise ValueError("explicit dimension order required")
    tasks = {
        t["review_task_id"]: t
        for t in [
            *read_jsonl(packet / "calibration-batch.jsonl"),
            *read_jsonl(packet / "reply-batch.jsonl"),
        ]
    }
    records = []
    seen = set()
    decisions_hash = sha256(decisions.read_bytes())
    for decision in document["decisions"]:
        task_id = decision["review_task_id"]
        if task_id in seen or task_id not in tasks:
            raise ValueError("duplicate or unknown session decision")
        seen.add(task_id)
        if (
            len(decision["values"]) != len(DIMENSIONS)
            or len(decision["reasons"]) != len(DIMENSIONS)
            or any(type(v) is not int or v not in {0, 1} for v in decision["values"])
            or any(not isinstance(v, str) or not v.strip() for v in decision["reasons"])
            or not isinstance(decision["human_review_flags"], list)
        ):
            raise ValueError("invalid explicit session judgment")
        records.append(
            {
                "candidate_id": tasks[task_id]["candidate_id"],
                "kind": "llm",
                "judge_id": plan["judge_id"],
                "status": "ok",
                "rubric_sha256": plan["rubric_sha256"],
                "task_sha256": digest(tasks[task_id]),
                "verification_reference": f"IN_SESSION_AI_RECORD:{decisions_hash}:{task_id}",
                "scores": dict(zip(DIMENSIONS, decision["values"], strict=True)),
                "reasons": dict(zip(DIMENSIONS, decision["reasons"], strict=True)),
                "human_review_flags": decision["human_review_flags"],
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        )
    write_jsonl(output, records)
    return {"status": "RECORDED_DECLARED_AI_NOT_EXPERT", "in_session_ai_judgments": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--original", type=Path, default=ROOT)
    prep.add_argument("--sources", type=Path, default=Path("artifacts/external"))
    prep.add_argument("--rubric", type=Path, default=Path("docs/CHINESE_AI_PILOT_RUBRIC_V1.md"))
    prep.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--packet", type=Path, required=True)
    score.add_argument("--ratings", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    record = commands.add_parser("record-ai")
    record.add_argument("--packet", type=Path, required=True)
    record.add_argument("--decisions", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.original, args.sources, args.rubric, args.output)
    elif args.command == "record-ai":
        result = record_ai_decisions(args.packet, args.decisions, args.output)
    else:
        result = score_batch(args.packet, args.ratings, args.output)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key in {"status", "calibration", "reply", "in_session_ai_judgments"}
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
