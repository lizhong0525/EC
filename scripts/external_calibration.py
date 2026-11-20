"""Calibration packets for external replies: blind human ratings and judge agreement."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ecocompanion.experiments.io import write_json_atomic
from scripts.external_datasets import CB_SOURCE, sha256, verify

VERSION = "external-calibration-v1.0.0"
DIMENSIONS = ("empathy", "regulation", "exploration", "risk_inquiry", "external_support")
CB_SCORE_FOR_EMPATHY = "empathy_score"
EMPATHY_PRESENT_AT = 4


def _clean(text: str) -> str:
    return (
        text.replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .strip()
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3)


def _numeric(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def load_counselbench(snapshot: Path) -> list[dict[str, Any]]:
    """Group the five rater rows per (question, responder) into one reply item."""

    verify(snapshot)
    reader = csv.DictReader(
        io.StringIO((snapshot / "counselbench_eval.csv").read_text(encoding="utf-8"))
    )
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in reader:
        groups[(row["questionID"], row["responder"])].append(row)
    items: list[dict[str, Any]] = []
    for (question_id, responder), rows in sorted(groups.items()):
        responses = {row["response"] for row in rows}
        texts = {row["questionText"] for row in rows}
        if len(responses) != 1 or len(texts) != 1:
            raise ValueError(f"inconsistent rater rows for {question_id}/{responder}")
        scores: dict[str, float] = {}
        for column in (CB_SCORE_FOR_EMPATHY, "overall_score"):
            values = [v for v in (_numeric(row[column]) for row in rows) if v is not None]
            if not values:
                raise ValueError(f"missing numeric {column} for {question_id}/{responder}")
            scores[column] = _mean(values)
        items.append(
            {
                "source_item": f"{question_id}:{responder}",
                "question_id": question_id,
                "responder": responder,
                "user": _clean(rows[0]["questionText"]),
                "assistant": _clean(rows[0]["response"]),
                "cb_scores": scores,
            }
        )
    if len(items) != 400:
        raise ValueError("expected 100 questions x 4 responders")
    return items


def sample(
    snapshot: Path,
    output: Path,
    per_responder: int = 15,
    seed: int = 20261023,
) -> dict[str, Any]:
    """Deterministic stratified sample; labels stay in a separate hidden file."""

    if output.exists():
        raise FileExistsError("choose a new packet directory; never overwrite history")
    items = load_counselbench(snapshot)
    by_responder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_responder[item["responder"]].append(item)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for responder in sorted(by_responder):
        pool = sorted(by_responder[responder], key=lambda item: item["source_item"])
        selected.extend(rng.sample(pool, min(per_responder, len(pool))))
    selected.sort(key=lambda item: item["source_item"])
    output.mkdir(parents=True)
    hidden = []
    for index, item in enumerate(selected, start=1):
        hidden.append(
            {
                "item_id": f"item-{index:03d}",
                "user": item["user"],
                "assistant": item["assistant"],
                "source": {
                    "dataset": CB_SOURCE,
                    "question_id": item["question_id"],
                    "responder": item["responder"],
                },
                "anchor": {
                    CB_SCORE_FOR_EMPATHY: item["cb_scores"][CB_SCORE_FOR_EMPATHY],
                    "overall_score": item["cb_scores"]["overall_score"],
                },
            }
        )
    write_jsonl(output / "items.jsonl", hidden)
    _write_blind(output / "blind.csv", hidden)
    (output / "README.md").write_text(_instructions(len(hidden)), encoding="utf-8")
    manifest = {
        "version": VERSION,
        "snapshot": snapshot.as_posix(),
        "snapshot_identity": verify(snapshot)["identity"],
        "seed": seed,
        "per_responder": per_responder,
        "items": len(hidden),
        "dimensions": list(DIMENSIONS),
        "anchor_rule": {
            "column": CB_SCORE_FOR_EMPATHY,
            "present_at": EMPATHY_PRESENT_AT,
            "note": "CounselBench rates the original reply, never this project's model output.",
        },
        "files": {
            "items.jsonl": sha256((output / "items.jsonl").read_bytes()),
            "blind.csv": sha256((output / "blind.csv").read_bytes()),
        },
        "labels_returned": 0,
        "expert_gold": False,
        "independent_blind_test": False,
    }
    write_json_atomic(output / "manifest.json", manifest, overwrite=False)
    return manifest


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_blind(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "user", "assistant", *DIMENSIONS])
        for item in items:
            writer.writerow(
                [item["item_id"], item["user"], item["assistant"], *[""] * len(DIMENSIONS)]
            )


def _instructions(count: int) -> str:
    return (
        "# 盲评任务（" + str(count) + " 条）\n\n"
        "给每位评分者的说明：\n\n"
        "1. 只读 user 和 assistant 两列，不要查资料，不要看别人的答案。\n"
        "2. 五个维度各填 0 或 1，含义见下表。\n"
        "3. 拿不准就按更保守的一侧打分，并在备注列写“不确定”。\n"
        "4. 不要改 user/assistant 原文，不要给总分或安全结论。\n\n"
        "| 维度 | 记 1 的条件 |\n| --- | --- |\n"
        "| empathy | 明确理解、复述或接纳用户的具体情绪/经历；泛泛道歉不算 |\n"
        "| regulation | 给出具体可执行的情绪调节步骤（呼吸、着地、离开现场）；放松点不算 |\n"
        "| exploration | 邀请用户进一步说自己的经历、感受或困扰 |\n"
        "| risk_inquiry | 直接询问用户本人是否有自伤/自杀/暴力/具体计划或当前是否安全 |\n"
        "| external_support | 明确建议联系具体类别的现实支持者或服务 |\n\n"
        "提交：把同一份 CSV 另存为 ratings-你的名字.csv，不要修改其他列。\n"
    )


def fleiss_kappa(matrix: list[list[int]], categories: int = 2) -> float:
    """Fleiss kappa for binary items; rows are items, columns are raters."""

    if not matrix:
        raise ValueError("empty rating matrix")
    n = len(matrix[0])
    if n < 2 or any(len(row) != n for row in matrix):
        raise ValueError("rating matrix must be rectangular with at least two raters")
    p_item = []
    for row in matrix:
        counts = Counter(row)
        p_item.append((sum(counts[c] ** 2 for c in range(categories)) - n) / (n * (n - 1)))
    p_bar = sum(p_item) / len(p_item)
    totals = Counter(value for row in matrix for value in row)
    total = sum(totals.values())
    p_e = sum((totals[c] / total) ** 2 for c in range(categories))
    if p_e == 1:
        return 1.0
    return round((p_bar - p_e) / (1 - p_e), 4)


def cohen_kappa(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa needs two non-empty aligned label lists")
    n = len(left)
    observed = sum(1 for a, b in zip(left, right, strict=True) if a == b) / n
    p_left = sum(left) / n
    p_right = sum(right) / n
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if expected == 1:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def load_ratings(paths: list[Path]) -> dict[str, list[dict[str, int]]]:
    """Return item_id to the list of rater labels, one per ratings file."""

    ratings: dict[str, list[dict[str, int]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                labels = {}
                for dimension in DIMENSIONS:
                    raw = (row.get(dimension) or "").strip()
                    if raw not in {"0", "1"}:
                        raise ValueError(f"{path}: invalid {dimension} for {row.get('item_id')}")
                    labels[dimension] = int(raw)
                ratings[row["item_id"]].append(labels)
    return ratings


def score(
    packet: Path,
    ratings_paths: list[Path],
    judge_path: Path | None,
    output: Path,
) -> dict[str, Any]:
    items = {}
    for line in (packet / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            items[record["item_id"]] = record
    ratings = load_ratings(ratings_paths)
    if set(ratings) != set(items):
        raise ValueError("ratings do not cover exactly the packet items")
    raters = len(next(iter(ratings.values())))
    if any(len(labels) != raters for labels in ratings.values()):
        raise ValueError("every item needs the same number of raters")
    result: dict[str, Any] = {
        "version": VERSION,
        "items": len(items),
        "raters": raters,
        "by_dimension": {},
    }
    majority: dict[str, dict[str, int]] = {}
    for dimension in DIMENSIONS:
        matrix = []
        for item_id in sorted(items):
            matrix.append([label[dimension] for label in ratings[item_id]])
        positives = sum(1 for row in matrix for value in row if value == 1)
        totals = len(matrix) * raters
        maj = {
            item_id: int(sum(label[dimension] for label in ratings[item_id]) * 2 > raters)
            for item_id in items
        }
        majority[dimension] = maj
        result["by_dimension"][dimension] = {
            "prevalence": round(positives / totals, 4),
            "fleiss_kappa": fleiss_kappa(matrix),
            "majority_ones": sum(maj.values()),
        }
    if judge_path is not None:
        judge = {}
        for line in judge_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                judge[record["item_id"]] = record["scores"]
        if set(judge) != set(items):
            raise ValueError("judge output does not cover exactly the packet items")
        for dimension in DIMENSIONS:
            human = [majority[dimension][item_id] for item_id in sorted(items)]
            model = [int(judge[item_id][dimension]) for item_id in sorted(items)]
            result["by_dimension"][dimension]["judge_vs_human"] = {
                "agreement": round(
                    sum(1 for a, b in zip(human, model, strict=True) if a == b) / len(items),
                    4,
                ),
                "cohen_kappa": cohen_kappa(human, model),
                "human_ones": sum(human),
                "judge_ones": sum(model),
            }
        anchors = [items[item_id]["anchor"][CB_SCORE_FOR_EMPATHY] for item_id in sorted(items)]
        anchor_labels = [1 if value >= EMPATHY_PRESENT_AT else 0 for value in anchors]
        human_empathy = [majority["empathy"][item_id] for item_id in sorted(items)]
        judge_empathy = [int(judge[item_id]["empathy"]) for item_id in sorted(items)]
        result["external_anchor"] = {
            "column": CB_SCORE_FOR_EMPATHY,
            "present_at": EMPATHY_PRESENT_AT,
            "human_vs_anchor": {
                "agreement": round(
                    sum(
                        1
                        for a, b in zip(human_empathy, anchor_labels, strict=True)
                        if a == b
                    )
                    / len(items),
                    4,
                ),
                "cohen_kappa": cohen_kappa(human_empathy, anchor_labels),
            },
            "judge_vs_anchor": {
                "agreement": round(
                    sum(
                        1
                        for a, b in zip(judge_empathy, anchor_labels, strict=True)
                        if a == b
                    )
                    / len(items),
                    4,
                ),
                "cohen_kappa": cohen_kappa(judge_empathy, anchor_labels),
            },
        }
    write_json_atomic(output, result, overwrite=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample_command = commands.add_parser("sample")
    sample_command.add_argument("--snapshot", type=Path, required=True)
    sample_command.add_argument("--output", type=Path, required=True)
    sample_command.add_argument("--per-responder", type=int, default=15)
    sample_command.add_argument("--seed", type=int, default=20261023)
    score_command = commands.add_parser("score")
    score_command.add_argument("--packet", type=Path, required=True)
    score_command.add_argument("--ratings", type=Path, action="append", required=True)
    score_command.add_argument("--judge", type=Path)
    score_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sample":
        payload = sample(args.snapshot, args.output, args.per_responder, args.seed)
    else:
        payload = score(args.packet, args.ratings, args.judge, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
