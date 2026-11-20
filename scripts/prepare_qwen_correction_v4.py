"""One development-guided correction using new, disjoint source dialogues.

The inspected frozen regression is never read. Existing groups remain intact;
new external dialogues are assigned as whole groups to train or development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from prepare_qwen_external_v3 import PINNED_REVISION, PINNED_SHA256

from ecocompanion.pet.training import TrainingExample, load_examples, write_sft

NEW_EXTERNAL = [
    (507, 43, "我很害怕，也很无助"),
    (1452, 3, "我现在非常焦虑"),
    (976, 47, "时常觉得焦虑和压力巨大"),
    (1787, 75, "这让我感到无助和困惑"),
    (480, 3, "最近总是会莫名的烦躁和焦虑"),
    (676, 47, "纠结和焦虑是目前最困扰我的问题"),
    (1586, 35, "我对冲突总是感觉很不安"),
    (1362, 13, "这种预期性的焦虑让我觉得整个工作都是一个巨大的负担"),
    (1684, 7, "心里一直很焦虑"),
    (190, 3, "我感到很迷茫"),
    (1624, 3, "有些时候真的感觉好迷茫"),
    (799, 39, "这种彻彻底底的决心让我有点害怕"),
]

NEW_BOUNDARY = [
    (
        "ordinary",
        "none",
        "我对以后的路感到迷茫和焦虑，不知道先做什么。",
        "我对以后的路感到迷茫和焦虑",
    ),
    ("ordinary", "none", "我对未来很不安，总觉得自己难以适应变化。", "我对未来很不安"),
    ("ordinary", "none", "最近工作总出错，我很害怕接下来还会失败。", "我很害怕接下来还会失败"),
    ("ordinary", "none", "我有时会突然恐慌，但没有伤害自己的想法。", "没有伤害自己的想法"),
    ("ordinary", "none", "考试前我心跳很快，害怕自己会考砸。", "害怕自己会考砸"),
    ("ordinary", "none", "我现在很崩溃，只想找人聊聊这两天的压力。", "只想找人聊聊这两天的压力"),
    ("quotation", "none", "论文里写着‘对未来的迷茫和焦虑’，我正在引用原文。", "我正在引用原文"),
    (
        "third_party",
        "uncertain",
        "我同事对未来很迷茫，还说不想活了。",
        "我同事对未来很迷茫，还说不想活了",
    ),
]


def split_group(group: str) -> str:
    digest = hashlib.sha256(f"v4:{group}".encode()).hexdigest()
    return "validation" if int(digest[:8], 16) % 5 == 0 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a previous dataset")
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != PINNED_SHA256:
        raise SystemExit("External source is not the pinned reviewed corpus")
    args.output.mkdir(parents=True)
    shutil.copyfile(args.previous / "source_registry.json", args.output / "source_registry.json")
    dialogues = json.loads(args.source.read_bytes())
    previous = load_examples(
        args.previous / "reviewed_examples.jsonl", args.previous / "source_registry.json"
    )
    used_dialogues = {row.group_id for row in previous}
    added: list[TrainingExample] = []
    for dialogue_id, turn_id, quote in NEW_EXTERNAL:
        group = f"psydial-dialogue-{dialogue_id:04d}"
        assert group not in used_dialogues
        message = dialogues[dialogue_id]["messages"][turn_id]
        assert message["role"] == "user" and quote in message["content"]
        high = any(word in quote for word in ("巨大", "负担", "无助"))
        added.append(
            TrainingExample(
                example_id=f"psydial-{dialogue_id:04d}-{turn_id:03d}",
                group_id=group,
                text=message["content"].strip(),
                target={
                    "safety_signal": "none",
                    "evidence_quotes": [quote],
                    "support_focus": "倾听当前情绪、生活影响与支持需要",
                    "needs_clarification": False,
                    "wellbeing_level": "distressed" if high else "strained",
                    "mood": "negative",
                    "stress": "high" if high else "moderate",
                },
                source_id="psydial-d4-acl2025",
                reviewer="codex-individual-text-review-not-clinical",
                review_status="reviewed",
                scenario="ordinary",
            )
        )
    for index, (scenario, signal, text, quote) in enumerate(NEW_BOUNDARY):
        added.append(
            TrainingExample(
                example_id=f"v4-correction-{index:03d}",
                group_id=f"v4-correction-group-{index:03d}",
                text=text,
                target={
                    "safety_signal": signal,
                    "evidence_quotes": [quote],
                    "support_focus": "确认相关人员的当前安全"
                    if signal == "uncertain"
                    else "倾听当前情绪与支持需要",
                    "needs_clarification": signal == "uncertain",
                    "wellbeing_level": "strained"
                    if signal == "none" and scenario == "ordinary"
                    else "uncertain",
                    "mood": "negative" if scenario == "ordinary" else "uncertain",
                    "stress": "moderate" if scenario == "ordinary" else "uncertain",
                },
                source_id="eco-synthetic-v3-boundary",
                reviewer="codex-project-boundary-review-not-clinical",
                review_status="reviewed",
                scenario=scenario,
            )
        )
    selected = previous + added
    (args.output / "reviewed_examples.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in selected), encoding="utf-8"
    )
    splits = {"train": [], "validation": [], "test": []}
    for row in selected:
        splits[split_group(row.group_id)].append(row.group_id)
    splits = {key: sorted(set(value)) for key, value in splits.items()}
    counts = write_sft(selected, splits, args.output / "prepared")
    (args.output / "correction_provenance.json").write_text(
        json.dumps(
            {
                "reason": (
                    "One development-only correction for exact quotation and "
                    "ordinary intense emotion"
                ),
                "source_revision": PINNED_REVISION,
                "frozen_test_read": False,
                "new_external": NEW_EXTERNAL,
                "new_boundary": NEW_BOUNDARY,
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
