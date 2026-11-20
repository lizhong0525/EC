"""Build v3 from explicitly reviewed external turns; never read frozen test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ecocompanion.pet.training import TrainingExample, load_examples, write_sft

PINNED_REVISION = "0ceef5ac19fb8174baa6be31ce1b2637711cb4c2"
PINNED_SHA256 = "29bfa723645f3c7213d00e34042c164545b27629487253fa6ebec97c7f3638b9"

# Individually read seeker turns. No assistant reply or diagnosis is a label.
# Selection precedes v3 training and does not use the inspected v1 test set.
REVIEWED_EXTERNAL = [
    (0, 5, "work_stress", "这么迷茫和空虚"),
    (2, 23, "ordinary", "对很多事情都提不起兴趣"),
    (3, 49, "work_stress", "工作时也常常陷入交流困难的困境"),
    (5, 9, "work_stress", "这种不一致会让我感到焦虑"),
    (7, 5, "physical_health", "根本无法正常睡眠"),
    (10, 11, "work_stress", "压力特别大"),
    (13, 5, "ordinary", "感觉很困惑"),
    (17, 5, "work_stress", "我感觉特别无力"),
    (19, 17, "work_stress", "自己压力很大"),
    (23, 7, "ordinary", "我内心的焦虑和压力"),
    (26, 7, "work_stress", "有时候情绪复发时会更加严重"),
    (31, 7, "work_stress", "压力特别大"),
    (34, 5, "work_stress", "最近我感觉压力很大"),
    (1, 7, "ordinary", "害怕一旦拒绝或是表示自己不满，就会被大家排挤"),
    (21, 7, "ordinary", "我害怕自己做得不够好"),
    (22, 53, "ordinary", "陷入一种迷茫"),
    (24, 3, "ordinary", "我又开始害怕"),
    (27, 39, "ordinary", "不由自主地感觉到孤独"),
    (35, 69, "ordinary", "还是感觉很紧张"),
    (36, 31, "ordinary", "更不安和自我怀疑"),
    (39, 27, "ordinary", "担心他们觉得我是个不好相处的人"),
    (40, 1, "ordinary", "这周过得其实挺紧张的"),
    (43, 41, "ordinary", "更多是对未来的迷茫和焦虑"),
    (45, 5, "ordinary", "我觉得很焦虑"),
    (51, 11, "ordinary", "要崩溃了，控制不住的恐慌"),
    (52, 7, "ordinary", "我担心自己只是小题大做"),
    (55, 9, "physical_health", "我就一直失眠"),
    (62, 13, "physical_health", "翻来覆去根本睡不着"),
    (63, 21, "physical_health", "只是觉得精神上很疲惫"),
    (69, 17, "physical_health", "精神上和身体上都很疲惫"),
    (81, 5, "physical_health", "睡眠质量也很差，经常失眠"),
    (87, 21, "physical_health", "几乎每天都失眠"),
    (99, 113, "physical_health", "总是失眠或者睡着后反反复复醒来"),
    (106, 31, "physical_health", "经常失眠"),
    (118, 11, "physical_health", "我经常失眠"),
    (141, 9, "physical_health", "总是会失眠，辗转反侧"),
    (162, 3, "physical_health", "总感觉疲惫不堪"),
    (173, 43, "physical_health", "有时候真的会觉得很疲惫"),
    (188, 5, "physical_health", "这周仍旧有些紧张和疲惫"),
]

# New, independently grouped boundary examples. Safety is not a severity score.
BOUNDARY = [
    ("current_safety", "current", "我现在总想伤害自己，觉得快控制不住了。", "总想伤害自己"),
    ("current_safety", "current", "我已经把药放在桌上了，今晚不想再撑下去。", "今晚不想再撑下去"),
    ("current_safety", "current", "我正在窗边，脑子里全是跳下去的念头。", "跳下去的念头"),
    ("current_safety", "current", "我这两天一直计划怎么结束生命。", "一直计划怎么结束生命"),
    ("current_safety", "current", "我刚割伤了手腕，现在不知道怎么办。", "刚割伤了手腕"),
    ("current_safety", "current", "我现在有强烈的轻生冲动，一个人在家。", "有强烈的轻生冲动"),
    ("current_safety", "current", "我准备好了绳子，打算半夜实施。", "准备好了绳子"),
    (
        "current_safety",
        "current",
        "我现在觉得自己马上就会做出伤害自己的事。",
        "马上就会做出伤害自己的事",
    ),
    ("negation", "none", "我现在没有轻生的想法，只是考试前很紧张。", "考试前很紧张"),
    ("negation", "none", "我并不是要自杀，最近只是累得不想说话。", "累得不想说话"),
    ("negation", "none", "别担心，我不会做傻事，只是工作上有些挫败。", "工作上有些挫败"),
    ("negation", "none", "我说不想活了是在开玩笑，其实刚赢了一局游戏。", "刚赢了一局游戏"),
    ("past", "none", "大学时有过自残经历，但这两年已经没有再发生。", "这两年已经没有再发生"),
    ("past", "none", "去年确实想过轻生，现在我正在接受专业帮助。", "正在接受专业帮助"),
    ("past", "none", "我小时候很怕黑，现在一个人住已经能适应。", "已经能适应"),
    ("past", "none", "那段想放弃自己的时期过去了，今天想讨论求职计划。", "想讨论求职计划"),
    ("quotation", "none", "这首歌写着‘想离开这个世界’，我在摘抄歌词。", "我在摘抄歌词"),
    ("quotation", "none", "小说角色说他不想活了，我想分析这个人物。", "小说角色说他不想活了"),
    ("quotation", "none", "课堂作业是讨论‘生存还是毁灭’，不是我的心情。", "不是我的心情"),
    ("quotation", "none", "我在写一篇关于青少年自杀预防的论文。", "自杀预防的论文"),
    (
        "third_party",
        "uncertain",
        "室友今天说她想结束生命，我该怎么陪她？",
        "室友今天说她想结束生命",
    ),
    (
        "third_party",
        "uncertain",
        "我弟弟刚发消息说不想活了，我现在赶过去。",
        "我弟弟刚发消息说不想活了",
    ),
    ("third_party", "uncertain", "网友透露正在准备自残，我只有他的账号。", "网友透露正在准备自残"),
    ("third_party", "uncertain", "妈妈说她活着没意思，我担心她今晚的安全。", "妈妈说她活着没意思"),
    ("metaphor", "none", "这局我又被Boss打死了，队友还在催我。", "被Boss打死了"),
    ("metaphor", "none", "方案第三次被毙掉，我真的有点沮丧。", "有点沮丧"),
    ("metaphor", "none", "电脑死机了，文件还没来得及保存。", "文件还没来得及保存"),
    ("metaphor", "none", "这个角色最后牺牲了，结局让我有点难受。", "有点难受"),
    ("work_stress", "none", "连续加班一周让我很疲惫，周末想好好睡一觉。", "很疲惫"),
    ("work_stress", "none", "临近答辩我很紧张，担心自己讲不清楚。", "很紧张"),
    ("work_stress", "none", "任务多到做不完，我现在压力特别大。", "压力特别大"),
    ("work_stress", "none", "今天终于完成汇报，感觉轻松了不少。", "感觉轻松了不少"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PINNED_SHA256:
        raise SystemExit("Source SHA-256 differs from the reviewed PsyDial-D4 revision")
    dialogues = json.loads(payload)
    selected: list[TrainingExample] = []
    provenance: list[dict] = []
    for dialogue_id, turn_id, scenario, quote in REVIEWED_EXTERNAL:
        message = dialogues[dialogue_id]["messages"][turn_id]
        text = message["content"].strip()
        assert message["role"] == "user" and quote in text
        example_id = f"psydial-{dialogue_id:04d}-{turn_id:03d}"
        high = any(word in quote for word in ("特别大", "崩溃", "根本无法", "根本睡不着"))
        selected.append(
            TrainingExample(
                example_id=example_id,
                group_id=f"psydial-dialogue-{dialogue_id:04d}",
                text=text,
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
                scenario=scenario,
            )
        )
        provenance.append(
            {"example_id": example_id, "dialogue_id": dialogue_id, "turn_id": turn_id}
        )

    for index, (scenario, signal, text, quote) in enumerate(BOUNDARY):
        current = signal == "current"
        troubled = any(
            word in text
            for word in ("紧张", "沮丧", "疲惫", "压力", "担心", "难受", "挫败", "累得")
        )
        selected.append(
            TrainingExample(
                example_id=f"v3-boundary-{index:03d}",
                group_id=f"v3-boundary-group-{index:03d}",
                text=text,
                target={
                    "safety_signal": signal,
                    "evidence_quotes": [quote],
                    "support_focus": "确认当前安全与现实支持"
                    if signal != "none"
                    else "倾听当前情绪与支持需要",
                    "needs_clarification": signal == "uncertain",
                    "wellbeing_level": "distressed"
                    if current
                    else "strained"
                    if troubled
                    else "uncertain",
                    "mood": "negative"
                    if current or troubled
                    else "positive"
                    if any(word in text for word in ("赢了", "轻松", "适应"))
                    else "uncertain",
                    "stress": "high"
                    if current or "特别大" in text
                    else "moderate"
                    if troubled
                    else "low"
                    if any(word in text for word in ("赢了", "轻松", "适应"))
                    else "uncertain",
                },
                source_id="eco-synthetic-v3-boundary",
                reviewer="codex-project-boundary-review-not-clinical",
                review_status="reviewed",
                scenario=scenario,
            )
        )
    selected.extend(load_examples(args.original, args.output / "source_registry.json"))
    (args.output / "reviewed_examples.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in selected), encoding="utf-8"
    )
    splits = {"train": [], "validation": [], "test": []}
    for row in selected:
        if row.source_id == "psydial-d4-acl2025":
            dialogue_id = int(row.group_id.rsplit("-", 1)[-1])
            digest = hashlib.sha256(f"v3-dialogue:{dialogue_id}".encode()).hexdigest()
            split = "validation" if int(digest[:8], 16) % 5 == 0 else "train"
        else:
            split = "validation" if int(row.example_id.rsplit("-", 1)[-1]) % 4 == 0 else "train"
        splits[split].append(row.group_id)
    splits = {split: sorted(set(groups)) for split, groups in splits.items()}
    counts = write_sft(selected, splits, args.output / "prepared")
    (args.output / "external_provenance.json").write_text(
        json.dumps(
            {
                "dataset": "qiuhuachuan/PsyDial-D4",
                "revision": PINNED_REVISION,
                "sha256": PINNED_SHA256,
                "license": "Apache-2.0",
                "citation": "Qiu and Lan, PsyDial, ACL 2025, pp. 21624-21655",
                "review": "Individual seeker-turn text review; no clinical annotations",
                "rows": provenance,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "sources": dict(Counter(row.source_id for row in selected)),
                "labels": dict(Counter(row.target.safety_signal for row in selected)),
                "review": "Engineering annotation, not clinical or double expert review",
                "frozen_test_read": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
