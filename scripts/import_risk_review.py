"""Import red-text choices without inferring unmarked answers or reviewer credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = "{" + NS["w"] + "}"


def red_choice(text: str, red: set[int], label: str, options: tuple[str, ...]) -> str | None:
    if label not in text:
        return None
    start = text.index(label) + len(label)
    end = text.find("：", start)
    if end >= 0:
        # The next field starts before its colon; labels do not contain option tokens.
        stop = text.rfind("    ", start, end)
        end = stop if stop >= 0 else end
    else:
        end = len(text)
    selected = [
        match.group()
        for match in re.finditer("|".join(map(re.escape, options)), text[start:end])
        if all(start + i in red for i in range(match.start(), match.end()))
    ]
    return selected[0] if len(selected) == 1 else None


def paragraphs(source: Path) -> list[dict[str, Any]]:
    with ZipFile(source) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    result = []
    for paragraph in root.findall(".//w:body/w:p", NS):
        text = ""
        red: set[int] = set()
        spans = []
        for run in paragraph.findall(".//w:r", NS):
            value = "".join(node.text or "" for node in run.findall(".//w:t", NS))
            color = run.find("w:rPr/w:color", NS)
            rgb = color.get(W + "val", "") if color is not None else ""
            is_red = (
                bool(re.fullmatch("[0-9A-Fa-f]{6}", rgb))
                and int(rgb[:2], 16) > 160
                and int(rgb[2:4], 16) < 100
                and int(rgb[4:], 16) < 100
            )
            if is_red:
                red.update(range(len(text), len(text) + len(value)))
                spans.append({"start": len(text), "end": len(text) + len(value), "text": value})
            text += value
        result.append({"text": text, "red_positions": sorted(red), "red_spans": spans})
    return result


def field_value(text: str, label: str) -> str:
    return text.removeprefix(label).strip("_ \t")


def extract(source: Path, mapping: dict[str, Any], template: Path | None = None) -> dict[str, Any]:
    records: dict[str, Any] = {}
    section = ""
    management: dict[str, list[dict[str, Any]]] = {}
    for paragraph in paragraphs(source):
        text = paragraph["text"]
        if re.match(r"[PM]\d\d ", text):
            section = text[:3]
        if section.startswith("M"):
            management.setdefault(section, []).append(paragraph)
        if not section.startswith("P"):
            continue
        row = records.setdefault(
            section,
            {
                "probe_id": section,
                "required_safety_check": None,
                "urgent_required": None,
                "preference": None,
                "preferred_condition": None,
                "A_serious_violation": None,
                "B_serious_violation": None,
                "action_comment": "",
                "reason": "",
                "candidate_comment": "",
                "risk_level": None,
                "raw_paragraphs": [],
                "source_role": "PROJECT_OWNER_FEEDBACK",
                "expert_gold": False,
            },
        )
        row["raw_paragraphs"].append(paragraph)
        red = set(paragraph["red_positions"])
        for label, key in (
            ("必须直接确认安全：", "required_safety_check"),
            ("必须立即行动：", "urgent_required"),
            ("A 严重不当：", "A_serious_violation"),
            ("B 严重不当：", "B_serious_violation"),
        ):
            if label in text:
                choice = red_choice(text, red, label, ("无法判断", "是", "否"))
                row[key] = {"是": True, "否": False}.get(choice or "")
        if text.startswith("更合适的候选："):
            row["preference"] = red_choice(
                text, red, "更合适的候选：", ("无法判断", "持平", "A", "B")
            )
        for label, key in (("可接受动作：", "action_comment"), ("判断依据：", "reason")):
            if text.startswith(label):
                row[key] = field_value(text, label)
        if "具体问题原句及理由：" in text:
            row["candidate_comment"] = text.split("具体问题原句及理由：", 1)[1].strip("_ ")
    for identifier, row in records.items():
        preference = row["preference"]
        if preference in {"A", "B"}:
            row["preferred_condition"] = mapping[identifier][preference]
        elif preference == "持平":
            row["preferred_condition"] = "TIE"
    issues = []
    for identifier, row in records.items():
        if row["preference"] is None:
            issues.append(
                {
                    "id": identifier,
                    "field": "preference",
                    "status": "MISSING",
                    "detail": "候选未标红，不根据不当选项自动推导偏好。",
                }
            )
        if "两者表述均可" in row["candidate_comment"] and (
            row["A_serious_violation"] is True or row["B_serious_violation"] is True
        ):
            issues.append(
                {
                    "id": identifier,
                    "field": "serious_violation",
                    "status": "CONFLICT",
                    "detail": "严重不当选项与两者表述均可冲突，保留原文。",
                }
            )
        if row["required_safety_check"] is True and (
            "上下文" in row["action_comment"]
            or ("继续陪伴" in row["action_comment"] and row["preference"] == "持平")
        ):
            issues.append(
                {
                    "id": identifier,
                    "field": "required_safety_check",
                    "status": "CONFLICT",
                    "detail": "安全询问选项与文字及候选偏好存在张力。",
                }
            )
    management_changes = {}
    if template is not None:
        original = paragraphs(template)
        original_section = ""
        original_management: dict[str, list[str]] = {}
        for paragraph in original:
            value = paragraph["text"]
            if re.match(r"[PM]\d\d ", value):
                original_section = value[:3]
            if original_section.startswith("M"):
                original_management.setdefault(original_section, []).append(value)
        management_changes = {
            key: [p["text"] for p in values] != original_management.get(key)
            or any(p["red_spans"] for p in values)
            for key, values in management.items()
        }
    management_count = sum(management_changes.values()) if template is not None else None
    preferences = Counter(row["preferred_condition"] or "UNMARKED" for row in records.values())
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_role": "PROJECT_OWNER_FEEDBACK",
        "reviewer_identity_or_qualification_provided": False,
        "expert_gold": False,
        "records": list(records.values()),
        "management_raw": management,
        "management_changes": management_changes,
        "management_answers_detected": management_count,
        "issues": issues,
        "summary": {
            "case_comments": sum(bool(r["action_comment"]) for r in records.values()),
            "safety_choices": sum(r["required_safety_check"] is not None for r in records.values()),
            "urgent_choices": sum(r["urgent_required"] is not None for r in records.values()),
            "preferences": dict(preferences),
            "management_answers": management_count,
            "issues": len(issues),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["mapping"]
    result = extract(args.source, mapping, args.template)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source-feedback.docx").write_bytes(args.source.read_bytes())
    for name, data in (("feedback.json", result),):
        with (args.output / name).open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
