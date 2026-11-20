"""Offline tests for the hosted judge reporting and comparison commands."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import deepseek_judge as judge


def _packet(tmp_path: Path, anchors: list[float]) -> Path:
    packet = tmp_path / "packet"
    packet.mkdir()
    items = [
        {
            "item_id": f"item-{index:03d}",
            "user": "u",
            "assistant": "a",
            "source": {"dataset": "counselbench-eval", "responder": "gpt4"},
            "anchor": {"empathy_score": anchor, "overall_score": 3.0},
        }
        for index, anchor in enumerate(anchors)
    ]
    (packet / "items.jsonl").write_text(
        "\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8"
    )
    return packet


def _judge(path: Path, scores: list[int], invalid: tuple[int, ...] = ()) -> Path:
    lines = []
    for index, score in enumerate(scores):
        record = {
            "item_id": f"item-{index:03d}",
            "status": "valid",
            "scores": {
                "empathy": score,
                "regulation": 0,
                "exploration": 0,
                "risk_inquiry": 0,
                "external_support": 1,
            },
        }
        if index in invalid:
            record = {"item_id": f"item-{index:03d}", "status": "INVALID:SCHEMA"}
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_report_scores_anchor_agreement(tmp_path: Path) -> None:
    packet = _packet(tmp_path, [5.0, 5.0, 2.0, 2.0])
    judge_path = _judge(tmp_path / "run.jsonl", [1, 1, 0, 0])
    summary = judge.report(packet, judge_path, tmp_path / "report.json")
    assert summary["valid"] == 4
    assert summary["external_anchor"]["agreement"] == 1.0
    assert summary["external_anchor"]["confusion"]["judge0_anchor1"] == 0
    assert summary["by_dimension"]["external_support"]["ones"] == 4


def test_report_ignores_invalid_records(tmp_path: Path) -> None:
    packet = _packet(tmp_path, [5.0, 2.0])
    judge_path = _judge(tmp_path / "run.jsonl", [1, 0], invalid=(1,))
    summary = judge.report(packet, judge_path, tmp_path / "report.json")
    assert summary["valid"] == 1
    assert summary["by_dimension"]["empathy"]["ones"] == 1


def test_compare_reports_self_consistency(tmp_path: Path) -> None:
    packet = _packet(tmp_path, [5.0, 2.0, 5.0, 2.0])
    first = _judge(tmp_path / "a.jsonl", [1, 0, 1, 0])
    second = _judge(tmp_path / "b.jsonl", [1, 0, 0, 0])
    summary = judge.compare(packet, first, second, tmp_path / "c.json")
    assert summary["common"] == 4
    assert summary["by_dimension"]["empathy"]["agreement"] == 0.75
    assert summary["by_dimension"]["regulation"]["agreement"] == 1.0
