"""Offline tests for the external calibration packet and agreement metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import external_calibration as cal


def test_fleiss_kappa_perfect_and_negative() -> None:
    assert cal.fleiss_kappa([[0, 0], [1, 1], [0, 0], [1, 1]]) == 1.0
    assert cal.fleiss_kappa([[0, 1], [1, 0], [0, 1], [1, 0]]) == -1.0


def test_cohen_kappa_bounds() -> None:
    assert cal.cohen_kappa([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0
    assert cal.cohen_kappa([0, 1, 1, 0], [1, 0, 0, 1]) == -1.0
    with pytest.raises(ValueError):
        cal.cohen_kappa([0, 1], [0])


def test_score_compares_two_raters_and_a_judge(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    items = [
        {
            "item_id": f"item-{index:03d}",
            "user": "u",
            "assistant": "a",
            "source": {"dataset": "counselbench-eval"},
            "anchor": {"empathy_score": 5.0 if index % 2 else 2.0, "overall_score": 3.0},
        }
        for index in range(4)
    ]
    (packet / "items.jsonl").write_text(
        "\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8"
    )
    labels = [0, 1, 0, 1]
    for rater in (1, 2):
        with (tmp_path / f"r{rater}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item_id", *cal.DIMENSIONS])
            for index, item in enumerate(items):
                writer.writerow([item["item_id"], *[labels[index]] * len(cal.DIMENSIONS)])
    judge = tmp_path / "judge.jsonl"
    judge.write_text(
        "\n".join(
            json.dumps(
                {"item_id": item["item_id"], "scores": {d: labels[i] for d in cal.DIMENSIONS}}
            )
            for i, item in enumerate(items)
        )
        + "\n",
        encoding="utf-8",
    )
    report = cal.score(
        packet, [tmp_path / "r1.csv", tmp_path / "r2.csv"], judge, tmp_path / "o.json"
    )
    assert report["items"] == 4
    assert report["raters"] == 2
    assert report["by_dimension"]["empathy"]["fleiss_kappa"] == 1.0
    assert report["by_dimension"]["empathy"]["judge_vs_human"]["cohen_kappa"] == 1.0
    assert report["external_anchor"]["human_vs_anchor"]["agreement"] == 1.0
