"""Validated, provenance-first data preparation for Qwen evidence extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecocompanion.pet.evidence import QwenAssessment, qwen_messages


class TrainingExample(BaseModel):
    """A reviewed task example, never a diagnosis or a user-chat export."""

    model_config = ConfigDict(extra="forbid")
    example_id: str = Field(pattern=r"^[A-Za-z0-9._-]{3,96}$")
    group_id: str = Field(min_length=3, max_length=96)
    text: str = Field(min_length=1, max_length=2000)
    target: QwenAssessment
    source_id: str
    reviewer: str = Field(min_length=2, max_length=96)
    review_status: Literal["reviewed", "double_reviewed"]
    scenario: Literal[
        "ordinary",
        "third_party",
        "past",
        "negation",
        "metaphor",
        "quotation",
        "physical_health",
        "work_stress",
        "current_safety",
    ]

    @model_validator(mode="after")
    def grounded(self) -> TrainingExample:
        if any(not quote or quote not in self.text for quote in self.target.evidence_quotes):
            raise ValueError(f"{self.example_id}: evidence must be exact quotes")
        if self.target.safety_signal == "current" and not self.target.evidence_quotes:
            raise ValueError(f"{self.example_id}: current requires evidence")
        return self


def read_registry(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("registry sources must be a list")
    result = {}
    for source in sources:
        if source["source_id"] in result:
            raise ValueError("duplicate source_id")
        result[source["source_id"]] = source
    return result


def load_examples(path: Path, registry_path: Path) -> list[TrainingExample]:
    registry = read_registry(registry_path)
    examples, ids, groups = [], set(), {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            example = TrainingExample.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        source = registry.get(example.source_id)
        if source is None or source.get("training_allowed") is not True:
            raise ValueError(
                f"{example.example_id}: source is not explicitly approved for training"
            )
        if not source.get("reviewed_at") or not source.get("license_url"):
            raise ValueError(f"{example.example_id}: source provenance is incomplete")
        if example.example_id in ids:
            raise ValueError(f"duplicate example_id: {example.example_id}")
        ids.add(example.example_id)
        normalized = " ".join(example.text.split()).casefold()
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        if fingerprint in groups and groups[fingerprint] != example.group_id:
            raise ValueError("identical text must share one group_id")
        groups[fingerprint] = example.group_id
        examples.append(example)
    if not examples:
        raise ValueError("no reviewed examples found")
    return examples


def deterministic_split(examples: list[TrainingExample]) -> dict[str, list[str]]:
    groups = sorted({example.group_id for example in examples})
    buckets: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for group in groups:
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        buckets[split].append(group)
    if any(not group_ids for group_ids in buckets.values()):
        raise ValueError("at least three independent groups are required for train/validation/test")
    # Aim to exercise every boundary; never move an entire validation group into test.
    scenarios = {example.scenario for example in examples}
    test_scenarios = {
        example.scenario for example in examples if example.group_id in buckets["test"]
    }
    if scenarios != test_scenarios:
        missing = scenarios - test_scenarios
        chosen: list[str] = []
        for scenario in sorted(missing):
            group = next(
                (
                    group
                    for group in sorted(buckets["train"])
                    if group not in chosen
                    and any(
                        example.group_id == group and example.scenario == scenario
                        for example in examples
                    )
                ),
                None,
            )
            if group is not None:
                chosen.append(group)
        buckets["train"] = [group for group in buckets["train"] if group not in chosen]
        buckets["test"].extend(chosen)
    return buckets


def write_sft(
    examples: list[TrainingExample],
    splits: dict[str, list[str]],
    output: Path,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, groups in splits.items():
        rows = [example for example in examples if example.group_id in groups]
        counts[split] = len(rows)
        with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
            for example in rows:
                messages = qwen_messages(example.text)
                messages.append({"role": "assistant", "content": example.target.model_dump_json()})
                target.write(
                    json.dumps(
                        {
                            "messages": messages,
                            "example_id": example.example_id,
                            "group_id": example.group_id,
                            "source_id": example.source_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    (output / "groups.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return counts
