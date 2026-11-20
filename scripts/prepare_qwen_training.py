"""Validate provenance and create group-isolated supervised fine-tuning files."""

import json
from pathlib import Path

from ecocompanion.pet.training import deterministic_split, load_examples, write_sft


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = load_examples(
        root / "training/data/reviewed_examples.jsonl",
        root / "training/data/source_registry.json",
    )
    counts = write_sft(examples, deterministic_split(examples), root / "training/prepared")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
