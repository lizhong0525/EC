"""Standalone benchmark data-generation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecocompanion.benchmark.cases import dump_cases_jsonl
from ecocompanion.benchmark.generation import (
    DEFAULT_CASE_SEED,
    generate_canonical_cases,
)
from ecocompanion.benchmark.splitting import canonical_split


def generate_example_dataset(
    output: str | Path,
    *,
    seed: int = DEFAULT_CASE_SEED,
    limit: int | None = 4,
) -> Path:
    """Generate a small auditable JSONL sample and adjacent split manifest."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")

    cases = generate_canonical_cases(seed=seed)
    splits = canonical_split(cases)
    selected = cases if limit is None else cases[:limit]
    destination = dump_cases_jsonl(selected, output)

    manifest = {
        "benchmark_version": "ecotriage-v1",
        "seed": seed,
        "canonical_case_count": len(cases),
        "exported_case_count": len(selected),
        "cases": {
            case.case_id: splits.split_for_case(case.case_id).value
            for case in selected
        },
    }
    manifest_path = destination.with_suffix(".splits.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ecocompanion.benchmark",
        description="Generate deterministic EcoTriage-Bench example data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CASE_SEED,
        help="Deterministic canonical generation seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Number of cases to export. Use --full for all 144 cases.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export the complete canonical suite.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_example_dataset(
        args.output,
        seed=args.seed,
        limit=None if args.full else args.limit,
    )
    return 0
