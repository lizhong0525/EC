"""Dataset-level validation used by the research CLI."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from ecocompanion.benchmark.cases import load_cases, validate_cases
from ecocompanion.benchmark.generation import (
    DEFAULT_CASE_SEED,
    generate_canonical_cases,
)
from ecocompanion.benchmark.leakage import audit_near_duplicate_leakage
from ecocompanion.benchmark.splitting import CaseSplits, DatasetSplit, canonical_split
from ecocompanion.domain.schemas import CaseSpecV1
from ecocompanion.experiments.models import BenchmarkValidationReport


def load_benchmark_cases(
    *,
    cases_path: Path | None = None,
    seed: int,
) -> tuple[CaseSpecV1, ...]:
    """Load an explicit dataset or regenerate the canonical suite."""

    return (
        load_cases(cases_path)
        if cases_path is not None
        else generate_canonical_cases(seed=seed)
    )


def dataset_digest(cases: tuple[CaseSpecV1, ...]) -> str:
    """Hash complete canonical case content independent of file formatting."""

    documents = [
        case.model_dump(mode="json")
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    serialized = json.dumps(
        documents,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_benchmark(
    *,
    cases_path: Path | None = None,
    seed: int = DEFAULT_CASE_SEED,
) -> BenchmarkValidationReport:
    """Validate schemas, invariants, factorial coverage, surfaces, and splits."""

    cases = validate_cases(
        load_benchmark_cases(cases_path=cases_path, seed=seed)
    )
    splits = canonical_split(cases)
    minimum_surfaces = min(
        len(step.frozen_utterances)
        for case in cases
        for step in case.timeline
    )
    decision_nodes = sum(len(case.timeline) for case in cases)
    surface_trajectories = sum(
        min(len(step.frozen_utterances) for step in case.timeline)
        for case in cases
    )
    surface_nodes = sum(
        sum(len(step.frozen_utterances) for step in case.timeline)
        for case in cases
    )
    benchmark_versions = {case.benchmark_version for case in cases}
    if len(benchmark_versions) != 1:
        raise ValueError(
            f"dataset mixes benchmark versions: {sorted(benchmark_versions)}"
        )

    checks = (
        "all documents satisfy CaseSpecV1 and benchmark invariants",
        "case identifiers and observation identifiers are unique",
        "factorial cells contain exactly two canonical replicates",
        "train/dev/public_test/hidden_test splits are non-overlapping",
        "surface rewrites remain nested within their canonical case",
        "event-template families do not cross dataset splits",
        "normalized utterances, events, and persona identities do not cross splits",
        "character 3-gram cross-split near duplicates stay below 0.85 Jaccard",
    )
    leakage = _cross_split_leakage_counts(splits)
    near_duplicate_audit = audit_near_duplicate_leakage(
        splits,
        ngram_size=3,
        threshold=0.85,
    )
    passed = (
        all(count == 0 for count in leakage.values())
        and near_duplicate_audit.passed
    )
    return BenchmarkValidationReport(
        benchmark_version=next(iter(benchmark_versions)),
        canonical_case_count=len(cases),
        decision_node_count=decision_nodes,
        surface_trajectory_count=surface_trajectories,
        surface_decision_node_count=surface_nodes,
        minimum_surface_variants=minimum_surfaces,
        split_counts={
            split.value: len(splits.for_split(split))
            for split in DatasetSplit
        },
        risk_trajectory_counts=dict(
            sorted(Counter(case.risk_trajectory.value for case in cases).items())
        ),
        social_ecology_counts=dict(
            sorted(Counter(case.social_ecology.value for case in cases).items())
        ),
        expression_style_counts=dict(
            sorted(Counter(case.expression_style.value for case in cases).items())
        ),
        cross_split_template_family_count=leakage["template_family"],
        cross_split_exact_utterance_count=leakage["utterance"],
        cross_split_exact_event_count=leakage["event"],
        cross_split_persona_count=leakage["persona"],
        near_duplicate_audit=near_duplicate_audit,
        dataset_digest=dataset_digest(cases),
        passed=passed,
        checks=checks,
    )


def _cross_split_leakage_counts(
    splits: CaseSplits,
) -> dict[str, int]:
    owners: dict[str, dict[str, set[str]]] = {
        "template_family": {},
        "utterance": {},
        "event": {},
        "persona": {},
    }
    for split in DatasetSplit:
        split_name = split.value
        for case in splits.for_split(split):
            values = {
                "template_family": (case.event_template_family,),
                "utterance": tuple(
                    utterance
                    for step in case.timeline
                    for utterance in step.frozen_utterances
                ),
                "event": tuple(
                    event
                    for step in case.timeline
                    for event in step.external_events
                ),
                "persona": ("|".join(case.persona_tags),),
            }
            for category, category_values in values.items():
                for value in category_values:
                    normalized = _normalize_fingerprint(value)
                    owners[category].setdefault(normalized, set()).add(split_name)
    return {
        category: sum(len(split_owners) > 1 for split_owners in values.values())
        for category, values in owners.items()
    }


def _normalize_fingerprint(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())
