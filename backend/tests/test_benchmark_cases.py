import json

import pytest

from ecocompanion.benchmark.cases import (
    CaseValidationError,
    dump_cases_jsonl,
    load_case,
    load_cases,
    validate_case,
)
from ecocompanion.benchmark.generation import (
    CANONICAL_CASE_COUNT,
    CANONICAL_DECISION_NODE_COUNT,
    CANONICAL_SURFACE_COUNT,
    CANONICAL_SURFACE_NODE_COUNT,
    STEPS_PER_CASE,
    generate_canonical_cases,
)
from ecocompanion.benchmark.splitting import DatasetSplit, canonical_split
from ecocompanion.domain.enums import ExpressionStyle, RiskTrajectory, SocialEcology
from ecocompanion.domain.schemas import CaseSpecV1


def test_canonical_generator_is_deterministic_and_complete() -> None:
    first = generate_canonical_cases(seed=1234)
    second = generate_canonical_cases(seed=1234)

    assert len(first) == CANONICAL_CASE_COUNT == 144
    assert CANONICAL_SURFACE_COUNT == 288
    assert CANONICAL_DECISION_NODE_COUNT == 2016
    assert CANONICAL_SURFACE_NODE_COUNT == 4032
    assert tuple(case.model_dump_json() for case in first) == tuple(
        case.model_dump_json() for case in second
    )
    assert {
        (
            case.risk_trajectory,
            case.social_ecology,
            case.expression_style,
        )
        for case in first
    } == {
        (trajectory, ecology, style)
        for trajectory in RiskTrajectory
        for ecology in SocialEcology
        for style in ExpressionStyle
    }
    assert all(len(case.timeline) == STEPS_PER_CASE for case in first)
    assert all(
        len(step.frozen_utterances) == 2
        for case in first
        for step in case.timeline
    )


def test_canonical_split_is_balanced_and_non_overlapping() -> None:
    cases = generate_canonical_cases()
    splits = canonical_split(cases)

    assert len(splits.train) == 72
    assert len(splits.dev) == 24
    assert len(splits.public_test) == 24
    assert len(splits.hidden_test) == 24

    all_ids = [
        case.case_id
        for split in DatasetSplit
        for case in splits.for_split(split)
    ]
    assert len(all_ids) == len(set(all_ids)) == 144
    for evaluation_split in (
        DatasetSplit.DEV,
        DatasetSplit.PUBLIC_TEST,
        DatasetSplit.HIDDEN_TEST,
    ):
        partition = splits.for_split(evaluation_split)
        assert set(case.risk_trajectory for case in partition) == set(RiskTrajectory)
        assert set(case.social_ecology for case in partition) == set(SocialEcology)
        assert set(case.expression_style for case in partition) == set(ExpressionStyle)

    family_owners: dict[str, DatasetSplit] = {}
    for split in DatasetSplit:
        for case in splits.for_split(split):
            assert family_owners.setdefault(case.event_template_family, split) is split


def test_surface_rewrites_cannot_cross_case_split() -> None:
    cases = generate_canonical_cases()
    splits = canonical_split(cases)
    case = cases[17]

    surface_episode_ids = {
        f"{case.case_id}-surface-{surface_index:02d}"
        for surface_index in range(2)
    }
    assert len(surface_episode_ids) == 2
    assert splits.split_for_case(case.case_id) in set(DatasetSplit)
    assert all(episode_id.startswith(case.case_id) for episode_id in surface_episode_ids)


def test_case_jsonl_round_trip(tmp_path) -> None:
    cases = generate_canonical_cases()[:3]
    destination = dump_cases_jsonl(cases, tmp_path / "cases.jsonl")

    loaded = load_cases(destination)

    assert loaded == cases
    assert destination.read_text(encoding="utf-8").count("\n") == 3


def test_json_array_loader(tmp_path) -> None:
    case = generate_canonical_cases()[0]
    source = tmp_path / "case.json"
    source.write_text(
        json.dumps([case.model_dump(mode="json")], ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_cases(source) == (case,)


def test_single_case_loader_rejects_ambiguous_collection(tmp_path) -> None:
    destination = dump_cases_jsonl(
        generate_canonical_cases()[:2],
        tmp_path / "cases.jsonl",
    )

    with pytest.raises(CaseValidationError, match="exactly one"):
        load_case(destination)


def test_benchmark_validation_rejects_nonconsecutive_timeline() -> None:
    payload = generate_canonical_cases()[0].model_dump(mode="json")
    payload["timeline"].pop(1)
    schema_valid_case = CaseSpecV1.model_validate(payload)

    with pytest.raises(CaseValidationError, match="consecutive"):
        validate_case(schema_valid_case)


def test_loader_rejects_invalid_json(tmp_path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text('{"not": "closed"\n', encoding="utf-8")

    with pytest.raises(CaseValidationError, match="invalid JSON"):
        load_cases(source)
