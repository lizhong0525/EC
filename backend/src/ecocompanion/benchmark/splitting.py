"""Leakage-resistant deterministic splits for the canonical case suite."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ecocompanion.benchmark.cases import CaseValidationError, validate_cases
from ecocompanion.domain.enums import ExpressionStyle, RiskTrajectory, SocialEcology
from ecocompanion.domain.schemas import CaseSpecV1


class DatasetSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    PUBLIC_TEST = "public_test"
    HIDDEN_TEST = "hidden_test"


@dataclass(frozen=True, slots=True)
class CaseSplits:
    """Canonical 72/24/24/24 case partitions."""

    train: tuple[CaseSpecV1, ...]
    dev: tuple[CaseSpecV1, ...]
    public_test: tuple[CaseSpecV1, ...]
    hidden_test: tuple[CaseSpecV1, ...]

    def for_split(self, split: DatasetSplit) -> tuple[CaseSpecV1, ...]:
        mapping = {
            DatasetSplit.TRAIN: self.train,
            DatasetSplit.DEV: self.dev,
            DatasetSplit.PUBLIC_TEST: self.public_test,
            DatasetSplit.HIDDEN_TEST: self.hidden_test,
        }
        return mapping[split]

    def as_id_manifest(self) -> dict[str, tuple[str, ...]]:
        return {
            split.value: tuple(case.case_id for case in self.for_split(split))
            for split in DatasetSplit
        }

    def split_for_case(self, case_id: str) -> DatasetSplit:
        for split in DatasetSplit:
            if any(case.case_id == case_id for case in self.for_split(split)):
                return split
        raise KeyError(case_id)

    def validate(self) -> None:
        expected_sizes = {
            DatasetSplit.TRAIN: 72,
            DatasetSplit.DEV: 24,
            DatasetSplit.PUBLIC_TEST: 24,
            DatasetSplit.HIDDEN_TEST: 24,
        }
        all_ids: list[str] = []
        for split, expected_size in expected_sizes.items():
            cases = self.for_split(split)
            if len(cases) != expected_size:
                raise CaseValidationError(
                    f"{split.value} contains {len(cases)} cases; expected {expected_size}"
                )
            all_ids.extend(case.case_id for case in cases)
        if len(all_ids) != len(set(all_ids)):
            raise CaseValidationError("case splits overlap")
        if len(all_ids) != 144:
            raise CaseValidationError("canonical split must cover exactly 144 cases")
        family_owners: dict[str, DatasetSplit] = {}
        for split in DatasetSplit:
            for case in self.for_split(split):
                owner = family_owners.setdefault(case.event_template_family, split)
                if owner is not split:
                    raise CaseValidationError(
                        "event_template_family crosses dataset splits: "
                        f"{case.event_template_family}"
                    )


def canonical_split(cases: tuple[CaseSpecV1, ...]) -> CaseSplits:
    """Split two replicates per factorial cell into 72/24/24/24 cases.

    The first deterministic replicate from each of the 72 cells forms training.
    The second replicate is assigned to one of three evaluation partitions using
    a balanced index rotation. Surface rewrites remain embedded in their parent
    ``CaseSpecV1`` and therefore cannot cross partitions.
    """

    validated = validate_cases(cases)
    groups: dict[
        tuple[RiskTrajectory, SocialEcology, ExpressionStyle],
        list[CaseSpecV1],
    ] = {}
    for case in validated:
        key = (
            case.risk_trajectory,
            case.social_ecology,
            case.expression_style,
        )
        groups.setdefault(key, []).append(case)

    expected_cells = (
        len(RiskTrajectory) * len(SocialEcology) * len(ExpressionStyle)
    )
    if len(groups) != expected_cells:
        raise CaseValidationError(
            f"canonical split requires {expected_cells} factorial cells; "
            f"found {len(groups)}"
        )

    train: list[CaseSpecV1] = []
    evaluation: dict[DatasetSplit, list[CaseSpecV1]] = {
        DatasetSplit.DEV: [],
        DatasetSplit.PUBLIC_TEST: [],
        DatasetSplit.HIDDEN_TEST: [],
    }
    evaluation_order = (
        DatasetSplit.DEV,
        DatasetSplit.PUBLIC_TEST,
        DatasetSplit.HIDDEN_TEST,
    )

    for key in sorted(groups, key=lambda item: tuple(part.value for part in item)):
        cell_cases = sorted(groups[key], key=lambda case: case.case_id)
        if len(cell_cases) != 2:
            raise CaseValidationError(
                "canonical split requires exactly two cases per "
                f"risk/ecology/expression cell; {key} has {len(cell_cases)}"
            )
        train.append(cell_cases[0])

        risk_index = list(RiskTrajectory).index(key[0])
        ecology_index = list(SocialEcology).index(key[1])
        # All expression variants of one risk/ecology event-template family
        # remain in the same evaluation partition. This prevents the shared
        # event template from leaking across dev/public/hidden test.
        target = evaluation_order[(risk_index + ecology_index) % 3]
        evaluation[target].append(cell_cases[1])

    partitions = CaseSplits(
        train=tuple(sorted(train, key=lambda case: case.case_id)),
        dev=tuple(sorted(evaluation[DatasetSplit.DEV], key=lambda case: case.case_id)),
        public_test=tuple(
            sorted(evaluation[DatasetSplit.PUBLIC_TEST], key=lambda case: case.case_id)
        ),
        hidden_test=tuple(
            sorted(evaluation[DatasetSplit.HIDDEN_TEST], key=lambda case: case.case_id)
        ),
    )
    partitions.validate()
    return partitions
