"""Deterministic cross-split near-duplicate audits for synthetic text."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import-untyped]

from ecocompanion.benchmark.splitting import CaseSplits, DatasetSplit

TextKind = Literal["utterance", "event"]
np: Any = import_module("numpy")


class _FrozenAuditModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextReferenceV1(_FrozenAuditModel):
    split: DatasetSplit
    kind: TextKind
    case_id: str
    observation_id: str
    surface_index: int | None = Field(default=None, ge=0)
    text: str = Field(min_length=1)


class CrossSplitSimilarityV1(_FrozenAuditModel):
    left_split: DatasetSplit
    right_split: DatasetSplit
    kind: TextKind
    compared_pair_count: int = Field(ge=0)
    maximum_jaccard: float = Field(ge=0.0, le=1.0)
    pairs_at_or_above_threshold: int = Field(ge=0)
    worst_left: TextReferenceV1 | None = None
    worst_right: TextReferenceV1 | None = None


class NearDuplicateAuditV1(_FrozenAuditModel):
    schema_version: Literal["1.0"] = "1.0"
    ngram_size: int = Field(ge=1)
    threshold: float = Field(gt=0.0, le=1.0)
    comparisons: tuple[CrossSplitSimilarityV1, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class _TextRecord:
    reference: TextReferenceV1
    normalized: str


def audit_near_duplicate_leakage(
    splits: CaseSplits,
    *,
    ngram_size: int = 3,
    threshold: float = 0.85,
) -> NearDuplicateAuditV1:
    """Audit character n-gram Jaccard overlap without quadratic Python loops."""

    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    comparisons = tuple(
        comparison
        for kind in ("utterance", "event")
        for comparison in _audit_kind(
            splits,
            kind=kind,
            ngram_size=ngram_size,
            threshold=threshold,
        )
    )
    return NearDuplicateAuditV1(
        ngram_size=ngram_size,
        threshold=threshold,
        comparisons=comparisons,
        passed=all(
            comparison.pairs_at_or_above_threshold == 0
            for comparison in comparisons
        ),
    )


def _audit_kind(
    splits: CaseSplits,
    *,
    kind: TextKind,
    ngram_size: int,
    threshold: float,
) -> tuple[CrossSplitSimilarityV1, ...]:
    records_by_split = {
        split: _records_for_split(splits, split=split, kind=kind)
        for split in DatasetSplit
    }
    ordered_records = tuple(
        record
        for split in DatasetSplit
        for record in records_by_split[split]
    )
    if not ordered_records:
        return ()
    boundaries: dict[DatasetSplit, tuple[int, int]] = {}
    offset = 0
    for split in DatasetSplit:
        count = len(records_by_split[split])
        boundaries[split] = (offset, offset + count)
        offset += count

    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(ngram_size, ngram_size),
        binary=True,
        dtype=np.uint8,
    )
    matrix = vectorizer.fit_transform(
        [record.normalized for record in ordered_records]
    ).astype(np.int32)
    lengths = np.asarray(matrix.sum(axis=1)).ravel()

    comparisons: list[CrossSplitSimilarityV1] = []
    split_order = tuple(DatasetSplit)
    for left_index, left_split in enumerate(split_order):
        for right_split in split_order[left_index + 1 :]:
            left_start, left_end = boundaries[left_split]
            right_start, right_end = boundaries[right_split]
            intersections = (
                matrix[left_start:left_end]
                @ matrix[right_start:right_end].T
            ).tocoo()
            compared_pair_count = (left_end - left_start) * (
                right_end - right_start
            )
            if intersections.nnz == 0:
                comparisons.append(
                    CrossSplitSimilarityV1(
                        left_split=left_split,
                        right_split=right_split,
                        kind=kind,
                        compared_pair_count=compared_pair_count,
                        maximum_jaccard=0.0,
                        pairs_at_or_above_threshold=0,
                    )
                )
                continue

            unions = (
                lengths[left_start:left_end][intersections.row]
                + lengths[right_start:right_end][intersections.col]
                - intersections.data
            )
            scores = intersections.data / unions
            maximum_index = int(np.argmax(scores))
            left_record = records_by_split[left_split][
                int(intersections.row[maximum_index])
            ]
            right_record = records_by_split[right_split][
                int(intersections.col[maximum_index])
            ]
            comparisons.append(
                CrossSplitSimilarityV1(
                    left_split=left_split,
                    right_split=right_split,
                    kind=kind,
                    compared_pair_count=compared_pair_count,
                    maximum_jaccard=float(scores[maximum_index]),
                    pairs_at_or_above_threshold=int(
                        np.count_nonzero(scores >= threshold)
                    ),
                    worst_left=left_record.reference,
                    worst_right=right_record.reference,
                )
            )
    return tuple(comparisons)


def _records_for_split(
    splits: CaseSplits,
    *,
    split: DatasetSplit,
    kind: TextKind,
) -> tuple[_TextRecord, ...]:
    records: list[_TextRecord] = []
    for case in splits.for_split(split):
        for step in case.timeline:
            texts = (
                step.frozen_utterances
                if kind == "utterance"
                else step.external_events
            )
            for index, text in enumerate(texts):
                reference = TextReferenceV1(
                    split=split,
                    kind=kind,
                    case_id=case.case_id,
                    observation_id=step.observation_id,
                    surface_index=index if kind == "utterance" else None,
                    text=text,
                )
                records.append(
                    _TextRecord(
                        reference=reference,
                        normalized=_normalize(text),
                    )
                )
    return tuple(records)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character for character in normalized if not character.isspace()
    )
