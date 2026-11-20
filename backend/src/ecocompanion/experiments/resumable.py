"""Shared progress invariants for append-only benchmark runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ecocompanion.experiments.io import ArtifactConflictError
from ecocompanion.experiments.models import (
    ExperimentCheckpoint,
    ExperimentStatus,
)


class ResumableManifest(Protocol):
    """Minimum immutable manifest contract needed by checkpoint logic."""

    run_fingerprint: str
    planned_unit_ids: tuple[str, ...]


def validate_max_units(max_units: int | None) -> None:
    if max_units is not None and max_units <= 0:
        raise ValueError("max_units must be positive")


def build_checkpoint(
    manifest: ResumableManifest,
    completed_unit_ids: Sequence[str],
) -> ExperimentCheckpoint:
    """Derive the replaceable checkpoint solely from durable result IDs."""

    completed_set = set(completed_unit_ids)
    completed = tuple(
        unit_id
        for unit_id in manifest.planned_unit_ids
        if unit_id in completed_set
    )
    status = (
        ExperimentStatus.COMPLETE
        if len(completed) == len(manifest.planned_unit_ids)
        else ExperimentStatus.IN_PROGRESS
    )
    return ExperimentCheckpoint(
        run_fingerprint=manifest.run_fingerprint,
        status=status,
        completed_unit_ids=completed,
        completed_count=len(completed),
        planned_count=len(manifest.planned_unit_ids),
    )


def validate_checkpoint(
    checkpoint: ExperimentCheckpoint,
    manifest: ResumableManifest,
    result_ids: set[str],
) -> None:
    """Reject a checkpoint that cannot be derived from its manifest/results."""

    if checkpoint.run_fingerprint != manifest.run_fingerprint:
        raise ArtifactConflictError(
            "checkpoint fingerprint does not match manifest"
        )
    if checkpoint.planned_count != len(manifest.planned_unit_ids):
        raise ArtifactConflictError(
            "checkpoint planned_count does not match manifest"
        )
    missing_results = set(checkpoint.completed_unit_ids) - result_ids
    if missing_results:
        raise ArtifactConflictError(
            "checkpoint is ahead of durable results for units: "
            f"{sorted(missing_results)}"
        )
