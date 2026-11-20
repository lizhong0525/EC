"""Read-only benchmark artifact discovery for the research Web console."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ecocompanion.application.models import (
    BenchmarkReportView,
    BenchmarkRunCatalogView,
    BenchmarkRunView,
)
from ecocompanion.domain.enums import Track
from ecocompanion.experiments.models import (
    ReportDocument,
    TrackBReportDocument,
    TrackCReportDocument,
)

_MANIFEST_NAME = "manifest.json"
_CHECKPOINT_NAME = "checkpoint.json"
_REPORT_NAME = "report.json"
class BenchmarkRunNotFound(KeyError):
    """Raised when a requested immutable run ID is absent from the catalog."""


class BenchmarkReportUnavailable(FileNotFoundError):
    """Raised when a known run has no generated report yet."""


class BenchmarkCatalog:
    """Query persisted run artifacts without creating or modifying files.

    Run identifiers are resolved by inspecting validated manifests rather than
    interpolating user input into filesystem paths. This keeps the HTTP query
    path independent from directory traversal and symlink tricks.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def list_runs(self) -> BenchmarkRunCatalogView:
        if not self._root.exists():
            return BenchmarkRunCatalogView(runs=())
        if not self._root.is_dir():
            raise NotADirectoryError(self._root)

        runs: list[BenchmarkRunView] = []
        ignored: list[str] = []
        for candidate in sorted(self._root.iterdir(), key=lambda path: path.name):
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            try:
                runs.append(self._read_summary(candidate))
            except (OSError, ValueError, ValidationError, json.JSONDecodeError):
                ignored.append(candidate.name)
        runs.sort(key=lambda run: (run.created_at, run.run_id), reverse=True)
        return BenchmarkRunCatalogView(
            runs=tuple(runs),
            ignored_directory_names=tuple(ignored),
        )

    def get_report(self, run_id: str) -> BenchmarkReportView:
        run_dir = self.get_run_directory(run_id)
        summary = self._read_summary(run_dir)
        report_path = run_dir / _REPORT_NAME
        if not summary.report_available or not report_path.is_file():
            raise BenchmarkReportUnavailable(run_id)
        if report_path.is_symlink():
            raise ValueError("report symlinks are not supported")

        payload = _read_object(report_path)
        report = _validate_report(Track(summary.track), payload)
        if report.run_id != summary.run_id:
            raise ValueError("report run_id does not match its manifest")
        if report.completed_unit_count != summary.completed_unit_count:
            raise ValueError("report completed count does not match checkpoint")
        if report.planned_unit_count != summary.planned_unit_count:
            raise ValueError("report planned count does not match checkpoint")
        return BenchmarkReportView(
            run=summary,
            report=report.model_dump(mode="json"),
        )

    def get_run_directory(self, run_id: str) -> Path:
        """Resolve an existing run through its validated manifest, never a user path."""

        normalized = run_id.strip()
        if not normalized:
            raise BenchmarkRunNotFound(run_id)
        matches = tuple(
            run
            for run in self.list_runs().runs
            if run.run_id == normalized
        )
        if not matches:
            raise BenchmarkRunNotFound(normalized)
        if len(matches) != 1:
            raise ValueError(f"duplicate benchmark run_id: {normalized}")
        summary = matches[0]
        run_dir = self._root / summary.directory_name
        if run_dir.is_symlink() or not run_dir.resolve().is_relative_to(self._root):
            raise ValueError("run must stay within the benchmark directory")
        return run_dir

    def _read_summary(self, run_dir: Path) -> BenchmarkRunView:
        manifest = _read_object(run_dir / _MANIFEST_NAME)
        checkpoint = _read_object(run_dir / _CHECKPOINT_NAME)
        run_id = _require_string(manifest, "run_id")
        track = Track(_require_string(manifest, "track", default=Track.A.value))
        created_at = datetime.fromisoformat(_require_string(manifest, "created_at"))
        planned_ids = _require_string_sequence(manifest, "planned_unit_ids")
        completed_ids = _require_string_sequence(
            checkpoint,
            "completed_unit_ids",
            allow_empty=True,
        )
        if len(planned_ids) != len(set(planned_ids)):
            raise ValueError("planned unit IDs are not unique")
        if len(completed_ids) != len(set(completed_ids)):
            raise ValueError("completed unit IDs are not unique")
        if not set(completed_ids).issubset(planned_ids):
            raise ValueError("checkpoint contains unplanned completed units")
        completed_count = _require_integer(checkpoint, "completed_count")
        planned_count = _require_integer(checkpoint, "planned_count")
        if completed_count != len(completed_ids):
            raise ValueError("checkpoint completed count is inconsistent")
        if planned_count != len(planned_ids):
            raise ValueError("checkpoint planned count is inconsistent")
        methods = _methods_for_track(manifest, track)
        return BenchmarkRunView(
            run_id=run_id,
            directory_name=run_dir.name,
            track=track.value,
            created_at=created_at,
            status="complete" if completed_count == planned_count else "in_progress",
            completed_unit_count=completed_count,
            planned_unit_count=planned_count,
            methods=methods,
            report_available=(run_dir / _REPORT_NAME).is_file(),
        )


def _methods_for_track(
    manifest: Mapping[str, Any],
    track: Track,
) -> tuple[str, ...]:
    field = "configurations" if track is Track.C else "policy_names"
    values = _require_string_sequence(manifest, field)
    return tuple(values)


def _validate_report(
    track: Track,
    payload: Mapping[str, Any],
) -> ReportDocument | TrackBReportDocument | TrackCReportDocument:
    if track is Track.A:
        return ReportDocument.model_validate(payload)
    if track is Track.B:
        return TrackBReportDocument.model_validate(payload)
    return TrackCReportDocument.model_validate(payload)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_string(
    payload: Mapping[str, Any],
    field: str,
    *,
    default: str | None = None,
) -> str:
    value = payload.get(field, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_string_sequence(
    payload: Mapping[str, Any],
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field} must be an array")
    normalized = tuple(item for item in value if isinstance(item, str) and item)
    if len(normalized) != len(value):
        raise ValueError(f"{field} must contain only non-empty strings")
    if not normalized and not allow_empty:
        raise ValueError(f"{field} cannot be empty")
    return normalized
