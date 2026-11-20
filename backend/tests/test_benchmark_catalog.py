import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ecocompanion.application.benchmark_catalog import (
    BenchmarkCatalog,
    BenchmarkReportUnavailable,
    BenchmarkRunNotFound,
)


def _write_run(
    root: Path,
    *,
    directory_name: str,
    run_id: str,
    track: str,
    methods: tuple[str, ...],
    complete: bool,
    report: dict[str, object] | None = None,
) -> Path:
    run_dir = root / directory_name
    run_dir.mkdir(parents=True)
    planned = ("unit-1", "unit-2")
    completed = planned if complete else planned[:1]
    method_field = "configurations" if track == "C" else "policy_names"
    manifest = {
        "run_id": run_id,
        "track": track,
        "created_at": datetime(2026, 9, 3, tzinfo=UTC).isoformat(),
        "planned_unit_ids": list(planned),
        method_field: list(methods),
    }
    checkpoint = {
        "completed_unit_ids": list(completed),
        "completed_count": len(completed),
        "planned_count": len(planned),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (run_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )
    if report is not None:
        (run_dir / "report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
    return run_dir


def test_catalog_lists_only_valid_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_run(
        runs,
        directory_name="valid-a",
        run_id="run-a",
        track="A",
        methods=("eco-safe",),
        complete=False,
    )
    invalid = runs / "invalid"
    invalid.mkdir()
    (invalid / "manifest.json").write_text("{}", encoding="utf-8")

    catalog = BenchmarkCatalog(runs).list_runs()

    assert tuple(run.run_id for run in catalog.runs) == ("run-a",)
    assert catalog.runs[0].status == "in_progress"
    assert catalog.runs[0].methods == ("eco-safe",)
    assert catalog.ignored_directory_names == ("invalid",)


def test_catalog_ignores_symlinked_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    valid = _write_run(
        runs,
        directory_name="valid-a",
        run_id="run-a",
        track="A",
        methods=("eco-safe",),
        complete=False,
    )
    try:
        (runs / "symlink").symlink_to(valid, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - Windows default
        pytest.skip(f"symlink creation unavailable on this host: {exc}")

    catalog = BenchmarkCatalog(runs).list_runs()

    assert tuple(run.run_id for run in catalog.runs) == ("run-a",)


def test_catalog_fails_closed_for_unknown_or_missing_report(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _write_run(
        runs,
        directory_name="known-c",
        run_id="run-c",
        track="C",
        methods=("PLAIN_ASSISTANT",),
        complete=True,
    )
    catalog = BenchmarkCatalog(runs)

    with pytest.raises(BenchmarkRunNotFound):
        catalog.get_report("../unknown")
    with pytest.raises(BenchmarkReportUnavailable):
        catalog.get_report("run-c")
