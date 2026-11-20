"""Bounded local-only Benchmark execution for the research Web console."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ecocompanion.application.benchmark_catalog import BenchmarkCatalog
from ecocompanion.application.models import (
    BenchmarkJobView,
    BenchmarkReportView,
    BenchmarkStartRequest,
)
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.benchmark.track_c import TrackCConfiguration
from ecocompanion.experiments.interactive_runner import (
    start_track_b_experiment,
    start_track_c_experiment,
)
from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.experiments.report import build_report
from ecocompanion.experiments.runner import start_experiment

logger = logging.getLogger(__name__)


class BenchmarkBusyError(RuntimeError):
    """Only one local Benchmark operation may run at a time."""


class BenchmarkJobs:
    def __init__(self, root: Path, catalog: BenchmarkCatalog) -> None:
        self._root = root.resolve()
        self._catalog = catalog
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._busy = False

    def start(self, configuration: BenchmarkStartRequest) -> BenchmarkJobView:
        if self._busy:
            raise BenchmarkBusyError("已有测试或报告正在生成，请等待完成。")
        job_id = uuid4().hex
        now = datetime.now(UTC)
        run_id = f"web-{configuration.track.lower()}-{now:%Y%m%d-%H%M%S}-{job_id[:8]}"
        job = BenchmarkJobView(
            job_id=job_id,
            run_id=run_id,
            status="queued",
            created_at=now,
            configuration=configuration,
        )
        self._save(job)
        self._busy = True
        self._tasks[job_id] = asyncio.create_task(self._execute(job))
        self._tasks[job_id].add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return job

    def get(self, job_id: str) -> BenchmarkJobView:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError(job_id)
        path = self._root / "benchmark-jobs" / f"{job_id}.json"
        if not path.is_file() or path.is_symlink():
            raise KeyError(job_id)
        job = BenchmarkJobView.model_validate(read_json_object(path))
        if job.status in {"queued", "running", "reporting"} and job_id not in self._tasks:
            job = job.model_copy(
                update={"status": "interrupted", "error": "服务已重启，运行记录保留。请新建测试。"}
            )
            self._save(job)
        run = next(
            (item for item in self._catalog.list_runs().runs if item.run_id == job.run_id), None
        )
        return job.model_copy(update={"run": run})

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)

    async def build_existing_report(self, run_id: str) -> BenchmarkReportView:
        if self._busy:
            raise BenchmarkBusyError("已有测试或报告正在生成，请等待完成。")
        directory = self._catalog.get_run_directory(run_id)
        self._busy = True
        try:
            # Existing reports remain read-only; generating twice returns the same artifact.
            if not (directory / "report.json").exists():
                await asyncio.to_thread(build_report, directory)
            return self._catalog.get_report(run_id)
        finally:
            self._busy = False

    def _save(self, job: BenchmarkJobView) -> None:
        write_json_atomic(self._root / "benchmark-jobs" / f"{job.job_id}.json", job, overwrite=True)

    async def _execute(self, job: BenchmarkJobView) -> None:
        directory = self._root / "runs" / job.run_id
        try:
            job = job.model_copy(update={"status": "running"})
            self._save(job)
            await asyncio.to_thread(self._run, directory, job.configuration)
            job = job.model_copy(update={"status": "reporting"})
            self._save(job)
            await asyncio.to_thread(build_report, directory)
            self._catalog.get_report(job.run_id)
            self._save(job.model_copy(update={"status": "complete"}))
        except Exception as exc:
            logger.exception("Web Benchmark failed: %s", job.run_id)
            self._save(
                job.model_copy(
                    update={
                        "status": "failed",
                        "error": f"生成失败（{type(exc).__name__}），运行记录已保留。",
                    }
                )
            )
        finally:
            self._busy = False

    @staticmethod
    def _run(directory: Path, configuration: BenchmarkStartRequest) -> None:
        if configuration.track == "A":
            start_experiment(
                run_dir=directory,
                split=DatasetSplit(configuration.split),
                policy_names=configuration.policies,
                surface_indices=(0, 1),
                repeats=1,
                case_limit=configuration.case_limit,
                run_id=directory.name,
            )
        elif configuration.track == "B":
            start_track_b_experiment(
                run_dir=directory,
                policy_names=configuration.policies,
                case_limit=configuration.case_limit,
                run_id=directory.name,
            )
        else:
            start_track_c_experiment(
                run_dir=directory,
                configurations=tuple(TrackCConfiguration),
                scenario_limit=configuration.case_limit,
                run_id=directory.name,
            )
