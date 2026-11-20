from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import RequestResponseEndpoint

from ecocompanion.api import create_app
from ecocompanion.application import benchmark_jobs as jobs
from ecocompanion.application.benchmark_catalog import BenchmarkCatalog
from ecocompanion.application.models import BenchmarkStartRequest
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.config import Settings
from ecocompanion.experiments.runner import start_experiment

BASE = "/api/v1/research/benchmark"


def local_client(app: FastAPI, host: str = "127.0.0.1") -> TestClient:
    @app.middleware("http")
    async def set_test_peer(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.scope["client"] = (host, 50000)
        return await call_next(request)

    return TestClient(app)


def configuration(root: Path, *, research: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        LITELLM_API_KEY=None,
        ECO_DEFAULT_MODEL=None,
        ECO_RESEARCH_MODE=research,
        ECO_DATABASE_URL=f"sqlite+aiosqlite:///{root / 'app.db'}",
        ECO_ARTIFACTS_DIR=root,
    )


@pytest.fixture(autouse=True)
def fast_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    original = jobs.build_report

    def small_report(directory: Path) -> Any:
        return original(directory, bootstrap_resamples=10, permutation_resamples=10)

    monkeypatch.setattr(jobs, "build_report", small_report)


def wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(1000):
        response = client.get(f"{BASE}/jobs/{job_id}")
        assert response.status_code == 200
        payload: dict[str, Any] = response.json()
        if payload["status"] in {"complete", "failed", "interrupted"}:
            return payload
        time.sleep(0.01)
    pytest.fail("local job did not reach a terminal state")


@pytest.mark.parametrize("track", ["A", "B", "C"])
def test_run_build_and_download_all_local_tracks(tmp_path: Path, track: str) -> None:
    app = create_app(settings=configuration(tmp_path))
    with local_client(app) as client:
        started = client.post(
            f"{BASE}/jobs",
            json={"track": track, "case_limit": 1, "policies": ["history-aware-rule"]},
        )
        assert started.status_code == 202
        job = wait_for_job(client, started.json()["job_id"])
        assert job["status"] == "complete", job
        assert job["execution"] == "local_rules"
        assert job["model_calls"] == 0
        assert job["run"]["report_available"]
        assert job["run"]["completed_unit_count"] == job["run"]["planned_unit_count"]
        run_id = job["run_id"]
        report = client.get(f"{BASE}/runs/{run_id}/report")
        assert report.status_code == 200
        assert report.json()["report"]["complete"]
        downloaded = client.get(f"{BASE}/runs/{run_id}/download?format=json")
        assert downloaded.status_code == 200
        assert downloaded.json() == report.json()["report"]
        assert "attachment" in downloaded.headers["content-disposition"]
        markdown = client.get(f"{BASE}/runs/{run_id}/download?format=md")
        assert markdown.status_code == 200
        assert run_id in markdown.text
        assert client.get(f"{BASE}/runs/{run_id}/download?format=html").status_code == 422
        assert app.state.runtime.client is None


def test_build_existing_report_is_idempotent_and_unknown_ids_do_not_create_paths(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "existing"
    start_experiment(
        run_dir=run_dir,
        run_id="existing",
        split=DatasetSplit.DEV,
        policy_names=("always-listen",),
        surface_indices=(0,),
        repeats=1,
        case_limit=1,
    )
    with local_client(create_app(settings=configuration(tmp_path))) as client:
        assert client.get(f"{BASE}/runs/existing/report").status_code == 409
        first = client.post(f"{BASE}/runs/existing/report")
        assert first.status_code == 200
        original = (run_dir / "report.json").read_bytes()
        second = client.post(f"{BASE}/runs/existing/report")
        assert second.json() == first.json()
        assert (run_dir / "report.json").read_bytes() == original
        assert client.post(f"{BASE}/runs/not-found/report").status_code == 404
        assert client.get(f"{BASE}/jobs/not-a-job").status_code == 404
        assert not (tmp_path / "runs" / "not-found").exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"case_limit": 0},
        {"case_limit": 25},
        {"track": "C", "case_limit": 13},
        {"policies": []},
        {"policies": ["always-listen", "always-listen"]},
        {"policies": ["eco-safe"]},
        {"model": "remote-model"},
        {"cases_path": "../../private.json"},
        {"split": "hidden_test"},
    ],
)
def test_web_run_input_is_bounded_and_provider_free(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    with local_client(create_app(settings=configuration(tmp_path))) as client:
        assert client.post(f"{BASE}/jobs", json=payload).status_code == 422
        assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(("research", "host"), [(False, "127.0.0.1"), (True, "192.0.2.1")])
def test_mutations_require_local_research_operator(
    tmp_path: Path, research: bool, host: str
) -> None:
    with local_client(
        create_app(settings=configuration(tmp_path, research=research)), host=host
    ) as client:
        assert client.post(f"{BASE}/jobs", json={}).status_code == 403
        assert client.post(f"{BASE}/runs/anything/report").status_code == 403


def test_busy_run_keeps_api_responsive_and_rejects_duplicate_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = jobs.BenchmarkJobs._run

    def paused(directory: Path, request: BenchmarkStartRequest) -> None:
        entered.set()
        assert release.wait(10)
        original(directory, request)

    monkeypatch.setattr(jobs.BenchmarkJobs, "_run", staticmethod(paused))
    with local_client(create_app(settings=configuration(tmp_path))) as client:
        try:
            first = client.post(
                f"{BASE}/jobs", json={"case_limit": 1, "policies": ["always-listen"]}
            )
            assert first.status_code == 202
            assert entered.wait(5)
            assert client.get("/healthz").status_code == 200
            assert client.post(f"{BASE}/jobs", json={}).status_code == 409
            assert client.post(f"{BASE}/runs/anything/report").status_code == 409
        finally:
            release.set()
        assert wait_for_job(client, first.json()["job_id"])["status"] == "complete"


def test_failed_run_and_restart_remain_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(directory: Path, request: BenchmarkStartRequest) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(jobs.BenchmarkJobs, "_run", staticmethod(fail))
    with local_client(create_app(settings=configuration(tmp_path))) as client:
        first = client.post(f"{BASE}/jobs", json={})
        assert first.status_code == 202
        failed = wait_for_job(client, first.json()["job_id"])
        assert failed["status"] == "failed"
        assert "RuntimeError" in failed["error"]
        assert client.post(f"{BASE}/jobs", json={}).status_code == 202
    identifier = first.json()["job_id"]
    manager = jobs.BenchmarkJobs(tmp_path, BenchmarkCatalog(tmp_path / "runs"))
    record = manager.get(identifier)
    manager._save(record.model_copy(update={"status": "running", "error": None}))
    recovered = manager.get(identifier)
    assert recovered.status == "interrupted"
    assert recovered.error


def test_external_browser_origin_cannot_trigger_local_work(tmp_path: Path) -> None:
    with local_client(create_app(settings=configuration(tmp_path))) as client:
        for endpoint in (f"{BASE}/jobs", f"{BASE}/runs/anything/report"):
            assert (
                client.post(
                    endpoint, json={}, headers={"Origin": "https://unrelated.example"}
                ).status_code
                == 403
            )
        assert not (tmp_path / "runs").exists()
