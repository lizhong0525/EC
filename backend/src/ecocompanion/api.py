"""FastAPI entrypoint for the local EcoCompanion research demo."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from ecocompanion.agent import MemoryService, ProactivityNotAllowed
from ecocompanion.application.benchmark_catalog import (
    BenchmarkCatalog,
    BenchmarkReportUnavailable,
    BenchmarkRunNotFound,
)
from ecocompanion.application.benchmark_jobs import BenchmarkBusyError, BenchmarkJobs
from ecocompanion.application.models import (
    AuthorizedMemoryRequest,
    BenchmarkJobView,
    BenchmarkReportView,
    BenchmarkRunCatalogView,
    BenchmarkStartRequest,
    CreateSessionRequest,
    MemoryCorrectionRequest,
    MessageView,
    ModelSnapshotView,
    ProactiveMessageView,
    ProactivityUpdateRequest,
    SessionView,
    TurnRequest,
    TurnResult,
)
from ecocompanion.application.renderer import LLMResponseRenderer
from ecocompanion.application.repository import SqlAlchemySessionRepository
from ecocompanion.application.service import (
    CompanionService,
    ModelSelectionError,
    SessionNotFound,
    TurnConflictError,
)
from ecocompanion.benchmark import generate_canonical_cases
from ecocompanion.config import Settings, get_settings
from ecocompanion.infrastructure import CostLedger, JsonlTraceStore, ModelPricing
from ecocompanion.providers import OpenAICompatibleClient


@dataclass(slots=True)
class AppRuntime:
    settings: Settings
    service: CompanionService
    client: OpenAICompatibleClient | None
    snapshot_path: Path
    benchmark_catalog: BenchmarkCatalog
    benchmark_jobs: BenchmarkJobs

    async def initialize(self) -> None:
        await self.service.initialize()

    async def close(self) -> None:
        await self.benchmark_jobs.close()
        await self.service.close()
        if self.client is not None:
            await self.client.aclose()


def build_runtime(settings: Settings) -> AppRuntime:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _find_snapshot_path(settings.artifacts_dir)
    # An explicitly configured model is an intentional escape hatch for the
    # Web/demo runtime: qualification snapshots remain mandatory for formal
    # benchmark runs, but a local operator should be able to exercise the
    # complete API path immediately after setting environment variables.
    explicit_model = settings.default_model
    selected_model = explicit_model or _qualified_model_from_snapshot(snapshot_path)
    api_key = (
        settings.litellm_api_key.get_secret_value().strip()
        if settings.litellm_api_key is not None
        else ""
    )
    client: OpenAICompatibleClient | None = None
    active_model: str | None = None
    pricing: dict[str, ModelPricing] = {}
    if selected_model and api_key:
        if (
            explicit_model is None
            and (
                settings.model_input_price_cny_per_million is None
                or settings.model_output_price_cny_per_million is None
            )
        ):
            raise ValueError(
                "ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION and "
                "ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION are required when "
                "a snapshot-selected provider-backed Web model is enabled; "
                "explicit ECO_DEFAULT_MODEL may be used for Web smoke tests"
            )
        client = OpenAICompatibleClient(
            base_url=settings.litellm_base_url,
            api_key=api_key,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=settings.read_timeout_seconds,
            max_same_model_retries=settings.max_same_model_retries,
            default_extra_body=settings.provider_extra_body,
        )
        active_model = selected_model
        if (
            settings.model_input_price_cny_per_million is not None
            and settings.model_output_price_cny_per_million is not None
        ):
            pricing[selected_model] = ModelPricing(
                input_per_million=Decimal(str(settings.model_input_price_cny_per_million)),
                output_per_million=Decimal(str(settings.model_output_price_cny_per_million)),
            )

    repository = SqlAlchemySessionRepository(settings.database_url)
    renderer = LLMResponseRenderer(client=client, model=active_model)
    trace_store = JsonlTraceStore(settings.artifacts_dir / "traces" / "web.jsonl")
    cost_ledger = CostLedger(
        pricing=pricing,
        pricing_version=(
            (settings.model_pricing_version or "runtime-env-v1")
            if pricing
            else None
        ),
        path=settings.artifacts_dir / "costs" / "web.jsonl",
    )
    known_case_ids = frozenset(case.case_id for case in generate_canonical_cases())
    service = CompanionService(
        repository=repository,
        renderer=renderer,
        trace_store=trace_store,
        cost_ledger=cost_ledger,
        memory=MemoryService(
            path=settings.artifacts_dir / "memory" / "events.jsonl"
        ),
        known_synthetic_case_ids=known_case_ids,
        api_budget_cny=settings.api_budget_cny,
        research_mode_enabled=settings.research_mode,
        web_conversation_mode=True,
    )
    catalog = BenchmarkCatalog(settings.artifacts_dir / "runs")
    return AppRuntime(
        settings=settings,
        service=service,
        client=client,
        snapshot_path=snapshot_path,
        benchmark_catalog=catalog,
        benchmark_jobs=BenchmarkJobs(settings.artifacts_dir, catalog),
    )


def create_app(
    *,
    settings: Settings | None = None,
    runtime: AppRuntime | None = None,
) -> FastAPI:
    if settings is not None and runtime is not None:
        raise ValueError("pass settings or runtime, not both")
    effective_runtime = runtime or build_runtime(settings or get_settings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await effective_runtime.initialize()
        try:
            yield
        finally:
            await effective_runtime.close()

    application = FastAPI(
        title="EcoCompanion Research API",
        version="0.1.0",
        description=(
            "Synthetic safety research platform; not a diagnostic, treatment, "
            "or emergency-response service."
        ),
        lifespan=lifespan,
    )
    application.state.runtime = effective_runtime
    application.add_middleware(
        CORSMiddleware,
        allow_origins=(
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ),
        allow_credentials=False,
        allow_methods=("GET", "POST", "PUT", "DELETE"),
        allow_headers=("Content-Type", "Accept"),
    )

    @application.get("/healthz")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "research_mode": effective_runtime.settings.research_mode,
            "provider_configured": effective_runtime.client is not None,
        }

    @application.get("/api/v1/models", response_model=ModelSnapshotView)
    async def models() -> ModelSnapshotView:
        return _snapshot_view(effective_runtime)

    @application.post(
        "/api/v1/sessions",
        response_model=SessionView,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(payload: CreateSessionRequest) -> SessionView:
        try:
            return await effective_runtime.service.create_session(payload)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ModelSelectionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.get("/api/v1/sessions/{session_id}", response_model=SessionView)
    async def get_session(session_id: str) -> SessionView:
        try:
            return await effective_runtime.service.get_session(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @application.get(
        "/api/v1/sessions/{session_id}/messages",
        response_model=tuple[MessageView, ...],
    )
    async def list_messages(session_id: str) -> tuple[MessageView, ...]:
        try:
            return await effective_runtime.service.list_messages(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @application.post("/api/v1/sessions/{session_id}/turn", response_model=None)
    async def send_turn(
        session_id: str,
        payload: TurnRequest,
        request: Request,
    ) -> Any:
        try:
            result = await effective_runtime.service.process_turn(session_id, payload)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except TurnConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if "text/event-stream" not in request.headers.get("accept", ""):
            return result
        return StreamingResponse(
            _sse_result(result),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/api/v1/sessions/{session_id}/memory")
    async def list_memory(session_id: str) -> list[dict[str, Any]]:
        try:
            records = await effective_runtime.service.list_memories(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        return [record.model_dump(mode="json") for record in records]

    @application.post(
        "/api/v1/sessions/{session_id}/memory",
        status_code=status.HTTP_201_CREATED,
    )
    async def add_memory(
        session_id: str,
        payload: AuthorizedMemoryRequest,
    ) -> dict[str, Any]:
        try:
            record = await effective_runtime.service.add_authorized_memory(
                session_id,
                payload,
            )
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return record.model_dump(mode="json")

    @application.put("/api/v1/sessions/{session_id}/memory/{memory_id}")
    async def correct_memory(
        session_id: str,
        memory_id: str,
        payload: MemoryCorrectionRequest,
    ) -> dict[str, Any]:
        try:
            record = await effective_runtime.service.correct_memory(
                session_id,
                memory_id,
                payload,
            )
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="memory not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return record.model_dump(mode="json")

    @application.delete(
        "/api/v1/sessions/{session_id}/memory/{memory_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_memory(session_id: str, memory_id: str) -> Response:
        try:
            await effective_runtime.service.delete_memory(session_id, memory_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session or memory not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.put(
        "/api/v1/sessions/{session_id}/proactivity",
        response_model=SessionView,
    )
    async def set_proactivity(
        session_id: str,
        payload: ProactivityUpdateRequest,
    ) -> SessionView:
        try:
            return await effective_runtime.service.set_proactivity(
                session_id,
                enabled=payload.enabled,
            )
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except (PermissionError, ProactivityNotAllowed) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @application.post(
        "/api/v1/sessions/{session_id}/proactivity/resume",
        response_model=SessionView,
    )
    async def resume_proactivity(session_id: str) -> SessionView:
        try:
            return await effective_runtime.service.resume_proactivity(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except (PermissionError, ProactivityNotAllowed) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @application.post(
        "/api/v1/sessions/{session_id}/proactivity/claim",
        response_model=ProactiveMessageView,
    )
    async def claim_proactive_message(session_id: str) -> ProactiveMessageView:
        try:
            return await effective_runtime.service.claim_proactive_message(
                session_id
            )
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ProactivityNotAllowed as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/research/sessions/{session_id}/trace")
    async def list_trace(session_id: str) -> list[dict[str, Any]]:
        try:
            events = await effective_runtime.service.list_trace(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return [event.model_dump(mode="json") for event in events]

    @application.get(
        "/api/v1/research/benchmark/runs",
        response_model=BenchmarkRunCatalogView,
    )
    async def list_benchmark_runs() -> BenchmarkRunCatalogView:
        try:
            return effective_runtime.benchmark_catalog.list_runs()
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="benchmark artifact catalog is unavailable",
            ) from exc

    def require_benchmark_operator(request: Request) -> None:
        if not effective_runtime.settings.research_mode:
            raise HTTPException(status_code=403, detail="研究模式未启用。")
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(status_code=403, detail="仅允许本机启动 Benchmark。")
        origin = request.headers.get("origin")
        allowed_origins = {
            str(request.base_url).rstrip("/"),
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        }
        if origin is not None and origin not in allowed_origins:
            raise HTTPException(status_code=403, detail="请求来源不受支持。")

    @application.post(
        "/api/v1/research/benchmark/jobs",
        response_model=BenchmarkJobView,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_benchmark(
        payload: BenchmarkStartRequest, request: Request
    ) -> BenchmarkJobView:
        require_benchmark_operator(request)
        try:
            return effective_runtime.benchmark_jobs.start(payload)
        except BenchmarkBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get(
        "/api/v1/research/benchmark/jobs/{job_id}", response_model=BenchmarkJobView
    )
    async def get_benchmark_job(job_id: str) -> BenchmarkJobView:
        try:
            return effective_runtime.benchmark_jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="benchmark job not found") from exc

    @application.post(
        "/api/v1/research/benchmark/runs/{run_id}/report", response_model=BenchmarkReportView
    )
    async def generate_benchmark_report(run_id: str, request: Request) -> BenchmarkReportView:
        require_benchmark_operator(request)
        try:
            return await effective_runtime.benchmark_jobs.build_existing_report(run_id)
        except BenchmarkRunNotFound as exc:
            raise HTTPException(status_code=404, detail="benchmark run not found") from exc
        except (BenchmarkBusyError, ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/research/benchmark/runs/{run_id}/download")
    async def download_benchmark_report(
        run_id: str, format: Literal["json", "md"] = "md"
    ) -> FileResponse:
        try:
            effective_runtime.benchmark_catalog.get_report(run_id)
            directory = effective_runtime.benchmark_catalog.get_run_directory(run_id)
            path = directory / f"report.{format}"
            if not path.is_file() or path.is_symlink():
                raise BenchmarkReportUnavailable(run_id)
            return FileResponse(
                path,
                media_type="application/json" if format == "json" else "text/markdown",
                filename=f"{directory.name}.{format}",
            )
        except BenchmarkRunNotFound as exc:
            raise HTTPException(status_code=404, detail="benchmark run not found") from exc
        except (BenchmarkReportUnavailable, ValueError) as exc:
            raise HTTPException(status_code=409, detail="报告文件不可用。") from exc

    @application.get(
        "/api/v1/research/benchmark/runs/{run_id}/report",
        response_model=BenchmarkReportView,
    )
    async def get_benchmark_report(run_id: str) -> BenchmarkReportView:
        try:
            return effective_runtime.benchmark_catalog.get_report(run_id)
        except BenchmarkRunNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail="benchmark run not found",
            ) from exc
        except BenchmarkReportUnavailable as exc:
            raise HTTPException(
                status_code=409,
                detail="benchmark report has not been generated",
            ) from exc
        except (ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=409,
                detail=f"benchmark artifacts are inconsistent: {exc}",
            ) from exc

    return application


async def _sse_result(result: TurnResult) -> AsyncIterator[str]:
    # Safety-sensitive generation is buffered until the final audit passes.
    # The already-approved text is then progressively delivered to the UI.
    for start in range(0, len(result.response), 8):
        chunk = result.response[start : start + 8]
        yield _sse_frame("token", {"token": chunk})
    yield _sse_frame(
        "done",
        {"result": result.model_dump(mode="json")},
    )


def _sse_frame(event: str, payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {serialized}\n\n"


def _find_snapshot_path(artifacts_dir: Path) -> Path:
    preferred = artifacts_dir / "models" / "model_snapshot.json"
    legacy = artifacts_dir / "model_snapshot.json"
    if preferred.exists() or not legacy.exists():
        return preferred
    return legacy


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _qualified_model_from_snapshot(path: Path) -> str | None:
    snapshot = _read_snapshot(path)
    if snapshot is None:
        return None
    capabilities = snapshot.get("capabilities")
    if not isinstance(capabilities, list):
        return None
    ready = {
        str(report["model_id"])
        for report in capabilities
        if isinstance(report, dict)
        and report.get("status") == "ready"
        and report.get("basic_chat") is True
        and report.get("chinese_text") is True
        and isinstance(report.get("model_id"), str)
    }
    if not ready:
        return None

    qualifications = snapshot.get("qualifications")
    if isinstance(qualifications, list):
        ranked = sorted(
            (
                (
                    float(report["qualification_score"]),
                    str(report["model_id"]),
                )
                for report in qualifications
                if isinstance(report, dict)
                and isinstance(report.get("model_id"), str)
                and report["model_id"] in ready
                and isinstance(report.get("qualification_score"), int | float)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if ranked:
            return ranked[0][1]
    return min(ready)


def _snapshot_view(runtime: AppRuntime) -> ModelSnapshotView:
    provider_location: Literal["local", "remote", "unavailable"] = (
        "local"
        if runtime.client is not None
        and _is_local_provider_url(runtime.settings.litellm_base_url)
        else "remote" if runtime.client is not None else "unavailable"
    )
    snapshot = _read_snapshot(runtime.snapshot_path)
    if snapshot is None:
        if runtime.service.model is not None:
            return ModelSnapshotView(
                source="runtime",
                active_model=runtime.service.model,
                provider_location=provider_location,
                detail="runtime model configured without a persisted capability snapshot",
            )
        return ModelSnapshotView(
            source="unavailable",
            active_model=None,
            provider_location="unavailable",
            detail=(
                "No qualified model snapshot is available. Run "
                "`eco models qualify` after configuring LITELLM_API_KEY."
            ),
        )
    models = snapshot.get("models")
    capabilities = snapshot.get("capabilities")
    return ModelSnapshotView(
        source="snapshot",
        active_model=runtime.service.model,
        provider_location=provider_location,
        discovered_at=(
            str(snapshot["discovered_at"])
            if snapshot.get("discovered_at") is not None
            else None
        ),
        models=tuple(item for item in models if isinstance(item, dict))
        if isinstance(models, list)
        else (),
        capabilities=tuple(
            item for item in capabilities if isinstance(item, dict)
        )
        if isinstance(capabilities, list)
        else (),
        authenticated_discovery=(
            bool(snapshot["authenticated_discovery"])
            if snapshot.get("authenticated_discovery") is not None
            else None
        ),
    )


def _is_local_provider_url(url: str) -> bool:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").casefold() in {"localhost", "127.0.0.1", "::1"}


app = create_app()
