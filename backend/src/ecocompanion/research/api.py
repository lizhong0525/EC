"""Standalone local annotation API; it never serves curator files or a future-node route."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from ecocompanion.research.annotation import AnnotationStore
from ecocompanion.research.contracts import Judgment


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    judgment: Judgment


def annotation_app(store: AnnotationStore) -> FastAPI:
    app = FastAPI(title="EcoTriage v2 annotation", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def private_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    def token_from_header(authorization: str) -> str:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Annotation access required")
        return authorization[7:]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return Path(__file__).with_name("annotation.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/task")
    def task(authorization: Annotated[str, Header()] = "") -> dict[str, Any]:
        try:
            return store.next_task(token_from_header(authorization))
        except PermissionError as exc:
            raise HTTPException(401, "Invalid annotation access") from exc

    @app.post("/judgments")
    def submit(body: Submission, authorization: Annotated[str, Header()] = "") -> dict[str, Any]:
        try:
            return store.submit(token_from_header(authorization), body.task_id, body.judgment)
        except PermissionError as exc:
            raise HTTPException(401, "Invalid annotation access") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/logout", status_code=204)
    def logout() -> Response:
        return Response(status_code=204)

    return app
