"""Local-only pet API, settings and model adapters."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ecocompanion.application.models import CreateSessionRequest, TurnRequest
from ecocompanion.application.repository import SqlAlchemySessionRepository
from ecocompanion.application.service import CompanionService, SessionNotFound, TurnConflictError
from ecocompanion.games.engine import view as game_view
from ecocompanion.games.service import Games
from ecocompanion.infrastructure import CostLedger, JsonlTraceStore
from ecocompanion.messaging.bridge import PersonalWeChatBridge
from ecocompanion.messaging.diagnostics import check_messaging
from ecocompanion.messaging.runtime import MessagingRuntime
from ecocompanion.messaging.settings import (
    MessagingSettingsStore,
    MessagingSettingsUpdate,
    messaging_home,
)
from ecocompanion.pet.deepseek import create_client, failure_detail, model_id, require_text
from ecocompanion.pet.pipeline import (
    DeepSeekNotConfigured,
    PetPipeline,
    PetRenderer,
    TurnContext,
    turn_context,
)
from ecocompanion.pet.reports import (
    CheckIn,
    ReportRequest,
    ReportStore,
    assessment_snapshot,
    evidence_for_request,
    extractive_summary,
    new_document,
    report_print_html,
)
from ecocompanion.pet.settings import SettingsStore, SettingsUpdate, default_home
from ecocompanion.providers.errors import ProviderError


class SessionSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(default="", max_length=200)
    limit: int = Field(default=40, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class DraftUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(default="", max_length=8_000)


class TitleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)


class WeChatDiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: bool
    account_wxid: str = Field(default="", max_length=128)
    query: str = Field(default="", max_length=80)


def create_pet_app(home: Path | None = None, *, transport: httpx.AsyncBaseTransport | None = None):
    store = SettingsStore(home or default_home())
    messaging_store = MessagingSettingsStore(messaging_home())
    messaging_runtime = MessagingRuntime(settings_store=store)
    wechat_bridge = PersonalWeChatBridge()
    http = httpx.AsyncClient(trust_env=False, follow_redirects=False, transport=transport)
    pipeline = PetPipeline(http)
    repository = SqlAlchemySessionRepository(
        f"sqlite+aiosqlite:///{store.home / 'conversations.db'}"
    )
    service = CompanionService(
        repository=repository,
        renderer=PetRenderer(store, transport=transport),
        trace_store=JsonlTraceStore(store.home / "traces.jsonl"),
        cost_ledger=CostLedger(pricing={}),
        turn_advisor=pipeline.advise,
        turn_snapshot=assessment_snapshot,
    )
    channel_repository = SqlAlchemySessionRepository(
        f"sqlite+aiosqlite:///{messaging_home() / 'conversations.db'}"
    )
    token = secrets.token_urlsafe(32)
    reports = ReportStore(store.home / "wellbeing.db")
    games = Games(store.home / "games", store.home)
    report_previews: dict[str, dict] = {}
    config_lock = asyncio.Lock()
    active_lock = asyncio.Lock()

    async def remember_session(session_id: str) -> None:
        async with active_lock:
            temporary = store.home / "active-session.tmp"
            temporary.write_text(json.dumps({"session_id": session_id}), encoding="utf-8")
            os.replace(temporary, store.home / "active-session.json")

    async def forget_session(session_id: str) -> None:
        async with active_lock:
            path = store.home / "active-session.json"
            try:
                active = json.loads(path.read_text(encoding="utf-8"))["session_id"]
            except (OSError, ValueError, KeyError, TypeError):
                return
            if active == session_id:
                path.unlink(missing_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.home.mkdir(parents=True, exist_ok=True)
        messaging_home().mkdir(parents=True, exist_ok=True)
        await service.initialize()
        await channel_repository.initialize()
        await reports.initialize()
        try:
            yield
        finally:
            await service.close()
            await channel_repository.close()
            await reports.close()
            await messaging_runtime.stop()
            await wechat_bridge.set_running(False)
            await http.aclose()

    app = FastAPI(title="EcoCompanion Desktop Preview", lifespan=lifespan)
    app.state.pet_store = store
    app.state.pet_service = service
    app.state.messaging_store = messaging_store
    app.state.report_store = reports
    app.state.games = games

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.url.path.startswith("/api/pet"):
            # DNS rebinding / cross-origin pages must not access local keys or conversations.
            if request.url.hostname not in {"127.0.0.1", "localhost", "::1", "testserver"}:
                return JSONResponse({"detail": "local access only"}, status_code=403)
            origin = request.headers.get("origin")
            if origin:
                parsed = urlsplit(origin)
                scheme = request.url.scheme
                default_port = 80 if scheme == "http" else 443
                same_host = parsed.scheme == scheme and parsed.hostname == request.url.hostname
                same_port = (parsed.port or (80 if parsed.scheme == "http" else 443)) == (
                    request.url.port or default_port
                )
                if not same_host or not same_port:
                    return JSONResponse({"detail": "same-origin access required"}, status_code=403)
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                if request.headers.get("x-pet-token") != token:
                    return JSONResponse({"detail": "reload the application"}, status_code=403)
        result = await call_next(request)
        if request.url.path.startswith("/api/pet"):
            result.headers["Cache-Control"] = "no-store"
        result.headers["X-Content-Type-Options"] = "nosniff"
        result.headers["Referrer-Policy"] = "no-referrer"
        return result

    @app.get("/api/pet/bootstrap")
    async def bootstrap():
        active = None
        try:
            active = json.loads((store.home / "active-session.json").read_text())["session_id"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {"token": token, "current_session_id": active, **store.view()}

    @app.put("/api/pet/settings")
    async def save_settings(payload: SettingsUpdate):
        async with config_lock:
            try:
                return store.update(payload)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(500, "本地配置写入失败，原配置未更改") from exc

    @app.post("/api/pet/games")
    async def create_game(payload: dict):
        kind, color = payload.get("kind"), payload.get("color", "white")
        if (
            kind not in {"chess", "undercover"}
            or color not in {"white", "black"}
            or len(payload) > 2
        ):
            raise HTTPException(400, "不支持的游戏类型")
        return games.create(kind, color)

    @app.get("/api/pet/games/{game_id}")
    async def get_game(game_id: str):
        return game_view(games.load(game_id))

    @app.post("/api/pet/games/{game_id}/actions")
    async def game_action(game_id: str, payload: dict):
        if set(payload) - {"revision", "action", "text", "target"}:
            raise HTTPException(400, "无效的游戏动作")
        try:
            return await games.action(
                game_id,
                payload["revision"],
                payload["action"],
                payload.get("text", ""),
                payload.get("target"),
            )
        except (KeyError, TypeError) as exc:
            raise HTTPException(400, "无效的游戏动作") from exc

    @app.get("/api/pet/messaging/settings")
    async def messaging_settings():
        return messaging_store.view()

    @app.put("/api/pet/messaging/settings")
    async def save_messaging_settings(payload: MessagingSettingsUpdate):
        async with config_lock:
            try:
                view = messaging_store.update(payload)
                if messaging_runtime.running:
                    # The gateway validates each callback against its startup
                    # snapshot; the bridge also owns its listener registrations.
                    bridge_was_running = wechat_bridge.status()["running"]
                    if bridge_was_running:
                        await wechat_bridge.set_running(False)
                    await messaging_runtime.restart()
                    if bridge_was_running:
                        await wechat_bridge.set_running(True)
                return view
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(500, "渠道配置写入失败，原配置未更改") from exc

    @app.post("/api/pet/messaging/preflight")
    async def messaging_preflight():
        return await check_messaging(
            messaging_store.settings,
            store.settings,
            deepseek_ready=bool(store.keys.get("deepseek")),
        )

    @app.get("/api/pet/messaging/runtime")
    async def messaging_runtime_status():
        return messaging_runtime.status()

    @app.post("/api/pet/messaging/runtime")
    async def set_messaging_runtime(payload: dict[str, bool]):
        if set(payload) != {"running"} or not isinstance(payload["running"], bool):
            raise HTTPException(422, "仅支持 running 布尔值")
        async with config_lock:
            try:
                if payload["running"]:
                    return await messaging_runtime.restart()
                return await messaging_runtime.stop()
            except (RuntimeError, ValueError, OSError) as exc:
                raise HTTPException(400, str(exc)) from exc

    @app.get("/api/pet/messaging/wechat-bridge")
    async def wechat_bridge_status():
        return wechat_bridge.status()

    @app.post("/api/pet/messaging/wechat-bridge/check")
    async def check_wechat_bridge():
        return await wechat_bridge.check()

    @app.post("/api/pet/messaging/wechat-contacts")
    async def discover_wechat_contacts(payload: WeChatDiscoveryRequest):
        try:
            return await wechat_bridge.discover(
                payload.account_wxid, payload.query, confirmed=payload.confirmed
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/pet/messaging/wechat-bridge/install")
    async def install_wechat_bridge():
        try:
            return await wechat_bridge.install()
        except (RuntimeError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/pet/messaging/wechat-bridge/install")
    async def wechat_bridge_installation_status():
        return wechat_bridge.installation_status()

    @app.post("/api/pet/messaging/wechat-bridge")
    async def set_wechat_bridge(payload: dict[str, bool]):
        if set(payload) != {"running"} or not isinstance(payload["running"], bool):
            raise HTTPException(422, "仅支持 running 布尔值")
        try:
            return await wechat_bridge.set_running(payload["running"])
        except (RuntimeError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/pet/test/{provider}")
    async def test_connection(provider: Literal["deepseek", "qwen", "jev"]):
        cfg, keys = store.settings, dict(store.keys)
        if provider == "deepseek" and (not cfg.cloud_consent or not keys.get("deepseek")):
            raise HTTPException(400, "请先保存 API Key 并允许云端传输")
        try:
            if provider == "qwen":
                response = await http.get(f"{cfg.qwen_base_url}/api/tags", timeout=4)
                response.raise_for_status()
                models = [item["name"] for item in response.json()["models"]]
                ready = cfg.qwen_model in models
                return {
                    "ok": ready,
                    "detail": (
                        f"本地模型已就绪 · {cfg.qwen_model}"
                        if ready
                        else f"Ollama 已连接，但未找到 {cfg.qwen_model}；"
                        f"当前可用：{'、'.join(models) if models else '无，请先安装模型'}"
                    ),
                    "models": models,
                }
            if provider == "jev":
                if not keys.get("jev"):
                    raise HTTPException(400, "请先保存 TypeSafe API Key")
                response = await http.post(
                    "https://api.typesafe.ai/v1/systemone",
                    headers={"Authorization": f"Bearer {keys['jev']}"},
                    json={
                        "model": cfg.jev_model,
                        "state": {"kind": "configuration_validation"},
                        "questions": {
                            "route": {
                                "type": "choice",
                                "instructions": "Choose an action.",
                                "criteria": {
                                    "listen": "Listen.",
                                    "clarify": "Ask a question.",
                                    "safety_check": "Check safety.",
                                },
                            }
                        },
                    },
                    timeout=20,
                )
                response.raise_for_status()
                if response.json().get("model") != cfg.jev_model:
                    raise ValueError("unexpected Jev model version")
                return {"ok": True, "detail": f"Jev 决策服务已连接 · {cfg.jev_model}"}
            # A model-list request cannot verify generation, billing or model identity.
            # This fixed probe never includes saved conversations/personality.
            async with create_client(cfg, keys["deepseek"], transport=transport) as client:
                completion = await client.chat_completion(
                    model=model_id(cfg),
                    messages=[
                        {"role": "system", "content": "Reply briefly."},
                        {"role": "system", "content": "You are an AI desktop companion."},
                        {"role": "user", "content": "Say hello in Chinese."},
                    ],
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                )
                require_text(completion)
            return {
                "ok": True,
                "detail": f"实际生成验证成功 · {completion.response_model}",
                "model": completion.response_model,
            }
        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
            ProviderError,
            RuntimeError,
        ) as exc:
            if (
                provider == "jev"
                and isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code == 451
            ):
                detail = (
                    "TypeSafe 当前地区不可用（HTTP 451），无法使用 Jev；本地规则与 Qwen 不受影响"
                )
            elif provider == "jev":
                detail = f"Jev 连接或响应校验失败：{type(exc).__name__}"
            elif provider == "qwen":
                detail = "本地模型连接失败，请检查 Ollama 是否运行以及模型名称"
            else:
                detail = failure_detail(exc)
            return {
                "ok": False,
                "detail": detail,
            }

    @app.post("/api/pet/sessions", status_code=201)
    async def create_session():
        session = await service.create_session(
            CreateSessionRequest(memory_enabled=False, research_mode=False)
        )
        await remember_session(session.session_id)
        return session

    @app.post("/api/pet/sessions/search")
    async def search_sessions(payload: SessionSearch):
        # Keep potentially sensitive search terms out of URL/access logs.
        items, total = await repository.search_sessions(
            query=payload.query, limit=payload.limit, offset=payload.offset
        )
        return {
            "items": items,
            "total": total,
            "has_more": payload.offset + len(items) < total,
        }

    @app.post("/api/pet/sessions/{session_id}/activate")
    async def activate_session(session_id: str):
        try:
            session = await service.get_session(session_id)
            history = await service.list_messages(session_id)
            await remember_session(session_id)
            return {
                "session_id": session_id,
                "messages": history,
                "draft": await repository.get_draft(session_id),
                "attention": session.state.risk_level.value in {"L3", "L4"},
            }
        except SessionNotFound as exc:
            raise HTTPException(404, "会话不存在") from exc

    @app.get("/api/pet/reports/sources")
    async def report_sources():
        local, _ = await repository.search_sessions(limit=100)
        items = [
            {
                "session_id": item.session_id,
                "source": "local",
                "title": item.title,
                "message_count": item.message_count,
                "updated_at": item.updated_at,
            }
            for item in local
        ]
        channels = await channel_repository.list_channel_sessions()
        labels = {"qq": "QQ", "personal_wechat": "个人微信", "wechat": "微信"}
        for index, item in enumerate(channels, 1):
            messages = await channel_repository.list_messages(item["session_id"], limit=1)
            items.append(
                {
                    "session_id": item["session_id"],
                    "source": item["channel"],
                    "title": f"{labels.get(item['channel'], '消息渠道')}会话 {index}",
                    "message_count": await channel_repository.message_count(item["session_id"]),
                    "updated_at": messages[-1].created_at if messages else None,
                }
            )
        return {"items": items}

    @app.get("/api/pet/sessions/{session_id}/messages")
    async def messages(session_id: str):
        try:
            return await service.list_messages(session_id)
        except SessionNotFound as exc:
            raise HTTPException(404, "会话不存在") from exc

    @app.get("/api/pet/reports/sources/{source}/{session_id}/messages")
    async def report_source_messages(
        source: Literal["local", "personal_wechat", "qq", "wechat"], session_id: str
    ):
        source_repository = repository if source == "local" else channel_repository
        if source != "local":
            allowed = await channel_repository.list_channel_sessions()
            if not any(
                item["session_id"] == session_id and item["channel"] == source for item in allowed
            ):
                raise HTTPException(404, "渠道会话不存在")
        elif await repository.get(session_id) is None:
            raise HTTPException(404, "会话不存在")
        return await source_repository.list_messages(session_id)

    @app.get("/api/pet/sessions/{session_id}/state")
    async def session_state(session_id: str):
        try:
            session = await service.get_session(session_id)
            return {"attention": session.state.risk_level.value in {"L3", "L4"}}
        except SessionNotFound as exc:
            raise HTTPException(404, "会话不存在") from exc

    @app.post("/api/pet/sessions/{session_id}/turn")
    async def turn(session_id: str, payload: TurnRequest):
        ctx = TurnContext(store.settings, dict(store.keys))
        marker = turn_context.set(ctx)
        try:
            result = await service.process_turn(session_id, payload)
            ctx.note(
                "audit",
                "ok" if result.decision.safety_passed else "warning",
                "已完成输出审查；明确标注的本地安全回退"
                if result.decision.used_fallback
                else "DeepSeek 回复审查通过",
            )
            return {
                "result": result,
                "stages": ctx.stages,
                "mode": "safety_fallback" if result.decision.used_fallback else "deepseek",
                "attention": result.decision.escalation_level.value in {"L3", "L4"},
            }
        except SessionNotFound as exc:
            raise HTTPException(404, "会话不存在，请新建对话") from exc
        except TurnConflictError as exc:
            recovered = await service.recover_turn(session_id, payload)
            if recovered is not None:
                session, response = recovered
                return {
                    "result": {
                        "session": session,
                        "response": response,
                        "decision": {"used_fallback": True, "recovered": True},
                    },
                    "stages": [
                        {
                            "name": "recovered",
                            "state": "ok",
                            "detail": "已从本地记录恢复，未重复生成或调用模型",
                        }
                    ],
                    "mode": "recovered",
                    "attention": session.state.risk_level.value in {"L3", "L4"},
                    "recovered": True,
                }
            raise HTTPException(409, "该消息已处理，请重新打开对话查看") from exc
        except DeepSeekNotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        except (RuntimeError, ProviderError) as exc:
            if not ctx.decision.get("safety_fallback") or "deepseek_fallback" not in exc.args:
                raise HTTPException(503, failure_detail(exc)) from exc
            raise HTTPException(503, "安全回复生成失败，请稍后重试") from exc
        finally:
            turn_context.reset(marker)

    @app.put("/api/pet/sessions/{session_id}/draft")
    async def save_draft(session_id: str, payload: DraftUpdate):
        try:
            await repository.save_draft(session_id, payload.text)
            return {"saved": True}
        except KeyError as exc:
            raise HTTPException(404, "会话不存在") from exc

    @app.put("/api/pet/sessions/{session_id}/title")
    async def rename_session(session_id: str, payload: TitleUpdate):
        try:
            await repository.rename_session(session_id, payload.title)
            return {"title": " ".join(payload.title.split())}
        except KeyError as exc:
            raise HTTPException(404, "会话不存在") from exc
        except ValueError as exc:
            raise HTTPException(422, "标题需要 1-48 个非空白字符") from exc

    @app.delete("/api/pet/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        if not await service.delete_session(session_id):
            raise HTTPException(404, "会话不存在")
        await reports.remove_session_evidence(session_id)
        await forget_session(session_id)

    @app.get("/api/pet/reports")
    async def list_reports():
        return {"items": await reports.list()}

    @app.post("/api/pet/reports/preview")
    async def preview_report(payload: ReportRequest):
        if payload.preview_id is not None:
            raise HTTPException(400, "预览请求不能指定已存在的预览编号")
        if not payload.check_in and not payload.message_ids:
            raise HTTPException(400, "请填写今日记录，或至少选择一条本人消息")
        try:
            source_repository = await report_repository(payload)
            evidence = await evidence_for_request(source_repository, payload)
            if payload.summary.strip():
                summary = payload.summary.strip()
                source = "user"
            else:
                summary = extractive_summary(evidence)
                source = "extractive" if summary else "none"
            assessments = (
                await source_repository.list_assessments(payload.session_id, payload.message_ids)
                if payload.session_id
                else []
            )
            saved_assessments = {item["message_id"]: item for item in assessments}
            assessments = [
                saved_assessments.get(
                    message_id,
                    {
                        "message_id": message_id,
                        "qwen": {"status": "unavailable"},
                        "decision": {},
                    },
                )
                for message_id in payload.message_ids
            ]
            document = new_document(payload, evidence, summary, source, assessments)
            preview_id = secrets.token_urlsafe(18)
            report_previews[preview_id] = document
            if len(report_previews) > 30:
                report_previews.pop(next(iter(report_previews)))
            return {"preview_id": preview_id, **document}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (KeyError, TypeError) as exc:
            raise HTTPException(400, "报告数据格式无效，未生成报告") from exc

    @app.post("/api/pet/reports", status_code=201)
    async def save_report(payload: ReportRequest):
        document = report_previews.pop(payload.preview_id, None) if payload.preview_id else None
        if document is None:
            raise HTTPException(409, "请先预览当前报告，再保存快照")
        if payload.check_in != (
            CheckIn.model_validate(document["check_in"]) if document.get("check_in") else None
        ):
            raise HTTPException(409, "自评已变化，请重新预览")
        source_repository = await report_repository(payload)
        evidence = await evidence_for_request(source_repository, payload)
        if [item.message_id for item in evidence] != [
            item["message_id"] for item in document["evidence"]
        ]:
            raise HTTPException(409, "引用消息已变化，请重新预览")
        if payload.summary.strip() != document.get("summary"):
            document["summary"] = payload.summary.strip()
            document["summary_source"] = (
                f"{document['summary_source']}+user-edit"
                if document["summary_source"] != "none"
                else "user"
            )
        if payload.session_id:
            if await source_repository.get(payload.session_id) is None:
                raise HTTPException(404, "会话不存在，报告未保存")
        return await reports.save(payload.check_in, document, evidence)

    async def report_repository(payload: ReportRequest) -> SqlAlchemySessionRepository:
        if payload.source == "local":
            return repository
        allowed = await channel_repository.list_channel_sessions()
        if not any(
            item["session_id"] == payload.session_id and item["channel"] == payload.source
            for item in allowed
        ):
            raise HTTPException(404, "渠道会话不存在")
        return channel_repository

    @app.get("/api/pet/reports/{report_id}")
    async def get_report(report_id: str):
        document = await reports.get(report_id)
        if document is None:
            raise HTTPException(404, "报告不存在或已删除")
        return document

    @app.get("/api/pet/reports/{report_id}/print")
    async def print_report(report_id: str):
        document = await reports.get(report_id)
        if document is None:
            raise HTTPException(404, "报告不存在或已删除")
        return HTMLResponse(report_print_html(document), media_type="text/html; charset=utf-8")

    @app.delete("/api/pet/reports/{report_id}", status_code=204)
    async def delete_report(report_id: str):
        if not await reports.delete(report_id):
            raise HTTPException(404, "报告不存在或已删除")

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "app": "desktop-pet-preview"}

    web = Path(__file__).resolve().parents[4] / "web" / "dist"
    if (web / "index.html").exists():

        @app.get("/games", include_in_schema=False)
        async def games_page():
            return FileResponse(web / "index.html")

        app.mount("/", StaticFiles(directory=web, html=True))
    return app


app = create_pet_app()
