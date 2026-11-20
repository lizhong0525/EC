"""Explicitly managed messaging runtime with an inert import-time surface."""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI

from ecocompanion.application.repository import SqlAlchemySessionRepository
from ecocompanion.application.service import CompanionService
from ecocompanion.infrastructure import CostLedger, JsonlTraceStore
from ecocompanion.messaging.personal_wechat import create_personal_wechat_app
from ecocompanion.messaging.processor import MessageProcessor, MessageWorker
from ecocompanion.messaging.qq import QQApi, QQRunner
from ecocompanion.messaging.settings import MessagingSettings, load_settings, messaging_home
from ecocompanion.pet.pipeline import PetPipeline, PetRenderer
from ecocompanion.pet.reports import assessment_snapshot
from ecocompanion.pet.settings import SettingsStore, default_home


async def wait_callback(
    host: str, port: int, task: asyncio.Task[None], timeout: float = 10
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if task.done():
            await task
            raise RuntimeError("消息回调服务未能持续运行")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError("消息回调端口未能监听，请检查端口占用和本机防火墙")


def enabled_channels(settings: MessagingSettings) -> list[str]:
    return [
        name
        for name, enabled in (
            ("qq", settings.qq.enabled),
            ("personal_wechat", settings.personal_wechat.enabled),
        )
        if enabled
    ]


class MessagingRuntime:
    """Own all messaging resources so UI start/stop cannot leave workers behind."""

    def __init__(
        self, home: Path | None = None, *, settings_store: SettingsStore | None = None
    ) -> None:
        self.home = home or messaging_home()
        self.settings_store = settings_store
        self.stack: AsyncExitStack | None = None
        self.channels: list[str] = []
        self.error = ""

    @property
    def running(self) -> bool:
        return self.stack is not None

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "channels": self.channels,
            "detail": self.error
            or ("消息网关正在运行" if self.running else "消息网关未启动"),
        }

    async def start(self) -> dict[str, Any]:
        if self.running:
            return self.status()
        stack = AsyncExitStack()
        try:
            settings = load_settings(self.home)
            channels = enabled_channels(settings)
            if not channels:
                raise RuntimeError("请先启用并保存至少一个消息渠道")
            if settings.callback_host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError("Bind callbacks to loopback and use a managed HTTPS proxy")
            self.home.mkdir(parents=True, exist_ok=True)
            http = await stack.enter_async_context(
                httpx.AsyncClient(trust_env=False, follow_redirects=False)
            )
            store = self.settings_store or SettingsStore(default_home())
            pipeline = PetPipeline(http)
            repository = SqlAlchemySessionRepository(
                f"sqlite+aiosqlite:///{self.home / 'conversations.db'}"
            )
            service = CompanionService(
                repository=repository,
                renderer=PetRenderer(store),
                trace_store=JsonlTraceStore(self.home / "traces.jsonl"),
                cost_ledger=CostLedger(path=self.home / "costs.jsonl"),
                turn_advisor=pipeline.advise,
                turn_snapshot=assessment_snapshot,
            )
            await service.initialize()
            stack.push_async_callback(service.close)
            processor = MessageProcessor(
                service,
                store,
                repository=repository,
                per_user_per_minute=settings.per_user_per_minute,
            )
            worker = MessageWorker(processor, queue_size=settings.max_inflight * 4)
            worker.start()
            stack.push_async_callback(worker.close)
            if settings.qq.enabled:
                api = QQApi(settings.qq)
                stack.push_async_callback(api.close)
                stop = asyncio.Event()
                task = asyncio.create_task(
                    QQRunner(api, processor, worker).run(stop), name="qq-official-bot"
                )

                async def stop_qq() -> None:
                    stop.set()
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                stack.push_async_callback(stop_qq)
            if settings.personal_wechat.enabled:
                callback = FastAPI(docs_url=None, redoc_url=None)
                callback.include_router(
                    create_personal_wechat_app(processor, settings.personal_wechat).router
                )

                @callback.get("/healthz")
                async def health() -> dict[str, bool]:
                    return {"ok": True}

                server = uvicorn.Server(
                    uvicorn.Config(
                        callback,
                        host=settings.callback_host,
                        port=settings.callback_port,
                        log_config=None,
                        access_log=False,
                    )
                )
                task = asyncio.create_task(server.serve(), name="wechat-callback")

                async def stop_callback() -> None:
                    server.should_exit = True
                    await task

                stack.push_async_callback(stop_callback)
                await wait_callback(settings.callback_host, settings.callback_port, task)
            self.stack = stack
            self.channels = channels
            self.error = ""
            return self.status()
        except Exception:
            await stack.aclose()
            raise

    async def stop(self) -> dict[str, Any]:
        stack, self.stack = self.stack, None
        self.channels = []
        if stack is not None:
            await stack.aclose()
        return self.status()

    async def restart(self) -> dict[str, Any]:
        await self.stop()
        return await self.start()


async def run(home: Path | None = None) -> None:
    runtime = MessagingRuntime(home)
    await runtime.start()
    try:
        await asyncio.Event().wait()
    finally:
        await runtime.stop()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.home))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
