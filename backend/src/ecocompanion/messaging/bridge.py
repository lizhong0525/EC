"""Lifecycle for the explicitly started, isolated personal WeChat transport."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scripts.wechat_test_bridge import preflight

BRIDGE_START_TIMEOUT = 90


class WeChatIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wxid: str = Field(min_length=1, max_length=128)
    name: str = Field(default="", max_length=80)


class WeChatDiscovery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: WeChatIdentity
    contacts: list[WeChatIdentity]


def decode_discovery_output(output: bytes) -> dict[str, Any]:
    """Ignore dependency diagnostics; accept only the small JSON result contract."""
    text = output.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if not line.startswith("{"):
            continue
        try:
            return WeChatDiscovery.model_validate(loads(line)).model_dump()
        except (JSONDecodeError, ValueError):
            continue
    raise RuntimeError("微信账号与好友读取结果无效")


class PersonalWeChatBridge:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.process_status: dict[str, Any] | None = None
        self.metrics: dict[str, int] = {}
        self.monitor_task: asyncio.Task[None] | None = None
        self.process_job: int | None = None
        self.lock = asyncio.Lock()
        self.install_lock = asyncio.Lock()
        self.install_progress = "尚未安装"
        self.install_running = False

    @staticmethod
    def _root() -> Path:
        return Path(__file__).resolve().parents[4]

    def status(self) -> dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        state = self.process_status or {}
        if not running and state.get("phase") == "error":
            detail = str(state.get("detail") or "微信测试桥接已停止")
        elif running and state.get("phase") == "listening":
            detail = str(state.get("detail") or "微信测试桥接监听已就绪")
        elif running:
            detail = str(state.get("detail") or "微信测试桥接正在初始化")
        else:
            detail = "独立微信测试桥接未启动"
        return {
            "running": running,
            "ready": running and state.get("phase") == "listening",
            "detail": detail,
            "metrics": dict(self.metrics),
        }

    def _read_process_status(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, b""):
                try:
                    report = json.loads(line.decode("utf-8", errors="replace"))
                except (JSONDecodeError, UnicodeError):
                    continue
                if (
                    report.get("wechat_bridge") == 1
                    and report.get("phase") in {"starting", "listening", "error"}
                    and isinstance(report.get("detail"), str)
                ):
                    self.process_status = {
                        "phase": report["phase"], "detail": report["detail"][:240]
                    }
                elif report.get("wechat_bridge_metrics") == 1:
                    self.metrics = {
                        key: value
                        for key, value in report.items()
                        if key != "wechat_bridge_metrics" and isinstance(value, int)
                    }
        except (OSError, ValueError):
            return

    async def _monitor(self, process: subprocess.Popen[bytes]) -> None:
        await asyncio.to_thread(self._read_process_status, process)
        await asyncio.to_thread(process.wait)
        if self.process is process and (self.process_status or {}).get("phase") != "error":
            self.process_status = {
                "phase": "error", "detail": "微信测试桥接已意外停止，请重新启动并检查窗口"
            }

    @staticmethod
    def _process_job() -> int | None:
        """Make console-less Python children exit with the bridge on Windows."""
        if os.name != "nt":
            return None
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        # The supported Windows Python ABI accepts this 64-bit job-info layout.
        # Only BasicLimitInformation.LimitFlags is populated; all limits stay zero.
        info = bytearray(144)
        info[16:20] = (0x2000).to_bytes(4, "little")  # KILL_ON_JOB_CLOSE
        buffer = (ctypes.c_char * len(info)).from_buffer(info)
        if not kernel32.SetInformationJobObject(job, 9, buffer, len(info)):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise OSError(error, "无法创建桥接进程退出保护")
        return int(job)

    @staticmethod
    def _assign_process_job(job: int | None, process: subprocess.Popen[bytes]) -> None:
        if job is None or os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x001F0FFF, False, process.pid)
        if handle:
            try:
                kernel32.AssignProcessToJobObject(job, handle)
            finally:
                kernel32.CloseHandle(handle)

    def _close_process_job(self) -> None:
        job, self.process_job = self.process_job, None
        if job is not None and os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.CloseHandle(job)

    async def _stop_locked(self) -> dict[str, Any]:
        process, self.process = self.process, None
        monitor, self.monitor_task = self.monitor_task, None
        if process is not None and process.poll() is None:
            await asyncio.to_thread(process.terminate)
            try:
                await asyncio.to_thread(process.wait, timeout=8)
            except subprocess.TimeoutExpired:
                await asyncio.to_thread(process.kill)
                await asyncio.to_thread(process.wait, timeout=4)
        if monitor is not None:
            try:
                await asyncio.wait_for(monitor, timeout=2)
            except TimeoutError:
                monitor.cancel()
        self.process_status = None
        self.metrics = {}
        self._close_process_job()
        return self.status()

    async def check(self) -> dict[str, Any]:
        return await asyncio.to_thread(preflight)

    def installation_status(self) -> dict[str, Any]:
        return {"running": self.install_running, "detail": self.install_progress}

    async def _installation_progress(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        phases = (
            ("Collecting ", "正在下载微信桥接组件及其依赖"),
            ("Using cached ", "正在使用已下载的组件"),
            ("Installing build dependencies", "正在准备本地构建环境"),
            ("Building wheel", "正在本地构建组件，首次安装可能需要几分钟"),
            ("Installing collected packages", "正在写入桥接组件"),
            ("component import OK", "组件安装完成，正在确认"),
        )
        while True:
            line = await process.stdout.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            for marker, detail in phases:
                if marker in text:
                    self.install_progress = detail

    async def discover(
        self, account_wxid: str, query: str, *, confirmed: bool
    ) -> dict[str, Any]:
        if not confirmed:
            raise ValueError("请先确认使用隔离测试账号与测试聊天")
        async with self.lock:
            if self.status()["running"]:
                raise RuntimeError("请先停止微信桥接，再读取账号与好友")
            root = self._root()
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "scripts.wechat_test_bridge",
                "--discover",
                "--confirm-test-account",
                "--account",
                account_wxid,
                "--query",
                query,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={
                    **os.environ,
                    "PYTHONPATH": str(root / "backend" / "src"),
                    "PYTHONUTF8": "1",
                },
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60)
            except TimeoutError as exc:
                process.kill()
                await process.communicate()
                raise RuntimeError("读取微信账号与好友超时，请确认微信已登录后重试") from exc
            if process.returncode != 0:
                detail = preflight()["detail"]
                if "尚未安装" in str(detail):
                    raise RuntimeError("尚未安装微信桥接组件，请先点击下方“安装桥接组件”")
                raise RuntimeError(
                    "无法读取微信账号与好友。请登录 Windows 微信后重试；"
                    "如果登录了多个账号，请只保留专用测试账号，并核对已选账号。"
                )
            return decode_discovery_output(stdout)

    async def install(self) -> dict[str, Any]:
        if os.name != "nt":
            raise RuntimeError("微信桥接组件仅支持 Windows 测试环境")
        async with self.install_lock:
            if self.status()["running"]:
                raise RuntimeError("请先停止微信桥接，再安装组件")
            if importlib.util.find_spec("wechatauto") is not None:
                self.install_progress = "桥接组件已安装，可以读取账号与好友"
                return {"ok": True, "detail": self.install_progress}
            root = self._root()
            script = root / "Install-WeChatTestBridge.ps1"
            if not script.is_file():
                raise RuntimeError("缺少微信桥接组件安装脚本")
            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.install_running = True
            self.install_progress = "正在启动桥接组件安装"
            progress = asyncio.create_task(self._installation_progress(process))
            try:
                await process.wait()
                await progress
            except BaseException:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                await progress
                raise
            finally:
                self.install_running = False
            if process.returncode != 0:
                self.install_progress = (
                    f"安装未完成：{self.install_progress}。请检查网络或稍后重试"
                )
                raise RuntimeError(self.install_progress)
            verify = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import wechatauto; from wechatauto import WeChatDB, WeChatGUI",
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if await verify.wait() != 0:
                self.install_progress = "桥接组件无法导入，请检查 Python 环境"
                raise RuntimeError(self.install_progress)
            self.install_progress = "桥接组件已安装，可以读取账号与好友"
            return {"ok": True, "detail": self.install_progress}

    async def set_running(self, running: bool) -> dict[str, Any]:
        async with self.lock:
            if not running:
                return await self._stop_locked()
            if self.status()["running"]:
                return self.status()
            ready = await self.check()
            if not ready["ok"]:
                raise RuntimeError(str(ready["detail"]))
            root = self._root()
            job = self._process_job()
            process = subprocess.Popen(
                [sys.executable, "-m", "scripts.wechat_test_bridge", "--run"],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={
                    **os.environ,
                    "PYTHONPATH": str(root / "backend" / "src"),
                    "PYTHONUTF8": "1",
                },
            )
            self.process = process
            self.process_status = {"phase": "starting", "detail": "正在启动微信测试桥接"}
            self.process_job = job
            self._assign_process_job(job, process)
            self.monitor_task = asyncio.create_task(self._monitor(process))
            deadline = time.monotonic() + BRIDGE_START_TIMEOUT
            while time.monotonic() < deadline:
                await asyncio.sleep(0.2)
                if self.status()["ready"]:
                    return self.status()
                if process.poll() is not None or self.process_status.get("phase") == "error":
                    detail = str(self.process_status.get("detail") or "微信测试桥接未能启动")
                    await self._stop_locked()
                    self.process_status = {"phase": "error", "detail": detail}
                    raise RuntimeError(detail)
            await self._stop_locked()
            raise RuntimeError("微信测试桥接初始化超时，请核对测试账号、客户端版本与窗口后重试")
