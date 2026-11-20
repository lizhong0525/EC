"""Append-only JSONL traces and immutable run manifests."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecocompanion.domain.schemas import TraceEventV1
from ecocompanion.providers.types import ModelRole

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "api-token",
    "access_token",
    "refresh_token",
    "client_secret",
    "litellm_api_key",
    "authorization",
    "cookie",
    "proxy-authorization",
    "secret",
    "set-cookie",
    "token",
    "x-api-key",
}
_SENSITIVE_VALUE_PATTERNS = (
    # OpenAI-compatible keys, including the LiteLLM key shape used by V1.
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    # Avoid persisting bearer credentials accidentally copied into an event.
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b"),
)


class TraceStoreError(RuntimeError):
    """Trace persistence or validation failure."""


class ManifestConflictError(RuntimeError):
    """A run manifest is immutable once written."""


class ManifestModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    family: str | None = None
    capability_status: str
    response_model_ids: tuple[str, ...] = ()


class ManifestRoleBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ModelRole
    model_id: str
    family: str | None = None


class RunManifest(BaseModel):
    """Inputs needed to audit and reproduce one experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    benchmark_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    gateway_base_url: str
    models: tuple[ManifestModel, ...]
    role_bindings: tuple[ManifestRoleBinding, ...]
    prompt_versions: dict[str, str]
    schema_versions: dict[str, str]
    random_seed: int
    research_mode: bool = True
    cross_model_fallback_allowed: bool = False

    @model_validator(mode="after")
    def enforce_research_contract(self) -> RunManifest:
        if self.research_mode and self.cross_model_fallback_allowed:
            raise ValueError("research mode forbids cross-model fallback")
        parsed = urlsplit(self.gateway_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("gateway_base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("gateway_base_url cannot contain credentials, query, or fragment")
        return self

    def sanitized_gateway_url(self) -> str:
        parsed = urlsplit(self.gateway_base_url)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


class JsonlTraceStore:
    """Append and validate complete TraceEventV1 records."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEventV1) -> None:
        payload = event.model_dump(mode="json")
        sanitized = _redact_sensitive(payload)
        serialized = json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            _append_line(self._path, serialized)

    def read_all(self) -> tuple[TraceEventV1, ...]:
        if not self._path.exists():
            return ()
        events: list[TraceEventV1] = []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                events.append(TraceEventV1.model_validate_json(line))
            except ValueError as exc:
                raise TraceStoreError(
                    f"invalid trace record at line {line_number}"
                ) from exc
        return tuple(events)


class RunManifestStore:
    """Write-once JSON manifest store."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, manifest: RunManifest) -> None:
        payload = manifest.model_dump(mode="json")
        payload["gateway_base_url"] = manifest.sanitized_gateway_url()
        serialized = json.dumps(
            _redact_sensitive(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        with self._lock:
            if self._path.exists():
                existing = self._path.read_text(encoding="utf-8")
                if json.loads(existing) != json.loads(serialized):
                    raise ManifestConflictError(
                        "run manifest already exists with different content"
                    )
                return
            descriptor = os.open(
                self._path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, f"{serialized}\n".encode())
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def read(self) -> RunManifest:
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        try:
            return RunManifest.model_validate_json(self._path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise TraceStoreError("invalid run manifest") from exc


def _append_line(path: Path, serialized: str) -> None:
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{serialized}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in _SENSITIVE_KEYS
                else _redact_sensitive(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(child) for child in value]
    if isinstance(value, tuple):
        return [_redact_sensitive(child) for child in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SENSITIVE_VALUE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value
