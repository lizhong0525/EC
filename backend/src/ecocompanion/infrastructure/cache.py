"""Exact, immutable SQLite response cache for reproducible model calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SECRET_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "proxy-authorization",
    "x-api-key",
}


class CacheConflictError(RuntimeError):
    """Raised when one exact request key maps to different validated results."""


@dataclass(frozen=True)
class CacheEntry:
    key: str
    namespace: str
    payload: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime


def canonical_json(value: Any) -> str:
    """Return stable JSON used for cache-key construction."""

    return json.dumps(
        _to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_exact_cache_key(*, namespace: str, request: Any) -> str:
    if not namespace.strip():
        raise ValueError("cache namespace cannot be empty")
    _reject_secret_fields(request)
    canonical = canonical_json({"namespace": namespace, "request": request})
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExactCache:
    """Process-safe exact cache.

    Entries are insert-only. Callers must include seed and replicate id in the
    request object when independent stochastic samples are required.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get(self, key: str) -> CacheEntry | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT cache_key, namespace, payload_json, metadata_json, created_at
                FROM exact_cache
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(
            key=str(row[0]),
            namespace=str(row[1]),
            payload=_load_json_object(str(row[2]), field_name="payload"),
            metadata=_load_json_object(str(row[3]), field_name="metadata"),
            created_at=datetime.fromisoformat(str(row[4])),
        )

    def put(
        self,
        *,
        key: str,
        namespace: str,
        payload: Any,
        metadata: Any | None = None,
    ) -> None:
        if not key.strip():
            raise ValueError("cache key cannot be empty")
        if not namespace.strip():
            raise ValueError("cache namespace cannot be empty")
        payload_json = canonical_json(payload)
        metadata_json = canonical_json(metadata or {})
        now = datetime.now(UTC).isoformat()

        with self._lock, self._connect() as connection:
            existing = connection.execute(
                """
                SELECT namespace, payload_json, metadata_json
                FROM exact_cache
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != (namespace, payload_json, metadata_json):
                    raise CacheConflictError(
                        "exact cache key already exists with different content"
                    )
                return
            connection.execute(
                """
                INSERT INTO exact_cache(
                    cache_key, namespace, payload_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (key, namespace, payload_json, metadata_json, now),
            )
            connection.commit()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS exact_cache(
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _reject_secret_fields(value: Any) -> None:
    jsonable = _to_jsonable(value)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower() in _SECRET_FIELD_NAMES:
                    raise ValueError(f"cache request cannot contain secret field: {key}")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(jsonable)


def _load_json_object(value: str, *, field_name: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"cached {field_name} must be a JSON object")
    return parsed
