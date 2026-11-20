"""Durable journal for provider calls whose sinks are written separately.

The exact response cache and the cost ledger intentionally remain independent
artifacts.  A process can therefore stop after one sink is durable but before
the other is written.  ``DurableCallJournal`` closes that recovery gap by
persisting an accepted provider completion first.  A subsequent invocation can
materialize the ledger/cache from the journal without sampling the provider
again.

Only provider completion metadata and the validated response are stored here;
request messages and credentials are never written to the journal.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ecocompanion.infrastructure.cache import canonical_json

JournalState = Literal["pending", "completed", "failed", "committed"]


class JournalConflictError(RuntimeError):
    """Raised when one call id is reused with a different immutable identity."""


class JournalPendingError(RuntimeError):
    """Raised when a previous provider call has an unresolved outcome."""


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """One durable provider-call state."""

    call_id: str
    cache_key: str
    namespace: str
    role: str
    model: str
    state: JournalState
    payload: dict[str, Any] | None
    metadata: dict[str, Any]
    error_type: str | None
    error_message: str | None
    updated_at: datetime


class DurableCallJournal:
    """SQLite-WAL call journal with idempotent state transitions."""

    _TABLE = "interactive_call_journal"

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.RLock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def get(self, call_id: str) -> JournalEntry | None:
        if not call_id.strip():
            raise ValueError("call_id cannot be empty")
        if self._path is None:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT call_id, cache_key, namespace, role, model, state,
                       payload_json, metadata_json, error_type, error_message,
                       updated_at
                FROM {self._TABLE}
                WHERE call_id = ?
                """,
                (call_id,),
            ).fetchone()
        return _row_to_entry(row) if row is not None else None

    def begin(
        self,
        *,
        call_id: str,
        cache_key: str,
        namespace: str,
        role: str,
        model: str,
    ) -> JournalEntry | None:
        """Reserve a call, returning a completed entry when one already exists.

        A PENDING row means the previous process may have died while the
        provider outcome was unknown.  It is deliberately not retried
        automatically: without a provider idempotency contract, doing so could
        double-bill the experiment.
        """

        self._validate_identity(
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            role=role,
            model=model,
        )
        if self._path is None:
            return None
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT call_id, cache_key, namespace, role, model, state,
                       payload_json, metadata_json, error_type, error_message,
                       updated_at
                FROM {self._TABLE}
                WHERE call_id = ?
                """,
                (call_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    f"""
                    INSERT INTO {self._TABLE}(
                        call_id, cache_key, namespace, role, model, state,
                        payload_json, metadata_json, error_type, error_message,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, '{{}}', NULL, NULL, ?)
                    """,
                    (call_id, cache_key, namespace, role, model, now),
                )
                connection.commit()
                return None
            existing = _row_to_entry(row)
            self._check_identity(existing, cache_key, namespace, role, model)
            if existing.state in {"completed", "committed"}:
                return existing
            if existing.state == "pending":
                raise JournalPendingError(
                    f"provider call has unresolved outcome: {call_id}"
                )
            # A recorded provider exception is retryable.  Keep the same
            # deterministic call id but clear the old failure payload.
            connection.execute(
                f"""
                UPDATE {self._TABLE}
                SET state = 'pending', payload_json = NULL, metadata_json = '{{}}',
                    error_type = NULL, error_message = NULL, updated_at = ?
                WHERE call_id = ?
                """,
                (now, call_id),
            )
            connection.commit()
        return None

    def complete(
        self,
        *,
        call_id: str,
        cache_key: str,
        namespace: str,
        role: str,
        model: str,
        payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Persist an accepted completion before writing any secondary sink."""

        self._validate_identity(
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            role=role,
            model=model,
        )
        if self._path is None:
            return
        payload_json = canonical_json(payload)
        metadata_json = canonical_json(metadata)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT call_id, cache_key, namespace, role, model, state,
                       payload_json, metadata_json, error_type, error_message,
                       updated_at
                FROM {self._TABLE}
                WHERE call_id = ?
                """,
                (call_id,),
            ).fetchone()
            if row is None:
                raise JournalConflictError(
                    f"cannot complete an unreserved provider call: {call_id}"
                )
            existing = _row_to_entry(row)
            self._check_identity(existing, cache_key, namespace, role, model)
            if existing.state in {"completed", "committed"}:
                if (
                    existing.payload is None
                    or canonical_json(existing.payload) != payload_json
                    or canonical_json(existing.metadata) != metadata_json
                ):
                    raise JournalConflictError(
                        f"provider call already completed with different payload: {call_id}"
                    )
                return
            if existing.state != "pending":
                raise JournalConflictError(
                    f"provider call is not pending: {call_id}"
                )
            connection.execute(
                f"""
                UPDATE {self._TABLE}
                SET state = 'completed', payload_json = ?, metadata_json = ?,
                    error_type = NULL, error_message = NULL, updated_at = ?
                WHERE call_id = ?
                """,
                (payload_json, metadata_json, now, call_id),
            )
            connection.commit()

    def fail(
        self,
        *,
        call_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """Record a provider failure so a later retry is explicitly known."""

        if self._path is None:
            return
        if not call_id.strip():
            raise ValueError("call_id cannot be empty")
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT state FROM {self._TABLE} WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            if row is None:
                raise JournalConflictError(
                    f"cannot fail an unreserved provider call: {call_id}"
                )
            state = str(row[0])
            if state in {"completed", "committed"}:
                return
            connection.execute(
                f"""
                UPDATE {self._TABLE}
                SET state = 'failed', payload_json = NULL, metadata_json = '{{}}',
                    error_type = ?, error_message = ?, updated_at = ?
                WHERE call_id = ?
                """,
                (error_type, error_message[:1000], now, call_id),
            )
            connection.commit()

    def mark_committed(self, call_id: str) -> None:
        """Mark both secondary sinks as materialized."""

        if self._path is None:
            return
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT state FROM {self._TABLE} WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            if row is None:
                raise JournalConflictError(
                    f"cannot commit an unknown provider call: {call_id}"
                )
            state = str(row[0])
            if state == "committed":
                return
            if state != "completed":
                raise JournalConflictError(
                    f"cannot commit provider call in state {state}: {call_id}"
                )
            connection.execute(
                f"""
                UPDATE {self._TABLE}
                SET state = 'committed', updated_at = ?
                WHERE call_id = ?
                """,
                (datetime.now(UTC).isoformat(), call_id),
            )
            connection.commit()

    @staticmethod
    def _validate_identity(
        *,
        call_id: str,
        cache_key: str,
        namespace: str,
        role: str,
        model: str,
    ) -> None:
        if not call_id.strip():
            raise ValueError("call_id cannot be empty")
        if not cache_key.strip():
            raise ValueError("cache_key cannot be empty")
        if not namespace.strip():
            raise ValueError("namespace cannot be empty")
        if not role.strip():
            raise ValueError("role cannot be empty")
        if not model.strip():
            raise ValueError("model cannot be empty")

    @staticmethod
    def _check_identity(
        existing: JournalEntry,
        cache_key: str,
        namespace: str,
        role: str,
        model: str,
    ) -> None:
        if (
            existing.cache_key != cache_key
            or existing.namespace != namespace
            or existing.role != role
            or existing.model != model
        ):
            raise JournalConflictError(
                f"provider call identity changed for call id: {existing.call_id}"
            )

    def _initialize(self) -> None:
        if self._path is None:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE}(
                    call_id TEXT PRIMARY KEY,
                    cache_key TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    role TEXT NOT NULL,
                    model TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(
                        state IN ('pending', 'completed', 'failed', 'committed')
                    ),
                    payload_json TEXT,
                    metadata_json TEXT NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._path is None:
            raise RuntimeError("call journal is disabled")
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection


def _row_to_entry(row: tuple[Any, ...]) -> JournalEntry:
    (
        call_id,
        cache_key,
        namespace,
        role,
        model,
        state,
        payload_json,
        metadata_json,
        error_type,
        error_message,
        updated_at,
    ) = row
    try:
        payload = (
            _load_object(str(payload_json), "payload")
            if payload_json is not None
            else None
        )
        metadata = _load_object(str(metadata_json), "metadata")
        parsed_state = str(state)
        if parsed_state not in {"pending", "completed", "failed", "committed"}:
            raise ValueError("unknown journal state")
        timestamp = datetime.fromisoformat(str(updated_at))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid interactive call journal row") from exc
    return JournalEntry(
        call_id=str(call_id),
        cache_key=str(cache_key),
        namespace=str(namespace),
        role=str(role),
        model=str(model),
        state=parsed_state,  # type: ignore[arg-type]
        payload=payload,
        metadata=metadata,
        error_type=str(error_type) if error_type is not None else None,
        error_message=(
            str(error_message) if error_message is not None else None
        ),
        updated_at=timestamp,
    )


def _load_object(raw: str, field_name: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"journal {field_name} must be an object")
    return value
