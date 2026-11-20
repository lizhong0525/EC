"""In-memory, consent-aware storage for the four memory layers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from ecocompanion.domain.enums import MemoryKind
from ecocompanion.domain.schemas import MemoryRecordV1


class MemoryConflictError(ValueError):
    """Raised when an existing memory id is reused for different content."""


class ProtectedMemoryError(PermissionError):
    """Raised when ordinary deletion is attempted on the safety ledger."""


class MemoryStoreError(RuntimeError):
    """Raised when persisted memory events are invalid or cannot be applied."""


class MemoryService:
    """A deterministic memory store with optional durable event persistence.

    Working memory is session-scoped and may be ephemeral. Episodic and
    preference memories require explicit user authorization. Safety-ledger
    records are protected from ordinary deletion so an audit event cannot be
    silently removed through the conversational memory API.
    """

    _CONSENT_REQUIRED_KINDS = frozenset(
        {
            MemoryKind.EPISODIC,
            MemoryKind.PREFERENCE,
        }
    )

    def __init__(self, *, path: Path | None = None) -> None:
        self._records: dict[str, MemoryRecordV1] = {}
        self._path = path
        self._lock = RLock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load_events()

    def store(self, record: MemoryRecordV1) -> None:
        """Store a record idempotently after validating its write contract."""

        self._validate_record(record)
        with self._lock:
            existing = self._records.get(record.memory_id)
            if existing is None:
                self._append_event(
                    {
                        "operation": "STORE",
                        "record": record.model_dump(mode="json"),
                    }
                )
                self._records[record.memory_id] = record
                return
            if existing != record:
                raise MemoryConflictError(
                    f"memory_id {record.memory_id!r} already exists with different content"
                )

    def get(
        self,
        memory_id: str,
        *,
        now: datetime | None = None,
        include_expired: bool = False,
    ) -> MemoryRecordV1 | None:
        """Return one visible record without mutating the store."""

        effective_now = self._normalize_now(now)
        with self._lock:
            record = self._records.get(memory_id)
            if record is None:
                return None
            if not include_expired and self._is_expired(record, effective_now):
                return None
            return record

    def list_for_session(
        self,
        session_id: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        now: datetime | None = None,
        include_expired: bool = False,
    ) -> tuple[MemoryRecordV1, ...]:
        """List records in stable chronological order."""

        effective_now = self._normalize_now(now)
        selected_kinds = None if kinds is None else frozenset(kinds)
        with self._lock:
            records = (
                record
                for record in self._records.values()
                if record.session_id == session_id
                and (selected_kinds is None or record.kind in selected_kinds)
                and (include_expired or not self._is_expired(record, effective_now))
            )
            return tuple(sorted(records, key=lambda item: (item.created_at, item.memory_id)))

    def delete(self, memory_id: str) -> None:
        """Delete an ordinary memory idempotently.

        Safety-ledger entries require a separate retention/governance process
        and therefore cannot be removed through this method.
        """

        with self._lock:
            record = self._records.get(memory_id)
            if record is None:
                return
            if record.kind is MemoryKind.SAFETY_LEDGER:
                raise ProtectedMemoryError(
                    f"safety-ledger memory {memory_id!r} cannot be deleted as ordinary memory"
                )
            self._append_event({"operation": "DELETE", "memory_id": memory_id})
            del self._records[memory_id]

    def replace(self, record: MemoryRecordV1) -> None:
        """Correct an ordinary memory while preserving its stable identifier."""

        self._validate_record(record)
        with self._lock:
            existing = self._records.get(record.memory_id)
            if existing is None:
                raise KeyError(record.memory_id)
            if existing.kind is MemoryKind.SAFETY_LEDGER:
                raise ProtectedMemoryError(
                    f"safety-ledger memory {record.memory_id!r} cannot be replaced"
                )
            if existing.session_id != record.session_id:
                raise MemoryConflictError(
                    "replacement memory belongs to a different session"
                )
            if existing.kind is not record.kind:
                raise MemoryConflictError("replacement cannot change memory kind")
            if existing == record:
                return
            self._append_event(
                {
                    "operation": "REPLACE",
                    "record": record.model_dump(mode="json"),
                }
            )
            self._records[record.memory_id] = record

    def delete_ordinary_for_session(self, session_id: str) -> None:
        """Delete all non-ledger memories for a session."""

        with self._lock:
            deletable_ids = [
                memory_id
                for memory_id, record in self._records.items()
                if record.session_id == session_id and record.kind is not MemoryKind.SAFETY_LEDGER
            ]
            for memory_id in deletable_ids:
                self._append_event({"operation": "DELETE", "memory_id": memory_id})
                del self._records[memory_id]

    def _load_events(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise MemoryStoreError(f"cannot read memory store: {self._path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                self._apply_event(event)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MemoryStoreError(
                    f"invalid memory event at line {line_number}"
                ) from exc

    def _apply_event(self, event: object) -> None:
        if not isinstance(event, dict):
            raise TypeError("memory event must be an object")
        operation = event["operation"]
        if operation == "STORE":
            record = MemoryRecordV1.model_validate(event["record"])
            self._validate_record(record)
            existing = self._records.get(record.memory_id)
            if existing is not None and existing != record:
                raise MemoryConflictError(
                    f"persisted memory_id {record.memory_id!r} conflicts"
                )
            self._records[record.memory_id] = record
            return
        if operation == "REPLACE":
            record = MemoryRecordV1.model_validate(event["record"])
            self._validate_record(record)
            existing = self._records.get(record.memory_id)
            if existing is None:
                raise KeyError(record.memory_id)
            if existing.kind is MemoryKind.SAFETY_LEDGER:
                raise ProtectedMemoryError(
                    "persisted REPLACE cannot modify a safety-ledger record"
                )
            if existing.session_id != record.session_id or existing.kind is not record.kind:
                raise MemoryConflictError(
                    "persisted replacement changes memory ownership or kind"
                )
            self._records[record.memory_id] = record
            return
        if operation == "DELETE":
            memory_id = event["memory_id"]
            if not isinstance(memory_id, str):
                raise TypeError("DELETE memory_id must be a string")
            existing = self._records.get(memory_id)
            if existing is not None and existing.kind is MemoryKind.SAFETY_LEDGER:
                raise ProtectedMemoryError(
                    "persisted DELETE cannot remove a safety-ledger record"
                )
            self._records.pop(memory_id, None)
            return
        raise ValueError(f"unknown memory operation: {operation!r}")

    def _append_event(self, event: dict[str, Any]) -> None:
        if self._path is None:
            return
        serialized = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            descriptor = os.open(
                self._path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, f"{serialized}\n".encode())
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise MemoryStoreError(f"cannot append memory event: {self._path}") from exc

    @classmethod
    def _validate_record(cls, record: MemoryRecordV1) -> None:
        if not record.memory_id.strip():
            raise ValueError("memory_id cannot be blank")
        if not record.session_id.strip():
            raise ValueError("session_id cannot be blank")
        if not record.content.strip():
            raise ValueError("memory content cannot be blank")
        if record.kind in cls._CONSENT_REQUIRED_KINDS and not record.user_authorized:
            raise PermissionError(
                f"{record.kind.value} memory requires explicit user authorization"
            )
        cls._require_aware(record.created_at, "created_at")
        if record.expires_at is not None:
            cls._require_aware(record.expires_at, "expires_at")
            if record.expires_at <= record.created_at:
                raise ValueError("expires_at must be later than created_at")

    @staticmethod
    def _normalize_now(value: datetime | None) -> datetime:
        effective = datetime.now(UTC) if value is None else value
        MemoryService._require_aware(effective, "now")
        return effective

    @staticmethod
    def _require_aware(value: datetime, field_name: str) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware")

    @staticmethod
    def _is_expired(record: MemoryRecordV1, now: datetime) -> bool:
        return record.expires_at is not None and record.expires_at <= now
