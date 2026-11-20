from datetime import UTC, datetime, timedelta

import pytest

from ecocompanion.agent.memory import (
    MemoryConflictError,
    MemoryService,
    ProtectedMemoryError,
)
from ecocompanion.domain.enums import MemoryKind
from ecocompanion.domain.schemas import MemoryRecordV1

NOW = datetime(2026, 9, 3, 8, 0, tzinfo=UTC)


def memory_record(
    memory_id: str,
    kind: MemoryKind,
    *,
    authorized: bool = False,
    content: str = "记忆内容",
    expires_at: datetime | None = None,
) -> MemoryRecordV1:
    return MemoryRecordV1(
        memory_id=memory_id,
        session_id="session-001",
        kind=kind,
        content=content,
        created_at=NOW,
        expires_at=expires_at,
        user_authorized=authorized,
    )


def test_memory_service_supports_all_four_layers() -> None:
    service = MemoryService()
    records = (
        memory_record("working", MemoryKind.WORKING),
        memory_record("episodic", MemoryKind.EPISODIC, authorized=True),
        memory_record("preference", MemoryKind.PREFERENCE, authorized=True),
        memory_record("ledger", MemoryKind.SAFETY_LEDGER),
    )

    for record in records:
        service.store(record)

    assert service.list_for_session("session-001", now=NOW) == tuple(
        sorted(records, key=lambda item: item.memory_id)
    )


@pytest.mark.parametrize("kind", [MemoryKind.EPISODIC, MemoryKind.PREFERENCE])
def test_persistent_conversational_memory_requires_authorization(
    kind: MemoryKind,
) -> None:
    service = MemoryService()

    with pytest.raises(PermissionError, match="explicit user authorization"):
        service.store(memory_record("memory-001", kind))


def test_ordinary_memory_can_be_deleted_but_safety_ledger_is_protected() -> None:
    service = MemoryService()
    service.store(memory_record("working", MemoryKind.WORKING))
    service.store(memory_record("ledger", MemoryKind.SAFETY_LEDGER))

    service.delete("working")
    service.delete("missing")

    assert service.get("working", now=NOW) is None
    with pytest.raises(ProtectedMemoryError, match="cannot be deleted"):
        service.delete("ledger")
    assert service.get("ledger", now=NOW) is not None


def test_session_deletion_preserves_safety_ledger() -> None:
    service = MemoryService()
    service.store(memory_record("working", MemoryKind.WORKING))
    service.store(
        memory_record("preference", MemoryKind.PREFERENCE, authorized=True)
    )
    service.store(memory_record("ledger", MemoryKind.SAFETY_LEDGER))

    service.delete_ordinary_for_session("session-001")

    remaining = service.list_for_session("session-001", now=NOW)
    assert tuple(record.memory_id for record in remaining) == ("ledger",)


def test_expired_memory_is_hidden_without_mutating_audit_storage() -> None:
    service = MemoryService()
    record = memory_record(
        "working",
        MemoryKind.WORKING,
        expires_at=NOW + timedelta(hours=1),
    )
    service.store(record)

    assert service.get("working", now=NOW + timedelta(hours=2)) is None
    assert (
        service.get(
            "working",
            now=NOW + timedelta(hours=2),
            include_expired=True,
        )
        == record
    )


def test_memory_store_is_idempotent_and_rejects_conflicting_id_reuse() -> None:
    service = MemoryService()
    record = memory_record("memory-001", MemoryKind.WORKING)

    service.store(record)
    service.store(record)

    with pytest.raises(MemoryConflictError, match="different content"):
        service.store(
            memory_record(
                "memory-001",
                MemoryKind.WORKING,
                content="不同内容",
            )
        )


def test_memory_rejects_invalid_expiry_and_naive_query_time() -> None:
    service = MemoryService()

    with pytest.raises(ValueError, match="expires_at"):
        service.store(
            memory_record(
                "memory-001",
                MemoryKind.WORKING,
                expires_at=NOW,
            )
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        service.list_for_session(
            "session-001",
            now=datetime(2026, 9, 3, 8, 0),
        )


def test_memory_store_replays_durable_events(tmp_path) -> None:
    path = tmp_path / "memory" / "events.jsonl"
    service = MemoryService(path=path)
    working = memory_record("working-persisted", MemoryKind.WORKING)
    ledger = memory_record("ledger-persisted", MemoryKind.SAFETY_LEDGER)
    service.store(working)
    service.store(ledger)
    service.delete(working.memory_id)

    reloaded = MemoryService(path=path)

    assert reloaded.get(working.memory_id, now=NOW, include_expired=True) is None
    assert reloaded.get(ledger.memory_id, now=NOW, include_expired=True) == ledger


def test_ordinary_memory_can_be_corrected_and_replayed(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    service = MemoryService(path=path)
    original = memory_record(
        "preference-corrected",
        MemoryKind.PREFERENCE,
        authorized=True,
        content="请叫我小林",
    )
    corrected = original.model_copy(update={"content": "请叫我林同学"})
    service.store(original)
    service.replace(corrected)

    assert MemoryService(path=path).get(corrected.memory_id, now=NOW) == corrected

    with pytest.raises(ProtectedMemoryError):
        ledger = memory_record("ledger-correct", MemoryKind.SAFETY_LEDGER)
        service.store(ledger)
        service.replace(ledger.model_copy(update={"content": "changed"}))
