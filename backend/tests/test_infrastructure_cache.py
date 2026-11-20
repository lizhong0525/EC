from pathlib import Path

import pytest

from ecocompanion.infrastructure import (
    CacheConflictError,
    ExactCache,
    build_exact_cache_key,
)


def test_exact_cache_key_is_canonical() -> None:
    left = build_exact_cache_key(
        namespace="chat-v1",
        request={"model": "a", "params": {"temperature": 0, "seed": 7}},
    )
    right = build_exact_cache_key(
        namespace="chat-v1",
        request={"params": {"seed": 7, "temperature": 0}, "model": "a"},
    )
    assert left == right


def test_exact_cache_round_trip_and_conflict(tmp_path: Path) -> None:
    cache = ExactCache(tmp_path / "responses.sqlite3")
    key = build_exact_cache_key(
        namespace="chat-v1",
        request={"model": "a", "replicate_id": 1},
    )
    cache.put(
        key=key,
        namespace="chat-v1",
        payload={"content": "first"},
        metadata={"schema": "AgentDecisionV1"},
    )
    cache.put(
        key=key,
        namespace="chat-v1",
        payload={"content": "first"},
        metadata={"schema": "AgentDecisionV1"},
    )

    entry = cache.get(key)
    assert entry is not None
    assert entry.payload == {"content": "first"}

    with pytest.raises(CacheConflictError):
        cache.put(
            key=key,
            namespace="chat-v1",
            payload={"content": "different stochastic sample"},
            metadata={"schema": "AgentDecisionV1"},
        )


def test_cache_key_rejects_secret_fields() -> None:
    with pytest.raises(ValueError, match="secret field"):
        build_exact_cache_key(
            namespace="chat-v1",
            request={"headers": {"Authorization": "Bearer do-not-store"}},
        )
