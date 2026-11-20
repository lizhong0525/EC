"""Idempotent per-call cost ledger with explicit unknown-price accounting."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ecocompanion.providers.types import CompletionUsage, ModelRole

_ONE_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPricing:
    """CNY prices per one million tokens."""

    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None
    reasoning_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        values = (
            self.input_per_million,
            self.output_per_million,
            self.cached_input_per_million,
            self.reasoning_per_million,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("model prices cannot be negative")


@dataclass(frozen=True)
class CostRecord:
    call_id: str
    role: ModelRole
    model: str
    usage: CompletionUsage
    retry_index: int
    local_cache_hit: bool
    cost_cny: Decimal | None
    pricing_version: str | None
    recorded_at: datetime


@dataclass(frozen=True)
class CostSummary:
    known_cost_cny: Decimal
    unknown_price_calls: int
    call_count: int
    local_cache_hits: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int


class CostConflictError(RuntimeError):
    """Raised when a call id is reused with different accounting data."""


class CostLedger:
    """Append-only cost ledger.

    Duplicate identical records are idempotent. A duplicate call id with
    different content fails fast so retries cannot be double-counted silently.
    """

    def __init__(
        self,
        *,
        pricing: dict[str, ModelPricing] | None = None,
        pricing_version: str | None = None,
        path: Path | None = None,
    ) -> None:
        self._pricing = dict(pricing or {})
        self._pricing_version = pricing_version
        self._path = path
        self._lock = threading.RLock()
        self._records: dict[str, CostRecord] = {}
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def record_call(
        self,
        *,
        call_id: str,
        role: ModelRole,
        model: str,
        usage: CompletionUsage,
        retry_index: int = 0,
        local_cache_hit: bool = False,
        recorded_at: datetime | None = None,
    ) -> None:
        if not call_id.strip():
            raise ValueError("call_id cannot be empty")
        if not model.strip():
            raise ValueError("model cannot be empty")
        if retry_index < 0:
            raise ValueError("retry_index cannot be negative")
        cost = (
            Decimal(0)
            if local_cache_hit
            else _calculate_cost(usage, self._pricing.get(model))
        )
        record = CostRecord(
            call_id=call_id,
            role=role,
            model=model,
            usage=usage,
            retry_index=retry_index,
            local_cache_hit=local_cache_hit,
            cost_cny=cost,
            pricing_version=self._pricing_version if cost is not None else None,
            recorded_at=recorded_at or datetime.now(UTC),
        )

        with self._lock:
            existing = self._records.get(call_id)
            if existing is not None:
                if not _same_logical_call(existing, record):
                    raise CostConflictError(
                        "call id already exists with different cost data"
                    )
                return
            self._records[call_id] = record
            if self._path is not None:
                _append_json_line(self._path, _record_to_json(record))

    def records(self) -> tuple[CostRecord, ...]:
        with self._lock:
            return tuple(
                sorted(self._records.values(), key=lambda item: (item.recorded_at, item.call_id))
            )

    def get(self, call_id: str) -> CostRecord | None:
        """Return one persisted call record by its id.

        Callers that replay an exact-cache entry need the original provider
        usage and cost, rather than a synthetic zero-usage cache-hit record.
        The lookup is read-only and follows the same lock discipline as
        :meth:`records`.
        """

        if not call_id.strip():
            raise ValueError("call_id cannot be empty")
        with self._lock:
            return self._records.get(call_id)

    def summary(self, *, role: ModelRole | None = None) -> CostSummary:
        selected = [
            record
            for record in self.records()
            if role is None or record.role is role
        ]
        return CostSummary(
            known_cost_cny=sum(
                (record.cost_cny for record in selected if record.cost_cny is not None),
                start=Decimal(0),
            ),
            unknown_price_calls=sum(record.cost_cny is None for record in selected),
            call_count=len(selected),
            local_cache_hits=sum(record.local_cache_hit for record in selected),
            input_tokens=sum(record.usage.input_tokens for record in selected),
            output_tokens=sum(record.usage.output_tokens for record in selected),
            cached_input_tokens=sum(
                record.usage.cached_input_tokens for record in selected
            ),
            reasoning_tokens=sum(record.usage.reasoning_tokens for record in selected),
        )

    def _load_existing(self) -> None:
        if self._path is None or not self._path.exists():
            return
        for line_number, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = _record_from_json(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"invalid cost ledger record at line {line_number}"
                ) from exc
            existing = self._records.get(record.call_id)
            if existing is not None and existing != record:
                raise CostConflictError(
                    f"conflicting persisted call id: {record.call_id}"
                )
            self._records[record.call_id] = record


def _calculate_cost(
    usage: CompletionUsage,
    pricing: ModelPricing | None,
) -> Decimal | None:
    if pricing is None:
        return None
    uncached_input = usage.input_tokens - usage.cached_input_tokens
    cached_price = pricing.cached_input_per_million or pricing.input_per_million
    reasoning_price = pricing.reasoning_per_million or Decimal(0)
    return (
        Decimal(uncached_input) * pricing.input_per_million
        + Decimal(usage.cached_input_tokens) * cached_price
        + Decimal(usage.output_tokens) * pricing.output_per_million
        + Decimal(usage.reasoning_tokens) * reasoning_price
    ) / _ONE_MILLION


def _same_logical_call(left: CostRecord, right: CostRecord) -> bool:
    return (
        left.call_id == right.call_id
        and left.role is right.role
        and left.model == right.model
        and left.usage == right.usage
        and left.retry_index == right.retry_index
        and left.local_cache_hit == right.local_cache_hit
        and left.cost_cny == right.cost_cny
        and left.pricing_version == right.pricing_version
    )


def _record_to_json(record: CostRecord) -> dict[str, Any]:
    usage = asdict(record.usage)
    return {
        "call_id": record.call_id,
        "role": record.role.value,
        "model": record.model,
        "usage": usage,
        "retry_index": record.retry_index,
        "local_cache_hit": record.local_cache_hit,
        "cost_cny": str(record.cost_cny) if record.cost_cny is not None else None,
        "pricing_version": record.pricing_version,
        "recorded_at": record.recorded_at.isoformat(),
    }


def _record_from_json(payload: object) -> CostRecord:
    if not isinstance(payload, dict):
        raise TypeError("cost record must be an object")
    usage_payload = payload["usage"]
    if not isinstance(usage_payload, dict):
        raise TypeError("usage must be an object")
    cost_raw = payload.get("cost_cny")
    return CostRecord(
        call_id=str(payload["call_id"]),
        role=ModelRole(str(payload["role"])),
        model=str(payload["model"]),
        usage=CompletionUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0)),
            output_tokens=int(usage_payload.get("output_tokens", 0)),
            cached_input_tokens=int(usage_payload.get("cached_input_tokens", 0)),
            reasoning_tokens=int(usage_payload.get("reasoning_tokens", 0)),
        ),
        retry_index=int(payload.get("retry_index", 0)),
        local_cache_hit=bool(payload.get("local_cache_hit", False)),
        cost_cny=Decimal(str(cost_raw)) if cost_raw is not None else None,
        pricing_version=(
            str(payload["pricing_version"])
            if payload.get("pricing_version") is not None
            else None
        ),
        recorded_at=datetime.fromisoformat(str(payload["recorded_at"])),
    )


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{serialized}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
