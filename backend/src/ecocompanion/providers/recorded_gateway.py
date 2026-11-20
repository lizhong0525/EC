"""Synchronous, model-pinned provider gateway with cache and cost accounting.

The benchmark runners are intentionally synchronous while the production
OpenAI-compatible client is asynchronous.  This module is the narrow bridge
between those two worlds.  It also centralizes the reproducibility contract:

* every request is pinned to exactly one frozen model binding;
* exact-cache keys include prompts, schema, seed, and replicate;
* successful provider calls and local cache hits are recorded idempotently;
* formal runs fail before the first request when any model price is unknown.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel

from ecocompanion.infrastructure import (
    CostLedger,
    DurableCallJournal,
    ExactCache,
    JournalConflictError,
    JournalPendingError,
    ModelPricing,
    build_exact_cache_key,
    canonical_json,
)
from ecocompanion.providers.types import (
    ChatCompletion,
    CompletionUsage,
    ModelRole,
    StructuredCompletion,
    StructuredMode,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
ValueT = TypeVar("ValueT")
_MAX_API_BUDGET_CNY = Decimal("5000")
_ONE_MILLION = Decimal(1_000_000)


class InteractiveRunIntegrityError(RuntimeError):
    """A formal run cannot continue without violating its frozen contract."""


class InteractiveBudgetExceeded(InteractiveRunIntegrityError):
    """Raised before a provider call that would exceed the frozen budget."""


class AsyncCompletionClient(Protocol):
    """Subset implemented by :class:`OpenAICompatibleClient`."""

    async def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        temperature: float | None = 0.0,
    ) -> ChatCompletion:
        """Return one model-pinned text completion."""

    async def complete_structured(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        schema: type[ModelT],
        max_tokens: int = 512,
        temperature: float | None = 0.0,
        modes: Sequence[StructuredMode] = (
            StructuredMode.JSON_SCHEMA,
            StructuredMode.PLAIN_JSON,
        ),
    ) -> StructuredCompletion[ModelT]:
        """Return one model-pinned, schema-validated completion."""

    async def aclose(self) -> None:
        """Close provider resources."""


class SyncCompletionGateway(Protocol):
    """Provider-neutral port consumed by benchmark adapters."""

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float | None,
    ) -> ChatCompletion:
        """Return one model-pinned text completion."""

    def complete_structured(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        schema: type[ModelT],
        max_tokens: int,
        temperature: float | None,
        modes: Sequence[StructuredMode],
    ) -> StructuredCompletion[ModelT]:
        """Return one model-pinned, schema-validated completion."""


class _BackgroundEventLoop:
    """Own one event loop so an async HTTP client never crosses event loops."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ecocompanion-provider-loop",
            daemon=True,
        )
        self._closed = False
        self._thread.start()
        self._started.wait()

    def submit(self, awaitable: Awaitable[ValueT]) -> ValueT:
        if self._closed:
            raise RuntimeError("provider event-loop bridge is closed")

        async def wait_for_value() -> ValueT:
            return await awaitable

        future = asyncio.run_coroutine_threadsafe(wait_for_value(), self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("provider event-loop bridge did not stop")
        self._loop.close()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()


class OpenAICompatibleSyncGateway:
    """Adapt a dedicated async client to synchronous benchmark ports.

    The supplied client must not be used concurrently from another event loop.
    Create a fresh client after model discovery when starting a formal run.
    """

    def __init__(
        self,
        client: AsyncCompletionClient,
        *,
        close_client: bool = False,
    ) -> None:
        self._client = client
        self._close_client = close_client
        self._bridge = _BackgroundEventLoop()
        self._closed = False

    def __enter__(self) -> OpenAICompatibleSyncGateway:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float | None,
    ) -> ChatCompletion:
        self._ensure_open()
        return self._bridge.submit(
            self._client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

    def complete_structured(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        schema: type[ModelT],
        max_tokens: int,
        temperature: float | None,
        modes: Sequence[StructuredMode],
    ) -> StructuredCompletion[ModelT]:
        self._ensure_open()
        return self._bridge.submit(
            self._client.complete_structured(
                model=model,
                messages=messages,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                modes=modes,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._close_client:
            self._bridge.submit(self._client.aclose())
        self._bridge.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OpenAI-compatible gateway is closed")


@dataclass(frozen=True, slots=True)
class InteractiveModelBinding:
    """One immutable model-role assignment used by an experiment."""

    model_id: str
    family: str | None
    role: ModelRole
    pricing: ModelPricing | None
    allowed_response_model_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be blank")
        if self.family is not None and not self.family.strip():
            raise ValueError("family cannot be blank when provided")


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Stable identifiers that make retries and resumes idempotent."""

    run_id: str
    invocation_id: str
    seed: int
    replicate_id: int

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be blank")
        if not self.invocation_id.strip():
            raise ValueError("invocation_id cannot be blank")
        if self.replicate_id < 0:
            raise ValueError("replicate_id cannot be negative")


@dataclass(frozen=True, slots=True)
class RecordedCompletion(Generic[ValueT]):
    """Validated value plus auditable provider/cost metadata."""

    value: ValueT
    completion: ChatCompletion
    call_id: str
    cache_key: str
    cache_hit: bool
    estimated_cost_cny: Decimal | None
    structured_mode: StructuredMode | None = None


class RecordedModelInvoker:
    """Command service for exact-cached, costed model calls."""

    def __init__(
        self,
        *,
        gateway: SyncCompletionGateway,
        cache: ExactCache,
        bindings: Sequence[InteractiveModelBinding],
        pricing_version: str | None,
        cost_ledger_path: Path | None = None,
        call_journal_path: Path | None = None,
        formal_mode: bool = True,
        api_budget_cny: Decimal = _MAX_API_BUDGET_CNY,
    ) -> None:
        normalized = tuple(bindings)
        if not normalized:
            raise ValueError("at least one model binding is required")
        model_ids = [binding.model_id for binding in normalized]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model bindings must use distinct model IDs")
        if api_budget_cny <= 0:
            raise ValueError("api_budget_cny must be positive")
        if api_budget_cny > _MAX_API_BUDGET_CNY:
            raise ValueError("api_budget_cny cannot exceed the 5000 CNY hard limit")
        if formal_mode:
            missing = sorted(
                binding.model_id
                for binding in normalized
                if binding.pricing is None
            )
            if missing:
                raise ValueError(
                    f"formal mode requires explicit prices for models: {missing}"
                )
            if pricing_version is None or not pricing_version.strip():
                raise ValueError("formal mode requires a pricing_version")

        self._gateway = gateway
        self._cache = cache
        self._bindings = {binding.model_id: binding for binding in normalized}
        self._ledger = CostLedger(
            pricing={
                binding.model_id: binding.pricing
                for binding in normalized
                if binding.pricing is not None
            },
            pricing_version=pricing_version,
            path=cost_ledger_path,
        )
        if call_journal_path is None and cost_ledger_path is not None:
            call_journal_path = cost_ledger_path.with_name(
                f"{cost_ledger_path.stem}.calls.sqlite3"
            )
        self._journal = DurableCallJournal(call_journal_path)
        self._formal_mode = formal_mode
        self._api_budget_cny = api_budget_cny

    @property
    def cost_ledger(self) -> CostLedger:
        return self._ledger

    @property
    def formal_mode(self) -> bool:
        return self._formal_mode

    def complete_text(
        self,
        *,
        binding: InteractiveModelBinding,
        context: InvocationContext,
        messages: Sequence[Mapping[str, Any]],
        prompt_version: str,
        max_tokens: int,
        temperature: float | None,
    ) -> RecordedCompletion[str]:
        self._require_binding(binding)
        request = self._request_payload(
            binding=binding,
            context=context,
            messages=messages,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=None,
            modes=(),
        )
        namespace = "interactive-text-v1"
        cache_key = build_exact_cache_key(namespace=namespace, request=request)
        call_id = self._call_id(context, cache_key)
        entry = self._cache.get(cache_key)
        journal_entry = self._journal.get(call_id)
        self._reject_unresolved_journal(journal_entry)
        if (
            journal_entry is not None
            and journal_entry.state in {"completed", "committed"}
            and (entry is None or self._ledger.get(call_id) is None)
        ):
            return self._replay_text_journal(
                entry=journal_entry,
                binding=binding,
                call_id=call_id,
                cache_key=cache_key,
                namespace=namespace,
                prompt_version=prompt_version,
            )
        if entry is not None:
            self._settle_completed_journal(
                journal_entry=journal_entry,
                cache_entry=entry,
                call_id=call_id,
            )
            content = entry.payload.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("cached text completion is invalid")
            usage, retry_count, cost = self._cache_hit_metadata(
                call_id=call_id,
                binding=binding,
                payload=entry.payload,
            )
            completion = self._cached_completion(
                binding,
                entry.payload,
                content,
                usage=usage,
                retry_count=retry_count,
            )
            return RecordedCompletion(
                value=content,
                completion=completion,
                call_id=call_id,
                cache_key=cache_key,
                cache_hit=True,
                estimated_cost_cny=cost,
            )

        self._ensure_budget_for_call(
            binding=binding,
            messages=messages,
            max_tokens=max_tokens,
        )
        self._begin_journal(
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            binding=binding,
        )
        try:
            completion = self._gateway.chat_completion(
                model=binding.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self._validate_model_identity(binding, completion)
            content = (completion.content or "").strip()
            if not content:
                raise RuntimeError("provider returned an empty text completion")
        except Exception as exc:
            self._fail_journal(call_id, exc)
            raise
        payload = self._completion_payload(completion)
        payload["content"] = content
        cost = self._estimate_cost(binding, completion.usage)
        payload["estimated_cost_cny"] = (
            str(cost) if cost is not None else None
        )
        self._complete_journal(
            call_id=call_id,
            binding=binding,
            cache_key=cache_key,
            namespace=namespace,
            payload=payload,
            metadata={"prompt_version": prompt_version, "kind": "text"},
        )
        cost = self._record_live_call(
            call_id=call_id,
            binding=binding,
            completion=completion,
        )
        try:
            self._cache.put(
                key=cache_key,
                namespace=namespace,
                payload=payload,
                metadata={"prompt_version": prompt_version, "kind": "text"},
            )
            self._journal.mark_committed(call_id)
        except Exception:
            # The completed journal is intentionally retained.  The next
            # invocation will repair the failed sink without another API call.
            raise
        return RecordedCompletion(
            value=content,
            completion=completion,
            call_id=call_id,
            cache_key=cache_key,
            cache_hit=False,
            estimated_cost_cny=cost,
        )

    def complete_structured(
        self,
        *,
        binding: InteractiveModelBinding,
        context: InvocationContext,
        messages: Sequence[Mapping[str, Any]],
        schema: type[ModelT],
        prompt_version: str,
        max_tokens: int,
        temperature: float | None,
        modes: Sequence[StructuredMode] = (
            StructuredMode.JSON_SCHEMA,
            StructuredMode.PLAIN_JSON,
        ),
    ) -> RecordedCompletion[ModelT]:
        self._require_binding(binding)
        normalized_modes = tuple(modes)
        if not normalized_modes:
            raise ValueError("at least one structured-output mode is required")
        request = self._request_payload(
            binding=binding,
            context=context,
            messages=messages,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
            modes=normalized_modes,
        )
        namespace = "interactive-structured-v1"
        cache_key = build_exact_cache_key(namespace=namespace, request=request)
        call_id = self._call_id(context, cache_key)
        entry = self._cache.get(cache_key)
        journal_entry = self._journal.get(call_id)
        self._reject_unresolved_journal(journal_entry)
        if (
            journal_entry is not None
            and journal_entry.state in {"completed", "committed"}
            and (entry is None or self._ledger.get(call_id) is None)
        ):
            return self._replay_structured_journal(
                entry=journal_entry,
                binding=binding,
                call_id=call_id,
                cache_key=cache_key,
                namespace=namespace,
                schema=schema,
                prompt_version=prompt_version,
            )
        if entry is not None:
            self._settle_completed_journal(
                journal_entry=journal_entry,
                cache_entry=entry,
                call_id=call_id,
            )
            raw_value = entry.payload.get("value")
            value = schema.model_validate(raw_value)
            usage, retry_count, cost = self._cache_hit_metadata(
                call_id=call_id,
                binding=binding,
                payload=entry.payload,
            )
            completion = self._cached_completion(
                binding,
                entry.payload,
                value.model_dump_json(),
                usage=usage,
                retry_count=retry_count,
            )
            raw_mode = entry.payload.get("structured_mode")
            mode = StructuredMode(str(raw_mode))
            return RecordedCompletion(
                value=value,
                completion=completion,
                call_id=call_id,
                cache_key=cache_key,
                cache_hit=True,
                estimated_cost_cny=cost,
                structured_mode=mode,
            )

        self._ensure_budget_for_call(
            binding=binding,
            messages=messages,
            max_tokens=max_tokens,
        )
        self._begin_journal(
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            binding=binding,
        )
        try:
            structured = self._gateway.complete_structured(
                model=binding.model_id,
                messages=messages,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                modes=normalized_modes,
            )
            self._validate_model_identity(binding, structured.completion)
        except Exception as exc:
            self._fail_journal(call_id, exc)
            raise
        payload = self._completion_payload(structured.completion)
        payload.update(
            {
                "value": structured.value.model_dump(mode="json"),
                "structured_mode": structured.mode.value,
            }
        )
        cost = self._estimate_cost(
            binding,
            structured.completion.usage,
        )
        payload["estimated_cost_cny"] = (
            str(cost) if cost is not None else None
        )
        self._complete_journal(
            call_id=call_id,
            binding=binding,
            cache_key=cache_key,
            namespace=namespace,
            payload=payload,
            metadata={
                "prompt_version": prompt_version,
                "kind": "structured",
                "schema": schema.__name__,
            },
        )
        cost = self._record_live_call(
            call_id=call_id,
            binding=binding,
            completion=structured.completion,
        )
        try:
            self._cache.put(
                key=cache_key,
                namespace=namespace,
                payload=payload,
                metadata={
                    "prompt_version": prompt_version,
                    "kind": "structured",
                    "schema": schema.__name__,
                },
            )
            self._journal.mark_committed(call_id)
        except Exception:
            raise
        return RecordedCompletion(
            value=structured.value,
            completion=structured.completion,
            call_id=call_id,
            cache_key=cache_key,
            cache_hit=False,
            estimated_cost_cny=cost,
            structured_mode=structured.mode,
        )

    def _begin_journal(
        self,
        *,
        call_id: str,
        cache_key: str,
        namespace: str,
        binding: InteractiveModelBinding,
    ) -> None:
        try:
            existing = self._journal.begin(
                call_id=call_id,
                cache_key=cache_key,
                namespace=namespace,
                role=binding.role.value,
                model=binding.model_id,
            )
        except (JournalConflictError, JournalPendingError) as exc:
            raise InteractiveRunIntegrityError(
                f"interactive call journal rejected {call_id}"
            ) from exc
        if existing is not None:
            raise InteractiveRunIntegrityError(
                f"interactive call journal already contains a completion: {call_id}"
            )

    def _complete_journal(
        self,
        *,
        call_id: str,
        cache_key: str,
        namespace: str,
        binding: InteractiveModelBinding,
        payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        try:
            self._journal.complete(
                call_id=call_id,
                cache_key=cache_key,
                namespace=namespace,
                role=binding.role.value,
                model=binding.model_id,
                payload=payload,
                metadata=metadata,
            )
        except (JournalConflictError, JournalPendingError) as exc:
            raise InteractiveRunIntegrityError(
                f"interactive call journal could not persist {call_id}"
            ) from exc

    def _fail_journal(self, call_id: str, error: BaseException) -> None:
        try:
            self._journal.fail(
                call_id=call_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        except Exception as journal_error:
            raise InteractiveRunIntegrityError(
                f"interactive call journal could not persist provider failure: "
                f"{call_id}"
            ) from journal_error

    @staticmethod
    def _estimate_cost(
        binding: InteractiveModelBinding,
        usage: CompletionUsage,
    ) -> Decimal | None:
        pricing = binding.pricing
        if pricing is None:
            return None
        uncached_input = usage.input_tokens - usage.cached_input_tokens
        cached_price = (
            pricing.cached_input_per_million
            or pricing.input_per_million
        )
        reasoning_price = pricing.reasoning_per_million or Decimal(0)
        return (
            Decimal(uncached_input) * pricing.input_per_million
            + Decimal(usage.cached_input_tokens) * cached_price
            + Decimal(usage.output_tokens) * pricing.output_per_million
            + Decimal(usage.reasoning_tokens) * reasoning_price
        ) / _ONE_MILLION

    def _replay_text_journal(
        self,
        *,
        entry: Any,
        binding: InteractiveModelBinding,
        call_id: str,
        cache_key: str,
        namespace: str,
        prompt_version: str,
    ) -> RecordedCompletion[str]:
        payload = self._validated_journal_payload(
            entry=entry,
            binding=binding,
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            expected_kind="text",
            expected_prompt_version=prompt_version,
        )
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise InteractiveRunIntegrityError(
                "completed journal contains an invalid text completion"
            )
        try:
            usage = _cached_usage(payload)
            retry_count = _cached_retry_count(payload)
            completion = self._cached_completion(
                binding,
                payload,
                content,
                usage=usage,
                retry_count=retry_count,
            )
        except (RuntimeError, ValueError) as exc:
            raise InteractiveRunIntegrityError(
                "completed journal contains invalid provider metadata"
            ) from exc
        materialized_cost = self._materialize_journal(
            entry=entry,
            binding=binding,
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            completion=completion,
            payload=payload,
        )
        estimated_cost: Decimal | None
        if materialized_cost is not None:
            # ``_materialize_journal`` is authoritative when the ledger was
            # already present; this preserves exact Decimal precision.
            estimated_cost = materialized_cost
        else:
            estimated_cost = _cached_cost(payload)
        return RecordedCompletion(
            value=content,
            completion=completion,
            call_id=call_id,
            cache_key=cache_key,
            cache_hit=False,
            estimated_cost_cny=estimated_cost,
        )

    def _replay_structured_journal(
        self,
        *,
        entry: Any,
        binding: InteractiveModelBinding,
        call_id: str,
        cache_key: str,
        namespace: str,
        schema: type[ModelT],
        prompt_version: str,
    ) -> RecordedCompletion[ModelT]:
        payload = self._validated_journal_payload(
            entry=entry,
            binding=binding,
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            expected_kind="structured",
            expected_prompt_version=prompt_version,
            expected_schema=schema.__name__,
        )
        try:
            value = schema.model_validate(payload.get("value"))
            usage = _cached_usage(payload)
            retry_count = _cached_retry_count(payload)
            completion = self._cached_completion(
                binding,
                payload,
                value.model_dump_json(),
                usage=usage,
                retry_count=retry_count,
            )
            mode = StructuredMode(str(payload["structured_mode"]))
        except (RuntimeError, ValueError, TypeError) as exc:
            raise InteractiveRunIntegrityError(
                "completed journal contains invalid structured metadata"
            ) from exc
        materialized_cost = self._materialize_journal(
            entry=entry,
            binding=binding,
            call_id=call_id,
            cache_key=cache_key,
            namespace=namespace,
            completion=completion,
            payload=payload,
        )
        return RecordedCompletion(
            value=value,
            completion=completion,
            call_id=call_id,
            cache_key=cache_key,
            cache_hit=False,
            estimated_cost_cny=(
                materialized_cost
                if materialized_cost is not None
                else _cached_cost(payload)
            ),
            structured_mode=mode,
        )

    def _materialize_journal(
        self,
        *,
        entry: Any,
        binding: InteractiveModelBinding,
        call_id: str,
        cache_key: str,
        namespace: str,
        completion: ChatCompletion,
        payload: dict[str, Any],
    ) -> Decimal | None:
        cost = self._record_live_call(
            call_id=call_id,
            binding=binding,
            completion=completion,
        )
        recorded_cost = _cached_cost(payload)
        if (
            recorded_cost is not None
            and cost is not None
            and recorded_cost != cost
        ):
            raise InteractiveRunIntegrityError(
                "journal completion cost conflicts with the cost ledger"
            )
        metadata = dict(entry.metadata)
        try:
            self._cache.put(
                key=cache_key,
                namespace=namespace,
                payload=payload,
                metadata=metadata,
            )
            self._journal.mark_committed(call_id)
        except Exception:
            # Keep the completed journal row for another repair attempt.
            raise
        return cost if cost is not None else recorded_cost

    def _validated_journal_payload(
        self,
        *,
        entry: Any,
        binding: InteractiveModelBinding,
        call_id: str,
        cache_key: str,
        namespace: str,
        expected_kind: str,
        expected_prompt_version: str,
        expected_schema: str | None = None,
    ) -> dict[str, Any]:
        if entry is None or entry.state not in {"completed", "committed"}:
            raise InteractiveRunIntegrityError(
                f"interactive call journal has no completed payload: {call_id}"
            )
        if (
            entry.call_id != call_id
            or entry.cache_key != cache_key
            or entry.namespace != namespace
            or entry.role != binding.role.value
            or entry.model != binding.model_id
        ):
            raise InteractiveRunIntegrityError(
                "interactive call journal identity conflicts with the request"
            )
        metadata = entry.metadata
        if (
            metadata.get("kind") != expected_kind
            or metadata.get("prompt_version") != expected_prompt_version
        ):
            raise InteractiveRunIntegrityError(
                "interactive call journal metadata conflicts with the request"
            )
        if expected_schema is not None and metadata.get("schema") != expected_schema:
            raise InteractiveRunIntegrityError(
                "interactive structured journal schema conflicts with the request"
            )
        if entry.payload is None:
            raise InteractiveRunIntegrityError(
                f"interactive call journal payload is missing: {call_id}"
            )
        return dict(entry.payload)

    @staticmethod
    def _reject_unresolved_journal(entry: Any) -> None:
        if entry is not None and entry.state == "pending":
            raise InteractiveRunIntegrityError(
                f"interactive call has an unresolved durable outcome: "
                f"{entry.call_id}"
            )

    def _settle_completed_journal(
        self,
        *,
        journal_entry: Any,
        cache_entry: Any,
        call_id: str,
    ) -> None:
        """Verify and close a journal row once both sinks are already present."""

        if journal_entry is None or journal_entry.state not in {
            "completed",
            "committed",
        }:
            return
        if journal_entry.payload is None:
            raise InteractiveRunIntegrityError(
                f"completed journal payload is missing: {call_id}"
            )
        if (
            canonical_json(journal_entry.payload)
            != canonical_json(cache_entry.payload)
            or canonical_json(journal_entry.metadata)
            != canonical_json(cache_entry.metadata)
        ):
            raise InteractiveRunIntegrityError(
                f"cache payload conflicts with the completed journal: {call_id}"
            )
        if journal_entry.state == "completed":
            try:
                self._journal.mark_committed(call_id)
            except Exception as exc:
                raise InteractiveRunIntegrityError(
                    f"interactive call journal could not be committed: {call_id}"
                ) from exc

    def _require_binding(self, binding: InteractiveModelBinding) -> None:
        configured = self._bindings.get(binding.model_id)
        if configured != binding:
            raise ValueError(
                f"model binding is not frozen in this invoker: {binding.model_id}"
            )

    @staticmethod
    def _request_payload(
        *,
        binding: InteractiveModelBinding,
        context: InvocationContext,
        messages: Sequence[Mapping[str, Any]],
        prompt_version: str,
        max_tokens: int,
        temperature: float | None,
        schema: type[BaseModel] | None,
        modes: Sequence[StructuredMode],
    ) -> dict[str, Any]:
        if not prompt_version.strip():
            raise ValueError("prompt_version cannot be blank")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        return {
            "model": binding.model_id,
            "role": binding.role.value,
            "messages": [dict(message) for message in messages],
            "prompt_version": prompt_version,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "schema": schema.model_json_schema() if schema is not None else None,
            "modes": [mode.value for mode in modes],
            "seed": context.seed,
            "replicate_id": context.replicate_id,
        }

    @staticmethod
    def _call_id(context: InvocationContext, cache_key: str) -> str:
        return (
            f"{context.run_id}:{context.invocation_id}:"
            f"{cache_key[:20]}"
        )

    def _record_live_call(
        self,
        *,
        call_id: str,
        binding: InteractiveModelBinding,
        completion: ChatCompletion,
    ) -> Decimal | None:
        self._ledger.record_call(
            call_id=call_id,
            role=binding.role,
            model=binding.model_id,
            usage=completion.usage,
            retry_index=completion.retry_count,
            local_cache_hit=False,
        )
        record = next(
            record
            for record in self._ledger.records()
            if record.call_id == call_id
        )
        if self._formal_mode and record.cost_cny is None:
            raise InteractiveRunIntegrityError(
                "formal model call produced unknown cost"
            )
        if self._ledger.summary().known_cost_cny > self._api_budget_cny:
            raise InteractiveBudgetExceeded(
                "provider usage exceeded the frozen API budget"
            )
        return record.cost_cny

    def _ensure_budget_for_call(
        self,
        *,
        binding: InteractiveModelBinding,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
    ) -> None:
        summary = self._ledger.summary()
        if self._formal_mode and summary.unknown_price_calls:
            raise InteractiveRunIntegrityError(
                "formal budget accounting contains unknown-price calls"
            )
        pricing = binding.pricing
        if pricing is None:
            if self._formal_mode:
                raise InteractiveRunIntegrityError(
                    f"formal model price is unavailable: {binding.model_id}"
                )
            return
        serialized = canonical_json([dict(message) for message in messages])
        # A UTF-8 byte is a conservative upper bound for ordinary tokenizer
        # units; the fixed allowance covers chat framing and special tokens.
        input_token_reserve = len(serialized.encode("utf-8")) + 512
        cached_price = (
            pricing.cached_input_per_million
            if pricing.cached_input_per_million is not None
            else pricing.input_per_million
        )
        del cached_price  # New calls are conservatively priced as uncached.
        reasoning_price = (
            pricing.reasoning_per_million
            if pricing.reasoning_per_million is not None
            else Decimal(0)
        )
        reserve = (
            Decimal(input_token_reserve) * pricing.input_per_million
            + Decimal(max_tokens) * pricing.output_per_million
            + Decimal(max_tokens) * reasoning_price
        ) / _ONE_MILLION
        if summary.known_cost_cny + reserve > self._api_budget_cny:
            raise InteractiveBudgetExceeded(
                "next provider call would exceed the frozen API budget"
            )

    def _cache_hit_metadata(
        self,
        *,
        call_id: str,
        binding: InteractiveModelBinding,
        payload: Mapping[str, Any],
    ) -> tuple[CompletionUsage, int, Decimal | None]:
        payload_has_usage = "usage" in payload
        usage = _cached_usage(payload)
        retry_count = _cached_retry_count(payload)
        existing = self._ledger.get(call_id)
        if existing is not None:
            if existing.model != binding.model_id or existing.role is not binding.role:
                raise InteractiveRunIntegrityError(
                    "persisted cost record conflicts with model binding"
                )
            # A live record is the accounting authority on resume.  If the
            # cache carries usage metadata, it must agree with that record;
            # otherwise a tampered/stale cache could alter the audit trail.
            if payload_has_usage and existing.usage != usage:
                raise InteractiveRunIntegrityError(
                    "cached completion usage conflicts with the cost ledger"
                )
            usage = existing.usage
            retry_count = existing.retry_index
            return usage, retry_count, existing.cost_cny
        self._ledger.record_call(
            call_id=call_id,
            role=binding.role,
            model=binding.model_id,
            # A cache hit performs no provider work and therefore has no
            # billable usage.  Preserve the original usage on the synthetic
            # completion above for trace fidelity, while keeping accounting
            # additive only for real calls.
            usage=CompletionUsage(),
            retry_index=0,
            local_cache_hit=True,
        )
        record = self._ledger.get(call_id)
        if record is None:
            raise RuntimeError("cache-hit cost record was not persisted")
        return usage, retry_count, record.cost_cny

    @staticmethod
    def _validate_model_identity(
        binding: InteractiveModelBinding,
        completion: ChatCompletion,
    ) -> None:
        if completion.requested_model != binding.model_id:
            raise InteractiveRunIntegrityError(
                "provider completion changed the requested model"
            )
        allowed = {binding.model_id, *binding.allowed_response_model_ids}
        if completion.response_model not in allowed:
            raise InteractiveRunIntegrityError(
                "provider returned a model outside the frozen response-model allow-list"
            )

    @staticmethod
    def _completion_payload(completion: ChatCompletion) -> dict[str, Any]:
        return {
            "requested_model": completion.requested_model,
            "response_model": completion.response_model,
            "finish_reason": completion.finish_reason,
            "response_id": completion.response_id,
            "tool_arguments": completion.tool_arguments,
            "retry_count": completion.retry_count,
            "usage": asdict(completion.usage),
        }

    @staticmethod
    def _cached_completion(
        binding: InteractiveModelBinding,
        payload: Mapping[str, Any],
        content: str,
        *,
        usage: CompletionUsage,
        retry_count: int,
    ) -> ChatCompletion:
        requested_model = str(
            payload.get("requested_model", binding.model_id)
        )
        if requested_model != binding.model_id:
            raise InteractiveRunIntegrityError(
                "cached completion conflicts with the frozen requested model"
            )
        response_model = str(payload.get("response_model", binding.model_id))
        synthetic = ChatCompletion(
            requested_model=requested_model,
            response_model=response_model,
            content=content,
            finish_reason=cast(str | None, payload.get("finish_reason")),
            usage=usage,
            retry_count=retry_count,
            tool_arguments=cast(str | None, payload.get("tool_arguments")),
            response_id=cast(str | None, payload.get("response_id")),
        )
        RecordedModelInvoker._validate_model_identity(binding, synthetic)
        return synthetic


def _cached_usage(payload: Mapping[str, Any]) -> CompletionUsage:
    """Decode provider usage retained in an exact-cache payload.

    Older cache rows may predate usage persistence; those rows fall back to
    the existing cost-ledger record when one is available, otherwise to zero.
    """

    raw_usage = payload.get("usage")
    if raw_usage is None:
        return CompletionUsage()
    if not isinstance(raw_usage, Mapping):
        raise RuntimeError("cached completion usage must be an object")
    try:
        return CompletionUsage(
            input_tokens=int(raw_usage.get("input_tokens", 0)),
            output_tokens=int(raw_usage.get("output_tokens", 0)),
            cached_input_tokens=int(raw_usage.get("cached_input_tokens", 0)),
            reasoning_tokens=int(raw_usage.get("reasoning_tokens", 0)),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cached completion usage is invalid") from exc


def _cached_retry_count(payload: Mapping[str, Any]) -> int:
    raw_retry_count = payload.get("retry_count", 0)
    try:
        retry_count = int(raw_retry_count)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cached completion retry_count is invalid") from exc
    if retry_count < 0:
        raise RuntimeError("cached completion retry_count cannot be negative")
    return retry_count


def _cached_cost(payload: Mapping[str, Any]) -> Decimal | None:
    """Decode the Decimal cost retained in a journal/cache payload."""

    raw_cost = payload.get("estimated_cost_cny")
    if raw_cost is None:
        return None
    try:
        cost = Decimal(str(raw_cost))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError("cached completion estimated cost is invalid") from exc
    if cost < 0:
        raise RuntimeError("cached completion estimated cost cannot be negative")
    return cost
