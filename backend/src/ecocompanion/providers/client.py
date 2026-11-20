"""Async OpenAI-compatible client with reproducible research safeguards."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ecocompanion.providers.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelIdentityMismatchError,
    ModelNotFoundError,
    ProtocolError,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
    StructuredOutputError,
    UnsupportedCapabilityError,
)
from ecocompanion.providers.types import (
    ChatCompletion,
    CompletionUsage,
    ModelDescriptor,
    StructuredAttempt,
    StructuredCompletion,
    StructuredMode,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
Sleep = Callable[[float], Awaitable[None]]
RandomFloat = Callable[[], float]

_PROTECTED_BODY_FIELDS = {"model", "messages", "response_format", "tools", "tool_choice"}
_UNSUPPORTED_MARKERS = (
    "not support",
    "unsupported",
    "doesn't support",
    "does not support",
    "unknown parameter",
    "unavailable",
    "not available",
)
_CAPABILITY_MARKERS = (
    "response_format",
    "json_schema",
    "json object",
    "tool_choice",
    "tools",
    "function call",
)


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible client.

    The client deliberately has no cross-model fallback argument. A research
    request is pinned to one model; only transient failures of that same request
    are retried.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 120.0,
        max_same_model_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
        sleeper: Sleep = asyncio.sleep,
        random_float: RandomFloat = random.random,
        allowed_response_model_ids: Mapping[str, frozenset[str]] | None = None,
        default_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        if not normalized_url.endswith("/v1"):
            normalized_url = f"{normalized_url}/v1"
        if max_same_model_retries < 0:
            raise ValueError("max_same_model_retries cannot be negative")
        if backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds cannot be negative")

        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._base_url = normalized_url
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._max_same_model_retries = max_same_model_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._sleeper = sleeper
        self._random_float = random_float
        self._allowed_response_model_ids = dict(allowed_response_model_ids or {})
        if default_extra_body:
            overlap = _PROTECTED_BODY_FIELDS & set(default_extra_body)
            if overlap:
                raise ValueError(
                    f"default_extra_body cannot override protected fields: {sorted(overlap)}"
                )
        self._default_extra_body = dict(default_extra_body or {})
        self._http = httpx.AsyncClient(
            base_url=f"{normalized_url}/",
            timeout=timeout,
            transport=transport,
            # Provider routing is explicitly controlled by LITELLM_BASE_URL.
            # Ignoring ambient proxy variables keeps local Web/API smoke tests
            # deterministic and avoids incompatible SOCKS proxy schemes.
            trust_env=False,
        )

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def discover_models(self) -> tuple[tuple[ModelDescriptor, ...], bool]:
        """Discover models, trying a public request before authenticated access.

        Returns the normalized descriptors and whether authenticated discovery
        was required.
        """

        try:
            payload, _ = await self._request_json("GET", "models", use_auth=False)
            authenticated = False
        except AuthenticationError:
            if self._api_key is None:
                raise
            payload, _ = await self._request_json("GET", "models", use_auth=True)
            authenticated = True

        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise ProtocolError("model discovery response must contain a data list")

        descriptors: dict[str, ModelDescriptor] = {}
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            owner = item.get("owned_by")
            created = item.get("created")
            descriptors[model_id] = ModelDescriptor(
                model_id=model_id,
                declared_owner=owner if isinstance(owner, str) else None,
                created=created if isinstance(created, int) else None,
            )

        if not descriptors:
            raise ProtocolError("model discovery returned no valid model ids")
        return tuple(descriptors[key] for key in sorted(descriptors)), authenticated

    async def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        temperature: float | None = 0.0,
        response_format: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Mapping[str, Any] | str | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletion:
        """Run one model-pinned chat completion."""

        if not model.strip():
            raise ValueError("model cannot be empty")
        if not messages:
            raise ValueError("messages cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        body: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if response_format is not None:
            body["response_format"] = dict(response_format)
        if tools is not None:
            body["tools"] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        merged_extra_body = dict(self._default_extra_body)
        if extra_body:
            overlap = _PROTECTED_BODY_FIELDS & set(extra_body)
            if overlap:
                raise ValueError(f"extra_body cannot override protected fields: {sorted(overlap)}")
            merged_extra_body.update(extra_body)
        if merged_extra_body:
            body.update(merged_extra_body)

        payload, retry_count = await self._request_json(
            "POST",
            "chat/completions",
            json_body=body,
            use_auth=True,
        )
        return self._parse_chat_completion(
            payload,
            requested_model=model,
            retry_count=retry_count,
        )

    async def stream_chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        temperature: float | None = 0.0,
    ) -> ChatCompletion:
        """Consume an OpenAI-compatible SSE stream and aggregate its text."""

        if not model.strip():
            raise ValueError("model cannot be empty")
        if not messages:
            raise ValueError("messages cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        body: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            body["temperature"] = temperature
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for retry_index in range(self._max_same_model_retries + 1):
            try:
                async with self._http.stream(
                    "POST",
                    "chat/completions",
                    json=body,
                    headers=headers,
                ) as response:
                    if response.is_success:
                        return await self._parse_stream_completion(
                            response,
                            requested_model=model,
                            retry_count=retry_index,
                        )
                    await response.aread()
                    error = self._classify_http_error(response)
            except (httpx.TimeoutException, httpx.TransportError):
                error = RetryableProviderError(
                    "provider stream failed after same-request retries"
                )

            if error.retryable and retry_index < self._max_same_model_retries:
                await self._wait_before_retry(retry_index)
                continue
            raise error

        raise AssertionError("stream retry loop exited unexpectedly")

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
            StructuredMode.TOOL_CALL,
            StructuredMode.JSON_OBJECT,
            StructuredMode.PLAIN_JSON,
        ),
    ) -> StructuredCompletion[ModelT]:
        """Return validated structured output through an explicit fallback chain."""

        if not modes:
            raise ValueError("at least one structured mode is required")

        attempts: list[StructuredAttempt] = []
        last_error: ProviderError | ValidationError | json.JSONDecodeError | None = None
        schema_payload = schema.model_json_schema()
        tool_name = f"emit_{schema.__name__.lower()}"

        for mode in modes:
            serialized: str | None = None
            try:
                completion = await self._completion_for_structured_mode(
                    mode=mode,
                    model=model,
                    messages=messages,
                    schema_payload=schema_payload,
                    schema_name=schema.__name__,
                    tool_name=tool_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                serialized = (
                    completion.tool_arguments
                    if mode is StructuredMode.TOOL_CALL
                    else completion.content
                )
                if not serialized:
                    raise StructuredOutputError(
                        f"{mode.value} returned no structured content"
                    )
                value = schema.model_validate_json(serialized)
                attempts.append(StructuredAttempt(mode=mode, succeeded=True))
                return StructuredCompletion(
                    value=value,
                    completion=completion,
                    mode=mode,
                    attempts=tuple(attempts),
                )
            except UnsupportedCapabilityError as exc:
                last_error = exc
                attempts.append(
                    StructuredAttempt(
                        mode=mode,
                        succeeded=False,
                        failure_category=exc.category,
                        failure_message=self._sanitize(str(exc)),
                    )
                )
                continue
            except (
                StructuredOutputError,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                attempts.append(
                    StructuredAttempt(
                        mode=mode,
                        succeeded=False,
                        failure_category=(
                            exc.category
                            if isinstance(exc, ProviderError)
                            else "schema_validation"
                        ),
                        failure_message=self._sanitize(str(exc)),
                    )
                )
                repaired = await self._repair_structured_output(
                    model=model,
                    messages=messages,
                    malformed=serialized or "",
                    schema=schema,
                    schema_payload=schema_payload,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                try:
                    repaired_value = schema.model_validate_json(
                        repaired.content or ""
                    )
                except ValidationError as repair_error:
                    attempts.append(
                        StructuredAttempt(
                            mode=mode,
                            succeeded=False,
                            repair=True,
                            failure_category="schema_validation",
                            failure_message=self._sanitize(str(repair_error)),
                        )
                    )
                    last_error = repair_error
                    break
                attempts.append(
                    StructuredAttempt(
                        mode=mode,
                        succeeded=True,
                        repair=True,
                    )
                )
                return StructuredCompletion(
                    value=repaired_value,
                    completion=repaired,
                    mode=mode,
                    attempts=tuple(attempts),
                )

        summary = "; ".join(
            f"{attempt.mode.value}:{attempt.failure_category}" for attempt in attempts
        )
        message = f"all structured-output modes failed ({summary})"
        if last_error is not None:
            message = f"{message}: {self._sanitize(str(last_error))}"
        raise StructuredOutputError(message)

    async def _repair_structured_output(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        malformed: str,
        schema: type[ModelT],
        schema_payload: Mapping[str, Any],
        max_tokens: int,
        temperature: float | None,
    ) -> ChatCompletion:
        repair_messages = (
            *(dict(message) for message in messages),
            {
                "role": "assistant",
                "content": malformed[:8_000],
            },
            {
                "role": "user",
                "content": (
                    "上一个输出未通过格式校验。只允许这一次修复："
                    f"请只返回符合 {schema.__name__} JSON Schema 的对象，"
                    "不要添加 Markdown 或解释。\n"
                    f"{json.dumps(schema_payload, ensure_ascii=False, sort_keys=True)}"
                ),
            },
        )
        return await self.chat_completion(
            model=model,
            messages=repair_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _completion_for_structured_mode(
        self,
        *,
        mode: StructuredMode,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        schema_payload: dict[str, Any],
        schema_name: str,
        tool_name: str,
        max_tokens: int,
        temperature: float | None,
    ) -> ChatCompletion:
        if mode is StructuredMode.JSON_SCHEMA:
            return await self.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema_payload,
                    },
                },
            )
        if mode is StructuredMode.TOOL_CALL:
            return await self.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": "Return the requested structured result.",
                            "strict": True,
                            "parameters": schema_payload,
                        },
                    },
                ),
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )

        instructed_messages = self._add_json_instruction(messages, schema_payload)
        if mode is StructuredMode.JSON_OBJECT:
            return await self.chat_completion(
                model=model,
                messages=instructed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        if mode is StructuredMode.PLAIN_JSON:
            return await self.chat_completion(
                model=model,
                messages=instructed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise ValueError(f"unknown structured mode: {mode}")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        use_auth: bool,
    ) -> tuple[dict[str, Any], int]:
        headers = {"Accept": "application/json"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        if use_auth and self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for retry_index in range(self._max_same_model_retries + 1):
            try:
                response = await self._http.request(
                    method,
                    path,
                    json=json_body,
                    headers=headers,
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if retry_index < self._max_same_model_retries:
                    await self._wait_before_retry(retry_index)
                    continue
                raise RetryableProviderError(
                    "provider transport failed after same-request retries"
                ) from None

            if response.is_success:
                try:
                    payload = response.json()
                except ValueError:
                    raise ProtocolError("provider returned non-JSON success response") from None
                if not isinstance(payload, dict):
                    raise ProtocolError("provider JSON response must be an object")
                return payload, retry_index

            error = self._classify_http_error(response)
            if error.retryable and retry_index < self._max_same_model_retries:
                await self._wait_before_retry(retry_index)
                continue
            raise error

        raise AssertionError("retry loop exited unexpectedly")

    def _parse_chat_completion(
        self,
        payload: Mapping[str, Any],
        *,
        requested_model: str,
        retry_count: int,
    ) -> ChatCompletion:
        response_model_raw = payload.get("model")
        response_model = (
            response_model_raw if isinstance(response_model_raw, str) else requested_model
        )
        self._validate_response_model(
            requested_model=requested_model,
            response_model=response_model,
        )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProtocolError("chat completion response must contain at least one choice")
        first_choice = choices[0]
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ProtocolError("chat completion choice must contain a message object")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProtocolError("chat completion message content must be a string or null")
        tool_arguments = self._extract_tool_arguments(message)
        if content is None and tool_arguments is None:
            raise ProtocolError("chat completion returned neither content nor tool arguments")

        finish_reason = first_choice.get("finish_reason")
        response_id = payload.get("id")
        return ChatCompletion(
            requested_model=requested_model,
            response_model=response_model,
            content=content,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=self._parse_usage(payload.get("usage")),
            retry_count=retry_count,
            tool_arguments=tool_arguments,
            response_id=response_id if isinstance(response_id, str) else None,
        )

    async def _parse_stream_completion(
        self,
        response: httpx.Response,
        *,
        requested_model: str,
        retry_count: int,
    ) -> ChatCompletion:
        chunks: list[str] = []
        response_model = requested_model
        response_id: str | None = None
        finish_reason: str | None = None
        usage = CompletionUsage()
        saw_done = False

        async for line in response.aiter_lines():
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            if not stripped.startswith("data:"):
                continue
            data = stripped.removeprefix("data:").strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProtocolError(
                    "provider stream contained invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise ProtocolError(
                    "provider stream event must be a JSON object"
                )
            model_value = payload.get("model")
            if isinstance(model_value, str):
                self._validate_response_model(
                    requested_model=requested_model,
                    response_model=model_value,
                )
                response_model = model_value
            identifier = payload.get("id")
            if response_id is None and isinstance(identifier, str):
                response_id = identifier
            if isinstance(payload.get("usage"), dict):
                usage = self._parse_usage(payload["usage"])
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ProtocolError(
                    "provider stream choice must be an object"
                )
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise ProtocolError(
                        "provider stream delta content must be a string or null"
                    )
                if content:
                    chunks.append(content)
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                finish_reason = reason

        if not saw_done:
            raise ProtocolError("provider stream ended without [DONE]")
        content = "".join(chunks)
        if not content:
            raise ProtocolError("provider stream returned no text content")
        self._validate_response_model(
            requested_model=requested_model,
            response_model=response_model,
        )
        return ChatCompletion(
            requested_model=requested_model,
            response_model=response_model,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            retry_count=retry_count,
            response_id=response_id,
        )

    def _validate_response_model(
        self,
        *,
        requested_model: str,
        response_model: str,
    ) -> None:
        allowed_ids = {requested_model}
        allowed_ids.update(
            self._allowed_response_model_ids.get(requested_model, frozenset())
        )
        if response_model not in allowed_ids:
            raise ModelIdentityMismatchError(
                "provider returned a model id outside the explicit alias allow-list"
            )

    @staticmethod
    def _parse_usage(raw_usage: Any) -> CompletionUsage:
        if not isinstance(raw_usage, dict):
            return CompletionUsage()
        input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
        output_tokens = raw_usage.get(
            "completion_tokens",
            raw_usage.get("output_tokens", 0),
        )
        details = raw_usage.get("prompt_tokens_details")
        cached_input_tokens = (
            details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        )
        completion_details = raw_usage.get("completion_tokens_details")
        reasoning_tokens = (
            completion_details.get("reasoning_tokens", 0)
            if isinstance(completion_details, dict)
            else 0
        )
        return CompletionUsage(
            input_tokens=input_tokens if isinstance(input_tokens, int) else 0,
            output_tokens=output_tokens if isinstance(output_tokens, int) else 0,
            cached_input_tokens=(
                cached_input_tokens if isinstance(cached_input_tokens, int) else 0
            ),
            reasoning_tokens=reasoning_tokens if isinstance(reasoning_tokens, int) else 0,
        )

    @staticmethod
    def _extract_tool_arguments(message: Mapping[str, Any]) -> str | None:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return None
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                return arguments
        return None

    def _classify_http_error(self, response: httpx.Response) -> ProviderError:
        status_code = response.status_code
        message = self._error_message(response)
        lowered = message.lower()
        if status_code in {401, 403}:
            return AuthenticationError(message, status_code=status_code)
        if status_code == 404:
            return ModelNotFoundError(message, status_code=status_code)
        if status_code == 429:
            return RateLimitError(message, status_code=status_code)
        if status_code == 408 or 500 <= status_code <= 599:
            return RetryableProviderError(message, status_code=status_code)
        if status_code in {400, 422}:
            unsupported = any(marker in lowered for marker in _UNSUPPORTED_MARKERS)
            capability = any(marker in lowered for marker in _CAPABILITY_MARKERS)
            if unsupported and capability:
                return UnsupportedCapabilityError(message, status_code=status_code)
            return InvalidRequestError(message, status_code=status_code)
        return ProviderError(message, status_code=status_code)

    def _error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return self._sanitize(
                f"provider returned HTTP {response.status_code}: {response.text[:500]}"
            )
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    return self._sanitize(message)
            if isinstance(error, str):
                return self._sanitize(error)
        return f"provider returned HTTP {response.status_code}"

    async def _wait_before_retry(self, retry_index: int) -> None:
        exponential = self._backoff_base_seconds * (2**retry_index)
        jitter = exponential * self._random_float()
        await self._sleeper(exponential + jitter)

    def _sanitize(self, value: str) -> str:
        sanitized = value.replace("\n", " ").strip()
        if self._api_key:
            sanitized = sanitized.replace(self._api_key, "[REDACTED]")
        return sanitized[:500]

    @staticmethod
    def _add_json_instruction(
        messages: Sequence[Mapping[str, Any]],
        schema_payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        instruction = (
            "Return only one valid JSON object matching this JSON Schema. "
            "Do not use Markdown fences or add commentary.\n"
            f"{json.dumps(schema_payload, ensure_ascii=False, sort_keys=True)}"
        )
        return (
            {"role": "system", "content": instruction},
            *(dict(message) for message in messages),
        )
