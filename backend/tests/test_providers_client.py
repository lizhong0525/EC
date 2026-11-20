import json
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from ecocompanion.providers import (
    InvalidRequestError,
    ModelIdentityMismatchError,
    OpenAICompatibleClient,
    StructuredMode,
    StructuredOutputError,
)


class ProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


def chat_response(
    *,
    model: str,
    content: str | None = "ok",
    tool_arguments: str | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_arguments is not None:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "emit_probepayload",
                    "arguments": tool_arguments,
                },
            }
        ]
    return {
        "id": "completion-1",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


@pytest.mark.asyncio
async def test_discovers_public_models_without_sending_authorization() -> None:
    seen_authorization: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "model-b", "owned_by": "family-b"},
                    {"id": "model-a", "owned_by": "family-a"},
                    {"id": "model-a", "owned_by": "duplicate"},
                ]
            },
        )

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        api_key="test-token",
        transport=httpx.MockTransport(handler),
    ) as client:
        models, authenticated = await client.discover_models()

    assert authenticated is False
    assert [model.model_id for model in models] == ["model-a", "model-b"]
    assert seen_authorization == [None]


@pytest.mark.asyncio
async def test_discovery_retries_with_authentication_only_after_401() -> None:
    seen_authorization: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers.get("authorization")
        seen_authorization.append(authorization)
        if authorization is None:
            return httpx.Response(
                401,
                json={"error": {"message": "authentication required"}},
            )
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    async with OpenAICompatibleClient(
        base_url="https://relay.example",
        api_key="test-secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        models, authenticated = await client.discover_models()

    assert authenticated is True
    assert models[0].model_id == "model-a"
    assert seen_authorization == [None, "Bearer test-secret"]


@pytest.mark.asyncio
async def test_retries_rate_limit_on_the_same_model() -> None:
    calls: list[str] = []
    waits: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(str(body["model"]))
        if len(calls) < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=chat_response(model="model-a"))

    async def fake_sleep(delay: float) -> None:
        waits.append(delay)

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
        max_same_model_retries=2,
        backoff_base_seconds=1.0,
        sleeper=fake_sleep,
        random_float=lambda: 0.0,
    ) as client:
        result = await client.chat_completion(
            model="model-a",
            messages=({"role": "user", "content": "hello"},),
        )

    assert calls == ["model-a", "model-a", "model-a"]
    assert waits == [1.0, 2.0]
    assert result.retry_count == 2
    assert result.usage.cached_input_tokens == 2


@pytest.mark.asyncio
async def test_invalid_request_fails_without_retry() -> None:
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
        max_same_model_retries=3,
        backoff_base_seconds=0.0,
    ) as client:
        with pytest.raises(InvalidRequestError):
            await client.chat_completion(
                model="model-a",
                messages=({"role": "user", "content": "hello"},),
            )

    assert call_count == 1


@pytest.mark.asyncio
async def test_structured_output_falls_back_to_tool_call() -> None:
    modes_seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("response_format", {}).get("type") == "json_schema":
            modes_seen.append("json_schema")
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "response_format json_schema is not supported"
                    }
                },
            )
        if "tools" in body:
            modes_seen.append("tool_call")
            return httpx.Response(
                200,
                json=chat_response(
                    model="model-a",
                    content=None,
                    tool_arguments='{"status":"ok"}',
                ),
            )
        raise AssertionError("unexpected structured-output mode")

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete_structured(
            model="model-a",
            messages=({"role": "user", "content": "return status"},),
            schema=ProbePayload,
        )

    assert result.value.status == "ok"
    assert result.mode is StructuredMode.TOOL_CALL
    assert modes_seen == ["json_schema", "tool_call"]
    assert [attempt.succeeded for attempt in result.attempts] == [False, True]


@pytest.mark.asyncio
async def test_stream_completion_aggregates_sse_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'data: {"id":"stream-1","model":"model-a","choices":'
                '[{"delta":{"content":"流式"},"finish_reason":null}]}\n\n'
                'data: {"id":"stream-1","model":"model-a","choices":'
                '[{"delta":{"content":"可用"},"finish_reason":"stop"}]}\n\n'
                'data: {"model":"model-a","choices":[],"usage":'
                '{"prompt_tokens":7,"completion_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        completion = await client.stream_chat_completion(
            model="model-a",
            messages=({"role": "user", "content": "测试"},),
        )

    assert completion.content == "流式可用"
    assert completion.finish_reason == "stop"
    assert completion.response_id == "stream-1"
    assert completion.usage.input_tokens == 7
    assert completion.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_structured_output_allows_exactly_one_schema_repair() -> None:
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        content = '{"status":"wrong"}' if call_count == 1 else '{"status":"ok"}'
        return httpx.Response(
            200,
            json=chat_response(model="model-a", content=content),
        )

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.complete_structured(
            model="model-a",
            messages=({"role": "user", "content": "return status"},),
            schema=ProbePayload,
        )

    assert result.value.status == "ok"
    assert call_count == 2
    assert [(attempt.succeeded, attempt.repair) for attempt in result.attempts] == [
        (False, False),
        (True, True),
    ]


@pytest.mark.asyncio
async def test_structured_output_does_not_retry_modes_after_failed_repair() -> None:
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json=chat_response(model="model-a", content='{"status":"wrong"}'),
        )

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(StructuredOutputError):
            await client.complete_structured(
                model="model-a",
                messages=({"role": "user", "content": "return status"},),
                schema=ProbePayload,
            )

    assert call_count == 2


@pytest.mark.asyncio
async def test_rejects_silent_cross_model_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(model="fallback-model"))

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ModelIdentityMismatchError):
            await client.chat_completion(
                model="evaluated-model",
                messages=({"role": "user", "content": "hello"},),
            )

@pytest.mark.asyncio
async def test_default_extra_body_is_sent_with_every_completion() -> None:
    bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response(model="model-a"))

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        api_key="test-token",
        transport=httpx.MockTransport(handler),
        default_extra_body={"thinking": {"type": "disabled"}},
    ) as client:
        await client.chat_completion(
            model="model-a",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
        )

    assert bodies[0]["thinking"] == {"type": "disabled"}


def test_default_extra_body_cannot_override_protected_fields() -> None:
    with pytest.raises(ValueError, match="protected"):
        OpenAICompatibleClient(
            base_url="https://relay.example/v1",
            default_extra_body={"model": "other"},
        )
