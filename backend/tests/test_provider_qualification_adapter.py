from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from ecocompanion.providers import (
    QUALIFICATION_SUITE_V1,
    ChatCompletion,
    CompletionUsage,
    FixedTokenPriceEstimator,
    OpenAIQualificationExecutor,
    ProbeExecution,
    QualificationJudgment,
    SafetyQualificationProbe,
)


class FakeChatClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[tuple[str, Sequence[Mapping[str, Any]]]] = []

    async def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        temperature: float | None = 0.0,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatCompletion:
        del max_tokens, temperature, response_format
        self.requests.append((model, messages))
        return ChatCompletion(
            requested_model=model,
            response_model=model,
            content=self.content,
            finish_reason="stop",
            usage=CompletionUsage(input_tokens=100, output_tokens=50),
            retry_count=0,
        )


class PassingEvaluator:
    def evaluate(
        self,
        probe: SafetyQualificationProbe,
        response_text: str,
    ) -> QualificationJudgment:
        assert response_text
        return QualificationJudgment(
            passed_criteria=probe.required_criteria,
            persona_consistent=(True if probe.score_persona_consistency else None),
            boundary_refusal_stable=(True if probe.score_boundary_refusal else None),
        )


@pytest.mark.asyncio
async def test_openai_adapter_uses_fake_client_without_network() -> None:
    client = FakeChatClient('{"response_text":"谢谢你告诉我，我在听。"}')
    executor = OpenAIQualificationExecutor(
        client=client,
        evaluator=PassingEvaluator(),
        cost_estimator=FixedTokenPriceEstimator(
            input_cny_per_million=2.0,
            output_cny_per_million=4.0,
        ),
    )

    result = await executor.execute(
        model_id="fixture-model",
        probe=QUALIFICATION_SUITE_V1[0],
    )

    assert isinstance(result, ProbeExecution)
    assert result.raw_schema_valid is True
    assert result.cost_cny == pytest.approx(0.0004)
    assert client.requests[0][0] == "fixture-model"
    assert client.requests[0][1][0]["role"] == "system"


@pytest.mark.asyncio
async def test_openai_adapter_records_raw_schema_failure_before_repair() -> None:
    client = FakeChatClient("谢谢你告诉我，我在听。")
    executor = OpenAIQualificationExecutor(
        client=client,
        evaluator=PassingEvaluator(),
    )

    result = await executor.execute(
        model_id="fixture-model",
        probe=QUALIFICATION_SUITE_V1[0],
    )

    assert result.raw_schema_valid is False
    assert result.cost_cny is None
