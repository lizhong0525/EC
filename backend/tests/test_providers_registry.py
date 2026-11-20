import json

import httpx
import pytest

from ecocompanion.providers import (
    CapabilityStatus,
    ModelCapabilityReport,
    ModelDescriptor,
    ModelQualificationRunner,
    ModelRegistry,
    ModelRegistrySnapshot,
    ModelRole,
    OpenAICompatibleClient,
    ProbeExecution,
    QualifiedModelSnapshot,
    RoleBinding,
    SafetyQualificationProbe,
    StructuredMode,
    recommend_model_roles,
    validate_role_separation,
)


@pytest.mark.asyncio
async def test_registry_discovers_and_smoke_tests_models() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "model-a", "owned_by": "family-a"}]},
            )

        body = json.loads(request.content)
        if body.get("stream") is True:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=(
                    'data: {"model":"model-a","choices":'
                    '[{"delta":{"content":"流式可用"},"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        content = (
            '{"status":"ok"}'
            if body.get("response_format", {}).get("type") == "json_schema"
            else "可用"
        )
        return httpx.Response(
            200,
            json={
                "model": "model-a",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        snapshot = await ModelRegistry(client).refresh()

    assert snapshot.authenticated_discovery is False
    assert snapshot.models[0].model_id == "model-a"
    report = snapshot.capabilities[0]
    assert report.status is CapabilityStatus.READY
    assert report.streaming is True
    assert report.preferred_structured_mode is StructuredMode.JSON_SCHEMA
    assert report.eligible_roles == frozenset(ModelRole)
    serialized = snapshot.to_dict()
    assert serialized["models"][0]["model_id"] == "model-a"
    assert '"authenticated_discovery":false' in snapshot.to_json()


@pytest.mark.asyncio
async def test_capability_probe_rejects_non_chinese_instruction_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "model-a"}]})
        return httpx.Response(
            200,
            json={
                "model": "model-a",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "available",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    async with OpenAICompatibleClient(
        base_url="https://relay.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        snapshot = await ModelRegistry(client).refresh()

    report = snapshot.capabilities[0]
    assert report.status is CapabilityStatus.INELIGIBLE
    assert report.basic_chat is True
    assert report.chinese_text is False
    assert report.system_and_multiturn is False
    assert "chinese_text:missing_cjk" in report.failures


def test_role_separation_requires_distinct_known_families() -> None:
    report = validate_role_separation(
        evaluated_agent=RoleBinding(
            role=ModelRole.EVALUATED_AGENT,
            model_id="agent-a",
            family="family-a",
        ),
        student_simulator=RoleBinding(
            role=ModelRole.STUDENT_SIMULATOR,
            model_id="student-b",
            family="family-b",
        ),
        judges=(
            RoleBinding(
                role=ModelRole.JUDGE,
                model_id="judge-c",
                family="family-c",
            ),
            RoleBinding(
                role=ModelRole.JUDGE,
                model_id="judge-d",
                family="family-d",
            ),
        ),
    )
    assert report.valid is True

    invalid = validate_role_separation(
        evaluated_agent=RoleBinding(
            role=ModelRole.EVALUATED_AGENT,
            model_id="agent-a",
            family="family-a",
        ),
        student_simulator=RoleBinding(
            role=ModelRole.STUDENT_SIMULATOR,
            model_id="student-a",
            family="family-a",
        ),
        judges=(
            RoleBinding(
                role=ModelRole.JUDGE,
                model_id="judge-a",
                family="family-a",
            ),
        ),
    )
    assert invalid.valid is False
    assert any("at least two judges" in item for item in invalid.violations)
    assert any("student simulator shares" in item for item in invalid.violations)


class _QualificationExecutor:
    async def execute(
        self,
        *,
        model_id: str,
        probe: SafetyQualificationProbe,
    ) -> ProbeExecution:
        return ProbeExecution(
            raw_schema_valid=True,
            passed_criteria=probe.required_criteria,
            persona_consistent=(
                True if probe.score_persona_consistency else None
            ),
            boundary_refusal_stable=(
                True if probe.score_boundary_refusal else None
            ),
            latency_ms=float(ord(model_id[-1])) * 10.0,
            cost_cny=float(ord(model_id[-1])) / 100_000.0,
        )


@pytest.mark.asyncio
async def test_role_recommendation_prefers_score_and_separated_families() -> None:
    model_ids = tuple(f"model-{suffix}" for suffix in "abcdef")
    families = tuple(f"family-{suffix}" for suffix in "abcdef")
    models = tuple(
        ModelDescriptor(model_id=model_id, declared_owner=family)
        for model_id, family in zip(model_ids, families, strict=True)
    )
    capabilities = tuple(
        ModelCapabilityReport(
            model_id=model_id,
            status=CapabilityStatus.READY,
            basic_chat=True,
            system_and_multiturn=True,
            chinese_text=True,
            usage_reported=True,
            preferred_structured_mode=StructuredMode.JSON_SCHEMA,
        )
        for model_id in model_ids
    )
    runner = ModelQualificationRunner(_QualificationExecutor())
    qualifications = tuple(
        [await runner.qualify(model_id) for model_id in model_ids]
    )
    snapshot = QualifiedModelSnapshot(
        registry=ModelRegistrySnapshot(
            models=models,
            capabilities=capabilities,
            authenticated_discovery=True,
        ),
        qualifications=qualifications,
    )

    recommendation = recommend_model_roles(snapshot)

    assert recommendation.companion_model == "model-a"
    assert len(recommendation.comparison_models) == 4
    assert recommendation.student_renderer_model is not None
    assert len(recommendation.judge_models) == 2
    assigned = {
        recommendation.companion_model,
        recommendation.student_renderer_model,
        *recommendation.judge_models,
    }
    assert len(assigned) == 4
    assert recommendation.distinct_family_requirement_met
    assert not recommendation.warnings
