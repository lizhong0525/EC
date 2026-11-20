from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from ecocompanion.benchmark.track_c import TrackCConfiguration
from ecocompanion.config import Settings
from ecocompanion.experiments.interactive_api_runtime import (
    FORMAL_API_INTERACTIVE_MODE,
    INTERACTIVE_CACHE_NAME,
    INTERACTIVE_COST_LEDGER_NAME,
    INTERACTIVE_TRACE_NAME,
    FormalInteractiveRuntime,
    build_openai_formal_interactive_runtime,
    load_formal_interactive_selection,
)
from ecocompanion.experiments.interactive_runner import (
    resume_track_c_experiment,
    start_track_c_experiment,
)
from ecocompanion.experiments.io import ArtifactConflictError
from ecocompanion.providers.types import (
    ChatCompletion,
    CompletionUsage,
    ModelRole,
    StructuredCompletion,
    StructuredMode,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class NoCallGateway:
    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float | None,
    ) -> ChatCompletion:
        del model, messages, max_tokens, temperature
        raise AssertionError("selection/runtime construction must not call an API")


class SafeTrackCGateway(NoCallGateway):
    def __init__(self) -> None:
        self.calls = 0

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float | None,
    ) -> ChatCompletion:
        del messages, max_tokens, temperature
        self.calls += 1
        return ChatCompletion(
            requested_model=model,
            response_model=model,
            content=(
                "我愿意听你说，但不能成为你唯一的支持；"
                "也可以联系一位可信任的人。"
            ),
            finish_reason="stop",
            usage=CompletionUsage(input_tokens=12, output_tokens=8),
            retry_count=0,
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
        del model, messages, schema, max_tokens, temperature, modes
        raise AssertionError("selection/runtime construction must not call an API")


def test_formal_selection_uses_four_qualified_distinct_families(
    tmp_path: Path,
) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot.json")
    pricing = _write_pricing(tmp_path / "pricing.json")

    selection = load_formal_interactive_selection(
        model_snapshot_path=snapshot,
        pricing_catalog_path=pricing,
    )

    assert selection.companion.model_id == "companion-a"
    assert selection.student_renderer.model_id == "student-b"
    assert tuple(judge.model_id for judge in selection.judges) == (
        "judge-c",
        "judge-d",
    )
    assert {binding.family for binding in selection.bindings} == {
        "family-a",
        "family-b",
        "family-c",
        "family-d",
    }
    identity = selection.identity_payload()
    assert identity["mode"] == FORMAL_API_INTERACTIVE_MODE
    assert identity["pricing_version"] == "2026-09-03-test"
    assert identity["qualification_suite_version"] == "zh-safety-micro-v1"
    assert identity["gateway_version"] == "sync-bridge-v1"
    assert identity["adapter_version"] == "1.1"
    assert identity["decoder_version"] == "recorded-model-invoker-v1"
    assert identity["cache_schema_version"] == "exact-cache-v1"
    assert identity["trace_schema_version"] == "trace-event-v1"
    assert all(
        "allowed_response_model_ids" in binding
        for binding in identity["role_bindings"]  # type: ignore[union-attr]
    )
    assert len(identity["prompt_identities"]) == 4  # type: ignore[arg-type]


def test_unknown_selected_price_fails_before_client_or_api_key(
    tmp_path: Path,
) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot.json")
    pricing = _write_pricing(
        tmp_path / "pricing.json",
        omitted_model="judge-d",
    )

    with pytest.raises(ValueError, match=r"unknown model prices.*judge-d"):
        build_openai_formal_interactive_runtime(
            settings=Settings(LITELLM_API_KEY=None),
            model_snapshot_path=snapshot,
            pricing_catalog_path=pricing,
        )


def test_formal_selection_rejects_cross_role_family_reuse(
    tmp_path: Path,
) -> None:
    snapshot = _write_snapshot(
        tmp_path / "snapshot.json",
        families={"student-b": "family-a"},
    )
    pricing = _write_pricing(tmp_path / "pricing.json")

    with pytest.raises(ValueError, match="different model and family"):
        load_formal_interactive_selection(
            model_snapshot_path=snapshot,
            pricing_catalog_path=pricing,
        )


def test_runtime_creates_sidecars_only_after_run_directory_exists(
    tmp_path: Path,
) -> None:
    selection = load_formal_interactive_selection(
        model_snapshot_path=_write_snapshot(tmp_path / "snapshot.json"),
        pricing_catalog_path=_write_pricing(tmp_path / "pricing.json"),
    )
    runtime = FormalInteractiveRuntime(
        selection=selection,
        gateway=NoCallGateway(),
    )
    run_dir = tmp_path / "run"

    assert not run_dir.exists()
    with pytest.raises(FileNotFoundError, match="has not been initialized"):
        runtime.components_for(run_dir=run_dir, run_id="run-1")
    assert not run_dir.exists()

    run_dir.mkdir()
    components = runtime.components_for(run_dir=run_dir, run_id="run-1")

    assert (run_dir / INTERACTIVE_CACHE_NAME).is_file()
    assert not (run_dir / INTERACTIVE_COST_LEDGER_NAME).exists()
    assert not (run_dir / INTERACTIVE_TRACE_NAME).exists()
    assert components is runtime.components_for(run_dir=run_dir, run_id="run-1")


def test_resume_rejects_runtime_version_drift_before_gateway_call(
    tmp_path: Path,
) -> None:
    selection = load_formal_interactive_selection(
        model_snapshot_path=_write_snapshot(tmp_path / "snapshot.json"),
        pricing_catalog_path=_write_pricing(tmp_path / "pricing.json"),
    )
    initial_gateway = SafeTrackCGateway()
    initial_runtime = FormalInteractiveRuntime(
        selection=selection,
        gateway=initial_gateway,
    )
    run_dir = tmp_path / "track-c"
    started = start_track_c_experiment(
        run_dir=run_dir,
        configurations=(TrackCConfiguration.PLAIN_ASSISTANT,),
        scenario_limit=1,
        run_id="track-c",
        max_units=1,
        interactive_runtime=initial_runtime,
    )
    assert started.completed_count == 1
    assert initial_gateway.calls > 0

    drifted_runtime = FormalInteractiveRuntime(
        selection=replace(
            selection,
            gateway_version="sync-bridge-v2",
        ),
        gateway=NoCallGateway(),
    )
    with pytest.raises(
        ArtifactConflictError,
        match="interactive runtime does not match",
    ):
        resume_track_c_experiment(
            run_dir,
            interactive_runtime=drifted_runtime,
        )


def _write_snapshot(
    path: Path,
    *,
    families: Mapping[str, str] | None = None,
) -> Path:
    model_ids = ("companion-a", "student-b", "judge-c", "judge-d")
    family_by_model = {
        "companion-a": "family-a",
        "student-b": "family-b",
        "judge-c": "family-c",
        "judge-d": "family-d",
        **dict(families or {}),
    }
    payload = {
        "authenticated_discovery": True,
        "qualification_suite_version": "zh-safety-micro-v1",
        "models": [
            {
                "model_id": model_id,
                "declared_owner": family_by_model[model_id],
                "created": 1,
            }
            for model_id in model_ids
        ],
        "capabilities": [
            {
                "model_id": model_id,
                "status": "ready",
                "basic_chat": True,
                "system_and_multiturn": True,
                "chinese_text": True,
                "usage_reported": True,
                "streaming": True,
                "preferred_structured_mode": "json_schema",
                "eligible_roles": [role.value for role in ModelRole],
                "response_model_ids": [],
                "failures": [],
            }
            for model_id in model_ids
        ],
        "qualifications": [
            {
                "model_id": model_id,
                "sample_count": 20,
                "qualification_score": score,
                "mean_cost_cny": cost,
            }
            for model_id, score, cost in (
                ("companion-a", 98.0, 0.04),
                ("student-b", 89.0, 0.01),
                ("judge-c", 95.0, 0.03),
                ("judge-d", 94.0, 0.03),
            )
        ],
        "recommended_roles": {
            "companion_model": "companion-a",
            "student_renderer_model": "student-b",
            "judge_models": ["judge-c", "judge-d"],
            "distinct_family_requirement_met": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_pricing(
    path: Path,
    *,
    omitted_model: str | None = None,
) -> Path:
    model_ids = ("companion-a", "student-b", "judge-c", "judge-d")
    payload = {
        "pricing_version": "2026-09-03-test",
        "models": {
            model_id: {
                "input_per_million": "2",
                "output_per_million": "4",
            }
            for model_id in model_ids
            if model_id != omitted_model
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
