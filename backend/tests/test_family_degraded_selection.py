from __future__ import annotations

import json
from pathlib import Path

from ecocompanion.experiments.interactive_api_runtime import (
    load_formal_interactive_selection,
)
from ecocompanion.experiments.models import InteractiveRuntimeRecord
from ecocompanion.providers.types import ModelRole


def _write_artifacts(
    tmp_path: Path,
    *,
    families: dict[str, str | None],
) -> tuple[Path, Path]:
    model_ids = tuple(families)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "authenticated_discovery": True,
                "qualification_suite_version": "zh-safety-micro-v1",
                "models": [
                    {
                        "model_id": model_id,
                        "declared_owner": family,
                        "created": 1,
                    }
                    for model_id, family in families.items()
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
                        "qualification_score": 98.0 - index,
                        "mean_cost_cny": 0.01,
                    }
                    for index, model_id in enumerate(model_ids)
                ],
            }
        ),
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps(
            {
                "pricing_version": "family-degraded-test-v1",
                "models": {
                    model_id: {
                        "input_per_million": "2",
                        "output_per_million": "4",
                    }
                    for model_id in model_ids
                },
            }
        ),
        encoding="utf-8",
    )
    return snapshot, pricing


def test_formal_selection_degrades_with_two_families_but_keeps_four_ids(
    tmp_path: Path,
) -> None:
    snapshot, pricing = _write_artifacts(
        tmp_path,
        families={
            "companion-a": "family-a",
            "student-b": "family-a",
            "judge-c": "family-b",
            "judge-d": "family-b",
        },
    )

    selection = load_formal_interactive_selection(
        model_snapshot_path=snapshot,
        pricing_catalog_path=pricing,
    )

    assert len({binding.model_id for binding in selection.bindings}) == 4
    assert selection.family_separation_status == "degraded"
    assert selection.expert_evaluation_required is True
    assert selection.family_warnings
    assert any(
        "judge shares companion/student family" in warning
        for warning in selection.family_warnings
    )

    runtime = InteractiveRuntimeRecord.model_validate(selection.identity_payload())
    assert runtime.family_separation_status == "degraded"
    assert runtime.expert_evaluation_required is True
    assert runtime.family_warnings == selection.family_warnings


def test_formal_selection_accepts_unknown_family_as_audited_degradation(
    tmp_path: Path,
) -> None:
    snapshot, pricing = _write_artifacts(
        tmp_path,
        families={
            "companion-a": None,
            "student-b": "family-b",
            "judge-c": "family-c",
            "judge-d": "family-d",
        },
    )

    selection = load_formal_interactive_selection(
        model_snapshot_path=snapshot,
        pricing_catalog_path=pricing,
    )

    assert selection.family_separation_status == "degraded"
    assert selection.expert_evaluation_required is True
    assert "one or more model families are unknown" in selection.family_warnings
