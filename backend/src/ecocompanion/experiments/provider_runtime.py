"""Build reproducible benchmark policy runtimes from a frozen model snapshot."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from ecocompanion.benchmark.model_provider import PolicyRuntime
from ecocompanion.config import Settings
from ecocompanion.experiments.io import read_json_object
from ecocompanion.experiments.models import PolicyRuntimeRecord
from ecocompanion.experiments.policy_registry import policy_requires_provider
from ecocompanion.infrastructure import ModelPricing
from ecocompanion.providers import OpenAICompatibleClient
from ecocompanion.providers.benchmark_adapter import (
    BenchmarkProviderConfig,
    OpenAICompatibleBenchmarkProvider,
)
from ecocompanion.providers.recorded_gateway import OpenAICompatibleSyncGateway

_PROVIDER_NAME = "litellm-openai-compatible"


def build_new_policy_runtime(
    *,
    settings: Settings,
    policy_names: Sequence[str],
    model_id: str | None,
    snapshot_path: Path | None,
    run_dir: Path | None = None,
    run_id: str | None = None,
) -> PolicyRuntime:
    """Create a provider only when a selected policy requires one."""

    requires_provider = policy_requires_provider(policy_names)
    if not requires_provider:
        if model_id is not None or snapshot_path is not None:
            raise ValueError(
                "--agent-model/--model-snapshot require a model-backed policy"
            )
        return PolicyRuntime()

    path = _resolve_snapshot_path(settings, snapshot_path)
    payload = read_json_object(path)
    selected_model = _select_qualified_model(
        payload,
        requested=model_id or settings.default_model,
    )
    destination, selected_run_id = _require_run_location(
        run_dir=run_dir,
        run_id=run_id,
    )
    return _build_runtime(
        settings=settings,
        model_id=selected_model,
        model_family=_model_family(payload, selected_model),
        allowed_response_model_ids=_allowed_response_model_ids(
            payload,
            selected_model,
        ),
        snapshot_digest=_file_digest(path),
        run_dir=destination,
        run_id=selected_run_id,
    )


def build_resume_policy_runtime(
    *,
    settings: Settings,
    policy_runtimes: Sequence[PolicyRuntimeRecord],
    model_id: str | None,
    snapshot_path: Path | None,
    run_dir: Path | None = None,
    run_id: str | None = None,
) -> PolicyRuntime:
    """Recreate exactly the provider identity frozen in a run manifest."""

    provider_records = tuple(
        record for record in policy_runtimes if record.provider_name is not None
    )
    if not provider_records:
        if model_id is not None or snapshot_path is not None:
            raise ValueError(
                "--agent-model/--model-snapshot are unused by this run"
            )
        return PolicyRuntime()

    identities = {
        (
            record.provider_name,
            record.model_id,
            record.model_snapshot_digest,
            record.pricing_version,
            record.input_price_cny_per_million,
            record.output_price_cny_per_million,
            record.api_budget_cny,
        )
        for record in provider_records
    }
    if len(identities) != 1:
        raise ValueError(
            "this CLI supports one frozen benchmark provider identity per run"
        )
    (
        provider_name,
        recorded_model,
        recorded_digest,
        recorded_pricing_version,
        recorded_input_price,
        recorded_output_price,
        recorded_budget,
    ) = next(iter(identities))
    if provider_name != _PROVIDER_NAME:
        raise ValueError(
            f"unsupported benchmark provider in manifest: {provider_name!r}"
        )
    if recorded_model is None or recorded_digest is None:
        raise ValueError(
            "model-backed manifest lacks model or snapshot provenance"
        )
    if model_id is not None and model_id != recorded_model:
        raise ValueError(
            "resume model does not match the immutable run manifest"
        )
    current_pricing = _model_pricing(settings)
    current_accounting = (
        settings.model_pricing_version,
        (
            current_pricing.input_per_million
            if current_pricing is not None
            else None
        ),
        (
            current_pricing.output_per_million
            if current_pricing is not None
            else None
        ),
        Decimal(str(settings.api_budget_cny)),
    )
    recorded_accounting = (
        recorded_pricing_version,
        recorded_input_price,
        recorded_output_price,
        recorded_budget,
    )
    if recorded_accounting != current_accounting:
        raise ValueError(
            "resume pricing or API budget does not match the immutable run "
            "manifest"
        )

    path = _resolve_snapshot_path(settings, snapshot_path)
    current_digest = _file_digest(path)
    if current_digest != recorded_digest:
        raise ValueError(
            "model snapshot content no longer matches the immutable run manifest"
        )
    payload = read_json_object(path)
    _select_qualified_model(payload, requested=recorded_model)
    destination, selected_run_id = _require_run_location(
        run_dir=run_dir,
        run_id=run_id,
    )
    return _build_runtime(
        settings=settings,
        model_id=recorded_model,
        model_family=_model_family(payload, recorded_model),
        allowed_response_model_ids=_allowed_response_model_ids(
            payload,
            recorded_model,
        ),
        snapshot_digest=current_digest,
        run_dir=destination,
        run_id=selected_run_id,
    )


def _build_runtime(
    *,
    settings: Settings,
    model_id: str,
    model_family: str | None,
    allowed_response_model_ids: frozenset[str],
    snapshot_digest: str,
    run_dir: Path,
    run_id: str,
) -> PolicyRuntime:
    pricing = _model_pricing(settings)
    config = BenchmarkProviderConfig(
        model_id=model_id,
        run_id=run_id,
        cache_path=run_dir / "api_cache.sqlite3",
        cost_ledger_path=run_dir / "api_cost_ledger.jsonl",
        trace_path=run_dir / "api_trace.jsonl",
        model_snapshot_digest=snapshot_digest,
        model_family=model_family,
        pricing=pricing,
        pricing_version=settings.model_pricing_version,
        formal_mode=settings.research_mode,
        api_budget_cny=Decimal(str(settings.api_budget_cny)),
        allowed_response_model_ids=allowed_response_model_ids,
    )
    client = OpenAICompatibleClient(
        base_url=settings.litellm_base_url,
        api_key=settings.require_api_key(),
        connect_timeout_seconds=settings.connect_timeout_seconds,
        read_timeout_seconds=settings.read_timeout_seconds,
        max_same_model_retries=settings.max_same_model_retries,
        allowed_response_model_ids={
            model_id: allowed_response_model_ids,
        },
        default_extra_body=settings.provider_extra_body,
    )
    gateway = OpenAICompatibleSyncGateway(client, close_client=True)
    provider = OpenAICompatibleBenchmarkProvider(
        gateway=gateway,
        config=config,
        close_callback=gateway.close,
    )
    return PolicyRuntime(
        provider=provider,
        scope_factory=provider.for_scope,
    )


def _model_pricing(settings: Settings) -> ModelPricing | None:
    input_price = settings.model_input_price_cny_per_million
    output_price = settings.model_output_price_cny_per_million
    if input_price is None or output_price is None:
        return None
    return ModelPricing(
        input_per_million=Decimal(str(input_price)),
        output_per_million=Decimal(str(output_price)),
    )


def _require_run_location(
    *,
    run_dir: Path | None,
    run_id: str | None,
) -> tuple[Path, str]:
    if run_dir is None:
        raise ValueError("model-backed policy runtime requires run_dir")
    destination = run_dir.resolve()
    selected_run_id = run_id or destination.name
    if not selected_run_id.strip():
        raise ValueError("model-backed policy runtime requires a non-empty run_id")
    return destination, selected_run_id


def _resolve_snapshot_path(
    settings: Settings,
    requested: Path | None,
) -> Path:
    path = (
        requested
        if requested is not None
        else settings.artifacts_dir / "models" / "model_snapshot.json"
    )
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"qualified model snapshot not found: {resolved}"
        )
    return resolved


def _select_qualified_model(
    payload: dict[str, Any],
    *,
    requested: str | None,
) -> str:
    capabilities = payload.get("capabilities")
    qualifications = payload.get("qualifications")
    if not isinstance(capabilities, list) or not isinstance(
        qualifications,
        list,
    ):
        raise ValueError(
            "benchmark policies require a qualified model snapshot"
        )
    ready = {
        str(item["model_id"])
        for item in capabilities
        if isinstance(item, dict)
        and item.get("status") == "ready"
        and item.get("basic_chat") is True
        and item.get("chinese_text") is True
        and isinstance(item.get("model_id"), str)
    }
    scores = {
        str(item["model_id"]): float(item["qualification_score"])
        for item in qualifications
        if isinstance(item, dict)
        and isinstance(item.get("model_id"), str)
        and isinstance(item.get("qualification_score"), int | float)
        and item["model_id"] in ready
    }
    eligible = ready & set(scores)
    if not eligible:
        raise ValueError(
            "qualified snapshot contains no ready, scored Chinese model"
        )
    if requested is not None:
        if requested not in eligible:
            raise ValueError(
                f"model {requested!r} is not eligible in the frozen snapshot"
            )
        return requested

    recommended = payload.get("recommended_roles")
    if isinstance(recommended, dict):
        companion = recommended.get("companion_model")
        if isinstance(companion, str) and companion in eligible:
            return companion
    return min(eligible, key=lambda item: (-scores[item], item))


def _model_family(payload: dict[str, Any], model_id: str) -> str | None:
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, dict) or item.get("model_id") != model_id:
            continue
        owner = item.get("declared_owner")
        return owner if isinstance(owner, str) and owner.strip() else None
    return None


def _allowed_response_model_ids(
    payload: dict[str, Any],
    model_id: str,
) -> frozenset[str]:
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        return frozenset()
    for item in capabilities:
        if not isinstance(item, dict) or item.get("model_id") != model_id:
            continue
        values = item.get("response_model_ids", ())
        if not isinstance(values, list | tuple):
            raise ValueError(
                f"model {model_id!r} has invalid response_model_ids"
            )
        normalized = frozenset(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
        if len(normalized) != len(values):
            raise ValueError(
                f"model {model_id!r} has invalid response_model_ids"
            )
        return normalized
    return frozenset()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
