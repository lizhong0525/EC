"""Single immutable registry for benchmark baselines and ECO-SAFE ablations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ecocompanion.benchmark.llm_policies import (
    LLMBenchmarkPolicy,
    LLMContextMode,
    LLMPolicyConfig,
)
from ecocompanion.benchmark.model_provider import (
    BenchmarkModelProvider,
    PolicyFamily,
    PolicyRuntime,
)
from ecocompanion.benchmark.policies import (
    AlwaysEscalatePolicy,
    AlwaysListenPolicy,
    DeterministicRandomPolicy,
    ECOSafeBenchmarkPolicy,
    ECOSafeFeatureConfig,
    HistoryAwareRulePolicy,
    KeywordTriagePolicy,
    RuleGradedReferralPolicy,
)
from ecocompanion.benchmark.tracks import BenchmarkPolicy
from ecocompanion.experiments.models import PolicyRuntimeRecord

PolicyFactory = Callable[[PolicyRuntime], BenchmarkPolicy]

_IMPLEMENTATION_VERSION = "benchmark-policies-v1"
_LLM_PROMPT_VERSION = "agent-decision-v1"
_EMPATHY_PROMPT_VERSION = "agent-decision-empathy-v1"
_MEMORY_PROMPT_VERSION = "agent-decision-memory-v1"


@dataclass(frozen=True, slots=True)
class PolicyDefinition:
    """One registry entry and its immutable experimental treatment."""

    name: str
    family: PolicyFamily
    factory: PolicyFactory
    requires_provider: bool
    implementation_version: str = _IMPLEMENTATION_VERSION
    prompt_version: str | None = None
    enabled_features: tuple[str, ...] = ()
    ablated_component: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("policy definition name cannot be blank")
        if not self.implementation_version.strip():
            raise ValueError("implementation_version cannot be blank")
        if len(self.enabled_features) != len(set(self.enabled_features)):
            raise ValueError("enabled_features cannot contain duplicates")


def _current_message_llm(runtime: PolicyRuntime) -> BenchmarkPolicy:
    return LLMBenchmarkPolicy(
        provider=_require_provider(runtime, "current-message-llm"),
        config=LLMPolicyConfig(
            name="current-message-llm",
            prompt_version=_LLM_PROMPT_VERSION,
            context_mode=LLMContextMode.CURRENT_MESSAGE,
        ),
    )


def _full_history_llm(runtime: PolicyRuntime) -> BenchmarkPolicy:
    return LLMBenchmarkPolicy(
        provider=_require_provider(runtime, "full-history-llm"),
        config=LLMPolicyConfig(
            name="full-history-llm",
            prompt_version=_LLM_PROMPT_VERSION,
            context_mode=LLMContextMode.FULL_HISTORY,
        ),
    )


def _strong_empathy_llm(runtime: PolicyRuntime) -> BenchmarkPolicy:
    return LLMBenchmarkPolicy(
        provider=_require_provider(runtime, "strong-empathy-llm"),
        config=LLMPolicyConfig(
            name="strong-empathy-llm",
            prompt_version=_EMPATHY_PROMPT_VERSION,
            context_mode=LLMContextMode.FULL_HISTORY,
            strong_empathy_prompt=True,
        ),
    )


def _long_term_memory_companion(runtime: PolicyRuntime) -> BenchmarkPolicy:
    return LLMBenchmarkPolicy(
        provider=_require_provider(
            runtime,
            "long-term-memory-companion",
        ),
        config=LLMPolicyConfig(
            name="long-term-memory-companion",
            prompt_version=_MEMORY_PROMPT_VERSION,
            context_mode=LLMContextMode.LONG_TERM_MEMORY,
        ),
    )


def _rule_graded_referral(runtime: PolicyRuntime) -> BenchmarkPolicy:
    return RuleGradedReferralPolicy(
        provider=_require_provider(runtime, "rule-graded-referral")
    )


def _eco_safe(
    runtime: PolicyRuntime,
    *,
    name: str = "eco-safe",
    features: ECOSafeFeatureConfig | None = None,
) -> BenchmarkPolicy:
    return ECOSafeBenchmarkPolicy(
        name=name,
        features=features or ECOSafeFeatureConfig(),
        provider=_require_provider(runtime, name),
    )


def _eco_safe_full(runtime: PolicyRuntime) -> BenchmarkPolicy:
    return _eco_safe(runtime)


def _eco_safe_without_long_term_memory(
    runtime: PolicyRuntime,
) -> BenchmarkPolicy:
    return _eco_safe(
        runtime,
        name="eco-safe-no-long-term-memory",
        features=replace(
            ECOSafeFeatureConfig(),
            long_term_memory=False,
        ),
    )


def _eco_safe_without_uncertainty(runtime: PolicyRuntime) -> BenchmarkPolicy:
    return _eco_safe(
        runtime,
        name="eco-safe-no-uncertainty",
        features=replace(
            ECOSafeFeatureConfig(),
            uncertainty=False,
        ),
    )


def _eco_safe_without_dependency_risk(
    runtime: PolicyRuntime,
) -> BenchmarkPolicy:
    return _eco_safe(
        runtime,
        name="eco-safe-no-dependency-risk",
        features=replace(
            ECOSafeFeatureConfig(),
            dependency_risk=False,
        ),
    )


def _eco_safe_without_evidence_constraint(
    runtime: PolicyRuntime,
) -> BenchmarkPolicy:
    return _eco_safe(
        runtime,
        name="eco-safe-no-evidence-constraint",
        features=replace(
            ECOSafeFeatureConfig(),
            evidence_constraint=False,
        ),
    )


def _eco_safe_without_graded_planning(
    runtime: PolicyRuntime,
) -> BenchmarkPolicy:
    return _eco_safe(
        runtime,
        name="eco-safe-no-graded-planning",
        features=replace(
            ECOSafeFeatureConfig(),
            graded_action_planning=False,
        ),
    )


def _eco_safe_without_output_audit(
    runtime: PolicyRuntime,
) -> BenchmarkPolicy:
    return _eco_safe(
        runtime,
        name="eco-safe-no-output-audit",
        features=replace(
            ECOSafeFeatureConfig(),
            output_safety_audit=False,
        ),
    )


def _no_runtime(factory: Callable[[], BenchmarkPolicy]) -> PolicyFactory:
    def create(runtime: PolicyRuntime) -> BenchmarkPolicy:
        del runtime
        return factory()

    return create


_ECO_SAFE_FEATURES = ECOSafeFeatureConfig().enabled_feature_names


def _eco_safe_features_without(component: str) -> tuple[str, ...]:
    return (
        *(
            feature
            for feature in _ECO_SAFE_FEATURES
            if feature != component
        ),
        "model_response_rendering",
    )


_POLICY_DEFINITIONS: tuple[PolicyDefinition, ...] = (
    PolicyDefinition(
        name="random",
        family=PolicyFamily.DIAGNOSTIC_BASELINE,
        factory=_no_runtime(DeterministicRandomPolicy),
        requires_provider=False,
        enabled_features=("random_action",),
    ),
    PolicyDefinition(
        name="always-listen",
        family=PolicyFamily.DIAGNOSTIC_BASELINE,
        factory=_no_runtime(AlwaysListenPolicy),
        requires_provider=False,
        enabled_features=("unconditional_listening",),
    ),
    PolicyDefinition(
        name="always-escalate",
        family=PolicyFamily.DIAGNOSTIC_BASELINE,
        factory=_no_runtime(AlwaysEscalatePolicy),
        requires_provider=False,
        enabled_features=("unconditional_escalation",),
    ),
    PolicyDefinition(
        name="keyword-triage",
        family=PolicyFamily.DIAGNOSTIC_BASELINE,
        factory=_no_runtime(KeywordTriagePolicy),
        requires_provider=False,
        enabled_features=("current_message", "keyword_rules"),
    ),
    PolicyDefinition(
        name="history-aware-rule",
        family=PolicyFamily.DIAGNOSTIC_BASELINE,
        factory=_no_runtime(HistoryAwareRulePolicy),
        requires_provider=False,
        enabled_features=("full_history", "keyword_rules"),
    ),
    PolicyDefinition(
        name="current-message-llm",
        family=PolicyFamily.LLM_BASELINE,
        factory=_current_message_llm,
        requires_provider=True,
        prompt_version=_LLM_PROMPT_VERSION,
        enabled_features=("current_message", "model_decision"),
    ),
    PolicyDefinition(
        name="full-history-llm",
        family=PolicyFamily.LLM_BASELINE,
        factory=_full_history_llm,
        requires_provider=True,
        prompt_version=_LLM_PROMPT_VERSION,
        enabled_features=("full_history", "model_decision"),
    ),
    PolicyDefinition(
        name="strong-empathy-llm",
        family=PolicyFamily.LLM_BASELINE,
        factory=_strong_empathy_llm,
        requires_provider=True,
        prompt_version=_EMPATHY_PROMPT_VERSION,
        enabled_features=(
            "full_history",
            "model_decision",
            "strong_empathy_prompt",
        ),
    ),
    PolicyDefinition(
        name="long-term-memory-companion",
        family=PolicyFamily.LLM_BASELINE,
        factory=_long_term_memory_companion,
        requires_provider=True,
        prompt_version=_MEMORY_PROMPT_VERSION,
        enabled_features=(
            "long_term_memory",
            "recent_working_memory",
            "model_decision",
        ),
    ),
    PolicyDefinition(
        name="rule-graded-referral",
        family=PolicyFamily.RULE_AGENT,
        factory=_rule_graded_referral,
        requires_provider=True,
        prompt_version="rule-referral-render-v1",
        enabled_features=(
            "rule_risk_assessment",
            "graded_referral",
            "model_response_rendering",
        ),
    ),
    PolicyDefinition(
        name="eco-safe",
        family=PolicyFamily.ECO_SAFE,
        factory=_eco_safe_full,
        requires_provider=True,
        prompt_version="eco-safe-render-v1",
        enabled_features=(
            *_ECO_SAFE_FEATURES,
            "model_response_rendering",
        ),
    ),
    PolicyDefinition(
        name="eco-safe-no-long-term-memory",
        family=PolicyFamily.ECO_SAFE_ABLATION,
        factory=_eco_safe_without_long_term_memory,
        requires_provider=True,
        prompt_version="eco-safe-render-v1",
        enabled_features=_eco_safe_features_without("long_term_memory"),
        ablated_component="long_term_memory",
    ),
    PolicyDefinition(
        name="eco-safe-no-uncertainty",
        family=PolicyFamily.ECO_SAFE_ABLATION,
        factory=_eco_safe_without_uncertainty,
        requires_provider=True,
        prompt_version="eco-safe-render-v1",
        enabled_features=_eco_safe_features_without("uncertainty"),
        ablated_component="uncertainty",
    ),
    PolicyDefinition(
        name="eco-safe-no-dependency-risk",
        family=PolicyFamily.ECO_SAFE_ABLATION,
        factory=_eco_safe_without_dependency_risk,
        requires_provider=True,
        prompt_version="eco-safe-render-v1",
        enabled_features=_eco_safe_features_without("dependency_risk"),
        ablated_component="dependency_risk",
    ),
    PolicyDefinition(
        name="eco-safe-no-evidence-constraint",
        family=PolicyFamily.ECO_SAFE_ABLATION,
        factory=_eco_safe_without_evidence_constraint,
        requires_provider=True,
        prompt_version="eco-safe-render-v1",
        enabled_features=_eco_safe_features_without("evidence_constraint"),
        ablated_component="evidence_constraint",
    ),
    PolicyDefinition(
        name="eco-safe-no-graded-planning",
        family=PolicyFamily.ECO_SAFE_ABLATION,
        factory=_eco_safe_without_graded_planning,
        requires_provider=True,
        prompt_version="eco-safe-render-v1",
        enabled_features=_eco_safe_features_without(
            "graded_action_planning"
        ),
        ablated_component="graded_action_planning",
    ),
    PolicyDefinition(
        name="eco-safe-no-output-audit",
        family=PolicyFamily.ECO_SAFE_ABLATION,
        factory=_eco_safe_without_output_audit,
        requires_provider=True,
        prompt_version="eco-safe-render-v1",
        enabled_features=_eco_safe_features_without("output_safety_audit"),
        ablated_component="output_safety_audit",
    ),
)

_DEFINITIONS_BY_NAME = {
    definition.name: definition for definition in _POLICY_DEFINITIONS
}
if len(_DEFINITIONS_BY_NAME) != len(_POLICY_DEFINITIONS):
    raise RuntimeError("policy registry contains duplicate names")


def available_policy_names() -> tuple[str, ...]:
    """Return stable policy identifiers accepted by Tracks A and B."""

    return tuple(sorted(_DEFINITIONS_BY_NAME))


def normalize_policy_names(policy_names: Sequence[str]) -> tuple[str, ...]:
    """Normalize and validate a user-provided policy selection."""

    normalized = tuple(dict.fromkeys(name.strip() for name in policy_names))
    if not normalized or any(not name for name in normalized):
        raise ValueError("at least one non-empty policy name is required")
    unknown = set(normalized) - set(_DEFINITIONS_BY_NAME)
    if unknown:
        raise ValueError(
            f"unknown policies: {sorted(unknown)}; "
            f"available: {list(available_policy_names())}"
        )
    return normalized


def policy_requires_provider(policy_names: Sequence[str]) -> bool:
    """Query whether any selected policy needs a model provider."""

    normalized = normalize_policy_names(policy_names)
    return any(
        _DEFINITIONS_BY_NAME[name].requires_provider for name in normalized
    )


def describe_policy_runtimes(
    policy_names: Sequence[str],
    runtime: PolicyRuntime | None = None,
) -> tuple[PolicyRuntimeRecord, ...]:
    """Build immutable manifest records for the exact selected treatments."""

    normalized = normalize_policy_names(policy_names)
    effective_runtime = runtime or PolicyRuntime()
    records: list[PolicyRuntimeRecord] = []
    for name in normalized:
        definition = _DEFINITIONS_BY_NAME[name]
        provider = (
            _require_provider(effective_runtime, name)
            if definition.requires_provider
            else None
        )
        identity = provider.identity if provider is not None else None
        records.append(
            PolicyRuntimeRecord(
                policy_name=name,
                family=definition.family,
                implementation_version=definition.implementation_version,
                prompt_version=definition.prompt_version,
                provider_name=(
                    identity.provider_name if identity is not None else None
                ),
                model_id=identity.model_id if identity is not None else None,
                model_snapshot_digest=(
                    identity.model_snapshot_digest
                    if identity is not None
                    else None
                ),
                pricing_version=(
                    identity.pricing_version if identity is not None else None
                ),
                input_price_cny_per_million=(
                    identity.input_price_cny_per_million
                    if identity is not None
                    else None
                ),
                output_price_cny_per_million=(
                    identity.output_price_cny_per_million
                    if identity is not None
                    else None
                ),
                api_budget_cny=(
                    identity.api_budget_cny if identity is not None else None
                ),
                enabled_features=definition.enabled_features,
                ablated_component=definition.ablated_component,
            )
        )
    return tuple(records)


def create_policy(
    policy_name: str,
    runtime: PolicyRuntime | None = None,
) -> BenchmarkPolicy:
    """Create a fresh policy instance using explicit injected dependencies."""

    try:
        definition = _DEFINITIONS_BY_NAME[policy_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown policy {policy_name!r}; "
            f"available: {list(available_policy_names())}"
        ) from exc
    return definition.factory(runtime or PolicyRuntime())


def _require_provider(
    runtime: PolicyRuntime,
    policy_name: str,
) -> BenchmarkModelProvider:
    if runtime.provider is None:
        raise ValueError(
            f"policy {policy_name!r} requires an injected benchmark model provider"
        )
    return runtime.provider
