"""Provider-neutral contracts for model-backed benchmark policies.

The benchmark core is intentionally synchronous and deterministic.  A concrete
provider may bridge to an asynchronous API client, while tests inject an
in-memory fake.  Requests contain only explicitly selected observable text and
never expose hidden simulator state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Protocol

from ecocompanion.domain.enums import Action
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    DecisionPlan,
    ObservationV1,
    ResponsePayload,
    RiskEstimate,
)


class PolicyFamily(StrEnum):
    DIAGNOSTIC_BASELINE = "diagnostic_baseline"
    LLM_BASELINE = "llm_baseline"
    RULE_AGENT = "rule_agent"
    ECO_SAFE = "eco_safe"
    ECO_SAFE_ABLATION = "eco_safe_ablation"


class BenchmarkProviderFatalError(RuntimeError):
    """A formal provider invariant failed and the run must stop."""


@dataclass(frozen=True, slots=True)
class BenchmarkProviderIdentity:
    """Secret-free identity persisted in an experiment manifest."""

    provider_name: str
    model_id: str
    model_snapshot_digest: str | None = None
    pricing_version: str | None = None
    input_price_cny_per_million: Decimal | None = None
    output_price_cny_per_million: Decimal | None = None
    api_budget_cny: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.provider_name.strip():
            raise ValueError("provider_name cannot be blank")
        if not self.model_id.strip():
            raise ValueError("model_id cannot be blank")
        if (
            self.model_snapshot_digest is not None
            and not self.model_snapshot_digest.strip()
        ):
            raise ValueError("model_snapshot_digest cannot be blank")
        if self.pricing_version is not None and not self.pricing_version.strip():
            raise ValueError("pricing_version cannot be blank")
        prices = (
            self.input_price_cny_per_million,
            self.output_price_cny_per_million,
        )
        if any(price is not None and price < 0 for price in prices):
            raise ValueError("model prices cannot be negative")
        if (prices[0] is None) != (prices[1] is None):
            raise ValueError("input and output model prices must be set together")
        if self.api_budget_cny is not None and not (
            Decimal(0) < self.api_budget_cny <= Decimal(5000)
        ):
            raise ValueError("api_budget_cny must be in (0, 5000]")


@dataclass(frozen=True, slots=True)
class BenchmarkInvocationScope:
    """Immutable experiment-unit context used for caching and accounting."""

    run_id: str
    unit_id: str
    seed: int
    replicate_id: int

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be blank")
        if not self.unit_id.strip():
            raise ValueError("unit_id cannot be blank")
        if self.replicate_id < 0:
            raise ValueError("replicate_id cannot be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkPromptMessage:
    """One immutable provider-facing message."""

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("prompt message content cannot be blank")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class DecisionProviderRequest:
    """Visible-only request for a complete structured agent decision."""

    policy_name: str
    prompt_version: str
    episode_id: str
    step_id: int
    observation: ObservationV1
    messages: tuple[BenchmarkPromptMessage, ...]
    allowed_actions: tuple[Action, ...]
    allowed_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name cannot be blank")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version cannot be blank")
        if not self.episode_id.strip():
            raise ValueError("episode_id cannot be blank")
        if self.step_id < 0:
            raise ValueError("step_id cannot be negative")
        if self.observation.episode_id != self.episode_id:
            raise ValueError("request episode_id must match observation")
        if self.observation.step_id != self.step_id:
            raise ValueError("request step_id must match observation")
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if not self.allowed_actions:
            raise ValueError("allowed_actions cannot be empty")
        if len(self.allowed_actions) != len(set(self.allowed_actions)):
            raise ValueError("allowed_actions cannot contain duplicates")
        if len(self.allowed_evidence_ids) != len(set(self.allowed_evidence_ids)):
            raise ValueError("allowed_evidence_ids cannot contain duplicates")
        if self.allowed_actions != self.observation.allowed_primary_actions:
            raise ValueError(
                "allowed_actions must match the observation action contract"
            )


@dataclass(frozen=True, slots=True)
class ResponseProviderRequest:
    """Visible-only request for rendering an already fixed local action."""

    policy_name: str
    prompt_version: str
    episode_id: str
    step_id: int
    observation: ObservationV1
    assessment: RiskEstimate
    decision: DecisionPlan
    messages: tuple[BenchmarkPromptMessage, ...]

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name cannot be blank")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version cannot be blank")
        if not self.episode_id.strip():
            raise ValueError("episode_id cannot be blank")
        if self.step_id < 0:
            raise ValueError("step_id cannot be negative")
        if self.observation.episode_id != self.episode_id:
            raise ValueError("request episode_id must match observation")
        if self.observation.step_id != self.step_id:
            raise ValueError("request step_id must match observation")
        if (
            self.decision.primary_action
            not in self.observation.allowed_primary_actions
        ):
            raise ValueError(
                "planned response action is outside the observation contract"
            )
        if not self.messages:
            raise ValueError("messages cannot be empty")


class BenchmarkModelProvider(Protocol):
    """Replaceable model boundary used by LLM policies and response renderers."""

    @property
    def identity(self) -> BenchmarkProviderIdentity:
        """Return a secret-free, immutable provider identity."""

    def decide(self, request: DecisionProviderRequest) -> AgentDecisionV1:
        """Return one schema-valid decision."""

    def render(self, request: ResponseProviderRequest) -> ResponsePayload:
        """Render an already selected local action without changing it."""

    def close(self) -> None:
        """Release provider resources idempotently."""


@dataclass(frozen=True, slots=True)
class PolicyRuntime:
    """Injected runtime dependencies shared by freshly built policies."""

    provider: BenchmarkModelProvider | None = None
    scope_factory: (
        Callable[[BenchmarkInvocationScope], BenchmarkModelProvider] | None
    ) = None

    def __post_init__(self) -> None:
        if self.scope_factory is not None and self.provider is None:
            raise ValueError("scope_factory requires a root provider")

    def for_unit(self, scope: BenchmarkInvocationScope) -> PolicyRuntime:
        """Return an immutable unit-scoped view without changing the root."""

        if self.scope_factory is None:
            return self
        return PolicyRuntime(provider=self.scope_factory(scope))

    def close(self) -> None:
        """Release the injected provider, if any."""

        if self.provider is not None:
            self.provider.close()
