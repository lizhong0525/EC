"""Opaque, persisted identities and a frozen, input-only Track A view."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from ecocompanion.benchmark.model_provider import (
    BenchmarkModelProvider,
    BenchmarkProviderIdentity,
    DecisionProviderRequest,
    ResponseProviderRequest,
)
from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    ExpressionStyle,
    RiskTrajectory,
    SocialEcology,
    Track,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    CaseSpecV1,
    ConversationMessage,
    CurrentObservation,
    ObservationV1,
    ResponsePayload,
    VisibleContext,
)
from ecocompanion.research import VERSION
from ecocompanion.research.contracts import Provenance, ResearchCase

_ENCODED_ID = re.compile(r"case-r\d+-e\d+-x\d+-p\d+", re.I)
_PRIVATE_KEY = re.compile(
    r'"(?:hidden_state|world_state_before|world_state_after|preferred_action|'
    r"prohibited_actions|acceptable_actions|risk_trajectory|social_ecology|source_path|"
    r'persona_tags|event_template_family|dataset_split|provenance|reference_action)"\s*:'
)


@dataclass
class IdentityMap:
    """Persist this curator-only map; randomness is allocated once, not per request."""

    mapping: dict[str, str] = field(default_factory=dict)

    def opaque(self, internal: str) -> str:
        if internal not in self.mapping:
            self.mapping[internal] = uuid4().hex
        return self.mapping[internal]


def audit_visible_payload(payload: str, internal_ids: Iterable[str] = ()) -> None:
    private_values = {
        e.value for cls in (RiskTrajectory, SocialEcology, ExpressionStyle) for e in cls
    }

    def inspect(value: object) -> None:
        if isinstance(value, str):
            if (
                _ENCODED_ID.search(value)
                or _PRIVATE_KEY.search(value)
                or any(re.search(rf"\b{re.escape(v)}\b", value) for v in private_values)
            ):
                raise ValueError("encoded identity or curator-only field in visible input")
            if any(identifier and identifier in value for identifier in internal_ids):
                raise ValueError("internal identifier in visible input")
        elif isinstance(value, dict):
            inspect(json.dumps(list(value), ensure_ascii=False))
            for key, item in value.items():
                if _PRIVATE_KEY.search(json.dumps({key: None})):
                    raise ValueError("curator-only key in visible input")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(payload)
    try:
        parsed = json.loads(payload)
    except (ValueError, TypeError):
        return
    inspect(parsed)


def observable_case(case: CaseSpecV1, identities: IdentityMap) -> ResearchCase:
    case_id = identities.opaque(case.case_id)
    observations: list[ObservationV1] = []
    history: list[ConversationMessage] = []
    for step in case.timeline:
        observation_id = identities.opaque(step.observation_id)
        observation = ObservationV1(
            benchmark_version=VERSION,
            track=Track.A,
            episode_id=case_id,
            step_id=step.step_id,
            simulated_time=step.simulated_time,
            visible_profile=case.visible_profile,
            current_observation=CurrentObservation(
                observation_id=observation_id,
                message=step.frozen_utterances[0],
            ),
            visible_context=VisibleContext(
                recent_events=step.external_events,
                conversation_history=tuple(history),
            ),
            allowed_primary_actions=tuple(Action),
        )
        audit_visible_payload(observation.model_dump_json(), (case.case_id,))
        observations.append(observation)
        # A frozen observation track must not depend on earlier candidate responses.
        history.append(
            ConversationMessage(
                message_id=observation_id,
                role=ActorRole.STUDENT_SIMULATOR,
                content=step.frozen_utterances[0],
                simulated_time=step.simulated_time,
            )
        )
    return ResearchCase(
        case_id=case_id,
        observations=tuple(observations),
        provenance=Provenance(
            origin="public_generator",
            source_id=case.case_id,
            author_id="legacy-generator",
            person_group=case.case_id,
            plot_group=case.risk_trajectory.value,
            template_group=case.event_template_family,
            source_group="public-v1-generator",
        ),
        split="development",
    )


def public_input(case: ResearchCase) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": case.case_id,
        "observations": [o.model_dump(mode="json") for o in case.observations],
    }
    audit_visible_payload(json.dumps(payload, ensure_ascii=False))
    return payload


class AuditedProvider:
    """Inspect the entire provider request, not just the latest user utterance."""

    def __init__(self, provider: BenchmarkModelProvider) -> None:
        self.provider = provider

    @property
    def identity(self) -> BenchmarkProviderIdentity:
        return self.provider.identity

    @staticmethod
    def _audit(request: DecisionProviderRequest | ResponseProviderRequest) -> None:
        if request.observation.benchmark_version != VERSION:
            raise ValueError("v2 provider refuses legacy observations")
        payload = json.dumps(
            asdict(request),
            ensure_ascii=False,
            default=lambda obj: obj.model_dump(mode="json"),
        )
        audit_visible_payload(payload)
        for identifier in (
            request.episode_id,
            request.observation.current_observation.observation_id,
            *(m.message_id for m in request.observation.visible_context.conversation_history),
        ):
            if re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
                raise ValueError("v2 provider requires opaque IDs in all request fields")
        if isinstance(request, DecisionProviderRequest):
            visible = {
                request.observation.current_observation.observation_id,
                *(m.message_id for m in request.observation.visible_context.conversation_history),
            }
            if not set(request.allowed_evidence_ids) <= visible:
                raise ValueError("provider evidence contract includes invisible observations")

    def decide(self, request: DecisionProviderRequest) -> AgentDecisionV1:
        self._audit(request)
        return self.provider.decide(request)

    def render(self, request: ResponseProviderRequest) -> ResponsePayload:
        self._audit(request)
        return self.provider.render(request)

    def close(self) -> None:
        self.provider.close()
