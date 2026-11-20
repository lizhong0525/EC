"""Track A execution primitives and compatibility exports for Tracks B/C."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ecocompanion.benchmark.cases import validate_case
from ecocompanion.benchmark.generation import relationship_cues_for
from ecocompanion.domain.enums import Action, ActorRole, Track
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    CaseSpecV1,
    CaseStepSpec,
    ConversationMessage,
    CurrentObservation,
    ObservationV1,
    VisibleContext,
)


class PolicyContractError(ValueError):
    """Raised when a benchmark policy returns an invalid decision."""


class BenchmarkPolicy(Protocol):
    """Minimal policy interface shared by deterministic runners."""

    @property
    def name(self) -> str:
        """Stable policy identifier used in result artifacts."""

    def decide(self, observation: ObservationV1) -> AgentDecisionV1:
        """Return one structured decision without mutating the observation."""


@dataclass(frozen=True, slots=True)
class TrackAStepResult:
    reference: CaseStepSpec
    observation: ObservationV1
    decision: AgentDecisionV1


@dataclass(frozen=True, slots=True)
class TrackARun:
    case: CaseSpecV1
    surface_index: int
    policy_name: str
    steps: tuple[TrackAStepResult, ...]


class TrackARunner:
    """Run a policy against a frozen, policy-independent trajectory."""

    def __init__(self, allowed_actions: Sequence[Action] = tuple(Action)) -> None:
        actions = tuple(allowed_actions)
        if not actions:
            raise ValueError("allowed_actions cannot be empty")
        if len(actions) != len(set(actions)):
            raise ValueError("allowed_actions cannot contain duplicates")
        self._allowed_actions = actions

    def run(
        self,
        case: CaseSpecV1,
        policy: BenchmarkPolicy,
        *,
        surface_index: int = 0,
    ) -> TrackARun:
        validated = validate_case(case)
        if surface_index < 0:
            raise ValueError("surface_index cannot be negative")
        if any(surface_index >= len(step.frozen_utterances) for step in validated.timeline):
            raise ValueError(f"{validated.case_id}: surface_index {surface_index} is unavailable")

        history: list[ConversationMessage] = []
        known_observation_ids: set[str] = set()
        results: list[TrackAStepResult] = []

        for step in validated.timeline:
            utterance = step.frozen_utterances[surface_index]
            observation = ObservationV1(
                track=Track.A,
                episode_id=(f"{validated.case_id}-surface-{surface_index:02d}"),
                step_id=step.step_id,
                simulated_time=step.simulated_time,
                visible_profile=validated.visible_profile,
                current_observation=CurrentObservation(
                    observation_id=step.observation_id,
                    message=utterance,
                ),
                visible_context=VisibleContext(
                    recent_events=step.external_events,
                    relationship_cues=relationship_cues_for(validated.social_ecology),
                    conversation_history=tuple(history),
                ),
                allowed_primary_actions=self._allowed_actions,
            )
            decision = policy.decide(observation)
            known_observation_ids.add(step.observation_id)
            self._validate_decision(
                observation,
                decision,
                known_observation_ids=known_observation_ids,
            )
            results.append(
                TrackAStepResult(
                    reference=step,
                    observation=observation,
                    decision=decision,
                )
            )
            history.extend(
                (
                    ConversationMessage(
                        message_id=f"{step.observation_id}-student",
                        role=ActorRole.STUDENT_SIMULATOR,
                        content=utterance,
                        simulated_time=step.simulated_time,
                    ),
                    ConversationMessage(
                        message_id=f"{step.observation_id}-companion",
                        role=ActorRole.COMPANION,
                        content=decision.response.text,
                        simulated_time=step.simulated_time,
                    ),
                )
            )
            known_observation_ids.update(
                {
                    f"{step.observation_id}-student",
                    f"{step.observation_id}-companion",
                }
            )

        return TrackARun(
            case=validated,
            surface_index=surface_index,
            policy_name=policy.name,
            steps=tuple(results),
        )

    @staticmethod
    def _validate_decision(
        observation: ObservationV1,
        decision: AgentDecisionV1,
        *,
        known_observation_ids: set[str],
    ) -> None:
        if decision.decision.primary_action not in observation.allowed_primary_actions:
            raise PolicyContractError(
                f"policy selected disallowed action {decision.decision.primary_action.value}"
            )
        if decision.decision.primary_action in decision.decision.secondary_actions:
            raise PolicyContractError("primary_action cannot also appear in secondary_actions")
        if len(decision.decision.secondary_actions) != len(
            set(decision.decision.secondary_actions)
        ):
            raise PolicyContractError("secondary_actions cannot contain duplicates")
        unknown_evidence = {
            evidence.observation_id
            for evidence in decision.evidence
            if evidence.observation_id not in known_observation_ids
        }
        if unknown_evidence:
            raise PolicyContractError(
                f"decision cites unknown observations: {sorted(unknown_evidence)}"
            )


# Kept here as compatibility imports for callers that previously imported all
# track primitives from ``ecocompanion.benchmark.tracks``.
from ecocompanion.benchmark.track_b import (  # noqa: E402
    TrackBDisclosureEngine,
    TrackBDisclosureState,
    TrackBTransition,
)
from ecocompanion.benchmark.track_c import (  # noqa: E402
    TrackCScenario,
    define_track_c_scenarios,
)

__all__ = [
    "BenchmarkPolicy",
    "PolicyContractError",
    "TrackARun",
    "TrackARunner",
    "TrackAStepResult",
    "TrackBDisclosureEngine",
    "TrackBDisclosureState",
    "TrackBTransition",
    "TrackCScenario",
    "define_track_c_scenarios",
]
