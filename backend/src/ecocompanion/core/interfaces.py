"""Dependency-inversion contracts for the companion pipeline."""

from typing import Protocol

from ecocompanion.core.policy import PolicyOutcome
from ecocompanion.core.risk import AssessmentResult
from ecocompanion.domain.schemas import (
    ObservationV1,
    ResponsePayload,
    SafetyAudit,
    SessionStateV1,
)


class RiskAssessor(Protocol):
    def assess(self, observation: ObservationV1) -> AssessmentResult:
        """Query risk evidence without mutating session state."""


class CompanionPolicy(Protocol):
    def decide(
        self,
        *,
        assessment: AssessmentResult,
        observation: ObservationV1,
        session: SessionStateV1,
    ) -> PolicyOutcome:
        """Select a structured action without rendering language."""


class ResponseRenderer(Protocol):
    async def render(
        self,
        *,
        observation: ObservationV1,
        assessment: AssessmentResult,
        policy: PolicyOutcome,
        session: SessionStateV1,
    ) -> ResponsePayload:
        """Render a structured action as user-facing language."""


class OutputAuditor(Protocol):
    async def audit(
        self,
        *,
        response: ResponsePayload,
        policy: PolicyOutcome,
        session: SessionStateV1,
    ) -> SafetyAudit:
        """Audit a rendered response without mutating state."""

