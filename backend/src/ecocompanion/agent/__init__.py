"""Safety-oriented building blocks for the EcoCompanion agent."""

from ecocompanion.agent.handoff import (
    HandoffConflictError,
    HandoffRecord,
    HandoffRequest,
    InvalidHandoffTransition,
    MockHandoffGateway,
    UnknownHandoffRequest,
)
from ecocompanion.agent.memory import (
    MemoryConflictError,
    MemoryService,
    MemoryStoreError,
    ProtectedMemoryError,
)
from ecocompanion.agent.proactivity import (
    ProactivePrompt,
    ProactivityEligibility,
    ProactivityEligibilityReason,
    ProactivityNotAllowed,
    ProactivityPauseReason,
    ProactivityScheduler,
    ProactivityState,
    PromptConflictError,
    PromptStatus,
    UnknownPrompt,
)
from ecocompanion.agent.safety import AuditedResponse, SafetyAuditor

__all__ = [
    "AuditedResponse",
    "HandoffConflictError",
    "HandoffRecord",
    "HandoffRequest",
    "InvalidHandoffTransition",
    "MemoryConflictError",
    "MemoryService",
    "MemoryStoreError",
    "MockHandoffGateway",
    "ProactivePrompt",
    "ProactivityEligibility",
    "ProactivityEligibilityReason",
    "ProactivityNotAllowed",
    "ProactivityPauseReason",
    "ProactivityScheduler",
    "ProactivityState",
    "PromptConflictError",
    "PromptStatus",
    "ProtectedMemoryError",
    "SafetyAuditor",
    "UnknownHandoffRequest",
    "UnknownPrompt",
]
