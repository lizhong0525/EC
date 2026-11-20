"""Immutable provider-facing value objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


class StructuredMode(StrEnum):
    """Ordered structured-output strategies supported by the client."""

    JSON_SCHEMA = "json_schema"
    TOOL_CALL = "tool_call"
    JSON_OBJECT = "json_object"
    PLAIN_JSON = "plain_json"


class ModelRole(StrEnum):
    EVALUATED_AGENT = "evaluated_agent"
    STUDENT_SIMULATOR = "student_simulator"
    JUDGE = "judge"


class CapabilityStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CompletionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.reasoning_tokens,
        )
        if any(value < 0 for value in values):
            raise ValueError("token usage cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")


@dataclass(frozen=True)
class ChatCompletion:
    requested_model: str
    response_model: str
    content: str | None
    finish_reason: str | None
    usage: CompletionUsage
    retry_count: int
    tool_arguments: str | None = None
    response_id: str | None = None


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredAttempt:
    mode: StructuredMode
    succeeded: bool
    repair: bool = False
    failure_category: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class StructuredCompletion(Generic[ModelT]):
    value: ModelT
    completion: ChatCompletion
    mode: StructuredMode
    attempts: tuple[StructuredAttempt, ...]


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    declared_owner: str | None = None
    created: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "declared_owner": self.declared_owner,
            "created": self.created,
        }


@dataclass(frozen=True)
class ModelCapabilityReport:
    model_id: str
    status: CapabilityStatus
    basic_chat: bool
    system_and_multiturn: bool
    chinese_text: bool
    usage_reported: bool
    preferred_structured_mode: StructuredMode | None
    streaming: bool = False
    eligible_roles: frozenset[ModelRole] = field(default_factory=frozenset)
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "basic_chat": self.basic_chat,
            "system_and_multiturn": self.system_and_multiturn,
            "chinese_text": self.chinese_text,
            "usage_reported": self.usage_reported,
            "streaming": self.streaming,
            "preferred_structured_mode": (
                self.preferred_structured_mode.value
                if self.preferred_structured_mode is not None
                else None
            ),
            "eligible_roles": sorted(role.value for role in self.eligible_roles),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class ModelRegistrySnapshot:
    models: tuple[ModelDescriptor, ...]
    capabilities: tuple[ModelCapabilityReport, ...]
    authenticated_discovery: bool
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [model.to_dict() for model in self.models],
            "capabilities": [
                capability.to_dict() for capability in self.capabilities
            ],
            "authenticated_discovery": self.authenticated_discovery,
            "discovered_at": self.discovered_at.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class RoleBinding:
    role: ModelRole
    model_id: str
    family: str | None


@dataclass(frozen=True)
class RoleSeparationReport:
    valid: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]


JsonObject = dict[str, Any]
