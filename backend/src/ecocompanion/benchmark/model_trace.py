"""Trace adapter for API-backed benchmark components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ecocompanion.domain.enums import Action, Track
from ecocompanion.domain.schemas import (
    ConversationMessage,
    CurrentObservation,
    ModelCallUsage,
    ObservationV1,
    RiskEstimate,
    SafetyAudit,
    TraceEventV1,
    VisibleContext,
    VisibleProfile,
)
from ecocompanion.infrastructure import JsonlTraceStore
from ecocompanion.providers.recorded_gateway import RecordedCompletion


@dataclass(frozen=True, slots=True)
class ModelTraceContext:
    """Observable context needed to emit one versioned TraceEventV1."""

    trace_id: str
    scenario_id: str
    episode_id: str
    step_id: int
    track: Track
    simulated_time: str
    observation_id: str
    message: str
    selected_action: Action
    allowed_actions: tuple[Action, ...]
    random_seed: int
    assessment: RiskEstimate = field(
        default_factory=lambda: RiskEstimate(
            depression_worsening=0.0,
            immediate_safety_risk=0.0,
            dependency_risk=0.0,
        )
    )
    conversation_history: tuple[ConversationMessage, ...] = ()
    memory_refs: tuple[str, ...] = ()
    external_event: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id.strip():
            raise ValueError("trace_id cannot be blank")
        if self.step_id < 0:
            raise ValueError("step_id cannot be negative")
        if not self.message.strip():
            raise ValueError("trace observation message cannot be blank")
        if self.selected_action not in self.allowed_actions:
            raise ValueError("selected_action must be included in allowed_actions")


class ModelTraceRecorder:
    """Idempotently append model-operation traces without chain-of-thought."""

    def __init__(self, store: JsonlTraceStore | None) -> None:
        self._store = store
        self._known = (
            {event.trace_id: event for event in store.read_all()}
            if store is not None
            else {}
        )

    def get(self, trace_id: str) -> TraceEventV1 | None:
        """Return a previously persisted event, if one exists.

        A trace is written before the surrounding benchmark result is
        checkpointed.  During a resume, callers can use this read-only lookup
        to replay the durable response instead of invoking a provider again.
        Returning the immutable Pydantic event also keeps the replay source
        single and avoids adding a second response journal.
        """

        if not trace_id.strip():
            raise ValueError("trace_id cannot be blank")
        self._refresh_known()
        return self._known.get(trace_id)

    def _refresh_known(self) -> None:
        """Refresh externally appended events before a resume lookup.

        A runner may create a recorder before another process finishes
        persisting a trace (for example, a supervisor can rebuild components
        while the worker is still draining its file buffers).  Keeping only
        the constructor snapshot would miss that durable response and cause a
        duplicate provider call.  The store remains the source of truth; any
        conflicting duplicate id is rejected just as ``record`` does.
        """

        if self._store is None:
            return
        for event in self._store.read_all():
            existing = self._known.get(event.trace_id)
            if existing is None:
                self._known[event.trace_id] = event
                continue
            if _logical_trace(existing) != _logical_trace(event):
                raise RuntimeError(
                    f"trace id already exists with different content: {event.trace_id}"
                )

    def record(
        self,
        *,
        context: ModelTraceContext,
        response_text: str,
        safety_audit: SafetyAudit,
        invocations: tuple[RecordedCompletion[Any], ...] = (),
    ) -> None:
        if self._store is None:
            return
        event = TraceEventV1(
            trace_id=context.trace_id,
            scenario_id=context.scenario_id,
            episode_id=context.episode_id,
            step_id=context.step_id,
            external_event={
                **context.external_event,
                "component_model_calls": len(invocations),
                "all_prices_known": all(
                    invocation.estimated_cost_cny is not None
                    for invocation in invocations
                ),
            },
            observation=ObservationV1(
                track=context.track,
                episode_id=context.episode_id,
                step_id=context.step_id,
                simulated_time=context.simulated_time,
                visible_profile=VisibleProfile(
                    age_group=_trace_age_group(context.track)
                ),
                current_observation=CurrentObservation(
                    observation_id=context.observation_id,
                    message=context.message,
                ),
                visible_context=VisibleContext(
                    conversation_history=context.conversation_history
                ),
                allowed_primary_actions=context.allowed_actions,
            ),
            memory_refs=context.memory_refs,
            assessment=context.assessment,
            selected_action=context.selected_action,
            response_text=response_text,
            safety_audit=safety_audit,
            random_seed=context.random_seed,
            model_usage=tuple(_model_usage(invocation) for invocation in invocations),
        )
        existing = self._known.get(event.trace_id)
        if existing is not None:
            if _logical_trace(existing) != _logical_trace(event):
                raise RuntimeError(
                    f"trace id already exists with different content: {event.trace_id}"
                )
            return
        self._store.append(event)
        self._known[event.trace_id] = event


def _model_usage(invocation: RecordedCompletion[Any]) -> ModelCallUsage:
    usage = invocation.completion.usage
    return ModelCallUsage(
        model=invocation.completion.requested_model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        estimated_cost_cny=(
            float(invocation.estimated_cost_cny)
            if invocation.estimated_cost_cny is not None
            else 0.0
        ),
        retry_index=invocation.completion.retry_count,
        cache_hit=invocation.cache_hit,
    )


def _trace_age_group(track: Track) -> str:
    if track in {Track.A, Track.B}:
        return "13-17 synthetic"
    return "synthetic relationship-safety scenario"


def _logical_trace(event: TraceEventV1) -> dict[str, Any]:
    payload = event.model_dump(
        mode="json",
        exclude={"recorded_at", "model_usage"},
    )
    payload["model_ids"] = tuple(usage.model for usage in event.model_usage)
    return payload
