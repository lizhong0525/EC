"""Offline Track B assumption stress tests, not empirical psychological validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from statistics import fmean
from typing import Any

from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.track_b import (
    TrackBDisclosureEngine,
    TrackBEpisodeRunner,
    build_track_b_episode_specs,
)
from ecocompanion.core.world import WorldEngine, WorldEvent, WorldTransition
from ecocompanion.domain.enums import Action
from ecocompanion.domain.schemas import HiddenState
from ecocompanion.research.governance import rank_sensitivity
from ecocompanion.research.workflows import LOCAL_POLICIES


@dataclass(frozen=True)
class AssumptionScales:
    trust: float = 1
    disclosure: float = 1
    offline_help: float = 1
    dependency: float = 1

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or not 0 <= value <= 2
            for value in (self.trust, self.disclosure, self.offline_help, self.dependency)
        ):
            raise ValueError("assumption scales must be finite in [0,2]")


class ScaledWorldEngine(WorldEngine):
    def __init__(self, scales: AssumptionScales) -> None:
        self.scales = scales

    def transition(
        self, *, state: HiddenState, event: WorldEvent, action: Action, seed: int
    ) -> WorldTransition:
        original = super().transition(state=state, event=event, action=action, seed=seed)
        trust, disclosure, help_delta, dependency = self._action_effects(action)
        scales = self.scales
        # Recompute from the pre-clipped state, preserving the legacy jitter exactly.
        values = original.after.model_dump()
        deltas = {
            "trust_in_companion": trust * scales.trust,
            "disclosure_readiness": (
                disclosure * scales.disclosure + Random(seed).uniform(-0.01, 0.01)
            ),
            "offline_help_willingness": help_delta * scales.offline_help,
            "ai_dependency": dependency * scales.dependency,
        }
        values.update(
            {
                key: round(max(0.0, min(1.0, getattr(state, key) + delta)), 6)
                for key, delta in deltas.items()
            }
        )
        return WorldTransition(
            before=state,
            after=HiddenState.model_validate(values),
            event=event,
            action=action,
            seed=seed,
        )


def run_sensitivity(*, max_cases: int = 24) -> dict[str, Any]:
    if not 1 <= max_cases <= 24:
        raise ValueError("max_cases must be in [1,24]")
    cases = generate_canonical_cases()
    by_id = {c.case_id: c for c in cases}
    specs = build_track_b_episode_specs(cases)
    case_ids = sorted({s.case_id for s in specs})[:max_cases]
    specs = tuple(s for s in specs if s.case_id in case_ids)
    settings = {
        "baseline": AssumptionScales(),
        "weaker_disclosure": AssumptionScales(disclosure=0.5, dependency=1.5),
        "stronger_disclosure": AssumptionScales(disclosure=1.5, dependency=0.5),
    }
    measurements: dict[str, dict[str, float]] = {}
    episodes: list[dict[str, Any]] = []
    for name, scales in settings.items():
        runner = TrackBEpisodeRunner(
            disclosure_engine=TrackBDisclosureEngine(world_engine=ScaledWorldEngine(scales))
        )
        measurements[name] = {}
        for policy_name, factory in LOCAL_POLICIES.items():
            outputs = [runner.run(by_id[s.case_id], s, factory()) for s in specs]
            measurements[name][policy_name] = fmean(
                o.metrics.critical_evidence_recall for o in outputs
            )
            episodes.extend(
                {
                    "setting": name,
                    "policy": policy_name,
                    "case_id": o.spec.case_id,
                    "seed": o.spec.seed,
                    "replay_digest": o.replay_digest,
                    "critical_evidence_recall": o.metrics.critical_evidence_recall,
                    "final_dependency": o.turns[-1].transition.after.dependency,
                    "final_help_willingness": o.turns[-1].transition.after.offline_help_willingness,
                }
                for o in outputs
            )
    return {
        "status": "EXPLORATORY_LOCAL_ASSUMPTION_TEST",
        "assumption_plausibility": "PENDING_INDEPENDENT_EXPERT_REVIEW",
        "primary_metric": "critical_evidence_recall",
        "case_count": max_cases,
        "settings": {name: s.__dict__ for name, s in settings.items()},
        "source": "public reconstructible v1 generator",
        "model_calls": 0,
        "rank_sensitivity": rank_sensitivity(measurements),
        "episodes": episodes,
        "warning": (
            "Parameter robustness is not evidence of real trust, disclosure or dependency effects."
        ),
    }
