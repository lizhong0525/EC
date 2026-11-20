import json
from collections import Counter
from dataclasses import dataclass

import pytest

from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.policies import (
    AlwaysListenPolicy,
    HistoryAwareRulePolicy,
)
from ecocompanion.benchmark.splitting import canonical_split
from ecocompanion.benchmark.track_b import (
    TRACK_B_SEEDS,
    StudentRenderRequest,
    TrackBDisclosureEngine,
    TrackBEpisodeRunner,
    build_track_b_episode_specs,
    select_track_b_cases,
)
from ecocompanion.benchmark.track_c import (
    TRACK_C_SEEDS,
    DeterministicTrackCBaselineResponder,
    RelationshipViolation,
    TrackCConfiguration,
    TrackCEpisodeRunner,
    build_track_c_episode_matrix,
    define_track_c_scenarios,
    run_track_c_matrix,
)
from ecocompanion.benchmark.tracks import (
    PolicyContractError,
    TrackARunner,
)
from ecocompanion.domain.enums import (
    Action,
    ExpressionStyle,
    RiskTrajectory,
    SocialEcology,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    EvidenceRef,
    ObservationV1,
    StudentTurnV1,
)


def test_track_a_is_frozen_and_builds_history() -> None:
    case = generate_canonical_cases()[0]
    before = case.model_dump_json()

    run = TrackARunner().run(case, HistoryAwareRulePolicy(), surface_index=1)

    assert case.model_dump_json() == before
    assert len(run.steps) == len(case.timeline)
    assert run.steps[0].observation.visible_context.conversation_history == ()
    assert len(run.steps[3].observation.visible_context.conversation_history) == 6
    assert "depression_burden" not in run.steps[0].observation.model_dump_json()
    assert (
        run.steps[0].observation.current_observation.message
        == case.timeline[0].frozen_utterances[1]
    )


@dataclass(frozen=True)
class UnknownEvidencePolicy:
    name: str = "unknown-evidence"

    def decide(self, observation: ObservationV1) -> AgentDecisionV1:
        baseline = AlwaysListenPolicy().decide(observation)
        return baseline.model_copy(
            update={
                "evidence": (
                    EvidenceRef(
                        observation_id="future-observation",
                        evidence_code="UNKNOWN",
                    ),
                )
            }
        )


def test_track_a_rejects_future_or_unknown_evidence() -> None:
    case = generate_canonical_cases()[0]

    with pytest.raises(PolicyContractError, match="unknown observations"):
        TrackARunner().run(case, UnknownEvidencePolicy())


def test_track_b_disclosure_is_deterministic_and_gated() -> None:
    case = generate_canonical_cases()[-1]
    engine = TrackBDisclosureEngine()
    initial = engine.initial_state(case)

    first = engine.advance(case, initial, Action.CLARIFY_SYMPTOMS)
    repeated = engine.advance(case, initial, Action.CLARIFY_SYMPTOMS)

    assert first == repeated
    assert first.newly_disclosed_fact_ids == ()
    assert first.after.disclosure_readiness > initial.disclosure_readiness

    state = first.after
    state = engine.advance(case, state, Action.CLARIFY_SYMPTOMS).after
    third = engine.advance(case, state, Action.CLARIFY_SYMPTOMS)
    assert len(third.newly_disclosed_fact_ids) == 1
    assert third.student_turn.disclosed_fact_ids == third.newly_disclosed_fact_ids


def test_track_b_direct_check_prioritizes_safety_fact_when_available() -> None:
    case = next(
        case
        for case in generate_canonical_cases()
        if case.risk_trajectory is RiskTrajectory.ACUTE_CRISIS
    )
    engine = TrackBDisclosureEngine()
    state = engine.initial_state(case)
    for _ in range(10):
        state = engine.advance(case, state, Action.CLARIFY_SYMPTOMS).after

    transition = engine.advance(case, state, Action.DIRECT_SAFETY_CHECK)
    fact_map = {fact.fact_id: fact for fact in case.disclosure_facts}

    assert len(transition.newly_disclosed_fact_ids) == 1
    disclosed = fact_map[transition.newly_disclosed_fact_ids[0]]
    assert disclosed.evidence_code == "PASSIVE_DEATH_WISH"


def test_track_b_stops_after_timeline() -> None:
    case = generate_canonical_cases()[0]
    engine = TrackBDisclosureEngine()
    state = engine.initial_state(case)
    for _ in case.timeline:
        state = engine.advance(case, state, Action.SUPPORTIVE_LISTENING).after

    with pytest.raises(StopIteration, match="complete"):
        engine.advance(case, state, Action.SUPPORTIVE_LISTENING)


def test_track_b_rejects_state_from_another_case() -> None:
    cases = generate_canonical_cases()
    engine = TrackBDisclosureEngine()
    foreign_state = engine.initial_state(cases[0])

    with pytest.raises(ValueError, match="state belongs"):
        engine.advance(cases[1], foreign_state, Action.SUPPORTIVE_LISTENING)


def test_track_b_selects_24_stratified_non_hidden_cases() -> None:
    cases = generate_canonical_cases()
    selected = select_track_b_cases(cases)
    splits = canonical_split(cases)
    hidden_ids = {case.case_id for case in splits.hidden_test}

    assert len(selected) == 24
    assert not hidden_ids & {case.case_id for case in selected}
    assert Counter(case.risk_trajectory for case in selected) == {
        trajectory: 4 for trajectory in RiskTrajectory
    }
    assert Counter(case.social_ecology for case in selected) == {
        ecology: 6 for ecology in SocialEcology
    }
    assert Counter(case.expression_style for case in selected) == {
        style: 8 for style in ExpressionStyle
    }


def test_track_b_builds_120_fixed_seed_specs_with_12_to_20_turns() -> None:
    specs = build_track_b_episode_specs(generate_canonical_cases())

    assert len(specs) == 120
    assert {spec.seed for spec in specs} == set(TRACK_B_SEEDS)
    assert all(12 <= spec.turn_limit <= 20 for spec in specs)
    assert set(Counter(spec.case_id for spec in specs).values()) == {5}
    assert len({spec.episode_id for spec in specs}) == len(specs)


def test_track_b_episode_is_json_serializable_and_replayable() -> None:
    cases = generate_canonical_cases()
    case_by_id = {case.case_id: case for case in cases}
    spec = build_track_b_episode_specs(cases)[0]
    runner = TrackBEpisodeRunner()

    first = runner.run(case_by_id[spec.case_id], spec, HistoryAwareRulePolicy())
    second = runner.run(case_by_id[spec.case_id], spec, HistoryAwareRulePolicy())

    assert first == second
    assert first.replay_digest == second.replay_digest
    assert len(first.turns) == spec.turn_limit
    assert json.loads(json.dumps(first.to_dict(), ensure_ascii=False))["track"] == "B"
    assert all(
        "depression_burden" not in turn.observation.model_dump_json() for turn in first.turns
    )
    assert 0.0 <= first.metrics.critical_evidence_recall <= 1.0
    assert 0.0 <= first.metrics.question_efficiency <= 1.0
    assert 0.0 <= first.metrics.disclosure_interruption_rate <= 1.0


@dataclass(frozen=True)
class DisallowedFactRenderer:
    name: str = "disallowed-fact-renderer"

    def render(self, request: StudentRenderRequest) -> StudentTurnV1:
        return StudentTurnV1(
            episode_id=request.episode_id,
            step_id=request.step_id,
            disclosed_fact_ids=("fact-outside-allow-list",),
            text="尝试泄漏未允许事实。",
            renderer_model="mock-model",
        )


def test_track_b_renderer_violation_uses_template_without_state_authority() -> None:
    case = generate_canonical_cases()[0]
    engine = TrackBDisclosureEngine(renderer=DisallowedFactRenderer())
    state = engine.initial_state(case)
    transition = engine.advance(
        case,
        state,
        Action.CLARIFY_SYMPTOMS,
        seed=7,
    )

    assert transition.renderer_fallback_used
    assert set(transition.newly_disclosed_fact_ids) <= set(transition.offered_fact_ids)
    assert transition.after.world_state == transition.world_transition.after
    assert (
        transition.after.world_state.depression_burden
        == transition.before.world_state.depression_burden
    )


def test_track_c_defines_12_text_scenarios_and_240_episode_matrix() -> None:
    scenarios = define_track_c_scenarios()
    specs = build_track_c_episode_matrix(scenarios)

    assert len(scenarios) == 12
    assert len({scenario.scenario_id for scenario in scenarios}) == 12
    assert {scenario.channel for scenario in scenarios} == {"text"}
    assert len(specs) == 240
    assert {spec.configuration for spec in specs} == set(TrackCConfiguration)
    assert {spec.seed for spec in specs} == set(TRACK_C_SEEDS)
    assert set(Counter(spec.scenario_id for spec in specs).values()) == {20}
    assert set(Counter(spec.configuration for spec in specs).values()) == {60}
    assert {spec.features.channel for spec in specs} == {"text"}


def test_track_c_safe_configuration_reduces_relationship_violations() -> None:
    scenarios = define_track_c_scenarios()
    scenario = scenarios[0]
    specs = build_track_c_episode_matrix(scenarios)
    plain_spec = next(
        spec
        for spec in specs
        if spec.scenario_id == scenario.scenario_id
        and spec.configuration is TrackCConfiguration.PLAIN_ASSISTANT
    )
    safe_spec = next(
        spec
        for spec in specs
        if spec.scenario_id == scenario.scenario_id
        and spec.configuration is TrackCConfiguration.MEMORY_PROACTIVE_ECO_SAFE
    )
    runner = TrackCEpisodeRunner()
    responder = DeterministicTrackCBaselineResponder()

    plain = runner.run(scenario, plain_spec, responder)
    safe = runner.run(scenario, safe_spec, responder)

    assert plain.metrics.total_violations > 0
    assert safe.metrics.total_violations == 0
    assert any(RelationshipViolation.EXCLUSIVITY in turn.violations for turn in plain.turns)


def test_track_c_full_matrix_is_json_serializable_and_replayable() -> None:
    first = run_track_c_matrix()
    second = run_track_c_matrix()

    assert len(first) == 240
    assert tuple(result.replay_digest for result in first) == tuple(
        result.replay_digest for result in second
    )
    payload = [result.to_dict() for result in first]
    assert len(json.loads(json.dumps(payload, ensure_ascii=False))) == 240
