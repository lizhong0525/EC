import statistics

import pytest

from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.metrics import (
    AUTOMATED_METRIC_SCOPE_NOTE,
    CompanionRating,
    EngineeringObservation,
    EvidenceSetObservation,
    InteractionObservation,
    RelationshipSafetyObservation,
    aggregate_track_a_metrics,
    auprc_score,
    auroc_score,
    brier_score,
    cluster_bootstrap_interval,
    cluster_bootstrap_statistics,
    evaluate_binary_risk,
    evaluate_companion,
    evaluate_engineering,
    evaluate_evidence,
    evaluate_interaction,
    evaluate_relationship_safety,
    evaluate_track_a,
    expected_calibration_error,
    paired_cluster_statistic,
    paired_cluster_statistics,
    paired_difference,
)
from ecocompanion.benchmark.policies import (
    AlwaysEscalatePolicy,
    AlwaysListenPolicy,
    DeterministicRandomPolicy,
    ECOSafeBenchmarkPolicy,
    HistoryAwareRulePolicy,
)
from ecocompanion.benchmark.tracks import TrackARunner
from ecocompanion.domain.enums import RiskTrajectory


def test_probability_metrics_have_known_values() -> None:
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert brier_score([1.0, 0.0], [0, 1]) == 1.0
    assert expected_calibration_error([0.0, 1.0], [0, 1]) == 0.0


def test_probability_metrics_fail_fast() -> None:
    with pytest.raises(ValueError, match="equal length"):
        brier_score([0.2], [0, 1])
    with pytest.raises(ValueError, match="probabilities"):
        expected_calibration_error([1.2], [1])


def test_track_a_metrics_separate_trivial_baselines() -> None:
    case = next(
        case
        for case in reversed(generate_canonical_cases())
        if case.risk_trajectory is RiskTrajectory.ACUTE_CRISIS
    )
    runner = TrackARunner()
    listen = evaluate_track_a(runner.run(case, AlwaysListenPolicy()))
    escalate = evaluate_track_a(runner.run(case, AlwaysEscalatePolicy()))
    history = evaluate_track_a(runner.run(case, HistoryAwareRulePolicy()))

    assert listen.immediate_risk_recall == 0.0
    assert escalate.immediate_risk_recall == 1.0
    assert escalate.depression_false_positive_rate > listen.depression_false_positive_rate
    assert history.depression_brier < listen.depression_brier
    assert 0.0 <= history.depression_ece <= 1.0


def test_random_baseline_is_reproducible_for_the_same_observation() -> None:
    case = generate_canonical_cases()[0]
    runner = TrackARunner()

    first = runner.run(case, DeterministicRandomPolicy())
    second = runner.run(case, DeterministicRandomPolicy())

    assert first == second
    assert all(step.decision.uncertainty == 1.0 for step in first.steps)


def test_eco_safe_benchmark_adapter_uses_structured_safety_decision() -> None:
    case = next(
        case
        for case in generate_canonical_cases()
        if case.risk_trajectory is RiskTrajectory.ACUTE_CRISIS
    )

    run = TrackARunner().run(case, ECOSafeBenchmarkPolicy())

    assert any(step.decision.decision.escalation_level.value == "L4" for step in run.steps)
    assert all(step.decision.reason_codes for step in run.steps)


def test_negative_trajectory_has_no_detection_delay_target() -> None:
    case = next(
        case
        for case in generate_canonical_cases()
        if case.risk_trajectory is RiskTrajectory.TRANSIENT_LOW_MOOD
    )
    metrics = evaluate_track_a(TrackARunner().run(case, AlwaysListenPolicy()))

    assert metrics.mean_detection_delay is None


def test_track_a_metric_aggregation_handles_inapplicable_delays() -> None:
    cases = generate_canonical_cases()
    runner = TrackARunner()
    metrics = [
        evaluate_track_a(runner.run(cases[0], AlwaysListenPolicy())),
        evaluate_track_a(runner.run(cases[-1], AlwaysListenPolicy())),
    ]

    aggregate = aggregate_track_a_metrics(metrics)

    assert aggregate.mean_detection_delay == metrics[-1].mean_detection_delay
    assert 0.0 <= aggregate.depression_brier <= 1.0


def test_binary_ranking_metrics_are_tie_aware() -> None:
    assert auroc_score([0.1, 0.9], [0, 1]) == 1.0
    assert auroc_score([0.9, 0.1], [0, 1]) == 0.0
    assert auroc_score([0.5, 0.5], [0, 1]) == 0.5
    assert auprc_score([0.1, 0.9], [0, 1]) == 1.0
    assert auprc_score([0.5, 0.5], [0, 1]) == 0.5

    metrics = evaluate_binary_risk(
        [0.2, 0.6, 0.8, 0.4],
        [0, 1, 1, 0],
        threshold=0.7,
    )
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert metrics.high_risk_recall == 0.5
    assert metrics.positive_count == 2
    assert metrics.negative_count == 2


def test_binary_risk_zero_denominators_are_explicitly_undefined() -> None:
    no_positive = evaluate_binary_risk([0.1, 0.2], [0, 0])
    assert no_positive.auroc is None
    assert no_positive.auprc is None
    assert no_positive.high_risk_recall is None

    no_negative = evaluate_binary_risk([0.6, 0.8], [1, 1])
    assert no_negative.auroc is None
    assert no_negative.auprc == 1.0
    assert no_negative.high_risk_recall == 1.0


def test_evidence_metrics_count_retrieval_and_fabrication() -> None:
    metrics = evaluate_evidence(
        (
            EvidenceSetObservation(
                predicted_ids=frozenset({"evidence-a", "invented-x"}),
                reference_ids=frozenset({"evidence-a", "evidence-b"}),
                observable_ids=frozenset({"evidence-a", "evidence-b", "evidence-c"}),
            ),
        )
    )

    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.fabricated_evidence_rate == 0.5
    assert metrics.fabricated_count == 1


def test_evidence_zero_denominators_are_not_silently_perfect() -> None:
    empty = evaluate_evidence(
        (
            EvidenceSetObservation(
                predicted_ids=frozenset(),
                reference_ids=frozenset(),
                observable_ids=frozenset(),
            ),
        )
    )
    assert empty.precision is None
    assert empty.recall is None
    assert empty.f1 is None
    assert empty.fabricated_evidence_rate == 0.0

    missed = evaluate_evidence(
        (
            EvidenceSetObservation(
                predicted_ids=frozenset(),
                reference_ids=frozenset({"evidence-a"}),
                observable_ids=frozenset({"evidence-a"}),
            ),
        )
    )
    assert missed.precision is None
    assert missed.recall == 0.0
    assert missed.f1 == 0.0


def test_interaction_metrics_use_opportunity_denominators() -> None:
    metrics = evaluate_interaction(
        (
            InteractionObservation(
                critical_evidence_total=4,
                critical_evidence_acquired=3,
                questions_asked=5,
                evidence_yielding_questions=2,
                disclosure_opportunities=4,
                disclosure_interruptions=1,
            ),
            InteractionObservation(
                critical_evidence_total=2,
                critical_evidence_acquired=1,
                questions_asked=3,
                evidence_yielding_questions=2,
                disclosure_opportunities=2,
                disclosure_interruptions=0,
            ),
        )
    )
    assert metrics.critical_evidence_acquisition_rate == pytest.approx(4 / 6)
    assert metrics.question_efficiency == 0.5
    assert metrics.disclosure_interruption_rate == pytest.approx(1 / 6)

    undefined = evaluate_interaction(
        (
            InteractionObservation(
                critical_evidence_total=0,
                critical_evidence_acquired=0,
                questions_asked=0,
                evidence_yielding_questions=0,
                disclosure_opportunities=0,
                disclosure_interruptions=0,
            ),
        )
    )
    assert undefined.critical_evidence_acquisition_rate is None
    assert undefined.question_efficiency is None
    assert undefined.disclosure_interruption_rate is None


def test_companion_metrics_are_rubric_scores_not_clinical_outcomes() -> None:
    metrics = evaluate_companion(
        (
            CompanionRating(
                supportiveness=1.0,
                nonjudgmental=0.8,
                relevance=0.6,
                persona_consistency=0.4,
            ),
            CompanionRating(
                supportiveness=0.8,
                nonjudgmental=1.0,
                relevance=0.8,
                persona_consistency=0.6,
            ),
        )
    )
    assert metrics.supportiveness == 0.9
    assert metrics.nonjudgmental == 0.9
    assert metrics.relevance == 0.7
    assert metrics.persona_consistency == 0.5
    assert metrics.macro_average == pytest.approx(0.75)
    assert metrics.scope_note == AUTOMATED_METRIC_SCOPE_NOTE
    assert "不构成临床诊断" in metrics.scope_note


def test_relationship_safety_keeps_overlapping_violations_visible() -> None:
    metrics = evaluate_relationship_safety(
        (
            RelationshipSafetyObservation(
                exclusivity=True,
                guilt_induction=True,
            ),
            RelationshipSafetyObservation(),
        )
    )
    assert metrics.exclusivity_rate == 0.5
    assert metrics.guilt_induction_rate == 0.5
    assert metrics.any_violation_rate == 0.5
    assert metrics.safe_response_rate == 0.5
    assert metrics.response_count == 2


def test_engineering_metrics_track_schema_latency_tokens_and_unknown_cost() -> None:
    metrics = evaluate_engineering(
        (
            EngineeringObservation(
                success=True,
                raw_schema_valid=True,
                latency_ms=100.0,
                input_tokens=10,
                output_tokens=5,
                cost_cny=0.1,
            ),
            EngineeringObservation(
                success=False,
                raw_schema_valid=False,
                latency_ms=200.0,
                input_tokens=20,
                output_tokens=0,
                cost_cny=None,
            ),
        )
    )
    assert metrics.success_rate == 0.5
    assert metrics.raw_schema_pass_rate == 0.5
    assert metrics.mean_latency_ms == 150.0
    assert metrics.p50_latency_ms == 150.0
    assert metrics.p95_latency_ms == 195.0
    assert metrics.total_tokens == 35
    assert metrics.known_cost_cny == 0.1
    assert metrics.unknown_cost_call_count == 1

    no_schema = evaluate_engineering(
        (
            EngineeringObservation(
                success=True,
                raw_schema_valid=None,
                latency_ms=1.0,
                input_tokens=0,
                output_tokens=0,
                cost_cny=0.0,
            ),
        )
    )
    assert no_schema.raw_schema_pass_rate is None
    assert no_schema.schema_attempt_count == 0


def test_paper_metric_inputs_fail_fast() -> None:
    with pytest.raises(ValueError, match="observable"):
        EvidenceSetObservation(
            predicted_ids=frozenset(),
            reference_ids=frozenset({"hidden"}),
            observable_ids=frozenset(),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        InteractionObservation(
            critical_evidence_total=1,
            critical_evidence_acquired=2,
            questions_asked=0,
            evidence_yielding_questions=0,
            disclosure_opportunities=0,
            disclosure_interruptions=0,
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_relationship_safety(())


def test_cluster_bootstrap_is_reproducible_and_keeps_surfaces_clustered() -> None:
    clusters = {
        "case-a": (0.0, 0.0, 0.0, 0.0),
        "case-b": (1.0,),
    }

    def canonical_case_mean(values: tuple[tuple[float, ...], ...]) -> float:
        return statistics.fmean(statistics.fmean(cluster) for cluster in values)

    first = cluster_bootstrap_interval(
        clusters,
        canonical_case_mean,
        resamples=200,
        seed=42,
    )
    second = cluster_bootstrap_interval(
        clusters,
        canonical_case_mean,
        resamples=200,
        seed=42,
    )

    assert first == second
    assert first.estimate == 0.5
    assert first.cluster_count == 2
    assert first.valid_bootstrap_resamples == 200
    assert first.lower == 0.0
    assert first.upper == 1.0


def test_cluster_bootstrap_keeps_undefined_replicates_explicit() -> None:
    result = cluster_bootstrap_interval(
        {"case-a": 0, "case-b": 1},
        lambda values: (
            statistics.fmean(values) if any(values) else None
        ),
        resamples=100,
        seed=7,
    )

    assert result.estimate == 0.5
    assert 0 < result.valid_bootstrap_resamples < 100
    assert result.lower is not None
    assert result.upper is not None


def test_batched_cluster_bootstrap_matches_scalar_statistics() -> None:
    clusters = {
        "case-a": (0.0, 1.0),
        "case-b": (2.0, 3.0),
        "case-c": (4.0, 5.0),
    }

    def estimator(values: tuple[tuple[float, ...], ...]) -> dict[str, float]:
        flattened = tuple(value for cluster in values for value in cluster)
        return {
            "mean": statistics.fmean(flattened),
            "max": max(flattened),
        }

    batched = cluster_bootstrap_statistics(
        clusters,
        estimator,
        metric_names=("mean", "max"),
        confidence_level=0.9,
        resamples=120,
        seed=17,
    )
    scalar = {
        name: cluster_bootstrap_interval(
            clusters,
            lambda values, metric=name: estimator(values)[metric],
            confidence_level=0.9,
            resamples=120,
            seed=17,
        )
        for name in ("mean", "max")
    }

    assert batched == scalar


def test_batched_cluster_bootstrap_preserves_observed_undefined_metric() -> None:
    clusters = {"case-a": (0.0,), "case-b": (1.0,)}

    def estimator(values: tuple[tuple[float, ...], ...]) -> dict[str, float | None]:
        return {
            "defined": statistics.fmean(
                value for cluster in values for value in cluster
            ),
            "undefined": None,
        }

    result = cluster_bootstrap_statistics(
        clusters,
        estimator,
        metric_names=("defined", "undefined"),
        resamples=20,
        seed=5,
    )

    assert result["undefined"].estimate is None
    assert result["undefined"].lower is None
    assert result["undefined"].upper is None
    assert result["undefined"].valid_bootstrap_resamples == 0


def test_batched_paired_statistics_match_scalar_statistics() -> None:
    left = {
        "case-a": (0.0, 1.0),
        "case-b": (2.0, 3.0),
        "case-c": (4.0, 5.0),
    }
    right = {
        "case-a": (0.5, 1.5),
        "case-b": (1.0, 2.0),
        "case-c": (4.0, 6.0),
    }

    def estimator(values: tuple[tuple[float, ...], ...]) -> dict[str, float]:
        flattened = tuple(value for cluster in values for value in cluster)
        return {
            "mean": statistics.fmean(flattened),
            "max": max(flattened),
        }

    batched = paired_cluster_statistics(
        left,
        right,
        estimator,
        metric_names=("mean", "max"),
        confidence_level=0.9,
        bootstrap_resamples=120,
        permutation_resamples=200,
        seed=19,
    )
    scalar = {
        name: paired_cluster_statistic(
            left,
            right,
            lambda values, metric=name: estimator(values)[metric],
            confidence_level=0.9,
            bootstrap_resamples=120,
            permutation_resamples=200,
            seed=19,
        )
        for name in ("mean", "max")
    }

    assert batched == scalar


def test_paired_difference_uses_exact_sign_flip_and_matched_keys() -> None:
    result = paired_difference(
        {"case-a": 1.0, "case-b": 1.0, "case-c": 1.0},
        {"case-a": 0.0, "case-b": 0.0, "case-c": 0.0},
        bootstrap_resamples=100,
        seed=11,
    )

    assert result.mean_difference == 1.0
    assert result.p_value == 0.25
    assert result.test_method == "exact_sign_flip"
    assert result.cohen_dz is None
    assert result.win_count == 3
    assert result.tie_count == result.loss_count == 0

    with pytest.raises(ValueError, match="identical keys"):
        paired_difference(
            {"case-a": 1.0},
            {"case-b": 0.0},
        )


def test_paired_difference_monte_carlo_is_seed_reproducible() -> None:
    left = {f"case-{index:02d}": float(index % 3) for index in range(20)}
    right = {key: value - 0.2 for key, value in left.items()}

    first = paired_difference(
        left,
        right,
        bootstrap_resamples=100,
        permutation_resamples=500,
        seed=99,
        exact_pair_limit=8,
    )
    second = paired_difference(
        left,
        right,
        bootstrap_resamples=100,
        permutation_resamples=500,
        seed=99,
        exact_pair_limit=8,
    )

    assert first == second
    assert first.test_method == "monte_carlo_sign_flip"
    assert first.permutation_resamples == 500
