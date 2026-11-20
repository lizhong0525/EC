import json

import pytest

from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.benchmark.track_c import TrackCConfiguration
from ecocompanion.experiments.interactive_runner import (
    start_track_b_experiment,
    start_track_c_experiment,
)
from ecocompanion.experiments.report import (
    _compile_track_a_risk_cluster,
    _track_a_fast_clusters_compatible,
    _track_a_risk_values,
    _track_a_risk_values_fast,
    _TrackANode,
    build_report,
)
from ecocompanion.experiments.runner import start_experiment


def test_track_a_compiled_risk_metrics_match_reference() -> None:
    nodes = (
        _TrackANode(
            step_id=0,
            depression_prediction=0.1,
            depression_target=0,
            immediate_prediction=0.1,
            immediate_target=0,
            critical_action_probability=0.0,
            acceptable_action_probability=1.0,
            preferred_action_probability=1.0,
            prohibited_action_probability=0.0,
        ),
        _TrackANode(
            step_id=1,
            depression_prediction=0.8,
            depression_target=1,
            immediate_prediction=0.9,
            immediate_target=1,
            critical_action_probability=1.0,
            acceptable_action_probability=0.5,
            preferred_action_probability=0.0,
            prohibited_action_probability=0.0,
        ),
        _TrackANode(
            step_id=2,
            depression_prediction=0.7,
            depression_target=1,
            immediate_prediction=0.8,
            immediate_target=1,
            critical_action_probability=0.5,
            acceptable_action_probability=1.0,
            preferred_action_probability=0.5,
            prohibited_action_probability=0.0,
        ),
    )
    reference = _track_a_risk_values((nodes, nodes), risk_threshold=0.5)
    compiled = _compile_track_a_risk_cluster(nodes)
    optimized = _track_a_risk_values_fast(
        (compiled, compiled),
        risk_threshold=0.5,
    )

    assert _track_a_fast_clusters_compatible((compiled, compiled))
    assert all(
        optimized[name] == pytest.approx(value)
        if value is not None
        else optimized[name] is None
        for name, value in reference.items()
    )


def test_report_builds_json_and_markdown_main_table(tmp_path) -> None:
    run_dir = tmp_path / "run"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.HIDDEN_TEST,
        policy_names=("always-listen", "always-escalate"),
        surface_indices=(0,),
        repeats=1,
        case_limit=2,
        run_id="report-test",
    )

    json_path, markdown_path, report = build_report(
        run_dir,
        bootstrap_resamples=80,
        permutation_resamples=200,
        random_seed=7,
        failure_case_limit=5,
    )

    assert report.schema_version == "2.0"
    assert report.track == "A"
    assert report.complete is True
    assert report.completed_unit_count == report.planned_unit_count == 4
    assert [row.policy_name for row in report.rows] == [
        "always-escalate",
        "always-listen",
    ]
    assert all(row.episode_count == 2 for row in report.rows)
    assert all(row.metric_estimates["depression_brier"].cluster_count == 2 for row in report.rows)
    assert all(row.canonical_node_count == 28 for row in report.rows)
    assert all("depression_auroc" in row.metric_estimates for row in report.rows)
    brier_comparison = next(
        comparison
        for comparison in report.paired_comparisons
        if comparison.metric_name == "depression_brier"
    )
    assert brier_comparison.pair_count == 2
    assert len(report.failure_cases) <= 5
    assert [failure.rank for failure in report.failure_cases] == list(
        range(1, len(report.failure_cases) + 1)
    )
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["run_id"] == "report-test"
    assert persisted["schema_version"] == "2.0"
    assert persisted["statistical_protocol"]["cluster_unit"] == "canonical_case"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Track A 风险识别与校准" in markdown
    assert "## Track A 动作与转介时机" in markdown
    assert "## Track A 证据质量" in markdown
    assert "## 配对比较" in markdown
    assert "## 失败案例（Top-5）" in markdown
    assert "| always-escalate |" in markdown
    assert "| always-listen |" in markdown


def test_report_marks_intentionally_partial_run_incomplete(tmp_path) -> None:
    run_dir = tmp_path / "partial"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.PUBLIC_TEST,
        policy_names=("always-listen",),
        surface_indices=(0, 1),
        repeats=1,
        case_limit=1,
        run_id="partial",
        max_units=1,
    )

    _, _, report = build_report(
        run_dir,
        bootstrap_resamples=40,
        permutation_resamples=100,
    )

    assert report.complete is False
    assert report.completed_unit_count == 1
    assert report.planned_unit_count == 2


def test_report_summarizes_track_b_interaction_metrics(tmp_path) -> None:
    run_dir = tmp_path / "track-b"
    start_track_b_experiment(
        run_dir=run_dir,
        policy_names=("always-listen", "history-aware-rule"),
        case_limit=1,
        run_id="track-b-report",
    )

    json_path, markdown_path, report = build_report(
        run_dir,
        bootstrap_resamples=60,
        permutation_resamples=200,
        random_seed=11,
    )

    assert report.track == "B"
    assert report.schema_version == "2.0"
    assert report.complete is True
    assert len(report.rows) == 2
    assert {row.policy_name for row in report.rows} == {
        "always-listen",
        "history-aware-rule",
    }
    assert all(row.episode_count == 5 for row in report.rows)
    assert all(
        row.metric_estimates["critical_evidence_recall"].cluster_count == 1
        for row in report.rows
    )
    comparison = next(
        item
        for item in report.paired_comparisons
        if item.metric_name == "critical_evidence_recall"
    )
    assert comparison.matched_pair_count == 5
    assert comparison.cluster_count == comparison.pair_count == 1
    assert comparison.pair_unit == "canonical_case"
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["track"] == "B"
    assert "## Track B 交互式渐进披露" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_report_summarizes_track_c_relationship_safety(tmp_path) -> None:
    run_dir = tmp_path / "track-c"
    start_track_c_experiment(
        run_dir=run_dir,
        configurations=tuple(TrackCConfiguration),
        scenario_limit=1,
        run_id="track-c-report",
    )

    _, markdown_path, report = build_report(
        run_dir / "results.jsonl",
        bootstrap_resamples=60,
        permutation_resamples=200,
        random_seed=13,
        failure_case_limit=3,
    )

    assert report.track == "C"
    assert report.schema_version == "2.0"
    assert report.complete is True
    assert report.planned_unit_count == 20
    assert len(report.rows) == 4
    by_configuration = {
        row.configuration: row for row in report.rows
    }
    assert (
        by_configuration[
            TrackCConfiguration.MEMORY_PROACTIVE_ECO_SAFE
        ].violation_free_episode_rate
        == 1.0
    )
    assert (
        by_configuration[
            TrackCConfiguration.PLAIN_ASSISTANT
        ].violation_turn_rate
        > 0.0
    )
    plain = by_configuration[TrackCConfiguration.PLAIN_ASSISTANT]
    assert plain.relationship_safety.exclusivity_rate > 0.0
    assert (
        by_configuration[
            TrackCConfiguration.MEMORY_PROACTIVE_ECO_SAFE
        ].relationship_safety.discourage_offline_help_rate
        == 0.0
    )
    assert all(
        row.metric_estimates["violation_turn_rate"].cluster_count == 1
        for row in report.rows
    )
    comparison = next(
        item
        for item in report.paired_comparisons
        if item.metric_name == "violation_turn_rate"
    )
    assert comparison.matched_pair_count == 5
    assert comparison.cluster_count == comparison.pair_count == 1
    assert comparison.pair_unit == "scenario"
    assert len(report.failure_cases) <= 3
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Track C 文本关系安全总览" in markdown
    assert "## Track C 关系安全分栏" in markdown
    assert "## 失败案例（Top-3）" in markdown


def test_track_c_full_scenario_matrix_allows_feature_specific_pairing(
    tmp_path,
) -> None:
    """A complete Track C run must report structural opportunity NAs safely."""

    run_dir = tmp_path / "track-c-full-matrix"
    start_track_c_experiment(
        run_dir=run_dir,
        configurations=tuple(TrackCConfiguration),
        run_id="track-c-full-matrix",
    )

    _, _, report = build_report(
        run_dir,
        bootstrap_resamples=20,
        permutation_resamples=40,
        random_seed=17,
    )

    assert report.complete is True
    assert report.planned_unit_count == 240
    assert len(report.rows) == len(tuple(TrackCConfiguration))
    comparison = next(
        item
        for item in report.paired_comparisons
        if item.metric_name == "violation_turn_rate"
        and item.left_name.startswith("LONG_TERM_MEMORY@")
        and item.right_name.startswith("MEMORY_PROACTIVE@")
    )
    assert comparison.matched_pair_count == 55
    assert comparison.left_only_count == 0
    assert comparison.right_only_count == 5
    assert comparison.complete_pairing is False


def test_track_a_surfaces_are_repeated_measures_not_independent_cases(
    tmp_path,
) -> None:
    run_dir = tmp_path / "surface-clusters"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.HIDDEN_TEST,
        policy_names=("always-listen", "always-escalate"),
        surface_indices=(0, 1),
        repeats=2,
        case_limit=2,
        run_id="surface-clusters",
    )

    _, _, report = build_report(
        run_dir,
        bootstrap_resamples=60,
        permutation_resamples=100,
        random_seed=17,
    )

    assert all(row.episode_count == 8 for row in report.rows)
    assert all(row.case_count == 2 for row in report.rows)
    assert all(row.canonical_node_count == 28 for row in report.rows)
    assert all(
        estimate.cluster_count == 2
        for row in report.rows
        for estimate in row.metric_estimates.values()
    )
    comparison = next(
        item
        for item in report.paired_comparisons
        if item.metric_name == "depression_brier"
    )
    assert comparison.pair_count == 2


def test_report_statistics_are_reproducible_for_same_seed(tmp_path) -> None:
    run_dir = tmp_path / "reproducible"
    start_track_c_experiment(
        run_dir=run_dir,
        configurations=tuple(TrackCConfiguration),
        scenario_limit=2,
        run_id="reproducible",
    )

    _, _, first = build_report(
        run_dir,
        output_dir=tmp_path / "first",
        bootstrap_resamples=80,
        permutation_resamples=200,
        random_seed=23,
    )
    _, _, second = build_report(
        run_dir,
        output_dir=tmp_path / "second",
        bootstrap_resamples=80,
        permutation_resamples=200,
        random_seed=23,
    )

    assert first.rows == second.rows
    assert first.paired_comparisons == second.paired_comparisons
    assert first.failure_cases == second.failure_cases


def test_failure_extraction_supports_ranked_top_thirty(tmp_path) -> None:
    run_dir = tmp_path / "top-thirty"
    start_track_c_experiment(
        run_dir=run_dir,
        configurations=(TrackCConfiguration.PLAIN_ASSISTANT,),
        run_id="top-thirty",
    )

    _, _, report = build_report(
        run_dir,
        bootstrap_resamples=30,
        permutation_resamples=50,
        failure_case_limit=30,
    )

    assert len(report.failure_cases) == 30
    assert tuple(item.rank for item in report.failure_cases) == tuple(
        range(1, 31)
    )
    assert all(
        left.severity_score >= right.severity_score
        for left, right in zip(
            report.failure_cases,
            report.failure_cases[1:],
            strict=False,
        )
    )
