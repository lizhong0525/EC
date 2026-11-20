import json

import pytest

from ecocompanion.benchmark.annotations import (
    AnnotationBatchV1,
    AnnotationRole,
    NodeAnnotationV1,
    validate_annotation_batch,
)
from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.metrics import evaluate_track_a
from ecocompanion.benchmark.policies import (
    AlwaysEscalatePolicy,
)
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.benchmark.track_c import TrackCConfiguration
from ecocompanion.benchmark.tracks import TrackARunner
from ecocompanion.domain.enums import RiskLevel, RiskTrajectory
from ecocompanion.domain.schemas import CaseSpecV1, EvidenceRef
from ecocompanion.experiments.interactive_runner import (
    start_track_b_experiment,
    start_track_c_experiment,
)
from ecocompanion.experiments.io import read_jsonl_models
from ecocompanion.experiments.models import ExperimentResult
from ecocompanion.experiments.report import build_report
from ecocompanion.experiments.runner import start_experiment
from ecocompanion.experiments.validation import load_benchmark_cases


def test_track_a_pooled_comparison_matches_headline_and_holm(tmp_path) -> None:
    run_dir = tmp_path / "track-a-pooled"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.HIDDEN_TEST,
        policy_names=("always-listen", "always-escalate"),
        surface_indices=(0,),
        repeats=1,
        run_id="track-a-pooled",
    )

    _, markdown_path, report = build_report(
        run_dir,
        confidence_level=0.90,
        bootstrap_resamples=30,
        permutation_resamples=80,
        random_seed=41,
    )

    rows = {row.policy_name: row for row in report.rows}
    comparison = next(
        item
        for item in report.paired_comparisons
        if item.metric_name == "depression_auroc"
    )
    assert comparison.mean_difference == pytest.approx(
        rows[comparison.left_name].depression_auroc
        - rows[comparison.right_name].depression_auroc
    )
    assert comparison.cluster_count == comparison.pair_count == 24
    assert comparison.matched_pair_count == 24
    assert comparison.holm_adjusted_p_value >= comparison.raw_p_value
    assert rows["always-escalate"].critical_action_recall == 1.0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "90% CI" in markdown
    assert "95% CI" not in markdown


def test_detection_delay_starts_at_onset_and_reports_early_alarm() -> None:
    case = next(
        item
        for item in generate_canonical_cases()
        if item.risk_trajectory is RiskTrajectory.GRADUAL_WORSENING
    )

    metrics = evaluate_track_a(
        TrackARunner().run(case, AlwaysEscalatePolicy())
    )

    assert metrics.mean_detection_delay == 0.0
    assert metrics.pre_onset_alarm_rate == 1.0
    assert metrics.mean_pre_onset_lead_time is not None
    assert metrics.mean_pre_onset_lead_time > 0.0
    assert metrics.undetected_onset_rate == 0.0


def test_track_b_seed_pairs_are_clustered_with_missing_units_disclosed(
    tmp_path,
) -> None:
    run_dir = tmp_path / "track-b-partial-pairs"
    start_track_b_experiment(
        run_dir=run_dir,
        policy_names=("always-listen", "history-aware-rule"),
        case_limit=1,
        run_id="track-b-partial-pairs",
    )
    result_path = run_dir / "results.jsonl"
    payloads = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    removed = next(
        payload
        for payload in payloads
        if payload["policy_name"] == "always-listen"
    )
    payloads.remove(removed)
    result_path.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False)
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )

    _, _, report = build_report(
        run_dir,
        bootstrap_resamples=20,
        permutation_resamples=40,
    )

    assert report.complete is False
    comparison = next(
        item
        for item in report.paired_comparisons
        if item.metric_name == "critical_evidence_recall"
    )
    assert comparison.matched_pair_count == 4
    assert comparison.cluster_count == comparison.pair_count == 1
    assert comparison.left_only_count + comparison.right_only_count == 1
    assert comparison.complete_pairing is False


def test_track_c_uses_response_and_feature_specific_opportunities(
    tmp_path,
) -> None:
    run_dir = tmp_path / "track-c-opportunities"
    start_track_c_experiment(
        run_dir=run_dir,
        configurations=(TrackCConfiguration.PLAIN_ASSISTANT,),
        run_id="track-c-opportunities",
    )

    _, _, report = build_report(
        run_dir,
        bootstrap_resamples=20,
        permutation_resamples=40,
    )

    row = report.rows[0]
    assert row.episode_count == 60
    assert row.response_opportunity_count == 110
    assert (
        row.metric_estimates["proactive_persistence_rate"].estimate
        is None
    )
    assert row.violation_opportunity_counts["PROACTIVE_PERSISTENCE"] == 0
    assert row.violation_opportunity_counts["FALSE_CONFIDENTIALITY"] > 0


def test_report_requires_gold_for_evidence_precision_recall_and_f1(
    tmp_path,
) -> None:
    run_dir = tmp_path / "track-a-no-gold"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=("always-listen",),
        surface_indices=(0,),
        repeats=1,
        case_limit=1,
        run_id="track-a-no-gold",
    )

    _, _, report = build_report(
        run_dir,
        bootstrap_resamples=20,
        permutation_resamples=40,
    )

    row = report.rows[0]
    assert report.evidence_scoring.status == "NO_GOLD_ANNOTATION"
    assert row.evidence_code_precision is None
    assert row.evidence_code_recall is None
    assert row.evidence_code_f1 is None
    assert row.fabricated_evidence_rate == 0.0


def test_adjudicated_gold_scores_full_evidence_references(tmp_path) -> None:
    run_dir = tmp_path / "track-a-gold"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=("always-listen",),
        surface_indices=(0,),
        repeats=1,
        case_limit=1,
        run_id="track-a-gold",
    )
    case = _selected_case(run_dir)
    result_path = run_dir / "results.jsonl"
    result = read_jsonl_models(result_path, ExperimentResult)[0]
    target_step = next(
        step for step in case.timeline if step.available_fact_ids
    )
    fact_by_id = {
        fact.fact_id: fact for fact in case.disclosure_facts
    }
    evidence = EvidenceRef(
        observation_id=target_step.observation_id,
        evidence_code=fact_by_id[
            target_step.available_fact_ids[0]
        ].evidence_code,
    )
    updated_steps = tuple(
        step
        if step.step_id != target_step.step_id
        else step.model_copy(
            update={
                "evidence_refs": (evidence,),
                "evidence_codes": (evidence.evidence_code,),
            }
        )
        for step in result.steps
    )
    updated = ExperimentResult.model_validate(
        {
            **result.model_dump(mode="python"),
            "metrics": {
                **result.metrics.model_dump(mode="python"),
                "evidence_code_precision": 1.0,
            },
            "steps": updated_steps,
        }
    )
    result_path.write_text(
        updated.model_dump_json() + "\n",
        encoding="utf-8",
    )
    annotation_path = tmp_path / "adjudicated-gold.json"
    _write_gold(
        annotation_path,
        case,
        evidence_by_step={target_step.step_id: (evidence,)},
    )

    _, _, report = build_report(
        run_dir,
        annotation_path=annotation_path,
        bootstrap_resamples=20,
        permutation_resamples=40,
    )

    row = report.rows[0]
    assert (
        report.evidence_scoring.status
        == "VALIDATED_ADJUDICATED_GOLD"
    )
    assert row.evidence_code_precision == 1.0
    assert row.evidence_code_recall == 1.0
    assert row.evidence_code_f1 == 1.0
    assert row.fabricated_evidence_rate == 0.0


def test_annotation_rejects_observation_code_pair_that_never_cooccurred() -> None:
    case = generate_canonical_cases()[0]
    fact_by_id = {
        fact.fact_id: fact for fact in case.disclosure_facts
    }
    earlier = next(
        step for step in case.timeline if step.available_fact_ids
    )
    later = next(
        step
        for step in case.timeline
        if len(step.available_fact_ids) > len(earlier.available_fact_ids)
    )
    later_only = next(
        fact_id
        for fact_id in later.available_fact_ids
        if fact_id not in earlier.available_fact_ids
    )
    invalid = EvidenceRef(
        observation_id=earlier.observation_id,
        evidence_code=fact_by_id[later_only].evidence_code,
    )
    annotation = _annotation_for_step(
        case,
        later.step_id,
        evidence=(invalid,),
    )
    batch = AnnotationBatchV1(
        manual_version="test-v1",
        batch_id="invalid-pair",
        annotations=(annotation,),
    )

    with pytest.raises(ValueError, match="observation/code pairs"):
        validate_annotation_batch(batch, (case,))


def _selected_case(run_dir) -> CaseSpecV1:
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    cases = load_benchmark_cases(seed=manifest["dataset_seed"])
    selected = manifest["selected_case_ids"][0]
    return next(case for case in cases if case.case_id == selected)


def _write_gold(
    path,
    case: CaseSpecV1,
    *,
    evidence_by_step: dict[int, tuple[EvidenceRef, ...]],
) -> None:
    batch = AnnotationBatchV1(
        manual_version="test-v1",
        batch_id="adjudicated-gold",
        annotations=tuple(
            _annotation_for_step(
                case,
                step.step_id,
                evidence=evidence_by_step.get(step.step_id, ()),
            )
            for step in case.timeline
        ),
    )
    path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")


def _annotation_for_step(
    case: CaseSpecV1,
    step_id: int,
    *,
    evidence: tuple[EvidenceRef, ...],
) -> NodeAnnotationV1:
    step = case.timeline[step_id]
    return NodeAnnotationV1(
        case_id=case.case_id,
        step_id=step_id,
        annotator_id="expert-adjudicator",
        role=AnnotationRole.ADJUDICATOR,
        risk_level=RiskLevel.L0,
        confidence=1.0,
        evidence=evidence,
        acceptable_actions=step.acceptable_actions,
        preferred_action=step.preferred_action,
        prohibited_actions=step.prohibited_actions,
        direct_safety_check_required=False,
    )
