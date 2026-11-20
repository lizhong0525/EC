import pytest

from ecocompanion.benchmark.annotations import (
    AnnotationBatchV1,
    NodeAnnotationV1,
    annotation_agreement,
    find_adjudication_nodes,
    validate_annotation_batch,
)
from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.domain.enums import Action, EscalationTarget, RiskLevel
from ecocompanion.domain.schemas import EvidenceRef


def _annotation(
    *,
    case_id: str,
    step_id: int,
    observation_id: str,
    evidence_code: str,
    annotator_id: str,
    risk_level: RiskLevel = RiskLevel.L2,
    preferred_action: Action = Action.TRUSTED_ADULT_BRIDGE,
) -> NodeAnnotationV1:
    acceptable = (
        preferred_action,
        Action.CLARIFY_SYMPTOMS,
    )
    return NodeAnnotationV1(
        case_id=case_id,
        step_id=step_id,
        annotator_id=annotator_id,
        risk_level=risk_level,
        confidence=0.8,
        evidence=(
            EvidenceRef(
                observation_id=observation_id,
                evidence_code=evidence_code,
            ),
        ),
        acceptable_actions=acceptable,
        preferred_action=preferred_action,
        prohibited_actions=(Action.LOW_RISK_COPING,),
        direct_safety_check_required=risk_level in {RiskLevel.L3, RiskLevel.L4},
        referral_target=EscalationTarget.TRUSTED_ADULT,
        earliest_referral_step=step_id,
        optimal_referral_start_step=step_id,
        optimal_referral_end_step=step_id,
        latest_safe_referral_step=step_id,
        reason_codes=("VISIBLE_LONGITUDINAL_SIGNAL",),
    )


def test_annotation_batch_validates_only_observable_evidence() -> None:
    cases = generate_canonical_cases()
    case = cases[-1]
    step = case.timeline[-1]
    fact_by_id = {fact.fact_id: fact for fact in case.disclosure_facts}
    evidence_code = fact_by_id[step.available_fact_ids[0]].evidence_code
    batch = AnnotationBatchV1(
        manual_version="0.1.0-pilot",
        batch_id="pilot-a",
        annotations=(
            _annotation(
                case_id=case.case_id,
                step_id=step.step_id,
                observation_id=step.observation_id,
                evidence_code=evidence_code,
                annotator_id="expert-a",
            ),
        ),
    )

    assert validate_annotation_batch(batch, cases) is batch

    future = batch.annotations[0].model_copy(
        update={
            "step_id": 0,
            "evidence": (
                EvidenceRef(
                    observation_id=step.observation_id,
                    evidence_code=evidence_code,
                ),
            ),
            "earliest_referral_step": 0,
            "optimal_referral_start_step": 0,
            "optimal_referral_end_step": 0,
            "latest_safe_referral_step": 0,
        }
    )
    invalid = batch.model_copy(update={"annotations": (future,)})
    with pytest.raises(ValueError, match="future or unknown"):
        validate_annotation_batch(invalid, cases)


def test_annotation_contract_rejects_action_overlap_and_partial_window() -> None:
    with pytest.raises(ValueError, match="preferred_action"):
        NodeAnnotationV1(
            case_id="case-1",
            step_id=0,
            annotator_id="expert-a",
            risk_level=RiskLevel.L1,
            confidence=0.6,
            acceptable_actions=(Action.CLARIFY_SYMPTOMS,),
            preferred_action=Action.SUPPORTIVE_LISTENING,
            direct_safety_check_required=False,
        )

    with pytest.raises(ValueError, match="all four"):
        NodeAnnotationV1(
            case_id="case-1",
            step_id=0,
            annotator_id="expert-a",
            risk_level=RiskLevel.L1,
            confidence=0.6,
            acceptable_actions=(Action.CLARIFY_SYMPTOMS,),
            preferred_action=Action.CLARIFY_SYMPTOMS,
            direct_safety_check_required=False,
            earliest_referral_step=0,
        )


def test_agreement_and_adjudication_queue_keep_high_risk_nodes() -> None:
    cases = generate_canonical_cases()
    case = cases[-1]
    step_a = case.timeline[-2]
    step_b = case.timeline[-1]
    fact_by_id = {fact.fact_id: fact for fact in case.disclosure_facts}

    def evidence_code(step_id: int) -> str:
        fact_id = case.timeline[step_id].available_fact_ids[0]
        return fact_by_id[fact_id].evidence_code

    left_annotations = (
        _annotation(
            case_id=case.case_id,
            step_id=step_a.step_id,
            observation_id=step_a.observation_id,
            evidence_code=evidence_code(step_a.step_id),
            annotator_id="expert-a",
            risk_level=RiskLevel.L3,
            preferred_action=Action.PROFESSIONAL_ASSESSMENT,
        ),
        _annotation(
            case_id=case.case_id,
            step_id=step_b.step_id,
            observation_id=step_b.observation_id,
            evidence_code=evidence_code(step_b.step_id),
            annotator_id="expert-a",
            risk_level=RiskLevel.L4,
            preferred_action=Action.URGENT_ESCALATION,
        ),
    )
    right_annotations = (
        left_annotations[0].model_copy(update={"annotator_id": "expert-b"}),
        left_annotations[1].model_copy(
            update={
                "annotator_id": "expert-b",
                "risk_level": RiskLevel.L3,
                "preferred_action": Action.PROFESSIONAL_ASSESSMENT,
                "acceptable_actions": (
                    Action.PROFESSIONAL_ASSESSMENT,
                    Action.CLARIFY_SYMPTOMS,
                ),
            }
        ),
    )
    left = AnnotationBatchV1(
        manual_version="0.1.0-pilot",
        batch_id="pilot-a",
        annotations=left_annotations,
    )
    right = AnnotationBatchV1(
        manual_version="0.1.0-pilot",
        batch_id="pilot-b",
        annotations=right_annotations,
    )

    report = annotation_agreement(left, right)
    queue = find_adjudication_nodes(left, right)

    assert report.paired_node_count == 2
    assert 0.0 <= report.action_set_mean_jaccard <= 1.0
    assert report.preferred_action_raw_agreement == 0.5
    assert report.adjudication_node_count == 2
    assert queue == (
        (case.case_id, step_a.step_id),
        (case.case_id, step_b.step_id),
    )
