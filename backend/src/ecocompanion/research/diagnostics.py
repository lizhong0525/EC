"""Frozen, offline single-factor experiments with replayable executed-stage artifacts."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ecocompanion.benchmark.policies import ECOSafeBenchmarkPolicy, ECOSafeFeatureConfig
from ecocompanion.core.policy import PolicyOutcome, PolicyThresholds
from ecocompanion.core.risk import AssessmentResult, RuleBasedRiskAssessor
from ecocompanion.domain.schemas import AgentDecisionV1, ObservationV1, ResponsePayload
from ecocompanion.experiments.io import (
    append_jsonl_model,
    read_jsonl_models,
    write_json_atomic,
)
from ecocompanion.research.annotation import AnnotationStore, dataset_digest, digest
from ecocompanion.research.contracts import Prediction, Protocol, Record, ResearchCase
from ecocompanion.research.evaluation import primary_clusters
from ecocompanion.research.fingerprint import runtime_manifest, source_fingerprint
from ecocompanion.research.identity import audit_visible_payload
from ecocompanion.research.measurement import (
    MEASUREMENT_VERSION,
    PredictionKey,
    anchored_gold,
    prediction_key,
    score_measurement,
)
from ecocompanion.research.review import ReplyRating, blind_reply_packet
from ecocompanion.research.workflows import require_private_output


class Condition(Record):
    name: str
    change: Literal[
        "none",
        "history_removed",
        "structured_ecology_removed",
        "uncertainty_disabled",
        "evidence_constraint_disabled",
        "output_audit_disabled",
    ]


CONDITIONS = (
    Condition(name="full", change="none"),
    Condition(name="no-history", change="history_removed"),
    Condition(name="no-structured-ecology", change="structured_ecology_removed"),
    Condition(name="no-uncertainty", change="uncertainty_disabled"),
    Condition(name="no-evidence-constraint", change="evidence_constraint_disabled"),
    Condition(name="no-output-audit", change="output_audit_disabled"),
)
DESIGN = {
    "hypotheses": [
        "History removal may change evidence and actions; no benefit direction assumed.",
        "Structured ecology removal may be inert when the existing assessor ignores it.",
        "Uncertainty/evidence/output-audit switches may be inactive on some inputs.",
    ],
    "primary_metrics": [
        "required_safety_action_recall",
        "prohibited_action_rate",
        "acceptable_action_rate",
    ],
    "diagnostic_metrics": [
        "all_emitted_actions_acceptable",
        "action_set_norm_conjunction",
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
    ],
    "controls": "same cases, node times, rule assessor, thresholds, templates, zero model calls",
    "comparisons": "full vs each single-factor condition; no cross-factor combination",
    "statistics": "paired provenance-cluster descriptive bootstrap; template sensitivity",
    "limitations": [
        "No causal mechanism claim from correlational stage errors.",
        "A changed observation is not rescored against a newly invented expert label.",
        "Ablated inputs use full-evidence Gold only as a lost-information diagnostic target.",
        "Unknown/missing information is not assumed normal or safe.",
        "No direct-vs-evidence-first LLM claim: true model experiments remain pending.",
        "No model chain-of-thought is requested or treated as ground-truth reasoning.",
    ],
}


class DiagnosticPlan(Record):
    schema_version: Literal["p0p1-plan-v1"] = "p0p1-plan-v1"
    measurement_version: Literal["ecotriage-measurement-v2.1.0"] = "ecotriage-measurement-v2.1.0"
    classification: Literal["OFFLINE_RULE_MECHANISM_STUDY"] = "OFFLINE_RULE_MECHANISM_STUDY"
    parent_protocol_digest: str
    protocol: Protocol
    conditions: tuple[Condition, ...] = CONDITIONS
    design_digest: str
    input_views_digest: str
    plan_digest: str

    @model_validator(mode="after")
    def validate_design(self) -> DiagnosticPlan:
        if self.conditions != CONDITIONS or self.design_digest != digest(DESIGN):
            raise ValueError("unsupported or altered single-factor design")
        if (
            self.protocol.execution != "local_rules"
            or self.protocol.policy_names != tuple(c.name for c in self.conditions)
            or self.protocol.repeats != 1
            or self.protocol.model_identity != "no-model"
        ):
            raise ValueError("this design executes deterministic rules, not model experiments")
        if self.plan_digest != digest(self.model_dump(mode="json", exclude={"plan_digest"})):
            raise ValueError("plan digest mismatch")
        return self


class StageArtifact(Record):
    policy: str
    case_id: str
    step_id: int = Field(ge=0)
    repeat: Literal[0] = 0
    plan_digest: str
    observation: ObservationV1
    observation_digest: str
    prediction_digest: str
    raw_assessment: dict[str, Any] | None
    effective_assessment: dict[str, Any] | None
    executed_plan: dict[str, Any] | None
    candidate_reply: dict[str, Any] | None
    emitted_reply: dict[str, Any] | None
    implementation: Literal["LOCAL_RULE_EXECUTION_NOT_MODEL_REASONING"] = (
        "LOCAL_RULE_EXECUTION_NOT_MODEL_REASONING"
    )


def observation_view(observation: ObservationV1, condition: Condition) -> ObservationV1:
    payload = observation.model_dump(mode="json")
    if condition.change == "history_removed":
        payload["visible_context"]["conversation_history"] = []
    elif condition.change == "structured_ecology_removed":
        payload["visible_context"]["recent_events"] = []
        payload["visible_context"]["relationship_cues"] = []
        payload["visible_profile"]["disclosed_background"] = []
    result = ObservationV1.model_validate(payload)
    audit_visible_payload(result.model_dump_json())
    return result


def input_views(
    cases: tuple[ResearchCase, ...],
) -> dict[tuple[str, str, int], ObservationV1]:
    return {
        (condition.name, case.case_id, observation.step_id): observation_view(
            observation, condition
        )
        for condition in CONDITIONS
        for case in cases
        for observation in case.observations
    }


def views_digest(views: dict[tuple[str, str, int], ObservationV1]) -> str:
    return digest(
        [[list(key), value.model_dump(mode="json")] for key, value in sorted(views.items())]
    )


def freeze_plan(packet: Path, output: Path) -> DiagnosticPlan:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    store.audit()
    cases = store.cases()
    require_private_output(cases, packet, output)
    parent = Protocol.model_validate_json((packet / "protocol.json").read_text(encoding="utf-8"))
    if parent.dataset_digest != dataset_digest(cases):
        raise ValueError("parent packet digest mismatch")
    source = source_fingerprint()
    protocol = Protocol(
        dataset_digest=dataset_digest(cases),
        policy_names=tuple(c.name for c in CONDITIONS),
        model_identity="no-model",
        prompt_digest=digest(
            {
                "policy": "ECOSafeBenchmarkPolicy",
                "source": source,
                "thresholds": PolicyThresholds().__dict__,
                "design": DESIGN,
            }
        ),
        source_digest=source,
        runtime_digest=digest(runtime_manifest()),
    )
    body = {
        "schema_version": "p0p1-plan-v1",
        "measurement_version": MEASUREMENT_VERSION,
        "classification": "OFFLINE_RULE_MECHANISM_STUDY",
        "parent_protocol_digest": digest(parent.model_dump(mode="json")),
        "protocol": protocol.model_dump(mode="json"),
        "conditions": [c.model_dump(mode="json") for c in CONDITIONS],
        "design_digest": digest(DESIGN),
        "input_views_digest": views_digest(input_views(cases)),
    }
    plan = DiagnosticPlan.model_validate({**body, "plan_digest": digest(body)})
    write_json_atomic(output, plan, overwrite=False)
    return plan


def _assessment_payload(value: AssessmentResult) -> dict[str, Any]:
    return {
        "estimate": value.estimate.model_dump(mode="json"),
        "uncertainty": value.uncertainty,
        "evidence": [e.model_dump(mode="json") for e in value.evidence],
        "reason_codes": list(value.reason_codes),
    }


@dataclass(frozen=True)
class InstrumentedPolicy(ECOSafeBenchmarkPolicy):
    """Observe the actual existing hooks; do not reimplement the action policy."""

    artifacts: dict[str, Any] = field(default_factory=dict, compare=False)

    def _transform_assessment(self, assessment: AssessmentResult) -> AssessmentResult:
        effective = super()._transform_assessment(assessment)
        self.artifacts["raw_assessment"] = _assessment_payload(assessment)
        self.artifacts["effective_assessment"] = _assessment_payload(effective)
        return effective

    def _enforce_output(self, response: ResponsePayload, outcome: PolicyOutcome) -> ResponsePayload:
        emitted = super()._enforce_output(response, outcome)
        self.artifacts["executed_plan"] = outcome.plan.model_dump(mode="json")
        self.artifacts["candidate_reply"] = response.model_dump(mode="json")
        self.artifacts["emitted_reply"] = emitted.model_dump(mode="json")
        return emitted


def execute_node(
    condition: Condition,
    observation: ObservationV1,
) -> tuple[AgentDecisionV1, dict[str, Any]]:
    policy = InstrumentedPolicy(
        name=condition.name,
        assessor=RuleBasedRiskAssessor(),
        features=ECOSafeFeatureConfig(
            uncertainty=condition.change != "uncertainty_disabled",
            evidence_constraint=condition.change != "evidence_constraint_disabled",
            output_safety_audit=condition.change != "output_audit_disabled",
        ),
    )
    decision = policy.decide(observation)
    return decision, policy.artifacts


def _validated_context(
    packet: Path,
    plan_path: Path,
) -> tuple[tuple[ResearchCase, ...], DiagnosticPlan, dict[tuple[str, str, int], ObservationV1]]:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    store.audit()
    cases = store.cases()
    plan = DiagnosticPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    parent = Protocol.model_validate_json((packet / "protocol.json").read_text(encoding="utf-8"))
    if plan.protocol.dataset_digest != dataset_digest(
        cases
    ) or plan.parent_protocol_digest != digest(parent.model_dump(mode="json")):
        raise ValueError("plan is bound to another packet")
    if (
        plan.protocol.source_digest != source_fingerprint()
        or plan.protocol.runtime_digest != digest(runtime_manifest())
    ):
        raise ValueError("source/runtime drift since plan freeze; create a new plan")
    views = input_views(cases)
    if views_digest(views) != plan.input_views_digest:
        raise ValueError("frozen input views changed")
    return cases, plan, views


def run_diagnostics(packet: Path, plan_path: Path, output: Path) -> dict[str, Any]:
    cases, plan, views = _validated_context(packet, plan_path)
    require_private_output(cases, packet, output)
    if output.exists():
        raise FileExistsError("choose a new diagnostic run; historical results are immutable")
    output.mkdir(parents=True)
    write_json_atomic(output / "plan.json", plan, overwrite=False)
    write_json_atomic(output / "design.json", DESIGN, overwrite=False)
    write_json_atomic(output / "runtime.json", runtime_manifest(), overwrite=False)
    write_json_atomic(
        output / "started.json",
        {
            "status": "STARTED",
            "plan_digest": plan.plan_digest,
            "started_at": datetime.now(UTC).isoformat(),
            "expected_nodes": len(views),
            "execution": "local_rules",
        },
        overwrite=False,
    )
    for condition in plan.conditions:
        for case in cases:
            for observation in case.observations:
                view = views[(condition.name, case.case_id, observation.step_id)]
                artifacts: dict[str, Any] = {}
                started = time.perf_counter()
                try:
                    decision, artifacts = execute_node(condition, view)
                    prediction = Prediction(
                        policy=condition.name,
                        case_id=case.case_id,
                        step_id=view.step_id,
                        status="ok",
                        decision=decision,
                        cost_cny=0,
                        protocol_digest=digest(plan.protocol.model_dump(mode="json")),
                        latency_seconds=time.perf_counter() - started,
                    )
                except Exception as exc:
                    prediction = Prediction(
                        policy=condition.name,
                        case_id=case.case_id,
                        step_id=view.step_id,
                        status="error",
                        error_code=type(exc).__name__,
                        cost_cny=0,
                        protocol_digest=digest(plan.protocol.model_dump(mode="json")),
                        latency_seconds=time.perf_counter() - started,
                    )
                stage = StageArtifact(
                    policy=condition.name,
                    case_id=case.case_id,
                    step_id=view.step_id,
                    plan_digest=plan.plan_digest,
                    observation=view,
                    observation_digest=digest(view.model_dump(mode="json")),
                    prediction_digest=digest(prediction.model_dump(mode="json")),
                    raw_assessment=artifacts.get("raw_assessment"),
                    effective_assessment=artifacts.get("effective_assessment"),
                    executed_plan=artifacts.get("executed_plan"),
                    candidate_reply=artifacts.get("candidate_reply"),
                    emitted_reply=artifacts.get("emitted_reply"),
                )
                append_jsonl_model(output / "predictions.jsonl", prediction)
                append_jsonl_model(output / "stages.jsonl", stage)
    report = report_diagnostics(packet, plan_path, output, output / "report.json")
    write_json_atomic(
        output / "completed.json",
        {
            "status": "EXECUTED_NOT_RESEARCH_VALIDATED",
            "plan_digest": plan.plan_digest,
            "completed_at": datetime.now(UTC).isoformat(),
            "report_digest": digest(report),
            "stage_replay": report["stage_replay"],
        },
        overwrite=False,
    )
    return report


def report_diagnostics(
    packet: Path,
    plan_path: Path,
    run: Path,
    output: Path,
    *,
    gold: Path | None = None,
) -> dict[str, Any]:
    cases, plan, views = _validated_context(packet, plan_path)
    require_private_output(cases, packet, output)
    if output.exists():
        raise FileExistsError("choose a new report path")
    if DiagnosticPlan.model_validate_json((run / "plan.json").read_text(encoding="utf-8")) != plan:
        raise ValueError("run plan differs from requested frozen plan")
    import json

    started = json.loads((run / "started.json").read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(started["started_at"])
    if (
        started.get("plan_digest") != plan.plan_digest
        or started.get("expected_nodes") != len(views)
        or started_at.tzinfo is None
        or started_at < plan.protocol.created_at
    ):
        raise ValueError("run start is not bound to a previously frozen plan")
    for path in (run / "predictions.jsonl", run / "stages.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    predictions = read_jsonl_models(run / "predictions.jsonl", Prediction)
    stages = read_jsonl_models(run / "stages.jsonl", StageArtifact)
    by_key = {prediction_key(p): p for p in predictions}
    traces: dict[PredictionKey, StageArtifact] = {
        (s.policy, s.case_id, s.step_id, s.repeat): s for s in stages
    }
    if len(traces) != len(stages) or len(by_key) != len(predictions):
        raise ValueError("duplicate prediction or stage")
    if set(by_key) != set(traces):
        raise ValueError("predictions and executed-stage coverage differ")
    replay_failures = []
    for key, prediction in by_key.items():
        if key[:3] not in views or key[3] != 0:
            raise ValueError("unknown stage node")
        stage = traces[key]
        view = views[key[:3]]
        if (
            stage.plan_digest != plan.plan_digest
            or stage.observation != view
            or stage.observation_digest != digest(view.model_dump(mode="json"))
            or stage.prediction_digest != digest(prediction.model_dump(mode="json"))
            or prediction.protocol_digest != digest(plan.protocol.model_dump(mode="json"))
            or prediction.calls
            or prediction.retries
            or prediction.input_tokens
            or prediction.output_tokens
            or prediction.cost_cny != 0
        ):
            raise ValueError("input, plan, prediction or zero-call accounting mismatch")
        condition = next(c for c in plan.conditions if c.name == key[0])
        try:
            replay, artifacts = execute_node(condition, view)
        except Exception as exc:
            if prediction.status != "error" or prediction.error_code != type(exc).__name__:
                replay_failures.append(list(key))
            continue
        if (
            prediction.status != "ok"
            or prediction.decision != replay
            or any(getattr(stage, name) != value for name, value in artifacts.items())
        ):
            replay_failures.append(list(key))
    if replay_failures:
        raise ValueError(f"executed-stage replay failed for {len(replay_failures)} nodes")
    document = anchored_gold(packet, gold)
    report = score_measurement(
        cases, plan.protocol, predictions, gold_document=document, views=views
    )
    changes = []
    for condition in plan.conditions[1:]:
        paired, changed_actions, changed_evidence, changed_inputs = 0, 0, 0, 0
        for case in cases:
            for observation in case.observations:
                base = by_key.get(("full", case.case_id, observation.step_id, 0))
                other = by_key.get((condition.name, case.case_id, observation.step_id, 0))
                if base is None or other is None or not base.decision or not other.decision:
                    continue
                paired += 1
                changed_actions += base.decision.decision != other.decision.decision
                changed_evidence += base.decision.evidence != other.decision.evidence
                changed_inputs += (
                    views[("full", case.case_id, observation.step_id)]
                    != views[(condition.name, case.case_id, observation.step_id)]
                )
        changes.append(
            {
                "condition": condition.name,
                "changed_factor": condition.change,
                "complete_output_pairs": paired,
                "changed_inputs": changed_inputs,
                "changed_decision_plans": changed_actions,
                "changed_evidence": changed_evidence,
                "benefit": "UNASSESSED_WITHOUT_APPLICABLE_EXPERT_REFERENCE",
                "interpretation": "activation/count diagnostic, not accuracy or causal proof",
            }
        )
    report.update(
        {
            "plan_digest": plan.plan_digest,
            "execution_started_at": started["started_at"],
            "classification": plan.classification,
            "stage_replay": {
                "status": "PASS" if len(stages) == len(views) else "INCOMPLETE",
                "verified_nodes": len(stages),
                "expected_nodes": len(views),
                "missing_nodes": len(views) - len(stages),
                "warning": "local deterministic replay; hashes do not defeat a malicious curator",
            },
            "input_ablation_reference_policy": {
                "full": "expert action norms on original evidence, if applicable Gold exists",
                "ablated": (
                    "full-evidence target diagnostic only; not normative correctness under "
                    "reduced information. New blinded labels needed for that claim."
                ),
            },
            "stage_completion": {
                "implementation": "LOCAL_RULE_EXECUTION_NOT_MODEL_REASONING",
                "recorded_predictions": dict(Counter(p.status for p in predictions)),
                "complete_stage_artifacts": sum(
                    all(
                        getattr(s, name) is not None
                        for name in (
                            "raw_assessment",
                            "effective_assessment",
                            "executed_plan",
                            "candidate_reply",
                            "emitted_reply",
                        )
                    )
                    for s in stages
                ),
            },
            "mechanism_activation": changes,
            "declared_primary_clusters": len(set(primary_clusters(cases).values())),
            "human_evidence_entailment_review": "PENDING",
            "missing_information_quality": "UNMEASURED_EXISTING_RULE_POLICY_HAS_NO_SUCH_OUTPUT",
            "direct_vs_evidence_first_model_experiment": "NOT_RUN_REQUIRES_QUALIFIED_MODEL",
            "design": DESIGN,
            "trace_digest": digest([s.model_dump(mode="json") for s in stages]),
            "predictions_digest": digest([p.model_dump(mode="json") for p in predictions]),
        }
    )
    report["paired_primary_comparisons"] = [
        row for row in report["paired_primary_comparisons"] if row["left"] == "full"
    ]
    # Do not expose reduced-input scores as though they were matched expert norms.
    if document:
        for condition in plan.conditions:
            if condition.change in {"history_removed", "structured_ecology_removed"}:
                item = report["policies"][condition.name]
                item["full_evidence_target_diagnostics"] = item.pop("expert_action_norms")
                item["expert_action_norms"] = {
                    "status": "UNAVAILABLE_REQUIRES_INPUT_MATCHED_GOLD",
                    "metrics": None,
                }
                item["action_set_diagnostics"]["reference_scope"] = (
                    "FULL_EVIDENCE_TARGET_NOT_INPUT_MATCHED_NORMS"
                )
                item["action_set_diagnostics"]["status"] = "FULL_EVIDENCE_TARGET_DIAGNOSTIC_ONLY"
        for comparison in report["paired_primary_comparisons"]:
            if {comparison["left"], comparison["right"]} & {"no-history", "no-structured-ecology"}:
                comparison["reference_scope"] = "FULL_EVIDENCE_TARGET_DIAGNOSTIC_ONLY"
    write_json_atomic(output, report, overwrite=False)
    return report


def diagnostic_replies(
    packet: Path,
    plan_path: Path,
    run: Path,
    output: Path,
) -> dict[str, Any]:
    """Reuse anonymous review with each condition's ACTUAL input, not full future context."""
    cases, plan, views = _validated_context(packet, plan_path)
    require_private_output(cases, packet, output)
    if output.exists():
        raise FileExistsError("choose a new anonymous diagnostic reply packet")
    report_diagnostics(packet, plan_path, run, output / "private" / "verification.json")
    result = blind_reply_packet(
        cases,
        read_jsonl_models(run / "predictions.jsonl", Prediction),
        output / "review",
        seed=plan.protocol.seeds[0],
        views=views,
    )
    write_json_atomic(
        output / "review" / "ReplyRating.schema.json",
        ReplyRating.model_json_schema(),
        overwrite=False,
    )
    write_json_atomic(
        output / "review" / "scope.json",
        {
            **result,
            "plan_digest": plan.plan_digest,
            "input_contract": "each candidate uses its own executed observation",
            "blinding_limit": "input differences may reveal the ablation, model names hidden",
            "reviewer_separation": (
                "Model-reply reviewers must not contaminate independent normative annotation."
            ),
            "evidence_entailment": "PENDING_SEPARATE_HUMAN_REVIEW_NOT_SCORED_BY_REPLY_SCHEMA",
            "ratings": "NOT_COLLECTED",
        },
        overwrite=False,
    )
    return result
