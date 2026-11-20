"""P0/P1 tests use artificial expectations, not real human annotations."""

from __future__ import annotations

import json

import pytest
from test_research_v2 import frozen_gold, judgment
from typer.testing import CliRunner

from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.policies import AlwaysListenPolicy
from ecocompanion.domain.enums import Action, RiskLevel
from ecocompanion.domain.schemas import EvidenceRef
from ecocompanion.experiments.io import read_jsonl_models, write_json_atomic
from ecocompanion.research.annotation import AnnotationStore, dataset_digest, derive_windows, digest
from ecocompanion.research.cli import app
from ecocompanion.research.contracts import Prediction, Protocol
from ecocompanion.research.diagnostics import (
    CONDITIONS,
    DiagnosticPlan,
    StageArtifact,
    diagnostic_replies,
    execute_node,
    freeze_plan,
    observation_view,
    report_diagnostics,
    run_diagnostics,
)
from ecocompanion.research.fingerprint import runtime_manifest, source_fingerprint
from ecocompanion.research.identity import IdentityMap, observable_case
from ecocompanion.research.measurement import (
    action_set_metrics,
    anchored_gold,
    contract_issues,
    score_measurement,
)
from ecocompanion.research.validity import audit_validity, challenge_suite


@pytest.fixture
def case():
    value = observable_case(generate_canonical_cases()[0], IdentityMap())
    return value.model_copy(update={"observations": value.observations[:6]})


@pytest.fixture
def packet(tmp_path, case):
    packet = tmp_path / "packet"
    AnnotationStore.create(
        packet / "private" / "annotations.sqlite3", (case,), manual_digest="TEST_ONLY"
    )
    protocol = Protocol(
        dataset_digest=dataset_digest((case,)),
        policy_names=("always-listen",),
        model_identity="no-model",
        prompt_digest="TEST_ONLY",
        source_digest=source_fingerprint(),
        runtime_digest=digest(runtime_manifest()),
        bootstrap_resamples=100,
    )
    write_json_atomic(packet / "protocol.json", protocol, overwrite=False)
    return packet


def prediction(case, **changes):
    observation = case.observations[-1]
    base = Prediction(
        policy="always-listen",
        case_id=case.case_id,
        step_id=observation.step_id,
        status="ok",
        decision=AlwaysListenPolicy().decide(observation),
        cost_cny=0,
    )
    return base.model_copy(update=changes)


def protocol(case):
    return Protocol(
        dataset_digest=dataset_digest((case,)),
        policy_names=("always-listen",),
        model_identity="no-model",
        prompt_digest="TEST_ONLY",
        bootstrap_resamples=100,
    )


def test_challenges_are_executable_and_never_gold():
    result = challenge_suite()
    assert result["status"] == "PASS"
    assert len(result["checks"]) >= 20
    assert all(c["observed"] == c["expected"] for c in result["checks"])
    assert result["classification"] == "PROGRAMMING_FIXTURES_NOT_EXPERT_OR_MODEL_RESULTS"


@pytest.mark.parametrize(
    "issue",
    [
        "INVISIBLE_EVIDENCE",
        "DUPLICATE_EVIDENCE",
        "DUPLICATE_ACTION",
        "EMPTY_EVIDENCE_CODE",
        "EMPTY_REPLY",
        "FORBIDDEN_FALLBACK",
        "DISALLOWED_ACTION",
    ],
)
def test_strict_contract_retains_failures_in_denominators(case, issue):
    item = prediction(case)
    decision = item.decision
    observation = case.observations[-1]
    if issue == "INVISIBLE_EVIDENCE":
        decision = decision.model_copy(
            update={"evidence": (EvidenceRef(observation_id="unknown", evidence_code="TEST"),)}
        )
    elif issue == "DUPLICATE_EVIDENCE":
        ref = EvidenceRef(
            observation_id=observation.current_observation.observation_id, evidence_code="TEST"
        )
        decision = decision.model_copy(update={"evidence": (ref, ref)})
    elif issue == "DUPLICATE_ACTION":
        decision = decision.model_copy(
            update={
                "decision": decision.decision.model_copy(
                    update={"secondary_actions": (decision.decision.primary_action,)}
                )
            }
        )
    elif issue == "EMPTY_EVIDENCE_CODE":
        decision = decision.model_copy(
            update={
                "evidence": (
                    EvidenceRef(
                        observation_id=observation.current_observation.observation_id,
                        evidence_code=" ",
                    ),
                )
            }
        )
    elif issue == "EMPTY_REPLY":
        decision = decision.model_copy(
            update={"response": decision.response.model_copy(update={"text": "   "})}
        )
    elif issue == "DISALLOWED_ACTION":
        observation = observation.model_copy(
            update={"allowed_primary_actions": (Action.CLARIFY_SYMPTOMS,)}
        )
    if issue == "FORBIDDEN_FALLBACK":
        item = item.model_copy(update={"status": "fallback"})
    item = item.model_copy(update={"decision": decision})
    assert issue in contract_issues(item, observation)
    views = {("always-listen", case.case_id, o.step_id): o for o in case.observations}
    views[("always-listen", case.case_id, observation.step_id)] = observation
    report = score_measurement((case,), protocol(case), [item], views=views)
    engineering = report["policies"]["always-listen"]["engineering"]
    assert engineering["strict_contract_failures"] == 1
    assert engineering["raw_success_rate"] == 0
    assert report["missing_results"] == 5
    assert report["strict_contract_issue_counts"][issue] == 1


def test_secondary_actions_cannot_hide_behind_valid_primary(case):
    item = prediction(case)
    item = item.model_copy(
        update={
            "decision": item.decision.model_copy(
                update={
                    "decision": item.decision.decision.model_copy(
                        update={
                            "secondary_actions": (Action.URGENT_ESCALATION,),
                        }
                    ),
                }
            )
        }
    )
    reference = judgment(case.observations[-1]).model_copy(
        update={"prohibited_actions": (Action.URGENT_ESCALATION,)}
    )
    result = action_set_metrics(item, reference)
    assert result["all_emitted_actions_acceptable"] == 0
    assert result["contains_unacceptable_action"] == 1
    assert result["action_set_norm_conjunction"] == 0


def test_gold_mutation_changes_stage_metrics_not_simulation(tmp_path, case):
    gold = frozen_gold(tmp_path, (case,))
    truth = {(case.case_id, o.step_id): (0.1, 0.2) for o in case.observations}
    before = score_measurement(
        (case,), protocol(case), [prediction(case)], gold_document=gold, simulation_truth=truth
    )
    changed = json.loads(json.dumps(gold))
    for row in changed["records"]:
        value = row["judgment"]
        value["acceptable_actions"] = [Action.CLARIFY_SYMPTOMS.value]
        value["preferred_action"] = Action.CLARIFY_SYMPTOMS.value
        value["prohibited_actions"] = [Action.SUPPORTIVE_LISTENING.value]
        row["independent_a"] = value
        row["independent_b"] = value
        row["adjudication"] = value
    changed["windows"] = derive_windows(changed["records"])
    changed["gold_digest"] = digest({k: v for k, v in changed.items() if k != "gold_digest"})
    after = score_measurement(
        (case,), protocol(case), [prediction(case)], gold_document=changed, simulation_truth=truth
    )
    a, b = (r["policies"]["always-listen"] for r in (before, after))
    assert a["simulation_state_recovery"] == b["simulation_state_recovery"]
    assert a["action_set_diagnostics"] != b["action_set_diagnostics"]
    assert after["node_diagnostics"][-1]["action_norms"]["all_emitted_actions_acceptable"] == 0


def test_missing_outputs_are_not_good_action_sets(tmp_path, case):
    gold = frozen_gold(tmp_path, (case,))
    result = score_measurement((case,), protocol(case), [], gold_document=gold)
    summary = result["policies"]["always-listen"]["action_set_diagnostics"]
    assert summary["metrics"]["all_emitted_actions_acceptable"]["estimate"] == 0
    assert summary["metrics"]["action_set_norm_conjunction"]["estimate"] == 0
    assert summary["metrics"]["all_emitted_actions_acceptable"]["applicable_node_repeats"] == 6
    assert result["missing_results"] == 6


def test_reduced_input_evidence_checked_against_actual_view(case):
    item = prediction(case)
    old = case.observations[0].current_observation.observation_id
    item = item.model_copy(
        update={
            "decision": item.decision.model_copy(
                update={
                    "evidence": (EvidenceRef(observation_id=old, evidence_code="TEST"),),
                }
            )
        }
    )
    condition = CONDITIONS[1]
    views = {
        ("always-listen", case.case_id, o.step_id): observation_view(o, condition)
        for o in case.observations
    }
    result = score_measurement((case,), protocol(case), [item], views=views)
    assert result["strict_contract_issue_counts"]["INVISIBLE_EVIDENCE"] == 1


def test_single_factor_views_preserve_unrelated_fields(case):
    observation = case.observations[-1].model_copy(
        update={
            "visible_context": case.observations[-1].visible_context.model_copy(
                update={
                    "recent_events": ("TEST_EVENT",),
                    "relationship_cues": ("TEST_CUE",),
                }
            ),
        }
    )
    full = observation_view(observation, CONDITIONS[0])
    history = observation_view(observation, CONDITIONS[1])
    ecology = observation_view(observation, CONDITIONS[2])
    assert full == observation
    assert not history.visible_context.conversation_history
    assert history.visible_context.recent_events == full.visible_context.recent_events
    assert ecology.visible_context.conversation_history == full.visible_context.conversation_history
    assert not ecology.visible_context.recent_events
    assert not ecology.visible_context.relationship_cues
    assert ecology.current_observation == full.current_observation
    assert ecology.allowed_primary_actions == full.allowed_primary_actions


def test_instrumentation_observes_existing_implementation(case):
    observation = case.observations[-1]
    full, trace = execute_node(CONDITIONS[0], observation)
    assert trace["executed_plan"] == full.decision.model_dump(mode="json")
    assert trace["emitted_reply"] == full.response.model_dump(mode="json")
    assert trace["effective_assessment"]["evidence"] == [
        e.model_dump(mode="json") for e in full.evidence
    ]
    no_uncertainty, changed = execute_node(CONDITIONS[3], observation)
    assert changed["effective_assessment"]["uncertainty"] == 0
    assert changed["raw_assessment"] == trace["raw_assessment"]
    assert no_uncertainty.uncertainty == 0


def test_plan_is_frozen_before_execution_and_rejects_drift(packet, tmp_path):
    path = tmp_path / "plan.json"
    plan = freeze_plan(packet, path)
    assert plan.protocol.model_identity == "no-model"
    assert plan.protocol.max_cost_cny == 0
    changed = plan.model_dump(mode="json")
    changed["conditions"][1]["change"] = "uncertainty_disabled"
    changed["plan_digest"] = digest({k: v for k, v in changed.items() if k != "plan_digest"})
    with pytest.raises(ValueError, match="single-factor"):
        DiagnosticPlan.model_validate(changed)
    changed = plan.model_dump(mode="json")
    changed["protocol"]["source_digest"] = "different"
    changed["plan_digest"] = digest({k: v for k, v in changed.items() if k != "plan_digest"})
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="source/runtime"):
        run_diagnostics(packet, path, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_complete_run_replays_without_gold_or_paid_calls(packet, tmp_path):
    plan = tmp_path / "plan.json"
    freeze_plan(packet, plan)
    run = tmp_path / "run"
    result = run_diagnostics(packet, plan, run)
    assert result["stage_replay"]["verified_nodes"] == 36
    assert result["stage_replay"]["missing_nodes"] == 0
    assert result["stage_completion"]["complete_stage_artifacts"] == 36
    assert result["declared_primary_clusters"] == 1
    assert all(
        item["expert_action_norms"]["status"] == "UNAVAILABLE_NO_APPLICABLE_GOLD"
        for item in result["policies"].values()
    )
    predictions = read_jsonl_models(run / "predictions.jsonl", Prediction)
    assert len(predictions) == 36
    assert all(p.calls == p.input_tokens == p.output_tokens == 0 for p in predictions)
    stages = read_jsonl_models(run / "stages.jsonl", StageArtifact)
    assert len(stages) == 36
    assert all(s.plan_digest == result["plan_digest"] for s in stages)
    with pytest.raises(FileExistsError):
        run_diagnostics(packet, plan, run)
    with pytest.raises(FileExistsError):
        report_diagnostics(packet, plan, run, run / "report.json")


@pytest.mark.parametrize("tamper", ["input", "prediction", "stage", "missing-stage", "budget"])
def test_report_rejects_fabricated_execution(packet, tmp_path, tamper):
    plan = tmp_path / "plan.json"
    freeze_plan(packet, plan)
    run = tmp_path / "run"
    run_diagnostics(packet, plan, run)
    stages = [
        json.loads(line) for line in (run / "stages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    predictions = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if tamper == "input":
        stages[0]["observation"]["current_observation"]["message"] = "TAMPERED"
        stages[0]["observation_digest"] = digest(stages[0]["observation"])
    elif tamper == "prediction":
        predictions[0]["decision"]["response"]["text"] = "FABRICATED"
        stages[0]["prediction_digest"] = digest(predictions[0])
    elif tamper == "stage":
        stages[0]["raw_assessment"]["uncertainty"] = 0.1234
    elif tamper == "missing-stage":
        stages.pop()
    else:
        predictions[0]["calls"] = 1
        stages[0]["prediction_digest"] = digest(predictions[0])
    (run / "stages.jsonl").write_text("\n".join(json.dumps(s) for s in stages), encoding="utf-8")
    (run / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in predictions), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        report_diagnostics(packet, plan, run, run / "must-not-exist.json")
    assert not (run / "must-not-exist.json").exists()


def test_unanchored_gold_is_rejected(packet, tmp_path):
    gold = tmp_path / "forged.json"
    gold.write_text('{"status":"GOLD","gold_digest":"TEST_NOT_REAL"}', encoding="utf-8")
    with pytest.raises(ValueError, match="audit trail"):
        anchored_gold(packet, gold)


def test_p0_audit_preserves_empty_human_store(packet):
    result = audit_validity(packet)
    assert result["challenge_suite"]["status"] == "PASS"
    assert result["label_counts"] == {"UNLABELED": 6}
    assert result["expert_identity_attestation_count"] == 0
    assert result["publication_readiness"] == "NOT_ESTABLISHED"
    stress = result["packet_stress_test"]
    assert len(stress["predictions"]) == 48
    assert (
        stress["report"]["policies"]["fabricated-evidence"]["engineering"][
            "strict_contract_failures"
        ]
        == 6
    )
    assert (
        stress["report"]["policies"]["fabricated-evidence"]["engineering"][
            "fabricated_reference_node_count"
        ]
        == 6
    )
    assert result["expert_gold_status"] == "NOT_FROZEN"


def test_cli_p0_p1_chain(packet, tmp_path):
    runner = CliRunner()
    plan, run = tmp_path / "plan.json", tmp_path / "run"
    for arguments in (
        ["validity-audit", "--packet", str(packet), "--output", str(tmp_path / "audit.json")],
        ["diagnostic-plan", "--packet", str(packet), "--output", str(plan)],
        ["diagnostic-run", "--packet", str(packet), "--plan", str(plan), "--output", str(run)],
        [
            "diagnostic-report",
            "--packet",
            str(packet),
            "--plan",
            str(plan),
            "--run",
            str(run),
            "--output",
            str(run / "rechecked.json"),
        ],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
    assert (run / "rechecked.json").exists()


def test_multi_policy_gold_serialization_and_input_reference_scope(tmp_path, case):
    """TEMPORARY fabricated identities exercise plumbing only, not human study results."""
    packet = tmp_path / "TEST_ONLY_PACKET"
    store, tokens = AnnotationStore.create(
        packet / "private" / "annotations.sqlite3", (case,), manual_digest="TEST_ONLY"
    )
    write_json_atomic(packet / "protocol.json", protocol(case), overwrite=False)
    for role in ("expert-a", "expert-b", "adjudicator"):
        store.attest(
            role,
            reference="TEST_NOT_REAL",
            administrator="TEST_NOT_REAL",
            person_id=f"TEST_PERSON_{role}",
            qualification="TEST_NOT_REAL",
        )
    for role in ("expert-a", "expert-b"):
        while (task := store.next_task(tokens[role]))["status"] == "ready":
            store.submit(
                tokens[role],
                task["task_id"],
                judgment(case.observations[task["step_id"]], list(RiskLevel)[task["step_id"]]),
            )
    while (task := store.next_task(tokens["adjudicator"]))["status"] == "ready":
        from ecocompanion.research.contracts import Judgment

        store.submit(
            tokens["adjudicator"],
            task["task_id"],
            Judgment.model_validate(task["independent_labels"]["a"]),
        )
    gold_path = packet / "private" / "TEST_ONLY_GOLD.json"
    store.freeze(gold_path)
    plan, run = tmp_path / "plan.json", tmp_path / "run"
    freeze_plan(packet, plan)
    run_diagnostics(packet, plan, run)
    result = report_diagnostics(packet, plan, run, run / "gold-report.json", gold=gold_path)
    assert result["policies"]["full"]["expert_action_norms"]["status"] == "GOLD"
    assert result["policies"]["no-history"]["expert_action_norms"]["status"] == (
        "UNAVAILABLE_REQUIRES_INPUT_MATCHED_GOLD"
    )
    assert result["policies"]["no-history"]["action_set_diagnostics"]["status"] == (
        "FULL_EVIDENCE_TARGET_DIAGNOSTIC_ONLY"
    )
    assert result["paired_primary_comparisons"]
    assert all(r["left"] == "full" for r in result["paired_primary_comparisons"])
    assert result["paired_primary_comparisons"][0]["difference_ci"]["cluster_count"] == 1
    assert (
        result["policies"]["full"]["expert_action_norms"]["metrics"]["acceptable_action_rate"][
            "clusters"
        ]
        == 1
    )


def test_partial_records_are_reported_incomplete_not_passed(packet, tmp_path):
    plan, run = tmp_path / "plan.json", tmp_path / "run"
    freeze_plan(packet, plan)
    run_diagnostics(packet, plan, run)
    for name in ("predictions.jsonl", "stages.jsonl"):
        path = run / name
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]), encoding="utf-8")
    result = report_diagnostics(packet, plan, run, run / "partial.json")
    assert result["stage_replay"]["status"] == "INCOMPLETE"
    assert result["missing_results"] == 1
    assert result["stage_replay"]["missing_nodes"] == 1


def test_normalization_does_not_erase_raw_fabricated_evidence(tmp_path, case):
    gold = frozen_gold(tmp_path, (case,))
    item = prediction(case)
    item = item.model_copy(
        update={
            "decision": item.decision.model_copy(
                update={
                    "evidence": (
                        EvidenceRef(observation_id="NONEXISTENT", evidence_code="TEST_ONLY"),
                    ),
                }
            )
        }
    )
    result = score_measurement((case,), protocol(case), [item], gold_document=gold)
    policy = result["policies"]["always-listen"]
    assert policy["expert_action_norms"]["metrics"]["fabricated_evidence_rate"]["estimate"] == 1
    assert policy["engineering"]["fabricated_reference_node_count"] == 1
    assert policy["engineering"]["raw_success_rate"] == 0


def test_anonymous_replies_use_condition_matched_inputs(packet, tmp_path):
    plan, run = tmp_path / "plan.json", tmp_path / "run"
    freeze_plan(packet, plan)
    run_diagnostics(packet, plan, run)
    output = tmp_path / "reply-review"
    result = diagnostic_replies(packet, plan, run, output)
    assert result["tasks"] == 36
    tasks = json.loads((output / "review" / "reply-tasks.json").read_text(encoding="utf-8"))
    mapping = json.loads(
        (output / "review" / "private" / "mapping.json").read_text(encoding="utf-8")
    )["mapping"]
    for task in tasks["tasks"]:
        assert "policy" not in task
        if mapping[task["candidate_id"]]["policy"] == "no-history":
            assert task["observation"]["visible_context"]["conversation_history"] == []
    assert (output / "review" / "ReplyRating.schema.json").exists()


def test_blind_packet_outputs_stay_private(tmp_path, case):
    from ecocompanion.research.contracts import ResearchCase

    value = case.model_dump(mode="json")
    value["split"] = "independent_blind"
    value["provenance"].update(
        {
            "origin": "independent_author",
            "review_status": "reviewed",
            "review_reference": "TEST_ONLY_NOT_REAL",
        }
    )
    independent = ResearchCase.model_validate(value)
    packet = tmp_path / "TEST_BLIND"
    AnnotationStore.create(
        packet / "private" / "annotations.sqlite3", (independent,), manual_digest="TEST_ONLY"
    )
    parent = Protocol(
        dataset_digest=dataset_digest((independent,)),
        policy_names=("always-listen",),
        model_identity="no-model",
        prompt_digest="TEST_ONLY",
    )
    write_json_atomic(packet / "protocol.json", parent, overwrite=False)
    with pytest.raises(ValueError, match="private"):
        freeze_plan(packet, tmp_path / "public-plan.json")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validity-audit",
            "--packet",
            str(packet),
            "--output",
            str(tmp_path / "public-audit.json"),
        ],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "public-audit.json").exists()
