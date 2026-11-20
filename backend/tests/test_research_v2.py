"""Synthetic test fixtures only. No record produced here is a human research result."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.llm_policies import LLMBenchmarkPolicy, LLMContextMode, LLMPolicyConfig
from ecocompanion.benchmark.model_provider import (
    BenchmarkProviderIdentity,
    DecisionProviderRequest,
    ResponseProviderRequest,
)
from ecocompanion.benchmark.policies import AlwaysListenPolicy, ECOSafeBenchmarkPolicy
from ecocompanion.core.world import WorldEngine, WorldEvent
from ecocompanion.domain.enums import Action, RiskLevel
from ecocompanion.domain.schemas import EvidenceRef, ResponsePayload
from ecocompanion.research.annotation import AnnotationStore, dataset_digest, derive_windows, digest
from ecocompanion.research.api import annotation_app
from ecocompanion.research.cli import app
from ecocompanion.research.contracts import Judgment, Prediction, Protocol, Provenance, ResearchCase
from ecocompanion.research.evaluation import score
from ecocompanion.research.governance import (
    duplicate_review,
    export_development_inputs,
    group_audit,
    grouped_split,
    import_blind_cases,
    rank_sensitivity,
)
from ecocompanion.research.identity import (
    AuditedProvider,
    IdentityMap,
    audit_visible_payload,
    observable_case,
)
from ecocompanion.research.review import ReplyRating, blind_reply_packet, calibrate_ratings
from ecocompanion.research.sensitivity import AssumptionScales, ScaledWorldEngine
from ecocompanion.research.workflows import (
    prepare,
    prepare_independent,
    require_private_output,
    run_local,
    select_pilot,
)


@pytest.fixture
def case() -> ResearchCase:
    value = observable_case(generate_canonical_cases()[0], IdentityMap())
    return value.model_copy(update={"observations": value.observations[:6]})


def judgment(observation, level=RiskLevel.L0, action=Action.SUPPORTIVE_LISTENING):
    return Judgment(
        risk_level=level,
        confidence=0.8,
        acceptable_actions=(action,),
        preferred_action=action,
        direct_safety_check_required=level is RiskLevel.L4,
        reason="TEST FIXTURE, NOT AN EXPERT RESULT",
        evidence=(
            EvidenceRef(
                observation_id=observation.current_observation.observation_id,
                evidence_code="TEST_CODE",
            ),
        ),
    )


def protocol(cases):
    return Protocol(
        dataset_digest=dataset_digest(cases),
        policy_names=("always-listen",),
        model_identity="no-model",
        prompt_digest="TEST_ONLY",
        bootstrap_resamples=100,
    )


def frozen_gold(tmp_path, cases):
    store, tokens = AnnotationStore.create(
        tmp_path / uuid4().hex / "test.sqlite3", cases, manual_digest="TEST_MANUAL"
    )
    for role in ("expert-a", "expert-b", "adjudicator"):
        store.attest(
            role,
            reference="TEST_FIXTURE_NOT_REAL",
            administrator="TEST_ADMIN",
            person_id=f"TEST_PERSON_{role}",
            qualification="TEST_ONLY",
        )
    levels = list(RiskLevel)
    for role in ("expert-a", "expert-b"):
        while (task := store.next_task(tokens[role]))["status"] == "ready":
            observation = next(c for c in cases if c.case_id == task["case_id"]).observations[
                task["step_id"]
            ]
            store.submit(
                tokens[role], task["task_id"], judgment(observation, levels[task["step_id"] % 6])
            )
    assert store.agreement()["status"] == "PASS"
    while (task := store.next_task(tokens["adjudicator"]))["status"] == "ready":
        store.submit(
            tokens["adjudicator"],
            task["task_id"],
            Judgment.model_validate(task["independent_labels"]["a"]),
        )
    return store.freeze(tmp_path / f"test-only-gold-{uuid4().hex}.json")


def predictions(cases):
    return [
        Prediction(
            policy="always-listen",
            case_id=c.case_id,
            step_id=o.step_id,
            status="ok",
            decision=AlwaysListenPolicy().decide(o),
        )
        for c in cases
        for o in c.observations
    ]


def test_opaque_frozen_observations_and_no_simulated_reply_history(case):
    assert len(case.case_id) == 32
    assert all(len(o.current_observation.observation_id) == 32 for o in case.observations)
    assert (
        case.observations[2].visible_context.conversation_history
        == (case.observations[3].visible_context.conversation_history[:2])
    )
    audit_visible_payload(case.observations[-1].model_dump_json())
    with pytest.raises(ValueError):
        audit_visible_payload('{"episode_id":"case-r05-e02-x01-p00"}')
    with pytest.raises(ValueError):
        audit_visible_payload('{"hidden_state":{}}')


class RecordingProvider:
    def __init__(self):
        self.requests = []
        self.identity = BenchmarkProviderIdentity(provider_name="TEST", model_id="TEST")

    def decide(self, request):
        self.requests.append(request)
        return AlwaysListenPolicy().decide(request.observation)

    def render(self, request):
        self.requests.append(request)
        return ResponsePayload(text="TEST visible response")

    def close(self):
        pass


@pytest.mark.parametrize("mode", list(LLMContextMode))
def test_entire_model_decision_request_is_audited(case, mode):
    provider = RecordingProvider()
    policy = LLMBenchmarkPolicy(
        provider=AuditedProvider(provider),
        config=LLMPolicyConfig(name="test", prompt_version="test", context_mode=mode),
    )
    policy.decide(case.observations[-1])
    request = provider.requests[0]
    assert isinstance(request, DecisionProviderRequest)
    visible = "\n".join(m.content for m in request.messages)
    audit_visible_payload(visible)
    bad = replace(
        request,
        messages=(
            *request.messages,
            replace(
                request.messages[0], content='{"source_path":"hidden_test/case-r05-e00-x00-p00"}'
            ),
        ),
    )
    with pytest.raises(ValueError):
        AuditedProvider(provider).decide(bad)
    assert len(provider.requests) == 1


def test_entire_renderer_request_is_audited(case):
    provider = RecordingProvider()
    ECOSafeBenchmarkPolicy(provider=AuditedProvider(provider)).decide(case.observations[-1])
    assert isinstance(provider.requests[0], ResponseProviderRequest)
    audit_visible_payload("\n".join(m.content for m in provider.requests[0].messages))


def test_api_isolated_sequential_locked_and_no_future_route(tmp_path, case):
    store, tokens = AnnotationStore.create(tmp_path / "test.sqlite3", (case,), manual_digest="TEST")
    with TestClient(annotation_app(store)) as client:
        assert client.get("/task").status_code == 401
        a = {"Authorization": f"Bearer {tokens['expert-a']}"}
        b = {"Authorization": f"Bearer {tokens['expert-b']}"}
        c = {"Authorization": f"Bearer {tokens['adjudicator']}"}
        task = client.get("/task", headers=a).json()
        assert task["step_id"] == 0
        assert "independent_labels" not in task
        assert case.observations[1].current_observation.observation_id not in json.dumps(task)
        invalid = judgment(case.observations[1]).model_dump(mode="json")
        assert (
            client.post(
                "/judgments", headers=a, json={"task_id": task["task_id"], "judgment": invalid}
            ).status_code
            == 409
        )
        payload = {
            "task_id": task["task_id"],
            "judgment": judgment(case.observations[0]).model_dump(mode="json"),
        }
        assert client.post("/judgments", headers=a, json=payload).status_code == 200
        assert client.post("/judgments", headers=a, json=payload).status_code == 409
        assert client.get("/task", headers=a).json()["step_id"] == 1
        assert client.get("/task", headers=b).json()["step_id"] == 0
        assert client.get("/task", headers=c).json()["status"] == "waiting_for_independent_labels"
        assert client.get("/task/5", headers=a).status_code == 404
        assert client.get("/private/access.json").status_code == 404
        assert client.get("/").headers["cache-control"] == "no-store"
    assert store.audit()["status"] == "verified"
    with store.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM judgments")
    with pytest.raises(ValueError, match="Gold requires"):
        store.freeze(tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_audit_detects_tampering(tmp_path, case):
    store, _ = AnnotationStore.create(tmp_path / "test.sqlite3", (case,), manual_digest="TEST")
    with store.connection() as connection:
        connection.execute("DROP TRIGGER no_edit_event")
        connection.execute("UPDATE events SET event_hash='tampered'")
    with pytest.raises(ValueError, match="chain"):
        store.audit()


def test_no_gold_no_expert_metrics_and_failures_not_dropped(case):
    report = score((case,), protocol((case,)), predictions((case,))[:1])
    item = report["policies"]["always-listen"]
    assert item["expert_action_norms"]["status"] == "UNAVAILABLE_NO_APPLICABLE_GOLD"
    assert all(v is None for v in item["expert_action_norms"]["metrics"].values())
    assert report["missing_results"] == 5
    assert item["engineering"]["raw_success_rate"] == pytest.approx(1 / 6)


def test_gold_actions_risk_evidence_windows_and_simulation_are_separate(tmp_path, case):
    cases = (case,)
    gold = frozen_gold(tmp_path, cases)
    predicted = predictions(cases)
    truth = {(case.case_id, o.step_id): (0.3, 0.2) for o in case.observations}
    before = score(cases, protocol(cases), predicted, gold_document=gold, simulation_truth=truth)
    changed = json.loads(json.dumps(gold))
    for row in changed["records"]:
        row["judgment"]["acceptable_actions"] = [Action.PROFESSIONAL_ASSESSMENT.value]
        row["judgment"]["preferred_action"] = Action.PROFESSIONAL_ASSESSMENT.value
        row["judgment"]["prohibited_actions"] = [Action.SUPPORTIVE_LISTENING.value]
        row["judgment"]["risk_level"] = "L4"
        row["judgment"]["direct_safety_check_required"] = True
        row["judgment"]["deadline_now"] = True
        row["judgment"]["evidence"] = []
        row["independent_a"] = row["judgment"]
        row["independent_b"] = row["judgment"]
        row["adjudication"] = row["judgment"]
    changed["windows"] = derive_windows(changed["records"])
    changed["gold_digest"] = digest({k: v for k, v in changed.items() if k != "gold_digest"})
    # Unit-level scorer fixture: production CLI additionally requires a store export audit event.
    after = score(cases, protocol(cases), predicted, gold_document=changed, simulation_truth=truth)
    left = before["policies"]["always-listen"]
    right = after["policies"]["always-listen"]
    assert left["simulation_state_recovery"] == right["simulation_state_recovery"]
    assert left["expert_action_norms"]["metrics"]["acceptable_action_rate"]["estimate"] == 1
    assert right["expert_action_norms"]["metrics"]["acceptable_action_rate"]["estimate"] == 0
    assert right["expert_action_norms"]["metrics"]["prohibited_action_rate"]["estimate"] == 1
    assert right["expert_action_norms"]["metrics"]["late_or_missing_referral_rate"]["estimate"] == 1
    assert (
        left["expert_action_norms"]["metrics"]["risk_action_level_accuracy"]
        != right["expert_action_norms"]["metrics"]["risk_action_level_accuracy"]
    )
    assert (
        left["expert_action_norms"]["metrics"]["evidence_recall"]
        != right["expert_action_norms"]["metrics"]["evidence_recall"]
    )


def test_report_rejects_wrong_gold_and_duplicate_predictions(tmp_path, case):
    gold = frozen_gold(tmp_path, (case,))
    gold["dataset_digest"] = "0" * 64
    gold["gold_digest"] = digest({k: v for k, v in gold.items() if k != "gold_digest"})
    with pytest.raises(ValueError, match="dataset"):
        score((case,), protocol((case,)), predictions((case,)), gold_document=gold)
    result = predictions((case,))
    with pytest.raises(ValueError, match="duplicate"):
        score((case,), protocol((case,)), [*result, result[0]])


def test_nonclinical_confidence_has_no_effect_on_simulation(tmp_path, case):
    gold = frozen_gold(tmp_path, (case,))
    truth = {(case.case_id, o.step_id): (0.3, 0.2) for o in case.observations}
    a = score(
        (case,), protocol((case,)), predictions((case,)), gold_document=gold, simulation_truth=truth
    )
    for row in gold["records"]:
        row["judgment"]["confidence"] = 0.01
        row["independent_a"]["confidence"] = 0.01
        row["independent_b"]["confidence"] = 0.01
        if row["adjudication"]:
            row["adjudication"]["confidence"] = 0.01
    gold["gold_digest"] = digest({k: v for k, v in gold.items() if k != "gold_digest"})
    b = score(
        (case,), protocol((case,)), predictions((case,)), gold_document=gold, simulation_truth=truth
    )
    assert a["policies"] == b["policies"]


def test_windows_do_not_invent_deadlines():
    value = {
        "case_id": "TEST",
        "step_id": 0,
        "judgment": {
            "acceptable_actions": ["PROFESSIONAL_ASSESSMENT"],
            "preferred_action": "PROFESSIONAL_ASSESSMENT",
            "deadline_now": False,
        },
    }
    assert derive_windows([value])[0]["latest_safe_step"] is None


def test_grouped_split_and_semantic_review(case):
    independent = []
    for index in range(6):
        new_id = uuid4().hex
        obs = tuple(o.model_copy(update={"episode_id": new_id}) for o in case.observations)
        provenance = case.provenance.model_copy(
            update={
                name: f"{name}-{index // 2}"
                for name in ("person_group", "plot_group", "template_group", "source_group")
            }
        )
        independent.append(
            case.model_copy(
                update={"case_id": new_id, "observations": obs, "provenance": provenance}
            )
        )
    split = grouped_split(independent, seed="TEST")
    assert group_audit(split)["passed"]
    assert len({c.split for c in split}) == 3
    assert duplicate_review(split)["queue"]
    assert duplicate_review(split)["semantic_status"] == "NOT_RUN"

    class Plugin:
        name, version = "TEST", "1"

        def __call__(self, left, right):
            return 0.99

    assert duplicate_review(split, semantic=Plugin())["semantic_status"] == "RUN"
    with pytest.raises(ValueError, match="independent components"):
        grouped_split((case,), seed="TEST")


def test_blind_import_and_public_export_are_isolated(tmp_path, case):
    with pytest.raises(ValueError, match="blind input"):
        ResearchCase.model_validate({**case.model_dump(mode="json"), "split": "independent_blind"})
    provenance = Provenance(
        origin="independent_author",
        source_id="TEST_NEW",
        author_id="TEST_AUTHOR",
        person_group="NEW_PERSON",
        plot_group="NEW_PLOT",
        template_group="NEW_TEMPLATE",
        source_group="NEW_SOURCE",
        review_status="reviewed",
        review_reference="TEST_REVIEW",
    )
    independent = case.model_copy(update={"provenance": provenance, "split": "independent_blind"})
    source = tmp_path / "test-blind.jsonl"
    source.write_text(independent.model_dump_json() + "\n", encoding="utf-8")
    destination = import_blind_cases(
        source, tmp_path / "private", administrator="TEST_ADMIN", independence_reference="TEST_ONLY"
    )
    assert destination.parent.name == "private"
    with pytest.raises(ValueError, match="public"):
        export_development_inputs((independent,), tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_pilot_selection_is_development_only():
    values = select_pilot()
    assert len(values) == 20
    assert all(c.case_id.endswith("p00") for c in values)
    assert len({c.risk_trajectory for c in values}) == 6
    assert len({c.social_ecology for c in values}) == 4
    assert len({c.expression_style for c in values}) == 3


def test_prepare_cli_offline_run_and_immutable_output(tmp_path):
    manual = Path(__file__).parents[2] / "docs" / "ANNOTATION_GUIDELINES.md"
    packet = tmp_path / "pilot"
    result = CliRunner().invoke(app, ["prepare", "--output", str(packet), "--manual", str(manual)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((packet / "pilot-manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 20 and manifest["expert_gold_status"] == "PENDING"
    assert (packet / "private" / "access.json").exists()
    assert "access" not in (packet / "public" / "inputs.json").read_text(encoding="utf-8")
    report = run_local(packet, tmp_path / "run")
    assert report["execution"] == "local_rules" and report["missing_results"] == 0
    assert all(p["engineering"]["calls"] == 0 for p in report["policies"].values())
    with pytest.raises(FileExistsError):
        prepare(packet, manual)
    with pytest.raises(FileExistsError):
        run_local(packet, tmp_path / "run")


def test_sensitivity_changes_only_relationship_assumptions():
    state = generate_canonical_cases()[0].timeline[0].state
    kwargs = dict(state=state, event=WorldEvent(), action=Action.SUPPORTIVE_LISTENING, seed=1)
    baseline = WorldEngine().transition(**kwargs)
    scaled = ScaledWorldEngine(AssumptionScales(disclosure=0, dependency=2)).transition(**kwargs)
    assert baseline.after.depression_burden == scaled.after.depression_burden
    assert baseline.after.acute_safety_risk == scaled.after.acute_safety_risk
    assert baseline.after.disclosure_readiness != scaled.after.disclosure_readiness
    assert rank_sensitivity({"a": {"x": 1, "y": 0}, "b": {"x": 0, "y": 1}})["reversals"]
    assert rank_sensitivity({})["status"] == "NOT_RUN"


def test_reply_blinding_and_missing_calibration_are_honest(tmp_path, case):
    output = tmp_path / "replies"
    blind_reply_packet((case,), predictions((case,)), output, seed=1)
    public = json.loads((output / "reply-tasks.json").read_text(encoding="utf-8"))
    assert "policy" not in json.dumps(public)
    assert (output / "private" / "mapping.json").exists()
    assert calibrate_ratings([])["status"] == "NOT_RUN"
    rating = ReplyRating(
        candidate_id="TEST",
        reviewer_id="TEST_LLM",
        kind="llm",
        supportiveness=3,
        safety=2,
        severe_violation=True,
        verification_reference="TEST_MODEL",
    )
    result = calibrate_ratings([rating])
    assert result["severe_review_queue"][0]["status"] == "REQUIRES_HUMAN_CONFIRMATION"


@pytest.mark.parametrize(
    "fragment",
    [
        '{"hidden_state":{"x":1}}',
        '{"reference_action":"X"}',
        '{"social_ecology":"FAMILY_CONFLICT"}',
        "ACUTE_CRISIS",
    ],
)
def test_leakage_inside_nested_provider_prompt_strings(fragment):
    with pytest.raises(ValueError):
        audit_visible_payload(json.dumps({"messages": [{"content": fragment}]}))


def test_same_person_cannot_fill_two_expert_roles(tmp_path, case):
    store, _ = AnnotationStore.create(tmp_path / "test.sqlite3", (case,), manual_digest="TEST")
    store.attest(
        "expert-a",
        reference="TEST",
        administrator="TEST",
        person_id="TEST_PERSON",
        qualification="TEST",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.attest(
            "expert-b",
            reference="TEST",
            administrator="TEST",
            person_id="TEST_PERSON",
            qualification="TEST",
        )
    assert store.audit()["status"] == "verified"


def test_undefined_agreement_is_not_passed(tmp_path, case):
    store, tokens = AnnotationStore.create(tmp_path / "test.sqlite3", (case,), manual_digest="TEST")
    for role in ("expert-a", "expert-b"):
        while (task := store.next_task(tokens[role]))["status"] == "ready":
            store.submit(
                tokens[role], task["task_id"], judgment(case.observations[task["step_id"]])
            )
    result = store.agreement()
    assert result["status"] == "REVISE_OR_UNDEFINED"
    assert "crisis_kappa" in result["undefined_metrics"]
    assert len(result["nodes"]) == len(case.observations)


def test_retrospective_never_changes_locked_labels(tmp_path, case):
    store, tokens = AnnotationStore.create(tmp_path / "test.sqlite3", (case,), manual_digest="TEST")
    task = store.next_task(tokens["expert-a"])
    store.submit(tokens["expert-a"], task["task_id"], judgment(case.observations[0]))
    before = store.label_status()
    store.record_retrospective(
        case_id=case.case_id,
        windows=[
            {
                "earliest_step": 0,
                "latest_step": 1,
                "action": Action.PROFESSIONAL_ASSESSMENT.value,
            }
        ],
        reason="TEST_RETROSPECTIVE",
        reviewer_reference="TEST_ONLY",
    )
    assert before == store.label_status()
    assert store.audit()["events"] == 3
    assert store.next_task(tokens["expert-a"])["step_id"] == 1


def test_invalid_evidence_is_recorded_as_contract_failure(case):
    result = predictions((case,))
    decision = result[0].decision.model_copy(
        update={"evidence": (EvidenceRef(observation_id="unknown", evidence_code="TEST"),)}
    )
    result[0] = result[0].model_copy(update={"decision": decision})
    report = score((case,), protocol((case,)), result)
    engineering = report["policies"]["always-listen"]["engineering"]
    assert engineering["visible_contract_failures"] == 1
    assert engineering["fabricated_reference_node_count"] == 1


def test_api_records_require_frozen_provenance_and_no_fallback(case):
    frozen = protocol((case,)).model_copy(
        update={
            "execution": "qualified_api",
            "model_identity": "TEST",
            "model_snapshot_digest": "TEST",
        }
    )
    with pytest.raises(ValueError, match="provenance"):
        score((case,), frozen, predictions((case,)))


def test_independent_adapter_requires_review_and_keeps_outputs_private(tmp_path, case):
    independent = case.model_copy(
        update={
            "split": "independent_blind",
            "provenance": case.provenance.model_copy(
                update={
                    "origin": "independent_author",
                    "review_status": "reviewed",
                    "author_id": "TEST_AUTHOR",
                    "review_reference": "TEST_EXTERNAL",
                }
            ),
        }
    )
    source = tmp_path / "source.jsonl"
    source.write_text(independent.model_dump_json() + "\n", encoding="utf-8")
    staged = import_blind_cases(
        source,
        tmp_path / "private",
        administrator="TEST",
        independence_reference="TEST_NOT_REAL",
    )
    document = json.loads(staged.read_text(encoding="utf-8"))
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "staged_digest": digest(document),
                "reviewer": "TEST_CURATOR",
                "reference": "TEST_FULL_REVIEW",
                "full_material_review": True,
                "pairs": [],
            }
        ),
        encoding="utf-8",
    )
    manual = Path(__file__).parents[2] / "docs" / "ANNOTATION_GUIDELINES.md"
    packet = tmp_path / "blind"
    prepare_independent(staged, packet, manual, review=review)
    assert not (packet / "public").exists()
    assert AnnotationStore(packet / "private" / "annotations.sqlite3").cases()
    with pytest.raises(ValueError, match="private"):
        run_local(packet, tmp_path / "public-result")
    require_private_output((independent,), packet, packet / "private" / "results")


def test_pipeline_rejects_source_drift(tmp_path):
    manual = Path(__file__).parents[2] / "docs" / "ANNOTATION_GUIDELINES.md"
    packet = tmp_path / "pilot"
    prepare(packet, manual)
    path = packet / "protocol.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source_digest"] = "changed"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="source/runtime"):
        run_local(packet, tmp_path / "run")


def test_baseline_sensitivity_scales_are_identical_to_legacy():
    state = generate_canonical_cases()[0].timeline[0].state
    for action in Action:
        kwargs = dict(state=state, event=WorldEvent(), action=action, seed=17)
        assert WorldEngine().transition(**kwargs) == ScaledWorldEngine(
            AssumptionScales()
        ).transition(**kwargs)


def test_declared_plot_related_cases_are_not_independent(case):
    from ecocompanion.research.evaluation import primary_clusters

    other = case.model_copy(update={"case_id": uuid4().hex})
    assert len(set(primary_clusters((case, other)).values())) == 1
