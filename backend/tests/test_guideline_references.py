"""All reviewer names, snapshots and mappings here are temporary programming fixtures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.policies import AlwaysListenPolicy
from ecocompanion.domain.enums import Action
from ecocompanion.research.annotation import AnnotationStore, dataset_digest, digest
from ecocompanion.research.cli import app
from ecocompanion.research.contracts import Prediction, Protocol
from ecocompanion.research.evaluation import load_gold
from ecocompanion.research.identity import IdentityMap, observable_case
from ecocompanion.research.reference_contracts import (
    EvidenceSpan,
    GuidelineCatalog,
    GuidelineMapping,
    MappingReview,
    PartialNorm,
    RuleApplication,
    RuleReview,
)
from ecocompanion.research.references import (
    ReferenceStore,
    fetch_sources,
    read_snapshots,
    reference_report,
)

ROOT = Path(__file__).parents[2]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    catalog = GuidelineCatalog.model_validate_json(
        (ROOT / "docs" / "GUIDELINE_CATALOG_V1.json").read_text(encoding="utf-8")
    )
    client_type = httpx.Client
    text = (
        "<html><p>TEST SNAPSHOT ONLY: "
        "about thoughts or plans of self-harm in the last month</p></html>"
    )

    def response(request):
        if "nice.org.uk" in str(request.url):
            return httpx.Response(403, request=request)
        return httpx.Response(200, text=text, headers={"content-type": "text/html"})

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: client_type(transport=httpx.MockTransport(response), **kw),
    )
    snapshots = tmp_path / "sources"
    fetch_sources(catalog, snapshots)
    value = observable_case(generate_canonical_cases()[0], IdentityMap())
    case = value.model_copy(update={"observations": value.observations[:6]})
    packet = tmp_path / "packet"
    annotations, _ = AnnotationStore.create(
        packet / "private" / "annotations.sqlite3", (case,), manual_digest="TEST_ONLY"
    )
    ledger = packet / "private" / "guideline-v1"
    store = ReferenceStore.create(
        ledger, packet, catalog, snapshots, ROOT / "docs" / "ANNOTATION_OPERATIONS_V2.md"
    )
    protocol = Protocol(
        dataset_digest=dataset_digest((case,)),
        policy_names=("always-listen",),
        model_identity="no-model",
        prompt_digest="TEST_ONLY",
        bootstrap_resamples=100,
    )
    (packet / "protocol.json").write_text(protocol.model_dump_json(), encoding="utf-8")
    return store, annotations, case, catalog, snapshots, packet, protocol


def allow_rule(store, catalog):
    return store.review_rule(
        RuleReview(
            rule_id=catalog.rules[0].rule_id,
            reviewer_id="TEST_SOURCE_CHECKER",
            verification_reference="TEST_NOT_A_REAL_PERSON_OR_REVIEW",
            source_and_scope_checked=True,
            disposition="allow_silver_mapping",
            reason="PROGRAMMING FIXTURE ONLY",
        )
    )


def mapping(case, catalog, *, author="TEST_CURATOR", origin="human_curator", step=0):
    observation = case.observations[step]
    rule = catalog.rules[0]
    return GuidelineMapping(
        author_id=author,
        origin=origin,
        case_id=case.case_id,
        step_id=step,
        observation_digest=digest(observation.model_dump(mode="json")),
        resolution="mapped",
        norm=PartialNorm(direct_safety_check_required=True),
        applications=(
            RuleApplication(
                rule_id=rule.rule_id,
                conditions={k: "met" for k in rule.conditions},
                exclusions={k: "absent" for k in rule.exclusions},
                fields=("direct_safety_check_required",),
                applicability_reason="TEST_ONLY, NOT ACTUAL CASE INTERPRETATION",
            ),
        ),
        evidence=(
            EvidenceSpan(
                observation_id=observation.current_observation.observation_id,
                quote=observation.current_observation.message,
            ),
        ),
        reason="TEST_ONLY_NOT_HUMAN_RESULTS",
    )


def review(event, *, reviewer="TEST_SECOND_CURATOR", disposition="confirm"):
    return MappingReview(
        mapping_digest=event,
        reviewer_id=reviewer,
        verification_reference="TEST_NOT_REAL",
        disposition=disposition,
        reason="PROGRAMMING FIXTURE ONLY",
    )


def predictions(case):
    return [
        Prediction(
            policy="always-listen",
            case_id=case.case_id,
            step_id=o.step_id,
            status="ok",
            decision=AlwaysListenPolicy().decide(o),
        )
        for o in case.observations
    ]


def test_initial_status_is_unresolved_no_fake_gold_or_curator(setup):
    store, annotations, case, _, _, _, _ = setup
    assert store.status()["coverage"] == {
        "GOLD": 0,
        "SILVER": 0,
        "UNRESOLVED": len(case.observations),
        "total_nodes": 6,
    }
    assert annotations.audit()["events"] == 1
    assert annotations.label_status()["gold_status"] == "NOT_FROZEN"
    assert store.audit()["events"] == 1
    with pytest.raises(ValueError, match="Gold requires"):
        annotations.freeze(store.path.parent / "must-not-exist.json")


def test_fetch_records_403_and_rejects_snapshot_tampering(setup):
    _, _, _, catalog, snapshots, _, _ = setup
    records = read_snapshots(snapshots, catalog)
    assert records["nice-ng225"]["status"] == "UNAVAILABLE"
    assert records["nice-ng225"]["http_status"] == 403
    file = next(snapshots.glob("*.json"))
    while file.name == "manifest.json":
        file = next(p for p in snapshots.glob("*.json") if p.name != "manifest.json")
    file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_snapshots(snapshots, catalog)
    with pytest.raises(FileExistsError):
        fetch_sources(catalog, snapshots)


def test_rule_must_be_source_reviewed_and_available(setup):
    store, _, case, catalog, _, _, _ = setup
    with pytest.raises(ValueError, match="pending"):
        store.submit(mapping(case, catalog))
    with pytest.raises(ValueError, match="source anchor"):
        store.review_rule(
            RuleReview(
                rule_id=catalog.rules[1].rule_id,
                reviewer_id="TEST",
                verification_reference="TEST",
                source_and_scope_checked=True,
                disposition="allow_silver_mapping",
                reason="TEST",
            )
        )
    allow_rule(store, catalog)
    with pytest.raises(ValueError, match="locked"):
        allow_rule(store, catalog)


def test_mapping_evidence_scope_conditions_fields_and_sequence(setup):
    store, _, case, catalog, _, _, _ = setup
    allow_rule(store, catalog)
    base = mapping(case, catalog)
    changes = [
        {"observation_digest": "0" * 64},
        {
            "evidence": (
                EvidenceSpan(
                    observation_id=case.observations[1].current_observation.observation_id,
                    quote=case.observations[1].current_observation.message,
                ),
            )
        },
        {
            "evidence": (
                EvidenceSpan(
                    observation_id=case.observations[0].current_observation.observation_id,
                    quote="A NONEXISTENT QUOTE",
                ),
            )
        },
        {"norm": PartialNorm(direct_safety_check_required=True, risk_level="L4")},
        {
            "applications": (
                base.applications[0].model_copy(
                    update={"conditions": {k: "unknown" for k in catalog.rules[0].conditions}}
                ),
            )
        },
    ]
    for update in changes:
        with pytest.raises(ValueError):
            store.submit(base.model_copy(update=update))
    with pytest.raises(ValueError, match="current node"):
        store.submit(mapping(case, catalog, step=1))
    assert store.audit()["events"] == 2


def test_independent_review_no_self_review_ai_drafts_remain_unresolved(setup):
    store, annotations, case, catalog, _, _, _ = setup
    allow_rule(store, catalog)
    event = store.submit(mapping(case, catalog))
    assert store.status()["coverage"]["SILVER"] == 0
    with pytest.raises(ValueError, match="own"):
        store.review_mapping(review(event, reviewer="TEST_CURATOR"))
    store.review_mapping(review(event))
    assert store.status()["coverage"]["SILVER"] == 1
    assert store.status()["records"][0]["norm"]["risk_level"] is None
    ai = store.submit(mapping(case, catalog, author="TEST_AI", origin="ai_draft", step=0))
    store.review_mapping(review(ai))
    assert store.status()["coverage"]["SILVER"] == 1
    ai_next = store.submit(mapping(case, catalog, author="TEST_AI", origin="ai_draft", step=1))
    store.review_mapping(review(ai_next))
    assert store.status()["coverage"]["SILVER"] == 1
    assert annotations.audit()["events"] == 1
    with pytest.raises(ValueError, match="Gold"):
        load_gold(store.status(), (case,))


def test_immutable_records_correction_and_disagreement(setup):
    store, _, case, catalog, _, _, _ = setup
    allow_rule(store, catalog)
    base = mapping(case, catalog)
    event = store.submit(base)
    store.review_mapping(review(event))
    with pytest.raises(ValueError, match="supersede"):
        store.submit(base)
    with store.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM events WHERE seq=2")
    changed = base.model_copy(update={"supersedes": event, "reason": "TEST_CORRECTION"})
    correction = store.submit(changed)
    assert store.status()["coverage"]["SILVER"] == 0
    store.review_mapping(review(correction, disposition="disagree"))
    assert store.status()["records"][0]["reason"] == "mapping_disagreement"
    assert len(store.status()["records"][0]["lineage"]) == 4
    assert store.audit()["status"] == "verified"


def test_silver_scoring_mutates_only_silver_not_expert_or_simulation(setup):
    store, _, case, catalog, _, _, protocol = setup
    predicted = predictions(case)
    truth = {(case.case_id, o.step_id): (0.2, 0.1) for o in case.observations}
    before = reference_report(store, protocol, predicted, simulation_truth=truth)
    allow_rule(store, catalog)
    event = store.submit(mapping(case, catalog))
    store.review_mapping(review(event))
    after = reference_report(store, protocol, predicted, simulation_truth=truth)
    a, b = before["policies"]["always-listen"], after["policies"]["always-listen"]
    assert a["expert_action_norms"] == b["expert_action_norms"]
    assert a["simulation_state_recovery"] == b["simulation_state_recovery"]
    assert a["guideline_silver_constraints"]["metrics"] is None
    silver = b["guideline_silver_constraints"]
    assert set(silver["metrics"]) == {"required_safety_action_recall"}
    assert silver["metrics"]["required_safety_action_recall"]["estimate"] == 0
    assert silver["metrics"]["required_safety_action_recall"]["applicable_node_repeats"] == 1
    action = predicted[0].decision.decision.model_copy(
        update={"primary_action": Action.DIRECT_SAFETY_CHECK}
    )
    predicted[0] = predicted[0].model_copy(
        update={"decision": predicted[0].decision.model_copy(update={"decision": action})}
    )
    changed = reference_report(store, protocol, predicted)["policies"]["always-listen"]
    assert (
        changed["guideline_silver_constraints"]["metrics"]["required_safety_action_recall"][
            "estimate"
        ]
        == 1
    )
    missing = reference_report(store, protocol, [])["policies"]["always-listen"]
    assert (
        missing["guideline_silver_constraints"]["metrics"]["required_safety_action_recall"][
            "estimate"
        ]
        == 0
    )
    assert missing["engineering"]["raw_success_rate"] == 0


def test_current_only_task_and_cli_output_boundaries(setup, tmp_path):
    store, _, case, _, _, packet, _ = setup
    task = store.next_task("TEST_CURATOR")
    assert task["step_id"] == 0
    assert "provenance" not in task
    assert case.observations[1].current_observation.observation_id not in json.dumps(task)
    runner, ledger = CliRunner(), store.path.parent
    path = ledger / "task-01.json"
    result = runner.invoke(
        app,
        [
            "references",
            "task",
            "--ledger",
            str(ledger),
            "--author",
            "TEST_CURATOR",
            "--output",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    for output in (path, tmp_path / "public-status.json"):
        result = runner.invoke(
            app,
            [
                "references",
                "status",
                "--ledger",
                str(ledger),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code != 0
    predicted = ledger / "TEST-predictions.jsonl"
    predicted.write_text(
        "\n".join(p.model_dump_json() for p in predictions(case)), encoding="utf-8"
    )
    result = runner.invoke(
        app,
        [
            "references",
            "report",
            "--ledger",
            str(ledger),
            "--packet",
            str(packet),
            "--predictions",
            str(predicted),
            "--output",
            str(ledger / "TEST-report.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        json.loads((ledger / "TEST-report.json").read_text(encoding="utf-8"))[
            "reference_provenance"
        ]["coverage"]["GOLD"]
        == 0
    )


def test_audit_detects_modified_events(setup):
    store, *_ = setup
    with store.connection() as connection:
        connection.execute("DROP TRIGGER no_update")
        connection.execute("UPDATE events SET event_hash='CORRUPTED'")
    with pytest.raises(ValueError, match="chain"):
        store.audit()


def test_blank_identity_and_ai_to_human_origin_relabel_are_rejected(setup):
    store, _, case, catalog, _, _, _ = setup
    allow_rule(store, catalog)
    base = mapping(case, catalog, origin="ai_draft")
    event = store.submit(base)
    with pytest.raises(ValueError, match="AI author"):
        store.submit(base.model_copy(update={"supersedes": event, "origin": "human_curator"}))
    with pytest.raises(ValueError, match="own"):
        store.review_mapping(review(event, reviewer="  TEST_CURATOR  "))
    with pytest.raises(ValueError):
        store.review_mapping(review(event, reviewer="  "))


def test_guideline_catalog_cannot_invent_unique_or_preferred_action():
    assert PartialNorm(direct_safety_check_required=True).preferred_action is None
    with pytest.raises(ValueError, match="supporting acceptable"):
        PartialNorm(preferred_action=Action.SUPPORTIVE_LISTENING)
    with pytest.raises(ValueError, match="overlap"):
        PartialNorm(
            acceptable_actions=(Action.SUPPORTIVE_LISTENING,),
            prohibited_actions=(Action.SUPPORTIVE_LISTENING,),
        )
