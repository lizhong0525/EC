"""Explicit offline entry points. No provider configuration or dotenv is read."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ecocompanion.benchmark.generation import DEFAULT_CASE_SEED, generate_canonical_cases
from ecocompanion.benchmark.policies import (
    AlwaysEscalatePolicy,
    AlwaysListenPolicy,
    HistoryAwareRulePolicy,
    KeywordTriagePolicy,
)
from ecocompanion.benchmark.splitting import canonical_split
from ecocompanion.benchmark.tracks import BenchmarkPolicy
from ecocompanion.domain.schemas import CaseSpecV1
from ecocompanion.experiments.io import read_jsonl_models, write_json_atomic
from ecocompanion.research import VERSION
from ecocompanion.research.annotation import AnnotationStore, dataset_digest, digest
from ecocompanion.research.contracts import Prediction, Protocol, ResearchCase
from ecocompanion.research.evaluation import score
from ecocompanion.research.fingerprint import runtime_manifest, source_fingerprint
from ecocompanion.research.governance import (
    duplicate_review,
    export_development_inputs,
    group_audit,
)
from ecocompanion.research.identity import IdentityMap, observable_case

LOCAL_POLICIES: dict[str, Callable[[], BenchmarkPolicy]] = {
    "always-listen": AlwaysListenPolicy,
    "always-escalate": AlwaysEscalatePolicy,
    "keyword-triage": KeywordTriagePolicy,
    "history-aware-rule": HistoryAwareRulePolicy,
}


def select_pilot() -> tuple[CaseSpecV1, ...]:
    """Greedy coverage of risk/ecology/style from TRAIN only; no test-set tuning."""
    pool = list(canonical_split(generate_canonical_cases()).train)
    counts: Counter[str] = Counter()
    selected: list[CaseSpecV1] = []
    while len(selected) < 20:

        def priority(case: CaseSpecV1) -> tuple[float, str]:
            cells = (
                f"risk:{case.risk_trajectory}",
                f"eco:{case.social_ecology}",
                f"style:{case.expression_style}",
            )
            return sum(1 / (1 + counts[c]) for c in cells), case.case_id

        case = max(pool, key=priority)
        pool.remove(case)
        selected.append(case)
        counts.update(
            (
                f"risk:{case.risk_trajectory}",
                f"eco:{case.social_ecology}",
                f"style:{case.expression_style}",
            )
        )
    return tuple(selected)


def prepare(output: Path, manual: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new immutable pilot directory")
    if not manual.is_file():
        raise FileNotFoundError(manual)
    legacy = select_pilot()
    identities = IdentityMap()
    cases = tuple(observable_case(c, identities) for c in legacy)
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir()
    manual_digest = hashlib.sha256(manual.read_bytes()).hexdigest()
    store, tokens = AnnotationStore.create(
        private / "annotations.sqlite3", cases, manual_digest=manual_digest
    )
    write_json_atomic(private / "access.json", tokens, overwrite=False)
    write_json_atomic(
        private / "identity-map.json", {"mapping": identities.mapping}, overwrite=False
    )
    write_json_atomic(
        private / "simulation-truth.json",
        {
            "source": "PUBLIC_GENERATOR_SILVER",
            "generator_seed": DEFAULT_CASE_SEED,
            "nodes": [
                {
                    "case_id": identities.opaque(c.case_id),
                    "step_id": s.step_id,
                    "depression": s.state.depression_burden,
                    "safety": s.state.acute_safety_risk,
                }
                for c in legacy
                for s in c.timeline
            ],
        },
        overwrite=False,
    )
    write_json_atomic(
        private / "manual-snapshot.json",
        {"sha256": manual_digest, "text": manual.read_text(encoding="utf-8")},
        overwrite=False,
    )
    export_development_inputs(cases, output / "public" / "inputs.json")
    # Four deterministic diagnostic baselines; no fake ECO-SAFE/model comparison.
    policy_digest = source_fingerprint()
    protocol = Protocol(
        dataset_digest=dataset_digest(cases),
        policy_names=tuple(LOCAL_POLICIES),
        model_identity="no-model",
        prompt_digest=f"local-source-sha256:{policy_digest}",
        source_digest=policy_digest,
        runtime_digest=digest(runtime_manifest()),
    )
    write_json_atomic(output / "protocol.json", protocol, overwrite=False)
    write_json_atomic(output / "runtime-inventory.json", runtime_manifest(), overwrite=False)
    audit = {
        "version": VERSION,
        "classification": "PUBLIC_DEVELOPMENT_PILOT",
        "case_count": len(cases),
        "node_count": sum(len(c.observations) for c in cases),
        "source_split": "legacy_train_only",
        "expert_gold_status": "PENDING",
        "human_research_status": "NOT_CONDUCTED",
        "independent_blind_status": "NOT_ACQUIRED",
        "coverage": {
            "risk": dict(Counter(c.risk_trajectory.value for c in legacy)),
            "ecology": dict(Counter(c.social_ecology.value for c in legacy)),
            "expression": dict(Counter(c.expression_style.value for c in legacy)),
        },
        "group_audit": group_audit(cases),
        "duplicate_review": duplicate_review(cases),
        "identity_policy": "random persisted opaque IDs; no category encoding",
        "history_policy": (
            "same frozen student inputs for every strategy; no generated assistant history"
        ),
        "legacy_holdout_policy": "reconstructible public data, not a confidential blind test",
        "annotation_audit": store.audit(),
        "runtime": {**runtime_manifest(), "platform": platform.platform()},
        "source_digest": policy_digest,
    }
    write_json_atomic(output / "pilot-manifest.json", audit, overwrite=False)
    return audit


def run_local(packet: Path, output: Path) -> dict[str, Any]:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    store.audit()
    cases = store.cases()
    require_private_output(cases, packet, output)
    protocol = Protocol.model_validate_json((packet / "protocol.json").read_text(encoding="utf-8"))
    if protocol.execution != "local_rules" or any(
        p not in LOCAL_POLICIES for p in protocol.policy_names
    ):
        raise ValueError("this command executes offline diagnostic rules only")
    if protocol.source_digest != source_fingerprint() or protocol.runtime_digest != digest(
        runtime_manifest()
    ):
        raise ValueError(
            "source/runtime changed since protocol freeze; prepare a new versioned run"
        )
    if output.exists():
        raise FileExistsError("choose a new run directory; historical runs are immutable")
    output.mkdir(parents=True)
    write_json_atomic(output / "protocol.json", protocol, overwrite=False)
    write_json_atomic(output / "runtime-inventory.json", runtime_manifest(), overwrite=False)
    predictions = []
    for name in protocol.policy_names:
        for repeat in range(protocol.repeats):
            policy = LOCAL_POLICIES[name]()
            for case in cases:
                for observation in case.observations:
                    started = time.perf_counter()
                    try:
                        decision = policy.decide(observation)
                        prediction = Prediction(
                            policy=name,
                            case_id=case.case_id,
                            step_id=observation.step_id,
                            repeat=repeat,
                            status="ok",
                            decision=decision,
                            latency_seconds=time.perf_counter() - started,
                            cost_cny=0,
                            protocol_digest=digest(protocol.model_dump(mode="json")),
                        )
                    except Exception as exc:
                        prediction = Prediction(
                            policy=name,
                            case_id=case.case_id,
                            step_id=observation.step_id,
                            repeat=repeat,
                            status="error",
                            error_code=type(exc).__name__,
                            latency_seconds=time.perf_counter() - started,
                            cost_cny=0,
                            protocol_digest=digest(protocol.model_dump(mode="json")),
                        )
                    predictions.append(prediction)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(prediction.model_dump_json() + "\n")
    return build_v2_report(packet, output / "predictions.jsonl", output / "report.json")


def build_v2_report(
    packet: Path,
    predictions: Path,
    output: Path,
    *,
    gold: Path | None = None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    store.audit()
    cases = store.cases()
    require_private_output(cases, packet, output)
    protocol = Protocol.model_validate_json(
        (protocol_path or packet / "protocol.json").read_text(encoding="utf-8")
    )
    truth_path = packet / "private" / "simulation-truth.json"
    truth = None
    if truth_path.exists():
        truth = {
            (r["case_id"], r["step_id"]): (r["depression"], r["safety"])
            for r in json.loads(truth_path.read_text(encoding="utf-8"))["nodes"]
        }
    document = json.loads(gold.read_text(encoding="utf-8")) if gold else None
    if document is not None:
        # A Gold document must be anchored to this curator store's export event.
        with store.connection() as connection:
            events = [
                json.loads(r["payload"]) for r in connection.execute("SELECT payload FROM events")
            ]
        if not any(
            e["kind"] == "gold_export" and e["gold_digest"] == document.get("gold_digest")
            for e in events
        ):
            raise ValueError("Gold has no matching export in this store's audit trail")
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    report = score(
        cases,
        protocol,
        read_jsonl_models(predictions, Prediction),
        gold_document=document,
        simulation_truth=truth,
    )
    report["prediction_file_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    write_json_atomic(output, report, overwrite=False)
    return report


def require_private_output(cases: tuple[ResearchCase, ...], packet: Path, output: Path) -> None:
    if any(c.split == "independent_blind" for c in cases):
        if not output.resolve().is_relative_to((packet / "private").resolve()):
            raise ValueError("blind runs/reports must remain inside this packet's private folder")


def prepare_independent(
    staged: Path, output: Path, manual: Path, *, review: Path
) -> dict[str, Any]:
    """Access-controlled adapter for independently supplied inputs, never a generator."""
    if output.exists():
        raise FileExistsError("choose a new blind packet")
    document = json.loads(staged.read_text(encoding="utf-8"))
    cases = tuple(ResearchCase.model_validate(c) for c in document["cases"])
    if not cases or any(c.split != "independent_blind" for c in cases):
        raise ValueError("independent adapter refuses public or generated material")
    decision = json.loads(review.read_text(encoding="utf-8"))
    queue = document["semantic_review"]["queue"]
    required_pairs = {(r["left_case"], r["right_case"]) for r in queue}
    supplied = {(r["left_case"], r["right_case"]) for r in decision.get("pairs", [])}
    if (
        decision.get("staged_digest") != digest(document)
        or not decision.get("reviewer")
        or not decision.get("reference")
        or decision.get("reviewer") in {c.provenance.author_id for c in cases}
        or decision.get("full_material_review") is not True
        or required_pairs != supplied
        or len(supplied) != len(decision.get("pairs", []))
        or any(
            r.get("disposition") != "independent" or not r.get("reason")
            for r in decision.get("pairs", [])
        )
    ):
        raise ValueError("complete independent human duplicate/material review is required")
    private = output / "private"
    private.mkdir(parents=True)
    manual_digest = hashlib.sha256(manual.read_bytes()).hexdigest()
    store, tokens = AnnotationStore.create(
        private / "annotations.sqlite3", cases, manual_digest=manual_digest
    )
    write_json_atomic(private / "access.json", tokens, overwrite=False)
    write_json_atomic(
        private / "manual-snapshot.json",
        {
            "sha256": manual_digest,
            "text": manual.read_text(encoding="utf-8"),
        },
        overwrite=False,
    )
    write_json_atomic(
        private / "independence-review.json",
        {
            "staged": document,
            "review": decision,
        },
        overwrite=False,
    )
    source_digest = source_fingerprint()
    protocol = Protocol(
        dataset_digest=dataset_digest(cases),
        policy_names=tuple(LOCAL_POLICIES),
        model_identity="no-model",
        prompt_digest=f"local-source-sha256:{source_digest}",
        source_digest=source_digest,
        runtime_digest=digest(runtime_manifest()),
    )
    write_json_atomic(output / "protocol.json", protocol, overwrite=False)
    manifest = {
        "classification": "INDEPENDENT_BLIND_PRIVATE_PENDING_GOLD",
        "case_count": len(cases),
        "node_count": sum(len(c.observations) for c in cases),
        "gold_status": "PENDING",
        "runtime": runtime_manifest(),
        "source_digest": source_digest,
        "audit": store.audit(),
    }
    write_json_atomic(output / "packet-manifest.json", manifest, overwrite=False)
    return manifest
