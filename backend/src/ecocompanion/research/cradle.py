"""Local, opt-in CRADLE snapshot and evaluation; no model or credentials loaded."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ecocompanion.benchmark.policies import (
    AlwaysEscalatePolicy,
    AlwaysListenPolicy,
    ECOSafeBenchmarkPolicy,
    HistoryAwareRulePolicy,
    KeywordTriagePolicy,
)
from ecocompanion.benchmark.tracks import BenchmarkPolicy
from ecocompanion.domain.enums import Action, ActorRole, Track
from ecocompanion.domain.schemas import (
    ConversationMessage,
    CurrentObservation,
    ObservationV1,
    VisibleContext,
    VisibleProfile,
)
from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research.annotation import digest
from ecocompanion.research.fingerprint import runtime_manifest, source_fingerprint

DATASET = "SungJoo/Cradle-Dialogue"
VERSION = "cradle-local-v1.0.0"
DEFAULT_REVISION = "49d5f40912e9afeadf4d0c1aa0ffd22c8d89c6a3"
SPLITS = ("train", "validation", "test")
TYPES = ("SI_passive", "SI_active", "SH", "DV", "CA", "SHA", "RA")
LABELS = (
    "alert_ongoing",
    "alert_past",
    *(f"confirm_{kind}_{temporal}" for kind in TYPES for temporal in ("ongoing", "past")),
)

# Fixed, deliberately weak smoke baseline. Not a clinical detector or paper reproduction.
LEXICON: dict[str, tuple[str, ...]] = {
    "SI_passive": ("wish i was dead", "wish i were dead", "better off dead", "not wake up"),
    "SI_active": ("kill myself", "end my life", "suicide plan", "planning suicide"),
    "SH": ("cut myself", "cutting myself", "hurt myself", "self-harm", "self harm"),
    "DV": ("partner hit me", "husband hit me", "boyfriend hit me", "domestic violence"),
    "CA": ("child abuse", "dad hits me", "father hits me", "mother hits me"),
    "SHA": ("sexual harassment", "sexually harassed", "unwanted sexual comments"),
    "RA": ("raped me", "was raped", "sex without my consent", "sexually assaulted"),
}
PAST_MARKERS = ("years ago", "last year", "when i was a child", "in the past", "used to")
NATIVE_POLICIES = ("no-signal", "english-lexicon-first-mention")
PROJECT_POLICIES: dict[str, Callable[[], BenchmarkPolicy]] = {
    "always-listen": AlwaysListenPolicy,
    "always-escalate": AlwaysEscalatePolicy,
    "keyword-triage": KeywordTriagePolicy,
    "history-aware-rule": HistoryAwareRulePolicy,
    "eco-safe": ECOSafeBenchmarkPolicy,
}


def visible_nodes(
    dialogues: list[list[dict[str, Any]]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Separate current/past text from reference labels and internal row coordinates."""
    for ordinal, dialogue in enumerate(dialogues):
        dialogue_id = uuid4().hex
        history: list[dict[str, Any]] = []
        for row in dialogue:
            observation_id = uuid4().hex
            if row["role"] == "User":
                yield (
                    {
                        "dialogue_id": dialogue_id,
                        "observation_id": observation_id,
                        "turn_index": row["turn_id"],
                        "language": "en",
                        "history": list(history),
                        "current_message": row["text"],
                    },
                    {
                        "observation_id": observation_id,
                        "dialogue_id": dialogue_id,
                        "labels": row["labels"],
                    },
                    {
                        "observation_id": observation_id,
                        "dialogue_id": dialogue_id,
                        "dialogue_ordinal": ordinal,
                        "source_row": row["row"],
                        "source_turn_id": row["turn_id"],
                    },
                )
            history.append(
                {
                    "message_id": observation_id,
                    "role": row["role"],
                    "text": row["text"],
                    "turn_index": row["turn_id"],
                }
            )


def project_observation(node: dict[str, Any]) -> ObservationV1:
    """Use existing student/user-only rules without inventing ages, time or ecology."""
    return ObservationV1(
        benchmark_version=VERSION,
        track=Track.A,
        episode_id=node["dialogue_id"],
        step_id=node["turn_index"],
        simulated_time=f"turn:{node['turn_index']}",
        visible_profile=VisibleProfile(age_group="unspecified", language="en"),
        current_observation=CurrentObservation(
            observation_id=node["observation_id"], message=node["current_message"]
        ),
        visible_context=VisibleContext(
            conversation_history=tuple(
                ConversationMessage(
                    message_id=m["message_id"],
                    role=ActorRole.USER,
                    content=m["text"],
                    simulated_time=f"turn:{m['turn_index']}",
                )
                for m in node["history"]
                if m["role"] == "User"
            )
        ),
        allowed_primary_actions=tuple(Action),
    )


def native_labels(policy: str, node: dict[str, Any], seen: set[str]) -> tuple[str, ...]:
    if policy == "no-signal":
        return ()
    if policy != "english-lexicon-first-mention":
        raise ValueError("unknown native policy")
    text = node["current_message"].casefold()
    temporal = "past" if any(marker in text for marker in PAST_MARKERS) else "ongoing"
    found = [
        f"confirm_{kind}_{temporal}"
        for kind, phrases in LEXICON.items()
        if any(phrase in text for phrase in phrases)
    ]
    emitted = tuple(label for label in found if label not in seen)
    seen.update(emitted)
    return emitted


def freeze_external_plan(snapshot: Path, output: Path, *, split: str = "test") -> dict[str, Any]:
    manifest = verify_snapshot(snapshot)
    # Malformed source rows are not silently dropped, merged, or assigned a guessed role.
    dialogues = load_split(snapshot, split)
    if output.exists():
        raise FileExistsError("choose a new experiment directory")
    output.mkdir(parents=True)
    nodes = list(visible_nodes(dialogues))
    _jsonl(output / "inputs.jsonl", (n[0] for n in nodes))
    _jsonl(output / "private" / "references.jsonl", (n[1] for n in nodes))
    _jsonl(output / "private" / "identity-map.jsonl", (n[2] for n in nodes))
    audit = audit_snapshot(snapshot)
    write_json_atomic(output / "data-audit.json", audit, overwrite=False)
    runtime = runtime_manifest()
    write_json_atomic(output / "runtime.json", runtime, overwrite=False)
    plan = {
        "version": VERSION,
        "status": "LOCAL_EXPLORATORY_PROTOCOL_NOT_PREREGISTRATION",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": DATASET,
        "revision": manifest["revision"],
        "snapshot_manifest_sha256": _sha((snapshot / "snapshot.json").read_bytes()),
        "source_split": split,
        "dialogues": len(dialogues),
        "user_nodes": len(nodes),
        "native_label_origin": "UPSTREAM_CLINICIAN_ANNOTATIONS" if split == "test" else "SYNTHETIC",
        "input_sha256": _sha((output / "inputs.jsonl").read_bytes()),
        "reference_sha256": _sha((output / "private" / "references.jsonl").read_bytes()),
        "identity_map_sha256": _sha((output / "private" / "identity-map.jsonl").read_bytes()),
        "audit_sha256": _sha((output / "data-audit.json").read_bytes()),
        "source_digest": source_fingerprint(),
        "runtime_digest": digest(runtime),
        "native_policies": list(NATIVE_POLICIES),
        "project_policies": list(PROJECT_POLICIES),
        "native_reference": "EXACT_USER_TURN_EVENTS_NO_FORWARD_FILL",
        "listener_labels": "PRESERVED_IN_SOURCE_AUDIT_NOT_MOVED_TO_USER",
        "native_context": "CURRENT_USER_AND_OBSERVED_USER_LISTENER_PREFIX",
        "project_context": "CURRENT_AND_PAST_USERS_ONLY_EXISTING_V2_CONVENTION",
        "primary_metric": "native_user_turn_micro_f1",
        "secondary_metrics": ["per_label_f1", "dialogue_union_micro_f1"],
        "empty_denominator": "null_NOT_PERFECT",
        "bootstrap": {"unit": "recovered_dialogue", "resamples": 1000, "seed": 20261021},
        "upstream_post_person_clusters": "UNKNOWN_CI_MAY_BE_OPTIMISTIC",
        "lexicon": LEXICON,
        "past_markers": list(PAST_MARKERS),
        "calls": 0,
        "retries": 0,
        "cost_cny": 0,
        "model": "NONE_LOCAL_RULES_ONLY",
        "translation": "NONE",
        "expert_action_norms": "UNAVAILABLE",
        "simulation_hidden_state": "UNAVAILABLE",
        "paper_reproduction": False,
        "restrictions": "LOCAL_ONLY_LICENSE_UNSPECIFIED_NO_REDISTRIBUTION_NO_TEST_TUNING",
        "limitations": [
            "English phrase baseline has no robust negation, experiencer or temporal resolution",
            "No ambiguous Alert detector in English baseline; Alert recall may be zero",
            "Original project rules/templates are Chinese and are not validated in English",
            "Source Listener replies are fixed; not an interactive intervention experiment",
            "Event labels cannot establish acceptable/referral actions or therapy outcomes",
            "Public source not an independent confidential blind set",
        ],
    }
    write_json_atomic(output / "plan.json", plan, overwrite=False)
    return plan


def _validate_plan(packet: Path) -> dict[str, Any]:
    plan = read_json_object(packet / "plan.json")
    for filename, key in (
        ("inputs.jsonl", "input_sha256"),
        ("private/references.jsonl", "reference_sha256"),
        ("private/identity-map.jsonl", "identity_map_sha256"),
        ("data-audit.json", "audit_sha256"),
    ):
        if _sha((packet / filename).read_bytes()) != plan[key]:
            raise ValueError(f"frozen artifact changed: {filename}")
    runtime = read_json_object(packet / "runtime.json")
    if (
        plan["source_digest"] != source_fingerprint()
        or plan["runtime_digest"] != digest(runtime_manifest())
        or digest(runtime) != plan["runtime_digest"]
        or plan["native_policies"] != list(NATIVE_POLICIES)
        or plan["project_policies"] != list(PROJECT_POLICIES)
        or plan["lexicon"] != {k: list(v) for k, v in LEXICON.items()}
        or plan["past_markers"] != list(PAST_MARKERS)
    ):
        raise ValueError("source/runtime/policy drift; freeze a new plan")
    return plan


def execute_records(nodes: list[dict[str, Any]], plan_hash: str) -> Iterator[dict[str, Any]]:
    native_seen: dict[tuple[str, str], set[str]] = {}
    project = {name: factory() for name, factory in PROJECT_POLICIES.items()}
    for node in nodes:
        common = {
            "dialogue_id": node["dialogue_id"],
            "observation_id": node["observation_id"],
            "plan_sha256": plan_hash,
            "calls": 0,
            "retries": 0,
            "tokens": 0,
            "cost_cny": 0,
        }
        for policy in NATIVE_POLICIES:
            seen = native_seen.setdefault((policy, node["dialogue_id"]), set())
            yield {
                **common,
                "track": "native_crisis_events",
                "policy": policy,
                "status": "ok",
                "labels": list(native_labels(policy, node, seen)),
            }
        observation = project_observation(node)
        for name, engine in project.items():
            try:
                decision = engine.decide(observation)
                record = {
                    "status": "ok",
                    "decision": decision.model_dump(mode="json"),
                    "observation_sha256": digest(observation.model_dump(mode="json")),
                }
            except Exception as exc:
                record = {"status": "error", "error_code": type(exc).__name__}
            yield {**common, "track": "project_action_diagnostics", "policy": name, **record}


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def native_metrics(
    nodes: list[dict[str, Any]],
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    resamples: int = 1000,
    seed: int = 20261021,
) -> dict[str, Any]:
    """Exact event matching, not ongoing crisis-state prediction or action correctness."""
    ids = [n["observation_id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate visible nodes")
    ref = {r["observation_id"]: r for r in references}
    if len(ref) != len(references) or set(ref) != set(ids):
        raise ValueError("missing/duplicate/unknown native references")
    pred = {p["observation_id"]: p for p in predictions}
    if len(pred) != len(predictions) or set(pred) - set(ids):
        raise ValueError("duplicate/unknown prediction ids")
    counts = {label: [0, 0, 0] for label in LABELS}
    cluster: dict[str, list[int]] = {}
    dialogue_gold: dict[str, set[str]] = {}
    dialogue_pred: dict[str, set[str]] = {}
    failures = 0
    for node in nodes:
        obs = node["observation_id"]
        dialogue = node["dialogue_id"]
        reference = ref[obs]
        if reference["dialogue_id"] != dialogue or set(reference["labels"]) - set(LABELS):
            raise ValueError("reference identity or labels mismatch")
        gold = set(reference["labels"])
        p = pred.get(obs)
        valid = (
            p is not None
            and p.get("dialogue_id") == dialogue
            and p.get("status") == "ok"
            and isinstance(p.get("labels"), list)
            and all(isinstance(label, str) for label in p["labels"])
            and len(p["labels"]) == len(set(p["labels"]))
            and not set(p["labels"]) - set(LABELS)
        )
        emitted = set(p["labels"]) if valid and p else set()
        failures += int(not valid)
        aggregate = cluster.setdefault(dialogue, [0, 0, 0])
        for index, amount in enumerate(
            (len(gold & emitted), len(emitted - gold), len(gold - emitted))
        ):
            aggregate[index] += amount
        for label in LABELS:
            counts[label][0] += int(label in gold and label in emitted)
            counts[label][1] += int(label not in gold and label in emitted)
            counts[label][2] += int(label in gold and label not in emitted)
        dialogue_gold.setdefault(dialogue, set()).update(gold)
        dialogue_pred.setdefault(dialogue, set()).update(emitted)
    total = np.array(list(cluster.values()), dtype=np.int64)
    aggregate = total.sum(axis=0).tolist()
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(resamples):
        a, b, c = total[rng.integers(0, len(total), size=len(total))].sum(axis=0).tolist()
        if 2 * a + b + c:
            bootstrap.append(2 * a / (2 * a + b + c))
    dialogue_counts = [0, 0, 0]
    for key, gold in dialogue_gold.items():
        emitted = dialogue_pred[key]
        for i, amount in enumerate((len(gold & emitted), len(emitted - gold), len(gold - emitted))):
            dialogue_counts[i] += amount
    return {
        "scope": "EXACT_NATIVE_USER_TURN_EVENT_LABELS_NOT_ACTION_NORMS",
        "expected_nodes": len(nodes),
        "retained_nodes": len(nodes),
        "contract_failures_including_missing": failures,
        "micro": _prf(*aggregate),
        "per_label": {label: _prf(*values) for label, values in counts.items()},
        "dialogue_union_micro": _prf(*dialogue_counts),
        "micro_f1_dialogue_cluster_ci95": (
            [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])] if bootstrap else None
        ),
        "bootstrap": {"seed": seed, "resamples": resamples, "clusters": len(cluster)},
        "ci_limitation": "SOURCE_PERSON_POST_TEMPLATE_DEPENDENCE_UNKNOWN_NOT_INDEPENDENT_CASE_CI",
        "listener_label_handling": "EXCLUDED_FROM_USER_METRICS_RETAINED_IN_SOURCE_AUDIT",
    }


def report_external(packet: Path, run: Path, output: Path) -> dict[str, Any]:
    plan = _validate_plan(packet)
    nodes = _read_jsonl(packet / "inputs.jsonl")
    references = _read_jsonl(packet / "private" / "references.jsonl")
    predictions = _read_jsonl(run / "predictions.jsonl")
    expected = list(execute_records(nodes, _sha((packet / "plan.json").read_bytes())))
    if predictions != expected:
        raise ValueError("execution trace is missing, reordered, tampered or not reproducible")
    native = {
        policy: native_metrics(nodes, references, [p for p in predictions if p["policy"] == policy])
        for policy in NATIVE_POLICIES
    }
    project = {}
    for policy in PROJECT_POLICIES:
        records = [p for p in predictions if p["policy"] == policy]
        ok = [p for p in records if p["status"] == "ok"]
        project[policy] = {
            "expected_nodes": len(nodes),
            "recorded_nodes": len(records),
            "successful": len(ok),
            "failed": len(records) - len(ok),
            "primary_actions": dict(
                Counter(p["decision"]["decision"]["primary_action"] for p in ok)
            ),
            "evidence_nodes": sum(bool(p["decision"]["evidence"]) for p in ok),
            "expert_action_norms": "UNAVAILABLE_NO_INPUT_MATCHED_ACTION_GOLD",
            "native_event_f1": "NOT_APPLICABLE_NO_VALIDATED_ACTION_TO_EVENT_MAPPING",
            "response_quality": "NOT_EVALUATED_CHINESE_TEMPLATE_ON_ENGLISH_INPUT",
        }
    report = {
        "version": VERSION,
        "status": "COMPLETE_LOCAL_RULE_DIAGNOSTIC",
        "plan_sha256": _sha((packet / "plan.json").read_bytes()),
        "predictions_sha256": _sha((run / "predictions.jsonl").read_bytes()),
        "dataset_revision": plan["revision"],
        "split": plan["source_split"],
        "native_label_origin": plan["native_label_origin"],
        "dialogues": plan["dialogues"],
        "user_nodes": len(nodes),
        "stage_replay": {"status": "PASS", "verified_records": len(expected)},
        "native_crisis_events": native,
        "project_action_diagnostics": project,
        "model_calls": 0,
        "tokens": 0,
        "cost_cny": 0,
        "model_experiment": "NOT_RUN",
        "expert_action_gold": "NOT_CREATED",
        "paper_reproduction": False,
        "license": "UNSPECIFIED_PENDING_AUTHOR_CONFIRMATION",
        "limitations": plan["limitations"],
    }
    write_json_atomic(output, report, overwrite=False)
    return report


def run_external(packet: Path, output: Path) -> dict[str, Any]:
    plan = _validate_plan(packet)
    if output.exists():
        raise FileExistsError("choose a new run; historical artifacts are immutable")
    output.mkdir(parents=True)
    plan_hash = _sha((packet / "plan.json").read_bytes())
    write_json_atomic(
        output / "started.json",
        {"plan_sha256": plan_hash, "started_at": datetime.now(UTC).isoformat()},
        overwrite=False,
    )
    nodes = _read_jsonl(packet / "inputs.jsonl")
    _jsonl(output / "predictions.jsonl", execute_records(nodes, plan_hash))
    report = report_external(packet, output, output / "report.json")
    write_json_atomic(
        output / "completed.json",
        {
            "status": report["status"],
            "completed_at": datetime.now(UTC).isoformat(),
            "plan_sha256": plan_hash,
            "report_sha256": _sha((output / "report.json").read_bytes()),
            "user_nodes": plan["user_nodes"],
        },
        overwrite=False,
    )
    return report


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _jsonl(path: Path, records: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_snapshot(output: Path, *, revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    """Download only public files at an immutable commit, without HF tokens."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("an immutable 40-character commit is required")
    if output.exists():
        raise FileExistsError("choose a new snapshot directory; no overwrite")
    base = f"https://huggingface.co/datasets/{DATASET}"
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        response = client.get(f"https://huggingface.co/api/datasets/{DATASET}/revision/{revision}")
        response.raise_for_status()
        metadata = response.json()
        if metadata.get("sha") != revision or metadata.get("private") or metadata.get("gated"):
            raise ValueError("snapshot identity/access differs from the public pinned dataset")
        response = client.get(
            f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}?recursive=true"
        )
        response.raise_for_status()
        tree = response.json()
        expected = {entry["path"]: entry["lfs"]["oid"] for entry in tree if entry.get("lfs")}
        paths = ["README.md", *(f"data/{s}-00000-of-00001.parquet" for s in SPLITS)]
        output.mkdir(parents=True)
        files = {}
        for relative in paths:
            response = client.get(f"{base}/resolve/{revision}/{relative}")
            response.raise_for_status()
            content = response.content
            checksum = _sha(content)
            if relative in expected and checksum != expected[relative]:
                raise ValueError(f"upstream LFS checksum mismatch: {relative}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(content)
            files[relative] = {"sha256": checksum, "bytes": len(content)}
    write_json_atomic(output / "upstream-metadata.json", metadata, overwrite=False)
    write_json_atomic(output / "upstream-tree.json", {"entries": tree}, overwrite=False)
    manifest = {
        "version": VERSION,
        "dataset": DATASET,
        "revision": revision,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "files": files,
        "metadata_sha256": _sha((output / "upstream-metadata.json").read_bytes()),
        "tree_sha256": _sha((output / "upstream-tree.json").read_bytes()),
        "license": metadata.get("cardData", {}).get("license") or "UNSPECIFIED",
        "usage": "LOCAL_RESEARCH_ONLY_PENDING_LICENSE_CONFIRMATION",
        "redistribution": "NOT_AUTHORIZED_BY_THIS_TOOL",
        "expert_action_gold": "UNAVAILABLE",
        "independent_blind": False,
    }
    write_json_atomic(output / "snapshot.json", manifest, overwrite=False)
    return manifest


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest = read_json_object(snapshot / "snapshot.json")
    if manifest.get("dataset") != DATASET or not re.fullmatch(
        r"[0-9a-f]{40}", manifest.get("revision", "")
    ):
        raise ValueError("invalid snapshot identity")
    for relative in ["README.md", *(f"data/{s}-00000-of-00001.parquet" for s in SPLITS)]:
        expected = manifest["files"][relative]
        content = (snapshot / relative).read_bytes()
        if _sha(content) != expected["sha256"] or len(content) != expected["bytes"]:
            raise ValueError(f"snapshot file changed: {relative}")
    for filename, key in (
        ("upstream-metadata.json", "metadata_sha256"),
        ("upstream-tree.json", "tree_sha256"),
    ):
        if _sha((snapshot / filename).read_bytes()) != manifest[key]:
            raise ValueError(f"snapshot metadata changed: {filename}")
    return manifest


def parse_dialogues(
    rows: list[dict[str, Any]], *, strict_roles: bool = True
) -> list[list[dict[str, Any]]]:
    """Recover boundaries from strict zero resets, not an invented source dialogue ID."""
    dialogues: list[list[dict[str, Any]]] = []
    for offset, raw in enumerate(rows):
        if set(raw) != {"turn_id", "text", "labels"}:
            raise ValueError("unexpected upstream columns; review the adapter")
        turn = raw["turn_id"]
        if type(turn) is not int or turn < 0:
            raise ValueError("invalid turn index")
        if turn == 0:
            dialogues.append([])
        if not dialogues or turn != len(dialogues[-1]):
            raise ValueError("noncontiguous turn sequence; cannot recover dialogue boundaries")
        text, labels = raw["text"], raw["labels"]
        if not isinstance(text, str) or not isinstance(labels, str):
            raise ValueError("null/non-string upstream text or labels")
        match = re.fullmatch(r"(User|Listener):\s*(.+)", text, re.DOTALL)
        if strict_roles and (match is None or not match[2].strip()):
            raise ValueError("unknown role prefix or empty utterance")
        role = match[1] if match else "Unknown"
        content = match[2] if match else text
        tokens = tuple(part.strip() for part in labels.split(";") if part.strip())
        if len(set(tokens)) != len(tokens) or set(tokens) - set(LABELS):
            raise ValueError("unknown/duplicate native label; review schema, do not guess")
        if strict_roles and dialogues[-1] and dialogues[-1][-1]["role"] == role:
            raise ValueError("nonalternating roles; review source before adaptation")
        dialogues[-1].append(
            {
                "row": offset,
                "turn_id": turn,
                "role": role,
                "text": content,
                "labels": list(tokens),
            }
        )
    if not dialogues:
        raise ValueError("empty source split")
    return dialogues


def load_split(
    snapshot: Path, split: str, *, strict_roles: bool = True
) -> list[list[dict[str, Any]]]:
    if split not in SPLITS:
        raise ValueError("unsupported source split")
    return parse_dialogues(
        pq.read_table(snapshot / f"data/{split}-00000-of-00001.parquet").to_pylist(),
        strict_roles=strict_roles,
    )


def audit_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest = verify_snapshot(snapshot)
    splits = {}
    fingerprints: dict[str, set[str]] = {}
    for split in SPLITS:
        dialogues = load_split(snapshot, split, strict_roles=False)
        rows = [row for dialogue in dialogues for row in dialogue]
        labels = Counter(label for row in rows for label in row["labels"])
        fingerprints[split] = {
            _sha("\n".join(r["text"].strip().casefold() for r in d).encode("utf-8"))
            for d in dialogues
        }
        splits[split] = {
            "dialogues": len(dialogues),
            "starting_roles": dict(Counter(d[0]["role"] for d in dialogues)),
            "turns": len(rows),
            "user_turns": sum(r["role"] == "User" for r in rows),
            "listener_turns": sum(r["role"] == "Listener" for r in rows),
            "unknown_role_rows": [r["row"] for r in rows if r["role"] == "Unknown"],
            "same_role_adjacencies": sum(
                a["role"] == b["role"] for d in dialogues for a, b in pairwise(d)
            ),
            "label_instances": sum(labels.values()),
            "labels": dict(labels),
            "listener_labeled_rows": [
                {"row": r["row"], "turn_id": r["turn_id"], "labels": r["labels"]}
                for r in rows
                if r["role"] == "Listener" and r["labels"]
            ],
            "exact_duplicate_dialogues": len(dialogues) - len(fingerprints[split]),
            "label_origin": "UPSTREAM_CLINICIAN_ANNOTATIONS" if split == "test" else "SYNTHETIC",
        }
    return {
        "version": VERSION,
        "revision": manifest["revision"],
        "license": manifest["license"],
        "splits": splits,
        "dialogue_boundaries": "RECOVERED_FROM_CONTIGUOUS_TURN_ID_ZERO_RESETS",
        "upstream_dialogue_id": "ABSENT",
        "upstream_person_post_template_groups": "NOT_PROVIDED",
        "cross_split_exact_dialogue_overlap": {
            f"{a}__{b}": len(fingerprints[a] & fingerprints[b])
            for i, a in enumerate(SPLITS)
            for b in SPLITS[i + 1 :]
        },
        "semantic_leakage_review": "NOT_PERFORMED",
        "not_a_no_leakage_proof": True,
        "expert_action_gold": "UNAVAILABLE_SOURCE_LABELS_ARE_CRISIS_EVENTS_NOT_ACTION_NORMS",
    }
