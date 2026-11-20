"""Offline design lock and source-review gates; never infer, label or train."""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ecocompanion.application.conversation import (
    CONVERSATION_POLICY_VERSION,
    QUALITY_GATE_VERSION,
    RENDER_PROMPT_VERSION,
)
from ecocompanion.application.renderer import LLMResponseRenderer
from ecocompanion.domain.schemas import SessionStateV1
from ecocompanion.research.contracts import Record
from scripts import distillation as d

ROOT = d.ROOT
SPEC = ROOT / "docs/RENDERER_EVALUATION_SPEC_V1.json"
VERSION = "renderer-human-evaluation-v1.0.0"
GROUPS = ("person_group", "conversation_group", "plot_group", "template_group", "source_group")
CHECKS = frozenset({"rights", "privacy", "source_independence", "semantic_dedup", "fixed_action"})
SETUP_FILES = ("protocol.json", "exposure.json", "case_groups.csv", "readiness.json")
AUDIT_FILES = ("candidates.jsonl", "reviews.jsonl", "audit.json")
ID = r"^[A-Za-z0-9_.:-]{1,160}$"


class History(Record):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Candidate(Record):
    case_id: str = Field(pattern=ID)
    category: str
    text: str = Field(min_length=1)
    history: tuple[History, ...] = ()
    source_id: str = Field(pattern=ID)
    source_revision: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    author_id: str = Field(pattern=ID)
    origin: Literal["independent_author"]
    adult_fictional: Literal[True]
    rights_reference: str = Field(min_length=1)
    privacy_reference: str = Field(min_length=1)
    person_group: str = Field(pattern=ID)
    conversation_group: str = Field(pattern=ID)
    plot_group: str = Field(pattern=ID)
    template_group: str = Field(pattern=ID)
    source_group: str = Field(pattern=ID)


class Review(Record):
    case_id: str = Field(pattern=ID)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(pattern=ID)
    review_reference: str = Field(min_length=1)
    decision: Literal["APPROVE", "REJECT", "PENDING"]
    checks: tuple[str, ...]
    approved_action: str


def canonical(value: Any) -> str:
    return d.digest(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    )


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def grams(text: str) -> set[str]:
    return {text[index : index + 3] for index in range(max(1, len(text) - 2))}


def similar(left: str, right: str) -> bool:
    a, b = normalized(left), normalized(right)
    if not a or not b:
        return False
    x, y = grams(a), grams(b)
    return a == b or len(x & y) / len(x | y) >= 0.85


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def code_identity() -> dict[str, str]:
    paths = (
        Path(__file__),
        ROOT / "scripts/distillation.py",
        ROOT / "backend/src/ecocompanion/core/risk.py",
        ROOT / "backend/src/ecocompanion/core/policy.py",
        ROOT / "backend/src/ecocompanion/application/conversation.py",
        ROOT / "backend/src/ecocompanion/application/renderer.py",
        ROOT / "backend/src/ecocompanion/domain/enums.py",
        ROOT / "backend/src/ecocompanion/domain/schemas.py",
        ROOT / "backend/src/ecocompanion/research/contracts.py",
    )
    return {path.relative_to(ROOT).as_posix(): d.digest(path.read_bytes()) for path in paths}


def seal(output: Path, files: tuple[str, ...], **metadata: Any) -> dict[str, Any]:
    manifest = {
        "version": VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "files": {name: d.digest((output / name).read_bytes()) for name in files},
        **metadata,
    }
    d.write_json(output / "manifest.json", manifest)
    return manifest


def verify(directory: Path, files: tuple[str, ...]) -> dict[str, Any]:
    manifest = d.read_json(directory / "manifest.json")
    if manifest.get("version") != VERSION or set(manifest["files"]) != set(files):
        raise ValueError("packet version/file contract changed")
    if any(d.digest((directory / name).read_bytes()) != manifest["files"][name] for name in files):
        raise ValueError("frozen packet changed")
    return dict(manifest)


def load_setup(setup: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify(setup, SETUP_FILES)
    if manifest["code_identity"] != code_identity():
        raise ValueError("evaluation/renderer code changed; prepare a new version")
    protocol = d.read_json(setup / "protocol.json")
    if protocol["specification"] != d.read_json(SPEC):
        raise ValueError("evaluation specification changed; prepare a new version")
    versions = {
        "policy": CONVERSATION_POLICY_VERSION,
        "prompt": RENDER_PROMPT_VERSION,
        "quality_gate": QUALITY_GATE_VERSION,
    }
    if protocol["runtime_versions"] != versions:
        raise ValueError("runtime versions changed")
    return protocol, d.read_json(setup / "exposure.json")


def prepare(output: Path, *, data_root: Path = ROOT) -> dict[str, Any]:
    specification = d.read_json(SPEC)
    seeds_path = data_root / "docs/DISTILLATION_PILOT_SEEDS_V1.json"
    diagnostic_path = data_root / "artifacts/research/judge-diagnostic-v2/cases.json"
    calibration_path = data_root / "artifacts/research/human-calibration-20261023-v1/items.jsonl"
    inventory_path = data_root / "docs/DATA_SOURCE_INVENTORY_20261025.md"
    seeds = d.read_json(seeds_path)
    diagnostic = d.read_json(diagnostic_path)["cases"]
    calibration = d.jsonl(calibration_path)
    corpus = [
        {"source_id": "distillation_pilot_v1", "text": text}
        for item in seeds
        for text in [item["text"], *(m["content"] for m in item.get("history", []))]
    ]
    corpus += [
        {"source_id": "judge_diagnostic_30", "text": item[key]}
        for item in diagnostic
        for key in ("user", "assistant")
    ]
    corpus += [
        {"source_id": "counselbench", "text": item[key]}
        for item in calibration
        for key in ("user", "assistant")
    ]
    # Other used corpora are blocked at source level, not falsely claimed as text-screened.
    exposure = {
        "blocked_source_ids": specification["source_isolation"]["exclude_used_sources"],
        "text_screened_source_ids": sorted({item["source_id"] for item in corpus}),
        "source_references": {
            path.relative_to(data_root).as_posix(): d.digest(path.read_bytes())
            for path in (seeds_path, diagnostic_path, calibration_path, inventory_path)
        },
        "corpus": corpus,
        "complete_global_exposure_history": False,
        "human_alias_and_semantic_review_required": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    d.write_json(
        output / "protocol.json",
        {
            "specification": specification,
            "runtime_versions": {
                "policy": CONVERSATION_POLICY_VERSION,
                "prompt": RENDER_PROMPT_VERSION,
                "quality_gate": QUALITY_GATE_VERSION,
            },
        },
    )
    d.write_json(output / "exposure.json", exposure)
    fields = ["case_id", "source_id", "source_revision", "split", *GROUPS, "review_status"]
    with (output / "case_groups.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in seeds:
            writer.writerow(
                {
                    "case_id": item["id"],
                    "source_id": "distillation_pilot_v1",
                    "source_revision": d.digest(seeds_path.read_bytes()),
                    "split": item["split"],
                    **dict.fromkeys(GROUPS, "UNKNOWN"),
                    "conversation_group": item["group"],
                    "review_status": "DEVELOPMENT_ONLY_NOT_INDEPENDENT",
                }
            )
    d.write_json(
        output / "readiness.json",
        {
            "design_locked": True,
            "independent_evaluation_cases": 0,
            "human_source_reviews": 0,
            "human_response_ratings": 0,
            "model_runs": 0,
            "expert_gold": False,
            "ready_for_evaluation": False,
            "blockers": [
                "NEED_40_INDEPENDENT_REVIEWED_CASES",
                "NEED_TWO_REAL_SOURCE_REVIEWERS",
                "NEED_INFERENCE_IDENTITY_REGISTRATION_AND_BLIND_RATING_PACKET",
            ],
        },
    )
    intake = output / "intake"
    intake.mkdir()
    write_jsonl(intake / "candidates.jsonl", [])
    write_jsonl(intake / "reviews.jsonl", [])
    d.write_json(intake / "candidate.schema.json", Candidate.model_json_schema())
    d.write_json(intake / "review.schema.json", Review.model_json_schema())
    return seal(output, SETUP_FILES, code_identity=code_identity(), expert_gold=False)


def components(cases: list[Candidate]) -> dict[str, str]:
    parent = {case.case_id: case.case_id for case in cases}

    def root(identifier: str) -> str:
        while parent[identifier] != identifier:
            identifier = parent[identifier]
        return identifier

    owners: dict[tuple[str, str], str] = {}
    for case in cases:
        for field in GROUPS:
            key = (field, normalized(getattr(case, field)))
            if key in owners:
                parent[root(case.case_id)] = root(owners[key])
            else:
                owners[key] = case.case_id
    return {case.case_id: root(case.case_id) for case in cases}


def inspect_cases(
    protocol: dict[str, Any],
    exposure: dict[str, Any],
    cases: list[Candidate],
    reviews: list[Review],
) -> dict[str, Any]:
    spec = protocol["specification"]
    errors: list[dict[str, str]] = []

    def fail(code: str, identifier: str = "BATCH") -> None:
        errors.append({"code": code, "case_id": identifier})

    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate identities")
    if len(cases) != spec["sample_plan"]["cases"]:
        fail("CASE_COUNT_MUST_BE_40")
    expected = dict.fromkeys(spec["sample_plan"]["categories"], 5)
    if dict(Counter(case.category for case in cases)) != expected:
        fail("NEED_FIVE_CASES_PER_CATEGORY")
    clusters = components(cases)
    if len(set(clusters.values())) < spec["sample_plan"]["minimum_connected_source_groups"]:
        fail("INSUFFICIENT_CONNECTED_SOURCE_GROUPS")
    review_keys = [(review.case_id, normalized(review.reviewer_id)) for review in reviews]
    if len(set(review_keys)) != len(review_keys) or any(r.case_id not in ids for r in reviews):
        raise ValueError("duplicate reviewer/case or unknown review case")
    for case in cases:
        data = case.model_dump(mode="json")
        visible = [case.text, *(m.content for m in case.history)]
        if d.KEY_PATTERN.search(json.dumps(data, ensure_ascii=False)):
            raise ValueError("credentials are forbidden in evaluation input")
        if any(d.UNSUPPORTED.search(text) for text in visible):
            fail("CRISIS_EXCLUDED_FROM_RENDERER_PILOT", case.case_id)
        if any(
            not str(data[field]).strip()
            or normalized(str(data[field])).startswith(("unknown", "pending", "unverified"))
            for field in (
                *GROUPS,
                "source_id",
                "source_revision",
                "source_record_id",
                "author_id",
                "rights_reference",
                "privacy_reference",
            )
        ):
            fail("UNKNOWN_PROVENANCE_OR_GROUP", case.case_id)
        if normalized(case.source_id) in {
            normalized(source) for source in exposure["blocked_source_ids"]
        }:
            fail("SOURCE_ALREADY_USED", case.case_id)
        if any(similar(text, old["text"]) for text in visible for old in exposure["corpus"]):
            fail("EXACT_OR_NEAR_EXPOSURE_MATCH", case.case_id)
        observation, _, outcome = d.context(data)
        if outcome.plan.escalation_level.value not in {"L0", "L1"}:
            fail("ELEVATED_ACTION_OUT_OF_SCOPE", case.case_id)
        bound = [r for r in reviews if r.case_id == case.case_id]
        valid = [
            r
            for r in bound
            if r.decision == "APPROVE"
            and r.candidate_sha256 == canonical(data)
            and frozenset(r.checks) == CHECKS
            and len(r.checks) == len(CHECKS)
            and normalized(r.reviewer_id) != normalized(case.author_id)
            and r.review_reference.strip()
            and not normalized(r.review_reference).startswith(("unknown", "pending", "unverified"))
            and not normalized(r.reviewer_id).startswith(("unknown", "pending", "unverified"))
            and r.approved_action == outcome.plan.primary_action.value
        ]
        if len(valid) != 2 or len(bound) != 2:
            fail("NEED_TWO_BOUND_INDEPENDENT_HUMAN_SOURCE_APPROVALS", case.case_id)
        # A reviewer must explicitly approve the action; automatic classification is not Gold.
        if not observation.current_observation.message.strip():
            fail("EMPTY_VISIBLE_INPUT", case.case_id)
    for index, left in enumerate(cases):
        for right in cases[index + 1 :]:
            if similar(left.text, right.text):
                fail("WITHIN_BATCH_EXACT_OR_NEAR_DUPLICATE", right.case_id)
            if clusters[left.case_id] != clusters[right.case_id] and any(
                len(normalized(a)) >= 12 and len(normalized(b)) >= 12 and similar(a, b)
                for a in [left.text, *(m.content for m in left.history)]
                for b in [right.text, *(m.content for m in right.history)]
            ):
                fail("SHARED_VISIBLE_CONTEXT_ACROSS_SOURCE_GROUPS", right.case_id)
    return {
        "version": VERSION,
        "eligible_for_case_freeze": not errors,
        "cases": len(cases),
        "connected_source_groups": len(set(clusters.values())),
        "clusters": clusters,
        "errors": errors,
        "source_approval_records": len(reviews),
        "identity_checks": "CURATOR_ATTESTATIONS_NOT_AUTOMATIC_PERSON_VERIFICATION",
        "independence": "KNOWN_EXPOSURE_SCREEN_PLUS_HUMAN_ATTESTATIONS_NOT_ABSOLUTE_PROOF",
        "expert_gold": False,
        "response_labels": 0,
        "model_runs": 0,
    }


def audit(setup: Path, candidates: Path, reviews: Path, output: Path) -> dict[str, Any]:
    protocol, exposure = load_setup(setup)
    cases = [Candidate.model_validate(item) for item in d.jsonl(candidates)]
    votes = [Review.model_validate(item) for item in d.jsonl(reviews)]
    if any(d.KEY_PATTERN.search(vote.model_dump_json()) for vote in votes):
        raise ValueError("credentials are forbidden in review metadata")
    report = inspect_cases(protocol, exposure, cases, votes)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "candidates.jsonl", [case.model_dump(mode="json") for case in cases])
    write_jsonl(output / "reviews.jsonl", [vote.model_dump(mode="json") for vote in votes])
    d.write_json(output / "audit.json", report)
    seal(
        output,
        AUDIT_FILES,
        setup_manifest_sha256=d.digest((setup / "manifest.json").read_bytes()),
    )
    return report


def freeze(setup: Path, audited: Path, output: Path) -> dict[str, Any]:
    protocol, exposure = load_setup(setup)
    manifest = verify(audited, AUDIT_FILES)
    if manifest["setup_manifest_sha256"] != d.digest((setup / "manifest.json").read_bytes()):
        raise ValueError("source audit belongs to a different setup")
    cases = [Candidate.model_validate(item) for item in d.jsonl(audited / "candidates.jsonl")]
    reviews = [Review.model_validate(item) for item in d.jsonl(audited / "reviews.jsonl")]
    report = inspect_cases(protocol, exposure, cases, reviews)
    if report != d.read_json(audited / "audit.json") or not report["eligible_for_case_freeze"]:
        raise ValueError("source audit blocked; no evaluation set frozen")
    if output.parent.name != "private":
        raise ValueError("evaluation cases and mappings must be in a private directory")
    tasks = []
    for case in cases:
        observation, assessment, outcome = d.context(case.model_dump(mode="json"))
        tasks.append(
            {
                "task_id": canonical(case.model_dump(mode="json"))[:32],
                "messages": [
                    dict(message)
                    for message in LLMResponseRenderer._messages(
                        observation=observation,
                        assessment=assessment,
                        policy=outcome,
                        session=SessionStateV1(session_id="synthetic-renderer-development"),
                        memory_snippets=(),
                    )
                ],
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "tasks.jsonl", tasks)
    write_jsonl(output / "cases.jsonl", [case.model_dump(mode="json") for case in cases])
    write_jsonl(output / "source_reviews.jsonl", [r.model_dump(mode="json") for r in reviews])
    d.write_json(output / "source_audit.json", report)
    return seal(
        output,
        ("tasks.jsonl", "cases.jsonl", "source_reviews.jsonl", "source_audit.json"),
        setup_manifest_sha256=d.digest((setup / "manifest.json").read_bytes()),
        audited_manifest_sha256=d.digest((audited / "manifest.json").read_bytes()),
        cases=len(cases),
        independent_source_status="CURATOR_ATTESTED_NOT_AUTOMATICALLY_VERIFIED",
        ready_for_model_evaluation=False,
        remaining_gate="INFERENCE_IDENTITY_REGISTRATION_AND_BLIND_RATING_PACKET",
        model_runs=0,
        human_response_ratings=0,
        expert_gold=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("audit")
    for name in ("setup", "candidates", "reviews", "output"):
        check.add_argument("--" + name, type=Path, required=True)
    lock = commands.add_parser("freeze")
    for name in ("setup", "audited", "output"):
        lock.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.output)
        elif args.command == "audit":
            result = audit(args.setup, args.candidates, args.reviews, args.output)
        else:
            result = freeze(args.setup, args.audited, args.output)
    except (ValueError, OSError, KeyError, TypeError):
        parser.exit(1, "Evaluation input/contract error; no inference or training performed.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
