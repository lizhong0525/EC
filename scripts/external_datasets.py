"""Verified intake for external mental-health benchmarks; no external label becomes project Gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ecocompanion.experiments.io import read_json_object, write_json_atomic

VERSION = "external-datasets-v1.0.0"
MHB_SOURCE = "mentalhealthbench"
MHB_URL = "https://cdn.openai.com/ctf-cdn/OAI_MentalHealthBench.zip"
MHB_ARCHIVE_SHA256 = "a3e94c7786bd514952c542d6a5694bb24055f136b417f15c5c6ab942976d7474"
MHB_MEMBERS = {
    "mentalhealthbench/README.md": "README.md",
    "mentalhealthbench/LICENSE": "LICENSE",
    "mentalhealthbench/mentalhealthbench_eval.jsonl": "mentalhealthbench_eval.jsonl",
}
MHB_RECORDS = 1215
MHB_RUBRIC_ITEMS = 5262
MHB_ACUITY = {"non_acute", "high_acuity", "emergent"}
MHB_PROFILES = {"adult", "teen", "clinician", "caregiver"}
MHB_LANGUAGES = {
    "en", "es", "hi", "ar", "pt", "de", "id", "it", "fa", "tr", "zh",
}
MHB_FIELDS = {
    "id", "conversation", "rubric_items", "acuity", "user_profile",
    "language", "has_prior_context", "canary_string",
}
CB_SOURCE = "counselbench-eval"
CB_REVISION = "8d56a96ea1de3f3f190f77f4ca9bc3503d731af7"
CB_BLOB = "7b5dc395091779e1724cbc03a9e166743d719365"
CB_URL = (
    "https://hf-mirror.com/datasets/izi-ano/CounselBench-Eval/resolve/"
    + CB_REVISION
    + "/counselbench_eval.csv"
)
CB_API = "https://hf-mirror.com/api/datasets/izi-ano/CounselBench-Eval"
CB_ROWS = 2000
CB_QUESTIONS = 100
CB_COLUMNS = (
    "questionID", "questionTitle", "questionText", "response", "topic",
    "responder", "survey_id", "overall_score", "overall_reason",
    "empathy_score", "specificity_score", "medical_advice_score",
    "factual_consistency_score", "toxicity_score",
)
CB_SCORE_COLUMNS = (
    "overall_score", "empathy_score", "specificity_score",
    "medical_advice_score", "factual_consistency_score", "toxicity_score",
)
USE_NOTICE = "LOCAL_ACADEMIC_RESEARCH_ONLY_NO_REDISTRIBUTION"
CLAIMS = {
    "new_project_action_gold": False,
    "expert_action_labels": False,
    "clinical_or_adolescent_deployment_claim": False,
    "remote_model_calls": 0,
    "upstream_code_execution": False,
}


class SourceUnavailable(RuntimeError):
    """Raised when a registered source cannot be fetched from this host."""


@dataclass(frozen=True)
class SourceSpec:
    key: str
    title: str
    publisher: str
    priority: int
    access: str
    status: str
    license: str
    identity_kind: str
    identity: str
    url: str
    notes: str


SOURCE_ORDER = (
    SourceSpec(
        key=MHB_SOURCE,
        title="MentalHealthBench: expert rubric benchmark for mental-health conversations",
        publisher="OpenAI",
        priority=1,
        access="PUBLIC_CDN",
        status="READY",
        license="MIT",
        identity_kind="ARCHIVE_SHA256",
        identity=MHB_ARCHIVE_SHA256,
        url=MHB_URL,
        notes="1215 synthetic conversations and 5262 expert rubric criteria.",
    ),
    SourceSpec(
        key="mentat-triage",
        title="MENTAT clinician-annotated psychiatric decision benchmark (triage subset)",
        publisher="Stanford University",
        priority=2,
        access="UNAVAILABLE_NETWORK",
        status="BLOCKED",
        license="UNVERIFIED",
        identity_kind="NONE",
        identity="",
        url="https://www.robotics.stanford.edu/blog/mentat/",
        notes="Project page unreachable from this host; no bytes acquired and no schema assumed.",
    ),
    SourceSpec(
        key=CB_SOURCE,
        title="CounselBench-Eval: expert evaluation of counselling responses",
        publisher="llm-eval-mental-health",
        priority=3,
        access="PUBLIC_HF_MIRROR",
        status="READY",
        license="CC-BY-NC-ND-4.0",
        identity_kind="HF_GIT_BLOB",
        identity=CB_BLOB,
        url=CB_URL,
        notes="2000 human evaluations on 100 questions; HuggingFace reached via mirror.",
    ),
    SourceSpec(
        key="reachout",
        title="ReachOut / CLPsych 2016 youth community triage",
        publisher="CLPsych shared task organisers",
        priority=4,
        access="CONTROLLED_ACCESS",
        status="BLOCKED",
        license="ACCESS_TERMS_REQUIRED",
        identity_kind="NONE",
        identity="",
        url="http://clpsych.org/shared-task-2016/",
        notes="Reddit-derived data requires an approved access request; nothing fetched.",
    ),
    SourceSpec(
        key="sos-hl-1k",
        title="SOS-HL-1K Chinese suicide-risk posts",
        publisher="Hongzhi Qi",
        priority=99,
        access="DEFERRED",
        status="DEFERRED",
        license="UNVERIFIED",
        identity_kind="NONE",
        identity="",
        url="https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval",
        notes="Deferred: overlaps with the PsyCrisis source already in use; audit before intake.",
    ),
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git_blob_sha(content: bytes) -> str:
    # Git object identity is a content address, not an authenticity signature.
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def registry() -> tuple[SourceSpec, ...]:
    return SOURCE_ORDER


def source(key: str) -> SourceSpec:
    for spec in SOURCE_ORDER:
        if spec.key == key:
            return spec
    raise KeyError(key)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inventory_mentalhealthbench(content: bytes) -> dict[str, Any]:
    """Validate the pinned JSONL and return only counted properties."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}") from exc
        _require(isinstance(record, dict), "record must be an object")
        _require(MHB_FIELDS <= set(record), "record is missing documented fields")
        messages = record["conversation"]["messages"]
        _require(bool(messages), "conversation must contain messages")
        _require(
            all(m.get("role") in {"system", "user", "assistant"} for m in messages),
            "unknown message role",
        )
        _require(record["acuity"] in MHB_ACUITY, "unknown acuity label")
        _require(record["user_profile"] in MHB_PROFILES, "unknown user profile")
        _require(record["language"] in MHB_LANGUAGES, "unknown language tag")
        _require(
            isinstance(record["has_prior_context"], bool),
            "has_prior_context must be boolean",
        )
        for item in record["rubric_items"]:
            _require(
                {"criterion_text", "points", "behavior_axis"} <= set(item),
                "rubric item is missing documented fields",
            )
            _require(isinstance(item["points"], int), "rubric points must be integers")
        records.append(record)
    _require(len(records) == MHB_RECORDS, "unexpected record count for the pinned archive")
    ids = [record["id"] for record in records]
    _require(len(set(ids)) == len(ids), "duplicate record ids")
    _require(all(len(str(i)) == 32 for i in ids), "record ids must be 32 hexadecimal characters")
    rubric_items = [item for record in records for item in record["rubric_items"]]
    _require(len(rubric_items) == MHB_RUBRIC_ITEMS, "unexpected rubric item count")
    return {
        "records": len(records),
        "unique_ids": len(set(ids)),
        "languages": dict(sorted(Counter(r["language"] for r in records).items())),
        "acuity": dict(sorted(Counter(r["acuity"] for r in records).items())),
        "user_profile": dict(sorted(Counter(r["user_profile"] for r in records).items())),
        "has_prior_context": dict(
            sorted(Counter(str(r["has_prior_context"]).lower() for r in records).items())
        ),
        "rubric_items": len(rubric_items),
        "behavior_axis": dict(
            sorted(Counter(i["behavior_axis"] for i in rubric_items).items())
        ),
        "rubric_points": sum(i["points"] for i in rubric_items),
        "messages_min": min(len(r["conversation"]["messages"]) for r in records),
        "messages_max": max(len(r["conversation"]["messages"]) for r in records),
        "canary_strings": len({r["canary_string"] for r in records}),
    }


def _score_summary(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    """Summarise a mixed column: native numbers are counted, free text is preserved."""

    numeric: list[int] = []
    text: Counter[str] = Counter()
    for row in rows:
        raw = (row.get(column) or "").strip()
        if not raw:
            continue
        try:
            numeric.append(int(float(raw)))
        except ValueError:
            text[raw] += 1
    summary: dict[str, Any] = {
        "answered": len(numeric) + sum(text.values()),
        "numeric": len(numeric),
        "text_values": dict(sorted(text.items())),
    }
    if numeric:
        summary["mean"] = round(sum(numeric) / len(numeric), 3)
        summary["min"] = min(numeric)
        summary["max"] = max(numeric)
    else:
        summary["mean"] = None
        summary["min"] = None
        summary["max"] = None
    return summary
def inventory_counselbench(content: bytes) -> dict[str, Any]:
    """Validate the pinned expert-evaluation CSV and summarise native scores."""

    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    _require(reader.fieldnames is not None, "missing CSV header")
    missing = [column for column in CB_COLUMNS if column not in reader.fieldnames]
    _require(not missing, f"missing documented columns: {missing}")
    rows = list(reader)
    _require(len(rows) == CB_ROWS, "unexpected row count for the pinned CSV")
    questions = {row["questionID"] for row in rows}
    _require(len(questions) == CB_QUESTIONS, "unexpected question count")
    return {
        "rows": len(rows),
        "unique_questions": len(questions),
        "responders": dict(sorted(Counter(row["responder"] for row in rows).items())),
        "topics": len({row["topic"] for row in rows}),
        "survey_ids": len({row["survey_id"] for row in rows}),
        "scores": {column: _score_summary(rows, column) for column in CB_SCORE_COLUMNS},
    }


def _inventory_for(spec: SourceSpec, name: str, content: bytes) -> dict[str, Any]:
    if spec.key == MHB_SOURCE and name.endswith(".jsonl"):
        return inventory_mentalhealthbench(content)
    if spec.key == CB_SOURCE and name.endswith(".csv"):
        return inventory_counselbench(content)
    raise ValueError("no inventory rule for this file")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def _snapshot_documents(
    spec: SourceSpec,
    files: dict[str, bytes],
    inventory: dict[str, Any],
    inventory_file: str,
    retrieved_at: str,
    extra: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    entries: dict[str, dict[str, Any]] = {}
    for name, content in sorted(files.items()):
        entry: dict[str, Any] = {"bytes": len(content), "sha256": sha256(content)}
        if spec.key == CB_SOURCE and name.endswith(".csv"):
            entry["git_blob"] = git_blob_sha(content)
        entries[name] = entry
    document = {
        "version": VERSION,
        "source": spec.key,
        "title": spec.title,
        "publisher": spec.publisher,
        "priority": spec.priority,
        "access_mode": spec.access,
        "retrieved_at": retrieved_at,
        "identity_kind": spec.identity_kind,
        "identity": spec.identity,
        "license": spec.license,
        "license_evidence": extra.get("license_evidence", ""),
        "inventory_file": inventory_file,
        "files": entries,
        "inventory": inventory,
        "verification": extra.get("verification", ""),
        "use": USE_NOTICE,
        "redistribution": "NOT_AUTHORIZED_BY_THIS_TOOL",
        "upstream_source_rights_independently_verified": False,
        "download_attempts": extra.get("download_attempts", []),
        **CLAIMS,
    }
    return document, entries


def verify(snapshot: Path) -> dict[str, Any]:
    """Recompute every recorded hash and the inventory; drift is an error."""

    document = read_json_object(snapshot / "snapshot.json")
    _require(document.get("version") == VERSION, "unexpected snapshot version")
    spec = source(document["source"])
    for name, entry in document["files"].items():
        target = snapshot / name
        _require(target.is_file(), "snapshot file is missing")
        content = target.read_bytes()
        if len(content) != entry["bytes"] or sha256(content) != entry["sha256"]:
            raise ValueError(f"snapshot file changed: {name}")
        if "git_blob" in entry and git_blob_sha(content) != entry["git_blob"]:
            raise ValueError(f"snapshot git blob mismatch: {name}")
    if spec.identity_kind == "ARCHIVE_SHA256":
        archive = snapshot / "archive.zip"
        _require(archive.is_file(), "archive missing")
        _require(sha256(archive.read_bytes()) == spec.identity, "archive identity changed")
    inventory_file = document["inventory_file"]
    recomputed = _inventory_for(spec, inventory_file, (snapshot / inventory_file).read_bytes())
    _require(recomputed == document["inventory"], "inventory changed")
    return document


def _download(client: httpx.Client, url: str, attempts: list[dict[str, Any]]) -> bytes:
    response = client.get(url)
    attempts.append(
        {"url": url, "status": response.status_code, "bytes": len(response.content)}
    )
    response.raise_for_status()
    return response.content


def write_snapshot(
    output: Path,
    spec: SourceSpec,
    files: dict[str, bytes],
    inventory: dict[str, Any],
    inventory_file: str,
    retrieved_at: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Write one immutable snapshot; refuse to replace existing history."""

    if output.exists():
        raise FileExistsError("choose a new snapshot directory; never overwrite history")
    document, _ = _snapshot_documents(
        spec, files, inventory, inventory_file, retrieved_at, extra
    )
    output.mkdir(parents=True)
    for name, content in sorted(files.items()):
        _write_once(output / name, content)
    write_json_atomic(output / "snapshot.json", document, overwrite=False)
    return document


def fetch(key: str, output: Path) -> dict[str, Any]:
    """Download one READY source; verification happens before any write."""

    spec = source(key)
    if spec.status != "READY":
        raise SourceUnavailable(f"{spec.key}: {spec.access}")
    attempts: list[dict[str, Any]] = []
    retrieved_at = datetime.now(UTC).isoformat()
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        if spec.key == MHB_SOURCE:
            archive = _download(client, MHB_URL, attempts)
            _require(sha256(archive) == MHB_ARCHIVE_SHA256, "archive sha256 differs from pin")
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                members = {info.filename for info in bundle.infolist() if not info.is_dir()}
                _require(members == set(MHB_MEMBERS), "unexpected archive members")
                files = {
                    MHB_MEMBERS[name]: bundle.read(name) for name in sorted(MHB_MEMBERS)
                }
            files["archive.zip"] = archive
            inventory_file = "mentalhealthbench_eval.jsonl"
            inventory = inventory_mentalhealthbench(files[inventory_file])
            extra = {
                "license_evidence": "LICENSE in archive plus dataset card license: mit",
                "verification": "PINNED_ARCHIVE_SHA256_AND_MEMBER_SHA256",
                "download_attempts": attempts,
            }
        elif spec.key == CB_SOURCE:
            metadata = _download(client, CB_API, attempts)
            content = _download(client, CB_URL, attempts)
            _require(git_blob_sha(content) == CB_BLOB, "CSV git blob differs from pin")
            files = {
                "counselbench_eval.csv": content,
                "upstream-metadata.json": metadata,
            }
            inventory_file = "counselbench_eval.csv"
            inventory = inventory_counselbench(content)
            extra = {
                "license_evidence": "dataset card license: cc-by-nc-nd-4.0",
                "verification": "PINNED_GIT_BLOB_AND_LOCAL_SHA256_VIA_MIRROR",
                "download_attempts": attempts,
            }
        else:  # pragma: no cover - the registry guards this branch
            raise SourceUnavailable(spec.key)
    return write_snapshot(
        output, spec, files, inventory, inventory_file, retrieved_at, extra
    )

def plan(root: Path | None = None) -> list[dict[str, Any]]:
    """Ordered intake plan; snapshot presence is read from disk, never assumed."""

    rows: list[dict[str, Any]] = []
    for spec in sorted(SOURCE_ORDER, key=lambda item: item.priority):
        snapshot = None
        if root is not None:
            matches = sorted((root / spec.key).glob("*/snapshot.json"))
            snapshot = matches[-1].parent.as_posix() if matches else None
        rows.append(
            {
                "priority": spec.priority,
                "source": spec.key,
                "status": spec.status,
                "access": spec.access,
                "license": spec.license,
                "snapshot": snapshot,
                "notes": spec.notes,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    plan_command = commands.add_parser("plan")
    plan_command.add_argument("--root", type=Path, default=Path("artifacts/external"))
    fetch_command = commands.add_parser("fetch")
    fetch_command.add_argument("source")
    fetch_command.add_argument("--output", type=Path, required=True)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--snapshot", type=Path, required=True)
    inventory_command = commands.add_parser("inventory")
    inventory_command.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        payload: Any = [
            {"priority": s.priority, "source": s.key, "status": s.status, "access": s.access}
            for s in sorted(SOURCE_ORDER, key=lambda item: item.priority)
        ]
    elif args.command == "plan":
        payload = plan(args.root)
    elif args.command == "fetch":
        payload = fetch(args.source, args.output)
    elif args.command == "inventory":
        document = verify(args.snapshot)
        payload = {"source": document["source"], "inventory": document["inventory"]}
    else:
        payload = verify(args.snapshot)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
