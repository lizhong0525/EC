"""Local MentalHealthBench reply-rubric evaluation, separate from action Gold.

Public source rubrics are kept outside generation requests. The paper's Judge
identity is not silently replaced: a local run is explicitly an uncalibrated
diagnostic, even when it follows the source scoring formula.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research.annotation import digest
from ecocompanion.research.mhb_protocol import (
    ENDPOINT,
    OFFICIAL_COMPLETIONS,
    PAPER_SHA256,
    PAPER_URL,
    READ_TIMEOUT_SECONDS,
    SEED,
    SOURCE,
    VERSION,
    generation_request,
    grader_request,
    sha256,
    visible_conversation,
)

DATA_FILE = "mentalhealthbench_eval.jsonl"
PAPER_MANIFEST = "paper-manifest.json"
PAPER_SNAPSHOT_VERSION = "mentalhealthbench-paper-v1.0.0"
GENERATION_MAX_TOKENS = 2048
GRADING_MAX_TOKENS = 1200
BOOTSTRAP_RESAMPLES = 1000
MIN_CI_TASKS = 30
CLAIMS: dict[str, Any] = {
    "new_project_action_gold": 0,
    "expert_action_metrics": None,
    "simulation_state_recovery": "NOT_APPLICABLE_EXTERNAL_NEXT_REPLY_RUBRICS",
    "independent_blind_test": False,
    "public_reconstructible_source": True,
    "clinical_or_adolescent_deployment_claim": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON at line {number}") from exc
        _require(isinstance(value, dict), f"{path}: expected an object at line {number}")
        records.append(value)
    return records


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records
    ).encode()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Create a new local artifact; never replace a previously recorded trace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_jsonl_bytes(records).decode())


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest = read_json_object(snapshot / "snapshot.json")
    _require(
        manifest.get("version") == "external-datasets-v1.0.0"
        and manifest.get("source") == SOURCE,
        "unexpected MentalHealthBench source snapshot",
    )
    _require(
        sha256((snapshot / "archive.zip").read_bytes()) == manifest.get("identity"),
        "source archive identity changed",
    )
    for name, item in manifest["files"].items():
        content = (snapshot / name).read_bytes()
        _require(
            len(content) == item["bytes"] and sha256(content) == item["sha256"],
            f"source file changed: {name}",
        )
    records = _read_jsonl(snapshot / DATA_FILE)
    _require(len(records) == 1215, "unexpected pinned task count")
    _require(
        sum(len(record["rubric_items"]) for record in records) == 5262,
        "unexpected pinned rubric count",
    )
    return manifest


def freeze_paper(output: Path) -> dict[str, Any]:
    """Fetch only the pinned public paper; no benchmark data are uploaded."""

    if output.exists():
        raise FileExistsError("use a new paper snapshot; do not overwrite history")
    retrieved_at = datetime.now(UTC).isoformat()
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        response = client.get(PAPER_URL)
        response.raise_for_status()
        content = response.content
    _require(sha256(content) == PAPER_SHA256, "paper SHA-256 differs from the frozen identity")
    manifest = {
        "version": PAPER_SNAPSHOT_VERSION,
        "source": SOURCE,
        "paper_url": PAPER_URL,
        "paper_sha256": PAPER_SHA256,
        "bytes": len(content),
        "retrieved_at": retrieved_at,
        "method_identity": "PINNED_PDF_BYTES_NOT_A_CLAIM_OF_OFFICIAL_MODEL_REPRODUCTION",
    }
    _write_once(output / "paper.pdf", content)
    write_json_atomic(output / PAPER_MANIFEST, manifest, overwrite=False)
    return manifest


def verify_paper(paper: Path) -> dict[str, Any]:
    manifest_path = paper / PAPER_MANIFEST
    if manifest_path.exists():
        manifest = read_json_object(manifest_path)
        if manifest.get("version") == "mentalhealthbench-external-v1.0.0":
            # Historical paper snapshots accidentally shared the evaluation
            # version. A reviewed successor manifest preserves the same bytes.
            manifest_path = paper / "paper-manifest-v2.json"
            manifest = read_json_object(manifest_path)
    else:
        manifest = read_json_object(manifest_path)
    content = (paper / "paper.pdf").read_bytes()
    _require(
        manifest.get("version") == PAPER_SNAPSHOT_VERSION
        and manifest.get("paper_url") == PAPER_URL
        and manifest.get("paper_sha256") == PAPER_SHA256
        and manifest.get("bytes") == len(content)
        and sha256(content) == PAPER_SHA256,
        "paper identity or frozen manifest changed",
    )
    return manifest


def implementation_digest() -> str:
    root = Path(__file__).resolve().parent
    return digest(
        {
            path.name: sha256(path.read_bytes())
            for path in (
                root / "mhb.py",
                root / "mhb_protocol.py",
                root / "annotation.py",
                root.parent / "experiments" / "io.py",
            )
        }
    )


def _task_id(source_id: str) -> str:
    return "mhb-" + sha256(f"{VERSION}|{source_id}".encode())[:24]


def _sample_seed(task_id: str, sample_index: int) -> int:
    value = sha256(f"{SEED}|{task_id}|{sample_index}".encode())[:8]
    return int(value, 16)


def _generation_options(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": protocol["generation_temperature"],
        "num_ctx": 16384,
        "num_predict": GENERATION_MAX_TOKENS,
        "seed_derivation": "SHA256(SEED|opaque_task_id|sample_index), first 32 bits",
    }


def _generation_payload(
    protocol: dict[str, Any], model: str, task: dict[str, Any], sample_index: int
) -> dict[str, Any]:
    return generation_request(
        model,
        task["messages"],
        max_tokens=GENERATION_MAX_TOKENS,
        temperature=protocol["generation_temperature"],
        seed=_sample_seed(task["task_id"], sample_index),
    )


def _grading_payload(
    model: str,
    task: dict[str, Any],
    completion: str,
    criterion: dict[str, Any],
) -> dict[str, Any]:
    return grader_request(
        model,
        task["messages"],
        completion,
        criterion["criterion_text"],
        max_tokens=GRADING_MAX_TOKENS,
    )


def _selected_records(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    return (
        sorted(
            records,
            key=lambda record: (
                sum(
                    len(str(message["content"]))
                    for message in record["conversation"]["messages"]
                ),
                record["id"],
            ),
        )[:2]
        if scope == "smoke" else records
    )


def prepare(snapshot: Path, paper: Path, output: Path, *, scope: str) -> dict[str, Any]:
    """Freeze selected visible inputs and private native rubrics before inference."""

    if output.exists():
        raise FileExistsError("use a new packet; never replace a frozen selection")
    _require(scope in {"smoke", "public"}, "scope must be smoke or public")
    source_manifest = verify_snapshot(snapshot)
    verify_paper(paper)
    source_records = _read_jsonl(snapshot / DATA_FILE)
    selected = _selected_records(source_records, scope)
    tasks: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for record in selected:
        messages = visible_conversation(record)
        _require(
            bool(messages) and messages[-1]["role"] == "user",
            "source prefix must end at user",
        )
        _require(
            all(
                message["role"] in {"system", "user", "assistant"} and message["content"]
                for message in messages
            ),
            "source has an invalid visible message",
        )
        task_id = _task_id(record["id"])
        rubrics = [
            {
                "criterion_id": sha256(f"{task_id}|criterion|{index}".encode())[:24],
                "criterion_text": str(item["criterion_text"]),
                "points": item["points"],
                "behavior_axis": str(item["behavior_axis"]),
            }
            for index, item in enumerate(record["rubric_items"])
        ]
        _require(
            all(type(item["points"]) is int and item["points"] != 0 for item in rubrics)
            and any(item["points"] > 0 for item in rubrics),
            "pinned source has unsupported point weights or no positive denominator",
        )
        tasks.append({"task_id": task_id, "messages": messages})
        references.append({"task_id": task_id, "rubric_items": rubrics})
        mapping.append(
            {
                "task_id": task_id,
                "source_id": record["id"],
                "source_index": source_records.index(record),
                "source_metadata": {
                    key: record[key]
                    for key in ("acuity", "user_profile", "language", "has_prior_context")
                },
            }
        )
    completions = 1 if scope == "smoke" else OFFICIAL_COMPLETIONS
    protocol: dict[str, Any] = {
        "version": VERSION,
        "scope": scope,
        "status": "FROZEN_NOT_EXECUTED",
        "selection": (
            "TWO_SHORTEST_VISIBLE_PREFIXES_BY_CHARACTERS_BEFORE_MODEL_OUTPUT_ENGINEERING_ONLY"
            if scope == "smoke"
            else "ALL_PINNED_SOURCE_RECORDS_NO_OUTCOME_DEPENDENT_SELECTION"
        ),
        "tasks": len(tasks),
        "rubric_items": sum(len(item["rubric_items"]) for item in references),
        "completions_per_task": completions,
        "generation_temperature": 0.0 if scope == "smoke" else 1.0,
        "source_snapshot": str(snapshot.resolve()),
        "source_snapshot_manifest_sha256": sha256((snapshot / "snapshot.json").read_bytes()),
        "source_archive_sha256": source_manifest["identity"],
        "source_data_sha256": sha256((snapshot / DATA_FILE).read_bytes()),
        "paper_manifest_sha256": sha256((paper / PAPER_MANIFEST).read_bytes()),
        "paper_sha256": PAPER_SHA256,
        "paper_url": PAPER_URL,
        "implementation_digest": implementation_digest(),
        "generation": _generation_options({"generation_temperature": 0.0}),
        "grading": {
            "prompt_sha256": sha256(
                grader_request(
                    "identity-not-used", [], "", "", max_tokens=GRADING_MAX_TOKENS
                )["messages"][0]["content"].encode()
            ),
            "options": {
                "temperature": 0,
                "seed": SEED,
                "num_ctx": 16384,
                "num_predict": GRADING_MAX_TOKENS,
            },
            "source_judge": "GPT-5.6 Sol, high reasoning effort",
            "local_judge_is_source_judge": False,
        },
        "sampling_scope": (
            "ONE_COMPLETION_PIPELINE_SMOKE_NOT_OFFICIAL_REPRODUCTION"
            if scope == "smoke"
            else "FOUR_PREREGISTERED_SEEDED_LOCAL_SAMPLES_NOT_IDENTICAL_TO_OFFICIAL_SAMPLING"
        ),
        "analysis": {
            "primary": "MACRO_MEAN_TASK_CLIPPED_SCORE",
            "signed": "SUM_MET_SIGNED_POINTS / SUM_ALL_POSITIVE_POINTS",
            "clipped": "MAX(SIGNED_SCORE, 0)",
            "axis_denominator": "TASK_TOTAL_POSITIVE_POINTS_NOT_AXIS_POSITIVE_POINTS",
            "macro_unit": "SOURCE_TASK_EQUAL_WEIGHT",
            "missing": "INVALID_OR_ABSENT_JUDGMENTS_ARE_INCOMPLETE_NOT_ZERO",
            "empty_completion": "VALID_EMPTY_COMPLETION_EARNS_ZERO_POINTS",
            "bootstrap": {
                "cluster": "SOURCE_TASK_NOT_VERIFIED_PERSON",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": SEED,
                "minimum_tasks_for_ci": MIN_CI_TASKS,
                "complete_tasks_only": True,
            },
        },
        "max_generation_calls": len(tasks) * completions,
        "max_grading_calls": sum(len(item["rubric_items"]) for item in references) * completions,
        "retry_budget": 0,
        "read_timeout_seconds": READ_TIMEOUT_SECONDS,
        "model_binding": "PENDING_EXPLICIT_LOCAL_MODEL",
        "official_paper_reproduction": False,
        "official_source_rubrics_reused_not_new_expert_labels": True,
        **CLAIMS,
    }
    protocol["generation"] = _generation_options(protocol)
    protocol["artifacts"] = {
        name: sha256(content)
        for name, content in (
            ("tasks.jsonl", _jsonl_bytes(tasks)),
            ("private/references.jsonl", _jsonl_bytes(references)),
            ("private/mapping.jsonl", _jsonl_bytes(mapping)),
        )
    }
    write_json_atomic(output / "protocol.json", protocol, overwrite=False)
    _write_jsonl(output / "tasks.jsonl", tasks)
    _write_jsonl(output / "private/references.jsonl", references)
    _write_jsonl(output / "private/mapping.jsonl", mapping)
    return protocol


def load_packet(packet: Path, snapshot: Path | None = None) -> dict[str, Any]:
    protocol = read_json_object(packet / "protocol.json")
    _require(
        protocol.get("version") == VERSION
        and protocol.get("implementation_digest") == implementation_digest(),
        "protocol or implementation changed; create a new packet",
    )
    source_path = snapshot or Path(protocol["source_snapshot"])
    source_manifest = verify_snapshot(source_path)
    _require(
        sha256((source_path / "snapshot.json").read_bytes())
        == protocol["source_snapshot_manifest_sha256"]
        and sha256((source_path / DATA_FILE).read_bytes()) == protocol["source_data_sha256"]
        and source_manifest["identity"] == protocol["source_archive_sha256"],
        "packet and source snapshot differ",
    )
    tasks = _read_jsonl(packet / "tasks.jsonl")
    references = _read_jsonl(packet / "private/references.jsonl")
    mapping = _read_jsonl(packet / "private/mapping.jsonl")
    for name in protocol["artifacts"]:
        _require(
            name in {"tasks.jsonl", "private/references.jsonl", "private/mapping.jsonl"},
            "unreviewed packet artifact path",
        )
        _require(
            sha256((packet / name).read_bytes()) == protocol["artifacts"][name],
            "frozen packet artifact changed",
        )
    _require(
        len(tasks) == protocol["tasks"] == len(references) == len(mapping),
        "packet task inventory changed",
    )
    by_task = {task["task_id"]: task for task in tasks}
    by_reference = {item["task_id"]: item for item in references}
    by_mapping = {item["task_id"]: item for item in mapping}
    _require(
        len(by_task) == len(by_reference) == len(by_mapping) == len(tasks)
        and set(by_task) == set(by_reference) == set(by_mapping),
        "task/reference/mapping identity mismatch",
    )
    source_records = _read_jsonl(source_path / DATA_FILE)
    by_source = {record["id"]: record for record in source_records}
    source_positions = {record["id"]: index for index, record in enumerate(source_records)}
    for task in tasks:
        task_id = task["task_id"]
        reference = by_reference[task_id]
        identity = by_mapping[task_id]
        _require(set(task) == {"task_id", "messages"}, "generation task has leaked fields")
        source = by_source.get(identity["source_id"])
        if source is None:
            raise ValueError("unknown source task mapping")
        _require(
            task_id == _task_id(identity["source_id"])
            and identity["source_index"] == source_positions[identity["source_id"]],
            "opaque task identity is not stable",
        )
        _require(task["messages"] == visible_conversation(source), "visible source input changed")
        expected_rubrics = [
            {
                "criterion_id": sha256(f"{task_id}|criterion|{index}".encode())[:24],
                "criterion_text": str(item["criterion_text"]),
                "points": item["points"],
                "behavior_axis": str(item["behavior_axis"]),
            }
            for index, item in enumerate(source["rubric_items"])
        ]
        _require(reference == {"task_id": task_id, "rubric_items": expected_rubrics},
                 "private reference differs from the source snapshot")
        expected_metadata = {
            key: source[key]
            for key in ("acuity", "user_profile", "language", "has_prior_context")
        }
        _require(
            set(identity) == {"task_id", "source_id", "source_index", "source_metadata"}
            and identity["source_metadata"] == expected_metadata,
            "private source metadata changed",
        )
    expected_ids = {
        record["id"] for record in _selected_records(source_records, protocol["scope"])
    }
    _require({item["source_id"] for item in mapping} == expected_ids, "frozen selection changed")
    completions = 1 if protocol["scope"] == "smoke" else OFFICIAL_COMPLETIONS
    _require(
        protocol["completions_per_task"] == completions
        and protocol["generation_temperature"] == (0.0 if completions == 1 else 1.0)
        and protocol["generation"] == _generation_options(protocol)
        and protocol["retry_budget"] == 0
        and protocol["read_timeout_seconds"] == READ_TIMEOUT_SECONDS
        and protocol["max_generation_calls"] == len(tasks) * completions
        and protocol["max_grading_calls"]
        == sum(len(item["rubric_items"]) for item in references) * completions,
        "frozen sampling or call budget changed",
    )
    return {
        "protocol": protocol,
        "tasks": tasks,
        "references": by_reference,
        "mapping": by_mapping,
    }


def local_models() -> list[dict[str, Any]]:
    with httpx.Client(timeout=5, trust_env=False, follow_redirects=False) as client:
        response = client.get(ENDPOINT + "/api/tags")
        response.raise_for_status()
        raw = response.json()
    _require(isinstance(raw, dict) and isinstance(raw.get("models"), list),
             "unexpected local model inventory")
    return [
        {"name": item["name"], "digest": item["digest"]}
        for item in raw["models"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def _model_digest(model: str) -> str:
    installed = [item for item in local_models() if item["name"] == model]
    _require(
        len(installed) == 1 and isinstance(installed[0].get("digest"), str),
        "exact installed local model required; no download or fallback",
    )
    return str(installed[0]["digest"])


def bind_local(packet: Path, model: str, output: Path) -> dict[str, Any]:
    loaded = load_packet(packet)
    protocol = loaded["protocol"]
    if output.exists():
        raise FileExistsError("use a new model binding; never replace a frozen identity")
    if not model.strip():
        raise ValueError("an exact local model name is required")
    config = {
        "version": VERSION,
        "packet_protocol_sha256": sha256((packet / "protocol.json").read_bytes()),
        "provider": "OLLAMA_LOCAL_ONLY",
        "endpoint": ENDPOINT,
        "model": model,
        "model_digest": _model_digest(model),
        "runtime_version": _runtime_version(),
        "retry_budget": 0,
        "read_timeout_seconds": READ_TIMEOUT_SECONDS,
        "max_attempts_per_planned_call": 1,
        "fallback": "FORBIDDEN",
        "generation": protocol["generation"],
        "grading": protocol["grading"],
        "completions_per_task": protocol["completions_per_task"],
        "trust_env": False,
        "follow_redirects": False,
        "judge_scope": "UNCALIBRATED_LOCAL_DIAGNOSTIC_NOT_OFFICIAL_PAPER_JUDGE",
        "judge_calibration": "NOT_ESTABLISHED_AGAINST_INDEPENDENT_EXPERT_MHB_JUDGMENTS",
        "remote_model_calls": 0,
        "reproducibility": "FROZEN_REQUESTS_AND_TRACE_NOT_BIT_DETERMINISTIC_REGENERATION",
    }
    write_json_atomic(output, config, overwrite=False)
    return config


def _runtime_version() -> str:
    with httpx.Client(timeout=5, trust_env=False, follow_redirects=False) as client:
        response = client.get(ENDPOINT + "/api/version")
        response.raise_for_status()
        raw = response.json()
    _require(isinstance(raw, dict) and isinstance(raw.get("version"), str),
             "local runtime version unavailable")
    return str(raw["version"])


def validate_config(packet: Path, path: Path) -> dict[str, Any]:
    loaded = load_packet(packet)
    protocol = loaded["protocol"]
    config = read_json_object(path)
    _require(
        config.get("version") == VERSION
        and config.get("packet_protocol_sha256")
        == sha256((packet / "protocol.json").read_bytes())
        and config.get("provider") == "OLLAMA_LOCAL_ONLY"
        and config.get("endpoint") == ENDPOINT
        and config.get("retry_budget") == 0
        and config.get("read_timeout_seconds") == READ_TIMEOUT_SECONDS
        and config.get("max_attempts_per_planned_call") == 1
        and config.get("fallback") == "FORBIDDEN"
        and config.get("generation") == protocol["generation"]
        and config.get("grading") == protocol["grading"]
        and config.get("completions_per_task") == protocol["completions_per_task"]
        and config.get("trust_env") is False
        and config.get("follow_redirects") is False
        and config.get("judge_scope")
        == "UNCALIBRATED_LOCAL_DIAGNOSTIC_NOT_OFFICIAL_PAPER_JUDGE",
        "model identity, prompt, or budget changed; use a new version",
    )
    _require(isinstance(config.get("model"), str) and config["model"].strip(),
             "model identity missing")
    return config


def visible_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Persist only reviewed fields; never collect arbitrary server or thinking data."""

    _require(isinstance(raw, dict) and isinstance(raw.get("message"), dict),
             "unexpected local response structure")
    content = raw["message"].get("content")
    _require(isinstance(content, str), "local response content must be a string")
    return {
        **{
            key: raw[key]
            for key in (
                "model", "done", "done_reason", "prompt_eval_count", "eval_count",
                "total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
            )
            if key in raw
        },
        "message": {"content": content},
    }


def parse_verdict(text: str) -> dict[str, Any]:
    value = json.loads(text)
    _require(
        isinstance(value, dict)
        and set(value) == {"explanation", "satisfaction"}
        and isinstance(value["explanation"], str)
        and bool(value["explanation"].strip())
        and type(value["satisfaction"]) is int
        and value["satisfaction"] in {0, 1},
        "Judge response violates the criterion binary-output contract",
    )
    return dict(value)


def _valid_token_counts(raw: dict[str, Any]) -> bool:
    return all(
        type(raw.get(key)) is int and raw[key] >= 0
        for key in ("prompt_eval_count", "eval_count")
    )


def _validate_raw(
    raw: dict[str, Any], config: dict[str, Any], *, kind: str
) -> None:
    _require(
        raw.get("model") == config["model"] and raw.get("done") is True,
        "model switched or response incomplete",
    )
    _require(_valid_token_counts(raw), "local token accounting unavailable")
    if kind == "generation":
        _require(raw.get("done_reason") != "length", "truncated generation is not a completion")
    else:
        parse_verdict(raw["message"]["content"])


def _generation_path(run: Path, task_id: str, sample_index: int) -> Path:
    return run / "records" / "generations" / f"{task_id}-{sample_index}.json"


def _judgment_path(
    run: Path, task_id: str, sample_index: int, criterion_id: str
) -> Path:
    return run / "records" / "judgments" / f"{task_id}-{sample_index}-{criterion_id}.json"


def _expected_generation_paths(
    run: Path, tasks: list[dict[str, Any]], completions: int
) -> set[Path]:
    return {
        _generation_path(run, task["task_id"], sample)
        for task in tasks
        for sample in range(completions)
    }


def _expected_judgment_paths(
    run: Path, tasks: list[dict[str, Any]], references: dict[str, Any], completions: int
) -> set[Path]:
    return {
        _judgment_path(run, task["task_id"], sample, criterion["criterion_id"])
        for task in tasks
        for sample in range(completions)
        for criterion in references[task["task_id"]]["rubric_items"]
    }


def _generation_record(
    packet: dict[str, Any], config: dict[str, Any], config_hash: str, path: Path,
    task: dict[str, Any], sample_index: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = read_json_object(path)
    task_id = task["task_id"]
    _require(
        set(record)
        <= {
            "version", "kind", "task_id", "sample_index", "task_sha256", "config_sha256",
            "request", "request_sha256", "calls", "attempt", "status", "error_type",
            "model_digest_verified_before_call", "raw_response", "latency_ms", "created_at_utc",
        }
        and record.get("version") == VERSION
        and record.get("kind") == "generation"
        and record.get("task_id") == task_id
        and record.get("sample_index") == sample_index
        and record.get("task_sha256") == digest(task)
        and record.get("config_sha256") == config_hash
        and record.get("request")
        == _generation_payload(packet["protocol"], config["model"], task, sample_index)
        and record.get("request_sha256") == digest(record["request"])
        and record.get("calls") in {0, 1}
        and type(record.get("calls")) is int
        and record.get("attempt") == 1
        and record.get("status") in {"ok", "error"},
        "generation identity or complete request changed",
    )
    if record["status"] == "ok":
        _require(
            record["calls"] == 1
            and record.get("model_digest_verified_before_call") == config["model_digest"]
            and isinstance(record.get("raw_response"), dict),
            "successful generation lacks identity/accounting",
        )
        _validate_raw(record["raw_response"], config, kind="generation")
    return record


def _judgment_record(
    config: dict[str, Any], config_hash: str, path: Path, task: dict[str, Any],
    criterion: dict[str, Any], generation: dict[str, Any], sample_index: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = read_json_object(path)
    _require(
        set(record)
        <= {
            "version", "kind", "task_id", "sample_index", "criterion_id", "task_sha256",
            "criterion_sha256", "generation_record_sha256", "config_sha256", "request",
            "request_sha256", "calls", "attempt", "status", "error_type",
            "model_digest_verified_before_call", "raw_response", "verdict", "latency_ms",
            "created_at_utc",
        }
        and record.get("version") == VERSION
        and record.get("kind") == "grading"
        and record.get("task_id") == task["task_id"]
        and record.get("sample_index") == sample_index
        and record.get("criterion_id") == criterion["criterion_id"]
        and record.get("task_sha256") == digest(task)
        and record.get("criterion_sha256") == digest(criterion)
        and record.get("generation_record_sha256") == digest(generation)
        and record.get("config_sha256") == config_hash
        and record.get("request")
        == _grading_payload(
            config["model"], task, generation["raw_response"]["message"]["content"], criterion
        )
        and record.get("request_sha256") == digest(record["request"])
        and record.get("calls") in {0, 1}
        and type(record.get("calls")) is int
        and record.get("attempt") == 1
        and record.get("status") in {"ok", "error"},
        "grading identity, source generation, or full request changed",
    )
    if record["status"] == "ok":
        _require(
            record["calls"] == 1
            and record.get("model_digest_verified_before_call") == config["model_digest"]
            and isinstance(record.get("raw_response"), dict)
            and record.get("verdict")
            == parse_verdict(record["raw_response"]["message"]["content"]),
            "successful judgment lacks a valid identity or verdict",
        )
        _validate_raw(record["raw_response"], config, kind="grading")
    return record


def score_completion(
    criteria: list[dict[str, Any]], satisfaction: dict[str, int] | None
) -> dict[str, Any]:
    """Apply native signed weights; incomplete judgment coverage is not a zero."""

    ids = [item["criterion_id"] for item in criteria]
    _require(len(ids) == len(set(ids)), "duplicate criterion identity")
    denominator = sum(max(item["points"], 0) for item in criteria)
    if satisfaction is None:
        return {
            "status": "INCOMPLETE_MISSING_GENERATION",
            "positive_points_available": denominator,
            "signed_score": None,
            "clipped_score": None,
        }
    _require(set(satisfaction) == set(ids), "criterion judgments are missing or unknown")
    _require(all(type(value) is int and value in {0, 1} for value in satisfaction.values()),
             "invalid binary criterion judgment")
    positive = sum(
        item["points"] * satisfaction[item["criterion_id"]]
        for item in criteria if item["points"] > 0
    )
    penalty = sum(
        -item["points"] * satisfaction[item["criterion_id"]]
        for item in criteria if item["points"] < 0
    )
    _require(denominator > 0, "native score undefined without positive-point denominator")
    signed = (positive - penalty) / denominator
    axes: dict[str, Any] = {}
    for axis in sorted({item["behavior_axis"] for item in criteria}):
        selected = [item for item in criteria if item["behavior_axis"] == axis]
        axis_positive = sum(
            item["points"] * satisfaction[item["criterion_id"]]
            for item in selected if item["points"] > 0
        )
        axis_penalty = sum(
            -item["points"] * satisfaction[item["criterion_id"]]
            for item in selected if item["points"] < 0
        )
        axes[axis] = {
            "positive_contribution": axis_positive / denominator,
            "penalty_burden": axis_penalty / denominator,
            "signed_contribution": (axis_positive - axis_penalty) / denominator,
            "criteria": len(selected),
            "denominator": "TASK_TOTAL_POSITIVE_POINTS",
        }
    return {
        "status": "COMPLETE",
        "positive_points_available": denominator,
        "positive_points_met": positive,
        "penalty_points_incurred": penalty,
        "positive_contribution": positive / denominator,
        "penalty_burden": penalty / denominator,
        "signed_score": signed,
        "clipped_score": max(signed, 0),
        "behavior_axes": axes,
    }


def _task_report(
    task: dict[str, Any], reference: dict[str, Any], completions: list[dict[str, Any]]
) -> dict[str, Any]:
    scores = [
        score_completion(
            reference["rubric_items"],
            {
                judgment["criterion_id"]: judgment["verdict"]["satisfaction"]
                for judgment in completion["judgments"]
            }
            if completion["complete"]
            else None,
        )
        for completion in completions
    ]
    complete = [score for score in scores if score["status"].startswith("COMPLETE")]
    axes = sorted({axis for score in complete for axis in score["behavior_axes"]})
    mean_axes = {
        axis: {
            key: float(np.mean([score["behavior_axes"][axis][key] for score in complete]))
            for key in ("positive_contribution", "penalty_burden", "signed_contribution")
        }
        for axis in axes
    } if complete else None
    return {
        "task_id": task["task_id"],
        "status": "COMPLETE" if len(complete) == len(completions) else "INCOMPLETE",
        "expected_completions": len(completions),
        "complete_completions": len(complete),
        "scores": scores,
        "mean_signed_score": (
            float(np.mean([score["signed_score"] for score in complete]))
            if len(complete) == len(completions) else None
        ),
        "mean_clipped_score": (
            float(np.mean([score["clipped_score"] for score in complete]))
            if len(complete) == len(completions) else None
        ),
        "mean_positive_contribution": (
            float(np.mean([score["positive_contribution"] for score in complete]))
            if len(complete) == len(completions) else None
        ),
        "mean_penalty_burden": (
            float(np.mean([score["penalty_burden"] for score in complete]))
            if len(complete) == len(completions) else None
        ),
        "behavior_axes": mean_axes if len(complete) == len(completions) else None,
    }


def _bootstrap(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [task for task in tasks if task["status"] == "COMPLETE"]
    axes = sorted({axis for task in complete for axis in task["behavior_axes"]})
    result: dict[str, Any] = {
        "unit": "SOURCE_TASK_NOT_VERIFIED_PERSON",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
        "clusters": len(complete),
        "scope": "DESCRIPTIVE_COMPLETE_TASKS_ONLY_NOT_CLINICAL_INFERENCE",
    }
    if len(complete) < MIN_CI_TASKS:
        result["status"] = "UNAVAILABLE_FEWER_THAN_30_COMPLETE_TASK_CLUSTERS"
        result["intervals"] = None
        return result
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(complete), (BOOTSTRAP_RESAMPLES, len(complete)))
    intervals: dict[str, Any] = {}
    metrics = {
        "signed_score": [task["mean_signed_score"] for task in complete],
        "clipped_score": [task["mean_clipped_score"] for task in complete],
        "positive_contribution": [task["mean_positive_contribution"] for task in complete],
        "penalty_burden": [task["mean_penalty_burden"] for task in complete],
    }
    for name, values in metrics.items():
        samples = np.asarray(values, dtype=float)[indices].mean(axis=1)
        intervals[name] = [float(value) for value in np.quantile(samples, [0.025, 0.975])]
    for axis in axes:
        for key in ("positive_contribution", "penalty_burden", "signed_contribution"):
            values = [
                task["behavior_axes"].get(axis, {}).get(key, 0.0) for task in complete
            ]
            samples = np.asarray(values, dtype=float)[indices].mean(axis=1)
            intervals[f"axis:{axis}:{key}"] = [
                float(value) for value in np.quantile(samples, [0.025, 0.975])
            ]
    result["status"] = "DESCRIPTIVE_TASK_CLUSTER_BOOTSTRAP"
    result["intervals"] = intervals
    return result


def _token_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["status"] == "ok"]
    failed_with_counts = [
        record for record in records
        if record["status"] == "error" and _valid_token_counts(record.get("raw_response", {}))
    ]
    return {
        "successful_calls": {
            "prompt": sum(record["raw_response"]["prompt_eval_count"] for record in successful),
            "completion": sum(record["raw_response"]["eval_count"] for record in successful),
        },
        "failed_calls_with_returned_counts": {
            "calls": len(failed_with_counts),
            "prompt": sum(
                record["raw_response"]["prompt_eval_count"] for record in failed_with_counts
            ),
            "completion": sum(
                record["raw_response"]["eval_count"] for record in failed_with_counts
            ),
        },
        "failed_calls_without_counts": sum(
            record["status"] == "error" for record in records
        ) - len(failed_with_counts),
    }


def score_run(packet: Path, config_path: Path, run: Path) -> dict[str, Any]:
    """Offline replay: rebuild every request, then rescore only valid judgments."""

    config = validate_config(packet, config_path)
    loaded = load_packet(packet)
    protocol, tasks, references = loaded["protocol"], loaded["tasks"], loaded["references"]
    config_hash = sha256(config_path.read_bytes())
    started = read_json_object(run / "started.json")
    _require(
        started.get("version") == VERSION
        and started.get("config_sha256") == config_hash
        and started.get("packet_protocol_sha256")
        == sha256((packet / "protocol.json").read_bytes())
        and started.get("task_set_digest") == digest(tasks)
        and started.get("stage") == "all"
        and started.get("max_generation_calls") == protocol["max_generation_calls"]
        and started.get("max_grading_calls") == protocol["max_grading_calls"]
        and started.get("maximum_inference_calls_before_empty_reply_exemption")
        == protocol["max_generation_calls"] + protocol["max_grading_calls"]
        and started.get("max_retries") == 0
        and started.get("read_timeout_seconds") == READ_TIMEOUT_SECONDS
        and started.get("runtime_version") == config["runtime_version"],
        "run is not bound to this frozen packet, model, or call budget",
    )
    generation_records: list[dict[str, Any]] = []
    judgment_records: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        reference = references[task_id]
        completions = []
        for sample in range(protocol["completions_per_task"]):
            generation = _generation_record(
                loaded,
                config,
                config_hash,
                _generation_path(run, task_id, sample),
                task,
                sample,
            )
            if generation is not None:
                generation_records.append(generation)
            generation_ok = generation is not None and generation["status"] == "ok"
            judgments: list[dict[str, Any]] = []
            empty_completion = generation is not None and generation["status"] == "ok" and not (
                generation["raw_response"]["message"]["content"]
            )
            criteria = [] if empty_completion else reference["rubric_items"]
            for criterion in criteria:
                path = _judgment_path(run, task_id, sample, criterion["criterion_id"])
                judgment = None
                if generation is not None and generation["status"] == "ok":
                    judgment = _judgment_record(
                        config, config_hash, path, task, criterion, generation, sample
                    )
                else:
                    _require(not path.exists(), "judgment has no valid source generation")
                if judgment is not None:
                    judgment_records.append(judgment)
                    judgments.append(judgment)
            complete = generation_ok and len(judgments) == len(reference["rubric_items"])
            complete = complete and all(item["status"] == "ok" for item in judgments)
            if empty_completion:
                judgments = [
                    {"criterion_id": criterion["criterion_id"], "verdict": {"satisfaction": 0}}
                    for criterion in reference["rubric_items"]
                ]
                complete = True
            completions.append(
                {
                    "sample_index": sample,
                    "generation": generation,
                    "judgments": judgments if complete else [],
                    "complete": complete,
                    "empty_score": empty_completion,
                }
            )
        task_results.append(_task_report(task, reference, completions))
    generation_dir = run / "records" / "generations"
    judgment_dir = run / "records" / "judgments"
    actual_generations = set(generation_dir.glob("*.json")) if generation_dir.exists() else set()
    actual_judgments = set(judgment_dir.glob("*.json")) if judgment_dir.exists() else set()
    _require(
        actual_generations
        <= _expected_generation_paths(run, tasks, protocol["completions_per_task"])
        and actual_judgments
        <= _expected_judgment_paths(run, tasks, references, protocol["completions_per_task"]),
        "unknown or extra model records",
    )
    empty_completions = [
        record for record in generation_records
        if record["status"] == "ok" and not record["raw_response"]["message"]["content"]
    ]
    for record in empty_completions:
        _require(
            not any(
                _judgment_path(run, record["task_id"], record["sample_index"], criterion[
                    "criterion_id"
                ]).exists()
                for criterion in references[record["task_id"]]["rubric_items"]
            ),
            "empty source-protocol zero must not contain hidden Judge calls",
        )
    complete_tasks = [task for task in task_results if task["status"] == "COMPLETE"]
    all_complete = len(complete_tasks) == len(tasks)

    def mean(name: str) -> float | None:
        return float(np.mean([task[name] for task in complete_tasks])) if complete_tasks else None

    axes = sorted({axis for task in complete_tasks for axis in task["behavior_axes"]})
    axis_means = {
        axis: {
            key: float(np.mean([
                task["behavior_axes"].get(axis, {}).get(key, 0.0)
                for task in complete_tasks
            ]))
            for key in ("positive_contribution", "penalty_burden", "signed_contribution")
        }
        for axis in axes
    }
    records = [*generation_records, *judgment_records]
    failed = [record for record in records if record["status"] == "error"]
    skipped_grading_calls = sum(
        len(references[record["task_id"]]["rubric_items"]) for record in empty_completions
    )
    return {
        "version": VERSION,
        "status": "COMPLETE_LOCAL_UNCALIBRATED_DIAGNOSTIC"
        if all_complete else "INCOMPLETE_LOCAL_DIAGNOSTIC_NOT_BENCHMARK_SCORE",
        "config_sha256": config_hash,
        "packet_protocol_sha256": started["packet_protocol_sha256"],
        "model": config["model"],
        "model_digest": config["model_digest"],
        "runtime_version": config["runtime_version"],
        "judge_scope": config["judge_scope"],
        "judge_calibration": config["judge_calibration"],
        "official_paper_reproduction": False,
        "official_judge_used": False,
        "sampling_scope": protocol["sampling_scope"],
        "expected_tasks": len(tasks),
        "complete_tasks": len(complete_tasks),
        "incomplete_tasks": len(tasks) - len(complete_tasks),
        "generation": {
            "expected_calls": protocol["max_generation_calls"],
            "recorded_calls": sum(record["calls"] for record in generation_records),
            "valid": sum(record["status"] == "ok" for record in generation_records),
            "failed": sum(record["status"] == "error" for record in generation_records),
            "missing_records": protocol["max_generation_calls"] - len(generation_records),
            "tokens": _token_totals(generation_records),
        },
        "grading": {
            "expected_calls": protocol["max_grading_calls"],
            "recorded_calls": sum(record["calls"] for record in judgment_records),
            "valid": sum(record["status"] == "ok" for record in judgment_records),
            "failed": sum(record["status"] == "error" for record in judgment_records),
            "missing_or_not_applicable_records": (
                protocol["max_grading_calls"] - len(judgment_records)
            ),
            "empty_completions_scored_zero": len(empty_completions),
            "skipped_empty_completion_calls": skipped_grading_calls,
            "missing_records_excluding_empty_completions": (
                protocol["max_grading_calls"] - len(judgment_records) - skipped_grading_calls
            ),
            "tokens": _token_totals(judgment_records),
        },
        "total_model_calls": sum(record["calls"] for record in records),
        "failed_records": [
            {
                "kind": record["kind"],
                "task_id": record["task_id"],
                "error_type": record.get("error_type", "UNKNOWN"),
            }
            for record in failed
        ],
        "remote_api_cost_cny": 0,
        "local_compute_cost": "NOT_MEASURED",
        "primary_macro_clipped_score_overall": mean("mean_clipped_score")
        if all_complete else None,
        "observed_complete_task_descriptives_NOT_FULL_BENCHMARK": {
            "tasks": len(complete_tasks),
            "signed_score": mean("mean_signed_score"),
            "clipped_score": mean("mean_clipped_score"),
            "positive_contribution": mean("mean_positive_contribution"),
            "penalty_burden": mean("mean_penalty_burden"),
            "behavior_axes": axis_means,
        },
        "bootstrap": _bootstrap(task_results),
        "tasks": task_results,
        "records_digest": digest(
            {
                str(path.relative_to(run)): sha256(path.read_bytes())
                for path in sorted([*actual_generations, *actual_judgments])
            }
        ),
        **CLAIMS,
    }


def report(packet: Path, config: Path, run: Path, output: Path) -> dict[str, Any]:
    result = score_run(packet, config, run)
    saved = read_json_object(run / "report.json")
    _require(result == saved, "independent rescore differs from the frozen run report")
    write_json_atomic(output, result, overwrite=False)
    return result


def _call_local(
    config: dict[str, Any], record: dict[str, Any], payload: dict[str, Any], kind: str
) -> None:
    start = time.monotonic()
    record["created_at_utc"] = datetime.now(UTC).isoformat()
    try:
        record["model_digest_verified_before_call"] = _model_digest(config["model"])
        _require(
            record["model_digest_verified_before_call"] == config["model_digest"],
            "model digest changed; no inference or fallback",
        )
        with httpx.Client(
            timeout=config["read_timeout_seconds"], trust_env=False, follow_redirects=False
        ) as client:
            record["calls"] = 1
            response = client.post(ENDPOINT + "/api/chat", json=payload)
            response.raise_for_status()
            raw = visible_response(response.json())
        record["raw_response"] = raw
        _validate_raw(raw, config, kind=kind)
        if kind == "grading":
            record["verdict"] = parse_verdict(raw["message"]["content"])
        record["status"] = "ok"
    except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        record.update(status="error", error_type=type(exc).__name__)
    record["latency_ms"] = (time.monotonic() - start) * 1000


def run_local(
    packet: Path, config_path: Path, output: Path, *, resume: bool = False
) -> dict[str, Any]:
    """Run once per planned request; explicit resume never repeats a recorded attempt."""

    config = validate_config(packet, config_path)
    loaded = load_packet(packet)
    protocol, tasks, references = loaded["protocol"], loaded["tasks"], loaded["references"]
    config_hash = sha256(config_path.read_bytes())
    if resume:
        _require(output.is_dir(), "resume requires an existing interrupted run")
        started = read_json_object(output / "started.json")
        _require(
            started.get("config_sha256") == config_hash
            and started.get("packet_protocol_sha256")
            == sha256((packet / "protocol.json").read_bytes())
            and started.get("stage") == "all"
            and started.get("task_set_digest") == digest(tasks),
            "resume cannot change packet, model, or protocol",
        )
        # Existing failed attempts are evidence, not an invitation to retry silently.
        replay = score_run(packet, config_path, output)
        if (output / "report.json").exists():
            _require(
                read_json_object(output / "report.json") == replay,
                "finalized run report does not replay",
            )
            _require(
                replay["complete_tasks"] == len(tasks),
                "an incomplete finalized run cannot be retried; declare a new experiment",
            )
            return replay
        _require(
            not any(path.exists() for path in (output / "records").rglob("*.json"))
            or all(
                read_json_object(path)["status"] == "ok"
                for path in (output / "records").rglob("*.json")
            ),
            "resume cannot retry a failed attempt; create a separately declared experiment",
        )
    else:
        if output.exists():
            raise FileExistsError("new run required; use --resume only for an interrupted run")
        started = {
            "version": VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_sha256": config_hash,
            "packet_protocol_sha256": sha256((packet / "protocol.json").read_bytes()),
            "task_set_digest": digest(tasks),
            "stage": "all",
            "max_generation_calls": protocol["max_generation_calls"],
            "max_grading_calls": protocol["max_grading_calls"],
            "maximum_inference_calls_before_empty_reply_exemption": (
                protocol["max_generation_calls"] + protocol["max_grading_calls"]
            ),
            "max_retries": 0,
            "read_timeout_seconds": READ_TIMEOUT_SECONDS,
            "runtime_version": config["runtime_version"],
            "resume_policy": "EXPLICIT_RESUME_OF_UNATTEMPTED_CALLS_ONLY_NO_RETRY",
            "remote_model_calls": 0,
        }
        write_json_atomic(output / "started.json", started, overwrite=False)
    _require(_runtime_version() == config["runtime_version"], "local runtime version changed")
    _require(
        _model_digest(config["model"]) == config["model_digest"],
        "model identity changed before execution",
    )
    stop = False
    for task in tasks:
        for sample in range(protocol["completions_per_task"]):
            task_id = task["task_id"]
            generation_path = _generation_path(output, task_id, sample)
            generation = _generation_record(
                loaded, config, config_hash, generation_path, task, sample
            )
            if generation is None:
                payload = _generation_payload(protocol, config["model"], task, sample)
                generation = {
                    "version": VERSION,
                    "kind": "generation",
                    "task_id": task_id,
                    "sample_index": sample,
                    "task_sha256": digest(task),
                    "config_sha256": config_hash,
                    "request": payload,
                    "request_sha256": digest(payload),
                    "calls": 0,
                    "attempt": 1,
                }
                _call_local(config, generation, payload, "generation")
                write_json_atomic(generation_path, generation, overwrite=False)
            if generation["status"] != "ok":
                stop = True
                break
            completion = generation["raw_response"]["message"]["content"]
            # The pinned source protocol gives an actually empty completion zero
            # points. It does not ask a Judge to reinterpret conditional criteria.
            if completion:
                for criterion in references[task_id]["rubric_items"]:
                    path = _judgment_path(output, task_id, sample, criterion["criterion_id"])
                    judgment = _judgment_record(
                        config, config_hash, path, task, criterion, generation, sample
                    )
                    if judgment is None:
                        payload = _grading_payload(config["model"], task, completion, criterion)
                        judgment = {
                            "version": VERSION,
                            "kind": "grading",
                            "task_id": task_id,
                            "sample_index": sample,
                            "criterion_id": criterion["criterion_id"],
                            "task_sha256": digest(task),
                            "criterion_sha256": digest(criterion),
                            "generation_record_sha256": digest(generation),
                            "config_sha256": config_hash,
                            "request": payload,
                            "request_sha256": digest(payload),
                            "calls": 0,
                            "attempt": 1,
                        }
                        _call_local(config, judgment, payload, "grading")
                        write_json_atomic(path, judgment, overwrite=False)
                    if judgment["status"] != "ok":
                        stop = True
                        break
            if stop:
                break
        if stop:
            break
    _require(_model_digest(config["model"]) == config["model_digest"],
             "model identity changed during execution")
    result = score_run(packet, config_path, output)
    write_json_atomic(output / "report.json", result, overwrite=False)
    return result

