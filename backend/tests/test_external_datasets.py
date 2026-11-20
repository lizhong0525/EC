"""Offline tests for the external dataset intake registry and verifier."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts import external_datasets as ed


def _mhb_record(index: int, rubric_items: int) -> dict[str, object]:
    return {
        "id": f"{index:032x}",
        "conversation": {
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"reply {index}"},
            ]
        },
        "rubric_items": [
            {
                "criterion_text": f"criterion {index}-{n}",
                "points": 1 if n % 2 == 0 else -1,
                "behavior_axis": "empathy_emotional_attunement_and_support",
            }
            for n in range(rubric_items)
        ],
        "acuity": ("non_acute", "high_acuity", "emergent")[index % 3],
        "user_profile": ("adult", "teen", "clinician", "caregiver")[index % 4],
        "language": "zh" if index == 0 else "en",
        "has_prior_context": index % 10 == 0,
        "canary_string": "mentalhealthbench:test-canary",
    }


def _mhb_jsonl(records: int = ed.MHB_RECORDS) -> bytes:
    base, extra = divmod(ed.MHB_RUBRIC_ITEMS, records)
    lines = [
        json.dumps(_mhb_record(i, base + (1 if i < extra else 0)), ensure_ascii=False)
        for i in range(records)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _mhb_archive(jsonl: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("mentalhealthbench/README.md", "# card\n")
        bundle.writestr("mentalhealthbench/LICENSE", "MIT\n")
        bundle.writestr("mentalhealthbench/mentalhealthbench_eval.jsonl", jsonl)
    return buffer.getvalue()


def _csv(rows: int = ed.CB_ROWS, questions: int = ed.CB_QUESTIONS) -> bytes:
    header = list(ed.CB_COLUMNS)
    lines = [",".join(header)]
    for index in range(rows):
        question = index % questions
        responder = ("gpt4", "llama3", "gemini", "human")[question % 4]
        values = {
            "questionID": f"q{question:03d}",
            "questionTitle": "title",
            "questionText": "text",
            "response": "reply",
            "topic": f"topic{question % 20}",
            "responder": responder,
            "survey_id": f"s{question:03d}",
            "overall_score": "3",
            "overall_reason": "reason",
            "empathy_score": "4",
            "specificity_score": "2",
            "medical_advice_score": "",
            "factual_consistency_score": "5",
            "toxicity_score": "1",
        }
        lines.append(",".join(f'"{values[column]}"' for column in header))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_snapshot(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    inventory: dict[str, object],
    inventory_file: str,
    identity: str,
) -> Path:
    spec = ed.source("mentalhealthbench" if "archive.zip" in files else ed.CB_SOURCE)
    spec = dataclasses.replace(spec, identity=identity)
    monkeypatch.setattr(ed, "SOURCE_ORDER", (spec,))
    return ed.write_snapshot(
        root, spec, files, inventory, inventory_file, "2026-10-23T00:00:00+00:00", {}
    ) and root


def test_registry_is_ordered_and_scoped() -> None:
    priorities = [spec.priority for spec in ed.registry()]
    assert priorities == sorted(priorities)
    assert [spec.key for spec in ed.registry()] == [
        "mentalhealthbench",
        "mentat-triage",
        "counselbench-eval",
        "reachout",
        "sos-hl-1k",
    ]
    assert ed.source("mentalhealthbench").status == "READY"
    assert ed.source("counselbench-eval").status == "READY"
    assert ed.source("reachout").status == "BLOCKED"
    assert ed.source("sos-hl-1k").status == "DEFERRED"
    with pytest.raises(KeyError):
        ed.source("unknown")


def test_git_blob_matches_known_object() -> None:
    assert ed.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_mentalhealthbench_inventory_counts() -> None:
    inventory = ed.inventory_mentalhealthbench(_mhb_jsonl())
    assert inventory["records"] == ed.MHB_RECORDS
    assert inventory["unique_ids"] == ed.MHB_RECORDS
    assert inventory["rubric_items"] == ed.MHB_RUBRIC_ITEMS
    assert inventory["languages"] == {"en": ed.MHB_RECORDS - 1, "zh": 1}
    assert inventory["canary_strings"] == 1
    assert inventory["messages_min"] == 2


def test_mentalhealthbench_rejects_unknown_acuity() -> None:
    records = [_mhb_record(0, 1)]
    records[0]["acuity"] = "urgent"
    with pytest.raises(ValueError):
        ed.inventory_mentalhealthbench((json.dumps(records[0]) + "\n").encode())


def test_mentalhealthbench_rejects_duplicate_ids() -> None:
    record = _mhb_record(7, 1)
    content = ((json.dumps(record) + "\n") * 2).encode("utf-8")
    with pytest.raises(ValueError):
        ed.inventory_mentalhealthbench(content)


def test_counselbench_inventory_counts() -> None:
    inventory = ed.inventory_counselbench(_csv())
    assert inventory["rows"] == ed.CB_ROWS
    assert inventory["unique_questions"] == ed.CB_QUESTIONS
    assert sum(inventory["responders"].values()) == ed.CB_ROWS
    assert inventory["scores"]["medical_advice_score"]["answered"] == 0
    assert inventory["scores"]["overall_score"]["mean"] == 3.0


def test_counselbench_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        ed.inventory_counselbench(b"questionID,response\nq1,hi\n")


def test_mentalhealthbench_snapshot_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl = _mhb_jsonl()
    archive = _mhb_archive(jsonl)
    inventory = ed.inventory_mentalhealthbench(jsonl)
    snapshot = _write_snapshot(
        tmp_path / "mhb",
        monkeypatch,
        {
            "archive.zip": archive,
            "README.md": b"# card\n",
            "LICENSE": b"MIT\n",
            "mentalhealthbench_eval.jsonl": jsonl,
        },
        inventory,
        "mentalhealthbench_eval.jsonl",
        hashlib.sha256(archive).hexdigest(),
    )
    document = ed.verify(snapshot)
    assert document["inventory"]["records"] == ed.MHB_RECORDS
    assert document["new_project_action_gold"] is False


def test_verify_detects_file_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jsonl = _mhb_jsonl()
    archive = _mhb_archive(jsonl)
    snapshot = _write_snapshot(
        tmp_path / "mhb",
        monkeypatch,
        {
            "archive.zip": archive,
            "mentalhealthbench_eval.jsonl": jsonl,
            "README.md": b"# card\n",
            "LICENSE": b"MIT\n",
        },
        ed.inventory_mentalhealthbench(jsonl),
        "mentalhealthbench_eval.jsonl",
        hashlib.sha256(archive).hexdigest(),
    )
    (snapshot / "README.md").write_bytes(b"# changed\n")
    with pytest.raises(ValueError, match="snapshot file changed"):
        ed.verify(snapshot)


def test_verify_detects_inventory_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jsonl = _mhb_jsonl()
    archive = _mhb_archive(jsonl)
    snapshot = _write_snapshot(
        tmp_path / "mhb",
        monkeypatch,
        {
            "archive.zip": archive,
            "mentalhealthbench_eval.jsonl": jsonl,
            "README.md": b"# card\n",
            "LICENSE": b"MIT\n",
        },
        ed.inventory_mentalhealthbench(jsonl),
        "mentalhealthbench_eval.jsonl",
        hashlib.sha256(archive).hexdigest(),
    )
    document = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    document["inventory"]["records"] = 0
    (snapshot / "snapshot.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory changed"):
        ed.verify(snapshot)


def test_counselbench_snapshot_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = _csv()
    snapshot = _write_snapshot(
        tmp_path / "cb",
        monkeypatch,
        {"counselbench_eval.csv": csv, "upstream-metadata.json": b"{}"},
        ed.inventory_counselbench(csv),
        "counselbench_eval.csv",
        ed.git_blob_sha(csv),
    )
    document = ed.verify(snapshot)
    assert document["files"]["counselbench_eval.csv"]["git_blob"] == ed.git_blob_sha(csv)
    assert document["expert_action_labels"] is False


def test_write_snapshot_refuses_existing_history(tmp_path: Path) -> None:
    spec = ed.source("mentalhealthbench")
    target = tmp_path / "taken"
    target.mkdir()
    with pytest.raises(FileExistsError):
        ed.write_snapshot(target, spec, {}, {}, "x", "y", {})


def test_fetch_refuses_blocked_source(tmp_path: Path) -> None:
    with pytest.raises(ed.SourceUnavailable):
        ed.fetch("reachout", tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_plan_reads_snapshots_from_disk(tmp_path: Path) -> None:
    root = tmp_path / "external"
    (root / "mentalhealthbench" / "abc").mkdir(parents=True)
    (root / "mentalhealthbench" / "abc" / "snapshot.json").write_text("{}", encoding="utf-8")
    rows = ed.plan(root)
    assert [row["priority"] for row in rows] == sorted(row["priority"] for row in rows)
    found = next(row for row in rows if row["source"] == "mentalhealthbench")
    assert found["snapshot"] is not None
    blocked = next(row for row in rows if row["source"] == "reachout")
    assert blocked["snapshot"] is None

def test_score_summary_keeps_free_text() -> None:
    summary = ed._score_summary(
        [{"c": "5"}, {"c": "I am not sure"}, {"c": ""}, {"c": "Yes"}], "c"
    )
    assert summary["answered"] == 3
    assert summary["numeric"] == 1
    assert summary["text_values"] == {"I am not sure": 1, "Yes": 1}
