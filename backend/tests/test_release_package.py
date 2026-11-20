import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ecocompanion.benchmark.prompt_identity import build_prompt_identity
from ecocompanion.experiments.release import (
    build_release_package,
    verify_release_package,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_release_package_withholds_test_labels_and_verifies_checksums(
    tmp_path: Path,
) -> None:
    destination = build_release_package(
        tmp_path / "release",
        project_root=PROJECT_ROOT,
        seed=41,
    )

    assert (destination / "labels" / "train.jsonl").exists()
    assert (destination / "labels" / "dev.jsonl").exists()
    assert not (destination / "labels" / "public_test.jsonl").exists()
    assert not (destination / "labels" / "hidden_test.jsonl").exists()
    assert (destination / "docs" / "EXTERNAL_LOG_IMPORT.md").exists()
    assert (destination / "schemas" / "ExternalEventLogRecordV1.schema.json").exists()
    assert (destination / "schemas" / "ImportedCaseDraftV1.schema.json").exists()
    assert (destination / "schemas" / "ReleaseManifest.schema.json").exists()
    assert (destination / "evaluation" / "metrics.py").exists()
    assert (destination / "baselines" / "policies.py").exists()
    assert (destination / "prompts" / "catalog.json").exists()
    assert (destination / "docker" / "docker-compose.yml").exists()
    assert (destination / "artifacts" / "README.md").exists()
    assert (destination / "reproducibility" / "requirements.lock").exists()
    nested_compose = yaml.safe_load(
        (destination / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    assert nested_compose["services"]["backend"]["build"]["context"] == ".."
    assert nested_compose["services"]["web"]["build"]["context"] == ".."
    assert nested_compose["services"]["backend"]["volumes"][0]["source"] == "../artifacts"
    # The release keeps an auditable ``source/`` snapshot and mirrors the
    # canonical checkout layout so the copied Dockerfile can build from the
    # package root without path rewriting.
    assert (destination / "source" / "backend" / "src" / "ecocompanion").exists()
    assert (destination / "backend" / "src" / "ecocompanion").exists()
    assert (destination / "scripts" / "verify_release.py").exists()
    assert (destination / "models" / "MODEL_SNAPSHOT_PENDING.md").exists()
    prompt_catalog = json.loads(
        (destination / "prompts" / "catalog.json").read_text(encoding="utf-8")
    )
    assert len(prompt_catalog["entries"]) == 5
    for entry in prompt_catalog["entries"]:
        content = (destination / entry["content_file"]).read_text(encoding="utf-8").rstrip()
        identity = build_prompt_identity(
            component=entry["component"],
            version=entry["version"],
            content=content,
        )
        assert identity.digest == entry["digest"]

    public_text = (destination / "inputs" / "public_test.jsonl").read_text(encoding="utf-8")
    assert public_text.count("\n") == 24
    assert "depression_burden" not in public_text
    assert "acute_safety_risk" not in public_text
    assert "preferred_action" not in public_text

    manifest = json.loads((destination / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "research-preview"
    assert manifest["split_counts"] == {
        "dev": 24,
        "hidden_test": 24,
        "public_test": 24,
        "train": 72,
    }
    assert manifest["withheld_splits"] == ["hidden_test"]
    assert manifest["model_snapshot_status"] == "placeholder"
    assert manifest["run_artifact_status"] == "not_provided"
    assert manifest["source_revision"] is None
    assert "reproducibility/requirements.lock" in manifest["dependency_lock_files"]
    assert any(
        item["name"] == "evaluation-scripts" and item["status"] == "included"
        for item in manifest["artifact_inventory"]
    )

    validation = json.loads((destination / "VALIDATION_REPORT.json").read_text(encoding="utf-8"))
    assert validation["passed"] is True
    assert validation["near_duplicate_audit"]["ngram_size"] == 3
    assert validation["near_duplicate_audit"]["threshold"] == 0.85
    assert validation["near_duplicate_audit"]["passed"] is True

    for line in (destination / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", maxsplit=1)
        payload = (destination / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
    verify_release_package(destination)


def test_release_package_never_overwrites_existing_directory(tmp_path: Path) -> None:
    destination = tmp_path / "release"
    build_release_package(destination, project_root=PROJECT_ROOT)

    with pytest.raises(FileExistsError):
        build_release_package(destination, project_root=PROJECT_ROOT)


def test_release_package_redacts_optional_run_artifacts_and_strips_hidden_state(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "formal-run"
    run_dir.mkdir()
    (run_dir / "api_trace.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "trace-1",
                "world_state_before": {"acute_safety_risk": 0.9},
                "response_text": "Bearer secret-token-12345678",
                # Build the credential-shaped fixture at runtime so the
                # repository scanner does not mistake a test value for a
                # checked-in credential.
                "api_key": "sk-" + "should-not-escape",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "raw_outputs.jsonl").write_text(
        '{"content":"普通回复","authorization":"Bearer another-token-12345678"}\n',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        (
            '{"risk_trajectory":"ACUTE_CRISIS","case_id":"case-1","metrics":{},'
            '"reference_value":1.0,"expected_action":"URGENT_ESCALATION",'
            '"annotation_digest":"should-not-escape","full_reference_record_count":2}\n'
        ),
        encoding="utf-8",
    )
    (run_dir / "cache.db").write_bytes(b"not published")
    destination = build_release_package(
        tmp_path / "release",
        project_root=PROJECT_ROOT,
        seed=41,
        run_dirs=(run_dir,),
    )

    trace_text = (destination / "runs" / "formal-run" / "traces" / "api_trace.jsonl").read_text(
        encoding="utf-8"
    )
    assert "world_state_before" not in trace_text
    assert "acute_safety_risk" not in trace_text
    assert "[REDACTED]" in trace_text
    raw_text = (
        destination / "runs" / "formal-run" / "raw_outputs" / "raw_outputs.jsonl"
    ).read_text(encoding="utf-8")
    assert "another-token" not in raw_text
    assert "[REDACTED]" in raw_text
    metadata_text = (destination / "runs" / "formal-run" / "metadata" / "results.jsonl").read_text(
        encoding="utf-8"
    )
    assert "risk_trajectory" not in metadata_text
    assert "reference_value" not in metadata_text
    assert "expected_action" not in metadata_text
    assert "annotation_digest" not in metadata_text
    assert "full_reference_record_count" not in metadata_text
    assert not (destination / "runs" / "formal-run" / "cache.db").exists()

    manifest = json.loads((destination / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["run_artifact_status"] == "included"
    verify_release_package(destination)
