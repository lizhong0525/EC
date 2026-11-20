import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ecocompanion import cli
from ecocompanion.benchmark.log_contract import (
    REQUIRED_EXPERT_ANNOTATIONS,
    ExternalEventLogRecordV1,
)
from ecocompanion.benchmark.log_importer import (
    ExternalLogConflictError,
    ExternalLogImportError,
    build_expert_relabel_drafts,
    import_external_log,
    load_imported_drafts,
    read_external_log_records,
)


def _record(
    *,
    record_id: str,
    sequence_index: int,
    text: str,
    episode_id: str = "episode-alpha",
    visible_events: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "record_type": "external_visible_event",
        "synthetic": True,
        "source_system": "neutral-simulator",
        "source_episode_id": episode_id,
        "source_record_id": record_id,
        "sequence_index": sequence_index,
        "simulated_time": f"day-{sequence_index + 1:02d}-evening",
        "person": {
            "source_person_id": "synthetic-student-001",
            "age_group": "12-15",
            "language": "zh-CN",
            "disclosed_background": ["最近学习任务变多"],
            "persona_tags": ["谨慎表达", "重视同伴评价"],
        },
        "visible_events": visible_events or ["阶段测验临近"],
        "text": text,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_json_import_creates_observable_only_expert_draft(tmp_path: Path) -> None:
    source = _write_json(
        tmp_path / "source.json",
        [
            _record(record_id="event-2", sequence_index=1, text="最近一直睡不好。"),
            _record(record_id="event-1", sequence_index=0, text="这周有点累。"),
        ],
    )

    outcome = import_external_log(source, tmp_path / "drafts.jsonl")
    drafts = load_imported_drafts(outcome.output_path)

    assert outcome.status == "created"
    assert outcome.draft_count == 1
    assert outcome.imported_record_count == 2
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.annotation_status == "PENDING_EXPERT_RELABEL"
    assert draft.benchmark_eligible is False
    assert draft.required_expert_annotations == REQUIRED_EXPERT_ANNOTATIONS
    assert [step.utterance for step in draft.timeline] == [
        "这周有点累。",
        "最近一直睡不好。",
    ]
    assert draft.visible_profile.age_group == "12-15"
    assert draft.persona_tags == ("谨慎表达", "重视同伴评价")

    serialized = outcome.output_path.read_text(encoding="utf-8")
    document = json.loads(serialized)
    forbidden_fields = {
        "hidden_state",
        "state",
        "risk_trajectory",
        "social_ecology",
        "expression_style",
        "acceptable_actions",
        "preferred_action",
        "prohibited_actions",
        "intervention_windows",
    }
    assert forbidden_fields.isdisjoint(document)
    assert all(
        forbidden_fields.isdisjoint(step)
        for step in document["timeline"]
    )
    assert "episode-alpha" not in serialized
    assert "synthetic-student-001" not in serialized
    if os.name == "posix":
        assert stat.S_IMODE(outcome.output_path.stat().st_mode) == 0o600
    # On Windows chmod(0o600) only toggles the read-only flag and stat() reports
    # 0o666, so the POSIX mode is not a meaningful privacy check there. The
    # importer still requests the mode; that intent is covered separately.
    assert outcome.output_path.is_file()


def test_shuffled_jsonl_duplicates_are_idempotent_and_stable(tmp_path: Path) -> None:
    first = _record(record_id="event-1", sequence_index=0, text="今天有点累。")
    second = _record(record_id="event-2", sequence_index=1, text="不太想参加活动。")
    source_with_duplicate = _write_jsonl(
        tmp_path / "first.jsonl",
        [second, first, first],
    )
    source_reordered = _write_json(
        tmp_path / "second.json",
        [first, second],
    )

    first_outcome = import_external_log(
        source_with_duplicate,
        tmp_path / "drafts-a.jsonl",
    )
    repeated_outcome = import_external_log(
        source_with_duplicate,
        tmp_path / "drafts-a.jsonl",
    )
    reordered_outcome = import_external_log(
        source_reordered,
        tmp_path / "drafts-b.jsonl",
    )

    assert first_outcome.status == "created"
    assert first_outcome.input_record_count == 3
    assert first_outcome.imported_record_count == 2
    assert first_outcome.duplicate_record_count == 1
    assert repeated_outcome.status == "unchanged"
    assert reordered_outcome.duplicate_record_count == 0
    assert (tmp_path / "drafts-a.jsonl").read_bytes() == (
        tmp_path / "drafts-b.jsonl"
    ).read_bytes()


def test_import_redacts_credentials_and_direct_contact_values(tmp_path: Path) -> None:
    credential = "sk-" + "sampleCredentialMaterial12345"
    phone = "138" + "00138000"
    national_id = "320101" + "20000101123" + "4"
    email = "student" + "@example.org"
    record = _record(
        record_id="event-1",
        sequence_index=0,
        text=f"误贴了 {credential}，邮箱 {email}，手机 {phone}，证件 {national_id}",
        visible_events=[f"Authorization: Bearer {credential}"],
    )
    source = _write_json(tmp_path / "source.json", record)

    outcome = import_external_log(source, tmp_path / "drafts.jsonl")
    serialized = outcome.output_path.read_text(encoding="utf-8")

    assert outcome.redaction_count == 5
    for sensitive in (credential, phone, national_id, email):
        assert sensitive not in serialized
    assert "[REDACTED_API_CREDENTIAL]" in serialized
    assert "[REDACTED_EMAIL]" in serialized
    assert "[REDACTED_PHONE]" in serialized
    assert "[REDACTED_NATIONAL_ID]" in serialized


def test_conflicting_duplicate_and_ambiguous_sequence_are_rejected() -> None:
    first = ExternalEventLogRecordV1.model_validate(
        _record(record_id="event-1", sequence_index=0, text="第一条。")
    )
    conflicting = ExternalEventLogRecordV1.model_validate(
        _record(record_id="event-1", sequence_index=0, text="内容被改过。")
    )
    same_sequence = ExternalEventLogRecordV1.model_validate(
        _record(record_id="event-2", sequence_index=0, text="另一条。")
    )

    with pytest.raises(ExternalLogConflictError, match="same source record identity"):
        build_expert_relabel_drafts((first, conflicting))
    with pytest.raises(ExternalLogConflictError, match="same sequence_index"):
        build_expert_relabel_drafts((first, same_sequence))


def test_direct_builder_rejects_credentials_used_as_identifiers() -> None:
    credential = "sk-" + "identifierCredentialMaterial12345"
    payload = _record(record_id="event-1", sequence_index=0, text="今天有点累。")
    payload["source_system"] = credential
    record = ExternalEventLogRecordV1.model_validate(payload)

    with pytest.raises(ExternalLogImportError, match="opaque synthetic IDs"):
        build_expert_relabel_drafts((record,))


def test_input_contract_rejects_labels_real_people_and_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    labeled = _record(record_id="event-1", sequence_index=0, text="今天有点累。")
    labeled["risk_trajectory"] = "ACUTE_CRISIS"
    labeled_source = _write_json(tmp_path / "labeled.json", labeled)
    with pytest.raises(ExternalLogImportError, match="schema validation"):
        read_external_log_records(labeled_source)

    real_person = _record(record_id="event-1", sequence_index=0, text="今天有点累。")
    real_person["synthetic"] = False
    real_source = _write_json(tmp_path / "real.json", real_person)
    with pytest.raises(ExternalLogImportError, match="schema validation"):
        read_external_log_records(real_source)

    coerced_index = _record(
        record_id="event-1",
        sequence_index=0,
        text="今天有点累。",
    )
    coerced_index["sequence_index"] = "0"
    coerced_source = _write_json(tmp_path / "coerced.json", coerced_index)
    with pytest.raises(ExternalLogImportError, match="schema validation"):
        read_external_log_records(coerced_source)

    duplicate_key_source = tmp_path / "duplicate-key.json"
    duplicate_key_source.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(ExternalLogImportError, match="duplicate JSON object key"):
        read_external_log_records(duplicate_key_source)


def test_json_envelope_and_conflicting_existing_output(tmp_path: Path) -> None:
    record = _record(record_id="event-1", sequence_index=0, text="今天有点累。")
    envelope = {
        "schema_version": "1.0",
        "document_type": "ecocompanion_external_visible_event_log",
        "records": [record],
    }
    source = _write_json(tmp_path / "source.json", envelope)
    output = tmp_path / "drafts.jsonl"

    assert len(read_external_log_records(source)) == 1
    import_external_log(source, output)
    output.write_text('{"different":"artifact"}\n', encoding="utf-8")

    with pytest.raises(ExternalLogConflictError, match="different import artifact"):
        import_external_log(source, output)
    replaced = import_external_log(source, output, overwrite=True)
    assert replaced.status == "replaced"
    assert len(load_imported_drafts(output)) == 1


def test_research_cli_import_log_is_key_free_and_repeatable(tmp_path: Path) -> None:
    credential = "sk-" + "cliCredentialMaterial123456"
    source = _write_json(
        tmp_path / "source.json",
        _record(
            record_id="event-1",
            sequence_index=0,
            text=f"误贴内容 {credential}",
        ),
    )
    output = tmp_path / "drafts.jsonl"
    runner = CliRunner()

    first = runner.invoke(
        cli.app,
        ["benchmark", "import-log", str(source), "--output", str(output)],
    )
    repeated = runner.invoke(
        cli.app,
        ["benchmark", "import-log", str(source), "--output", str(output)],
    )

    assert first.exit_code == 0, first.output
    assert "status=created" in first.stdout
    assert "redactions=1" in first.stdout
    assert credential not in first.output
    assert repeated.exit_code == 0, repeated.output
    assert "status=unchanged" in repeated.stdout
    assert credential not in output.read_text(encoding="utf-8")

def test_import_requests_restrictive_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_json(
        tmp_path / "source.json",
        [_record(record_id="event-1", sequence_index=0, text="这周有点累。")],
    )
    modes: list[int] = []
    real_chmod = os.chmod

    def spy(path: str | os.PathLike[str], mode: int, *, dir_fd: int | None = None) -> None:
        modes.append(mode)
        real_chmod(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "chmod", spy)
    import_external_log(source, tmp_path / "drafts.jsonl")

    assert 0o600 in modes
