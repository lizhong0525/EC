"""Clean-room importer for neutral synthetic external event logs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from ecocompanion.benchmark.log_contract import (
    ExternalEventLogDocumentV1,
    ExternalEventLogRecordV1,
    ImportedCaseDraftV1,
    ImportedObservationDraftV1,
    ImportProvenanceV1,
)
from ecocompanion.domain.schemas import VisibleProfile

MAX_IMPORT_BYTES = 64 * 1024 * 1024
MAX_IMPORT_RECORDS = 100_000

_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "API_CREDENTIAL",
        re.compile(
            r"(?i)\b(?:authorization|proxy-authorization|x-api-key|api[_-]?key)"
            r"\s*[:=]\s*(?:bearer\s+)?[\"']?[A-Za-z0-9._~+/\-=]{8,}[\"']?"
        ),
    ),
    (
        "API_CREDENTIAL",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-=]{12,}"),
    ),
    ("API_CREDENTIAL", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    (
        "EMAIL",
        re.compile(
            r"(?i)(?<![A-Za-z0-9._%+-])"
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
            r"(?![A-Za-z0-9._%+-])"
        ),
    ),
    (
        "NATIONAL_ID",
        re.compile(r"(?<![A-Za-z0-9])\d{17}[\dXx](?![A-Za-z0-9])"),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<![A-Za-z0-9])(?:\+?86[- ]?)?1[3-9]\d{9}(?![A-Za-z0-9])"
        ),
    ),
)


class ExternalLogImportError(ValueError):
    """Raised when a source log cannot satisfy the neutral import contract."""


class ExternalLogConflictError(ExternalLogImportError):
    """Raised for ambiguous duplicate records or conflicting output artifacts."""


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True)
class SanitizedText:
    value: str
    replacement_count: int


class DeterministicRedactor:
    """Redact common credentials and direct contact identifiers deterministically."""

    def sanitize(self, value: str) -> SanitizedText:
        sanitized = value
        replacement_count = 0
        for kind, pattern in _SENSITIVE_PATTERNS:
            sanitized, count = pattern.subn(f"[REDACTED_{kind}]", sanitized)
            replacement_count += count
        return SanitizedText(
            value=sanitized.strip(),
            replacement_count=replacement_count,
        )

    def contains_sensitive_value(self, value: str) -> bool:
        return any(pattern.search(value) is not None for _, pattern in _SENSITIVE_PATTERNS)


@dataclass(frozen=True)
class DraftBuildResult:
    drafts: tuple[ImportedCaseDraftV1, ...]
    input_record_count: int
    imported_record_count: int
    duplicate_record_count: int
    redaction_count: int


ImportWriteStatus = Literal["created", "unchanged", "replaced"]


@dataclass(frozen=True)
class LogImportOutcome:
    output_path: Path
    status: ImportWriteStatus
    draft_count: int
    input_record_count: int
    imported_record_count: int
    duplicate_record_count: int
    redaction_count: int


def read_external_log_records(path: str | Path) -> tuple[ExternalEventLogRecordV1, ...]:
    """Read a JSON or JSONL log without accepting unknown or duplicate fields."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ExternalLogImportError(f"{source}: expected a file")
    if source.suffix.lower() not in {".json", ".jsonl"}:
        raise ExternalLogImportError(f"{source}: expected .json or .jsonl")
    if source.stat().st_size > MAX_IMPORT_BYTES:
        raise ExternalLogImportError(
            f"{source}: input exceeds the {MAX_IMPORT_BYTES}-byte safety limit"
        )
    try:
        raw = source.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExternalLogImportError(f"{source}: input must be UTF-8") from exc

    records = (
        _parse_jsonl_records(raw, source)
        if source.suffix.lower() == ".jsonl"
        else _parse_json_records(raw, source)
    )
    if not records:
        raise ExternalLogImportError(f"{source}: external log cannot be empty")
    if len(records) > MAX_IMPORT_RECORDS:
        raise ExternalLogImportError(
            f"{source}: input exceeds the {MAX_IMPORT_RECORDS}-record safety limit"
        )
    _reject_sensitive_identifiers(records)
    return tuple(records)


def build_expert_relabel_drafts(
    records: Sequence[ExternalEventLogRecordV1],
    *,
    redactor: DeterministicRedactor | None = None,
) -> DraftBuildResult:
    """Map visible source records to sanitized drafts without generating labels."""

    if not records:
        raise ExternalLogImportError("external log cannot be empty")
    _reject_sensitive_identifiers(records)
    active_redactor = redactor or DeterministicRedactor()
    unique_records, duplicate_count = _deduplicate_records(records)
    grouped = _group_records(unique_records)

    drafts: list[ImportedCaseDraftV1] = []
    redaction_count = 0
    for group_key in sorted(grouped):
        group_records = sorted(
            grouped[group_key],
            key=lambda record: (record.sequence_index, record.source_record_id),
        )
        _validate_episode_records(group_records)
        draft, group_redactions = _build_episode_draft(
            group_records,
            redactor=active_redactor,
        )
        drafts.append(draft)
        redaction_count += group_redactions

    return DraftBuildResult(
        drafts=tuple(drafts),
        input_record_count=len(records),
        imported_record_count=len(unique_records),
        duplicate_record_count=duplicate_count,
        redaction_count=redaction_count,
    )


def import_external_log(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> LogImportOutcome:
    """Import a neutral log into deterministic, idempotent draft JSONL."""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ExternalLogImportError("source and output paths must be different")
    if output_path.suffix.lower() != ".jsonl":
        raise ExternalLogImportError("draft output must use the .jsonl extension")

    records = read_external_log_records(source_path)
    build = build_expert_relabel_drafts(records)
    serialized = serialize_imported_drafts(build.drafts)
    if DeterministicRedactor().contains_sensitive_value(serialized):
        raise ExternalLogImportError(
            "sanitized draft still contains a recognized sensitive value"
        )
    status = _persist_idempotent(
        output_path,
        serialized,
        overwrite=overwrite,
    )
    return LogImportOutcome(
        output_path=output_path,
        status=status,
        draft_count=len(build.drafts),
        input_record_count=build.input_record_count,
        imported_record_count=build.imported_record_count,
        duplicate_record_count=build.duplicate_record_count,
        redaction_count=build.redaction_count,
    )


def serialize_imported_drafts(drafts: Iterable[ImportedCaseDraftV1]) -> str:
    """Serialize drafts as canonical JSONL with a stable draft ordering."""

    ordered = tuple(
        sorted(
            drafts,
            key=lambda draft: (
                draft.provenance.source_system,
                draft.provenance.source_episode_ref,
                draft.draft_id,
            ),
        )
    )
    if not ordered:
        raise ExternalLogImportError("draft collection cannot be empty")
    draft_ids = tuple(draft.draft_id for draft in ordered)
    if len(draft_ids) != len(set(draft_ids)):
        raise ExternalLogConflictError("draft IDs must be unique")
    serialized = (
        "\n".join(
            _canonical_json(draft.model_dump(mode="json"))
            for draft in ordered
        )
        + "\n"
    )
    if DeterministicRedactor().contains_sensitive_value(serialized):
        raise ExternalLogImportError(
            "draft serialization contains a recognized sensitive value"
        )
    return serialized


def load_imported_drafts(path: str | Path) -> tuple[ImportedCaseDraftV1, ...]:
    """Validate an imported draft JSONL artifact."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".jsonl":
        raise ExternalLogImportError(f"{source}: expected .jsonl")
    drafts: list[ImportedCaseDraftV1] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        document = _decode_json(line, source, line_number=line_number)
        if not isinstance(document, dict):
            raise ExternalLogImportError(
                f"{source}:{line_number}: expected a JSON object"
            )
        try:
            drafts.append(ImportedCaseDraftV1.model_validate(document))
        except ValidationError as exc:
            raise ExternalLogImportError(
                f"{source}:{line_number}: invalid expert-relabel draft "
                f"({exc.error_count()} schema error(s))"
            ) from exc
    if not drafts:
        raise ExternalLogImportError(f"{source}: draft collection cannot be empty")
    draft_ids = tuple(draft.draft_id for draft in drafts)
    if len(draft_ids) != len(set(draft_ids)):
        raise ExternalLogConflictError(f"{source}: duplicate draft IDs")
    return tuple(drafts)


def _parse_json_records(
    raw: str,
    source: Path,
) -> list[ExternalEventLogRecordV1]:
    document = _decode_json(raw, source)
    if isinstance(document, dict):
        if document.get("document_type") == "ecocompanion_external_visible_event_log":
            try:
                envelope = ExternalEventLogDocumentV1.model_validate(document)
            except ValidationError as exc:
                raise _schema_error(source, exc) from exc
            return list(envelope.records)
        return [_validate_record(document, source)]
    if isinstance(document, list):
        return [
            _validate_record(item, source, record_index=index)
            for index, item in enumerate(document)
        ]
    raise ExternalLogImportError(f"{source}: expected a JSON object or array")


def _parse_jsonl_records(
    raw: str,
    source: Path,
) -> list[ExternalEventLogRecordV1]:
    records: list[ExternalEventLogRecordV1] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        document = _decode_json(line, source, line_number=line_number)
        records.append(
            _validate_record(
                document,
                source,
                line_number=line_number,
            )
        )
    return records


def _validate_record(
    document: object,
    source: Path,
    *,
    record_index: int | None = None,
    line_number: int | None = None,
) -> ExternalEventLogRecordV1:
    if not isinstance(document, dict):
        location = _source_location(
            source,
            record_index=record_index,
            line_number=line_number,
        )
        raise ExternalLogImportError(f"{location}: expected a JSON object")
    try:
        return ExternalEventLogRecordV1.model_validate(document)
    except ValidationError as exc:
        location = _source_location(
            source,
            record_index=record_index,
            line_number=line_number,
        )
        raise ExternalLogImportError(
            f"{location}: failed external-log schema validation "
            f"({exc.error_count()} error(s))"
        ) from exc


def _decode_json(
    raw: str,
    source: Path,
    *,
    line_number: int | None = None,
) -> object:
    try:
        return cast(
            object,
            json.loads(raw, object_pairs_hook=_reject_duplicate_keys),
        )
    except json.JSONDecodeError as exc:
        location = _source_location(source, line_number=line_number)
        raise ExternalLogImportError(f"{location}: invalid JSON syntax") from exc
    except _DuplicateJsonKeyError as exc:
        location = _source_location(source, line_number=line_number)
        raise ExternalLogImportError(
            f"{location}: duplicate JSON object key is not allowed"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateJsonKeyError
        document[key] = value
    return document


def _schema_error(source: Path, error: ValidationError) -> ExternalLogImportError:
    return ExternalLogImportError(
        f"{source}: failed external-log document validation "
        f"({error.error_count()} error(s))"
    )


def _source_location(
    source: Path,
    *,
    record_index: int | None = None,
    line_number: int | None = None,
) -> str:
    if line_number is not None:
        return f"{source}:{line_number}"
    if record_index is not None:
        return f"{source}:record[{record_index}]"
    return str(source)


def _reject_sensitive_identifiers(
    records: Sequence[ExternalEventLogRecordV1],
) -> None:
    redactor = DeterministicRedactor()
    for record in records:
        identifiers = (
            record.source_system,
            record.source_episode_id,
            record.source_record_id,
            record.person.source_person_id,
        )
        if any(redactor.contains_sensitive_value(value) for value in identifiers):
            raise ExternalLogImportError(
                "source identifiers must be opaque synthetic IDs, not credentials "
                "or direct contact identifiers"
            )


def _deduplicate_records(
    records: Sequence[ExternalEventLogRecordV1],
) -> tuple[tuple[ExternalEventLogRecordV1, ...], int]:
    indexed: dict[tuple[str, str, str], ExternalEventLogRecordV1] = {}
    duplicate_count = 0
    for record in records:
        identity = (
            record.source_system,
            record.source_episode_id,
            record.source_record_id,
        )
        existing = indexed.get(identity)
        if existing is None:
            indexed[identity] = record
            continue
        if existing != record:
            raise ExternalLogConflictError(
                "conflicting records use the same source record identity"
            )
        duplicate_count += 1
    return tuple(indexed.values()), duplicate_count


def _group_records(
    records: Sequence[ExternalEventLogRecordV1],
) -> dict[tuple[str, str], list[ExternalEventLogRecordV1]]:
    grouped: dict[tuple[str, str], list[ExternalEventLogRecordV1]] = {}
    for record in records:
        key = (record.source_system, record.source_episode_id)
        grouped.setdefault(key, []).append(record)
    return grouped


def _validate_episode_records(
    records: Sequence[ExternalEventLogRecordV1],
) -> None:
    if not records:
        raise ExternalLogImportError("episode record collection cannot be empty")
    first = records[0]
    sequence_indices: set[int] = set()
    for record in records:
        if record.person != first.person:
            raise ExternalLogConflictError(
                "all records in one source episode must use one consistent person profile"
            )
        if record.sequence_index in sequence_indices:
            raise ExternalLogConflictError(
                "one source episode cannot contain two records at the same sequence_index"
            )
        sequence_indices.add(record.sequence_index)


def _build_episode_draft(
    records: Sequence[ExternalEventLogRecordV1],
    *,
    redactor: DeterministicRedactor,
) -> tuple[ImportedCaseDraftV1, int]:
    first = records[0]
    source_episode_ref = _stable_ref(
        "episode",
        first.source_system,
        first.source_episode_id,
    )
    draft_id = _stable_ref("draft", first.source_system, first.source_episode_id)
    profile, persona_tags, profile_redactions = _sanitize_profile(
        first,
        redactor=redactor,
    )

    timeline: list[ImportedObservationDraftV1] = []
    redaction_count = profile_redactions
    for step_id, record in enumerate(records):
        source_event_ref = _stable_ref(
            "event",
            record.source_system,
            record.source_episode_id,
            record.source_record_id,
        )
        simulated_time = redactor.sanitize(record.simulated_time)
        utterance = redactor.sanitize(record.text)
        visible_events, event_redactions = _sanitize_texts(
            record.visible_events,
            redactor=redactor,
        )
        redaction_count += (
            simulated_time.replacement_count
            + utterance.replacement_count
            + event_redactions
        )
        timeline.append(
            ImportedObservationDraftV1(
                step_id=step_id,
                observation_id=_stable_ref("observation", draft_id, source_event_ref),
                source_event_ref=source_event_ref,
                simulated_time=simulated_time.value,
                visible_events=visible_events,
                utterance=utterance.value,
            )
        )

    content_payload = {
        "visible_profile": profile.model_dump(mode="json"),
        "persona_tags": persona_tags,
        "timeline": [
            step.model_dump(mode="json")
            for step in timeline
        ],
    }
    draft = ImportedCaseDraftV1(
        draft_id=draft_id,
        provenance=ImportProvenanceV1(
            source_system=first.source_system,
            source_episode_ref=source_episode_ref,
            source_content_sha256=_sha256_text(_canonical_json(content_payload)),
            source_record_count=len(records),
        ),
        visible_profile=profile,
        persona_tags=persona_tags,
        timeline=tuple(timeline),
    )
    return draft, redaction_count


def _sanitize_profile(
    record: ExternalEventLogRecordV1,
    *,
    redactor: DeterministicRedactor,
) -> tuple[VisibleProfile, tuple[str, ...], int]:
    age_group = redactor.sanitize(record.person.age_group)
    language = redactor.sanitize(record.person.language)
    background, background_redactions = _sanitize_texts(
        record.person.disclosed_background,
        redactor=redactor,
    )
    persona_tags, persona_redactions = _sanitize_texts(
        record.person.persona_tags,
        redactor=redactor,
    )
    profile = VisibleProfile(
        age_group=age_group.value,
        language=language.value,
        disclosed_background=background,
    )
    redaction_count = (
        age_group.replacement_count
        + language.replacement_count
        + background_redactions
        + persona_redactions
    )
    return profile, persona_tags, redaction_count


def _sanitize_texts(
    values: Sequence[str],
    *,
    redactor: DeterministicRedactor,
) -> tuple[tuple[str, ...], int]:
    sanitized_values: list[str] = []
    redaction_count = 0
    for value in values:
        sanitized = redactor.sanitize(value)
        sanitized_values.append(sanitized.value)
        redaction_count += sanitized.replacement_count
    return tuple(sanitized_values), redaction_count


def _stable_ref(namespace: str, *parts: str) -> str:
    digest = _sha256_text(
        "\x1f".join(("ecocompanion-external-log-import-v1", namespace, *parts))
    )
    return f"{namespace}-{digest[:24]}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _persist_idempotent(
    destination: Path,
    serialized: str,
    *,
    overwrite: bool,
) -> ImportWriteStatus:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing == serialized:
            os.chmod(destination, 0o600)
            return "unchanged"
        if not overwrite:
            raise ExternalLogConflictError(
                f"{destination} already contains a different import artifact; "
                "choose a new output or explicitly overwrite it"
            )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.chmod(temporary_path, 0o600)
        if overwrite:
            os.replace(temporary_path, destination)
            return "replaced"
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            existing = destination.read_text(encoding="utf-8")
            if existing == serialized:
                os.chmod(destination, 0o600)
                return "unchanged"
            raise ExternalLogConflictError(
                f"{destination} was created concurrently with different content"
            ) from None
        temporary_path.unlink()
        return "created"
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
