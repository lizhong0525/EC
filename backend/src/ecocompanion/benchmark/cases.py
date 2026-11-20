"""Case loading, persistence, and benchmark-level validation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from ecocompanion.domain.schemas import CaseSpecV1


class CaseValidationError(ValueError):
    """Raised when a case is schema-valid but violates benchmark invariants."""


def validate_case(
    case: CaseSpecV1,
    *,
    minimum_surface_variants: int = 2,
) -> CaseSpecV1:
    """Validate invariants required by deterministic benchmark runners."""

    step_ids = tuple(step.step_id for step in case.timeline)
    expected_step_ids = tuple(range(len(case.timeline)))
    if step_ids != expected_step_ids:
        raise CaseValidationError(
            f"{case.case_id}: timeline step_ids must be consecutive from zero"
        )

    observation_ids = tuple(step.observation_id for step in case.timeline)
    if len(observation_ids) != len(set(observation_ids)):
        raise CaseValidationError(f"{case.case_id}: observation_ids must be unique")

    for step in case.timeline:
        if len(step.frozen_utterances) < minimum_surface_variants:
            raise CaseValidationError(
                f"{case.case_id}: step {step.step_id} requires at least "
                f"{minimum_surface_variants} frozen utterances"
            )

    last_step = case.timeline[-1].step_id
    window_actions: set[str] = set()
    for window in case.intervention_windows:
        if window.latest_safe_step > last_step:
            raise CaseValidationError(
                f"{case.case_id}: intervention window for {window.action.value} "
                "extends beyond the timeline"
            )
        if window.action.value in window_actions:
            raise CaseValidationError(
                f"{case.case_id}: duplicate intervention window for {window.action.value}"
            )
        window_actions.add(window.action.value)

    fact_ids = tuple(fact.fact_id for fact in case.disclosure_facts)
    if len(fact_ids) != len(set(fact_ids)):
        raise CaseValidationError(f"{case.case_id}: disclosure fact_ids must be unique")

    return case


def validate_cases(
    cases: Sequence[CaseSpecV1],
    *,
    minimum_surface_variants: int = 2,
) -> tuple[CaseSpecV1, ...]:
    """Validate a case collection and reject duplicate identifiers."""

    if not cases:
        raise CaseValidationError("case collection cannot be empty")

    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise CaseValidationError("case_ids must be unique")

    return tuple(
        validate_case(case, minimum_surface_variants=minimum_surface_variants)
        for case in cases
    )


def _parse_json_document(raw: str, source: Path) -> list[object]:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CaseValidationError(f"{source}: invalid JSON: {exc}") from exc
    if isinstance(document, list):
        return list(document)
    if isinstance(document, dict):
        return [document]
    raise CaseValidationError(f"{source}: expected a JSON object or array")


def _parse_json_lines(raw: str, source: Path) -> list[object]:
    documents: list[object] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CaseValidationError(
                f"{source}:{line_number}: invalid JSON: {exc}"
            ) from exc
    return documents


def load_cases(
    path: str | Path,
    *,
    minimum_surface_variants: int = 2,
) -> tuple[CaseSpecV1, ...]:
    """Load one JSON/JSONL file or all such files in a directory."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_dir():
        files = tuple(
            sorted(
                file
                for file in source.rglob("*")
                if file.is_file() and file.suffix.lower() in {".json", ".jsonl"}
            )
        )
        if not files:
            raise CaseValidationError(f"{source}: no JSON or JSONL case files found")
    else:
        if source.suffix.lower() not in {".json", ".jsonl"}:
            raise CaseValidationError(f"{source}: expected .json or .jsonl")
        files = (source,)

    documents: list[object] = []
    for file in files:
        raw = file.read_text(encoding="utf-8")
        if file.suffix.lower() == ".jsonl":
            documents.extend(_parse_json_lines(raw, file))
        else:
            documents.extend(_parse_json_document(raw, file))

    parsed: list[CaseSpecV1] = []
    for index, document in enumerate(documents):
        try:
            parsed.append(CaseSpecV1.model_validate(document))
        except ValidationError as exc:
            raise CaseValidationError(
                f"{source}: case document {index} failed schema validation: {exc}"
            ) from exc

    return validate_cases(
        parsed,
        minimum_surface_variants=minimum_surface_variants,
    )


def load_case(
    path: str | Path,
    *,
    minimum_surface_variants: int = 2,
) -> CaseSpecV1:
    """Load exactly one case and reject ambiguous multi-case inputs."""

    cases = load_cases(
        path,
        minimum_surface_variants=minimum_surface_variants,
    )
    if len(cases) != 1:
        raise CaseValidationError(
            f"{Path(path)}: expected exactly one case; found {len(cases)}"
        )
    return cases[0]


def dump_cases_jsonl(cases: Iterable[CaseSpecV1], path: str | Path) -> Path:
    """Atomically persist validated cases as deterministic JSONL."""

    validated = validate_cases(tuple(cases))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = "\n".join(
        case.model_dump_json(exclude_none=False) for case in validated
    )
    payload += "\n"

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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination
