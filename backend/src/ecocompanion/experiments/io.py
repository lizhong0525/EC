"""Small, explicit persistence primitives for research artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def _replace_with_retry(source: Path, destination: Path) -> None:
    # Windows readers can briefly hold a handle that prevents atomic replacement.
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 7:
                raise
            time.sleep(0.01 * (attempt + 1))


class ArtifactConflictError(RuntimeError):
    """Raised when an immutable artifact would be replaced."""


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object, rejecting arrays and scalar documents."""

    if not path.exists():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def read_jsonl_models(
    path: Path,
    model: type[ModelT],
) -> tuple[ModelT, ...]:
    """Validate every non-empty JSONL record with ``model``."""

    if not path.exists():
        return ()
    records: list[ModelT] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"{path}: invalid record at line {line_number}"
            ) from exc
    return tuple(records)


def write_json_atomic(
    path: Path,
    payload: BaseModel | dict[str, Any],
    *,
    overwrite: bool,
    mode: int = 0o600,
) -> Path:
    """Atomically write JSON, optionally enforcing write-once semantics."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ArtifactConflictError(
            f"{destination} already exists; choose a new path or explicitly overwrite it"
        )
    document = (
        payload.model_dump(mode="json")
        if isinstance(payload, BaseModel)
        else payload
    )
    serialized = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
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
        os.chmod(temporary_path, mode)
        if overwrite:
            _replace_with_retry(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise ArtifactConflictError(
                    f"{destination} was created concurrently; refusing to replace it"
                ) from exc
            temporary_path.unlink()
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def write_text_atomic(path: Path, text: str, *, mode: int = 0o600) -> Path:
    """Atomically replace a derived UTF-8 text artifact."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.chmod(temporary_path, mode)
        _replace_with_retry(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def append_jsonl_model(path: Path, record: BaseModel) -> None:
    """Append one validated model as a durable JSONL record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{serialized}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def unique_records_by_id(
    records: Iterable[ModelT],
    *,
    id_attribute: str,
) -> dict[str, ModelT]:
    """Index records while tolerating only byte-equivalent logical duplicates."""

    indexed: dict[str, ModelT] = {}
    for record in records:
        record_id = getattr(record, id_attribute, None)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"record has no valid {id_attribute}")
        existing = indexed.get(record_id)
        if existing is not None and existing != record:
            raise ArtifactConflictError(
                f"conflicting records use the same {id_attribute}: {record_id}"
            )
        indexed[record_id] = record
    return indexed
