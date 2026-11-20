"""Build and verify a sanitized EcoTriage-Bench research package.

The release builder deliberately treats a package as a *research preview*.
It can include optional model/run artifacts, but it never fabricates a model
snapshot, never copies credentials, and strips hidden state from traces.  The
result is self-describing: the manifest records which optional components were
available and ``checksums.sha256`` covers every file except itself.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecocompanion.benchmark import (
    DEFAULT_CASE_SEED,
    DatasetSplit,
    canonical_split,
    dump_cases_jsonl,
    generate_canonical_cases,
)
from ecocompanion.benchmark.generation import relationship_cues_for
from ecocompanion.benchmark.prompt_identity import build_prompt_identity
from ecocompanion.domain.enums import Action
from ecocompanion.domain.schemas import CaseSpecV1
from ecocompanion.experiments.validation import dataset_digest, validate_benchmark


class ReleaseArtifactRecord(BaseModel):
    """Status of one optional or required release component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    status: Literal[
        "included",
        "placeholder",
        "withheld",
        "not_provided",
        "omitted",
    ]
    paths: tuple[str, ...] = ()
    note: str = Field(min_length=1)


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    release_name: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_seed: int
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["research-preview"] = "research-preview"
    expert_annotation_status: Literal["pending"] = "pending"
    split_counts: dict[str, int]
    published_case_labels: tuple[str, ...]
    published_input_splits: tuple[str, ...]
    withheld_splits: tuple[str, ...]
    file_sha256: dict[str, str]
    limitations: tuple[str, ...]
    source_revision: str | None = None
    code_version: str | None = None
    dependency_lock_files: tuple[str, ...] = ()
    artifact_inventory: tuple[ReleaseArtifactRecord, ...] = ()
    model_snapshot_status: Literal[
        "included",
        "placeholder",
        "not_provided",
    ] = "not_provided"
    run_artifact_status: Literal[
        "included",
        "not_provided",
    ] = "not_provided"
    license_status: Literal[
        "declared",
        "not_declared",
    ] = "not_declared"
    reproducibility_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_release_inventory(self) -> ReleaseManifest:
        if any(value < 0 for value in self.split_counts.values()):
            raise ValueError("split_counts cannot contain negative values")
        if len(set(self.published_case_labels)) != len(self.published_case_labels):
            raise ValueError("published_case_labels cannot contain duplicates")
        if len(set(self.published_input_splits)) != len(self.published_input_splits):
            raise ValueError("published_input_splits cannot contain duplicates")
        if len(set(self.withheld_splits)) != len(self.withheld_splits):
            raise ValueError("withheld_splits cannot contain duplicates")
        if set(self.published_input_splits) & set(self.withheld_splits):
            raise ValueError("published_input_splits and withheld_splits must be disjoint")
        if any(
            not path.strip() or Path(path).is_absolute() or ".." in Path(path).parts
            for path in self.file_sha256
        ):
            raise ValueError("file_sha256 keys must be safe relative paths")
        if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in self.file_sha256.values()):
            raise ValueError("file_sha256 values must be SHA-256 hex digests")
        names = [record.name for record in self.artifact_inventory]
        if len(names) != len(set(names)):
            raise ValueError("artifact_inventory names cannot be duplicated")
        return self


def build_release_package(
    destination: Path,
    *,
    project_root: Path,
    seed: int = DEFAULT_CASE_SEED,
    run_dirs: Sequence[Path] | None = None,
    source_run_dirs: Sequence[Path] | None = None,
    model_snapshot_path: Path | None = None,
    pricing_catalog_path: Path | None = None,
) -> Path:
    """Create a new package without exposing public/hidden reference labels.

    ``run_dirs``/``source_run_dirs`` and ``model_snapshot_path`` are optional
    because a clean checkout may not have authenticated API artifacts yet.  In
    that case the package contains explicit placeholders rather than synthetic
    records that could be mistaken for real model results.
    """

    from ecocompanion.schema_export import export_schemas

    if run_dirs is not None and source_run_dirs is not None:
        raise ValueError("provide only one of run_dirs and source_run_dirs")
    requested_run_dirs = (source_run_dirs if source_run_dirs is not None else run_dirs) or ()

    target = destination.resolve()
    if target.exists():
        raise FileExistsError(f"release destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    cases = generate_canonical_cases(seed=seed)
    splits = canonical_split(cases)
    validation = validate_benchmark(seed=seed)
    if not validation.passed:
        raise RuntimeError("benchmark validation failed; release package was not built")

    with tempfile.TemporaryDirectory(
        dir=target.parent,
        prefix=f".{target.name}.",
    ) as temporary:
        root = Path(temporary)
        labels_dir = root / "labels"
        inputs_dir = root / "inputs"
        splits_dir = root / "splits"
        docs_dir = root / "docs"
        cases_dir = root / "cases"
        runtime_artifacts_dir = root / "artifacts"
        labels_dir.mkdir(parents=True)
        inputs_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        cases_dir.mkdir(parents=True)
        runtime_artifacts_dir.mkdir(parents=True)
        (runtime_artifacts_dir / "README.md").write_text(
            """# Runtime artifacts

This directory is intentionally present and writable for the local Compose
demo. It is the host bind mount for `/app/artifacts`; runtime databases,
memory, traces, caches and model snapshots are created here and are not part
of the sanitized release payload.
""",
            encoding="utf-8",
        )

        # Keep the historical labels/inputs paths for compatibility while also
        # exposing the canonical layout documented in the Datasheet.
        dump_cases_jsonl(splits.train, labels_dir / "train.jsonl")
        dump_cases_jsonl(splits.dev, labels_dir / "dev.jsonl")
        dump_cases_jsonl(splits.train, cases_dir / "train.jsonl")
        dump_cases_jsonl(splits.dev, cases_dir / "dev.jsonl")
        _write_public_inputs(
            splits.public_test,
            inputs_dir / "public_test.jsonl",
        )
        _write_public_inputs(
            splits.public_test,
            cases_dir / "public_test.jsonl",
        )
        _write_split_manifest(splits.as_id_manifest(), splits_dir / "manifest.json")
        export_schemas(root / "schemas")

        inventory: list[ReleaseArtifactRecord] = []
        inventory.append(
            ReleaseArtifactRecord(
                name="runtime-artifacts-directory",
                status="included",
                paths=("artifacts/README.md",),
                note=(
                    "Pre-created host bind-mount directory for the Docker demo; "
                    "runtime state is intentionally excluded from the release payload."
                ),
            )
        )
        inventory.extend(_copy_release_docs(project_root.resolve(), docs_dir, root))
        inventory.extend(_copy_project_readme_and_plan(project_root.resolve(), root))
        inventory.extend(_write_prompt_catalog(project_root.resolve(), root))
        inventory.extend(
            _copy_model_snapshot(
                project_root.resolve(),
                root,
                explicit_path=model_snapshot_path,
                explicit_pricing_path=pricing_catalog_path,
            )
        )
        inventory.extend(_copy_reproducibility_material(project_root.resolve(), root))
        inventory.extend(
            _copy_run_artifacts(
                root,
                requested_run_dirs,
            )
        )
        inventory.extend(_write_reproducibility_scripts(root))
        inventory.extend(_write_release_readme(root))

        validation_path = root / "VALIDATION_REPORT.json"
        validation_path.write_text(
            json.dumps(
                validation.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        inventory.append(
            ReleaseArtifactRecord(
                name="benchmark-validation",
                status="included",
                paths=("VALIDATION_REPORT.json",),
                note="Deterministic benchmark and leakage validation report.",
            )
        )
        inventory.append(
            ReleaseArtifactRecord(
                name="json-schemas",
                status="included",
                paths=tuple(
                    sorted(
                        path.relative_to(root).as_posix()
                        for path in (root / "schemas").glob("*.json")
                    )
                ),
                note="Exported Pydantic contracts, including the release manifest schema.",
            )
        )
        payload_hashes = _hash_files(root)
        model_status = _status_for_inventory(inventory, "model-snapshot")
        run_status: Literal["included", "not_provided"] = (
            "included"
            if any(record.name.startswith("run-artifacts:") for record in inventory)
            else "not_provided"
        )
        manifest = ReleaseManifest(
            release_name=target.name,
            benchmark_version=cases[0].benchmark_version,
            dataset_seed=seed,
            dataset_digest=dataset_digest(cases),
            split_counts={split.value: len(splits.for_split(split)) for split in DatasetSplit},
            published_case_labels=("train", "dev"),
            published_input_splits=("public_test",),
            withheld_splits=("hidden_test",),
            file_sha256=payload_hashes,
            limitations=(
                "All cases are synthetic and have not completed expert gold-label review.",
                "The package supports research engineering evaluation, not diagnosis or treatment.",
                "Reference files are omitted, but public generator/source/seed reconstruct v1 "
                "holdouts; these are public development/regression data, not confidential tests.",
                "A missing authenticated model snapshot is represented by a placeholder.",
                "Optional run traces are redacted and must not be interpreted as "
                "clinical evidence.",
            ),
            source_revision=_detect_source_revision(project_root.resolve()),
            code_version=_project_version(project_root.resolve()),
            dependency_lock_files=tuple(
                path
                for path in (
                    "requirements.lock",
                    "requirements-dev.lock",
                    "web/package-lock.json",
                    "reproducibility/requirements.lock",
                    "reproducibility/requirements-dev.lock",
                    "reproducibility/web-package-lock.json",
                )
                if (root / path).exists()
            ),
            artifact_inventory=tuple(inventory),
            model_snapshot_status=model_status,
            run_artifact_status=run_status,
            license_status="declared" if (root / "LICENSE").exists() else "not_declared",
            reproducibility_notes=(
                "Run scripts/verify_release.py before consuming package files.",
                "Repeatable case generation uses the recorded dataset_seed and benchmark_version.",
                "source_revision is null when the input checkout has no Git metadata.",
                "Real LiteLLM discovery, qualification, expert labels and Docker acceptance "
                "remain external release gates.",
            ),
        )
        (root / "RELEASE_MANIFEST.json").write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_checksums(root)
        verify_release_package(root)
        Path(temporary).rename(target)
    return target


def _write_public_inputs(
    cases: tuple[CaseSpecV1, ...],
    path: Path,
) -> None:
    documents = [_public_case(case) for case in cases]
    path.write_text(
        "".join(
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )


def _public_case(case: CaseSpecV1) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "benchmark_version": case.benchmark_version,
        "case_id": case.case_id,
        "synthetic": True,
        "visible_profile": case.visible_profile.model_dump(mode="json"),
        "persona_tags": list(case.persona_tags),
        "relationship_cues": list(relationship_cues_for(case.social_ecology)),
        "allowed_primary_actions": [action.value for action in Action],
        "timeline": [
            {
                "step_id": step.step_id,
                "simulated_time": step.simulated_time,
                "external_events": list(step.external_events),
                "observation_id": step.observation_id,
                "frozen_utterances": list(step.frozen_utterances),
            }
            for step in case.timeline
        ],
    }


def _write_split_manifest(
    split_ids: dict[str, tuple[str, ...]],
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {key: list(value) for key, value in split_ids.items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_release_docs(
    project_root: Path,
    destination: Path,
    package_root: Path,
) -> tuple[ReleaseArtifactRecord, ...]:
    sources = {
        project_root / "docs" / "README.md": "INDEX.md",
        project_root / "docs" / "BENCHMARK_CARD.md": "BENCHMARK_CARD.md",
        project_root / "docs" / "DATASHEET.md": "DATASHEET.md",
        project_root / "docs" / "ANNOTATION_GUIDELINES.md": ("ANNOTATION_GUIDELINES.md"),
        project_root / "docs" / "EXTERNAL_LOG_IMPORT.md": "EXTERNAL_LOG_IMPORT.md",
        project_root / "CHANGELOG.md": "CHANGELOG.md",
    }
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"release documentation is missing: {missing}")
    written: list[str] = []
    for source, name in sources.items():
        shutil.copyfile(source, destination / name)
        written.append((destination / name).relative_to(package_root).as_posix())
    return (
        ReleaseArtifactRecord(
            name="release-documentation",
            status="included",
            paths=tuple(written),
            note="Benchmark Card, Datasheet, annotation and import contracts, and changelog.",
        ),
    )


def _copy_project_readme_and_plan(
    project_root: Path,
    package_root: Path,
) -> tuple[ReleaseArtifactRecord, ...]:
    """Copy top-level context without copying ignored runtime state."""

    sources = {
        "README.md": project_root / "README.md",
        "DEVELOPMENT_PLAN.md": (project_root / "EcoCompanion_完整Benchmark与陪伴Agent开发计划.md"),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"release context is missing: {missing}")
    paths: list[str] = []
    for name, source in sources.items():
        destination = package_root / name
        shutil.copyfile(source, destination)
        paths.append(destination.relative_to(package_root).as_posix())
    return (
        ReleaseArtifactRecord(
            name="project-context",
            status="included",
            paths=tuple(paths),
            note="Project README and the frozen development plan.",
        ),
    )


_PROMPT_SOURCES: tuple[tuple[str, str, str, str], ...] = (
    (
        "agent-decision-system",
        "agent-decision-v1",
        "backend/src/ecocompanion/benchmark/llm_policies.py",
        "_BASE_SYSTEM_PROMPT",
    ),
    (
        "agent-empathy-treatment",
        "agent-empathy-v1",
        "backend/src/ecocompanion/benchmark/llm_policies.py",
        "_EMPATHY_TREATMENT",
    ),
    (
        "companion-renderer",
        "companion-render-v1",
        "backend/src/ecocompanion/application/renderer.py",
        "_SYSTEM_PROMPT",
    ),
    (
        "safety-rewriter",
        "safety-rewrite-v1",
        "backend/src/ecocompanion/application/renderer.py",
        "_REWRITE_SYSTEM_PROMPT",
    ),
    (
        "anonymous-response-judge",
        "anonymous-response-judge-v1",
        "backend/src/ecocompanion/benchmark/judging.py",
        "_JUDGE_SYSTEM_PROMPT",
    ),
)


def _write_prompt_catalog(
    project_root: Path,
    package_root: Path,
) -> tuple[ReleaseArtifactRecord, ...]:
    """Extract static provider prompts and bind each to a content digest."""

    prompts_dir = package_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    paths: list[str] = []
    missing: list[str] = []
    for component, version, relative_source, constant_name in _PROMPT_SOURCES:
        source = project_root / relative_source
        if not source.exists():
            missing.append(str(source))
            continue
        content = _extract_string_constant(source, constant_name)
        content = content.rstrip()
        identity = build_prompt_identity(
            component=component,
            version=version,
            content=content,
        )
        filename = f"{component}.txt"
        target = prompts_dir / filename
        target.write_text(content + "\n", encoding="utf-8")
        paths.append(target.relative_to(package_root).as_posix())
        entries.append(
            {
                **identity.to_dict(),
                "source": relative_source,
                "constant": constant_name,
                "content_file": f"prompts/{filename}",
            }
        )
    if missing:
        raise FileNotFoundError(f"prompt sources are missing: {missing}")
    catalog = {
        "catalog_version": "prompt-catalog-v1",
        "entries": entries,
        "dynamic_prompt_note": (
            "Track B/C dynamic context templates remain in the source snapshot; "
            "the static system contracts above are hashed individually."
        ),
    }
    catalog_path = prompts_dir / "catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.append(catalog_path.relative_to(package_root).as_posix())
    readme_path = prompts_dir / "README.md"
    readme_path.write_text(
        """# Prompt catalog

This directory contains the provider-facing static prompts used by the
benchmark and companion renderer. Each text file is bound to a SHA-256 digest
in `catalog.json`; a changed prompt must receive a new version in an
experiment manifest. Dynamic observation/context interpolation is retained in
the source snapshot and is not treated as a hidden chain-of-thought artifact.
""",
        encoding="utf-8",
    )
    paths.append(readme_path.relative_to(package_root).as_posix())
    return (
        ReleaseArtifactRecord(
            name="prompts",
            status="included",
            paths=tuple(paths),
            note="Static prompt text, identity digests and dynamic-template boundary.",
        ),
    )


def _extract_string_constant(source: Path, constant_name: str) -> str:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except SyntaxError as exc:
        raise ValueError(f"{source}: invalid Python source") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not any(
                isinstance(target, ast.Name) and target.id == constant_name
                for target in node.targets
            ):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise ValueError(f"{source}: {constant_name} is not a literal string") from exc
            if isinstance(value, str) and value.strip():
                return value
            raise ValueError(f"{source}: {constant_name} must be a non-empty string")
    raise ValueError(f"{source}: constant {constant_name} was not found")


def _copy_model_snapshot(
    project_root: Path,
    package_root: Path,
    *,
    explicit_path: Path | None,
    explicit_pricing_path: Path | None,
) -> tuple[ReleaseArtifactRecord, ...]:
    """Copy authenticated snapshots only when they actually exist."""

    models_dir = package_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    default_snapshot = project_root / "artifacts" / "models" / "model_snapshot.json"
    source = explicit_path.resolve() if explicit_path is not None else default_snapshot
    if source.exists():
        if source.is_dir() or source.is_symlink():
            raise ValueError(f"model snapshot must be a regular file: {source}")
        payload = _load_json_document(source)
        _reject_release_secrets(payload, path=source)
        destination = models_dir / "model_snapshot.json"
        _write_json_document(destination, _sanitize_payload(payload))
        paths = [destination.relative_to(package_root).as_posix()]
        note = (
            "Copied an existing key-free discovery/qualification snapshot; "
            "verify authenticated_discovery and qualification fields before use."
        )
        status: Literal["included", "placeholder", "not_provided"] = "included"
    else:
        destination = models_dir / "MODEL_SNAPSHOT_PENDING.md"
        destination.write_text(
            """# Model snapshot pending

No authenticated LiteLLM model snapshot was present when this research
preview was built. This placeholder is intentional; it is not a fabricated
model result. Run `eco models discover` and `eco models qualify` with a
runtime-provided `LITELLM_API_KEY`, review the output, and create a new release
package with that immutable snapshot before reporting API-model results.
""",
            encoding="utf-8",
        )
        paths = [destination.relative_to(package_root).as_posix()]
        note = "No snapshot was available; the package contains an explicit pending placeholder."
        status = "placeholder"

    records = [
        ReleaseArtifactRecord(
            name="model-snapshot",
            status=status,
            paths=tuple(paths),
            note=note,
        )
    ]

    pricing_default = project_root / "artifacts" / "models" / "pricing_catalog.json"
    pricing_source = (
        explicit_pricing_path.resolve() if explicit_pricing_path is not None else pricing_default
    )
    if pricing_source.exists():
        if pricing_source.is_dir() or pricing_source.is_symlink():
            raise ValueError(f"pricing catalog must be a regular file: {pricing_source}")
        pricing_payload = _load_json_document(pricing_source)
        _reject_release_secrets(pricing_payload, path=pricing_source)
        pricing_destination = models_dir / "pricing_catalog.json"
        _write_json_document(
            pricing_destination,
            _sanitize_payload(pricing_payload),
        )
        records.append(
            ReleaseArtifactRecord(
                name="pricing-catalog",
                status="included",
                paths=(pricing_destination.relative_to(package_root).as_posix(),),
                note="Versioned per-model pricing metadata used for cost accounting.",
            )
        )
    else:
        records.append(
            ReleaseArtifactRecord(
                name="pricing-catalog",
                status="not_provided",
                note=(
                    "No pricing catalog was available; unknown-cost calls must be "
                    "reported separately."
                ),
            )
        )
    return tuple(records)


def _nested_compose_contents(contents: str) -> str:
    """Adapt the Compose copy stored under ``docker/`` to its own directory.

    The package also contains an exact root-level ``docker-compose.yml``. A
    user may nevertheless invoke ``docker compose -f docker/docker-compose.yml``
    from the package root; Compose resolves relative paths against the Compose
    file's directory, so the nested copy must point one level up for its build
    context and the host ``artifacts`` bind mount.
    """

    context_count = contents.count("      context: .")
    if context_count != 2:
        raise ValueError(
            "root Compose file must contain exactly two standard build contexts"
        )
    if contents.count("        source: ./artifacts") != 1:
        raise ValueError(
            "root Compose file must contain exactly one artifacts bind source"
        )
    adapted = contents.replace("      context: .", "      context: ..")
    adapted = adapted.replace(
        "        source: ./artifacts",
        "        source: ../artifacts",
    )
    return (
        "# This nested Compose file is adapted for execution from the docker/ "
        "directory. The package-root docker-compose.yml is the canonical copy.\n"
        + adapted
    )


def _copy_reproducibility_material(
    project_root: Path,
    package_root: Path,
) -> tuple[ReleaseArtifactRecord, ...]:
    """Include source, evaluation modules, baselines, locks and Docker files."""

    records: list[ReleaseArtifactRecord] = []

    required_files = {
        project_root / "pyproject.toml": (
            package_root / "reproducibility" / "pyproject.toml"
        ),
        project_root / "requirements.lock": (
            package_root / "reproducibility" / "requirements.lock"
        ),
        project_root / "requirements-dev.lock": (
            package_root / "reproducibility" / "requirements-dev.lock"
        ),
        project_root / "web" / "package.json": (
            package_root / "reproducibility" / "web-package.json"
        ),
        project_root / "web" / "package-lock.json": (
            package_root / "reproducibility" / "web-package-lock.json"
        ),
        project_root / "Makefile": package_root / "reproducibility" / "Makefile",
        project_root / ".env.example": package_root / "reproducibility" / ".env.example",
    }
    lock_paths: list[str] = []
    for source, destination in required_files.items():
        if not source.exists():
            raise FileNotFoundError(f"reproducibility material is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        lock_paths.append(destination.relative_to(package_root).as_posix())
    # Mirror the files at package root so the copied Dockerfiles and
    # ``pyproject.toml`` remain executable without path rewriting.
    root_files = {
        "pyproject.toml": project_root / "pyproject.toml",
        "requirements.lock": project_root / "requirements.lock",
        "requirements-dev.lock": project_root / "requirements-dev.lock",
        "Makefile": project_root / "Makefile",
        ".env.example": project_root / ".env.example",
    }
    for name, source in root_files.items():
        destination = package_root / name
        shutil.copyfile(source, destination)
        lock_paths.append(destination.relative_to(package_root).as_posix())
    web_root = project_root / "web"
    if not web_root.exists():
        raise FileNotFoundError(f"frontend source is missing: {web_root}")
    web_paths: list[str] = []
    allowed_web_suffixes = {
        ".css",
        ".html",
        ".json",
        ".ts",
        ".tsx",
    }
    for source in sorted(web_root.rglob("*")):
        if (
            not source.is_file()
            or source.is_symlink()
            or "node_modules" in source.parts
            or "dist" in source.parts
            or source.suffix.lower() not in allowed_web_suffixes
            or source.name.endswith(".tsbuildinfo")
        ):
            continue
        relative = source.relative_to(project_root)
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        web_paths.append(destination.relative_to(package_root).as_posix())
    if not web_paths:
        raise FileNotFoundError("frontend source has no publishable files")
    lock_paths.extend(web_paths)
    records.append(
        ReleaseArtifactRecord(
            name="dependency-locks",
            status="included",
            paths=tuple(lock_paths),
            note="Python, frontend and build dependency lock/configuration files.",
        )
    )

    docker_sources = {
        "docker/backend.Dockerfile": "docker/backend.Dockerfile",
        "docker/web.Dockerfile": "docker/web.Dockerfile",
        "docker/nginx.conf": "docker/nginx.conf",
        "docker-compose.yml": "docker/docker-compose.yml",
        ".dockerignore": "docker/.dockerignore",
        "docker/README.md": "docker/README.md",
    }
    docker_paths: list[str] = []
    for relative_source, relative_destination in docker_sources.items():
        source = project_root / relative_source
        if not source.exists():
            raise FileNotFoundError(f"Docker material is missing: {source}")
        destination = package_root / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative_source == "docker-compose.yml":
            destination.write_text(
                _nested_compose_contents(source.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        else:
            shutil.copyfile(source, destination)
        docker_paths.append(destination.relative_to(package_root).as_posix())
    root_compose = package_root / "docker-compose.yml"
    shutil.copyfile(project_root / "docker-compose.yml", root_compose)
    docker_paths.append(root_compose.relative_to(package_root).as_posix())
    root_dockerignore = package_root / ".dockerignore"
    shutil.copyfile(project_root / ".dockerignore", root_dockerignore)
    docker_paths.append(root_dockerignore.relative_to(package_root).as_posix())
    records.append(
        ReleaseArtifactRecord(
            name="docker",
            status="included",
            paths=tuple(docker_paths),
            note="Reviewed local Compose, backend/web images and SSE reverse proxy.",
        )
    )

    source_root = project_root / "backend" / "src" / "ecocompanion"
    if not source_root.exists():
        raise FileNotFoundError(f"source snapshot is missing: {source_root}")
    source_paths: list[str] = []
    runtime_source_paths: list[str] = []
    sources = [*source_root.rglob("*.py")]
    annotation_ui = source_root / "research" / "annotation.html"
    if annotation_ui.is_file():
        sources.append(annotation_ui)
    for source in sorted(sources):
        if "__pycache__" in source.parts or source.is_symlink():
            continue
        relative = source.relative_to(project_root)
        # Keep an auditable copy under ``source/`` and mirror the canonical
        # checkout layout at the package root.  The latter is required by the
        # copied ``pyproject.toml`` and Dockerfile (both expect
        # ``backend/src`` relative to the build context).
        for destination_root, paths in (
            (package_root / "source", source_paths),
            (package_root, runtime_source_paths),
        ):
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            paths.append(destination.relative_to(package_root).as_posix())
    source_paths.extend(runtime_source_paths)
    records.append(
        ReleaseArtifactRecord(
            name="source-snapshot",
            status="included",
            paths=tuple(source_paths),
            note=(
                "Python source snapshot backing baselines, runners and report generation; "
                "the root-level checkout mirror keeps Docker and local package commands "
                "runnable from the release directory."
            ),
        )
    )

    baseline_modules = (
        "backend/src/ecocompanion/benchmark/policies.py",
        "backend/src/ecocompanion/benchmark/llm_policies.py",
        "backend/src/ecocompanion/benchmark/metrics.py",
        "backend/src/ecocompanion/benchmark/model_provider.py",
        "backend/src/ecocompanion/benchmark/track_b.py",
        "backend/src/ecocompanion/benchmark/track_c.py",
    )
    baseline_paths: list[str] = []
    for relative_source in baseline_modules:
        source = project_root / relative_source
        if not source.exists():
            raise FileNotFoundError(f"baseline source is missing: {source}")
        destination = package_root / "baselines" / Path(relative_source).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        baseline_paths.append(destination.relative_to(package_root).as_posix())
    baseline_readme = package_root / "baselines" / "README.md"
    baseline_readme.write_text(
        """# Baselines

The source files in this directory expose the deterministic baselines
(`random`, `always-listen`, `always-escalate`, keyword triage and
history-aware rules), the controlled LLM context treatments, and ECO-SAFE
policy implementation. They are copied from the source snapshot for audit
convenience; run them through the pinned package environment rather than
editing release files in place.
""",
        encoding="utf-8",
    )
    baseline_paths.append(baseline_readme.relative_to(package_root).as_posix())
    records.append(
        ReleaseArtifactRecord(
            name="baselines",
            status="included",
            paths=tuple(baseline_paths),
            note="Local rule baselines and controlled/public model policy adapters.",
        )
    )

    evaluation_modules = {
        "backend/src/ecocompanion/benchmark/metrics.py": "evaluation/metrics.py",
        "backend/src/ecocompanion/benchmark/leakage.py": "evaluation/leakage.py",
        "backend/src/ecocompanion/experiments/report.py": "evaluation/report.py",
        "backend/src/ecocompanion/experiments/validation.py": "evaluation/validation.py",
        "scripts/secret_scan.py": "evaluation/secret_scan.py",
    }
    evaluation_paths: list[str] = []
    for relative_source, relative_destination in evaluation_modules.items():
        source = project_root / relative_source
        if not source.exists():
            raise FileNotFoundError(f"evaluation source is missing: {source}")
        destination = package_root / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        evaluation_paths.append(destination.relative_to(package_root).as_posix())
    evaluation_readme = package_root / "evaluation" / "README.md"
    evaluation_readme.write_text(
        """# Evaluation and statistics

`metrics.py`, `leakage.py`, `report.py` and `validation.py` are the exact
evaluation sources used by the package's `eco benchmark` and `eco report`
commands. `secret_scan.py` is included as the release hygiene check. The
source snapshot and lock files are required to execute these modules.
""",
        encoding="utf-8",
    )
    evaluation_paths.append(evaluation_readme.relative_to(package_root).as_posix())
    records.append(
        ReleaseArtifactRecord(
            name="evaluation-scripts",
            status="included",
            paths=tuple(evaluation_paths),
            note="Metric, leakage, report, validation and secret-scan implementations.",
        )
    )
    return tuple(records)


def _copy_run_artifacts(
    package_root: Path,
    run_dirs: Sequence[Path],
) -> tuple[ReleaseArtifactRecord, ...]:
    """Copy selected run outputs after redaction, never raw databases/caches."""

    if not run_dirs:
        readme = package_root / "runs" / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            """# Run artifacts pending

No run directory was supplied to the release builder. This is intentional:
real model outputs and AgentTrace are only published after redaction and
review. Rebuild with `build_release_package(..., run_dirs=[...])` after a
formal run has been audited.
""",
            encoding="utf-8",
        )
        return (
            ReleaseArtifactRecord(
                name="run-artifacts",
                status="not_provided",
                paths=("runs/README.md",),
                note="No raw outputs or AgentTrace were supplied for this preview.",
            ),
        )

    records: list[ReleaseArtifactRecord] = []
    seen_names: set[str] = set()
    for source_dir in run_dirs:
        source = source_dir.resolve()
        if not source.exists() or not source.is_dir() or source.is_symlink():
            raise FileNotFoundError(f"run directory is not a regular directory: {source}")
        if source == package_root or package_root in source.parents:
            raise ValueError("run directory cannot contain the release destination")
        safe_name = _safe_path_component(source.name)
        if safe_name in seen_names:
            raise ValueError(f"run directories resolve to the same release name: {safe_name}")
        seen_names.add(safe_name)
        run_destination = package_root / "runs" / safe_name
        copied: list[str] = []
        skipped: list[str] = []
        for candidate in sorted(source.rglob("*")):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(source)
            lower_name = candidate.name.lower()
            if _should_skip_run_file(relative):
                skipped.append(relative.as_posix())
                continue
            category = _run_file_category(lower_name)
            if category is None and candidate.suffix.lower() in {
                ".json",
                ".jsonl",
                ".txt",
                ".md",
                ".csv",
            }:
                # Preserve otherwise unclassified textual outputs as metadata;
                # the recursive sanitizer below still removes credentials and
                # hidden labels.
                category = "metadata"
            if category is None:
                skipped.append(relative.as_posix())
                continue
            destination = run_destination / category / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Every exported run file is treated as potentially containing
            # labels or world state; public releases must not rely on naming
            # conventions to protect hidden fields.
            _copy_sanitized_file(candidate, destination, strip_hidden=True)
            copied.append(destination.relative_to(package_root).as_posix())
        if not copied:
            note = (
                "The supplied run directory had no eligible text/JSON artifacts after "
                "redaction; caches and databases were withheld."
            )
            status: Literal["included", "not_provided"] = "not_provided"
        else:
            note = (
                f"Copied {len(copied)} redacted run files; withheld {len(skipped)} "
                "cache/database/unknown files."
            )
            status = "included"
        records.append(
            ReleaseArtifactRecord(
                name=f"run-artifacts:{safe_name}",
                status=status,
                paths=tuple(copied),
                note=note,
            )
        )
    return tuple(records)


def _should_skip_run_file(relative: Path) -> bool:
    parts = {part.lower() for part in relative.parts}
    if any(part in {"cache", "__pycache__", ".git"} for part in parts):
        return True
    if relative.name.lower() in {
        ".env",
        ".env.local",
        "model_snapshot.json",
        "pricing_catalog.json",
    }:
        return True
    return relative.suffix.lower() in {
        ".db",
        ".sqlite",
        ".sqlite3",
        ".parquet",
        ".pyc",
        ".bin",
    }


def _run_file_category(filename: str) -> str | None:
    if "trace" in filename:
        return "traces"
    if any(token in filename for token in ("raw", "output", "response")):
        return "raw_outputs"
    if filename in {
        "manifest.json",
        "checkpoint.json",
        "cost_ledger.jsonl",
        "results.jsonl",
        "result.json",
    }:
        return "metadata"
    return None


def _copy_sanitized_file(
    source: Path,
    destination: Path,
    *,
    strip_hidden: bool,
) -> None:
    suffix = source.suffix.lower()
    if suffix not in {".json", ".jsonl", ".txt", ".md", ".csv"}:
        raise ValueError(f"unsupported run artifact format: {source}")
    if suffix == ".json":
        payload = _load_json_document(source)
        sanitized = _sanitize_payload(payload, strip_hidden=strip_hidden)
        _write_json_document(destination, sanitized)
        return
    lines = source.read_text(encoding="utf-8").splitlines()
    sanitized_lines: list[str] = []
    for line in lines:
        if suffix == ".jsonl" and line.strip():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                sanitized_lines.append(_redact_text(line))
                continue
            sanitized_lines.append(
                json.dumps(
                    _sanitize_payload(payload, strip_hidden=strip_hidden),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        else:
            sanitized_lines.append(_redact_text(line))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(sanitized_lines) + "\n", encoding="utf-8")


def _load_json_document(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: expected a UTF-8 JSON document") from exc


_HIDDEN_RELEASE_KEYS = {
    "hidden_state",
    "world_state_before",
    "world_state_after",
    "depression_burden",
    "acute_safety_risk",
    "trust_in_companion",
    "disclosure_readiness",
    "offline_help_willingness",
    "ai_dependency",
    "family_support",
    "peer_support",
    "school_support",
    "risk_trajectory",
    "social_ecology",
    "expression_style",
    "acceptable_actions",
    "preferred_action",
    "prohibited_actions",
    "intervention_windows",
    "gold_label",
    "reference_label",
    # Evaluation/report provenance and per-node reference values are
    # publication-time labels, not observable model output.  They must not
    # escape through optional run artifacts even when a filename is treated as
    # ordinary metadata.
    "reference_value",
    "reference_action",
    "reference_risk",
    "reference_level",
    "reference_evidence",
    "reference_evidence_codes",
    "reference_record",
    "reference_records",
    "expected_action",
    "expected_risk",
    "expected_level",
    "expected_evidence",
    "target_action",
    "target_risk",
    "target_level",
    "target_evidence",
    "gold_node_count",
    "full_reference_record_count",
    "legacy_record_count",
    "annotation_digest",
    "annotation_path",
    "manual_version",
    "batch_id",
}


def _sanitize_payload(value: object, *, strip_hidden: bool = False) -> object:
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _SENSITIVE_KEYS:
                sanitized[str(key)] = "[REDACTED]"
                continue
            if strip_hidden and normalized in _HIDDEN_RELEASE_KEYS:
                continue
            sanitized[str(key)] = _sanitize_payload(child, strip_hidden=strip_hidden)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_payload(item, strip_hidden=strip_hidden) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(item, strip_hidden=strip_hidden) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _reject_release_secrets(value: object, *, path: Path) -> None:
    """Fail closed on credential-shaped keys in supplied release artifacts."""

    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                raise ValueError(f"{path}: supplied artifact contains secret field {key!r}")
            _reject_release_secrets(child, path=path)
    elif isinstance(value, list):
        for child in value:
            _reject_release_secrets(child, path=path)
    elif isinstance(value, str) and _SECRET_VALUE_PATTERN.search(value):
        raise ValueError(f"{path}: supplied artifact contains a credential-shaped value")


def _write_json_document(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
}
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:sk-[a-z0-9][a-z0-9_-]{8,}|bearer\s+[a-z0-9._~+/=-]{8,})"
)


def _redact_text(text: str) -> str:
    return _SECRET_VALUE_PATTERN.sub("[REDACTED]", text)


def _safe_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "run"


def _status_for_inventory(
    inventory: Sequence[ReleaseArtifactRecord],
    name: str,
) -> Literal["included", "placeholder", "not_provided"]:
    records = tuple(record for record in inventory if record.name == name)
    if not records:
        return "not_provided"
    if any(record.status == "included" for record in records):
        return "included"
    if any(record.status == "placeholder" for record in records):
        return "placeholder"
    return "not_provided"


def _detect_source_revision(project_root: Path) -> str | None:
    """Return Git HEAD when available, otherwise make the gap explicit."""

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _project_version(project_root: Path) -> str | None:
    """Read the PEP 621 project version without importing build tooling."""

    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    match = re.search(
        r"(?m)^\s*version\s*=\s*[\"']([^\"']+)[\"']\s*$",
        pyproject.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def _write_reproducibility_scripts(
    package_root: Path,
) -> tuple[ReleaseArtifactRecord, ...]:
    scripts_dir = package_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    verify_path = scripts_dir / "verify_release.py"
    verify_path.write_text(
        '''#!/usr/bin/env python3
"""Verify a sanitized EcoCompanion release package.

The script intentionally uses only Python's standard library so it can run
before installing project dependencies.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe package path: {relative!r}")
    resolved = (root / path).resolve()
    resolved.relative_to(root)
    return resolved


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "RELEASE_MANIFEST.json"
    checksum_path = root / "checksums.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest.get("file_sha256", {}).items():
        path = package_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", maxsplit=1)
        path = package_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    if failures:
        print("checksum verification failed:", *sorted(set(failures)), sep="\\n")
        return 1
    print("release checksums verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
        encoding="utf-8",
    )
    verify_path.chmod(0o755)
    reproduce_path = scripts_dir / "reproduce_local.sh"
    reproduce_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/scripts/verify_release.py"
printf '%s\n' "Install the pinned environment from reproducibility/requirements*.lock"
printf '%s\n' "then run: eco benchmark validate --output artifacts/validation/report.json"
""",
        encoding="utf-8",
    )
    reproduce_path.chmod(0o755)
    readme_path = scripts_dir / "README.md"
    readme_path.write_text(
        """# Reproduction helpers

Run `python3 scripts/verify_release.py` first. `reproduce_local.sh` documents
the dependency installation and deterministic validation command. API-backed
model discovery and expert annotation are intentionally not faked by this
package; they require the external gates recorded in `RELEASE_MANIFEST.json`.
""",
        encoding="utf-8",
    )
    return (
        ReleaseArtifactRecord(
            name="reproduction-scripts",
            status="included",
            paths=(
                "scripts/verify_release.py",
                "scripts/reproduce_local.sh",
                "scripts/README.md",
            ),
            note="Dependency-free checksum verifier and deterministic local reproduction helper.",
        ),
    )


def _write_release_readme(
    package_root: Path,
) -> tuple[ReleaseArtifactRecord, ...]:
    path = package_root / "RELEASE_README.md"
    path.write_text(
        """# EcoTriage-Bench research preview

This package contains synthetic benchmark inputs, labels for Train/Dev,
Public Test inputs, versioned schemas, evaluation sources, prompt identities,
optional redacted run artifacts, and reproducibility material.

It is not a clinical dataset, diagnostic tool, treatment system, or real-time
emergency service. Hidden Test inputs and all public/hidden reference labels
are omitted as files, but the published deterministic generator and seeds can
reconstruct the v1 holdouts and their simulator references. Treat v1 as public
development/regression data, not an independent confidential blind test.
Independent v2 blind material must never be copied into this package.
A missing authenticated model snapshot is represented by an
explicit placeholder. Before using any optional run artifacts, inspect their
redaction and the `artifact_inventory` in `RELEASE_MANIFEST.json`.

Verify the package with:

```bash
python3 scripts/verify_release.py
```

For the local Compose demo, run `docker compose` from the package root using
the root-level `docker-compose.yml`. The copy at
`docker/docker-compose.yml` is adapted for execution from the `docker/`
directory and points its build context and `artifacts` bind mount one level
up.
""",
        encoding="utf-8",
    )
    return (
        ReleaseArtifactRecord(
            name="release-readme",
            status="included",
            paths=("RELEASE_README.md",),
            note="Scope, safety boundary and verification instructions.",
        ),
    )


def _hash_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {"checksums.sha256", "RELEASE_MANIFEST.json"}
    }


def _write_checksums(root: Path) -> None:
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "checksums.sha256"
    ]
    (root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def verify_release_package(package_root: Path) -> None:
    """Verify manifest hashes, checksum lines and public-data withholding.

    The function raises ``ValueError`` on any mismatch so callers can use it
    as a release gate.  It intentionally does not trust the manifest alone:
    both the manifest hash map and the independent checksum file are checked.
    """

    root = package_root.resolve()
    manifest_path = root / "RELEASE_MANIFEST.json"
    checksums_path = root / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ValueError("release package requires RELEASE_MANIFEST.json and checksums.sha256")
    try:
        manifest = ReleaseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("release manifest is invalid") from exc

    expected_manifest_paths = set(manifest.file_sha256)
    actual_payload_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {"checksums.sha256", "RELEASE_MANIFEST.json"}
    }
    if expected_manifest_paths != actual_payload_paths:
        missing = sorted(expected_manifest_paths - actual_payload_paths)
        unexpected = sorted(actual_payload_paths - expected_manifest_paths)
        raise ValueError(
            f"release manifest file inventory mismatch; missing={missing}; unexpected={unexpected}"
        )
    for relative, expected in manifest.file_sha256.items():
        actual = _sha256(_safe_package_path(root, relative))
        if actual != expected:
            raise ValueError(f"manifest hash mismatch for {relative}")

    checksum_records: dict[str, str] = {}
    for line_number, line in enumerate(
        checksums_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}") from exc
        if relative in checksum_records and checksum_records[relative] != expected:
            raise ValueError(f"conflicting checksum entries for {relative}")
        checksum_records[relative] = expected
        path = _safe_package_path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"checksum references missing or symlink path: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"checksum mismatch for {relative}")
    expected_checksum_paths = actual_payload_paths | {"RELEASE_MANIFEST.json"}
    if set(checksum_records) != expected_checksum_paths:
        missing = sorted(expected_checksum_paths - set(checksum_records))
        unexpected = sorted(set(checksum_records) - expected_checksum_paths)
        raise ValueError(
            f"checksum file inventory mismatch; missing={missing}; unexpected={unexpected}"
        )

    _verify_public_withholding(root, manifest)


def _verify_public_withholding(root: Path, manifest: ReleaseManifest) -> None:
    """Reject accidental hidden labels or hidden state in public inputs."""

    if manifest.withheld_splits != ("hidden_test",):
        raise ValueError("research preview must withhold the hidden_test split")
    public_paths = (
        root / "inputs" / "public_test.jsonl",
        root / "cases" / "public_test.jsonl",
    )
    for path in public_paths:
        if not path.is_file():
            raise ValueError(f"public input file is missing: {path.relative_to(root)}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid public input JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"public input must be an object at {path}:{line_number}")
            forbidden = _find_keys(payload, _HIDDEN_RELEASE_KEYS)
            if forbidden:
                raise ValueError(
                    f"public input leaks hidden fields at {path}:{line_number}: {forbidden}"
                )


def _find_keys(value: object, forbidden: set[str], *, prefix: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            location = f"{prefix}.{key}" if prefix else str(key)
            if normalized in forbidden:
                found.append(location)
            found.extend(_find_keys(child, forbidden, prefix=location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_keys(child, forbidden, prefix=f"{prefix}[{index}]"))
    return tuple(found)


def _safe_package_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path while rejecting traversal and absolute paths."""

    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"package path must be relative and traversal-free: {relative!r}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"package path escapes release root: {relative!r}") from exc
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
