"""Export versioned Pydantic contracts as JSON Schema files."""

import json
from pathlib import Path

from pydantic import BaseModel

from ecocompanion.benchmark.leakage import NearDuplicateAuditV1
from ecocompanion.benchmark.log_contract import (
    ExternalEventLogDocumentV1,
    ExternalEventLogRecordV1,
    ImportedCaseDraftV1,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    CaseSpecV1,
    ObservationV1,
    SessionStateV1,
    StudentTurnV1,
    TraceEventV1,
)
from ecocompanion.experiments.models import (
    BenchmarkValidationReport,
    ExperimentCheckpoint,
    ExperimentManifest,
    ExperimentResult,
    ReportDocument,
    TrackBExperimentManifest,
    TrackBExperimentResult,
    TrackBReportDocument,
    TrackCExperimentManifest,
    TrackCExperimentResult,
    TrackCReportDocument,
)
from ecocompanion.experiments.release import ReleaseArtifactRecord, ReleaseManifest

INTERFACE_SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    CaseSpecV1,
    ObservationV1,
    AgentDecisionV1,
    StudentTurnV1,
    SessionStateV1,
    TraceEventV1,
    ExternalEventLogRecordV1,
    ExternalEventLogDocumentV1,
    ImportedCaseDraftV1,
)

ARTIFACT_SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    BenchmarkValidationReport,
    NearDuplicateAuditV1,
    ExperimentManifest,
    ExperimentCheckpoint,
    ExperimentResult,
    ReportDocument,
    TrackBExperimentManifest,
    TrackBExperimentResult,
    TrackBReportDocument,
    TrackCExperimentManifest,
    TrackCExperimentResult,
    TrackCReportDocument,
)

RELEASE_SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    ReleaseArtifactRecord,
    ReleaseManifest,
)

SCHEMA_MODELS = (
    *INTERFACE_SCHEMA_MODELS,
    *ARTIFACT_SCHEMA_MODELS,
    *RELEASE_SCHEMA_MODELS,
)


def export_schemas(output_dir: Path) -> tuple[Path, ...]:
    """Atomically write all public JSON Schemas and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in SCHEMA_MODELS:
        target = output_dir / f"{model.__name__}.schema.json"
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
        written.append(target)
    return tuple(written)
