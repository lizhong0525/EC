import json
from pathlib import Path

from ecocompanion.schema_export import (
    ARTIFACT_SCHEMA_MODELS,
    SCHEMA_MODELS,
    export_schemas,
)


def test_export_all_public_schemas(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert len(written) == len(SCHEMA_MODELS)
    assert all(path.exists() for path in written)
    assert (tmp_path / "ReleaseArtifactRecord.schema.json").exists()
    assert (tmp_path / "ReleaseManifest.schema.json").exists()
    case_schema = json.loads((tmp_path / "CaseSpecV1.schema.json").read_text("utf-8"))
    assert case_schema["title"] == "CaseSpecV1"
    assert len(ARTIFACT_SCHEMA_MODELS) == 12
    validation_schema = json.loads(
        (tmp_path / "BenchmarkValidationReport.schema.json").read_text("utf-8")
    )
    leakage_schema = json.loads(
        (tmp_path / "NearDuplicateAuditV1.schema.json").read_text("utf-8")
    )
    assert "near_duplicate_audit" in validation_schema["properties"]
    assert leakage_schema["properties"]["schema_version"]["const"] == "1.0"
    track_b_schema = json.loads(
        (tmp_path / "TrackBExperimentResult.schema.json").read_text("utf-8")
    )
    track_c_schema = json.loads(
        (tmp_path / "TrackCExperimentResult.schema.json").read_text("utf-8")
    )
    assert track_b_schema["properties"]["track"]["const"] == "B"
    assert track_c_schema["properties"]["track"]["const"] == "C"
    report_schema = json.loads(
        (tmp_path / "ReportDocument.schema.json").read_text("utf-8")
    )
    assert report_schema["properties"]["schema_version"]["const"] == "2.0"
    assert "MetricEstimateRecord" in report_schema["$defs"]
    assert "PairedComparisonRecord" in report_schema["$defs"]
    assert "FailureCaseRecord" in report_schema["$defs"]
    assert "RelationshipSafetyBreakdown" in json.loads(
        (tmp_path / "TrackCReportDocument.schema.json").read_text("utf-8")
    )["$defs"]
