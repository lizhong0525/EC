"""Opt-in v2 commands; no credentials or paid providers are discovered here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ecocompanion.experiments.io import read_jsonl_models, write_json_atomic
from ecocompanion.research.annotation import AnnotationStore
from ecocompanion.research.chinese_cli import app as chinese_app
from ecocompanion.research.contracts import Judgment, Prediction, Protocol, Provenance, ResearchCase
from ecocompanion.research.governance import import_blind_cases
from ecocompanion.research.mhb_cli import app as mhb_app
from ecocompanion.research.reference_cli import app as reference_app
from ecocompanion.research.review import ReplyRating, blind_reply_packet, calibrate_ratings
from ecocompanion.research.workflows import (
    build_v2_report,
    prepare,
    prepare_independent,
    require_private_output,
    run_local,
)

app = typer.Typer(no_args_is_help=True, help="V2 auditable Track A research; offline by default.")
app.add_typer(chinese_app, name="zh")
app.add_typer(reference_app, name="references")
app.add_typer(mhb_app, name="mhb")
Packet = Annotated[Path, typer.Option("--packet", exists=True, file_okay=False)]


@app.command("cradle-download")
def cradle_download_command(
    output: Annotated[Path, typer.Option("--output")],
    revision: str = "49d5f40912e9afeadf4d0c1aa0ffd22c8d89c6a3",
) -> None:
    """Download a pinned public snapshot; do not upload data or discover model keys."""
    from ecocompanion.research.cradle import download_snapshot

    result = download_snapshot(output, revision=revision)
    typer.echo(f"revision={result['revision']} license={result['license']} model_calls=0")


@app.command("cradle-audit")
def cradle_audit_command(
    snapshot: Annotated[Path, typer.Option("--snapshot", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Audit actual columns, boundaries, labels and exact split overlap."""
    from ecocompanion.research.cradle import audit_snapshot

    result = audit_snapshot(snapshot)
    write_json_atomic(output, result, overwrite=False)
    typer.echo(json.dumps(result, ensure_ascii=True))


@app.command("cradle-plan")
def cradle_plan_command(
    snapshot: Annotated[Path, typer.Option("--snapshot", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    split: str = "test",
) -> None:
    """Freeze source, local strategies, native labels and isolated opaque inputs."""
    from ecocompanion.research.cradle import freeze_external_plan

    result = freeze_external_plan(snapshot, output, split=split)
    typer.echo(f"dialogues={result['dialogues']} user_nodes={result['user_nodes']} model_calls=0")


@app.command("cradle-run")
def cradle_run_command(
    packet: Packet,
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run local native-event baselines and project rules; never map events to Gold."""
    from ecocompanion.research.cradle import run_external

    result = run_external(packet, output)
    typer.echo(f"status={result['status']} replay={result['stage_replay']} model_calls=0")


@app.command("cradle-report")
def cradle_report_command(
    packet: Packet,
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Replay every prediction and recompute a new, immutable external report."""
    from ecocompanion.research.cradle import report_external

    result = report_external(packet, run, output)
    typer.echo(f"status={result['status']} replay={result['stage_replay']}")


@app.command("prepare")
def prepare_command(
    output: Annotated[Path, typer.Option("--output")],
    manual: Annotated[Path, typer.Option("--manual", exists=True)] = Path(
        "docs/ANNOTATION_OPERATIONS_V2.md"
    ),
) -> None:
    """Create twenty development-only pilot cases and an empty expert store."""
    result = prepare(output, manual)
    typer.echo(f"prepared={result['case_count']} nodes={result['node_count']} gold=PENDING")


@app.command("serve")
def serve_command(packet: Packet, port: int = 8765) -> None:
    """Serve role-isolated annotation on loopback; distribute credentials separately."""
    import uvicorn

    from ecocompanion.research.api import annotation_app

    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    store.audit()
    uvicorn.run(annotation_app(store), host="127.0.0.1", port=port, access_log=False)


@app.command("audit")
def audit_command(packet: Packet) -> None:
    typer.echo(
        json.dumps(AnnotationStore(packet / "private" / "annotations.sqlite3").audit(), indent=2)
    )


@app.command("agreement")
def agreement_command(packet: Packet, output: Annotated[Path, typer.Option("--output")]) -> None:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    require_private_output(store.cases(), packet, output)
    result = store.agreement()
    write_json_atomic(output, result, overwrite=False)
    typer.echo(f"agreement={result['status']}")


@app.command("attest")
def attest_command(
    packet: Packet,
    reviewer: str,
    reference: str,
    administrator: str,
    person_id: str,
    qualification: str,
) -> None:
    """Record an administrator's ACTUAL external human qualification/identity check."""
    AnnotationStore(packet / "private" / "annotations.sqlite3").attest(
        reviewer,
        reference=reference,
        administrator=administrator,
        person_id=person_id,
        qualification=qualification,
    )
    typer.echo(
        "attestation recorded; this command does not itself verify a person's qualifications"
    )


@app.command("freeze-gold")
def freeze_command(packet: Packet, output: Annotated[Path, typer.Option("--output")]) -> None:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    if not output.resolve().is_relative_to((packet / "private").resolve()):
        raise ValueError(
            "expert Gold exports remain curator-private until separate publication review"
        )
    store.freeze(output)
    typer.echo("Gold exported from locked independent annotations and adjudications")


@app.command("run-local")
def run_command(packet: Packet, output: Annotated[Path, typer.Option("--output")]) -> None:
    report = run_local(packet, output)
    typer.echo(f"execution={report['execution']} expert_metrics=UNAVAILABLE_WITHOUT_GOLD")


@app.command("report")
def report_command(
    packet: Packet,
    predictions: Annotated[Path, typer.Option("--predictions", exists=True)],
    output: Annotated[Path, typer.Option("--output")],
    gold: Annotated[Path | None, typer.Option("--gold", exists=True)] = None,
    protocol: Annotated[Path | None, typer.Option("--protocol", exists=True)] = None,
) -> None:
    build_v2_report(packet, predictions, output, gold=gold, protocol_path=protocol)
    typer.echo("v2 report created without modifying earlier runs")


@app.command("import-blind")
def import_blind_command(
    packet: Packet,
    source: Annotated[Path, typer.Option("--source", exists=True)],
    administrator: str,
    independence_reference: str,
) -> None:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    import_blind_cases(
        source,
        packet / "private",
        administrator=administrator,
        independence_reference=independence_reference,
        existing=store.cases(),
    )
    typer.echo(
        "independent blind material staged privately; Gold and semantic review remain pending"
    )


@app.command("blind-replies")
def replies_command(
    packet: Packet,
    predictions: Annotated[Path, typer.Option("--predictions", exists=True)],
    output: Annotated[Path, typer.Option("--output")],
    seed: int = 20261020,
) -> None:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    require_private_output(store.cases(), packet, output)
    result = blind_reply_packet(
        store.cases(), read_jsonl_models(predictions, Prediction), output, seed=seed
    )
    typer.echo(f"anonymous_tasks={result['tasks']} status={result['status']}")


@app.command("prepare-independent")
def independent_command(
    staged: Annotated[Path, typer.Option("--staged", exists=True)],
    review: Annotated[Path, typer.Option("--review", exists=True)],
    output: Annotated[Path, typer.Option("--output")],
    manual: Annotated[Path, typer.Option("--manual", exists=True)],
) -> None:
    result = prepare_independent(staged, output, manual, review=review)
    typer.echo(f"private_cases={result['case_count']} gold=PENDING")


@app.command("label-status")
def status_command(packet: Packet, output: Annotated[Path, typer.Option("--output")]) -> None:
    store = AnnotationStore(packet / "private" / "annotations.sqlite3")
    require_private_output(store.cases(), packet, output)
    write_json_atomic(output, store.label_status(), overwrite=False)
    typer.echo("unfrozen label provenance exported; no automatic Gold promotion")


@app.command("retrospective")
def retrospective_command(
    packet: Packet,
    source: Annotated[Path, typer.Option("--source", exists=True)],
) -> None:
    document = json.loads(source.read_text(encoding="utf-8"))
    AnnotationStore(packet / "private" / "annotations.sqlite3").record_retrospective(
        case_id=document["case_id"],
        windows=document["windows"],
        reason=document["reason"],
        reviewer_reference=document["reviewer_reference"],
    )
    typer.echo("retrospective review appended; prospective labels/windows unchanged")


@app.command("calibrate")
def calibrate_command(
    tasks: Annotated[Path, typer.Option("--tasks", exists=True)],
    ratings: Annotated[Path, typer.Option("--ratings", exists=True)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    packet = json.loads(tasks.read_text(encoding="utf-8"))
    from ecocompanion.research.annotation import digest

    if packet.get("task_digest") != digest(packet["tasks"]):
        raise ValueError("anonymous task packet digest mismatch")
    result = calibrate_ratings(
        read_jsonl_models(ratings, ReplyRating),
        allowed_candidates={t["candidate_id"] for t in packet["tasks"]},
    )
    write_json_atomic(output, result, overwrite=False)
    typer.echo(f"calibration={result['status']}")


@app.command("sensitivity")
def sensitivity_command(
    output: Annotated[Path, typer.Option("--output")],
    max_cases: int = 24,
) -> None:
    from ecocompanion.research.sensitivity import run_sensitivity

    result = run_sensitivity(max_cases=max_cases)
    write_json_atomic(output, result, overwrite=False)
    typer.echo(f"measured_episodes={len(result['episodes'])} assumption_validation=PENDING")


@app.command("schemas")
def schemas_command(output: Annotated[Path, typer.Option("--output")]) -> None:
    for model in (Judgment, Prediction, Protocol, Provenance, ResearchCase, ReplyRating):
        write_json_atomic(
            output / f"{model.__name__}.schema.json", model.model_json_schema(), overwrite=False
        )
    typer.echo("v2 schemas exported")


@app.command("validity-audit")
def validity_command(
    packet: Packet,
    output: Annotated[Path, typer.Option("--output")],
    gold: Annotated[Path | None, typer.Option("--gold", exists=True)] = None,
) -> None:
    """P0 engineering challenges and external-readiness gates; never creates Gold."""
    from ecocompanion.research.validity import audit_validity

    cases = AnnotationStore(packet / "private" / "annotations.sqlite3").cases()
    require_private_output(cases, packet, output)
    result = audit_validity(packet, gold=gold)
    write_json_atomic(output, result, overwrite=False)
    typer.echo(f"fixture_checks={result['challenge_suite']['status']} publication=NOT_ESTABLISHED")


@app.command("diagnostic-plan")
def diagnostic_plan_command(
    packet: Packet,
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Freeze P1 conditions, input hashes, implementation, budget and analysis first."""
    from ecocompanion.research.diagnostics import freeze_plan

    result = freeze_plan(packet, output)
    typer.echo(f"conditions={len(result.conditions)} execution=local_rules gold=NOT_CREATED")


@app.command("diagnostic-run")
def diagnostic_run_command(
    packet: Packet,
    plan: Annotated[Path, typer.Option("--plan", exists=True)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Execute the frozen deterministic study and replay all recorded stage artifacts."""
    from ecocompanion.research.diagnostics import run_diagnostics

    result = run_diagnostics(packet, plan, output)
    typer.echo(
        f"recorded_nodes={result['stage_replay']['verified_nodes']} "
        f"replay={result['stage_replay']['status']} model_calls=0"
    )


@app.command("diagnostic-report")
def diagnostic_report_command(
    packet: Packet,
    plan: Annotated[Path, typer.Option("--plan", exists=True)],
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    gold: Annotated[Path | None, typer.Option("--gold", exists=True)] = None,
) -> None:
    """Revalidate executed traces and optionally attach audit-anchored human Gold."""
    from ecocompanion.research.diagnostics import report_diagnostics

    result = report_diagnostics(packet, plan, run, output, gold=gold)
    typer.echo(f"measurement={result['measurement_version']} model_experiment=NOT_RUN")


@app.command("diagnostic-replies")
def diagnostic_replies_command(
    packet: Packet,
    plan: Annotated[Path, typer.Option("--plan", exists=True)],
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Revalidate stages, then export anonymous replies with condition-matched evidence."""
    from ecocompanion.research.diagnostics import diagnostic_replies

    result = diagnostic_replies(packet, plan, run, output)
    typer.echo(f"anonymous_tasks={result['tasks']} ratings=NOT_COLLECTED")
