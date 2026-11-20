"""Offline-first guideline reference commands; downloads require an explicit command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ecocompanion.experiments.io import read_jsonl_models, write_json_atomic
from ecocompanion.research.annotation import AnnotationStore, digest
from ecocompanion.research.contracts import Prediction, Protocol
from ecocompanion.research.reference_contracts import (
    GuidelineCatalog,
    GuidelineMapping,
    MappingReview,
    RuleReview,
)
from ecocompanion.research.references import ReferenceStore, fetch_sources, reference_report
from ecocompanion.research.workflows import prepare

app = typer.Typer(no_args_is_help=True, help="Private guideline Silver; never manufactures Gold.")
Ledger = Annotated[Path, typer.Option("--ledger", exists=True, file_okay=False)]
Input = Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)]
Catalog = Annotated[Path, typer.Option("--catalog", exists=True, dir_okay=False)]


def _store(ledger: Path) -> ReferenceStore:
    store = ReferenceStore(ledger / "references.sqlite3")
    store.audit()
    return store


def _write(store: ReferenceStore, output: Path, body: dict[str, Any]) -> None:
    if not output.resolve().is_relative_to(store.path.parent.resolve()):
        raise ValueError("reference tasks, lineage and reports stay within the ledger directory")
    write_json_atomic(output, body, overwrite=False)


@app.command("fetch-sources")
def fetch_command(catalog: Catalog, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Retrieve only the explicitly listed public guideline URLs, never model data."""
    result = fetch_sources(
        GuidelineCatalog.model_validate_json(catalog.read_text(encoding="utf-8")), output
    )
    typer.echo(json.dumps({s["source_id"]: s["status"] for s in result["sources"]}))


@app.command("prepare")
def prepare_command(
    catalog: Catalog,
    snapshots: Annotated[Path, typer.Option("--snapshots", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    manual: Annotated[Path, typer.Option("--manual", exists=True, dir_okay=False)] = Path(
        "docs/GUIDELINE_REFERENCE_OPERATIONS_V1.md"
    ),
    expert_manual: Annotated[Path, typer.Option("--expert-manual", exists=True)] = Path(
        "docs/ANNOTATION_OPERATIONS_V2.md"
    ),
) -> None:
    """New 20-case development packet, empty expert store and separate Silver ledger."""
    model = GuidelineCatalog.model_validate_json(catalog.read_text(encoding="utf-8"))
    from ecocompanion.research.references import read_snapshots

    read_snapshots(snapshots, model)
    prepare(output, expert_manual)
    store = ReferenceStore.create(
        output / "private" / "guideline-v1", output, model, snapshots, manual
    )
    _write(store, store.path.parent / "audit-initial.json", store.audit())
    typer.echo("development_cases=20 expert_gold=0 mappings=0 all_guideline_nodes=UNRESOLVED")


@app.command("init")
def init_command(
    packet: Annotated[Path, typer.Option("--packet", exists=True, file_okay=False)],
    catalog: Catalog,
    snapshots: Annotated[Path, typer.Option("--snapshots", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    manual: Annotated[Path, typer.Option("--manual", exists=True)] = Path(
        "docs/GUIDELINE_REFERENCE_OPERATIONS_V1.md"
    ),
) -> None:
    """Attach a new private reference ledger without changing existing expert records."""
    ReferenceStore.create(
        output,
        packet,
        GuidelineCatalog.model_validate_json(catalog.read_text(encoding="utf-8")),
        snapshots,
        manual,
    )
    typer.echo("guideline ledger created; expert database unchanged")


@app.command("task")
def task_command(
    ledger: Ledger,
    author: Annotated[str, typer.Option("--author")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Export only the author's next current-only task; this is a trusted-curator CLI."""
    store = _store(ledger)
    task = store.next_task(author)
    _write(store, output, task)
    typer.echo(f"task={task['status']}")


@app.command("review-rule")
def rule_command(ledger: Ledger, source: Input) -> None:
    """Record a real project source/scope check, not expert credential verification."""
    event = _store(ledger).review_rule(
        RuleReview.model_validate_json(source.read_text(encoding="utf-8"))
    )
    typer.echo(f"rule_review_event={event}")


@app.command("submit")
def submit_command(ledger: Ledger, source: Input) -> None:
    """Append an actual curator mapping or explicit AI draft; never overwrite."""
    event = _store(ledger).submit(
        GuidelineMapping.model_validate_json(source.read_text(encoding="utf-8"))
    )
    typer.echo(f"mapping_digest={event} expert_gold=UNCHANGED")


@app.command("review")
def review_command(ledger: Ledger, source: Input) -> None:
    """Append a different person's actual check; AI drafts cannot become Silver."""
    event = _store(ledger).review_mapping(
        MappingReview.model_validate_json(source.read_text(encoding="utf-8"))
    )
    typer.echo(f"mapping_review_event={event}")


@app.command("status")
def status_command(ledger: Ledger, output: Annotated[Path, typer.Option("--output")]) -> None:
    store = _store(ledger)
    body = store.status()
    _write(store, output, body)
    typer.echo(json.dumps(body["coverage"]))


@app.command("audit")
def audit_command(ledger: Ledger) -> None:
    typer.echo(json.dumps(_store(ledger).audit()))


@app.command("report")
def report_command(
    ledger: Ledger,
    packet: Annotated[Path, typer.Option("--packet", exists=True, file_okay=False)],
    predictions: Annotated[Path, typer.Option("--predictions", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Gold metrics remain unavailable; Silver partial constraints get a separate column."""
    store = _store(ledger)
    annotations = AnnotationStore(packet / "private" / "annotations.sqlite3")
    annotations.audit()
    if (
        digest([c.model_dump(mode="json") for c in annotations.cases()])
        != store.status()["dataset_digest"]
    ):
        raise ValueError("reference ledger belongs to a different packet")
    protocol = Protocol.model_validate_json((packet / "protocol.json").read_text(encoding="utf-8"))
    truth_path = packet / "private" / "simulation-truth.json"
    truth = (
        {
            (r["case_id"], r["step_id"]): (r["depression"], r["safety"])
            for r in json.loads(truth_path.read_text(encoding="utf-8"))["nodes"]
        }
        if truth_path.exists()
        else None
    )
    body = reference_report(
        store, protocol, read_jsonl_models(predictions, Prediction), simulation_truth=truth
    )
    _write(store, output, body)
    typer.echo("expert=UNAVAILABLE_WITHOUT_GOLD silver=SEPARATE_PARTIAL_CONSTRAINTS")
