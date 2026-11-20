"""Explicit local-only MentalHealthBench commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ecocompanion.experiments.io import write_json_atomic

app = typer.Typer(
    no_args_is_help=True,
    help="External next-response rubric track; never creates EcoTriage action Gold.",
)
Packet = Annotated[Path, typer.Option("--packet", exists=True, file_okay=False)]
Config = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]
Run = Annotated[Path, typer.Option("--run", exists=True, file_okay=False)]


@app.command("freeze-paper")
def freeze_paper_command(output: Annotated[Path, typer.Option("--output")]) -> None:
    from ecocompanion.research.mhb import freeze_paper

    result = freeze_paper(output)
    typer.echo(f"paper_sha256={result['paper_sha256']} model_calls=0")


@app.command("prepare")
def prepare_command(
    snapshot: Annotated[Path, typer.Option("--snapshot", exists=True, file_okay=False)],
    paper: Annotated[Path, typer.Option("--paper", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    scope: Annotated[str, typer.Option("--scope", help="smoke or public")] = "smoke",
) -> None:
    from ecocompanion.research.mhb import prepare

    result = prepare(snapshot, paper, output, scope=scope)
    typer.echo(
        f"scope={result['scope']} tasks={result['tasks']} "
        f"rubrics={result['rubric_items']} status=NOT_EXECUTED"
    )


@app.command("bind-local")
def bind_command(
    packet: Packet,
    model: Annotated[str, typer.Option("--model")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    from ecocompanion.research.mhb import bind_local

    result = bind_local(packet, model, output)
    typer.echo(f"model={result['model']} digest={result['model_digest']} scope=LOCAL_DIAGNOSTIC")


@app.command("run")
def run_command(
    packet: Packet,
    config: Config,
    output: Annotated[Path, typer.Option("--output")],
    resume: Annotated[
        bool, typer.Option("--resume", help="Only unattempted calls; never retry")
    ] = False,
) -> None:
    from ecocompanion.research.mhb import run_local

    result = run_local(packet, config, output, resume=resume)
    typer.echo(
        f"status={result['status']} complete_tasks={result['complete_tasks']}/"
        f"{result['expected_tasks']} calls={result['total_model_calls']}"
    )


@app.command("report")
def report_command(
    packet: Packet,
    config: Config,
    run: Run,
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    from ecocompanion.research.mhb import report

    result = report(packet, config, run, output)
    typer.echo(f"status={result['status']} replay=VERIFIED")


@app.command("verify")
def verify_command(packet: Packet) -> None:
    from ecocompanion.research.mhb import load_packet

    result = load_packet(packet)
    typer.echo(json.dumps({"status": "VERIFIED", "tasks": len(result["tasks"])}))


@app.command("probe")
def probe_command(output: Annotated[Path, typer.Option("--output")]) -> None:
    from ecocompanion.research.mhb import local_models

    try:
        models = local_models()
        result = {
            "status": "LOCAL_SERVICE_AVAILABLE",
            "models": models,
            "inference_calls": 0,
            "remote_uploads": 0,
        }
    except Exception as exc:  # probe reports service state without starting inference
        if exc.__class__.__module__.split(".")[0] not in {"httpx", "json"}:
            raise
        result = {
            "status": "LOCAL_SERVICE_UNAVAILABLE",
            "error_type": type(exc).__name__,
            "models": [],
            "inference_calls": 0,
            "remote_uploads": 0,
        }
    write_json_atomic(output, result, overwrite=False)
    typer.echo(f"status={result['status']} inference_calls=0")
