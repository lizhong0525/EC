"""Explicit Chinese source commands: local-only inference, separate human tasks."""

from pathlib import Path
from typing import Annotated

import typer

from ecocompanion.experiments.io import write_json_atomic

app = typer.Typer(no_args_is_help=True, help="Pinned Chinese sources; no real model calls.")
Snapshot = Annotated[Path, typer.Option("--snapshot", exists=True, file_okay=False)]
Packet = Annotated[Path, typer.Option("--packet", exists=True, file_okay=False)]
Output = Annotated[Path, typer.Option("--output")]


@app.command("download")
def download(source: str, output: Output) -> None:
    from ecocompanion.research.chinese_sources import download_chinese

    result = download_chinese(source, output)
    typer.echo(f"source={source} revision={result['revision']} no_remote_code=True")


@app.command("audit")
def audit(snapshot: Snapshot, output: Output) -> None:
    from ecocompanion.research.chinese_sources import audit_chinese

    result = audit_chinese(snapshot)
    write_json_atomic(output, result, overwrite=False)
    typer.echo(f"source={result['source']} gold=UNAVAILABLE")


@app.command("plan")
def plan(snapshot: Snapshot, output: Output, pilot_count: int = 20) -> None:
    from ecocompanion.research.chinese_study import freeze_chinese

    result = freeze_chinese(snapshot, output, pilot_count=pilot_count)
    typer.echo(f"cases={result['cases']} nodes={result['nodes']} model_calls=0")


@app.command("run")
def run(packet: Packet, output: Output) -> None:
    from ecocompanion.research.chinese_study import run_chinese

    result = run_chinese(packet, output)
    typer.echo(f"status={result['status']} replay={result['stage_replay']} model_calls=0")


@app.command("report")
def report(
    packet: Packet, run: Annotated[Path, typer.Option("--run", exists=True)], output: Output
) -> None:
    from ecocompanion.research.chinese_study import chinese_report

    result = chinese_report(packet, run, output)
    typer.echo(f"status={result['status']} replay={result['stage_replay']}")


@app.command("calibration-packet")
def calibration(snapshot: Snapshot, output: Output, holdout_groups: int = 100) -> None:
    from ecocompanion.research.chinese_sources import calibration_packet

    result = calibration_packet(snapshot, output, holdout_groups=holdout_groups)
    typer.echo(f"tasks={result['tasks']} groups={result['input_groups']} ratings=NOT_GENERATED")


@app.command("calibrate")
def calibrate(
    packet: Packet,
    predictions: Annotated[Path, typer.Option("--predictions", exists=True)],
    output: Output,
) -> None:
    from ecocompanion.research.chinese_study import score_calibration

    result = score_calibration(packet, predictions, output)
    typer.echo(f"status={result['status']} local_expert_calibration=NOT_CONDUCTED")


@app.command("annotation-packet")
def annotation(
    packet: Packet,
    output: Output,
    manual: Annotated[Path, typer.Option("--manual", exists=True)] = Path(
        "docs/CHINESE_EXTERNAL_ANNOTATION.md"
    ),
) -> None:
    from ecocompanion.research.chinese_study import annotation_packet

    result = annotation_packet(packet, output, manual)
    typer.echo(f"cases={result['cases']} nodes={result['nodes']} gold=NOT_FROZEN")


@app.command("reply-packet")
def replies(
    packet: Packet, run: Annotated[Path, typer.Option("--run", exists=True)], output: Output
) -> None:
    from ecocompanion.research.chinese_study import chinese_reply_packet

    result = chinese_reply_packet(packet, run, output)
    typer.echo(f"tasks={result['tasks']} status={result['status']}")
