"""Exercise the real local Web render/rewrite/fallback pipeline on retained probes."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ecocompanion.agent import SafetyAuditor
from ecocompanion.application.conversation import response_quality_codes, web_audit_action
from ecocompanion.application.models import CreateSessionRequest, TurnRequest
from ecocompanion.application.renderer import LLMResponseRenderer
from ecocompanion.application.repository import SqlAlchemySessionRepository, StoredMessage
from ecocompanion.application.service import CompanionService
from ecocompanion.domain.enums import ActorRole, RiskLevel
from ecocompanion.infrastructure import CostLedger, JsonlTraceStore
from ecocompanion.providers import OpenAICompatibleClient
from scripts import distillation as d
from scripts import risk_evaluation as r


class LocalCapture(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.probe = ""
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != "127.0.0.1" or request.url.port != 11434:
            raise RuntimeError("only the fixed loopback model is allowed")
        if len(self.calls) >= 32:
            raise RuntimeError("local pipeline call limit exceeded")
        record: dict[str, Any] = {
            "probe_id": self.probe,
            "request": json.loads(request.content),
        }
        self.calls.append(record)
        try:
            response = await self.inner.handle_async_request(request)
            await response.aread()
            record.update({"status": response.status_code, "response": response.json()})
            return response
        except (httpx.HTTPError, ValueError) as exc:
            record["error"] = type(exc).__name__
            raise

    async def aclose(self) -> None:
        await self.inner.aclose()


async def run(output: Path, raw_run: Path, feedback_path: Path) -> dict[str, Any]:
    plan = d.read_json(raw_run / "plan.json")
    for name, expected in plan["implementation_hashes"].items():
        if d.digest((d.ROOT / name).read_bytes()) != expected:
            raise ValueError("implementation changed after raw rerun")
    with httpx.Client(trust_env=False) as identity_client:
        if r.model_digest(identity_client) != plan["model_digest"]:
            raise ValueError("local model identity changed")
    probes = d.read_json(raw_run / "probes.json")
    feedback = d.read_json(feedback_path)
    output.mkdir(parents=True, exist_ok=False)
    d.write_json(
        output / "plan.json",
        {
            "raw_run_plan_sha256": d.digest((raw_run / "plan.json").read_bytes()),
            "feedback_sha256": d.digest(feedback_path.read_bytes()),
            "source_docx_sha256": feedback["source_sha256"],
            "model": r.MODEL,
            "model_digest": plan["model_digest"],
            "max_calls": 32,
            "max_tokens": 384,
            "temperature": 0,
            "seed": 20261026,
            "same_model_retries": 0,
            "paid_calls": 0,
            "history": "Seeded retained fictional context; only current turn is generated",
            "handoff": "SIMULATED_ONLY",
        },
    )
    capture = LocalCapture()
    client = OpenAICompatibleClient(
        base_url=r.BASE + "/v1",
        api_key=None,
        max_same_model_retries=0,
        default_extra_body={"seed": 20261026},
        transport=capture,
    )
    renderer = LLMResponseRenderer(client=client, model=r.MODEL, max_tokens=384, temperature=0)
    results = []
    try:
        for probe in probes:
            with httpx.Client(trust_env=False) as identity_client:
                if r.model_digest(identity_client) != plan["model_digest"]:
                    raise ValueError("local model identity changed during verification")
            capture.probe = probe["id"]
            repository = SqlAlchemySessionRepository("sqlite+aiosqlite:///:memory:")
            directory = output / probe["id"]
            directory.mkdir()
            service = CompanionService(
                repository=repository,
                renderer=renderer,
                trace_store=JsonlTraceStore(directory / "trace.jsonl"),
                cost_ledger=CostLedger(path=directory / "cost.jsonl"),
            )
            await service.initialize()
            try:
                view = await service.create_session(CreateSessionRequest(memory_enabled=False))
                state = view.state
                history = probe.get("history", [])
                now = datetime(2026, 10, 26, tzinfo=UTC)
                for index in range(0, len(history), 2):
                    messages = tuple(
                        StoredMessage(
                            message_id=f"{view.session_id}:seed:{ordinal}",
                            session_id=view.session_id,
                            ordinal=ordinal,
                            role=ActorRole.USER if item["role"] == "user" else ActorRole.COMPANION,
                            content=item["content"],
                            simulated_time=now.isoformat(),
                            created_at=now,
                        )
                        for ordinal, item in enumerate(history[index : index + 2], index)
                    )
                    next_state = state.model_copy(
                        update={
                            "turn_index": state.turn_index + 1,
                            "risk_level": RiskLevel(probe.get("prior_level", "L0")),
                        }
                    )
                    await repository.commit_turn(
                        session_id=view.session_id,
                        expected_turn_index=state.turn_index,
                        next_state=next_state,
                        messages=messages,
                    )
                    state = next_state
                result = await service.process_turn(
                    view.session_id, TurnRequest(text=probe["text"], request_id=probe["id"])
                )
                observation, _, web, _ = r.decisions(probe)
                audit = SafetyAuditor().audit(
                    result.response,
                    risk_level=web.outcome.plan.escalation_level,
                    handoff_status=result.session.state.handoff_status,
                    required_action=web_audit_action(web.outcome),
                )
                quality = response_quality_codes(
                    result.response, observation=observation, outcome=web.outcome
                )
                record = {
                    "probe_id": probe["id"],
                    "turn": result.model_dump(mode="json"),
                    "final_audit_codes": audit.violation_codes,
                    "final_quality_codes": quality,
                    "final_contract_pass": audit.passed and not quality,
                    "new_human_preference": None,
                }
                d.write_json(directory / "result.json", record)
                results.append(record)
                print(
                    probe["id"],
                    result.decision.primary_action.value,
                    "fallback=" + str(result.decision.used_fallback),
                    "final_codes=" + str((*audit.violation_codes, *quality)),
                    flush=True,
                )
            finally:
                await service.close()
    finally:
        await client.aclose()
        d.write_json(output / "wire-calls.json", capture.calls)
    with httpx.Client(trust_env=False) as identity_client:
        if r.model_digest(identity_client) != plan["model_digest"]:
            raise ValueError("local model identity changed at completion")
    summary = {
        "probes": len(results),
        "wire_calls": len(capture.calls),
        "render_calls": sum(
            1
            for row in capture.calls
            if "候选回复：" not in row["request"]["messages"][-1]["content"]
        ),
        "rewrite_calls": sum(
            1 for row in capture.calls if "候选回复：" in row["request"]["messages"][-1]["content"]
        ),
        "used_fallback": sum(row["turn"]["decision"]["used_fallback"] for row in results),
        "final_contract_pass": sum(row["final_contract_pass"] for row in results),
        "paid_calls": 0,
        "new_human_preferences": 0,
        "files": {
            str(path.relative_to(output)): d.digest(path.read_bytes())
            for path in output.rglob("*.json")
        },
    }
    d.write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output, args.raw_run, args.feedback))


if __name__ == "__main__":
    main()
