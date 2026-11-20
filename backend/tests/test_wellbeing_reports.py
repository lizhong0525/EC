import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ecocompanion.application.models import TurnRequest
from ecocompanion.pet.api import create_pet_app
from ecocompanion.pet.settings import PetSettings, SettingsStore, SettingsUpdate
from ecocompanion.pet.training import (
    TrainingExample,
    deterministic_split,
    load_examples,
    write_sft,
)


def auth(client: TestClient) -> dict[str, str]:
    return {"X-Pet-Token": client.get("/api/pet/bootstrap").json()["token"]}


def conversation(client: TestClient) -> tuple[dict[str, str], str, str]:
    headers = auth(client)
    client.app.state.pet_store.update(SettingsUpdate(
        settings=client.app.state.pet_store.settings.model_copy(update={"cloud_consent": True}),
        deepseek_api_key="test-only",
    ))
    session = client.post("/api/pet/sessions", json={}, headers=headers).json()["session_id"]
    client.post(
        f"/api/pet/sessions/{session}/turn",
        headers=headers,
        json={"text": "今天加班有点累", "request_id": "report-test-turn"},
    ).raise_for_status()
    messages = client.get(f"/api/pet/sessions/{session}/messages").json()
    return headers, session, messages[0]["message_id"]


def payload(session: str, message: str, **overrides) -> dict:
    return {
        "session_id": session,
        "message_ids": [message],
        "summary": "我的回顾",
        "check_in": {
            "local_date": "2026-10-27",
            "mood": 3,
            "energy": 2,
            "sleep": 3,
            "stress": 4,
            "note": "需要早点休息",
        },
        **overrides,
    }


def test_local_report_preview_snapshot_export_and_delete(tmp_path: Path):
    with TestClient(create_pet_app(tmp_path)) as client:
        headers, session, message = conversation(client)
        body = payload(session, message)
        preview = client.post("/api/pet/reports/preview", json=body, headers=headers)
        assert preview.status_code == 200, preview.text
        assert preview.json()["summary_source"] == "user"
        saved = client.post(
            "/api/pet/reports",
            json={**body, "preview_id": preview.json()["preview_id"]},
            headers=headers,
        )
        assert saved.status_code == 201, saved.text
        report_id = saved.json()["report_id"]
        assert client.post("/api/pet/reports", json=body, headers=headers).status_code == 409
        assert len(client.get("/api/pet/reports").json()["items"]) == 1
        printed = client.get(f"/api/pet/reports/{report_id}/print")
        assert printed.status_code == 200
        assert "今天加班有点累" in printed.text and "不是医学诊断" in printed.text
        assert client.delete(f"/api/pet/reports/{report_id}", headers=headers).status_code == 204
        assert client.get(f"/api/pet/reports/{report_id}").status_code == 404


def test_report_rejects_foreign_or_assistant_evidence_and_deletes_session_quotes(tmp_path: Path):
    with TestClient(create_pet_app(tmp_path)) as client:
        headers, session, message = conversation(client)
        other = client.post("/api/pet/sessions", json={}, headers=headers).json()["session_id"]
        response = client.post(
            "/api/pet/reports/preview",
            json=payload(session, message, session_id=other),
            headers=headers,
        )
        assert response.status_code == 400
        all_messages = client.get(f"/api/pet/sessions/{session}/messages").json()
        assert (
            client.post(
                "/api/pet/reports/preview",
                json=payload(session, all_messages[1]["message_id"]),
                headers=headers,
            ).status_code
            == 400
        )
        preview = client.post(
            "/api/pet/reports/preview", json=payload(session, message, summary=""), headers=headers
        ).json()
        saved = client.post(
            "/api/pet/reports",
            json=payload(session, message, summary="", preview_id=preview["preview_id"]),
            headers=headers,
        ).json()
        assert client.delete(f"/api/pet/sessions/{session}", headers=headers).status_code == 204
        remaining = client.get(f"/api/pet/reports/{saved['report_id']}").json()
        assert remaining["evidence"] == [] and remaining["summary"] == ""
        assert remaining["summary_source"] == "removed"
        assert "今天加班" not in json.dumps(remaining, ensure_ascii=False)


def test_report_uses_saved_qwen_assessment_not_generated_reply_wording(tmp_path: Path):
    requests = []
    sources = []

    def transport(request: httpx.Request):
        requests.append(request)
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/api/chat"
        sources.append(json.loads(json.loads(request.content)["messages"][1]["content"]))
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "safety_signal": "none",
                            "evidence_quotes": ["加班有点累"],
                            "support_focus": "关注工作后的疲劳",
                            "needs_clarification": False,
                            "wellbeing_level": "strained",
                            "mood": "negative",
                            "stress": "moderate",
                            "rationale": "当前原话明确提到工作后的疲劳。",
                        },
                        ensure_ascii=False,
                    )
                }
            },
        )

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(SettingsUpdate(settings=PetSettings(qwen_enabled=True)))
    with TestClient(app) as client:
        headers, session, message = conversation(client)
        assert sources[0]["current_message"] == "今天加班有点累"
        body = payload(session, message, summary="")
        preview = client.post("/api/pet/reports/preview", json=body, headers=headers)
        assert preview.status_code == 200, preview.text
        assert preview.json()["summary_source"] == "extractive"
        assessment = preview.json()["assessments"][0]
        assert assessment["qwen"]["wellbeing_level"] == "strained"
        assert assessment["qwen"]["rationale"] == "当前原话明确提到工作后的疲劳。"
        assert assessment["decision"]["action"]
        assert len(requests) == 2
        assert sum(request.url.host == "api.deepseek.com" for request in requests) == 1
        saved = client.post(
            "/api/pet/reports",
            json={**body, "preview_id": preview.json()["preview_id"]},
            headers=headers,
        )
        assert saved.status_code == 201
        edited = client.post(
            "/api/pet/reports/preview", json=body, headers=headers
        ).json()
        saved_edit = client.post(
            "/api/pet/reports",
            json={**body, "summary": "我修订后的回顾", "preview_id": edited["preview_id"]},
            headers=headers,
        ).json()
        assert saved_edit["summary_source"].endswith("+user-edit")
        assert saved_edit["summary"] == "我修订后的回顾"


def test_training_data_provenance_and_group_isolation(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    examples = load_examples(
        root / "training/data/reviewed_examples.jsonl", root / "training/data/source_registry.json"
    )
    splits = deterministic_split(examples)
    write_sft(examples, splits, tmp_path)
    assert all((tmp_path / f"{split}.jsonl").is_file() for split in splits)
    assert set().union(*[set(groups) for groups in splits.values()]) == {
        example.group_id for example in examples
    }
    (tmp_path / "bad.jsonl").write_text(
        TrainingExample(
            example_id="external-001",
            group_id="external",
            text="外部文本",
            target={
                "safety_signal": "none",
                "evidence_quotes": [],
                "support_focus": "倾听",
                "needs_clarification": False,
            },
            source_id="unapproved",
            reviewer="test-reviewer",
            review_status="reviewed",
            scenario="ordinary",
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not explicitly approved"):
        load_examples(tmp_path / "bad.jsonl", root / "training/data/source_registry.json")


def test_channel_report_uses_saved_assessment_without_exposing_external_id(tmp_path: Path):
    from ecocompanion.application.models import CreateSessionRequest
    from ecocompanion.application.repository import ChannelBinding
    from ecocompanion.pet.pipeline import TurnContext, turn_context

    async def create_channel_turn():
        from ecocompanion.application.repository import SqlAlchemySessionRepository
        from ecocompanion.application.service import CompanionService
        from ecocompanion.infrastructure import CostLedger, JsonlTraceStore
        from ecocompanion.pet.pipeline import PetPipeline, PetRenderer
        from ecocompanion.pet.reports import assessment_snapshot

        store = SettingsStore(tmp_path / "channel-store")
        store.update(SettingsUpdate(
            settings=PetSettings(qwen_enabled=False, cloud_consent=True),
            deepseek_api_key="test-only",
        ))
        (tmp_path / "test-messaging").mkdir(parents=True, exist_ok=True)
        repository = SqlAlchemySessionRepository(
            f"sqlite+aiosqlite:///{tmp_path / 'test-messaging' / 'conversations.db'}"
        )
        http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
        pipeline = PetPipeline(http)
        service = CompanionService(
            repository=repository,
            renderer=PetRenderer(store),
            trace_store=JsonlTraceStore(tmp_path / "channel-trace.jsonl"),
            cost_ledger=CostLedger(),
            turn_advisor=pipeline.advise,
            turn_snapshot=assessment_snapshot,
        )
        await service.initialize()
        session = await service.create_session(CreateSessionRequest(memory_enabled=False))
        await repository.save_channel_binding(
            ChannelBinding("qq", "PRIVATE-EXTERNAL-ID", session.session_id, True)
        )
        context = TurnContext(store.settings, {})
        marker = turn_context.set(context)
        try:
            await service.process_turn(session.session_id, TurnRequest(
                text="我今天很累", request_id="channel-turn"
            ))
        finally:
            turn_context.reset(marker)
        await service.close()
        await http.aclose()

    import asyncio
    asyncio.run(create_channel_turn())
    with TestClient(create_pet_app(tmp_path)) as client:
        headers = auth(client)
        sources = client.get("/api/pet/reports/sources", headers=headers).json()["items"]
        assert len(sources) == 1
        assert sources[0]["source"] == "qq"
        assert "PRIVATE-EXTERNAL-ID" not in json.dumps(sources)
        session = sources[0]["session_id"]
        messages = client.get(
            f"/api/pet/reports/sources/qq/{session}/messages", headers=headers
        ).json()
        preview = client.post(
            "/api/pet/reports/preview",
            json=payload(session, messages[0]["message_id"], source="qq"),
            headers=headers,
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["assessments"][0]["decision"]["action"]
