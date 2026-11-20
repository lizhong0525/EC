import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ecocompanion.messaging.runtime import MessagingRuntime
from ecocompanion.messaging.settings import (
    MessagingSettings,
    MessagingSettingsStore,
    MessagingSettingsUpdate,
)
from ecocompanion.pet.api import create_pet_app
from ecocompanion.pet.settings import PetSettings, SettingsStore, SettingsUpdate


@pytest.fixture(autouse=True)
def isolated_messaging_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ECO_MESSAGING_HOME", str(tmp_path / "messaging"))


def headers(client: TestClient) -> dict[str, str]:
    return {"X-Pet-Token": client.get("/api/pet/bootstrap").json()["token"]}


def test_saving_channels_reloads_running_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = MessagingSettingsStore(tmp_path / "messages").update(
        MessagingSettingsUpdate(settings=MessagingSettings())
    )["settings"]
    restarted = 0

    async def start(self):
        self.stack = AsyncExitStack()
        return self.status()

    async def restart(self):
        nonlocal restarted
        restarted += 1
        if self.stack is None:
            self.stack = AsyncExitStack()
        return self.status()

    monkeypatch.setattr(MessagingRuntime, "start", start)
    monkeypatch.setattr(MessagingRuntime, "restart", restart)
    app = create_pet_app(tmp_path)
    with TestClient(app) as client:
        started = client.post(
            "/api/pet/messaging/runtime", json={"running": True}, headers=headers(client)
        )
        assert started.status_code == 200
        assert started.json()["running"]
        response = client.put(
            "/api/pet/messaging/settings", json={"settings": settings}, headers=headers(client)
        )
        assert response.status_code == 200
        assert restarted == 2


def test_qwen_connection_check_identifies_missing_model(tmp_path: Path):
    app = create_pet_app(
        tmp_path,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        ),
    )
    with TestClient(app) as client:
        response = client.post("/api/pet/test/qwen", json={}, headers=headers(client))
        assert response.status_code == 200
        assert response.json()["ok"]
        assert "qwen2.5:7b" in response.json()["detail"]
        app.state.pet_store.update(SettingsUpdate(
            settings=PetSettings(qwen_model="missing-test-model")
        ))
        missing = client.post("/api/pet/test/qwen", json={}, headers=headers(client))
        assert not missing.json()["ok"]
        assert "missing-test-model" in missing.json()["detail"]
        assert "qwen2.5:7b" in missing.json()["detail"]


def test_jev_region_restriction_is_visible(tmp_path: Path):
    app = create_pet_app(
        tmp_path,
        transport=httpx.MockTransport(lambda _: httpx.Response(
            451, json={"title": "Typesafe is not available in your region.", "status": 451}
        )),
    )
    app.state.pet_store.update(SettingsUpdate(
        settings=PetSettings(jev_enabled=True, cloud_consent=True),
        jev_api_key="test-only",
    ))
    with TestClient(app) as client:
        response = client.post("/api/pet/test/jev", json={}, headers=headers(client))
        assert response.status_code == 200
        assert not response.json()["ok"]
        assert "当前地区不可用" in response.json()["detail"]


def test_settings_protect_credentials_and_validate_origins(tmp_path: Path):
    app = create_pet_app(tmp_path)
    with TestClient(app) as client:
        assert client.put("/api/pet/settings", json={}).status_code == 403
        assert (
            client.get(
                "/api/pet/bootstrap", headers={"Origin": "https://untrusted.example"}
            ).status_code
            == 403
        )
        assert (
            client.get("/api/pet/bootstrap", headers={"Host": "attacker.example"}).status_code
            == 403
        )
        data = client.get("/api/pet/bootstrap").json()
        assert data["settings"]["cloud_consent"] is False
        updated = client.put(
            "/api/pet/settings",
            headers=headers(client),
            json={
                "settings": {**data["settings"], "pet_name": "测试伙伴"},
                "deepseek_api_key": "sk-test-do-not-use",
            },
        )
        assert updated.status_code == 200
        assert updated.json()["deepseek_key_configured"]
        assert "sk-test" not in updated.text
        assert "sk-test" not in (tmp_path / "settings.json").read_text(encoding="utf-8")
        reloaded = SettingsStore(tmp_path)
        assert reloaded.settings.pet_name == "测试伙伴"
        if os.name == "nt":
            assert reloaded.keys["deepseek"] == "sk-test-do-not-use"
        # Empty field retains an existing credential; deletion is explicit.
        view = app.state.pet_store.update(SettingsUpdate(settings=reloaded.settings))
        assert view["deepseek_key_configured"]
        view = app.state.pet_store.update(
            SettingsUpdate(settings=reloaded.settings, clear_deepseek_key=True)
        )
        assert not view["deepseek_key_configured"]


def test_local_origin_is_checked_against_the_request_not_testserver(tmp_path: Path):
    with TestClient(create_pet_app(tmp_path)) as client:
        response = client.post(
            "/api/pet/sessions/search",
            json={},
            headers={
                **headers(client),
                "Host": "127.0.0.1:55642",
                "Origin": "http://127.0.0.1:55642",
            },
        )
        assert response.status_code == 200
        assert (
            client.post(
                "/api/pet/sessions/search",
                json={},
                headers={
                    **headers(client),
                    "Host": "127.0.0.1:55642",
                    "Origin": "http://127.0.0.1:9000",
                },
            ).status_code
            == 403
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "https://localhost:11434",
        "http://localhost:11434/path",
        "http://user:pass@localhost:11434",
        "http://localhost:11434?x=1",
    ],
)
def test_qwen_endpoint_cannot_send_local_assessments_to_arbitrary_hosts(url: str):
    with pytest.raises(ValidationError):
        PetSettings(qwen_base_url=url)


def test_local_chat_persists_and_emergency_skips_all_models(tmp_path: Path):
    requests = []

    def transport(request: httpx.Request):
        requests.append(str(request.url))
        raise AssertionError("No external calls allowed in this test")

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    with TestClient(app) as client:
        auth = headers(client)
        app.state.pet_store.update(
            SettingsUpdate(
                settings=PetSettings(qwen_enabled=False, jev_enabled=True, cloud_consent=True),
                deepseek_api_key="test-only",
                jev_api_key="test-only",
            )
        )
        session = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        result = client.post(
            f"/api/pet/sessions/{session}/turn",
            headers=auth,
            json={
                "text": "我现在正准备自杀，马上就要行动了。",
                "request_id": "urgent-test",
            },
        )
        assert result.status_code == 200, result.text
        data = result.json()
        assert data["attention"]
        assert data["mode"] == "safety_fallback"
        assert data["result"]["decision"]["escalation_level"] in {"L3", "L4"}
        assert not requests
        assert not data["result"]["decision"]["model_calls"]
        assert len(client.get(f"/api/pet/sessions/{session}/messages").json()) == 2
        repeated = client.post(
            f"/api/pet/sessions/{session}/turn",
            headers=auth,
            json={
                "text": "我现在正准备自杀，马上就要行动了。",
                "request_id": "urgent-test",
            },
        )
        assert repeated.status_code == 200
        assert repeated.json()["mode"] == "recovered"
        assert repeated.json()["result"]["response"] == data["result"]["response"]
        assert not requests
        assert len(client.get(f"/api/pet/sessions/{session}/messages").json()) == 2


def test_qwen_evidence_is_grounded_and_jev_cannot_lower_safety(tmp_path: Path):
    calls = []

    def transport(request: httpx.Request):
        body = json.loads(request.content)
        calls.append((str(request.url), body))
        if request.url.host == "127.0.0.1":
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": json.dumps(
                            {
                                "safety_signal": "current",
                                "evidence_quotes": ["每天都很难过"],
                                "support_focus": "倾听情绪",
                                "needs_clarification": False,
                            }
                        )
                    }
                },
            )
        assert request.url.path == "/v1/systemone"
        assert body["model"] == "jev-1.14.0"
        assert body["state"]["current_user_message"] == "每天都很难过"
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "listen",
                        "confidence": 0.99,
                        "probabilities": {"listen": 0.99, "clarify": 0.01, "safety_check": 0.0},
                    }
                },
            },
        )

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(jev_enabled=True, cloud_consent=True),
            jev_api_key="test-only",
        )
    )
    with TestClient(app) as client:
        auth = headers(client)
        session = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        result = client.post(
            f"/api/pet/sessions/{session}/turn", headers=auth, json={"text": "每天都很难过"}
        ).json()
        assert result["attention"]
        assert result["result"]["decision"]["primary_action"] == "DIRECT_SAFETY_CHECK"
        assert len(calls) == 2


def test_invalid_assessment_and_jev_failure_are_visible_not_safe(tmp_path: Path):
    def transport(request: httpx.Request):
        if request.url.host == "127.0.0.1":
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": json.dumps(
                            {
                                "safety_signal": "current",
                                "evidence_quotes": ["虚构的原话"],
                                "support_focus": "不可接受",
                                "needs_clarification": False,
                            }
                        )
                    }
                },
            )
        return httpx.Response(503)

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(jev_enabled=True, cloud_consent=True),
            jev_api_key="test-only",
        )
    )
    with TestClient(app) as client:
        auth = headers(client)
        session = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        response = client.post(
            f"/api/pet/sessions/{session}/turn", headers=auth, json={"text": "你好"}
        )
        assert response.status_code == 503
        data = response.json()
        assert (
            "尚未配置" in data["detail"]
        )
        assert "sk-test" not in json.dumps(data)


def test_cloud_disabled_never_calls_jev(tmp_path: Path):
    def transport(request: httpx.Request):
        raise AssertionError("Cloud transmission requires explicit opt-in")

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(qwen_enabled=False, jev_enabled=True),
            jev_api_key="test-only",
            deepseek_api_key="test-only",
        )
    )
    with TestClient(app) as client:
        auth = headers(client)
        session = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        data = client.post(f"/api/pet/sessions/{session}/turn", headers=auth, json={"text": "你好"})
        assert data.status_code == 503
        assert "尚未配置" in data.json()["detail"]
        assert client.get(f"/api/pet/sessions/{session}/messages").json() == []


@pytest.mark.parametrize(
    "configured,thinking,actual,mode",
    [
        ("deepseek-chat", True, "deepseek-flash", "disabled"),
        ("deepseek-reasoner", False, "deepseek-flash", "enabled"),
        ("deepseek-flash", False, "deepseek-flash", "disabled"),
        ("deepseek-flash", True, "deepseek-flash", "enabled"),
        ("deepseek-v4-pro", False, "deepseek-v4-pro", "disabled"),
    ],
)
def test_deepseek_uses_personality_and_preserves_primary_safety_prompt(
    tmp_path: Path, configured: str, thinking: bool, actual: str, mode: str
):
    bodies = []

    def transport(request: httpx.Request):
        body = json.loads(request.content)
        bodies.append(body)
        assert request.url.host == "api.deepseek.com"
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "test-deepseek-response",
                "model": actual,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "你好呀，今天有什么想聊的？"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220},
            },
        )

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(
                qwen_enabled=False,
                cloud_consent=True,
                personality_prompt="表达自然，称呼我小林。",
                deepseek_model=configured,
                deepseek_thinking=thinking,
            ),
            deepseek_api_key="test-only",
        )
    )
    with TestClient(app) as api:
        auth = headers(api)
        session = api.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        response = api.post(
            f"/api/pet/sessions/{session}/turn", json={"text": "你好"}, headers=auth
        )
        assert response.status_code == 200, response.text
        assert bodies
        assert "不诊断" in bodies[0]["messages"][0]["content"]
        assert "表达自然，称呼我小林" in bodies[0]["messages"][1]["content"]
        assert bodies[0]["model"] == actual
        assert bodies[0]["thinking"] == {"type": mode}
        assert response.json()["mode"] == "deepseek"
        assert response.json()["result"]["decision"]["model_calls"]


def test_deepseek_connection_checks_actual_text_without_private_context(tmp_path: Path):
    calls = []

    def transport(request: httpx.Request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "enabled"}
        assert "PRIVATE-PERSONALITY" not in request.content.decode()
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "你好！"}, "finish_reason": "stop"}],
            },
        )

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(
                cloud_consent=True,
                deepseek_model="deepseek-reasoner",
                personality_prompt="PRIVATE-PERSONALITY",
            ),
            deepseek_api_key="test-secret",
        )
    )
    with TestClient(app) as client:
        result = client.post("/api/pet/test/deepseek", json={}, headers=headers(client))
        assert result.json()["ok"]
        assert result.json()["model"] == "deepseek-flash"
        assert len(calls) == 1
        assert "test-secret" not in result.text


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "密钥"),
        (402, "余额不足"),
        (403, "访问权限"),
        (429, "限流"),
        (400, "请求参数"),
        (404, "模型不可用"),
        (503, "暂时不可用"),
    ],
)
def test_deepseek_errors_are_actionable_without_echoing_provider_body(
    tmp_path: Path, status: int, expected: str
):
    def transport(request: httpx.Request):
        return httpx.Response(
            status, json={"error": {"message": "private-dialogue test-secret PRIVATE-PERSONALITY"}}
        )

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(cloud_consent=True, qwen_enabled=False),
            deepseek_api_key="test-secret",
        )
    )
    with TestClient(app) as client:
        auth = headers(client)
        result = client.post("/api/pet/test/deepseek", json={}, headers=auth)
        assert not result.json()["ok"]
        assert expected in result.json()["detail"]
        session = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        reply = client.post(
            f"/api/pet/sessions/{session}/turn", json={"text": "你好"}, headers=auth
        )
        assert reply.status_code == 200
        assert reply.json()["mode"] == "safety_fallback"
        stage = next(s for s in reply.json()["stages"] if s["name"] == "deepseek")
        assert expected in stage["detail"]
        for private in ("private-dialogue", "test-secret", "PRIVATE-PERSONALITY"):
            assert private not in result.text
            assert private not in reply.text


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("model", "非预期模型"),
        ("empty", "未返回可用文本"),
        ("timeout", "超时"),
        ("network", "无法连接"),
    ],
)
def test_deepseek_probe_rejects_mismatches_empty_reasoning_and_network_failures(
    tmp_path: Path, failure: str, expected: str
):
    def transport(request: httpx.Request):
        if failure == "timeout":
            raise httpx.ReadTimeout("test-secret", request=request)
        if failure == "network":
            raise httpx.ConnectError("test-secret", request=request)
        return httpx.Response(
            200,
            json={
                "model": "unexpected-model" if failure == "model" else "deepseek-flash",
                "choices": [
                    {
                        "message": {"content": "" if failure == "empty" else "你好"},
                        "finish_reason": "length" if failure == "empty" else "stop",
                    }
                ],
            },
        )

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(cloud_consent=True),
            deepseek_api_key="test-secret",
        )
    )
    with TestClient(app) as client:
        response = client.post("/api/pet/test/deepseek", json={}, headers=headers(client))
        assert not response.json()["ok"]
        assert expected in response.json()["detail"]
        assert "test-secret" not in response.text


def test_deepseek_generation_probe_requires_consent(tmp_path: Path):
    def transport(request: httpx.Request):
        raise AssertionError("Must not send a probe without consent")

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(),
            deepseek_api_key="test-secret",
        )
    )
    with TestClient(app) as client:
        assert (
            client.post("/api/pet/test/deepseek", json={}, headers=headers(client)).status_code
            == 400
        )


def test_renderer_history_limit_keeps_the_newest_not_the_oldest_messages():
    from types import SimpleNamespace

    from ecocompanion.application.renderer import _compact_history
    from ecocompanion.domain.enums import ActorRole

    history = [SimpleNamespace(role=ActorRole.USER, content=f"turn-{index}") for index in range(20)]
    assert [message.content for message in _compact_history(history, 4)] == [
        "turn-16",
        "turn-17",
        "turn-18",
        "turn-19",
    ]


def test_active_session_survives_backend_restart(tmp_path: Path):
    app = create_pet_app(tmp_path)
    app.state.pet_store.update(SettingsUpdate(settings=PetSettings(qwen_enabled=False)))
    with TestClient(app) as client:
        auth = headers(client)
        session = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        result = client.post(
            f"/api/pet/sessions/{session}/turn", json={"text": "你好"}, headers=auth
        )
        assert result.status_code == 503
    with TestClient(create_pet_app(tmp_path)) as client:
        assert client.get("/api/pet/bootstrap").json()["current_session_id"] == session
        assert client.get(f"/api/pet/sessions/{session}/messages").json() == []


def test_history_search_pagination_activation_and_restart(tmp_path: Path):
    app = create_pet_app(tmp_path)
    app.state.pet_store.update(SettingsUpdate(
        settings=PetSettings(qwen_enabled=False, cloud_consent=True),
        deepseek_api_key="test-only",
    ))
    with TestClient(app) as client:
        auth = headers(client)
        first = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        client.post(
            f"/api/pet/sessions/{first}/turn",
            json={"text": "旅行预算已经完成 100%"},
            headers=auth,
        ).raise_for_status()
        second = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        client.post(
            f"/api/pet/sessions/{second}/turn",
            json={"text": "我现在正准备自杀，马上就要行动了。"},
            headers=auth,
        ).raise_for_status()
        third = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        assert client.post("/api/pet/sessions/search", json={}).status_code == 403
        listing = client.post("/api/pet/sessions/search", json={"limit": 1}, headers=auth).json()
        assert listing["total"] == 3 and listing["has_more"]
        assert listing["items"][0]["session_id"] == third
        assert listing["items"][0]["title"] == "新对话"
        next_page = client.post(
            "/api/pet/sessions/search", json={"limit": 2, "offset": 1}, headers=auth
        ).json()
        assert not next_page["has_more"]
        assert [item["session_id"] for item in next_page["items"]] == [second, first]
        for query in ("旅行", "%"):
            found = client.post(
                "/api/pet/sessions/search", json={"query": query}, headers=auth
            ).json()
            assert found["total"] == 1
            assert found["items"][0]["session_id"] == first
            assert found["items"][0]["message_count"] == 2
        urgent = client.post(f"/api/pet/sessions/{second}/activate", json={}, headers=auth)
        assert urgent.json()["attention"]
        normal = client.post(f"/api/pet/sessions/{first}/activate", json={}, headers=auth)
        assert not normal.json()["attention"]
        assert len(normal.json()["messages"]) == 2
        assert all(
            message["content"] != "我现在正准备自杀，马上就要行动了。"
            for message in normal.json()["messages"]
        )
        assert (
            client.post("/api/pet/sessions/missing/activate", json={}, headers=auth).status_code
            == 404
        )
        assert client.get("/api/pet/bootstrap").json()["current_session_id"] == first
        assert (
            client.post("/api/pet/sessions/search", json={"limit": 101}, headers=auth).status_code
            == 422
        )
    with TestClient(create_pet_app(tmp_path)) as client:
        assert client.get("/api/pet/bootstrap").json()["current_session_id"] == first
        listing = client.post("/api/pet/sessions/search", json={}, headers=headers(client)).json()
        assert listing["total"] == 3
        assert next(i for i in listing["items"] if i["session_id"] == first)["message_count"] == 2


def test_multiple_conversations_do_not_share_model_context(tmp_path: Path):
    states = []

    def transport(request: httpx.Request):
        body = json.loads(request.content)
        if request.url.path == "/api/chat":
                states.append(json.loads(body["messages"][-1]["content"]))
        else:
            states.append(body["state"])
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.14.0",
                    "answers": {
                        "route": {
                            "type": "choice",
                            "choice": "listen",
                            "confidence": 0.9,
                            "probabilities": {"listen": 0.9, "clarify": 0.1, "safety_check": 0},
                        }
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "safety_signal": "none",
                            "evidence_quotes": [],
                            "support_focus": "日常聊天",
                            "needs_clarification": False,
                        }
                    )
                }
            },
        )

    app = create_pet_app(tmp_path, transport=httpx.MockTransport(transport))
    app.state.pet_store.update(
        SettingsUpdate(
            settings=PetSettings(jev_enabled=True, cloud_consent=True),
            jev_api_key="test-only", deepseek_api_key="test-only",
        )
    )
    with TestClient(app) as client:
        auth = headers(client)
        first = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        second = client.post("/api/pet/sessions", json={}, headers=auth).json()["session_id"]
        for session, text in [
            (first, "我在准备周末去海边旅行"),
            (second, "我在读一本太空小说"),
            (first, "今天继续准备"),
        ]:
            client.post(
                f"/api/pet/sessions/{session}/turn", json={"text": text}, headers=auth
            ).raise_for_status()
        assert states[0]["current_message"] == "我在准备周末去海边旅行"
        assert states[2]["current_message"] == "我在读一本太空小说"
        assert states[1]["recent_conversation"] == []
        assert states[3]["recent_conversation"] == []
        last = json.dumps(states[5]["recent_conversation"], ensure_ascii=False)
        assert "海边旅行" in last
        assert "太空小说" not in last
