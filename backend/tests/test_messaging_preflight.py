import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from ecocompanion.messaging import diagnostics
from ecocompanion.pet.api import create_pet_app


def test_saved_channel_preflight_uses_official_endpoints_without_secret_leaks(
    tmp_path: Path, monkeypatch
):
    requests = []

    def transport(request: httpx.Request):
        requests.append(request)
        if request.url.path == "/app/getAppAccessToken":
            return httpx.Response(200, json={"access_token": "fake-access", "expires_in": 7200})
        if request.url.path == "/gateway":
            return httpx.Response(200, json={"url": "wss://api.sgroup.qq.com/websocket"})
        if request.url.path == "/cgi-bin/token":
            return httpx.Response(200, json={"access_token": "fake-access", "expires_in": 7200})
        return httpx.Response(404)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        diagnostics.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(transport), trust_env=False, timeout=2
        ),
    )
    with TestClient(create_pet_app(tmp_path)) as client:
        token = client.get("/api/pet/bootstrap").json()["token"]
        headers = {"X-Pet-Token": token}
        defaults = client.get("/api/pet/messaging/settings").json()["settings"]
        defaults["qq"]["app_id"] = "test-app-id"
        defaults["wechat"]["app_id"] = "test-app-id"
        saved = client.put(
            "/api/pet/messaging/settings",
            headers=headers,
            json={
                "settings": defaults,
                "qq_client_secret": "qq-secret-not-for-output",
                "wechat_token": "wechat-token-not-for-output",
                "wechat_app_secret": "wechat-secret-not-for-output",
            },
        )
        assert saved.status_code == 200, saved.text
        assert "not-for-output" not in saved.text
        forbidden = client.post("/api/pet/messaging/preflight", json={})
        assert forbidden.status_code == 403
        defaults["wechat"]["enabled"] = True
        saved = client.put(
            "/api/pet/messaging/settings", headers=headers, json={"settings": defaults}
        )
        assert saved.status_code == 200, saved.text
        retired = client.post("/api/pet/messaging/preflight", json={}, headers=headers)
        assert retired.json()["checks"] == []
        assert not retired.json()["ok"]
        assert requests == []
        defaults["qq"]["enabled"] = True
        defaults["qq"]["app_id"] = "test-app-id"
        defaults["wechat"]["enabled"] = True
        defaults["wechat"]["app_id"] = "test-app-id"
        saved = client.put(
            "/api/pet/messaging/settings", headers=headers, json={"settings": defaults}
        )
        assert saved.status_code == 200, saved.text
        result = client.post("/api/pet/messaging/preflight", json={}, headers=headers)
        assert result.status_code == 200
        assert result.json()["ok"]
        assert "wechat" not in {check["channel"] for check in result.json()["checks"]}
        assert "not-for-output" not in result.text
        assert "fake-access" not in result.text
    assert {request.url.host for request in requests} == {"api.bot.qq.com"}
    assert not any(
        "not-for-output" in request.headers.get("authorization", "") for request in requests
    )
    # Credentials belong only in their documented authentication requests.
    assert all(
        request.url.path in {"/app/getAppAccessToken", "/gateway", "/cgi-bin/token"}
        for request in requests
    )
    bodies = [json.loads(request.content) for request in requests if request.content]
    assert not any("fake-access" in json.dumps(body) for body in bodies)
