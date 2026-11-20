from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ecocompanion.messaging.bridge import decode_discovery_output
from ecocompanion.pet.api import create_pet_app
from scripts import wechat_contacts


class FakeDatabase:
    wxid = "wxid_test_bot"

    def get_self_info(self):
        return {"nick_name": "测试机器人", "remark": ""}

    def nickname_map(self):
        return {
            "wxid_test_bot": "测试机器人",
            "wxid_test_friend": "测试好友",
            "wxid_test_other": "另一位好友",
            "wxid_test_group@chatroom": "测试群",
        }

    def search_contact(self, query):
        assert query == "测试好友"
        return [{"username": "wxid_test_friend", "nick_name": "原始昵称"}]


def test_discovery_returns_names_and_ids_without_message_content(monkeypatch):
    monkeypatch.setattr(wechat_contacts, "discovery_dependency_error", lambda: None)
    result = wechat_contacts.discover_contacts(
        confirmed=True, database_factory=lambda **kwargs: FakeDatabase()
    )
    assert result == {
        "account": {"wxid": "wxid_test_bot", "name": "测试机器人"},
        "contacts": [
            {"wxid": "wxid_test_other", "name": "另一位好友"},
            {"wxid": "wxid_test_friend", "name": "测试好友"},
        ],
    }
    assert "不得返回" not in json.dumps(result)
    assert wechat_contacts.discover_contacts(
        query="测试好友", confirmed=True, database_factory=lambda **kwargs: FakeDatabase()
    )["contacts"] == [{"wxid": "wxid_test_friend", "name": "测试好友"}]


def test_discovery_does_not_truncate_the_contact_list(monkeypatch):
    class ManyContacts(FakeDatabase):
        def nickname_map(self):
            return {f"wxid_test_{index:03d}": f"好友{index:03d}" for index in range(135)}

    monkeypatch.setattr(wechat_contacts, "discovery_dependency_error", lambda: None)
    result = wechat_contacts.discover_contacts(
        confirmed=True, database_factory=lambda **kwargs: ManyContacts()
    )
    assert len(result["contacts"]) == 135
    assert decode_discovery_output(json.dumps(result, ensure_ascii=False).encode()) == result


def test_discovery_requires_confirmation_and_matching_account(monkeypatch):
    monkeypatch.setattr(wechat_contacts, "discovery_dependency_error", lambda: None)
    with pytest.raises(ValueError, match="确认"):
        wechat_contacts.discover_contacts(
            confirmed=False, database_factory=lambda **kwargs: FakeDatabase()
        )
    with pytest.raises(RuntimeError, match="不一致"):
        wechat_contacts.discover_contacts(
            account_wxid="wxid_different_account",
            confirmed=True,
            database_factory=lambda **kwargs: FakeDatabase(),
        )


def test_subprocess_result_ignores_library_diagnostics_and_rejects_invalid_json():
    result = {
        "account": {"wxid": "wxid_test_bot", "name": "测试机器人"},
        "contacts": [{"wxid": "wxid_test_friend", "name": "测试好友"}],
    }
    output = ("[library diagnostic]\n" + json.dumps(result, ensure_ascii=False)).encode()
    assert decode_discovery_output(output)["contacts"][0]["wxid"] == "wxid_test_friend"
    with pytest.raises(RuntimeError, match="无效"):
        decode_discovery_output(b"diagnostic only")


def test_contact_discovery_api_has_explicit_confirmation_and_no_wechat_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import ecocompanion.pet.api as pet_api

    calls = []

    async def discover(self, account, query, *, confirmed):
        calls.append((account, query, confirmed))
        return {"account": {"wxid": "wxid_test_bot", "name": "测试机器人"}, "contacts": []}

    monkeypatch.setattr(pet_api.PersonalWeChatBridge, "discover", discover)
    with TestClient(create_pet_app(tmp_path)) as client:
        token = client.get("/api/pet/bootstrap").json()["token"]
        denied = client.post("/api/pet/messaging/wechat-contacts", json={"confirmed": True})
        assert denied.status_code == 403
        response = client.post(
            "/api/pet/messaging/wechat-contacts",
            headers={"X-Pet-Token": token},
            json={"confirmed": True, "query": "测试好友"},
        )
        assert response.status_code == 200
        assert calls == [("", "测试好友", True)]
        assert not client.get("/api/pet/messaging/wechat-bridge").json()["running"]
