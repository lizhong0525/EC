"""Separate deployment credentials and limits from the desktop preferences."""

from __future__ import annotations

import base64
import json
import os
import secrets as secret_generator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from ecocompanion.pet.settings import SettingsStore, _dpapi, default_home


def _cloud_status() -> tuple[bool, bool]:
    try:
        desktop = SettingsStore(default_home())
        return bool(desktop.keys.get("deepseek")), desktop.settings.cloud_consent
    except (OSError, ValueError):
        return False, False


class QQSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    app_id: str = ""
    client_secret: str = Field(default="", repr=False)
    intents: int = Field(default=1 << 25, ge=0, le=2**31 - 1)


class WeChatSettings(BaseModel):
    """Official Account XML callback, not a personal WeChat login."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    token: str = Field(default="", repr=False)
    app_id: str = ""
    app_secret: str = Field(default="", repr=False)
    allow_ip_cidr: str = ""


class PersonalWeChatSettings(BaseModel):
    """Opt-in local bridge for a dedicated Windows WeChat test account."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    bridge_token: str = Field(default="", repr=False)
    account_wxid: str = Field(default="", max_length=128)
    allowed_chats: list[str] = Field(default_factory=list, max_length=20)
    allow_groups: bool = False
    test_account_confirmed: bool = False

    @field_validator("allowed_chats")
    @classmethod
    def validate_chat_ids(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if any(len(value) > 128 or any(char.isspace() for char in value) for value in cleaned):
            raise ValueError("测试会话 ID 不能为空、包含空白或超过 128 个字符")
        return cleaned


class MessagingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qq: QQSettings = Field(default_factory=QQSettings)
    wechat: WeChatSettings = Field(default_factory=WeChatSettings)
    personal_wechat: PersonalWeChatSettings = Field(default_factory=PersonalWeChatSettings)
    callback_host: str = "127.0.0.1"
    callback_port: int = Field(default=8765, ge=1024, le=65535)
    per_user_per_minute: int = Field(default=6, ge=1, le=30)
    max_inflight: int = Field(default=8, ge=1, le=32)

    @model_validator(mode="after")
    def callback_stays_local(self) -> MessagingSettings:
        if self.callback_host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("回调仅监听本机，请由受管理的 HTTPS 反向代理对外提供服务")
        return self

    def validate_ready(self) -> None:
        if self.qq.enabled and (not self.qq.app_id.strip() or not self.qq.client_secret.strip()):
            raise ValueError("启用 QQ 需要 AppID 与 ClientSecret")
        if self.wechat.enabled and not self.wechat.token:
            raise ValueError("启用微信公众号需要回调 Token")
        if self.wechat.enabled and (not self.wechat.app_id or not self.wechat.app_secret):
            raise ValueError("异步客服回复需要公众号 AppID 与 AppSecret")
        if self.personal_wechat.enabled:
            personal = self.personal_wechat
            if not personal.test_account_confirmed:
                raise ValueError("个人微信桥接仅限隔离测试账号，请先确认测试边界")
            if len(personal.bridge_token) < 32:
                raise ValueError("个人微信桥接 Token 至少需要 32 个字符")
            if not personal.allowed_chats:
                raise ValueError("个人微信桥接至少需要一个明确的测试会话 ID")
            if not personal.account_wxid.strip() or any(
                character.isspace() for character in personal.account_wxid
            ):
                raise ValueError("个人微信桥接需要锁定测试账号 wxid")
        if bool(self.wechat.app_id) != bool(self.wechat.app_secret):
            raise ValueError("公众号 AppID 与 AppSecret 需要一起配置")


def messaging_home() -> Path:
    override = os.environ.get("ECO_MESSAGING_HOME")
    return Path(override).resolve() if override else default_home() / "messaging"


def load_settings(home: Path) -> MessagingSettings:
    path = home / "messaging.json"
    if not path.exists():
        return MessagingSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protected_keys"):
        if os.name != "nt":
            raise ValueError("Windows-protected messaging credentials require the same user")
        secrets = json.loads(_dpapi(base64.b64decode(data["protected_keys"]), decrypt=True))
        data["settings"]["qq"]["client_secret"] = secrets.get("qq_client_secret", "")
        data["settings"]["wechat"]["token"] = secrets.get("wechat_token", "")
        data["settings"]["wechat"]["app_secret"] = secrets.get("wechat_app_secret", "")
        data["settings"].setdefault("personal_wechat", {})["bridge_token"] = secrets.get(
            "personal_wechat_bridge_token", ""
        )
        data = data["settings"]
    settings = MessagingSettings.model_validate(data)
    settings.validate_ready()
    return settings


def save_settings(home: Path, settings: MessagingSettings) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "messaging.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    if os.name != "nt":
        path.chmod(0o600)
    return path


class MessagingSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    settings: MessagingSettings
    qq_client_secret: SecretStr | None = Field(default=None, max_length=512)
    wechat_token: SecretStr | None = Field(default=None, max_length=512)
    wechat_app_secret: SecretStr | None = Field(default=None, max_length=512)
    clear_qq_client_secret: bool = False
    clear_wechat_token: bool = False
    clear_wechat_app_secret: bool = False
    personal_wechat_bridge_token: SecretStr | None = Field(default=None, max_length=512)
    clear_personal_wechat_bridge_token: bool = False


class MessagingSettingsStore:
    """Return credential presence only; retain writable secrets on the server."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.secrets: dict[str, str] = {}
        self.settings = MessagingSettings()
        if (home / "messaging.json").exists():
            self.settings = load_settings(home)
            raw = json.loads((home / "messaging.json").read_text(encoding="utf-8"))
            if raw.get("protected_keys") and os.name == "nt":
                self.secrets = json.loads(
                    _dpapi(base64.b64decode(raw["protected_keys"]), decrypt=True)
                )
            elif "settings" not in raw:
                self.secrets = {
                    "qq_client_secret": self.settings.qq.client_secret,
                    "wechat_token": self.settings.wechat.token,
                    "wechat_app_secret": self.settings.wechat.app_secret,
                    "personal_wechat_bridge_token": self.settings.personal_wechat.bridge_token,
                }
            else:
                self.secrets.setdefault(
                    "personal_wechat_bridge_token", self.settings.personal_wechat.bridge_token
                )

    def view(self) -> dict[str, object]:
        deepseek_configured, cloud_consent = _cloud_status()
        settings = self.settings.model_dump()
        settings["qq"].pop("client_secret")
        settings["wechat"].pop("token")
        settings["wechat"].pop("app_secret")
        settings["personal_wechat"].pop("bridge_token")
        return {
            "settings": settings,
            "qq_secret_configured": bool(self.secrets.get("qq_client_secret")),
            "wechat_token_configured": bool(self.secrets.get("wechat_token")),
            "wechat_secret_configured": bool(self.secrets.get("wechat_app_secret")),
            "personal_wechat_token_configured": bool(
                self.secrets.get("personal_wechat_bridge_token")
            ),
            "deepseek_key_configured": deepseek_configured,
            "cloud_consent": cloud_consent,
            "credential_storage": "windows-dpapi" if os.name == "nt" else "local-file-0600",
        }

    def update(self, update: MessagingSettingsUpdate) -> dict[str, object]:
        secrets = dict(self.secrets)
        if (
            update.settings.personal_wechat.enabled
            and not secrets.get("personal_wechat_bridge_token")
            and not update.personal_wechat_bridge_token
        ):
            secrets["personal_wechat_bridge_token"] = secret_generator.token_urlsafe(32)
        for name in (
            "qq_client_secret",
            "wechat_token",
            "wechat_app_secret",
            "personal_wechat_bridge_token",
        ):
            secret = getattr(update, name)
            if getattr(update, f"clear_{name}"):
                secrets.pop(name, None)
            elif secret and secret.get_secret_value().strip():
                value = secret.get_secret_value().strip()
                if any(character.isspace() for character in value):
                    raise ValueError("渠道密钥不能包含空白字符")
                secrets[name] = value
        settings = update.settings.model_copy(
            update={
                "qq": update.settings.qq.model_copy(
                    update={"client_secret": secrets.get("qq_client_secret", "")}
                ),
                "wechat": update.settings.wechat.model_copy(
                    update={
                        "token": secrets.get("wechat_token", ""),
                        "app_secret": secrets.get("wechat_app_secret", ""),
                    }
                ),
                "personal_wechat": update.settings.personal_wechat.model_copy(
                    update={
                        "bridge_token": secrets.get("personal_wechat_bridge_token", "")
                    }
                ),
            }
        )
        settings.validate_ready()
        data = settings.model_dump()
        data["qq"]["client_secret"] = ""
        data["wechat"]["token"] = ""
        data["wechat"]["app_secret"] = ""
        data["personal_wechat"]["bridge_token"] = ""
        if os.name == "nt":
            protected = base64.b64encode(_dpapi(json.dumps(secrets).encode())).decode()
            payload = {"settings": data, "protected_keys": protected}
        else:
            payload = data
        self.home.mkdir(parents=True, exist_ok=True)
        path = self.home / "messaging.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)
        self.settings, self.secrets = settings, secrets
        return self.view()
