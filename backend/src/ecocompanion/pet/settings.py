"""Validated local configuration; credentials never appear in API responses."""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DEFAULT_PERSONALITY = (
    "你叫小溟，是一个温和、真诚、有一点轻盈幽默感的 AI 桌面伙伴。"
    "像平等的聊天伙伴一样说话，少用套话和说教，先接住对方的具体处境。"
    "日常回复简短自然，不连续追问；对方想倾诉时认真听，想讨论办法时再一起拆解。"
    "不称用户为主人，不扮演恋人，不卖惨，不要求陪伴或回复。"
    "面对严肃或安全相关话题，收起玩笑，清楚、平静地提供支持。"
)


class PetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pet_name: str = Field(default="小溟", min_length=1, max_length=20)
    personality_prompt: str = Field(default=DEFAULT_PERSONALITY, max_length=6000)
    deepseek_base_url: Literal["https://api.deepseek.com", "https://api.deepseek.com/v1"] = (
        "https://api.deepseek.com"
    )
    deepseek_model: Literal[
        "deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"
    ] = "deepseek-flash"
    deepseek_thinking: bool = False
    temperature: float = Field(default=0.7, ge=0, le=1.5, allow_inf_nan=False)
    max_tokens: int = Field(default=512, ge=128, le=8192)
    context_turns: int = Field(default=6, ge=1, le=10)
    timeout_seconds: int = Field(default=35, ge=5, le=120)
    cloud_consent: bool = False
    qwen_enabled: bool = True
    qwen_base_url: str = "http://127.0.0.1:11434"
    qwen_model: str = Field(default="qwen2.5:7b", min_length=1, max_length=128)
    assessment_timeout_seconds: int = Field(default=25, ge=3, le=90)
    jev_enabled: bool = False
    jev_model: str = Field(default="jev-1.14.0", pattern=r"^jev-\d+\.\d+\.\d+$")
    jev_confidence: float = Field(default=0.8, ge=0.5, le=0.99, allow_inf_nan=False)
    pet_scale: int = Field(default=100, ge=70, le=140)
    animation_enabled: bool = True
    always_on_top: bool = True

    @field_validator("pet_name", "qwen_model")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cannot be blank")
        return value.strip()

    @field_validator("qwen_base_url")
    @classmethod
    def local_ollama_only(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Qwen must use a loopback Ollama HTTP origin")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("invalid Ollama port") from exc
        return value.rstrip("/")


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    settings: PetSettings
    deepseek_api_key: SecretStr | None = Field(default=None, max_length=512)
    jev_api_key: SecretStr | None = Field(default=None, max_length=512)
    clear_deepseek_key: bool = False
    clear_jev_key: bool = False


class _Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, *, decrypt: bool = False) -> bytes:
    buffer = ctypes.create_string_buffer(data)
    incoming = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = _Blob()
    if decrypt:
        ok = ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)
        )
    else:
        ok = ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)
        )
    if not ok:
        raise OSError("Windows credential protection failed")
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        ctypes.windll.kernel32.LocalFree(outgoing.data)


def default_home() -> Path:
    override = os.environ.get("ECO_PET_HOME")
    if override:
        return Path(override).resolve()
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "EcoCompanion" / "pet"


class SettingsStore:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.path = home / "settings.json"
        self.settings = PetSettings()
        self.keys: dict[str, str] = {}
        self.load_warning: str | None = None
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.settings = PetSettings.model_validate(raw["settings"])
                if raw.get("protected_keys") and os.name == "nt":
                    self.keys = json.loads(
                        _dpapi(base64.b64decode(raw["protected_keys"]), decrypt=True)
                    )
            except (ValueError, KeyError, TypeError, OSError):
                self.load_warning = "本地配置读取失败，已使用默认值，请重新检查配置。"

    def view(self) -> dict[str, object]:
        return {
            "settings": self.settings.model_dump(),
            "deepseek_key_configured": bool(self.keys.get("deepseek")),
            "jev_key_configured": bool(self.keys.get("jev")),
            "credential_storage": "windows-dpapi" if os.name == "nt" else "session-only",
            "warning": self.load_warning,
        }

    def update(self, update: SettingsUpdate) -> dict[str, object]:
        keys = dict(self.keys)
        for name in ("deepseek", "jev"):
            secret = getattr(update, f"{name}_api_key")
            if getattr(update, f"clear_{name}_key"):
                keys.pop(name, None)
            elif secret and secret.get_secret_value().strip():
                key = secret.get_secret_value().strip()
                if any(char.isspace() for char in key):
                    raise ValueError("API Key 不能包含空白字符")
                keys[name] = key
        protected = (
            base64.b64encode(_dpapi(json.dumps(keys).encode())).decode()
            if keys and os.name == "nt"
            else None
        )
        self.home.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"settings": update.settings.model_dump(), "protected_keys": protected},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        self.settings, self.keys = update.settings, keys
        self.load_warning = None
        return self.view()
