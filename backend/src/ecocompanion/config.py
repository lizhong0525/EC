"""Immutable application configuration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for process configuration."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        env_prefix="",
        env_ignore_empty=True,
        frozen=True,
        extra="ignore",
    )

    litellm_base_url: str = Field(
        default="https://litellm.dex-gem.ai/v1",
        validation_alias="LITELLM_BASE_URL",
    )
    litellm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="LITELLM_API_KEY",
        repr=False,
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./artifacts/ecocompanion.db",
        validation_alias="ECO_DATABASE_URL",
    )
    artifacts_dir: Path = Field(
        default=Path("artifacts"),
        validation_alias="ECO_ARTIFACTS_DIR",
    )
    default_model: str | None = Field(
        default=None,
        validation_alias="ECO_DEFAULT_MODEL",
    )
    demo_fallback_model: str | None = Field(
        default=None,
        validation_alias="ECO_DEMO_FALLBACK_MODEL",
    )
    api_budget_cny: float = Field(
        default=5000.0,
        gt=0,
        validation_alias="ECO_API_BUDGET_CNY",
    )
    model_input_price_cny_per_million: float | None = Field(
        default=None,
        ge=0,
        validation_alias="ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION",
    )
    model_output_price_cny_per_million: float | None = Field(
        default=None,
        ge=0,
        validation_alias="ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION",
    )
    model_pricing_version: str | None = Field(
        default=None,
        validation_alias="ECO_MODEL_PRICING_VERSION",
    )
    research_mode: bool = Field(
        default=True,
        validation_alias="ECO_RESEARCH_MODE",
    )
    provider_extra_body: dict[str, Any] | None = Field(
        default=None,
        validation_alias="ECO_PROVIDER_EXTRA_BODY",
    )
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0
    max_same_model_retries: int = 3

    @field_validator("litellm_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("LITELLM_BASE_URL must use http:// or https://")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    @field_validator("provider_extra_body", mode="before")
    @classmethod
    def parse_provider_extra_body(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("ECO_PROVIDER_EXTRA_BODY must be valid JSON") from exc
            if not isinstance(parsed, dict) or not parsed:
                raise ValueError("ECO_PROVIDER_EXTRA_BODY must be a non-empty JSON object")
            return parsed
        if isinstance(value, dict):
            return value
        raise ValueError("ECO_PROVIDER_EXTRA_BODY must be a JSON object")

    @model_validator(mode="after")
    def require_complete_model_pricing(self) -> Settings:
        if (
            self.model_input_price_cny_per_million is None
        ) != (
            self.model_output_price_cny_per_million is None
        ):
            raise ValueError(
                "model input and output prices must be configured together"
            )
        if (
            self.model_pricing_version is not None
            and not self.model_pricing_version.strip()
        ):
            raise ValueError("ECO_MODEL_PRICING_VERSION cannot be blank")
        return self

    def require_api_key(self) -> str:
        """Return the API key or fail before making a network request."""

        if self.litellm_api_key is None:
            raise RuntimeError(
                "LITELLM_API_KEY is not configured. Set it in the environment "
                "or a git-ignored .env.local file."
            )
        value = self.litellm_api_key.get_secret_value().strip()
        if not value:
            raise RuntimeError("LITELLM_API_KEY is empty.")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the immutable process settings."""

    return Settings()
