from pathlib import Path

import pytest
from pydantic import ValidationError

from ecocompanion.config import Settings


def test_settings_normalize_base_url() -> None:
    settings = Settings(
        LITELLM_BASE_URL="https://example.test/",
        ECO_ARTIFACTS_DIR=Path("artifacts"),
    )
    assert settings.litellm_base_url == "https://example.test/v1"


def test_settings_do_not_require_key_for_offline_construction() -> None:
    settings = Settings(LITELLM_API_KEY=None)
    with pytest.raises(RuntimeError, match="LITELLM_API_KEY"):
        settings.require_api_key()


def test_settings_reject_invalid_url() -> None:
    with pytest.raises(ValidationError):
        Settings(LITELLM_BASE_URL="example.test")


def test_settings_require_input_and_output_prices_together() -> None:
    with pytest.raises(ValidationError, match="configured together"):
        Settings(ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION=2.0)

    settings = Settings(
        ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION=2.0,
        ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION=8.0,
    )
    assert settings.model_input_price_cny_per_million == 2.0
    assert settings.model_output_price_cny_per_million == 8.0


def test_settings_ignore_blank_optional_values_in_env_file(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "LITELLM_API_KEY=\n"
        "ECO_DEFAULT_MODEL=\n"
        "ECO_MODEL_INPUT_PRICE_CNY_PER_MILLION=\n"
        "ECO_MODEL_OUTPUT_PRICE_CNY_PER_MILLION=\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.litellm_api_key is None
    assert settings.default_model is None
    assert settings.model_input_price_cny_per_million is None
    assert settings.model_output_price_cny_per_million is None

def test_settings_parse_provider_extra_body_json() -> None:
    settings = Settings(ECO_PROVIDER_EXTRA_BODY='{"thinking": {"type": "disabled"}}')
    assert settings.provider_extra_body == {"thinking": {"type": "disabled"}}


def test_settings_reject_invalid_provider_extra_body() -> None:
    with pytest.raises(ValidationError, match="ECO_PROVIDER_EXTRA_BODY"):
        Settings(ECO_PROVIDER_EXTRA_BODY="not-json")
