"""Current official DeepSeek routing, isolated from research model pinning."""

from __future__ import annotations

import httpx

from ecocompanion.pet.settings import PetSettings
from ecocompanion.providers import ChatCompletion, OpenAICompatibleClient
from ecocompanion.providers.errors import ModelIdentityMismatchError, ProviderError

# Legacy names no longer appear in official discovery. Preserve their modes,
# but send the current explicit model ID instead of relying on retired aliases.
LEGACY_MODELS = {"deepseek-chat": "deepseek-flash", "deepseek-reasoner": "deepseek-flash"}


def model_id(settings: PetSettings) -> str:
    return LEGACY_MODELS.get(settings.deepseek_model, settings.deepseek_model)


def create_client(
    settings: PetSettings, key: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> OpenAICompatibleClient:
    thinking = (
        settings.deepseek_model == "deepseek-reasoner"
        if settings.deepseek_model in LEGACY_MODELS
        else settings.deepseek_thinking
    )
    return OpenAICompatibleClient(
        base_url=settings.deepseek_base_url,
        api_key=key,
        connect_timeout_seconds=min(10, settings.timeout_seconds),
        read_timeout_seconds=settings.timeout_seconds,
        max_same_model_retries=0,
        transport=transport,
        default_extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )


def require_text(completion: ChatCompletion) -> None:
    if not (completion.content or "").strip():
        raise RuntimeError("empty DeepSeek text")


def failure_detail(error: Exception) -> str:
    """Never expose provider bodies: they may echo credentials or dialogue."""
    status = error.status_code if isinstance(error, ProviderError) else None
    if status in {401, 403}:
        return f"密钥无效或无访问权限（HTTP {status}），请检查 DeepSeek API Key"
    if status == 402:
        return "DeepSeek 账户余额不足（HTTP 402），请充值后重试"
    if status == 429:
        return "DeepSeek 请求限流（HTTP 429），请稍后重试"
    if status == 404:
        return "DeepSeek 模型不可用（HTTP 404），请重新选择当前模型"
    if status in {400, 422}:
        return f"DeepSeek 拒绝了请求参数（HTTP {status}），请检查模型与生成设置"
    if status == 408 or (status is not None and status >= 500):
        return f"DeepSeek 服务暂时不可用（HTTP {status}），请稍后重试"
    if isinstance(error, ModelIdentityMismatchError):
        return "DeepSeek 返回了非预期模型，已拒绝使用；请更新模型配置"
    if isinstance(error, httpx.TimeoutException):
        return "DeepSeek 请求超时，请检查网络或提高请求超时"
    if isinstance(error, httpx.TransportError) or (
        isinstance(error, ProviderError) and error.retryable
    ):
        return "无法连接 DeepSeek 或请求超时，请检查网络、服务状态与请求超时"
    if isinstance(error, ProviderError):
        return "DeepSeek 响应格式无效，请检查模型设置或稍后重试"
    if isinstance(error, RuntimeError):
        return "DeepSeek 未返回可用文本；请关闭思考模式或提高生成上限后重试"
    return "DeepSeek 响应格式无效，请检查模型设置或稍后重试"
