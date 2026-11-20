"""Typed failures for OpenAI-compatible provider calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderFailure:
    """Sanitized provider failure details safe to persist in experiment traces."""

    category: str
    message: str
    status_code: int | None = None
    retryable: bool = False


class ProviderError(RuntimeError):
    """Base class for provider errors with an explicit retry contract."""

    category = "provider_error"
    retryable = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    def as_failure(self) -> ProviderFailure:
        return ProviderFailure(
            category=self.category,
            message=str(self),
            status_code=self.status_code,
            retryable=self.retryable,
        )


class AuthenticationError(ProviderError):
    category = "authentication"


class ModelNotFoundError(ProviderError):
    category = "model_not_found"


class InvalidRequestError(ProviderError):
    category = "invalid_request"


class UnsupportedCapabilityError(InvalidRequestError):
    category = "unsupported_capability"


class RetryableProviderError(ProviderError):
    category = "retryable_provider"
    retryable = True


class RateLimitError(RetryableProviderError):
    category = "rate_limit"


class ProtocolError(ProviderError):
    category = "protocol"


class StructuredOutputError(ProviderError):
    category = "structured_output"


class ModelIdentityMismatchError(ProviderError):
    category = "model_identity_mismatch"
