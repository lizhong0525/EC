"""Stable prompt identities for immutable experiment manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptIdentity:
    """Version and content digest for one provider-facing prompt contract."""

    component: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        if not self.component.strip():
            raise ValueError("prompt component cannot be blank")
        if not self.version.strip():
            raise ValueError("prompt version cannot be blank")
        if len(self.digest) != 64:
            raise ValueError("prompt digest must be a SHA-256 hex digest")
        try:
            int(self.digest, 16)
        except ValueError as exc:
            raise ValueError("prompt digest must be hexadecimal") from exc

    def to_dict(self) -> dict[str, str]:
        return {
            "component": self.component,
            "version": self.version,
            "digest": self.digest,
        }


def build_prompt_identity(
    *,
    component: str,
    version: str,
    content: object,
) -> PromptIdentity:
    """Hash the complete static prompt material under an explicit version."""

    if not component.strip():
        raise ValueError("prompt component cannot be blank")
    if not version.strip():
        raise ValueError("prompt version cannot be blank")
    serialized = json.dumps(
        {
            "component": component,
            "version": version,
            "content": content,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return PromptIdentity(
        component=component,
        version=version,
        digest=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )
