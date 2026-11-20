"""Download plumbing fixtures; no runtime, model, or network access."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, ClassVar

import pytest

from scripts import local_runtime_download as downloader

CONTENT = b"ENGINEERING_DOWNLOAD_FIXTURE_NOT_A_MODEL"
FILENAME = "fixture.bin"


class FixtureResponse(io.BytesIO):
    status = 206
    headers: ClassVar[dict[str, str]] = {
        "Content-Range": f"bytes 0-{len(CONTENT) - 1}/{len(CONTENT)}",
    }


@pytest.fixture
def asset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "ASSETS", {
        "fixture": (
            "https://example.invalid/fixture", len(CONTENT),
            hashlib.sha256(CONTENT).hexdigest(), FILENAME,
        ),
    })
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)


def test_range_download_is_verified_before_final_install(
    asset: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def response(request: Any, timeout: int) -> FixtureResponse:
        assert request.get_header("Range") == f"bytes=0-{len(CONTENT) - 1}"
        assert timeout == 120
        return FixtureResponse(CONTENT)

    monkeypatch.setattr(downloader, "urlopen", response)
    downloader.download("fixture", tmp_path, workers=1)
    assert (tmp_path / FILENAME).read_bytes() == CONTENT
    assert not (tmp_path / f"{FILENAME}.assembling").exists()


def test_verified_existing_asset_does_not_download(
    asset: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / FILENAME).write_bytes(CONTENT)
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: pytest.fail("network attempted"))
    downloader.download("fixture", tmp_path, workers=1)


def test_existing_unverified_target_is_never_overwritten(asset: None, tmp_path: Path) -> None:
    target = tmp_path / FILENAME
    target.write_bytes(b"DO_NOT_OVERWRITE")
    with pytest.raises(ValueError, match="existing target"):
        downloader.download("fixture", tmp_path, workers=1)
    assert target.read_bytes() == b"DO_NOT_OVERWRITE"


@pytest.mark.parametrize("failure", ["range", "length", "digest"])
def test_bad_download_is_never_installed(
    asset: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str,
) -> None:
    response = FixtureResponse(
        CONTENT[:-1] if failure == "length"
        else b"X" * len(CONTENT) if failure == "digest" else CONTENT
    )
    if failure == "range":
        response.headers = {"Content-Range": "bytes 1-2/999"}

    def fresh_response(*args: Any, **kwargs: Any) -> FixtureResponse:
        result = FixtureResponse(response.getvalue())
        result.headers = response.headers
        return result

    monkeypatch.setattr(downloader, "urlopen", fresh_response)
    with pytest.raises(ValueError):
        downloader.download("fixture", tmp_path, workers=1)
    assert not (tmp_path / FILENAME).exists()


def test_cached_part_must_still_pass_final_sha256(
    asset: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    parts = tmp_path / f"{FILENAME}.parts"
    parts.mkdir()
    (parts / "00000.part").write_bytes(b"X" * len(CONTENT))
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: pytest.fail("network attempted"))
    with pytest.raises(ValueError, match="SHA256"):
        downloader.download("fixture", tmp_path, workers=1)
    assert not (tmp_path / FILENAME).exists()


@pytest.mark.parametrize("url", [
    "http://release-assets.githubusercontent.com/github-production-release-asset/fixture",
    "https://example.invalid/github-production-release-asset/fixture",
    "https://release-assets.githubusercontent.com/other-path",
    "https://user:password@release-assets.githubusercontent.com/github-production-release-asset/a",
])
def test_unreviewed_cdn_route_rejected(tmp_path: Path, url: str) -> None:
    headers = tmp_path / "fixture.headers"
    headers.write_bytes(f"HTTP/1.1 302 Found\r\nLocation: {url}\r\n\r\n".encode())
    with pytest.raises(ValueError, match="unreviewed"):
        downloader.official_cdn_url(headers)


def test_reviewed_cdn_route_preserves_signed_public_query(tmp_path: Path) -> None:
    url = (
        "https://release-assets.githubusercontent.com/"
        "github-production-release-asset/fixture?signature=PUBLIC_DOWNLOAD_FIXTURE"
    )
    headers = tmp_path / "fixture.headers"
    headers.write_bytes(f"HTTP/1.1 302 Found\r\nLocation: {url}\r\n\r\n".encode())
    assert downloader.official_cdn_url(headers) == url
