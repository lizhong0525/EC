"""Download two explicitly authorized, pinned official local-runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import parse_headers
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ASSETS = {
    "ollama": (
        "https://github.com/ollama/ollama/releases/download/v0.40.8/ollama-windows-amd64.zip",
        1582537058,
        "04d1d43ac62c2deb0393307bb49fed7caf6b192de41bdd497aaa9488089d5552",
        "ollama-windows-amd64.zip",
    ),
    "qwen-weight": (
        "https://registry.ollama.ai/v2/library/qwen2.5/blobs/"
        "sha256:2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
        4683073952,
        "2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
        "sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
    ),
}


def urlopen(request: Request, timeout: int) -> Any:
    """Public downloads must not inherit proxy credentials or authentication handlers."""
    return build_opener(ProxyHandler({})).open(request, timeout=timeout)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def official_cdn_url(headers: Path) -> str:
    stream = io.BytesIO(headers.read_bytes())
    if b" 302 " not in stream.readline():
        raise ValueError("requires official redirect response")
    parsed = parse_headers(stream)
    target = parsed.get("Location", "")
    url = urlsplit(target)
    if (
        url.scheme != "https" or url.hostname != "release-assets.githubusercontent.com"
        or url.username or url.password or url.port not in {None, 443}
        or not url.path.startswith("/github-production-release-asset/")
    ):
        raise ValueError("unreviewed redirect host/path")
    return target


def download(asset: str, directory: Path, workers: int, cdn_headers: Path | None = None) -> None:
    url, size, expected, filename = ASSETS[asset]
    if cdn_headers is not None:
        if asset != "ollama":
            raise ValueError("CDN route is only for the pinned official runtime archive")
        url = official_cdn_url(cdn_headers)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.exists():
        if target.stat().st_size == size and file_hash(target) == expected:
            print(json.dumps({"status": "EXISTING_VERIFIED", "asset": asset}), flush=True)
            return
        raise ValueError("existing target is not the pinned asset; retain and use a new path")
    parts = directory / f"{filename}.parts"
    parts.mkdir(exist_ok=True)
    chunk = 8 * 1024 * 1024
    ranges = [(i, start, min(start + chunk, size) - 1)
              for i, start in enumerate(range(0, size, chunk))]

    def fetch(item: tuple[int, int, int]) -> int:
        index, start, end = item
        path = parts / f"{index:05d}.part"
        needed = end - start + 1
        if path.exists():
            if path.stat().st_size == needed:
                return needed
            raise ValueError("unexpected completed part length")
        pending = parts / f"{index:05d}.pending"
        for attempt in range(3):
            try:
                separator = "&" if "?" in url else "?"
                request = Request(
                    f"{url}{separator}ecocompanion_range={start}-{end}",
                    headers={"Range": f"bytes={start}-{end}", "Cache-Control": "no-cache"},
                )
                with urlopen(request, timeout=120) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
                    ):
                        raise ValueError("server did not honor exact pinned range")
                    with pending.open("wb") as output:
                        while data := response.read(1024 * 1024):
                            output.write(data)
                if pending.stat().st_size != needed:
                    raise ValueError("incomplete part")
                pending.replace(path)
                return needed
            except Exception as exc:
                print(json.dumps({
                    "asset": asset, "part": index, "download_attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                    "transport": "DIRECT_PUBLIC_HTTPS_NO_ENV_PROXY",
                }), flush=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError("unreachable")

    start_time = time.monotonic()
    completed = 0
    last = 0.0
    print(json.dumps({
        "asset": asset, "status": "START", "source": ASSETS[asset][0],
        "download_host": urlsplit(url).hostname,
        "bytes": size, "sha256": expected, "workers": workers,
        "scope": "PUBLIC_ASSET_DOWNLOAD_NOT_MODEL_INFERENCE",
    }), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        for future in as_completed(futures):
            completed += future.result()
            now = time.monotonic()
            if now - last > 20 or completed == size:
                print(json.dumps({
                    "asset": asset, "verified_range_bytes": completed,
                    "total_bytes": size, "elapsed_seconds": round(now - start_time, 1),
                }), flush=True)
                last = now
    pending_target = directory / f"{filename}.assembling"
    with pending_target.open("xb") as output:
        for index, _, _ in ranges:
            with (parts / f"{index:05d}.part").open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    output.write(data)
    if pending_target.stat().st_size != size or file_hash(pending_target) != expected:
        raise ValueError("final SHA256 mismatch; never execute/load, preserve partial assets")
    pending_target.replace(target)
    print(json.dumps({
        "status": "VERIFIED", "asset": asset, "path": str(target), "sha256": expected,
        "elapsed_seconds": round(time.monotonic() - start_time, 1),
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=ASSETS)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--official-cdn-headers", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        parser.error("workers must be between 1 and 24")
    download(args.asset, args.directory, args.workers, args.official_cdn_headers)


if __name__ == "__main__":
    main()
