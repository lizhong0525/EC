"""Download the official, immutable Qwen base revision on the HPC login node."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    path = args.output
    path.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/{args.revision}"
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ):
        if not (path / name).exists():
            print(f"Downloading {name}", flush=True)
            urllib.request.urlretrieve(f"{base}/{name}", path / name)
    hashes = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix in {".json", ".safetensors", ".txt"}
    }
    record = {
        "repository": "Qwen/Qwen2.5-7B-Instruct",
        "revision": args.revision,
        "path": str(path),
        "sha256": hashes,
        "scope": "official base weights; not psychological training evidence",
    }
    (args.output / "SOURCE.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"downloaded": str(path), "revision": args.revision}, indent=2))


if __name__ == "__main__":
    main()
