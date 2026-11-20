"""Package only reproducible adapter artifacts, never the retained test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {
        "scope": "synthetic engineering adapter; deployment requires held-out comparison",
        "run": json.loads((args.run / "RUN.json").read_text(encoding="utf-8")),
        "validation": json.loads(args.validation.read_text(encoding="utf-8")),
        "files": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for item in sorted((args.run / "adapter").iterdir()):
            if item.is_file():
                archive.add(item, arcname=f"adapter/{item.name}")
                manifest["files"][f"adapter/{item.name}"] = hashlib.sha256(
                    item.read_bytes()
                ).hexdigest()
        for name in ("train.jsonl", "validation.jsonl", "groups.json"):
            item = args.data / name
            archive.add(item, arcname=f"data/{name}")
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(payload)
        import io

        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"package": str(args.output), "adapter_files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
