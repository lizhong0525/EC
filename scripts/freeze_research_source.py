"""Create a local source-only snapshot; never traverse research data or private stores."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def freeze(root: Path, output: Path) -> dict[str, str]:
    root = root.resolve()
    archive = output.with_suffix(".zip")
    manifest_path = output.with_suffix(".manifest.json")
    if archive.exists() or manifest_path.exists():
        raise FileExistsError("source snapshots are immutable; choose a new output stem")
    files = [
        *(root / "backend" / "src" / "ecocompanion").rglob("*.py"),
        *(root / "backend" / "tests").rglob("*.py"),
        *(root / "scripts").glob("*.py"),
        *(root / "scripts").glob("research_ui_*.cjs"),
        *(root / "docs").glob("*.md"),
        root / "backend/src/ecocompanion/research/annotation.html",
        *(
            root / name
            for name in (
                "pyproject.toml",
                "requirements.lock",
                "requirements-dev.lock",
                "README.md",
                "CHANGELOG.md",
                ".gitignore",
                ".dockerignore",
                "EcoCompanion_完整Benchmark与陪伴Agent开发计划.md",
            )
        ),
    ]
    paths = sorted(set(files))
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with ZipFile(archive, "x", ZIP_DEFLATED) as handle:
        for path in paths:
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("source snapshot contains an unsafe or missing source path")
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            handle.writestr(relative, data)
    manifest = {
        "classification": "LOCAL_SOURCE_ONLY_NO_DATA_NO_PRIVATE_RECORDS",
        "files": hashes,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return {"files": str(len(hashes)), "archive_sha256": manifest["archive_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path, help="New local snapshot stem")
    arguments = parser.parse_args()
    print(json.dumps(freeze(Path(__file__).resolve().parents[1], arguments.output)))
