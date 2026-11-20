"""Fail when repository text files contain likely API credentials."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    ".venv",
    ".venv311",
    ".research-venv",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".env",
    ".txt",
}
PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)(api[_-]?key|authorization)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}"),
)


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.name == ".env.example" or path.suffix in TEXT_SUFFIXES:
            files.append(path)
    return files


def scan(root: Path = ROOT) -> list[tuple[Path, int]]:
    findings: list[tuple[Path, int]] = []
    for path in iter_text_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(pattern.search(line) for pattern in PATTERNS):
                findings.append((path.relative_to(root), line_number))
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        print("secret scan: clean")
        return 0
    for path, line_number in findings:
        print(f"possible credential: {path}:{line_number}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
