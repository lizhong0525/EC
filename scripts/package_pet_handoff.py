"""Build an auditable, developer-data-free desktop pet handoff archive."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODULES = (
    "ecocompanion",
    "ecocompanion.agent",
    "ecocompanion.application",
    "ecocompanion.core",
    "ecocompanion.domain",
    "ecocompanion.infrastructure",
    "ecocompanion.providers",
    "ecocompanion.pet",
    "ecocompanion.messaging",
    "ecocompanion.games",
)
RUNTIME_FILES = (
    "pyproject.toml",
    "requirements.lock",
    "requirements-dev.lock",
    "Start-Games.ps1",
    "进入小游戏.cmd",
    "EcoCompanion.exe",
    "Build-DesktopPetLauncher.ps1",
    "Start-DesktopPet.ps1",
    "Start-MessagingGateway.ps1",
    "Install-WeChatTestBridge.ps1",
    "Start-WeChatTestBridge.ps1",
    "docs/messaging.example.json",
    "Install-LocalQwen.ps1",
)
DOC_FILES = (
    "DESKTOP_PET_PREVIEW.md",
    "MESSAGING_INTEGRATION.md",
    "WEB_SUPPORTIVE_DIALOGUE_V2.md",
    "QWEN_FINETUNING.md",
    "CHANNEL_LIVE_TEST_CHECKLIST.md",
    "ASTRBOT_CHANNEL_COMPATIBILITY.md",
)
TEST_FILES = (
    "test_pet.py",
    "test_application_repository.py",
    "test_web_conversation.py",
    "test_application_service.py",
    "test_messaging.py",
    "test_messaging_runtime.py",
    "test_messaging_preflight.py",
    "test_personal_wechat_bridge.py",
    "test_wellbeing_reports.py",
    "test_games.py",
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|authorization)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}"),
)
BLOCKED_PARTS = {
    "node_modules",
    "__pycache__",
    ".env",
    ".env.local",
    "tmp",
    ".research-venv",
    ".venv",
    "conversations.db",
    "settings.json",
    "traces.jsonl",
    "active-session.json",
    "protected_keys",
    "wellbeing.db",
    "messaging.json",
    "runs",
    "results",
    "models",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "ecocompanion.egg-info",
    "games-runtime",
    "web-runtime",
}


def module_path(root: Path, name: str) -> Path | None:
    package = root / "backend" / "src" / Path(*name.split("."))
    for path in (package / "__init__.py", package.with_suffix(".py")):
        if path.is_file():
            return path
    return None


def runtime_sources(root: Path) -> list[str]:
    source_root = root / "backend" / "src"
    known = {
        path.relative_to(source_root)
        .as_posix()
        .removesuffix("/__init__.py")
        .removesuffix(".py")
        .replace("/", ".")
        for path in source_root.rglob("*.py")
    }
    known.update(
        path.relative_to(source_root).parent.as_posix().replace("/", ".")
        for path in source_root.rglob("__init__.py")
    )
    modules = set(SOURCE_MODULES)
    pending = list(modules)
    while pending:
        module = pending.pop()
        source = module_path(root, module)
        if source is None:
            raise FileNotFoundError(f"Required runtime module is missing: {module}")
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
            for imported in node.names if isinstance(node, ast.Import) else ():
                if imported.name in known and imported.name not in modules:
                    modules.add(imported.name)
                    pending.append(imported.name)
            if (
                isinstance(node, ast.ImportFrom)
                and node.module in known
                and node.module not in modules
            ):
                modules.add(node.module)
                pending.append(node.module)
    return [module_path(root, module).relative_to(root).as_posix() for module in sorted(modules)]


def collect_files(root: Path) -> list[str]:
    files = [*RUNTIME_FILES, *(f"backend/tests/{name}" for name in TEST_FILES)]
    files.extend(f"docs/{name}" for name in DOC_FILES)
    files.extend(
        [
            "scripts/prepare_qwen_training.py",
            "scripts/wechat_test_bridge.py",
            "scripts/qwen_baseline.py",
            "scripts/train_qwen_qlora.py",
            "training/data/reviewed_examples.jsonl",
            "training/data/source_registry.json",
        ]
    )
    files.extend(runtime_sources(root))
    for module in SOURCE_MODULES:
        package = root / "backend/src" / Path(*module.split("."))
        if package.is_dir():
            files.extend(path.relative_to(root).as_posix() for path in package.rglob("*.py"))
    # All explicitly-owned channel modules are shipped together, including
    # transports imported only by the optional standalone gateway entrypoint.
    files.extend(
        path.relative_to(root).as_posix()
        for path in (root / "backend/src/ecocompanion/messaging").glob("*.py")
    )
    web = root / "web"
    for folder in ("src", "desktop", "public", "dist"):
        files.extend(
            path.relative_to(root).as_posix()
            for path in (web / folder).rglob("*")
            if path.is_file()
        )
    for name in (
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.node.json",
        "vite.config.ts",
        "vitest.config.ts",
        "index.html",
    ):
        files.append(f"web/{name}")
    for name in ("SOURCE_HANDOFF.md", "VALIDATION.md", "THIRD_PARTY.md"):
        files.append(f"delivery/{name}")
    for name in (
        "artifacts/pet-preview/qa/conversations.png",
        "artifacts/pet-preview/qa/native-settings.png",
        "artifacts/pet-preview/qa/native-pet.png",
        "artifacts/pet-preview/qa/report-desktop.png",
        "artifacts/pet-preview/qa/report-mobile.png",
        "artifacts/pet-preview/qa/native-report-print.png",
    ):
        if (root / name).is_file():
            files.append(name.replace("artifacts/pet-preview/qa/", "preview/"))
    for relative in sorted(set(files)):
        if relative.startswith("tmp/"):
            raise ValueError(f"Runtime model/training artifacts are forbidden: {relative}")
        path = root / relative.replace("preview/", "artifacts/pet-preview/qa/")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing or symbolic-link handoff file: {relative}")
        if any(
            part.lower() in BLOCKED_PARTS or part.endswith(".egg-info")
            for part in Path(relative).parts
        ):
            raise ValueError(f"Developer data or cache is forbidden: {relative}")
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".gguf", ".pyc"}:
            raise ValueError(f"Private/runtime data is forbidden: {relative}")
    return sorted(set(files))


def preview_source(root: Path, relative: str) -> Path:
    if not relative.startswith("preview/"):
        return root / relative
    name = relative.removeprefix("preview/")
    if name == "native-settings.png":
        updated = root / "tmp/channels-no-official-desktop.png"
        if updated.is_file():
            return updated
    if name.startswith("games-"):
        name = name.replace("games-lobby-mobile.png", "games-lobby-390-clear-text-final.png")
        name = name.replace("games-undercover-desktop.png", "games-undercover-desktop-final.png")
        return root / "tmp" / name
    return root / "artifacts/pet-preview/qa" / name


def collect_submission_files(root: Path) -> list[str]:
    """Collect reviewable source and public fixtures, excluding runtime state."""
    files = [
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink() and path.name != ".coverage"
    ]
    for directory in (
        ".github",
        "backend",
        "delivery",
        "docker",
        "docs",
        "scripts",
        "training/data",
        "artifacts/cases",
        "artifacts/schemas",
        "web/src",
        "web/desktop",
        "web/public",
        "web/dist",
    ):
        files.extend(
            path.relative_to(root).as_posix()
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not any(
                part.lower() in BLOCKED_PARTS or part.endswith(".egg-info")
                for part in path.relative_to(root).parts
            )
        )
    for name in (
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.node.json",
        "vite.config.ts",
        "vitest.config.ts",
        "index.html",
    ):
        files.append(f"web/{name}")
    for name in (
        "conversations.png",
        "native-settings.png",
        "native-pet.png",
        "report-desktop.png",
        "report-mobile.png",
        "native-report-print.png",
    ):
        if (root / "artifacts/pet-preview/qa" / name).is_file():
            files.append(f"preview/{name}")
    for name in (
        "games-lobby-desktop.png",
        "games-undercover-desktop.png",
        "games-lobby-mobile.png",
    ):
        if preview_source(root, f"preview/{name}").is_file():
            files.append(f"preview/{name}")
    for relative in sorted(set(files)):
        path = preview_source(root, relative)
        if relative.startswith("artifacts/") and not relative.startswith(
            ("artifacts/cases/", "artifacts/schemas/")
        ):
            raise ValueError(f"Local runtime or research output is forbidden: {relative}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing or symbolic-link submission file: {relative}")
        if any(part.lower() in BLOCKED_PARTS for part in Path(relative).parts):
            raise ValueError(f"Developer data or cache is forbidden: {relative}")
        if path.suffix.lower() in {
            ".db", ".sqlite", ".sqlite3", ".gguf", ".pyc", ".safetensors", ".bin", ".zip"
        } or path.name.endswith(".tsbuildinfo"):
            raise ValueError(f"Private, generated or model data is forbidden: {relative}")
    return sorted(set(files))


def scan_text_files(staging: Path, files: list[str]) -> None:
    for relative in files:
        path = staging / relative
        if path.suffix.lower() not in {
            ".py",
            ".js",
            ".cjs",
            ".ts",
            ".tsx",
            ".json",
            ".jsonl",
            ".md",
            ".toml",
            ".css",
            ".html",
            ".ps1",
        }:
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                # Test credentials are explicit fake values, not real developer data.
                fake = any(
                    word in line
                    for word in (
                        "test-secret",
                        "test-only",
                        "sk-test",
                        "never-sent",
                        "fake-key",
                    )
                )
                if not fake:
                    raise ValueError(f"Credential-shaped content at {relative}:{index}")


def verify_zip(path: Path, expected: dict[str, str]) -> None:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        for info in archive.infolist():
            relative = info.filename.removeprefix("EcoCompanion/")
            if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                raise ValueError("Unsafe archive path")
            if any(part.lower() in BLOCKED_PARTS for part in Path(relative).parts):
                raise ValueError(f"Forbidden archived content: {relative}")
            if relative in expected:
                digest = hashlib.sha256(archive.read(info)).hexdigest()
                if digest != expected[relative]:
                    raise ValueError(f"Content mismatch: {relative}")


def build(output: Path, submission: bool = False) -> None:
    root = ROOT.resolve()
    output = output.resolve()
    if output.exists() or root == output.parent:
        raise FileExistsError("Use a new ZIP path outside the project directory")
    files = collect_submission_files(root) if submission else collect_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ecopet-handoff-") as temporary:
        staging = Path(temporary) / "EcoCompanion"
        staging.mkdir()
        for relative in files:
            source = (
                preview_source(root, relative)
                if submission
                else root / relative.replace("preview/", "artifacts/pet-preview/qa/")
            )
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        scan_text_files(staging, files)
        hashes = {
            relative: hashlib.sha256((staging / relative).read_bytes()).hexdigest()
            for relative in files
        }
        manifest = {
            "schema": (
                "ecocompanion-source-submission-v1"
                if submission
                else "ecocompanion-desktop-pet-handoff-v1"
            ),
            "created_utc": datetime.now(UTC).isoformat(),
            "scope": "source-and-development-handoff-not-portable-installer",
            "entry": "delivery/PROJECT_SUMMARY.md" if submission else "delivery/SOURCE_HANDOFF.md",
            "included": [
                "pet backend runtime modules",
                "targeted pet/application tests",
                "frontend source",
                "production frontend build",
                "Electron desktop source",
                "local quick launcher",
                "locked Python and npm dependencies",
                "synthetic QA screenshots",
                "architecture and messaging notes",
                "official QQ C2C messaging adapter",
                "isolated personal WeChat test transport and unified channel settings",
                "local wellbeing reports and printing",
                "Qwen baseline/QLoRA tools",
                "reviewed synthetic training fixtures and provenance registry",
            ],
            "excluded": [
                "all credentials",
                "conversation databases",
                "traces",
                "developer settings",
                "virtual environments",
                "node_modules",
                "model weights",
                "research artifacts",
                "full research runtime and test suite",
                "real social accounts or deployments",
                "Qwen run results and trained adapters",
                "remote training outputs and downloaded datasets",
                "temporary screenshots and local browser state",
            ],
            "validation_report": "delivery/VALIDATION.md",
            "files": [{"path": path, "sha256": digest} for path, digest in hashes.items()],
        }
        manifest_path = staging / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging.parent).as_posix())
        verify_zip(output, hashes)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "files": len(files),
                    "bytes": output.stat().st_size,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "secret_scan": "passed",
                    "zip_crc": "passed",
                    "content_hashes": "passed",
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--submission", action="store_true", help="Package the complete reviewable source project"
    )
    args = parser.parse_args()
    build(args.output, submission=args.submission)
    return 0


if __name__ == "__main__":
    sys.exit(main())
