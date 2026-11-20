"""Fetch reviewed public reference bytes, not executable upstream installations."""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research.chinese_sources import (
    PSY_FILE,
    PSY_REPO,
    PSY_REVISION,
    RATING_FILE,
    blob_sha,
    sha256,
)

VERSION = "external-reference-v1.0.0"
VERA_REPO = "SpringCare/VERA-MH"
# First commit publishing the December clinician CSV under its corrected filename.
# Do not substitute a newer rubric or claim exact equivalence to the published study.
VERA_REVISION = "1775e23c520b03831a99a78c3f6d20322df20ba0"
VERA_PREFIX = "publication_data/human_validation/"
VERA_RATINGS = VERA_PREFIX + "clinician_judging/VERA-MH Clinician Ratings_December 2025.csv"
VERA_DIMENSIONS = (
    "clinician_detects",
    "clinician_confirms",
    "clinician_guides",
    "clinician_supportive",
    "clinician_boundaries",
)
VERA_CATEGORIES = (
    "Best Practice",
    "Suboptimal but Low Potential for Harm",
    "High Potential for Harm",
    "Not Relevant",
)
VERA_ID_ALIASES = {"#402e12": "402e12", "#0413e7": "0413e7"}
PSY_PATHS = ("README.md", "LICENSE", PSY_FILE, RATING_FILE, "code/evaluate.py")
VERA_PATHS = (
    "README.md",
    "LICENSE",
    VERA_RATINGS,
    VERA_PREFIX + "README.md",
    "data/rubric.tsv",
    "data/rubric_prompt_beginning.txt",
    "data/question_prompt.txt",
    "judge/constants.py",
    "judge/rubric_config.py",
    "judge/question_navigator.py",
    "judge/llm_judge.py",
)


def source_identity(source: str) -> tuple[str, str, tuple[str, ...]]:
    if source == "psycrisis":
        return PSY_REPO, PSY_REVISION, PSY_PATHS
    if source == "vera":
        return VERA_REPO, VERA_REVISION, VERA_PATHS
    raise ValueError("source must be psycrisis or vera")


def clinician_rows(content: bytes) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    needed = {"conversation_id", "clinician_id", "user_profile", *VERA_DIMENSIONS}
    if not rows or not needed <= set(rows[0]):
        raise ValueError("unreviewed clinician CSV schema")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        original_id = row["conversation_id"]
        row["source_conversation_id"] = original_id
        row["conversation_id"] = VERA_ID_ALIASES.get(original_id, original_id)
        key = row["conversation_id"], row["clinician_id"]
        if (
            not re.fullmatch(r"[0-9a-f]{6}", key[0])
            or not key[1].strip()
            or key in seen
            or any(row[d] not in VERA_CATEGORIES for d in VERA_DIMENSIONS)
        ):
            raise ValueError("invalid/duplicate clinician record")
        seen.add(key)
    return rows


def transcript_id(path: str) -> str:
    return path.rsplit("/", 1)[-1].split("_", 1)[0]


def fetch(source: str, output: Path) -> dict[str, Any]:
    """Only public GETs. All objects verified against a fixed Git tree."""
    if output.exists():
        raise FileExistsError("choose a new snapshot; never overwrite history")
    repo, revision, paths = source_identity(source)
    download_attempts: list[dict[str, Any]] = []
    metadata_attempts: list[dict[str, Any]] = []
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        def metadata_get(url: str) -> httpx.Response:
            response = client.get(url)
            metadata_attempts.append(
                {"url": url, "transport": "DEFAULT_PUBLIC_HTTPS", "status": response.status_code}
            )
            if response.status_code in {403, 429}:
                # A separate public network route may have a different rate-limit bucket.
                # No token/credential lookup and no weaker integrity fallback.
                with httpx.Client(
                    timeout=90, follow_redirects=True, trust_env=False
                ) as direct:
                    response = direct.get(url)
                metadata_attempts.append(
                    {"url": url, "transport": "DIRECT_PUBLIC_HTTPS", "status": response.status_code}
                )
            response.raise_for_status()
            return response

        response = metadata_get(f"https://api.github.com/repos/{repo}/commits/{revision}")
        commit = response.json()
        if commit.get("sha") != revision:
            raise ValueError("upstream commit differs from fixed identity")
        tree_sha = commit["commit"]["tree"]["sha"]
        response = metadata_get(
            f"https://api.github.com/repos/{repo}/git/trees/{tree_sha}?recursive=1"
        )
        tree = response.json()
        if tree.get("truncated") or tree.get("sha") != commit["commit"]["tree"]["sha"]:
            raise ValueError("incomplete or mismatched upstream tree")
        entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
        contents: dict[str, bytes] = {}

        def get(path: str) -> bytes:
            if path not in entries:
                raise ValueError("reviewed source file absent")
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{quote(path, safe='/')}"
            for attempt in range(3):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    download_attempts.append({"path": path, "attempt": attempt + 1, "status": 200})
                    break
                except httpx.TransportError as exc:
                    download_attempts.append(
                        {"path": path, "attempt": attempt + 1, "error_type": type(exc).__name__}
                    )
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            data = response.content
            entry = entries[path]
            if len(data) != entry["size"] or blob_sha(data) != entry["sha"]:
                raise ValueError("upstream Git object mismatch")
            contents[path] = data
            return data

        for path in paths:
            get(path)
        if source == "vera":
            wanted = {r["conversation_id"] for r in clinician_rows(contents[VERA_RATINGS])}
            selected = [
                p
                for p in entries
                if p.startswith(VERA_PREFIX + "conversations/")
                and p.endswith(".txt")
                and transcript_id(p) in wanted
            ]
            if len(selected) != len(wanted) or {transcript_id(p) for p in selected} != wanted:
                raise ValueError("clinician/transcript join is not one-to-one")
            for path in sorted(selected):
                get(path)
    files = {}
    # Upstream paths include Windows-forbidden characters and model/persona names.
    # Store by object identity; original paths remain private source metadata only.
    for path, content in contents.items():
        relative = f"objects/{entries[path]['sha']}.bin"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as handle:
                handle.write(content)
        files[path] = {
            "local": relative,
            "git_blob": entries[path]["sha"],
            "sha256": sha256(content),
            "bytes": len(content),
        }
    write_json_atomic(output / "commit.json", commit, overwrite=False)
    write_json_atomic(output / "tree.json", tree, overwrite=False)
    manifest = {
        "version": VERSION,
        "source": source,
        "repository": repo,
        "revision": revision,
        "created_at": datetime.now(UTC).isoformat(),
        "verification": "FIXED_COMMIT_TREE_GIT_BLOB_AND_LOCAL_SHA256",
        "files": files,
        "commit_sha256": sha256((output / "commit.json").read_bytes()),
        "tree_sha256": sha256((output / "tree.json").read_bytes()),
        "upstream_code_execution": False,
        "remote_model_calls": 0,
        "public_byte_download_attempts": download_attempts,
        "public_metadata_attempts": metadata_attempts,
        "uploads": 0,
        "rights_independently_cleared": False,
        "release": "LOCAL_RESEARCH_ONLY_NO_AUTOMATIC_REDISTRIBUTION",
        "new_project_action_gold": False,
    }
    write_json_atomic(output / "snapshot.json", manifest, overwrite=False)
    return manifest


def verify(snapshot: Path) -> dict[str, Any]:
    manifest = read_json_object(snapshot / "snapshot.json")
    repo, revision, required = source_identity(manifest["source"])
    if (
        manifest.get("version") != VERSION
        or manifest["repository"] != repo
        or manifest["revision"] != revision
        or not set(required) <= set(manifest["files"])
    ):
        raise ValueError("snapshot identity or required inventory changed")
    for name in ("commit", "tree"):
        if sha256((snapshot / f"{name}.json").read_bytes()) != manifest[f"{name}_sha256"]:
            raise ValueError("snapshot metadata changed")
    commit = read_json_object(snapshot / "commit.json")
    tree = read_json_object(snapshot / "tree.json")
    if (
        commit.get("sha") != revision
        or tree.get("truncated")
        or tree.get("sha") != commit["commit"]["tree"]["sha"]
    ):
        raise ValueError("snapshot Git tree/commit mismatch")
    entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    for path, item in manifest["files"].items():
        if (
            path not in entries
            or not re.fullmatch(r"objects/[0-9a-f]{40}\.bin", item["local"])
            or item["local"] != f"objects/{item['git_blob']}.bin"
        ):
            raise ValueError("unexpected source path/object location")
        content = (snapshot / item["local"]).read_bytes()
        if (
            sha256(content) != item["sha256"]
            or len(content) != item["bytes"]
            or blob_sha(content) != item["git_blob"]
            or item["git_blob"] != entries[path]["sha"]
            or item["bytes"] != entries[path]["size"]
        ):
            raise ValueError("reference snapshot object changed")
    allowed = set(required)
    if manifest["source"] == "vera":
        rows = clinician_rows(read_file(snapshot, manifest, VERA_RATINGS))
        wanted = {r["conversation_id"] for r in rows}
        transcripts = [
            p
            for p in manifest["files"]
            if p.startswith(VERA_PREFIX + "conversations/") and p.endswith(".txt")
        ]
        if (
            len(transcripts) != len(wanted)
            or {transcript_id(p) for p in transcripts} != wanted
        ):
            raise ValueError("reference transcript inventory mismatch")
        allowed.update(transcripts)
    if allowed != set(manifest["files"]):
        raise ValueError("files outside reviewed snapshot allowlist")
    return manifest


def read_file(snapshot: Path, manifest: dict[str, Any], path: str) -> bytes:
    item = manifest["files"][path]
    if not re.fullmatch(r"objects/[0-9a-f]{40}\.bin", item["local"]):
        raise ValueError("unsafe local object path")
    content: bytes = (snapshot / str(item["local"])).read_bytes()
    return content


def psy_system_prompt(content: bytes) -> str:
    """Extract a literal from Python syntax; never import/exec upstream code."""
    tree = ast.parse(content.decode("utf-8"))
    prompts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            literals = {
                key.value: value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            role = literals.get("role")
            text = literals.get("content")
            if (
                isinstance(role, ast.Constant)
                and role.value == "system"
                and isinstance(text, ast.Constant)
                and isinstance(text.value, str)
            ):
                prompts.append(text.value)
    if len(prompts) != 1:
        raise ValueError("upstream literal system prompt changed; review adapter")
    return prompts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch")
    download.add_argument("source", choices=("psycrisis", "vera"))
    download.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.source, args.output) if args.command == "fetch" else verify(args.snapshot)
    print(json.dumps({k: result[k] for k in ("source", "revision", "verification")}))


if __name__ == "__main__":
    main()
