"""Pinned, public Chinese data snapshots. Never execute upstream evaluators."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4
from zipfile import ZipFile

import httpx
import openpyxl  # type: ignore[import-untyped]

from ecocompanion.experiments.io import read_json_object, write_json_atomic
from ecocompanion.research.annotation import digest

VERSION = "chinese-external-v1.0.0"
PSY_REPO = "mental-health-llm-safety-eval/psycrisis-bench"
PSY_REVISION = "56a862aa9a5209b6836c17c2697ab4f29e644535"
CP_REPO = "CAS-SIAT-XinHai/CPsyCoun"
CP_REVISION = "8fe0fa0f77630b921932bdaa6bae7481bc611cf2"
PSY_FILE = "Data/PsyCrisis-Bench.xlsx"
RATING_FILE = "Data/output_response_human annotation_800.xlsx"
CP_FILE = "CPsyCounD.json"
PSY_COLUMNS = ("label", "user")
RATING_COLUMNS = ("来访者言论", "咨询师回复", "共情", "建议", "担忧", "风险评估", "外部干预")
DIMENSIONS = ("empathy", "regulation", "exploration", "risk_inquiry", "external_support")
PSY_CATEGORIES = {"0": "suicide", "1": "nssi", "2": "existential_distress"}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def blob_sha(content: bytes) -> str:
    # Git's object identity is not a security signature or authorization guarantee.
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_chinese(source: str, output: Path) -> dict[str, Any]:
    """Only allowlisted public bytes; no HF cache, credentials, or remote code."""
    if source not in {"psycrisis", "cpsycoun"}:
        raise ValueError("supported sources: psycrisis, cpsycoun")
    if output.exists():
        raise FileExistsError("choose a new immutable snapshot")
    metadata: dict[str, Any]
    tree: dict[str, Any]
    entries: dict[str, Any]
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        if source == "psycrisis":
            revision, repo = PSY_REVISION, PSY_REPO
            response = client.get(f"https://api.github.com/repos/{repo}/commits/{revision}")
            if response.status_code in {403, 429}:
                # Public API rate limits do not justify reading local credentials.
                metadata = {"sha": revision, "api_status": response.status_code}
                tree = {"tree": [], "status": "API_RATE_LIMIT_NOT_TREE_VERIFIED"}
                entries = {}
                verification = "PINNED_HTTPS_RAW_LOCAL_SHA256_NOT_UPSTREAM_TREE_VERIFIED"
            else:
                response.raise_for_status()
                metadata = response.json()
                if metadata.get("sha") != revision:
                    raise ValueError("upstream commit identity mismatch")
                response = client.get(
                    f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
                )
                if response.status_code in {403, 429}:
                    tree = {"tree": [], "status": "API_RATE_LIMIT_NOT_TREE_VERIFIED"}
                    entries = {}
                    verification = "PINNED_HTTPS_RAW_LOCAL_SHA256_NOT_UPSTREAM_TREE_VERIFIED"
                else:
                    response.raise_for_status()
                    tree = response.json()
                    if tree.get("truncated"):
                        raise ValueError("upstream tree is incomplete")
                    entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
                    verification = "PINNED_UPSTREAM_OBJECT_AND_LOCAL_SHA256"
            paths = ["README.md", "LICENSE", PSY_FILE, RATING_FILE]
            base = f"https://raw.githubusercontent.com/{repo}/{revision}"
            license_info = "CC-BY-NC-4.0_ACADEMIC_RESEARCH_NOTICE"
        else:
            revision, repo = CP_REVISION, CP_REPO
            response = client.get(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}")
            response.raise_for_status()
            metadata = response.json()
            if (
                metadata.get("sha") != revision
                or metadata.get("private")
                or metadata.get("gated")
                or metadata.get("cardData", {}).get("license") != "cc-by-sa-4.0"
            ):
                raise ValueError("upstream identity/access/license requires new review")
            response = client.get(
                f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true"
            )
            response.raise_for_status()
            tree = {"tree": response.json()}
            entries = {e["path"]: e for e in tree["tree"] if e["type"] == "file"}
            verification = "PINNED_UPSTREAM_OBJECT_AND_LOCAL_SHA256"
            paths = ["README.md", CP_FILE]
            base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}"
            license_info = "CC-BY-SA-4.0_HF_DATA_CARD"
        files = {}
        downloaded = {}
        for relative in paths:
            response = client.get(f"{base}/{relative}")
            response.raise_for_status()
            content = response.content
            if entries:
                entry = entries[relative]
                expected = entry.get("sha") or entry.get("oid")
                actual = blob_sha(content)
                if entry.get("lfs"):
                    expected, actual = entry["lfs"]["oid"], sha256(content)
                if expected != actual or len(content) != entry["size"]:
                    raise ValueError(f"upstream object identity/size mismatch: {relative}")
            downloaded[relative] = content
            files[relative] = {"sha256": sha256(content), "bytes": len(content)}
        output.mkdir(parents=True)
        for relative, content in downloaded.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
    write_json_atomic(output / "upstream.json", metadata, overwrite=False)
    write_json_atomic(output / "tree.json", tree, overwrite=False)
    manifest = {
        "version": VERSION,
        "source": source,
        "repository": repo,
        "revision": revision,
        "download_verification": verification,
        "license": license_info,
        "license_conflict": (
            "HF_DATA_CC_BY_SA_VS_GITHUB_CC_BY_REQUIRES_AUTHOR_CLARIFICATION"
            if source == "cpsycoun"
            else None
        ),
        "files": files,
        "upstream_sha256": sha256((output / "upstream.json").read_bytes()),
        "tree_sha256": sha256((output / "tree.json").read_bytes()),
        "use": "LOCAL_ACADEMIC_RESEARCH_ONLY_NO_PUBLICATION_BY_THIS_COMMAND",
        "upstream_source_rights_independently_verified": False,
    }
    write_json_atomic(output / "snapshot.json", manifest, overwrite=False)
    return manifest


def verify_chinese(snapshot: Path) -> dict[str, Any]:
    manifest = read_json_object(snapshot / "snapshot.json")
    source = manifest.get("source")
    expected = (
        {"README.md", "LICENSE", PSY_FILE, RATING_FILE}
        if source == "psycrisis"
        else {"README.md", CP_FILE}
    )
    if source not in {"psycrisis", "cpsycoun"} or set(manifest["files"]) != expected:
        raise ValueError("unsupported snapshot source or file allowlist")
    revision = PSY_REVISION if source == "psycrisis" else CP_REVISION
    if manifest["revision"] != revision:
        raise ValueError("snapshot revision requires new reviewed adapter")
    for relative, identity in manifest["files"].items():
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError("unsafe snapshot path")
        content = (snapshot / relative).read_bytes()
        if sha256(content) != identity["sha256"] or len(content) != identity["bytes"]:
            raise ValueError(f"snapshot file changed: {relative}")
    for filename, key in (("upstream.json", "upstream_sha256"), ("tree.json", "tree_sha256")):
        if sha256((snapshot / filename).read_bytes()) != manifest[key]:
            raise ValueError("snapshot metadata changed")
    metadata = read_json_object(snapshot / "upstream.json")
    tree = read_json_object(snapshot / "tree.json")
    if metadata.get("sha") != revision:
        raise ValueError("pinned upstream identity mismatch")
    entries = {entry["path"]: entry for entry in tree["tree"]}
    if (
        source == "psycrisis"
        and not entries
        and tree.get("status") == "API_RATE_LIMIT_NOT_TREE_VERIFIED"
    ):
        verification = "PINNED_HTTPS_RAW_LOCAL_SHA256_NOT_UPSTREAM_TREE_VERIFIED"
    else:
        for relative in expected:
            entry = entries.get(relative, {})
            content = (snapshot / relative).read_bytes()
            actual = sha256(content) if entry.get("lfs") else blob_sha(content)
            identity = (
                entry["lfs"]["oid"] if entry.get("lfs") else entry.get("sha") or entry.get("oid")
            )
            if identity != actual or entry.get("size") != len(content):
                raise ValueError("snapshot differs from pinned upstream object tree")
        verification = "PINNED_UPSTREAM_OBJECT_AND_LOCAL_SHA256"
    if manifest.get("download_verification", verification) != verification:
        raise ValueError("download verification claim differs from saved metadata")
    return {**manifest, "download_verification": verification}


def workbook_rows(path: Path, columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    """Read values only; reject formulas, extra sheets, links and unexpected schema."""
    with ZipFile(path) as archive:
        if any(
            name.startswith("xl/externalLinks/") or name.endswith("vbaProject.bin")
            for name in archive.namelist()
        ):
            raise ValueError("external workbook links/macros are not dataset records")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False, keep_links=False)
    try:
        if len(workbook.worksheets) != 1:
            raise ValueError("unexpected workbook sheets")
        rows = []
        for cells in workbook.worksheets[0].iter_rows():
            if any(cell.data_type == "f" for cell in cells):
                raise ValueError("formulas are not dataset records")
            rows.append(tuple(cell.value for cell in cells))
        if not rows or rows[0] != columns or any(len(r) != len(columns) for r in rows):
            raise ValueError("workbook columns differ from reviewed schema")
        return rows[1:]
    finally:
        workbook.close()


def normalized(text: str) -> str:
    import unicodedata

    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def load_psy(snapshot: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = []
    for index, (label, user) in enumerate(workbook_rows(snapshot / PSY_FILE, PSY_COLUMNS), 2):
        if not isinstance(user, str) or not user.strip() or label not in PSY_CATEGORIES:
            raise ValueError("invalid prompt category or text")
        prompts.append({"row": index, "category": PSY_CATEGORIES[label], "text": user})
    ratings = []
    for index, row in enumerate(workbook_rows(snapshot / RATING_FILE, RATING_COLUMNS), 2):
        if not all(isinstance(value, str) and value.strip() for value in row[:2]) or any(
            type(value) is not int or value not in {0, 1} for value in row[2:]
        ):
            raise ValueError("invalid response text or binary upstream rating")
        ratings.append(
            {
                "row": index,
                "user": row[0],
                "reply": row[1],
                "scores": dict(zip(DIMENSIONS, row[2:], strict=True)),
            }
        )
    if not prompts or not ratings:
        raise ValueError("empty source workbooks")
    return prompts, ratings


def load_cp(snapshot: Path) -> list[dict[str, Any]]:
    document = json.loads((snapshot / CP_FILE).read_text(encoding="utf-8"))
    if not isinstance(document, list) or not document:
        raise ValueError("expected nonempty CPsyCoun list")
    cases = []
    for index, raw in enumerate(document):
        if set(raw) != {"instruction", "input", "output", "history"}:
            raise ValueError("unexpected CPsyCoun fields")
        if raw["input"] != "" or not isinstance(raw["history"], list):
            raise ValueError("nonempty input/unknown history requires explicit schema review")
        pairs = [*raw["history"], [raw["instruction"], raw["output"]]]
        if any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(text, str) and text.strip() for text in pair)
            for pair in pairs
        ):
            raise ValueError("malformed conversation pair")
        cases.append({"row": index, "pairs": pairs})
    return cases


def audit_chinese(snapshot: Path) -> dict[str, Any]:
    manifest = verify_chinese(snapshot)
    common = {
        "version": VERSION,
        "source": manifest["source"],
        "revision": manifest["revision"],
        "license": manifest["license"],
        "license_conflict": manifest["license_conflict"],
        "download_verification": manifest.get("download_verification", "LEGACY_SNAPSHOT"),
        "expert_action_gold": "UNAVAILABLE",
        "source_person_groups": "NOT_PROVIDED",
        "semantic_deduplication": "NOT_PERFORMED",
        "independent_blind": False,
    }
    if manifest["source"] == "psycrisis":
        prompts, ratings = load_psy(snapshot)
        groups = Counter(normalized(r["user"]) for r in ratings)
        pairs: dict[tuple[str, str], list[dict[str, int]]] = {}
        for r in ratings:
            pairs.setdefault((normalized(r["user"]), normalized(r["reply"])), []).append(
                r["scores"]
            )
        return {
            **common,
            "prompts": len(prompts),
            "exact_unique_prompts": len({r["text"] for r in prompts}),
            "categories": dict(Counter(r["category"] for r in prompts)),
            "normalized_unique_prompts": len({normalized(r["text"]) for r in prompts}),
            "upstream_rated_responses": len(ratings),
            "unique_rated_inputs": len(groups),
            "responses_per_input": dict(Counter(groups.values())),
            "prompt_rating_input_overlap": len(
                {normalized(r["text"]) for r in prompts} & set(groups)
            ),
            "duplicate_input_reply_pairs": sum(len(v) - 1 for v in pairs.values()),
            "conflicting_duplicate_scores": sum(
                len({digest(s) for s in scores}) > 1 for scores in pairs.values()
            ),
            "upstream_positive_counts": {
                dimension: sum(r["scores"][dimension] for r in ratings) for dimension in DIMENSIONS
            },
            "upstream_label_status": "AUTHOR_SUPPLIED_HUMAN_RATINGS_NOT_PROJECT_GOLD",
            "individual_raters_qualifications_disagreements": "NOT_PROVIDED_IN_WORKBOOK",
            "negative_noncrisis_controls": "NOT_PROVIDED",
            "timing_or_longitudinal_labels": "NOT_PROVIDED",
            "upstream_evaluator_execution": "NOT_EXECUTED_NO_REMOTE_JUDGE",
        }
    cases = load_cp(snapshot)
    trajectories = [digest([normalized(p[0]) for p in c["pairs"]]) for c in cases]
    initial = Counter(normalized(c["pairs"][0][0]) for c in cases)
    return {
        **common,
        "dialogues": len(cases),
        "user_nodes": sum(len(c["pairs"]) for c in cases),
        "node_length_distribution": dict(Counter(len(c["pairs"]) for c in cases)),
        "duplicate_user_trajectories": len(cases) - len(set(trajectories)),
        "shared_initial_user_groups": sum(v > 1 for v in initial.values()),
        "topic_or_school_labels": "NOT_PRESENT_IN_RECORDS_DO_NOT_INFER_FROM_README",
        "age_group": "UNSPECIFIED_NOT_ASSUMED_ADOLESCENT",
        "data_kind": "REPORT_RECONSTRUCTED_DIALOGUE_NOT_ORIGINAL_CLINICAL_TRANSCRIPT",
        "source_replies": "NOT_ACTION_GOLD_NOT_CURRENT_NODE_MODEL_INPUT",
    }


def calibration_packet(
    snapshot: Path, output: Path, *, holdout_groups: int = 100
) -> dict[str, Any]:
    """Make anonymous, input-group-disjoint calibration materials; never rate anything."""
    manifest = verify_chinese(snapshot)
    if manifest["source"] != "psycrisis":
        raise ValueError("calibration requires upstream PsyCrisis reply ratings")
    prompts, ratings = load_psy(snapshot)
    groups = sorted(
        {normalized(r["user"]) for r in ratings},
        key=lambda text: sha256(text.encode("utf-8")),
    )
    if not 0 < holdout_groups < len(groups):
        raise ValueError("calibration and holdout both require nonempty input groups")
    if output.exists():
        raise FileExistsError("choose a new calibration packet")
    heldout = set(groups[:holdout_groups])
    group_ids = {text: uuid4().hex for text in groups}
    tasks: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for record in ratings:
        key = normalized(record["user"])
        candidate = uuid4().hex
        tasks.append(
            {
                "candidate_id": candidate,
                "input_group": group_ids[key],
                "user": record["user"],
                "reply": record["reply"],
            }
        )
        references.append(
            {
                "candidate_id": candidate,
                "input_group": group_ids[key],
                "partition": "public_holdout" if key in heldout else "calibration",
                "scores": record["scores"],
                "origin": "UPSTREAM_HUMAN_RATINGS_UNVERIFIED_RATER_LINEAGE",
            }
        )
        mapping.append({"candidate_id": candidate, "source_row": record["row"]})
    import random

    random.Random(20261021).shuffle(tasks)
    write_jsonl(output / "tasks.jsonl", tasks)
    write_jsonl(output / "private" / "references.jsonl", references)
    write_jsonl(output / "private" / "mapping.jsonl", mapping)
    result = {
        "version": VERSION,
        "status": "PENDING_JUDGE_AND_LOCAL_EXPERT_CALIBRATION",
        "source_snapshot_sha256": sha256((snapshot / "snapshot.json").read_bytes()),
        "tasks": len(tasks),
        "input_groups": len(groups),
        "calibration_groups": len(groups) - holdout_groups,
        "public_holdout_groups": holdout_groups,
        "partitions_not_confidential_blind": True,
        "tasks_sha256": sha256((output / "tasks.jsonl").read_bytes()),
        "references_sha256": sha256((output / "private" / "references.jsonl").read_bytes()),
        "dimensions": list(DIMENSIONS),
        "score_semantics": "FIVE_SEPARATE_BINARY_SOURCE_DIMENSIONS_NO_SUMMED_SAFETY_SCORE",
        "subjective_judge_scores": "NOT_GENERATED",
        "action_gold": "NOT_CREATED",
        "cross_dataset_overlap_inputs": len({normalized(p["text"]) for p in prompts} & set(groups)),
        "independence": "NORMALIZED_INPUT_GROUPS_NOT_VERIFIED_PERSON_OR_POST_GROUPS",
        "model_calls": 0,
    }
    write_json_atomic(output / "manifest.json", result, overwrite=False)
    return result
