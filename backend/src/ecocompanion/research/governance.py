"""Grouped independence audits, semantic review queues and blind-set boundaries."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ecocompanion.experiments.io import write_json_atomic
from ecocompanion.research.annotation import digest
from ecocompanion.research.contracts import ResearchCase
from ecocompanion.research.identity import audit_visible_payload, public_input

GROUP_FIELDS = ("person_group", "plot_group", "template_group", "source_group")


def grouped_split(
    cases: Sequence[ResearchCase], *, seed: str, groups: tuple[str, ...] = GROUP_FIELDS
) -> tuple[ResearchCase, ...]:
    """Assign connected components, not individual variants, to partitions."""
    if not groups or not set(groups) <= set(GROUP_FIELDS):
        raise ValueError("invalid independence grouping dimensions")
    parent = {c.case_id: c.case_id for c in cases}
    if len(parent) != len(cases):
        raise ValueError("duplicate case IDs")

    def root(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key

    owners: dict[tuple[str, str], str] = {}
    for case in cases:
        dimensions = [(name, getattr(case.provenance, name)) for name in groups]
        if case.contrast_group:
            dimensions.append(("contrast_group", case.contrast_group))
        for key in dimensions:
            if key in owners:
                parent[root(case.case_id)] = root(owners[key])
            else:
                owners[key] = case.case_id
    components: dict[str, list[ResearchCase]] = defaultdict(list)
    for case in cases:
        components[root(case.case_id)].append(case)
    if len(components) < 3:
        raise ValueError(
            "fewer than three independent components; do not split related cases to meet quotas"
        )
    ordered = sorted(
        components.values(), key=lambda cs: digest([seed, sorted(c.case_id for c in cs)])
    )
    result: list[ResearchCase] = []
    for index, component in enumerate(ordered):
        split = ("development", "validation", "public_test")[index % 3]
        result.extend(c.model_copy(update={"split": split}) for c in component)
    return tuple(sorted(result, key=lambda c: c.case_id))


def group_audit(cases: Sequence[ResearchCase]) -> dict[str, Any]:
    collisions = []
    for dimension in (*GROUP_FIELDS, "contrast_group"):
        owners: dict[str, set[str]] = defaultdict(set)
        for case in cases:
            group = (
                case.contrast_group
                if dimension == "contrast_group"
                else getattr(case.provenance, dimension)
            )
            if group:
                owners[group].add(case.split)
        for group, splits in sorted(owners.items()):
            if len(splits) > 1:
                collisions.append(
                    {"dimension": dimension, "group": group, "splits": sorted(splits)}
                )
    return {
        "passed": not collisions,
        "collisions": collisions,
        "warning": "A passing grouping audit does not establish semantic independence.",
    }


class SemanticSimilarity(Protocol):
    name: str
    version: str

    def __call__(self, left: str, right: str) -> float: ...


def duplicate_review(
    cases: Sequence[ResearchCase],
    *,
    semantic: SemanticSimilarity | None = None,
    threshold: float = 0.85,
) -> dict[str, Any]:
    if not 0 < threshold <= 1:
        raise ValueError("invalid similarity threshold")
    texts = [
        (c.case_id, c.split, "\n".join(o.current_observation.message for o in c.observations))
        for c in cases
    ]
    units: dict[str, list[tuple[str, int, str, set[str]]]] = {}
    for case in cases:
        units[case.case_id] = []
        for observation in case.observations:
            for kind, text in [
                ("utterance", observation.current_observation.message),
                *(("event", event) for event in observation.visible_context.recent_events),
            ]:
                normalized = re.sub(r"\s+", "", text).casefold()
                grams = {normalized[i : i + 3] for i in range(max(1, len(normalized) - 2))}
                units[case.case_id].append((kind, observation.step_id, normalized, grams))
    queue = []
    maximum = 0.0
    for index, (left_id, left_split, left) in enumerate(texts):
        left = re.sub(r"\s+", "", left).casefold()
        left_grams = {left[i : i + 3] for i in range(max(1, len(left) - 2))}
        for right_id, right_split, right in texts[index + 1 :]:
            if left_split == right_split:
                continue
            right = re.sub(r"\s+", "", right).casefold()
            right_grams = {right[i : i + 3] for i in range(max(1, len(right) - 2))}
            union = left_grams | right_grams
            score = len(left_grams & right_grams) / len(union) if union else 1.0
            maximum = max(maximum, score)
            semantic_score = semantic(left, right) if semantic is not None else None
            if semantic_score is not None and not 0 <= semantic_score <= 1:
                raise ValueError("semantic plugin returned invalid similarity")
            unit_matches = []
            for kind, step, text, grams in units[left_id]:
                for other_kind, other_step, other, other_grams in units[right_id]:
                    if kind != other_kind:
                        continue
                    combined = grams | other_grams
                    overlap = len(grams & other_grams) / len(combined) if combined else 1.0
                    maximum = max(maximum, overlap)
                    if text == other or overlap >= threshold:
                        unit_matches.append(
                            {
                                "kind": kind,
                                "left_step": step,
                                "right_step": other_step,
                                "exact": text == other,
                                "character_jaccard": overlap,
                            }
                        )
            if (
                left == right
                or score >= threshold
                or (semantic_score is not None and semantic_score >= threshold)
                or unit_matches
            ):
                queue.append(
                    {
                        "left_case": left_id,
                        "right_case": right_id,
                        "exact": left == right,
                        "character_jaccard": score,
                        "semantic_score": semantic_score,
                        "lexical_unit_matches": unit_matches,
                        "review_status": "PENDING_HUMAN",
                    }
                )
    return {
        "threshold": threshold,
        "maximum_character_jaccard": maximum,
        "semantic_status": "RUN" if semantic is not None else "NOT_RUN",
        "semantic_plugin": {"name": semantic.name, "version": semantic.version}
        if semantic
        else None,
        "queue": queue,
        "claim": "candidate review only; not proof of no leakage",
        "lexical_unit": "whole case, individual utterance and visible event",
        "semantic_unit": "whole case",
    }


def import_blind_cases(
    source: Path,
    private_root: Path,
    *,
    administrator: str,
    independence_reference: str,
    existing: Sequence[ResearchCase] = (),
) -> Path:
    """Stage independent material privately; no automatic generation or publication."""
    if (
        private_root.name != "private"
        or not administrator.strip()
        or not independence_reference.strip()
    ):
        raise ValueError("a private root and independent administrator attestation are required")
    from ecocompanion.experiments.io import read_jsonl_models

    cases = read_jsonl_models(source, ResearchCase)
    if not cases or any(c.split != "independent_blind" for c in cases):
        raise ValueError("only independently authored reviewed blind cases can be staged")
    if len({c.case_id for c in (*existing, *cases)}) != len(existing) + len(cases):
        raise ValueError("duplicate case identity")
    audit = group_audit((*existing, *cases))
    if not audit["passed"]:
        raise ValueError(
            "blind cases share identity/plot/template/source groups with development data"
        )
    for case in cases:
        public_input(case)
    destination = private_root / "independent-blind.json"
    write_json_atomic(
        destination,
        {
            "status": "STAGED_NOT_PUBLISHED",
            "cases": [c.model_dump(mode="json") for c in cases],
            "administrator": administrator,
            "independence_reference": independence_reference,
            "group_audit": audit,
            "semantic_review": duplicate_review((*existing, *cases)),
            "expert_gold_status": "PENDING",
        },
        overwrite=False,
    )
    return destination


def export_development_inputs(cases: Sequence[ResearchCase], destination: Path) -> None:
    """Allowlist public inputs; never recursively copy a curator workspace."""
    if any(c.split == "independent_blind" for c in cases):
        raise ValueError("blind cases cannot enter a public development export")
    payload = {
        "version": "ecotriage-v2.0.0",
        "classification": "PUBLIC_DEVELOPMENT",
        "gold_status": "PENDING",
        "cases": [public_input(c) for c in cases],
        "legacy_generator_reconstructible": True,
    }
    audit_visible_payload(__import__("json").dumps(payload, ensure_ascii=False))
    write_json_atomic(destination, payload, overwrite=False)


def rank_sensitivity(results: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Compare actual measured rankings; do not fabricate missing experiments."""
    if not results:
        return {"status": "NOT_RUN", "rankings": {}, "reversals": []}
    policies = set(next(iter(results.values())))
    if len(policies) < 2 or any(set(scores) != policies for scores in results.values()):
        raise ValueError("sensitivity settings must contain the same two or more policies")
    import math

    if any(not math.isfinite(score) for scores in results.values() for score in scores.values()):
        raise ValueError("nonfinite sensitivity result")
    rankings = {
        setting: sorted(scores, key=lambda p: (-scores[p], p))
        for setting, scores in results.items()
    }
    settings = list(results)
    reversals = []
    for i, left in enumerate(settings):
        for right in settings[i + 1 :]:
            for a_index, a in enumerate(sorted(policies)):
                for b in sorted(policies)[a_index + 1 :]:
                    if (results[left][a] - results[left][b]) * (
                        results[right][a] - results[right][b]
                    ) < 0:
                        reversals.append({"settings": [left, right], "policies": [a, b]})
    return {
        "status": "EXPLORATORY_MEASURED",
        "higher_is_better": True,
        "rankings": rankings,
        "reversals": reversals,
    }
