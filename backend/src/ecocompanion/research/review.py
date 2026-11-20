"""Offline anonymous reply packets and expert/Judge calibration."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from ecocompanion.domain.schemas import ObservationV1
from ecocompanion.experiments.io import write_json_atomic
from ecocompanion.research.annotation import digest
from ecocompanion.research.contracts import Prediction, Record, ResearchCase


class ReplyRating(Record):
    candidate_id: str
    reviewer_id: str
    kind: Literal["expert", "llm"]
    rubric_version: Literal["reply-v2"] = "reply-v2"
    supportiveness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    severe_violation: bool
    violation_codes: tuple[str, ...] = ()
    verification_reference: str = Field(min_length=1)


def blind_reply_packet(
    cases: Sequence[ResearchCase],
    predictions: Sequence[Prediction],
    output: Path,
    *,
    seed: int,
    views: Mapping[tuple[str, str, int], ObservationV1] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new reply review packet")
    observations = {(c.case_id, o.step_id): o for c in cases for o in c.observations}
    tasks, mapping = [], {}
    seen = set()
    for prediction in predictions:
        key = (prediction.policy, prediction.case_id, prediction.step_id, prediction.repeat)
        if key in seen:
            raise ValueError("duplicate reply candidate")
        seen.add(key)
        if not prediction.decision:
            continue
        candidate_id = uuid4().hex
        observation = (
            views[(prediction.policy, prediction.case_id, prediction.step_id)]
            if views is not None
            else observations[(prediction.case_id, prediction.step_id)]
        )
        tasks.append(
            {
                "candidate_id": candidate_id,
                "rubric_version": "reply-v2",
                "observation": observation.model_dump(mode="json"),
                "reply": prediction.decision.response.model_dump(mode="json"),
            }
        )
        mapping[candidate_id] = {
            "policy": prediction.policy,
            "case_id": prediction.case_id,
            "step_id": prediction.step_id,
            "repeat": prediction.repeat,
            "status": prediction.status,
        }
    if not tasks:
        raise ValueError("no actual model/rule replies to review")
    random.Random(seed).shuffle(tasks)
    output.mkdir(parents=True)
    write_json_atomic(
        output / "private" / "mapping.json", {"seed": seed, "mapping": mapping}, overwrite=False
    )
    public = {
        "rubric_version": "reply-v2",
        "status": "PENDING_EXPERT_REVIEW",
        "tasks": tasks,
        "task_digest": digest(tasks),
    }
    write_json_atomic(output / "reply-tasks.json", public, overwrite=False)
    return {"tasks": len(tasks), "status": "PENDING_EXPERT_REVIEW"}


def calibrate_ratings(
    ratings: Sequence[ReplyRating], *, allowed_candidates: set[str] | None = None
) -> dict[str, Any]:
    import math

    if not ratings:
        return {"status": "NOT_RUN", "paired_candidates": 0, "severe_review_queue": []}
    by_candidate: dict[str, list[ReplyRating]] = {}
    seen = set()
    for rating in ratings:
        if allowed_candidates is not None and rating.candidate_id not in allowed_candidates:
            raise ValueError("rating candidate is absent from the frozen anonymous packet")
        key = rating.candidate_id, rating.reviewer_id, rating.kind
        if key in seen:
            raise ValueError("duplicate review; retain corrections as a new version")
        seen.add(key)
        by_candidate.setdefault(rating.candidate_id, []).append(rating)
    pairs, pending, severe = [], [], []
    for candidate, reviews in by_candidate.items():
        expert = [r for r in reviews if r.kind == "expert"]
        judges = [r for r in reviews if r.kind == "llm"]
        if any(r.severe_violation for r in reviews):
            severe.append(
                {
                    "candidate_id": candidate,
                    "expert_reviewed": bool(expert),
                    "status": "REQUIRES_HUMAN_CONFIRMATION" if not expert else "HUMAN_RATED",
                }
            )
        if len(expert) != 2 or len({r.reviewer_id for r in expert}) != 2 or not judges:
            pending.append(candidate)
            continue
        if any(r.reviewer_id in {e.reviewer_id for e in expert} for r in judges):
            raise ValueError("expert and model Judge identities must be distinct")
        if (
            expert[0].severe_violation != expert[1].severe_violation
            or expert[0].safety != expert[1].safety
            or expert[0].supportiveness != expert[1].supportiveness
        ):
            pending.append(candidate)
            continue
        for judge in judges:
            pairs.append((expert[0], judge))
    metrics = {}
    for dimension in ("supportiveness", "safety"):
        a = [getattr(expert, dimension) for expert, _ in pairs]
        b = [getattr(judge, dimension) for _, judge in pairs]
        kappa = (
            float(cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5]))
            if a and len(set(a + b)) > 1
            else None
        )
        metrics[dimension] = {
            "mean_absolute_error": sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)
            if a
            else None,
            "weighted_kappa": kappa if kappa is not None and math.isfinite(kappa) else None,
        }
    return {
        "status": "CALIBRATION_ONLY_NOT_SAFETY_CERTIFICATION",
        "paired_candidates": len({a.candidate_id for a, _ in pairs}),
        "expert_judge_pairs": len(pairs),
        "metrics": metrics,
        "pending_adjudication_or_missing_reviews": sorted(pending),
        "severe_review_queue": severe,
    }
