"""Reference support and uncertainty diagnostics, never a replacement validation gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from ecocompanion.experiments.io import write_json_atomic
from ecocompanion.research.chinese_sources import DIMENSIONS, read_jsonl, sha256
from scripts.external_reference import DEFAULT_ROOT, GATES
from scripts.p0_validation import support_audit

VERSION = "reference-support-diagnostics-v1.0.0"


def wilson(successes: int, trials: int, confidence: float = 0.95) -> list[float] | None:
    if (
        type(successes) is not int or type(trials) is not int
        or not 0 <= successes <= trials or not 0 < confidence < 1
    ):
        raise ValueError("invalid binomial counts/confidence")
    if not trials:
        return None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    scale = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / scale
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def analyze(packet: Path) -> dict[str, Any]:
    plan_hash = sha256((packet / "plan.json").read_bytes())
    implementation_hash = sha256(Path(__file__).read_bytes())
    support = support_audit(packet)
    refs = read_jsonl(packet / "private/references.jsonl")
    dimensions = {}
    for dimension in DIMENSIONS:
        positive_groups = {
            r["input_group"] for r in refs if r["scores"][dimension] == 1
        }
        total = sum(r["scores"][dimension] for r in refs)
        phase_counts = {
            phase: {
                "positive_responses": data["dimensions"][dimension]["reference_positive"],
                "positive_input_groups": len({
                    r["input_group"] for r in refs
                    if r["partition"] == phase and r["scores"][dimension] == 1
                }),
            }
            for phase, data in support["phases"].items()
        }
        required = 2 * GATES["min_positive"]
        dimensions[dimension] = {
            "total_positive_responses": total,
            "total_positive_input_groups_NOT_VERIFIED_PERSONS": len(positive_groups),
            "phases": phase_counts,
            "minimum_total_positive_responses_under_existing_two_phase_gate": required,
            "total_positive_response_shortfall": max(0, required - total),
            "could_any_repartition_satisfy_positive_response_counts": total >= required,
            "repartition_authorized_or_performed": False,
            "caution": "ARITHMETIC_NECESSITY_NOT_POWER_ANALYSIS_OR_PERMISSION_TO_REPARTITION",
        }
    illustrative = [
        {
            "hypothetical_independent_positive_trials": n,
            "hypothetical_correct_predictions": n,
            "wilson_ci95_under_IID_assumption": wilson(n, n),
        }
        for n in (7, 20, 28, 50, 100)
    ]
    report = {
        "version": VERSION,
        "packet_plan_sha256": plan_hash,
        "implementation_sha256": implementation_hash,
        "support": support,
        "dimensions": dimensions,
        "hypothetical_precision_examples_NOT_MODEL_RESULTS": illustrative,
        "uncertainty_scope": (
            "ILLUSTRATIVE_BINOMIAL_IID_ONLY_SOURCE_RESPONSES_ARE_GROUPED;"
            "DO_NOT_REPORT_AS_ACTUAL_MODEL_CLUSTERED_CONFIDENCE_INTERVALS"
        ),
        "protocol_decisions": {
            "existing_five_dimension_gate": "UNCHANGED",
            "automatic_dimension_drop": "FORBIDDEN",
            "automatic_label_generation": "FORBIDDEN",
            "candidate_followup_not_executed": [
                "NEW_SOURCE_ORIGINAL_HUMAN_LABELS_WITH_MATCHED_RUBRIC_RIGHTS_AND_GROUP_AUDIT",
                "NEW_EXPLORATORY_SCOPE_VERSION_NOT_A_PASS_OF_THE_OLD_PROTOCOL",
            ],
            "new_material_can_inherit_old_response_labels": False,
            "data_acquisition_is_not_sample_size_justification": True,
        },
        "new_predictions": 0, "new_expert_labels": 0,
    }
    if (
        support_audit(packet) != support
        or sha256((packet / "plan.json").read_bytes()) != plan_hash
        or sha256(Path(__file__).read_bytes()) != implementation_hash
    ):
        raise ValueError("reference packet or implementation changed during execution")
    report["end_of_run_integrity_verified"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_ROOT / "psycrisis")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.packet)
    write_json_atomic(args.output, report, overwrite=False)
    print(json.dumps({
        "version": VERSION, "new_predictions": 0,
        "risk_inquiry": report["dimensions"]["risk_inquiry"],
    }))


if __name__ == "__main__":
    main()
