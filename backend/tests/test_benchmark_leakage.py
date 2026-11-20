from ecocompanion.benchmark.generation import generate_canonical_cases
from ecocompanion.benchmark.leakage import audit_near_duplicate_leakage
from ecocompanion.benchmark.splitting import canonical_split


def test_canonical_text_has_no_high_similarity_cross_split_pairs() -> None:
    report = audit_near_duplicate_leakage(
        canonical_split(generate_canonical_cases()),
    )

    assert report.passed is True
    assert len(report.comparisons) == 12
    assert all(
        comparison.pairs_at_or_above_threshold == 0
        for comparison in report.comparisons
    )
    train_evaluation = tuple(
        comparison
        for comparison in report.comparisons
        if comparison.left_split.value == "train"
        and comparison.kind == "utterance"
    )
    assert len(train_evaluation) == 3
    assert max(
        comparison.maximum_jaccard for comparison in train_evaluation
    ) < 0.5


def test_near_duplicate_audit_rejects_invalid_configuration() -> None:
    splits = canonical_split(generate_canonical_cases())

    for invalid_ngram_size in (0, -1):
        try:
            audit_near_duplicate_leakage(
                splits,
                ngram_size=invalid_ngram_size,
            )
        except ValueError as exc:
            assert "ngram_size" in str(exc)
        else:
            raise AssertionError("invalid ngram size was accepted")

    for invalid_threshold in (0.0, 1.01):
        try:
            audit_near_duplicate_leakage(
                splits,
                threshold=invalid_threshold,
            )
        except ValueError as exc:
            assert "threshold" in str(exc)
        else:
            raise AssertionError("invalid threshold was accepted")
