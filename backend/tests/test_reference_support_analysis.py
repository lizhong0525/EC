"""Hypothetical precision is never presented as actual Judge performance."""

from __future__ import annotations

from typing import Any

import pytest

from scripts import reference_support_analysis as support


def test_wilson_bounds_and_zero_denominator() -> None:
    assert support.wilson(0, 0) is None
    low, high = support.wilson(7, 7) or []
    assert low == pytest.approx(0.6456695649333126)
    assert high == pytest.approx(1.0)
    low0, high0 = support.wilson(0, 7) or []
    assert low0 == pytest.approx(0.0)
    assert high0 == pytest.approx(1 - low)


@pytest.mark.parametrize("args", [(8, 7), (-1, 7), (0, -1), (True, 7), (0, 7, 1)])
def test_invalid_binomial_counts_rejected(args: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        support.wilson(*args)


def test_more_hypothetical_independent_support_tightens_perfect_case_interval() -> None:
    bounds = [support.wilson(n, n) for n in (7, 20, 28, 50, 100)]
    lower = [b[0] for b in bounds if b is not None]
    assert lower == sorted(lower)
    assert all(b is not None and b[1] == pytest.approx(1.0) for b in bounds)
