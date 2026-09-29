"""Identity-level intervals: the percentile bootstrap and the dependence-adjusted Wilson check."""

import numpy as np
import pytest

from ryuk.evaluation.bootstrap import (
    adjusted_wilson,
    disagree,
    identity_weights,
    percentile_interval,
    ratio_interval,
)
from ryuk.evaluation.results import Interval


def test_with_one_trial_per_identity_the_adjusted_wilson_is_the_textbook_wilson() -> None:
    # 1 error in 100 independent trials: Wilson's 95% interval is [0.1767%, 5.4486%].
    errors = np.array([1] + [0] * 99)

    interval = adjusted_wilson(errors, np.ones(100, dtype=np.int_))

    assert interval.low == pytest.approx(0.001767432, abs=1e-8)
    assert interval.high == pytest.approx(0.054486196, abs=1e-8)


def test_errors_clustered_in_one_identity_shrink_the_effective_sample() -> None:
    # 3 errors in 30 trials, all from one of three identities. The cluster variance is
    # ((0 - 1)² + (0 - 1)² + (3 - 1)²) / 30² = 6/900, so N* = 0.1 * 0.9 / (6/900) = 13.5,
    # well above G = 3, and the interval is Wilson's at p = 0.1, n = 13.5.
    interval = adjusted_wilson(np.array([0, 0, 3]), np.array([10, 10, 10]))

    assert interval.low == pytest.approx(0.021910237, abs=1e-8)
    assert interval.high == pytest.approx(0.355304788, abs=1e-8)


def test_the_effective_sample_never_drops_below_the_number_of_identities() -> None:
    # One identity of 40 has 50 trials, all errors; 39 have one correct trial each. p = 50/89,
    # and the cluster variance alone gives N* ≈ 3.96; per-identity rates are floored at G = 40.
    errors = np.array([50] + [0] * 39)
    trials = np.array([50] + [1] * 39)

    interval = adjusted_wilson(errors, trials)

    # Wilson at p = 50/89 and n = 40.
    assert interval.low == pytest.approx(0.409413220, abs=1e-8)
    assert interval.high == pytest.approx(0.703352649, abs=1e-8)


def test_no_errors_takes_the_floor_not_every_trial() -> None:
    # 20 identities of 5 trials, no errors: the variance is 0 and says nothing about dependence.
    # Wilson over all 100 trials would give an upper bound of 3.70%; at n = G = 20 it is 16.11%.
    interval = adjusted_wilson(np.zeros(20, dtype=np.int_), np.full(20, 5))

    assert interval.low == 0.0
    assert interval.high == pytest.approx(0.161125158, abs=1e-8)


def test_every_trial_an_error_takes_the_floor_too() -> None:
    interval = adjusted_wilson(np.full(20, 5), np.full(20, 5))

    assert interval.low == pytest.approx(0.838874842, abs=1e-8)
    assert interval.high == 1.0


def test_the_bootstrap_resamples_whole_identities() -> None:
    rng = np.random.default_rng(0)
    weights = identity_weights(50, 2000, rng)

    assert weights.shape == (2000, 50)
    assert (weights.sum(axis=1) == 50).all()
    # One identity holds all 50 errors in 50 trials; 49 others one correct trial each. A pair-
    # level bootstrap would pin the rate near 50/99; the identity-level one does not.
    errors = np.array([50] + [0] * 49)
    trials = np.array([50] + [1] * 49)
    interval = ratio_interval(errors, trials, weights)
    assert interval.low < 0.1
    assert interval.high > 0.6


def test_identical_identities_give_a_degenerate_interval() -> None:
    weights = identity_weights(10, 500, np.random.default_rng(1))

    interval = ratio_interval(np.full(10, 3), np.full(10, 12), weights)

    assert interval == Interval(low=0.25, high=0.25)


def test_the_percentile_interval_is_the_middle_95_percent() -> None:
    interval = percentile_interval(np.arange(1001, dtype=np.float64) / 1000)

    assert interval.low == pytest.approx(0.025)
    assert interval.high == pytest.approx(0.975)


def test_an_identity_with_no_trials_is_refused() -> None:
    weights = identity_weights(2, 10, np.random.default_rng(0))

    with pytest.raises(ValueError, match="every identity needs at least one trial"):
        ratio_interval(np.array([0, 0]), np.array([3, 0]), weights)
    with pytest.raises(ValueError, match="every identity needs at least one trial"):
        adjusted_wilson(np.array([0, 0]), np.array([3, 0]))


def test_intervals_disagree_when_an_end_moves_by_more_than_a_quarter_of_the_wider_width() -> None:
    assert not disagree(Interval(low=0.01, high=0.03), Interval(low=0.012, high=0.034))
    assert disagree(Interval(low=0.01, high=0.03), Interval(low=0.01, high=0.038))
    assert disagree(Interval(low=0.004, high=0.012), Interval(low=0.0, high=0.012))
