"""Identity-level confidence intervals for open-set rates (#9).

Probes of one identity are not independent, so every interval treats the identity as the unit.
The primary interval is a percentile bootstrap: resample identities with replacement, keeping all
of each one's probes, 2,000 times with a fixed seed. For an error rate below 1%, where Fogliato
et al. (arXiv:2306.01198) show that bootstrap can under-cover, the Wilson interval with variance
adjusted for that dependence is computed as a check (their recommendation R1, section 4.1).

A rate here is a sum of per-identity numerators over a sum of per-identity denominators: errors
over trials, each identity contributing its own count of both.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

import numpy as np
from numpy.typing import NDArray

CONFIDENCE: Final = 0.95
RESAMPLES: Final = 2000
BOOTSTRAP_SEED: Final = 2000
"""The committed seed of every open-set bootstrap."""
WILSON_BELOW: Final = 0.01
"""An error rate below this also gets the adjusted Wilson interval (#9)."""
DISAGREEMENT: Final = 0.25
"""Two intervals disagree when an end moves by more than this share of the wider one's width."""


@dataclass(frozen=True, slots=True)
class Interval:
    low: float
    high: float


def identity_weights(groups: int, resamples: int, rng: np.random.Generator) -> NDArray[np.int_]:
    """How many times each of `groups` identities is drawn in each resample, resamples x groups.

    Each row sums to `groups`: a resample with replacement is a multinomial draw of counts.
    """
    if groups < 1:
        raise ValueError("need at least one identity to resample")
    return rng.multinomial(groups, np.full(groups, 1 / groups), size=resamples).astype(np.int_)


def percentile_interval(values: NDArray[np.float64], confidence: float = CONFIDENCE) -> Interval:
    """The central `confidence` share of a statistic's bootstrap values."""
    tail = (1 - confidence) / 2
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1 - tail])
    return Interval(float(low), float(high))


def ratio_interval(
    numerators: NDArray[np.int_],
    denominators: NDArray[np.int_],
    weights: NDArray[np.int_],
    confidence: float = CONFIDENCE,
) -> Interval:
    """The percentile bootstrap interval of sum(numerators) / sum(denominators).

    `numerators[g]` and `denominators[g]` are identity g's counts; `weights` comes from
    `identity_weights` over the same identities.
    """
    hits, trials = _counts(numerators, denominators)
    values = (weights @ hits) / (weights @ trials)
    return percentile_interval(values, confidence)


def adjusted_wilson(
    errors: NDArray[np.int_], trials: NDArray[np.int_], confidence: float = CONFIDENCE
) -> Interval:
    """Wilson's interval for sum(errors) / sum(trials) with an effective sample size N*.

    N* = max(p(1 - p) / Var(p), G / 2), Var(p) being the cluster-robust variance over the G
    identities, sum over g of (errors_g - p trials_g)² / N². With one trial per identity that is
    p(1 - p) / N, so N* = N and the interval is the textbook Wilson's. When the variance is 0
    (no errors, or every trial an error) it says nothing about dependence and N* = N.
    """
    wrong, tried = _counts(errors, trials)
    total = float(tried.sum())
    rate = float(wrong.sum()) / total
    variance = float(((wrong - rate * tried) ** 2).sum()) / total**2
    effective = total if variance == 0 else max(rate * (1 - rate) / variance, wrong.size / 2)
    return _wilson(rate, effective, confidence)


def disagree(first: Interval, second: Interval) -> bool:
    """Whether two intervals for one rate differ enough to report both (#9)."""
    width = max(first.high - first.low, second.high - second.low)
    moved = max(abs(first.low - second.low), abs(first.high - second.high))
    return moved > DISAGREEMENT * width


def _wilson(rate: float, n: float, confidence: float) -> Interval:
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    shrink = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / shrink
    half = z * math.sqrt(z * z / (4 * n * n) + rate * (1 - rate) / n) / shrink
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def _counts(
    numerators: NDArray[np.int_], denominators: NDArray[np.int_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    hits = np.asarray(numerators, dtype=np.float64)
    trials = np.asarray(denominators, dtype=np.float64)
    if hits.shape != trials.shape or hits.ndim != 1:
        raise ValueError("expected one numerator and one denominator per identity")
    if (trials < 1).any():
        raise ValueError("every identity needs at least one trial")
    if (hits < 0).any() or (hits > trials).any():
        raise ValueError("an identity's count must lie between 0 and its trials")
    return hits, trials
