"""Verification metrics over scored pairs: accuracy, LFW's k-fold protocol, ROC and TAR at FAR.

A pair is two face embeddings; its score is their cosine similarity and `same` says whether
they are the same person. Two rules decide a match, deliberately. Accuracy follows the LFW
recipe of OpenCV zoo and InsightFace, "same" iff score > threshold. The ROC and TAR at FAR
follow the live monitor, a match iff score >= threshold.
"""

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray


def _read_only(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values.flags.writeable = False
    return values


LFW_THRESHOLDS: Final[NDArray[np.float64]] = _read_only(1 - np.arange(0, 4, 0.01) / 2)
"""InsightFace's LFW threshold grid, in cosine.

InsightFace searches squared L2 distance between unit embeddings, d² = 2 - 2cos, over
`np.arange(0, 4, 0.01)` with "same" iff d² < t. That is "same" iff cos > 1 - t/2, so this is
1 - t/2 for each t in that grid, in the same order: 400 values descending from 1.0.
"""

# The cross product below which three ROC points count as collinear. Points on a ROC are
# (false accepts / N, true accepts / P), so a real turn has a cross product of at least
# 1/(N P); rounding contributes about 1e-15. This separates the two while N P < 1e14.
_COLLINEAR: Final = 1e-14


@dataclass(frozen=True, slots=True)
class KFoldAccuracy:
    """Accuracy under a k-fold protocol, one entry per fold in fold order.

    `thresholds` holds the threshold each fold was scored at, chosen on the other folds.
    `standard_error` is LFW's: the sample standard deviation (ddof=1) of the fold accuracies
    over the square root of the number of folds.
    """

    fold_accuracies: tuple[float, ...]
    thresholds: tuple[float, ...]
    mean: float
    standard_error: float


@dataclass(frozen=True, slots=True, eq=False)
class Roc:
    """A ROC curve, one point per threshold, from (0, 0) at +inf down to (1, 1).

    The arrays are read-only and the same length; `far` and `tar` never decrease along them.
    Compared by identity, as numpy arrays have no single truth value.
    """

    far: NDArray[np.float64]
    tar: NDArray[np.float64]
    thresholds: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TarAtFar:
    """The ROC point chosen for a FAR budget: its actual FAR, TAR and threshold."""

    target_far: float
    tar: float
    far: float
    threshold: float


def accuracy(scores: NDArray[np.float64], same: NDArray[np.bool_], threshold: float) -> float:
    """The fraction of pairs classified correctly, calling a pair "same" iff score > threshold."""
    scores, same = _pairs(scores, same)
    if math.isnan(threshold):
        raise ValueError("threshold is NaN")
    return float(np.mean((scores > threshold) == same))


def best_threshold(
    scores: NDArray[np.float64],
    same: NDArray[np.bool_],
    thresholds: NDArray[np.float64] = LFW_THRESHOLDS,
) -> float:
    """The threshold in `thresholds` with the highest `accuracy`; on a tie, the first in order."""
    scores, same = _pairs(scores, same)
    thresholds = _grid(thresholds)
    return float(thresholds[np.argmax(_correct(scores, same, thresholds))])


def kfold_accuracy(
    scores: NDArray[np.float64],
    same: NDArray[np.bool_],
    folds: NDArray[np.int_],
    thresholds: NDArray[np.float64] = LFW_THRESHOLDS,
) -> KFoldAccuracy:
    """Accuracy with each fold scored at the `best_threshold` of all the other folds.

    `folds[i]` is pair i's fold. Folds are numbered 0 to k - 1, k >= 2, none of them empty,
    and are taken in ascending order. For LFW View 2 that is 10 folds in file order.
    """
    scores, same = _pairs(scores, same)
    thresholds = _grid(thresholds)
    folds = np.asarray(folds)
    if folds.shape != scores.shape:
        raise ValueError(f"expected one fold per pair, got {folds.shape} for {scores.size} pairs")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError(f"folds must be integers, got {folds.dtype}")
    if folds.min() < 0:
        raise ValueError(f"folds are numbered from 0, got {folds.min()}")
    counts = np.bincount(folds)
    if counts.size < 2:
        raise ValueError("need at least two folds")
    if (empty := np.flatnonzero(counts == 0)).size:
        raise ValueError(f"fold {empty[0]} is empty; folds must be numbered 0 to {counts.size - 1}")

    fold_accuracies: list[float] = []
    fold_thresholds: list[float] = []
    for fold in range(counts.size):
        test = folds == fold
        threshold = best_threshold(scores[~test], same[~test], thresholds)
        fold_thresholds.append(threshold)
        fold_accuracies.append(accuracy(scores[test], same[test], threshold))
    return KFoldAccuracy(
        fold_accuracies=tuple(fold_accuracies),
        thresholds=tuple(fold_thresholds),
        mean=float(np.mean(fold_accuracies)),
        standard_error=float(np.std(fold_accuracies, ddof=1) / math.sqrt(counts.size)),
    )


def roc(scores: NDArray[np.float64], same: NDArray[np.bool_]) -> Roc:
    """The ROC curve, accepting a pair iff score >= threshold.

    It starts at (0, 0) with threshold +inf, then has one point per distinct score, highest
    first, where every pair with that score is accepted at once; it ends at (1, 1). FAR is
    accepted negatives over negatives, TAR accepted positives over positives.
    """
    scores, same = _pairs(scores, same)
    positives = int(np.count_nonzero(same))
    negatives = same.size - positives
    if positives == 0:
        raise ValueError("a ROC needs at least one positive pair")
    if negatives == 0:
        raise ValueError("a ROC needs at least one negative pair")

    order = np.argsort(-scores, kind="stable")
    ranked, accepted = scores[order], same[order]
    # The last pair of each run of equal scores, where the whole run has been accepted.
    run_ends = np.append(np.flatnonzero(np.diff(ranked)), ranked.size - 1)
    true_accepts = np.cumsum(accepted)[run_ends]
    false_accepts = (run_ends + 1) - true_accepts
    return Roc(
        far=_read_only(np.append(0.0, false_accepts / negatives)),
        tar=_read_only(np.append(0.0, true_accepts / positives)),
        thresholds=_read_only(np.append(math.inf, ranked[run_ends])),
    )


def roc_auc(scores: NDArray[np.float64], same: NDArray[np.bool_]) -> float:
    """The trapezoidal area under `roc`.

    That is the Mann-Whitney probability that a random positive outscores a random negative,
    a tie counting half.
    """
    curve = roc(scores, same)
    return float(np.trapezoid(curve.tar, curve.far))


def tar_at_far(scores: NDArray[np.float64], same: NDArray[np.bool_], target_far: float) -> TarAtFar:
    """The `roc` point with the lowest threshold whose FAR <= `target_far`.

    FAR and TAR only grow as the threshold falls, so this is the highest TAR within the FAR
    budget. When only the starting point qualifies, the threshold is +inf and the TAR 0.
    """
    if not 0 <= target_far <= 1:
        raise ValueError(f"target_far must be between 0 and 1, got {target_far}")
    curve = roc(scores, same)
    point = np.flatnonzero(curve.far <= target_far)[-1]
    return TarAtFar(
        target_far=float(target_far),
        tar=float(curve.tar[point]),
        far=float(curve.far[point]),
        threshold=float(curve.thresholds[point]),
    )


def compact(curve: Roc) -> Roc:
    """The same ROC curve without the interior points collinear with their neighbours.

    A run of points along one line, such as a stretch moving only along TAR or only along FAR,
    keeps just its two ends. The first and last points are always kept. Meant for a curve from
    `roc`, whose points are distinct and move monotonically.
    """
    if not curve.far.shape == curve.tar.shape == curve.thresholds.shape or curve.far.ndim != 1:
        raise ValueError(
            "a ROC's far, tar and thresholds must be 1-d and the same length, got "
            f"{curve.far.shape}, {curve.tar.shape} and {curve.thresholds.shape}"
        )
    keep = np.ones(curve.far.size, dtype=np.bool_)
    step_far, step_tar = np.diff(curve.far), np.diff(curve.tar)
    # Point i + 1 is a turn iff the steps into and out of it are not parallel.
    keep[1:-1] = np.abs(step_far[:-1] * step_tar[1:] - step_tar[:-1] * step_far[1:]) > _COLLINEAR
    return Roc(
        far=_read_only(curve.far[keep]),
        tar=_read_only(curve.tar[keep]),
        thresholds=_read_only(curve.thresholds[keep]),
    )


def _pairs(
    scores: NDArray[np.float64], same: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Scores and labels as 1-d float64 and bool arrays, checked to describe the same pairs."""
    scores = np.asarray(scores, dtype=np.float64)
    same = np.asarray(same, dtype=np.bool_)
    if scores.ndim != 1 or same.ndim != 1:
        raise ValueError(f"scores and same must be 1-d, got {scores.shape} and {same.shape}")
    if scores.size != same.size:
        raise ValueError(f"got {scores.size} scores but {same.size} same labels")
    if scores.size == 0:
        raise ValueError("no pairs")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return scores, same


def _grid(thresholds: NDArray[np.float64]) -> NDArray[np.float64]:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError(f"thresholds must be a non-empty 1-d array, got {thresholds.shape}")
    if np.any(np.isnan(thresholds)):
        raise ValueError("thresholds must not be NaN")
    return thresholds


def _correct(
    scores: NDArray[np.float64], same: NDArray[np.bool_], thresholds: NDArray[np.float64]
) -> NDArray[np.int_]:
    """For each threshold, how many pairs `accuracy`'s rule classifies correctly.

    A positive is right iff its score > t, a negative iff its score <= t. Counting both with
    a binary search keeps this O((pairs + thresholds) log pairs) rather than their product.
    """
    positives, negatives = np.sort(scores[same]), np.sort(scores[~same])
    positives_above = positives.size - np.searchsorted(positives, thresholds, side="right")
    negatives_at_or_below = np.searchsorted(negatives, thresholds, side="right")
    return positives_above + negatives_at_or_below
