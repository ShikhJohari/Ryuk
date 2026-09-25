import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ryuk.evaluation.metrics import (
    LFW_THRESHOLDS,
    Roc,
    accuracy,
    best_threshold,
    compact,
    kfold_accuracy,
    roc,
    roc_auc,
    tar_at_far,
)


def _scores(*values: float) -> NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _same(*values: bool) -> NDArray[np.bool_]:
    return np.array(values, dtype=np.bool_)


T, F = True, False

# Six pairs, three of each, with a positive and a negative tied at 0.7.
# Accept iff score >= threshold; P = N = 3.
#   threshold  +inf: nothing accepted              -> (FAR, TAR) = (0,   0)
#   threshold  0.9:  the 0.9 positive              -> (0,   1/3)
#   threshold  0.8:  + the 0.8 negative            -> (1/3, 1/3)
#   threshold  0.7:  + both 0.7 pairs, together    -> (2/3, 2/3)
#   threshold  0.5:  + the 0.5 positive            -> (2/3, 1)
#   threshold  0.3:  + the 0.3 negative            -> (1,   1)
SIX_SCORES = _scores(0.9, 0.8, 0.7, 0.7, 0.5, 0.3)
SIX_SAME = _same(T, F, T, F, T, F)


def test_lfw_thresholds_are_insightfaces_distance_grid_in_cosine() -> None:
    # d² = 2 - 2cos, so InsightFace's d² grid 0, 0.01, ..., 3.99 is cos 1, 0.995, ..., -0.995.
    assert LFW_THRESHOLDS.shape == (400,)
    assert LFW_THRESHOLDS[0] == 1.0
    assert LFW_THRESHOLDS[-1] == pytest.approx(1 - 3.99 / 2)
    assert np.all(np.diff(LFW_THRESHOLDS) < 0)


def test_accuracy_counts_a_pair_as_same_only_when_its_score_is_strictly_above() -> None:
    # A score exactly at the threshold is "different": right for a negative, wrong for a positive.
    assert accuracy(_scores(0.5), _same(T), 0.5) == 0.0
    assert accuracy(_scores(0.5), _same(F), 0.5) == 1.0


def test_accuracy_is_the_fraction_of_pairs_classified_correctly() -> None:
    # At 0.6: 0.9 T right, 0.8 F wrong, 0.7 T right, 0.7 F wrong, 0.5 T wrong, 0.3 F right -> 3/6.
    assert accuracy(SIX_SCORES, SIX_SAME, 0.6) == 0.5
    # At 0.75: 0.9 T right, 0.8 F wrong, 0.7 T wrong, 0.7 F right, 0.5 T wrong, 0.3 F right -> 3/6.
    assert accuracy(SIX_SCORES, SIX_SAME, 0.75) == 0.5
    # At 0.85: only 0.9 is "same": 0.9 T right, the negatives right, 0.7 T and 0.5 T wrong -> 4/6.
    assert accuracy(SIX_SCORES, SIX_SAME, 0.85) == pytest.approx(4 / 6)


def test_best_threshold_picks_the_most_accurate_threshold() -> None:
    # Grid 0.95, 0.85, 0.6: accuracies 3/6, 4/6, 3/6 (see the accuracy test, 0.95 calls all
    # "different" so only the three negatives are right).
    grid = _scores(0.95, 0.85, 0.6)

    assert best_threshold(SIX_SCORES, SIX_SAME, grid) == 0.85


def test_best_threshold_calls_a_score_on_the_threshold_different() -> None:
    # A positive at 0.5 is wrong at 0.5 (1/2) but right at 0.3, where 0.2 F is right too (1).
    assert best_threshold(_scores(0.5, 0.2), _same(T, F), _scores(0.5, 0.3)) == 0.3
    # A negative at 0.5 is right at 0.5, where 0.9 T is right too (1): the first perfect one.
    assert best_threshold(_scores(0.9, 0.5), _same(T, F), _scores(0.5, 0.7)) == 0.5


def test_the_first_threshold_in_grid_order_wins_a_tie() -> None:
    # Every threshold in [0.1, 0.9) separates these perfectly, so all three tie at 100%.
    scores, same = _scores(0.9, 0.1), _same(T, F)

    assert best_threshold(scores, same, _scores(0.8, 0.5, 0.2)) == 0.8
    assert best_threshold(scores, same, _scores(0.2, 0.5, 0.8)) == 0.2


def test_best_threshold_searches_the_lfw_grid_by_default() -> None:
    # The grid is 1 - k/200. A positive at 0.8975 is "same" from k = 21 (0.895) on, and the
    # negative at 0.3 stays "different" until well after, so k = 21 is the first perfect one.
    assert best_threshold(_scores(0.8975, 0.3), _same(T, F)) == pytest.approx(0.895)


def test_kfold_scores_each_fold_at_the_threshold_chosen_on_the_other_folds() -> None:
    # Grid 0.75, 0.5, 0.25. A positive at s is right at t iff s > t; a negative iff s <= t.
    # Right answers per threshold, per fold (folds interleaved in the arrays on purpose):
    #   fold 0: 0.9 T [1,1,1] + 0.6 F [1,0,0] + 0.4 T [0,0,1] = [2,1,2]  (3 pairs)
    #   fold 1: 0.4 T [0,0,1] + 0.1 F [1,1,1]                 = [1,1,2]  (2 pairs)
    #   fold 2: 0.6 T [0,1,1] + 0.3 F [1,1,0]                 = [1,2,1]  (2 pairs)
    # Fold 0 trains on 1+2 = [2,3,3]: 0.5 and 0.25 tie, 0.5 first; fold 0 at 0.5: 1/3.
    # Fold 1 trains on 0+2 = [3,3,3]: all tie, 0.75 first;         fold 1 at 0.75: 1/2.
    # Fold 2 trains on 0+1 = [3,2,4]: 0.25;                         fold 2 at 0.25: 1/2.
    scores = _scores(0.9, 0.4, 0.6, 0.6, 0.1, 0.3, 0.4)
    same = _same(T, T, T, F, F, F, T)
    folds = np.array([0, 1, 2, 0, 1, 2, 0])

    result = kfold_accuracy(scores, same, folds, thresholds=_scores(0.75, 0.5, 0.25))

    assert result.thresholds == (0.5, 0.75, 0.25)
    assert result.fold_accuracies == pytest.approx((1 / 3, 1 / 2, 1 / 2))
    # The mean of the fold accuracies, not of all pairs (which would be 3/7).
    assert result.mean == pytest.approx(4 / 9)
    # Deviations from 4/9: -1/9, 1/18, 1/18; squares sum to 4/324 + 1/324 + 1/324 = 1/54.
    # Sample variance 1/54 / 2 = 1/108, std 1/sqrt(108) = 1/(6 sqrt 3); over sqrt 3: 1/18.
    assert result.standard_error == pytest.approx(1 / 18)


def test_kfold_uses_the_lfw_grid_by_default() -> None:
    # The grid is 1 - k/200, descending. Each fold is separable by any threshold between its
    # scores, so training picks the first grid value below the other fold's positive:
    #   fold 0 trains on fold 1 (0.9475 T, 0.3 F): k = 11, 0.945; fold 0 at 0.945 gets
    #     0.8975 T wrong and 0.2 F right -> 1/2.
    #   fold 1 trains on fold 0 (0.8975 T, 0.2 F): k = 21, 0.895; fold 1 at 0.895 -> 1.
    # Mean 3/4; SE for two folds is |a - b| / 2 = 1/4.
    scores = _scores(0.8975, 0.2, 0.9475, 0.3)
    same = _same(T, F, T, F)

    result = kfold_accuracy(scores, same, np.array([0, 0, 1, 1]))

    assert result.thresholds == pytest.approx((0.945, 0.895))
    assert result.fold_accuracies == (0.5, 1.0)
    assert result.mean == 0.75
    assert result.standard_error == pytest.approx(0.25)


def test_roc_steps_through_each_distinct_score_from_the_top() -> None:
    curve = roc(SIX_SCORES, SIX_SAME)

    np.testing.assert_allclose(curve.far, [0, 0, 1 / 3, 2 / 3, 2 / 3, 1])
    np.testing.assert_allclose(curve.tar, [0, 1 / 3, 1 / 3, 2 / 3, 1, 1])
    np.testing.assert_array_equal(curve.thresholds, [math.inf, 0.9, 0.8, 0.7, 0.5, 0.3])


def test_roc_does_not_depend_on_pair_order() -> None:
    order = np.array([4, 1, 5, 3, 0, 2])

    shuffled = roc(SIX_SCORES[order], SIX_SAME[order])

    np.testing.assert_allclose(shuffled.far, [0, 0, 1 / 3, 2 / 3, 2 / 3, 1])
    np.testing.assert_allclose(shuffled.tar, [0, 1 / 3, 1 / 3, 2 / 3, 1, 1])


def test_roc_auc_is_the_trapezoidal_area_with_ties_counted_half() -> None:
    # Trapezoids between the six ROC points: 0 + 1/3 * 1/3 + 1/3 * (1/3 + 2/3)/2 + 0 + 1/3 * 1
    #   = 1/9 + 1/6 + 1/3 = 11/18.
    # Mann-Whitney over the 3 x 3 (positive, negative) pairs, a tie counting half:
    #   0.9 beats 0.8, 0.7, 0.3 -> 3;  0.7 beats 0.3, ties 0.7 -> 1.5;  0.5 beats 0.3 -> 1.
    #   5.5 / 9 = 11/18.
    assert roc_auc(SIX_SCORES, SIX_SAME) == pytest.approx(11 / 18)


def test_roc_auc_is_one_for_a_perfect_ranking_and_zero_for_a_reversed_one() -> None:
    assert roc_auc(_scores(0.9, 0.8, 0.2), _same(T, T, F)) == 1.0
    assert roc_auc(_scores(0.9, 0.8, 0.2), _same(F, F, T)) == 0.0


def test_roc_auc_equals_the_mann_whitney_probability_on_random_data() -> None:
    rng = np.random.default_rng(26)
    # Rounded to one decimal so that many scores tie across the classes.
    scores = np.round(rng.uniform(-1, 1, 300), 1)
    same = rng.random(300) < 0.4
    positives, negatives = scores[same][:, None], scores[~same][None, :]
    wins = np.sum(positives > negatives) + 0.5 * np.sum(positives == negatives)

    assert roc_auc(scores, same) == pytest.approx(wins / (positives.size * negatives.size))


def test_tar_at_far_zero_is_the_last_point_before_the_first_false_accept() -> None:
    # FAR <= 0 at +inf (TAR 0) and 0.9 (TAR 1/3); the lowest threshold is 0.9.
    point = tar_at_far(SIX_SCORES, SIX_SAME, 0.0)

    assert (point.target_far, point.far, point.threshold) == (0.0, 0.0, 0.9)
    assert point.tar == pytest.approx(1 / 3)


def test_tar_at_far_between_steps_reports_the_actual_far() -> None:
    # FAR <= 0.5 at +inf, 0.9 and 0.8 (FAR 1/3); 0.7 would take FAR to 2/3.
    point = tar_at_far(SIX_SCORES, SIX_SAME, 0.5)

    assert point.target_far == 0.5
    assert point.threshold == 0.8
    assert point.far == pytest.approx(1 / 3)
    assert point.tar == pytest.approx(1 / 3)


def test_tar_at_far_one_accepts_every_pair() -> None:
    point = tar_at_far(SIX_SCORES, SIX_SAME, 1.0)

    assert (point.far, point.tar, point.threshold) == (1.0, 1.0, 0.3)


def test_tar_at_far_is_zero_at_infinity_when_the_top_score_is_a_false_accept() -> None:
    # The first real point, 0.9, already has FAR 1; only (0, 0) at +inf is within 0.
    point = tar_at_far(_scores(0.9, 0.5), _same(F, T), 0.0)

    assert (point.far, point.tar, point.threshold) == (0.0, 0.0, math.inf)


def test_compact_keeps_only_the_corners_of_a_staircase() -> None:
    # Positives 0.9, 0.8, 0.5; negatives 0.7, 0.6, 0.4. The curve:
    #   (0,0) inf -> (0,1/3) 0.9 -> (0,2/3) 0.8 -> (1/3,2/3) 0.7 -> (2/3,2/3) 0.6
    #   -> (2/3,1) 0.5 -> (1,1) 0.4
    # (0,1/3) sits mid-way up a vertical run and (1/3,2/3) mid-way along a horizontal one.
    curve = roc(_scores(0.9, 0.8, 0.7, 0.6, 0.5, 0.4), _same(T, T, F, F, T, F))

    compacted = compact(curve)

    np.testing.assert_allclose(compacted.far, [0, 0, 2 / 3, 2 / 3, 1])
    np.testing.assert_allclose(compacted.tar, [0, 2 / 3, 2 / 3, 1, 1])
    np.testing.assert_array_equal(compacted.thresholds, [math.inf, 0.8, 0.6, 0.5, 0.4])


def test_compact_collapses_a_diagonal_run_to_its_ends() -> None:
    # Each score ties a positive with a negative: (0,0), (1/3,1/3), (2/3,2/3), (1,1), one line.
    curve = roc(_scores(0.9, 0.9, 0.5, 0.5, 0.1, 0.1), _same(T, F, T, F, T, F))

    compacted = compact(curve)

    np.testing.assert_array_equal(compacted.far, [0, 1])
    np.testing.assert_array_equal(compacted.tar, [0, 1])
    np.testing.assert_array_equal(compacted.thresholds, [math.inf, 0.1])


def test_compact_leaves_a_curve_with_no_collinear_points_alone() -> None:
    curve = roc(SIX_SCORES, SIX_SAME)

    compacted = compact(curve)

    np.testing.assert_array_equal(compacted.far, curve.far)
    np.testing.assert_array_equal(compacted.tar, curve.tar)
    np.testing.assert_array_equal(compacted.thresholds, curve.thresholds)


def test_compact_keeps_both_points_of_a_two_point_curve() -> None:
    curve = roc(_scores(0.5, 0.5), _same(T, F))

    compacted = compact(curve)

    np.testing.assert_array_equal(compacted.far, [0, 1])
    np.testing.assert_array_equal(compacted.thresholds, [math.inf, 0.5])


type PairMetric = Callable[[NDArray[np.float64], NDArray[np.bool_]], object]

EVERY_METRIC: list[PairMetric] = [
    lambda s, y: accuracy(s, y, 0.5),
    best_threshold,
    lambda s, y: kfold_accuracy(s, y, np.arange(s.size) % 2),
    roc,
    roc_auc,
    lambda s, y: tar_at_far(s, y, 0.1),
]
CURVE_METRICS: list[PairMetric] = [roc, roc_auc, lambda s, y: tar_at_far(s, y, 0.1)]


@pytest.mark.parametrize("metric", EVERY_METRIC)
@pytest.mark.parametrize(
    ("scores", "same"),
    [
        pytest.param(_scores(0.1, 0.2, 0.3, 0.4), _same(T, F, T), id="mismatched lengths"),
        pytest.param(_scores(), _same(), id="empty"),
        pytest.param(_scores(0.1, math.nan, 0.3, 0.4), _same(T, F, T, F), id="nan score"),
        pytest.param(_scores(0.1, 0.2, math.inf, 0.4), _same(T, F, T, F), id="infinite score"),
        pytest.param(_scores(0.1, 0.2, 0.3, 0.4).reshape(2, 2), _same(T, F, T, F), id="2-d"),
    ],
)
def test_every_metric_rejects_malformed_pairs(
    metric: PairMetric, scores: NDArray[np.float64], same: NDArray[np.bool_]
) -> None:
    with pytest.raises(ValueError, match=r"scores|same|pairs"):
        metric(scores, same)


@pytest.mark.parametrize("metric", CURVE_METRICS)
@pytest.mark.parametrize("label", [T, F])
def test_curve_metrics_need_both_positives_and_negatives(metric: PairMetric, label: bool) -> None:
    with pytest.raises(ValueError, match="positive" if not label else "negative"):
        metric(_scores(0.1, 0.2, 0.3), _same(label, label, label))


def test_accuracy_rejects_a_nan_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        accuracy(_scores(0.1), _same(T), math.nan)


@pytest.mark.parametrize(
    "thresholds",
    [_scores(), _scores(0.5, math.nan), _scores(0.5, 0.2).reshape(2, 1)],
    ids=["empty", "nan", "2-d"],
)
def test_best_threshold_rejects_a_malformed_grid(thresholds: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        best_threshold(_scores(0.9, 0.1), _same(T, F), thresholds)


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        (np.array([0, 0, 2, 2]), "fold 1 is empty"),
        (np.array([0, 1, -1, 1]), "numbered from 0"),
        (np.array([0, 1, 0]), "one fold per pair"),
        (np.array([0, 0, 0, 0]), "at least two folds"),
        (np.array([0.0, 1.0, 0.0, 1.0]), "integers"),
    ],
)
def test_kfold_rejects_malformed_folds(folds: NDArray[np.int_], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        kfold_accuracy(_scores(0.9, 0.1, 0.8, 0.2), _same(T, F, T, F), folds)


@pytest.mark.parametrize("target_far", [-0.1, 1.1, math.nan])
def test_tar_at_far_rejects_a_target_outside_zero_to_one(target_far: float) -> None:
    with pytest.raises(ValueError, match="target_far"):
        tar_at_far(SIX_SCORES, SIX_SAME, target_far)


def test_compact_rejects_a_curve_whose_arrays_differ_in_length() -> None:
    curve = Roc(far=_scores(0, 1), tar=_scores(0, 0.5, 1), thresholds=_scores(math.inf, 0.5))

    with pytest.raises(ValueError, match="same length"):
        compact(curve)
