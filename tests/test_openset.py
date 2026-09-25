"""Open-set identification on a draw: best-photo scoring, TPIR and FPIR, frozen thresholds."""

import numpy as np
import pytest
from numpy.typing import NDArray

from ryuk.eda.summary import Draw
from ryuk.evaluation.openset import (
    FrozenThreshold,
    Gallery,
    Probes,
    ScoredProbes,
    draw_result,
    freeze,
    open_set_curve,
    rank_1,
    score_probes,
    tpir_at_fpir,
)


def _unit(*values: float) -> NDArray[np.float32]:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _scored(draw: Draw = "validation") -> ScoredProbes:
    """Four mated probes (one misidentified) and four non-mated, worked by hand below."""
    return ScoredProbes(
        draw=draw,
        mated_identity=np.array([1, 1, 2, 2]),
        mated_score=np.array([0.9, 0.6, 0.8, 0.5]),
        mated_correct=np.array([True, True, False, True]),
        non_mated_identity=np.array([7, 7, 8, 9]),
        non_mated_score=np.array([0.7, 0.4, 0.55, 0.3]),
    )


def test_a_probe_scores_against_each_identitys_best_enrolled_photo() -> None:
    gallery = Gallery.enrol({5: [_unit(1, 0, 0), _unit(0, 1, 0)], 3: [_unit(0, 0, 1)]})
    mated = Probes(np.array([5, 3]), np.stack([_unit(0.1, 1, 0), _unit(0, 1, 0.2)]))
    non_mated = Probes(np.array([9]), np.stack([_unit(0.2, 0, 1)]))

    scored = score_probes("validation", gallery, mated, non_mated)

    # The first probe is nearest identity 5's second photo; the second is nearer 5 than 3, so
    # identity 3's probe is misidentified as 5.
    assert scored.mated_score == pytest.approx([1 / np.sqrt(1.01), 1 / np.sqrt(1.04)], abs=1e-6)
    assert scored.mated_correct.tolist() == [True, False]
    assert scored.non_mated_score == pytest.approx([1 / np.sqrt(1.04)], abs=1e-6)
    assert scored.draw == "validation"


def test_rank_1_is_the_share_of_mated_probes_whose_top_candidate_is_right() -> None:
    assert rank_1(_scored()) == 0.75


def test_tpir_at_fpir_takes_the_lowest_threshold_within_the_fpir_budget() -> None:
    # One false alarm of four allowed: the threshold falls to 0.6, just above the second non-
    # mated score. Mated probes at 0.9 and 0.6 are right; 0.8 names the wrong identity.
    point = tpir_at_fpir(_scored(), 0.25)

    assert (point.threshold, point.tpir, point.fpir) == (0.6, 0.5, 0.25)


def test_no_false_alarm_allowed_puts_the_threshold_above_every_non_mated_score() -> None:
    point = tpir_at_fpir(_scored(), 0.01)

    assert (point.threshold, point.tpir, point.fpir) == (0.8, 0.25, 0.0)


def test_the_curve_runs_from_accepting_nothing_to_accepting_every_probe() -> None:
    curve = open_set_curve(_scored())

    assert curve.thresholds[0] == np.inf
    assert curve.fpir.tolist() == [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 1.0]
    assert curve.tpir.tolist() == [0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75]
    assert curve.misidentification.tolist() == [0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]


def test_a_threshold_is_frozen_on_the_validation_draw() -> None:
    frozen = freeze(_scored("validation"), target_fpir=0.25)

    assert (frozen.value, frozen.target_fpir) == (0.6, 0.25)


def test_the_test_draw_can_never_set_a_threshold() -> None:
    with pytest.raises(ValueError, match="only the validation draw sets a threshold"):
        freeze(_scored("test"), target_fpir=0.25)


def test_a_threshold_cannot_be_made_except_by_freezing_one() -> None:
    with pytest.raises(TypeError, match="freeze"):
        FrozenThreshold(0.5, 0.01)


def _separable(draw: Draw, rng: np.random.Generator) -> ScoredProbes:
    """100 gallery identities x 15 mated probes and 200 held-out x 10 non-mated probes. Every
    mated probe is right; about 1 in 40 non-mated probes outscores the lowest mated ones."""
    mated_identity = np.repeat(np.arange(100), 15)
    non_mated_identity = np.repeat(np.arange(1000, 1200), 10)
    return ScoredProbes(
        draw=draw,
        mated_identity=mated_identity,
        mated_score=rng.uniform(0.5, 0.9, mated_identity.size),
        mated_correct=np.ones(mated_identity.size, dtype=np.bool_),
        non_mated_identity=non_mated_identity,
        non_mated_score=rng.uniform(-0.2, 0.51, non_mated_identity.size),
    )


def test_a_draw_reports_its_rates_at_the_frozen_threshold_with_intervals() -> None:
    rng = np.random.default_rng(4)
    frozen = freeze(_separable("validation", rng))
    test = _separable("test", rng)

    result = draw_result(test, frozen, seed=1)

    assert result.draw == "test"
    assert (result.mated_probes, result.non_mated_probes) == (1500, 2000)
    assert result.at_threshold.threshold == frozen.value
    # Every mated probe is right, so rank-1 is 1 and no probe is misidentified.
    assert result.rank_1.value == 1.0
    assert (result.rank_1.ci.low, result.rank_1.ci.high) == (1.0, 1.0)
    assert result.at_threshold.misidentification.value == 0.0
    for rate in (result.at_threshold.tpir, result.at_threshold.fpir):
        assert rate.ci.low <= rate.value <= rate.ci.high
    # Error rates under 1%, and success rates over 99%, carry the adjusted Wilson check.
    assert result.at_threshold.misidentification.adjusted_wilson is not None
    assert result.rank_1.adjusted_wilson is not None
    assert [(p.target_fpir, p.indicative) for p in result.operating_points] == [
        (0.01, False),
        (0.001, True),
    ]
    assert result.curve.fpir[0] == 0.0
    assert result.curve.fpir[-1] == result.curve.tpir[-1] == 1.0
    assert len(result.curve.fpir) <= 202


def test_a_draws_intervals_are_fixed_by_the_seed() -> None:
    rng = np.random.default_rng(9)
    frozen = freeze(_separable("validation", rng))
    test = _separable("test", rng)

    assert draw_result(test, frozen, seed=3) == draw_result(test, frozen, seed=3)
