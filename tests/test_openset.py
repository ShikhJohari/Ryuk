"""Open-set identification on a draw: best-photo scoring, TPIR and FPIR, frozen thresholds."""

import dataclasses

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
    paired_gain,
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
    frozen = freeze(_scored("validation"), target_fpir=0.25)

    with pytest.raises(TypeError, match="freeze"):
        FrozenThreshold(0.5, 0.01)
    # A frozen threshold is no dataclass, so it cannot be copied with another value either.
    with pytest.raises(TypeError):
        dataclasses.replace(frozen, value=0.1)  # type: ignore[type-var]
    with pytest.raises(AttributeError):
        frozen.value = 0.1  # type: ignore[misc]


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


def test_tpir_at_fpir_rechooses_its_threshold_in_every_resample() -> None:
    # Held-out scores all sit below every mated one: whichever identities a resample draws,
    # its FPIR 1% threshold still clears them, so TPIR is 1 in every resample.
    separated = ScoredProbes(
        draw="test",
        mated_identity=np.repeat(np.arange(50), 15),
        mated_score=np.linspace(0.6, 0.9, 750),
        mated_correct=np.ones(750, dtype=np.bool_),
        non_mated_identity=np.repeat(np.arange(1000, 1100), 10),
        non_mated_score=np.linspace(-0.3, 0.5, 1000),
    )
    frozen = freeze(_separable("validation", np.random.default_rng(2)))

    point = draw_result(separated, frozen, seed=5).operating_points[0]

    assert (point.target_fpir, point.tpir.value) == (0.01, 1.0)
    assert (point.tpir.ci.low, point.tpir.ci.high) == (1.0, 1.0)


def test_an_overlapping_draw_gets_a_tpir_at_fpir_interval_around_its_value() -> None:
    rng = np.random.default_rng(6)
    frozen = freeze(_separable("validation", rng))

    point = draw_result(_separable("test", rng), frozen, seed=5).operating_points[0]

    assert point.tpir.ci.low < point.tpir.value < point.tpir.ci.high


def test_the_averaged_gallery_scores_each_identity_by_its_renormalised_mean() -> None:
    rng = np.random.default_rng(3)
    enrolled = {
        identity: [_unit(*rng.normal(size=8)) for _ in range(photos)]
        for identity, photos in ((4, 3), (9, 1), (2, 5))
    }
    probes = np.stack([_unit(*rng.normal(size=8)) for _ in range(20)])

    identities, scores = Gallery.enrol(enrolled).averaged().top_candidates(probes)

    means = {i: np.mean(photos, axis=0) for i, photos in enrolled.items()}
    cosines = np.stack(
        [probes @ (means[i] / np.linalg.norm(means[i])) for i in sorted(enrolled)], axis=1
    )
    assert identities.tolist() == [sorted(enrolled)[i] for i in np.argmax(cosines, axis=1)]
    assert scores == pytest.approx(np.max(cosines, axis=1), abs=1e-6)


def test_an_identity_whose_photos_cancel_out_has_no_mean() -> None:
    gallery = Gallery.enrol({1: [_unit(1, 0), _unit(-1, 0)], 2: [_unit(0, 1)]})

    with pytest.raises(ValueError, match="cancel out"):
        gallery.averaged()


def test_the_runner_up_is_the_second_best_identity_not_the_top_ones_second_photo() -> None:
    # Identity 1's two photos both beat identity 2, which beats identity 3.
    gallery = Gallery.enrol(
        {
            1: [_unit(1, 0.1, 0), _unit(1, 0.2, 0)],
            2: [_unit(1, 1, 0)],
            3: [_unit(0, 0, 1)],
        }
    )
    probe = np.stack([_unit(1, 0, 0)])

    top_two = gallery.top_two(probe)

    assert top_two.identities.tolist() == [1]
    assert top_two.scores == pytest.approx([1 / np.sqrt(1.01)], abs=1e-6)
    assert top_two.runner_up_scores == pytest.approx([1 / np.sqrt(2)], abs=1e-6)
    assert top_two.gaps == pytest.approx([1 / np.sqrt(1.01) - 1 / np.sqrt(2)], abs=1e-6)


def test_the_top_two_agree_with_the_top_candidates() -> None:
    rng = np.random.default_rng(8)
    gallery = Gallery.enrol({i: [_unit(*rng.normal(size=6)) for _ in range(3)] for i in range(10)})
    probes = np.stack([_unit(*rng.normal(size=6)) for _ in range(50)])

    top_two = gallery.top_two(probes)
    identities, scores = gallery.top_candidates(probes)

    assert np.array_equal(top_two.identities, identities)
    assert np.array_equal(top_two.scores, scores)
    assert (top_two.gaps >= 0).all()


def test_a_tie_for_the_top_leaves_no_gap_to_the_runner_up() -> None:
    gallery = Gallery.enrol({5: [_unit(1, 0)], 3: [_unit(1, 0)]})

    top_two = gallery.top_two(np.stack([_unit(1, 0)]))

    assert top_two.identities.tolist() == [3]
    assert top_two.gaps.tolist() == [0.0]


def test_a_runner_up_needs_two_identities() -> None:
    gallery = Gallery.enrol({5: [_unit(1, 0), _unit(0, 1)]})

    with pytest.raises(ValueError, match="at least two identities"):
        gallery.top_two(np.stack([_unit(1, 0)]))


def _overlapping(rng: np.random.Generator, draw: Draw = "test") -> ScoredProbes:
    """Like `_separable`, but a tenth of the mated probes are misidentified and held-out scores
    overlap the mated ones, so TPIR at FPIR 1% varies from resample to resample."""
    probes = _separable(draw, rng)
    return dataclasses.replace(
        probes,
        mated_correct=rng.uniform(size=probes.mated_score.size) > 0.1,
        non_mated_score=rng.uniform(-0.2, 0.75, probes.non_mated_score.size),
    )


def test_a_method_identical_to_the_baseline_gains_nothing() -> None:
    baseline = _overlapping(np.random.default_rng(1))

    gain = paired_gain(baseline, baseline, 0.01, seed=5)

    assert (gain.target_fpir, gain.value) == (0.01, 0.0)
    assert (gain.ci.low, gain.ci.high) == (0.0, 0.0)
    assert not gain.improves


def test_the_gain_is_paired_so_a_rescored_baseline_gains_nothing_in_every_resample() -> None:
    # A monotone rescoring ranks every probe as before, so each resample picks the same
    # probes. Unpaired, the two TPIRs would still differ by their own resampling noise.
    baseline = _overlapping(np.random.default_rng(2))
    rescored = dataclasses.replace(
        baseline,
        mated_score=2 * baseline.mated_score + 0.1,
        non_mated_score=2 * baseline.non_mated_score + 0.1,
    )
    frozen = freeze(_separable("validation", np.random.default_rng(3)))
    alone = draw_result(baseline, frozen, seed=5).operating_points[0].tpir.ci

    gain = paired_gain(baseline, rescored, 0.01, seed=5)

    assert alone.low < alone.high
    assert (gain.value, gain.ci.low, gain.ci.high) == (0.0, 0.0, 0.0)


def test_a_method_that_ranks_mated_probes_higher_improves_on_the_baseline() -> None:
    baseline = _overlapping(np.random.default_rng(4))
    better = dataclasses.replace(
        baseline, mated_score=baseline.mated_score + 0.2 * baseline.mated_correct
    )

    gain = paired_gain(baseline, better, 0.01, seed=5)

    expected = tpir_at_fpir(better, 0.01).tpir - tpir_at_fpir(baseline, 0.01).tpir
    assert gain.value == pytest.approx(expected)
    assert 0 < gain.ci.low <= gain.value <= gain.ci.high
    assert gain.improves
    assert paired_gain(better, baseline, 0.01, seed=5).ci.high < 0
    assert paired_gain(baseline, better, 0.01, seed=5) == gain


@pytest.mark.parametrize(
    "change",
    [
        {"draw": "validation"},
        {"mated_identity": np.repeat(np.arange(1, 101), 15)},
        {"non_mated_identity": np.repeat(np.arange(2000, 2200), 10)},
    ],
)
def test_a_gain_compares_methods_on_the_same_probes_only(change: dict[str, object]) -> None:
    baseline = _overlapping(np.random.default_rng(5))

    with pytest.raises(ValueError, match="same probes"):
        paired_gain(baseline, dataclasses.replace(baseline, **change), 0.01, seed=5)  # type: ignore[arg-type]
