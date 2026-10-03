"""The live rule at its frozen threshold on smaller galleries and one enrolled photo (#49)."""

import datetime

import numpy as np
import pytest

from embedded_draws import embedded_draw
from ryuk.evaluation.learning import TopTwoDraw, fit_learned_rule, score_rule
from ryuk.evaluation.openset import EmbeddedDraw, freeze
from ryuk.evaluation.results import LearnedRule, MatchRule, ModelThreshold, RecognitionModelId
from ryuk.evaluation.small_galleries import partition, small_galleries

MODEL = RecognitionModelId(network="sface", provider="cpu", weights_sha256="a" * 64, dimension=16)


@pytest.fixture(scope="module")
def draw() -> EmbeddedDraw:
    return embedded_draw("test", 2, identities=20, noise=0.3)


def _frozen(
    rule: MatchRule, threshold: float, learned: LearnedRule | None = None
) -> ModelThreshold:
    return ModelThreshold(
        model=MODEL,
        rule=rule,
        threshold=threshold,
        target_fpir=0.01,
        learned_rule=learned,
        commit="1" * 40,
        date=datetime.date(2026, 10, 3),
    )


def _threshold(rule: MatchRule) -> ModelThreshold:
    validation = embedded_draw("validation", 1, identities=20, noise=0.3)
    return _frozen(rule, freeze(score_rule(rule, validation)).value)


def test_a_partition_splits_the_gallery_into_disjoint_galleries_of_one_size() -> None:
    galleries = partition(list(range(20)), 5, seed=49)

    assert galleries.shape == (4, 5)
    assert sorted(galleries.ravel().tolist()) == list(range(20))
    # Fixed by its seed and the size, whatever the photos enrolled.
    assert np.array_equal(galleries, partition(list(range(20)), 5, seed=49))
    assert not np.array_equal(galleries, partition(list(range(20)), 5, seed=50))


def test_a_size_that_does_not_divide_the_gallery_is_refused() -> None:
    with pytest.raises(ValueError, match="20 identities into galleries of 6"):
        partition(list(range(20)), 6, seed=49)


@pytest.mark.parametrize("rule", ["best-photo", "mean"])
def test_the_rehearsals_own_gallery_reproduces_its_rates_at_the_threshold(
    draw: EmbeddedDraw, rule: MatchRule
) -> None:
    frozen = _threshold(rule)

    full = small_galleries(draw, frozen, sizes=(), seed=7)[0]

    scored = score_rule(rule, draw)
    accepted = scored.mated_score >= frozen.threshold
    assert (full.identities, full.enrolled_photos, full.galleries) == (20, 5, 1)
    assert (full.mated_probes, full.non_mated_probes) == (120, 180)
    assert full.tpir.value == pytest.approx(np.mean(accepted & scored.mated_correct))
    assert full.misidentification.value == pytest.approx(np.mean(accepted & ~scored.mated_correct))
    assert full.fpir.value == pytest.approx(np.mean(scored.non_mated_score >= frozen.threshold))


def test_each_size_is_scored_with_every_photo_then_one(draw: EmbeddedDraw) -> None:
    cells = small_galleries(draw, _threshold("best-photo"), sizes=(5,), seed=7)

    assert [(c.identities, c.enrolled_photos, c.galleries) for c in cells] == [
        (20, 5, 1),
        (20, 1, 1),
        (5, 5, 4),
        (5, 1, 4),
    ]
    # Each mated probe is scored once, against its own gallery; each non-mated probe against
    # every gallery.
    assert [(c.mated_probes, c.non_mated_probes) for c in cells] == [
        (120, 180),
        (120, 180),
        (120, 720),
        (120, 720),
    ]


def test_one_photo_scores_against_each_identitys_first_enrolled_photo(draw: EmbeddedDraw) -> None:
    frozen = _threshold("best-photo")
    first_only = EmbeddedDraw(
        draw.draw,
        {identity: list(photos)[:1] for identity, photos in draw.enrolled.items()},
        draw.mated,
        draw.non_mated,
    )

    one = small_galleries(draw, frozen, sizes=(), seed=7)[1]

    expected = small_galleries(first_only, frozen, sizes=(), seed=7)[0]
    assert (one.tpir, one.fpir, one.misidentification) == (
        expected.tpir,
        expected.fpir,
        expected.misidentification,
    )


def test_a_smaller_gallery_never_raises_more_false_alarms_under_best_photo(
    draw: EmbeddedDraw,
) -> None:
    # A stranger alarms in a part of the gallery only if they alarm against the whole of it.
    cells = small_galleries(draw, _frozen("best-photo", 0.3), sizes=(10, 5, 2), seed=7)

    fpir = [c.fpir.value for c in cells if c.enrolled_photos == 5]
    assert fpir == sorted(fpir, reverse=True)
    assert fpir[0] > fpir[-1]


def test_the_learned_rule_is_scored_with_its_coefficients(draw: EmbeddedDraw) -> None:
    rule = fit_learned_rule(TopTwoDraw.of(embedded_draw("validation", 1, identities=20))).rule

    cells = small_galleries(draw, _frozen("learned", 0.5, rule), sizes=(5,), seed=7)

    scored = score_rule("learned", draw, rule)
    assert cells[0].fpir.value == pytest.approx(np.mean(scored.non_mated_score >= 0.5))


def test_the_learned_rule_needs_a_runner_up_in_every_gallery(draw: EmbeddedDraw) -> None:
    rule = fit_learned_rule(TopTwoDraw.of(embedded_draw("validation", 1, identities=20))).rule

    with pytest.raises(ValueError, match="at least two identities"):
        small_galleries(draw, _frozen("learned", 0.5, rule), sizes=(1,), seed=7)
