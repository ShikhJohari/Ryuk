"""The same-person threshold: a 1:1 cut-off frozen on impostor pairs (#49, #47 Q6)."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from embedded_draws import embedded_draw
from ryuk.eda.summary import Draw
from ryuk.evaluation.openset import EmbeddedDraw, Probes
from ryuk.evaluation.results import SamePerson
from ryuk.evaluation.same_person import (
    FrozenSamePerson,
    Pairs,
    freeze_same_person,
    pairs,
    same_person,
)


def _at(*degrees: float) -> list[NDArray[np.float32]]:
    """Unit vectors in the plane at these angles, so a pair's cosine is the cosine of the angle
    between them."""
    return [
        np.array([math.cos(math.radians(d)), math.sin(math.radians(d))], np.float32)
        for d in degrees
    ]


def _cos(degrees: float) -> float:
    return math.cos(math.radians(degrees))


def _probes(by_identity: dict[int, list[float]]) -> Probes:
    identities = [i for i, angles in by_identity.items() for _ in angles]
    return Probes(
        np.array(identities, dtype=np.int_),
        np.stack([v for angles in by_identity.values() for v in _at(*angles)]),
    )


def _plane(draw: Draw = "validation") -> EmbeddedDraw:
    """Identity 1 enrolled at 0° and 10° with probes at 20° and 40°; identity 2 enrolled at 90°
    and 95° with a probe at 100°; held-out identity 9 with probes at 180° and 60°."""
    return EmbeddedDraw(
        draw=draw,
        enrolled={1: _at(0, 10), 2: _at(90, 95)},
        mated=_probes({1: [20, 40], 2: [100]}),
        non_mated=_probes({9: [180, 60]}),
    )


def _sorted(identities: NDArray[np.int_], scores: NDArray[np.float64]) -> list[tuple[int, float]]:
    return sorted((int(i), round(float(s), 6)) for i, s in zip(identities, scores, strict=True))


def _expected(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return sorted((i, round(s, 6)) for i, s in pairs)


def test_one_photo_pairs_each_identitys_first_photo_with_its_own_probes_and_everyone_elses() -> (
    None
):
    scored = pairs(_plane(), photos=1)

    assert (scored.draw, scored.photos) == ("validation", 1)
    # Mated pairs are grouped by the gallery identity whose photo it is.
    assert _sorted(scored.mated_identity, scored.mated_score) == _expected(
        [(1, _cos(20)), (1, _cos(40)), (2, _cos(10))]
    )
    # Impostor pairs are grouped by the probe's identity: another gallery identity's mated
    # probe or a held-out identity's probe.
    assert _sorted(scored.impostor_identity, scored.impostor_score) == _expected(
        [
            (2, _cos(100)),  # identity 2's probe against identity 1's photo
            (9, _cos(180)),
            (9, _cos(60)),
            (1, _cos(70)),  # identity 1's probes against identity 2's photo
            (1, _cos(50)),
            (9, _cos(90)),
            (9, _cos(30)),
        ]
    )


def test_each_impostor_pair_records_whose_photos_it_compares_with() -> None:
    scored = pairs(_plane(), photos=1)

    assert sorted(
        zip(
            scored.impostor_gallery_identity.tolist(),
            scored.impostor_identity.tolist(),
            strict=True,
        )
    ) == [(1, 2), (1, 9), (1, 9), (2, 1), (2, 1), (2, 9), (2, 9)]


def test_with_several_photos_a_pair_scores_the_best_of_them() -> None:
    scored = pairs(_plane(), photos=2)

    assert _sorted(scored.mated_identity, scored.mated_score) == _expected(
        [(1, _cos(10)), (1, _cos(30)), (2, _cos(5))]
    )
    assert (2, round(_cos(90), 6)) in _sorted(scored.impostor_identity, scored.impostor_score)


def test_pairs_need_as_many_enrolled_photos_as_asked_for() -> None:
    with pytest.raises(ValueError, match="3 enrolled photos"):
        pairs(_plane(), photos=3)


def _pairs(mated: list[float], impostor: list[float], draw: Draw = "validation") -> Pairs:
    return Pairs(
        draw=draw,
        photos=1,
        mated_identity=np.arange(len(mated), dtype=np.int_),
        mated_score=np.array(mated),
        impostor_identity=np.arange(len(impostor), dtype=np.int_) + 100,
        impostor_gallery_identity=np.zeros(len(impostor), dtype=np.int_),
        impostor_score=np.array(impostor),
    )


def test_the_threshold_is_the_lowest_score_whose_false_accept_rate_meets_the_target() -> None:
    # One impostor pair in four may be accepted: 0.9 alone is; the mated 0.8 is the lowest
    # cut-off that still accepts no other, as an open-set threshold is chosen.
    frozen = freeze_same_person(_pairs([0.8, 0.4], [0.9, 0.5, 0.3, 0.1]), target_far=0.25)

    assert (frozen.value, frozen.target_far, frozen.far, frozen.impostor_pairs) == (
        0.8,
        0.25,
        0.25,
        4,
    )


def test_only_the_validation_draw_sets_the_threshold() -> None:
    with pytest.raises(ValueError, match="only the validation draw"):
        freeze_same_person(_pairs([0.8], [0.1], draw="test"))


def test_the_threshold_is_set_with_one_photo_enrolled() -> None:
    with pytest.raises(ValueError, match="one enrolled photo"):
        freeze_same_person(pairs(_plane(), photos=2))


def test_a_target_only_accepting_nothing_meets_is_refused() -> None:
    with pytest.raises(ValueError, match="accepting none"):
        freeze_same_person(_pairs([0.8], [0.9]), target_far=0.0)


def test_a_frozen_threshold_is_only_made_by_freezing() -> None:
    with pytest.raises(TypeError, match="freeze_same_person"):
        FrozenSamePerson(0.5, 0.001, 0.001, 10)


def _random_draw(draw: Draw, seed: int) -> EmbeddedDraw:
    """Noisy enough that a person's own photos sometimes score under the threshold."""
    return embedded_draw(draw, seed, noise=0.3)


def test_the_test_draw_is_scored_once_at_the_validation_threshold() -> None:
    validation, test = _random_draw("validation", 1), _random_draw("test", 2)

    result = same_person(validation, test, live_threshold=0.6, target_far=0.01, seed=7)

    frozen = freeze_same_person(pairs(validation, photos=1), target_far=0.01)
    assert (result.threshold, result.target_far, result.validation_far) == (
        frozen.value,
        0.01,
        frozen.far,
    )
    assert result.validation_impostor_pairs == 24 * (23 * 6 + 30 * 6)
    assert [r.enrolled_photos for r in result.test] == [1, 5]
    for rates in result.test:
        scored = pairs(test, photos=rates.enrolled_photos)
        assert (rates.mated_pairs, rates.impostor_pairs) == (24 * 6, 24 * (23 * 6 + 30 * 6))
        assert rates.warning_rate.value == pytest.approx(np.mean(scored.mated_score < frozen.value))
        assert rates.far.value == pytest.approx(np.mean(scored.impostor_score >= frozen.value))
        assert rates.far.ci.low <= rates.far.value <= rates.far.ci.high
        assert rates.warning_rate_at_live_threshold is not None
        assert rates.warning_rate_at_live_threshold.value == pytest.approx(
            np.mean(scored.mated_score < 0.6)
        )
    # More photos to compare with, fewer of a person's own photos warn.
    one, every = result.test
    assert every.warning_rate.value < one.warning_rate.value


def test_a_live_threshold_that_is_not_a_cosine_has_no_warning_rate() -> None:
    result = same_person(
        _random_draw("validation", 1), _random_draw("test", 2), live_threshold=None, seed=7
    )

    assert all(r.warning_rate_at_live_threshold is None for r in result.test)


@pytest.mark.parametrize("photos", [[5, 1], [2, 5], [1, 1, 5], []])
def test_the_test_draw_is_recorded_with_one_photo_first_then_more(photos: list[int]) -> None:
    result = same_person(
        _random_draw("validation", 1), _random_draw("test", 2), live_threshold=None, seed=7
    )
    document = result.model_dump(mode="json")
    one = document["test"][0]

    with pytest.raises(ValueError, match="one photo first"):
        SamePerson.model_validate(
            {**document, "test": [{**one, "enrolled_photos": n} for n in photos]}
        )
