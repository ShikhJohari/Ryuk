"""Who the system fails: per-group rates at the single frozen threshold, worked by hand."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from celeba_files import Row, solid, write_celeba
from ryuk.eda.aggregate import LabelledImage, group_stats, majority_label
from ryuk.eda.build import RULES
from ryuk.eda.summary import ATTRIBUTES, Attribute, IdentityAttribute, PhotoCondition
from ryuk.evaluation.bias import (
    MIN_IDENTITIES,
    GroupLabels,
    LabelledProbes,
    attribute_breakdowns,
    bias_model,
    read_group_labels,
)
from ryuk.evaluation.openset import ScoredProbes
from ryuk.evaluation.results import (
    AttributeBreakdown,
    BiasAttribute,
    BiasModel,
    GroupRates,
    Rate,
    RecognitionModelId,
)

MODEL = RecognitionModelId(network="arcface", provider="cpu", weights_sha256="a" * 64, dimension=4)
THRESHOLD = 0.5
NO_PHOTO_LABELS: Mapping[PhotoCondition, bool] = {
    "Eyeglasses": False,
    "Wearing_Hat": False,
    "Blurry": False,
}

type Identities = Mapping[int, tuple[bool | None, bool | None]]
"""Each identity's (Male, Young) majority label, None where mixed."""


def _scored(
    mated: Sequence[tuple[int, float, bool]], non_mated: Sequence[tuple[int, float]]
) -> ScoredProbes:
    """Mated probes as (identity, score, top candidate right); non-mated as (identity, score)."""
    return ScoredProbes(
        draw="test",
        mated_identity=np.array([i for i, _, _ in mated], dtype=np.int_),
        mated_score=np.array([s for _, s, _ in mated], dtype=np.float64),
        mated_correct=np.array([c for _, _, c in mated], dtype=np.bool_),
        non_mated_identity=np.array([i for i, _ in non_mated], dtype=np.int_),
        non_mated_score=np.array([s for _, s in non_mated], dtype=np.float64),
    )


def _images(scored: ScoredProbes) -> tuple[list[str], list[str]]:
    mated = [f"mated-{i}.png" for i in range(scored.mated_score.size)]
    return mated, [f"non-mated-{i}.png" for i in range(scored.non_mated_score.size)]


def _probes(
    scored: ScoredProbes,
    identities: Identities,
    photos: Mapping[str, Mapping[PhotoCondition, bool]] | None = None,
) -> LabelledProbes:
    """`scored` with every probe's image named by `_images`, photo labels False unless `photos`
    gives them."""
    mated_images, non_mated_images = _images(scored)
    given = photos or {}
    labels = GroupLabels(
        identities={
            identity: {"Male": male, "Young": young}
            for identity, (male, young) in identities.items()
        },
        photos={
            name: given.get(name, NO_PHOTO_LABELS) for name in [*mated_images, *non_mated_images]
        },
    )
    return LabelledProbes(scored, mated_images, non_mated_images, labels)


def _model(
    scored: ScoredProbes,
    identities: Identities,
    *,
    photos: Mapping[str, Mapping[PhotoCondition, bool]] | None = None,
    seed: int = 0,
    min_identities: int = 1,
) -> BiasModel:
    """The breakdown with a rate for any group with an identity, unless `min_identities` says
    otherwise, so hand-worked cases stay small."""
    return BiasModel(
        model=MODEL,
        rule="best-photo",
        threshold=THRESHOLD,
        attributes=attribute_breakdowns(
            _probes(scored, identities, photos),
            THRESHOLD,
            seed=seed,
            min_identities=min_identities,
        ),
    )


def _breakdown(model: BiasModel, attribute: BiasAttribute) -> AttributeBreakdown:
    return next(b for b in model.attributes if b.attribute == attribute)


def _group(model: BiasModel, attribute: BiasAttribute, label: str) -> GroupRates:
    return next(g for g in _breakdown(model, attribute).groups if g.label == label)


def _rates(group: GroupRates) -> tuple[float | None, float | None, float | None]:
    """The group's TPIR, misidentification and FPIR values, None for a rate too few to have."""
    return _value(group.tpir), _value(group.misidentification), _value(group.fpir)


def _value(rate: Rate | None) -> float | None:
    return None if rate is None else rate.value


# Gallery identities 1-4 and held-out identities 11-14; 4 and 14 have no Male majority, 11 no
# Young majority. At the threshold 0.5 a mated probe counts for TPIR if right and accepted, for
# misidentification if wrong and accepted; a non-mated probe is a false alarm if accepted.
SMALL: Identities = {
    1: (True, True),
    2: (True, False),
    3: (False, True),
    4: (None, True),
    11: (True, None),
    12: (False, True),
    13: (False, False),
    14: (None, False),
}


def _small() -> ScoredProbes:
    return _scored(
        mated=[
            (1, 0.9, True),  # a match
            (1, 0.6, False),  # a misidentification
            (1, 0.3, True),  # right but below the threshold
            (2, 0.7, True),
            (3, 0.8, True),
            (3, 0.4, False),  # wrong but below the threshold
            (4, 0.9, True),
        ],
        non_mated=[(11, 0.6), (11, 0.2), (12, 0.7), (13, 0.1), (13, 0.1), (13, 0.1), (14, 0.9)],
    )


def test_identity_groups_follow_the_majority_label_and_count_the_mixed() -> None:
    male = _breakdown(_model(_small(), SMALL), "Male")

    assert (male.basis, male.indicative) == ("identity", False)
    assert [g.label for g in male.groups] == ["Male", "Not male"]
    assert [g.values for g in male.groups] == [{"Male": True}, {"Male": False}]
    # Identity 4 and identity 14 have no Male majority: left out and counted, one per side.
    assert (male.mixed_gallery_identities, male.mixed_held_out_identities) == (1, 1)
    counts = [
        (g.gallery_identities, g.held_out_identities, g.mated_probes, g.non_mated_probes)
        for g in male.groups
    ]
    assert counts == [(2, 1, 4, 2), (1, 2, 2, 4)]


def test_each_rate_is_its_groups_share_at_the_single_threshold() -> None:
    model = _model(_small(), SMALL)
    male, not_male = _group(model, "Male", "Male"), _group(model, "Male", "Not male")

    # Male: identities 1 and 2 have 4 mated probes; 0.9 and 0.7 are accepted and right, 0.6
    # accepted and wrong. Identity 11 has 2 non-mated probes, 0.6 accepted.
    assert _rates(male) == (0.5, 0.25, 0.5)
    # Not male: identity 3's 0.8 is accepted and right, 0.4 wrong but below. Identities 12
    # and 13 have 4 non-mated probes, only 0.7 accepted.
    assert _rates(not_male) == (0.5, 0.0, 0.25)
    assert model.threshold == THRESHOLD
    assert model.rule == "best-photo"


def test_a_rate_under_one_percent_also_gets_the_adjusted_wilson_check() -> None:
    not_male = _group(_model(_small(), SMALL), "Male", "Not male")

    assert not_male.misidentification is not None
    assert not_male.misidentification.adjusted_wilson is not None
    assert not_male.tpir is not None
    assert not_male.tpir.adjusted_wilson is None


def test_the_male_and_young_cells_are_indicative_and_leave_out_either_mixed_label() -> None:
    cells = _breakdown(_model(_small(), SMALL), "Male_and_Young")

    assert (cells.basis, cells.indicative) == ("identity", True)
    assert [g.label for g in cells.groups] == [
        "Male, young",
        "Male, not young",
        "Not male, young",
        "Not male, not young",
    ]
    assert [g.values for g in cells.groups] == [
        {"Male": True, "Young": True},
        {"Male": True, "Young": False},
        {"Male": False, "Young": True},
        {"Male": False, "Young": False},
    ]
    # Gallery identity 4 has no Male majority; held-out 11 no Young majority, 14 no Male.
    assert (cells.mixed_gallery_identities, cells.mixed_held_out_identities) == (1, 2)
    assert [(g.gallery_identities, g.held_out_identities) for g in cells.groups] == [
        (1, 0),
        (1, 0),
        (1, 1),
        (0, 1),
    ]
    indicative = [b.attribute for b in _model(_small(), SMALL).attributes if b.indicative]
    assert indicative == ["Male_and_Young"]


def test_a_side_with_no_probes_is_still_reported_without_rates() -> None:
    cells = _breakdown(_model(_small(), SMALL), "Male_and_Young")
    male_not_young, not_male_not_young = cells.groups[1], cells.groups[3]

    assert (male_not_young.non_mated_probes, male_not_young.fpir) == (0, None)
    assert male_not_young.tpir is not None
    assert (not_male_not_young.mated_probes, not_male_not_young.tpir) == (0, None)
    assert not_male_not_young.misidentification is None
    assert not_male_not_young.fpir is not None


def test_attributes_come_in_order_with_true_before_false() -> None:
    model = _model(_small(), SMALL)

    assert [b.attribute for b in model.attributes] == [
        "Male",
        "Young",
        "Male_and_Young",
        "Eyeglasses",
        "Wearing_Hat",
        "Blurry",
    ]
    assert [[g.label for g in b.groups] for b in model.attributes if not b.indicative] == [
        ["Male", "Not male"],
        ["Young", "Not young"],
        ["Eyeglasses", "No eyeglasses"],
        ["Hat", "No hat"],
        ["Blurry", "Not blurry"],
    ]


def test_photo_conditions_group_each_probe_by_its_own_image() -> None:
    scored = _small()
    mated_images, non_mated_images = _images(scored)
    # Identity 1's first two probes wear glasses, its third does not; identity 12's one does.
    glasses = {**NO_PHOTO_LABELS, "Eyeglasses": True}
    photos = {mated_images[0]: glasses, mated_images[1]: glasses, non_mated_images[2]: glasses}

    eyeglasses = _breakdown(_model(scored, SMALL, photos=photos), "Eyeglasses")

    assert (eyeglasses.basis, eyeglasses.indicative) == ("photo", False)
    assert (eyeglasses.mixed_gallery_identities, eyeglasses.mixed_held_out_identities) == (0, 0)
    with_them, without = eyeglasses.groups
    assert with_them.values == {"Eyeglasses": True}
    # Mixed-label identities count here: a photo condition ignores the identity's majority.
    assert (with_them.gallery_identities, with_them.mated_probes) == (1, 2)
    assert (without.gallery_identities, without.mated_probes) == (4, 5)
    assert (with_them.held_out_identities, with_them.non_mated_probes) == (1, 1)
    assert (without.held_out_identities, without.non_mated_probes) == (3, 6)
    # 0.9 right and 0.6 wrong, both accepted; 12's one probe at 0.7 is a false alarm.
    assert _rates(with_them) == (0.5, 0.5, 1.0)
    assert _rates(without)[2] == pytest.approx(2 / 6)


def _population(male: int, not_male: int, held_male: int, held_not_male: int) -> ScoredProbes:
    """One mated probe per gallery identity and one non-mated per held-out identity, scores
    from a fixed stream: male gallery identities from 0, not male from 1000, held-out male from
    2000 and not male from 3000."""
    rng = np.random.default_rng(7)

    def ids(start: int, count: int) -> range:
        return range(start, start + count)

    gallery = [*ids(0, male), *ids(1000, not_male)]
    held_out = [*ids(2000, held_male), *ids(3000, held_not_male)]
    return _scored(
        mated=[(i, float(rng.uniform()), bool(rng.uniform() < 0.9)) for i in gallery],
        non_mated=[(i, float(rng.uniform(0, 0.6))) for i in held_out],
    )


def _population_labels(scored: ScoredProbes) -> Identities:
    everyone = np.concatenate([scored.mated_identity, scored.non_mated_identity])
    return {int(i): (int(i) % 2000 < 1000, True) for i in everyone}


def test_a_side_with_fewer_than_30_identities_is_too_few_but_still_reported() -> None:
    assert MIN_IDENTITIES == 30
    scored = _population(male=29, not_male=30, held_male=30, held_not_male=29)

    model = _model(scored, _population_labels(scored), min_identities=MIN_IDENTITIES)
    male, not_male = _group(model, "Male", "Male"), _group(model, "Male", "Not male")

    assert (male.gallery_identities, male.mated_probes) == (29, 29)
    assert (male.tpir, male.misidentification) == (None, None)
    assert male.fpir is not None
    assert (not_male.held_out_identities, not_male.non_mated_probes) == (29, 29)
    assert not_male.tpir is not None
    assert not_male.misidentification is not None
    assert not_male.fpir is None
    # With one group's FPIR too few, no ratio.
    assert _breakdown(model, "Male").fpir_ratio is None


def test_the_fpir_ratio_is_the_worst_group_over_the_best() -> None:
    model = _model(_small(), SMALL)

    assert _breakdown(model, "Male").fpir_ratio == 2.0
    # Young: 12 alarms 1 of 1, 13 and 14 alarm 1 of 4.
    assert _breakdown(model, "Young").fpir_ratio == 4.0


def test_no_fpir_ratio_when_the_best_group_has_no_false_alarms() -> None:
    scored = _scored(mated=[(1, 0.9, True), (3, 0.9, True)], non_mated=[(11, 0.9), (13, 0.1)])

    male = _breakdown(_model(scored, SMALL), "Male")

    assert [g.fpir.value for g in male.groups if g.fpir is not None] == [1.0, 0.0]
    assert male.fpir_ratio is None


def test_no_fpir_ratio_with_fewer_than_two_groups_to_compare() -> None:
    # Every held-out identity is not young, so Young has one group with an FPIR.
    scored = _scored(mated=[(1, 0.9, True)], non_mated=[(13, 0.9), (14, 0.1)])

    assert _breakdown(_model(scored, SMALL), "Young").fpir_ratio is None


def test_intervals_are_fixed_by_the_seed() -> None:
    scored = _population(male=40, not_male=40, held_male=40, held_not_male=40)
    labels = _population_labels(scored)

    first, again = _model(scored, labels, seed=3), _model(scored, labels, seed=3)
    other = _model(scored, labels, seed=4)

    assert first == again
    # Another seed moves the intervals, never the rates.
    assert first != other
    assert _values(first) == _values(other)


def _values(model: BiasModel) -> list[float | None]:
    return [
        None if rate is None else rate.value
        for breakdown in model.attributes
        for group in breakdown.groups
        for rate in (group.tpir, group.misidentification, group.fpir)
    ]


def test_a_groups_intervals_do_not_depend_on_other_groups() -> None:
    scored = _population(male=40, not_male=40, held_male=40, held_not_male=40)
    labels = _population_labels(scored)
    # The same draw without any not-male identity, gallery or held out.
    male_only = _scored(
        mated=[
            (int(i), float(s), bool(c))
            for i, s, c in zip(
                scored.mated_identity, scored.mated_score, scored.mated_correct, strict=True
            )
            if i < 1000
        ],
        non_mated=[
            (int(i), float(s))
            for i, s in zip(scored.non_mated_identity, scored.non_mated_score, strict=True)
            if i < 3000
        ],
    )

    full, alone = _model(scored, labels, seed=3), _model(male_only, labels, seed=3)

    assert _group(full, "Male", "Male") == _group(alone, "Male", "Male")
    assert _group(alone, "Male", "Not male").mated_probes == 0


def test_the_breakdown_validates_as_the_results_schema() -> None:
    scored = _population(male=40, not_male=40, held_male=40, held_not_male=40)
    probes = _probes(scored, _population_labels(scored))

    model = bias_model(probes, THRESHOLD, model=MODEL, rule="mean", seed=3)

    assert BiasModel.model_validate_json(model.model_dump_json()) == model
    assert (model.model, model.rule, model.threshold) == (MODEL, "mean", THRESHOLD)
    # At the committed minimum of 30 identities: 40 on each side of Male, none not young.
    assert model.attributes == attribute_breakdowns(probes, THRESHOLD, seed=3, min_identities=30)
    young = _breakdown(model, "Young").groups
    assert young[0].tpir is not None
    assert (young[1].mated_probes, young[1].tpir) == (0, None)


def test_the_breakdown_is_only_reported_on_the_test_draw() -> None:
    scored = _small()
    validation = ScoredProbes(
        draw="validation",
        mated_identity=scored.mated_identity,
        mated_score=scored.mated_score,
        mated_correct=scored.mated_correct,
        non_mated_identity=scored.non_mated_identity,
        non_mated_score=scored.non_mated_score,
    )

    with pytest.raises(ValueError, match="test draw"):
        _probes(validation, SMALL)


@pytest.mark.parametrize("threshold", [float("inf"), float("nan")])
def test_the_threshold_must_be_a_cut_off(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        attribute_breakdowns(_probes(_small(), SMALL), threshold, seed=0)


def test_every_rate_needs_at_least_one_identity() -> None:
    with pytest.raises(ValueError, match="min_identities"):
        attribute_breakdowns(_probes(_small(), SMALL), THRESHOLD, seed=0, min_identities=0)


def test_every_probe_needs_its_labels_and_its_image() -> None:
    scored = _small()
    mated_images, non_mated_images = _images(scored)
    labels = GroupLabels(
        identities={i: {"Male": m, "Young": y} for i, (m, y) in SMALL.items() if i != 13},
        photos=dict.fromkeys([*mated_images, *non_mated_images], NO_PHOTO_LABELS),
    )
    complete = GroupLabels(
        identities={i: {"Male": m, "Young": y} for i, (m, y) in SMALL.items()},
        photos=labels.photos,
    )

    def run(labels: GroupLabels, mated: Sequence[str]) -> list[AttributeBreakdown]:
        probes = LabelledProbes(scored, mated, non_mated_images, labels)
        return attribute_breakdowns(probes, THRESHOLD, seed=0)

    with pytest.raises(ValueError, match="identity 13"):
        run(labels, mated_images)
    with pytest.raises(ValueError, match="one image per probe"):
        run(complete, mated_images[:-1])
    with pytest.raises(ValueError, match=r"unknown\.png"):
        run(complete, [*mated_images[:-1], "unknown.png"])


# --- the labels, from CelebA's label table ------------------------------------------------------


def _labelled_images() -> list[LabelledImage]:
    """Three identities over ten images, some with no usable face. Identity 1 is male in 4 of
    5 images (80%, male) but in only 2 of its 3 usable ones, and young in 3 of 5 (mixed);
    identity 2 is in none of its 4 either; identity 3's one image is male and young."""

    def image(identity: int, male: bool, young: bool, usable: bool) -> LabelledImage:
        labels: dict[Attribute, bool] = dict.fromkeys(ATTRIBUTES, False)
        labels |= {"Male": male, "Young": young}
        return LabelledImage(identity, labels, detected=usable, usable=usable)

    return [
        image(1, True, True, True),
        image(1, True, True, False),
        image(1, True, True, True),
        image(1, True, False, False),
        image(1, False, False, True),
        image(2, False, False, False),
        image(2, False, False, True),
        image(2, False, False, False),
        image(2, False, False, True),
        image(3, True, True, False),
    ]


def _table(images: Sequence[LabelledImage]) -> pa.Table:
    """`images` as `read_labels` returns them: the location columns, then the attributes."""
    columns: dict[str, list[str] | list[int] | list[bool]] = {
        "split": ["test"] * len(images),
        "path": [f"{n:06d}.png" for n in range(len(images))],
        "celeb_id": [image.identity for image in images],
    }
    for attribute in ATTRIBUTES:
        columns[attribute] = [image.labels[attribute] for image in images]
    return pa.table(columns)


def test_identity_labels_are_the_eda_majority_over_every_image_usable_or_not() -> None:
    images = _labelled_images()

    labels = GroupLabels.from_table(_table(images))

    for identity in (1, 2, 3):
        own = [image for image in images if image.identity == identity]
        for attribute in ("Male", "Young"):
            expected = majority_label(
                sum(image.labels[attribute] for image in own), len(own), RULES.majority_agreement
            )
            assert labels.identities[identity][attribute] == expected
    assert labels.identities[1] == {"Male": True, "Young": None}
    assert labels.identities[2] == {"Male": False, "Young": False}


@pytest.mark.parametrize("attribute", ["Male", "Young"])
def test_identity_groups_match_the_eda_group_counts(attribute: IdentityAttribute) -> None:
    images = _labelled_images()

    labels = GroupLabels.from_table(_table(images))
    with_it, without = group_stats(
        attribute, images, agreement=RULES.majority_agreement, min_gallery_images=1
    )

    majorities = [labels.identities[i][attribute] for i in labels.identities]
    assert (majorities.count(True), majorities.count(False)) == (
        with_it.identities,
        without.identities,
    )


def test_photo_labels_are_each_images_own() -> None:
    images = _labelled_images()
    table = _table(images).set_column(
        ATTRIBUTES.index("Blurry") + 3, "Blurry", pa.array([n == 4 for n in range(len(images))])
    )

    labels = GroupLabels.from_table(table)

    assert len(labels.photos) == len(images)
    assert labels.photos["000004.png"] == {
        "Eyeglasses": False,
        "Wearing_Hat": False,
        "Blurry": True,
    }
    assert labels.photos["000003.png"]["Blurry"] is False


def test_a_missing_label_is_refused() -> None:
    table = _table(_labelled_images())
    blanked = table.set_column(
        ATTRIBUTES.index("Male") + 3, "Male", pa.array([None] + [True] * (table.num_rows - 1))
    )

    with pytest.raises(ValueError, match="Male"):
        GroupLabels.from_table(blanked)


def test_labels_are_read_from_the_draws_split(tmp_path: Path) -> None:
    male_young = {"Male": True, "Young": True}
    write_celeba(
        tmp_path,
        {
            "valid-00000-of-00001.parquet": [Row(5, solid(1), male_young)],
            "test-00000-of-00001.parquet": [
                Row(7, solid(2), male_young),
                Row(7, solid(3), {"Male": True, "Blurry": True}),
                Row(8, solid(4)),
            ],
        },
    )

    labels = read_group_labels(tmp_path, "test")

    # Identity 7 is male in both images and young in one of two, so mixed.
    assert labels.identities == {
        7: {"Male": True, "Young": None},
        8: {"Male": False, "Young": False},
    }
    assert sorted(labels.photos) == [
        "test-00000-of-00001-0.png",
        "test-00000-of-00001-1.png",
        "test-00000-of-00001-2.png",
    ]
    assert labels.photos["test-00000-of-00001-1.png"]["Blurry"] is True
