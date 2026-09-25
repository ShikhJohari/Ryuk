"""The EDA's arithmetic on small inputs whose answers are worked out by hand in the comments."""

import pytest

from ryuk.datasets.lfw import LfwImage, Pair, PairsFile
from ryuk.detector import Box, Detection, Landmarks
from ryuk.eda.aggregate import (
    FACE_SIZE_EDGES,
    POSE_EDGES,
    SHARE_EDGES,
    LabelledImage,
    attribute_prevalence,
    celeba_draw,
    detection_stats,
    distribution,
    eligible_identities,
    gallery_candidates,
    group_stats,
    histogram,
    images_per_identity,
    majority_label,
    min_usable_face_size,
    pair_list,
)
from ryuk.eda.scan import ImageScan
from ryuk.eda.summary import (
    ATTRIBUTES,
    Attribute,
    GroupStats,
    Histogram,
    ImagesPerIdentityRow,
    Quantiles,
    Rules,
)

# --- histograms and distributions --------------------------------------------------------------


def test_the_standard_edges() -> None:
    # 0-300 px in 2 px bins, -90..90 degrees in 5 degree bins, shares in tenths.
    assert (len(FACE_SIZE_EDGES), FACE_SIZE_EDGES[0], FACE_SIZE_EDGES[-1]) == (151, 0.0, 300.0)
    assert (len(POSE_EDGES), POSE_EDGES[0], POSE_EDGES[1], POSE_EDGES[-1]) == (37, -90, -85, 90)
    assert SHARE_EDGES == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def test_a_histogram_counts_half_open_bins_with_the_last_closed() -> None:
    # [0,1): 0, 0.5 | [1,2): 1 | [2,3]: 2.999, 3 | outside: -1, 3.5
    result = histogram([0, 0.5, 1, 2.999, 3, 3.5, -1], [0, 1, 2, 3])

    assert result == Histogram(edges=[0, 1, 2, 3], counts=[2, 1, 2])


def test_a_distribution_has_mean_quantiles_and_histogram() -> None:
    # Sorted 20, 25, 50. Linear quantiles sit at q * (n - 1) = 2q between the sorted values:
    # p1 at 0.02 -> 20.1, p5 at 0.1 -> 20.5, p25 at 0.5 -> 22.5, p50 at 1 -> 25,
    # p75 at 1.5 -> 37.5, p95 at 1.9 -> 47.5, p99 at 1.98 -> 49.5. Mean 95 / 3 = 31.67.
    result = distribution([50, 20, 25], [0, 30, 60], digits=2)

    assert result.n == 3
    assert result.mean == 31.67
    assert result.quantiles == Quantiles(
        p1=20.1, p5=20.5, p25=22.5, p50=25.0, p75=37.5, p95=47.5, p99=49.5
    )
    assert result.histogram.counts == [2, 1]


def test_a_distribution_rounds_to_its_digits() -> None:
    # 1/3 and 2/3: mean 0.5; p1 = 1/3 + 0.01 * 1/3 = 0.3367 -> 0.34.
    result = distribution([1 / 3, 2 / 3], [0, 1], digits=2)

    assert result.mean == 0.5
    assert result.quantiles is not None
    assert result.quantiles.p1 == 0.34
    assert result.quantiles.p99 == 0.66


def test_an_empty_distribution_has_no_mean_or_quantiles() -> None:
    result = distribution([], [0, 1, 2], digits=2)

    assert (result.n, result.mean, result.quantiles) == (0, None, None)
    assert result.histogram.counts == [0, 0]


# --- images per identity -----------------------------------------------------------------------


def test_images_per_identity_is_a_frequency_table() -> None:
    # Six identities, 13 images; sorted counts 1 1 1 2 3 5, so the median is (1 + 2) / 2.
    result = images_per_identity([1, 1, 3, 2, 1, 5])

    assert (result.identities, result.images, result.median, result.max) == (6, 13, 1.5, 5)
    assert result.table == [
        ImagesPerIdentityRow(images=1, identities=3),
        ImagesPerIdentityRow(images=2, identities=1),
        ImagesPerIdentityRow(images=3, identities=1),
        ImagesPerIdentityRow(images=5, identities=1),
    ]


def test_no_identities_have_no_median() -> None:
    result = images_per_identity([])

    assert (result.identities, result.images, result.median, result.max, result.table) == (
        0,
        0,
        None,
        None,
        [],
    )


def test_an_identity_needs_an_image() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        images_per_identity([2, 0])


# --- minimum usable face size ------------------------------------------------------------------


def test_the_minimum_keeps_at_least_the_share_asked_for() -> None:
    # 1 of 100 faces at 25 px, the rest at 60: 99% reach 30, 40, 50 and 60, none reach 70.
    result = min_usable_face_size([25.0] + [60.0] * 99)

    assert (result.value, result.step, result.keep, result.detections) == (60, 10, 0.99, 100)
    assert (result.kept, result.kept_at_next_step) == (0.99, 0.0)


def test_a_face_exactly_at_the_value_is_kept() -> None:
    # 98 faces at exactly 50 px and 2 at exactly 40: 100% reach 40, but only 98% reach 50.
    result = min_usable_face_size([50.0] * 98 + [40.0] * 2)

    assert (result.value, result.kept, result.kept_at_next_step) == (40, 1.0, 0.98)


def test_ties_just_under_a_round_value_count_against_it() -> None:
    # 2 of 200 faces at 49.99 px: 50 keeps 198 / 200 = 99%, still enough; 60 keeps none.
    # With a third face at 49.99, 50 keeps 197 / 200 = 98.5%, so the value drops to 40.
    assert min_usable_face_size([49.99] * 2 + [55.0] * 198).value == 50
    assert min_usable_face_size([49.99] * 3 + [55.0] * 197).value == 40


def test_step_and_keep_can_be_changed() -> None:
    # Keeping everything: the smallest face is 37.5 px, so the largest multiple of 5 below it.
    result = min_usable_face_size([37.5, 80.0, 90.0], step=5, keep=1.0)

    assert (result.value, result.step, result.keep) == (35, 5, 1.0)
    # Shares keep six places, so a share just under `keep` never rounds up to it.
    assert (result.kept, result.kept_at_next_step) == (1.0, 0.666667)


def test_there_is_no_minimum_without_enough_large_faces() -> None:
    with pytest.raises(ValueError, match="no detections"):
        min_usable_face_size([])
    # Half the faces are under the first step of 10 px.
    with pytest.raises(ValueError, match="10 px"):
        min_usable_face_size([5.0, 50.0])


@pytest.mark.parametrize("keep", [0.0, -0.5, 1.01, float("nan")])
def test_keep_must_be_a_share_the_search_can_reach(keep: float) -> None:
    # A keep of 0 or less is met at every size, so the search would never stop.
    with pytest.raises(ValueError, match=r"keep must be a share in \(0, 1\]"):
        min_usable_face_size([50.0], keep=keep)


@pytest.mark.parametrize("step", [0, -10])
def test_step_must_be_positive(step: int) -> None:
    with pytest.raises(ValueError, match="step must be a positive number of pixels"):
        min_usable_face_size([50.0], step=step)


# --- majority labels and attribute prevalence --------------------------------------------------


@pytest.mark.parametrize(
    ("labelled", "images", "expected"),
    [(4, 5, True), (5, 5, True), (1, 5, False), (0, 1, False), (3, 5, None), (7, 10, None)],
)
def test_a_majority_needs_the_agreement_share(
    labelled: int, images: int, expected: bool | None
) -> None:
    assert majority_label(labelled, images, 0.8) is expected


def test_a_majority_needs_images() -> None:
    with pytest.raises(ValueError, match="no images"):
        majority_label(0, 0, 0.8)


def image(
    identity: int, *, detected: bool = True, usable: bool = True, **labels: bool
) -> LabelledImage:
    """An image of `identity`; attributes not named are False."""
    return LabelledImage(
        identity=identity,
        labels={attribute: labels.get(attribute, False) for attribute in ATTRIBUTES},
        detected=detected,
        usable=usable,
    )


def labelled(identity: int, flags: str, attribute: Attribute) -> list[LabelledImage]:
    """Images of `identity` with `attribute` set where `flags` has a T, such as "TTF"."""
    return [image(identity, **{attribute: flag == "T"}) for flag in flags]


def test_prevalence_is_counted_per_identity() -> None:
    images = [
        *labelled(1, "TTTTF", "Eyeglasses"),  # share 0.8: with, bin [0.8, 0.9)
        *labelled(2, "FF", "Eyeglasses"),  # share 0: without, bin [0, 0.1)
        *labelled(3, "TTTFFFFFFF", "Eyeglasses"),  # 0.3: mixed, bin [0.3, 0.4)
        *labelled(4, "T", "Eyeglasses"),  # share 1: with, the last bin, closed
        *labelled(5, "TFF", "Eyeglasses"),  # 1/3: mixed, bin [0.3, 0.4)
    ]

    result = attribute_prevalence("Eyeglasses", images, agreement=0.8)

    assert result.attribute == "Eyeglasses"
    assert result.images_with == 4 + 0 + 3 + 1 + 1
    assert result.identities_with_any == 4
    assert result.identity_share.edges == list(SHARE_EDGES)
    assert result.identity_share.counts == [1, 0, 0, 2, 0, 0, 0, 0, 1, 1]
    assert (result.majority_with, result.majority_without, result.mixed) == (2, 1, 2)


# --- groups and eligibility --------------------------------------------------------------------


def draw_images() -> list[LabelledImage]:
    """Four identities, 15 images. Male by majority at 80%: 1 and 4 with, 2 without, 3 mixed.

    identity  Male labels   detected      usable        Eyeglasses
    1         T T T         T T F         T F F         image 0
    2         F F           T T           T T           -
    3         T T F F F     all           all           image 0
    4         T T T T F     T T T T F     T T T F F     image 4
    """
    rows = {
        1: ("TTT", "TTF", "TFF", "T--"),
        2: ("FF", "TT", "TT", "--"),
        3: ("TTFFF", "TTTTT", "TTTTT", "T----"),
        4: ("TTTTF", "TTTTF", "TTTFF", "----T"),
    }
    return [
        image(
            identity,
            Male=male == "T",
            detected=detected == "T",
            usable=usable == "T",
            Eyeglasses=glasses == "T",
        )
        for identity, columns in rows.items()
        for male, detected, usable, glasses in zip(*columns, strict=True)
    ]


def test_an_identity_group_holds_all_images_of_its_majority_identities() -> None:
    result = group_stats("Male", draw_images(), agreement=0.8, min_gallery_images=2)

    assert result == (
        # Identities 1 and 4: 3 + 5 images, 2 + 4 detected, 1 + 3 usable; only 4 has 2 usable.
        GroupStats(
            attribute="Male",
            value=True,
            identities=2,
            images=8,
            detected=6,
            usable=4,
            eligible_identities=1,
        ),
        # Identity 2: both images usable. Identity 3 is mixed and in neither group.
        GroupStats(
            attribute="Male",
            value=False,
            identities=1,
            images=2,
            detected=2,
            usable=2,
            eligible_identities=1,
        ),
    )


def test_a_photo_condition_group_holds_the_images_with_that_label() -> None:
    result = group_stats("Eyeglasses", draw_images(), agreement=0.8, min_gallery_images=2)

    assert result == (
        # First images of 1 and 3 (both usable), the last of 4 (not detected).
        GroupStats(
            attribute="Eyeglasses",
            value=True,
            identities=3,
            images=3,
            detected=2,
            usable=2,
            eligible_identities=None,
        ),
        # The other 12: detected 13 - 2 = 11 of all 15's 13, usable 11 - 2 = 9.
        GroupStats(
            attribute="Eyeglasses",
            value=False,
            identities=4,
            images=12,
            detected=11,
            usable=9,
            eligible_identities=None,
        ),
    )


def test_eligibility_counts_usable_images_and_candidacy_counts_all() -> None:
    images = draw_images()

    # Images per identity 3, 2, 5, 5; usable 1, 2, 5, 3.
    assert gallery_candidates(images, 2) == 4
    assert gallery_candidates(images, 3) == 3
    assert eligible_identities(images, 2) == 3
    assert eligible_identities(images, 4) == 1


def test_a_draw_puts_it_all_together() -> None:
    images = draw_images()
    stats = detection_stats([], min_face_size=10)

    result = celeba_draw(
        "validation", images, stats, Rules(min_gallery_images=3, majority_agreement=0.8)
    )

    assert (result.draw, result.split) == ("validation", "valid")
    assert result.images_per_identity.table == [
        ImagesPerIdentityRow(images=2, identities=1),
        ImagesPerIdentityRow(images=3, identities=1),
        ImagesPerIdentityRow(images=5, identities=2),
    ]
    assert (result.gallery_candidates, result.eligible_identities) == (3, 2)
    assert [a.attribute for a in result.attributes] == list(ATTRIBUTES)
    assert [(g.attribute, g.value) for g in result.groups] == [
        (attribute, value) for attribute in ATTRIBUTES for value in (True, False)
    ]
    # Nobody wears a hat: everyone is in the without group, nobody in the with group.
    hats = [g for g in result.groups if g.attribute == "Wearing_Hat"]
    assert [(g.identities, g.images) for g in hats] == [(0, 0), (4, 15)]
    assert result.detection is stats


def test_the_test_draw_is_the_test_split() -> None:
    rules = Rules(min_gallery_images=1, majority_agreement=0.8)

    result = celeba_draw("test", [image(1)], detection_stats([], min_face_size=10), rules)

    assert (result.draw, result.split) == ("test", "test")


# --- LFW pairs ---------------------------------------------------------------------------------


def test_a_pairs_list_counts_its_identities_images_and_exclusions() -> None:
    a1, a2, a3 = LfwImage("A", 1), LfwImage("A", 2), LfwImage("A", 3)
    b1, b2, c1 = LfwImage("B", 1), LfwImage("B", 2), LfwImage("C", 1)
    pairs = PairsFile(
        "pairs",
        2,
        (Pair(a1, a2, 0), Pair(a1, b1, 0), Pair(b1, b2, 1), Pair(c1, a3, 1)),
    )

    # A2 and C1 have no usable face: the first and last pairs lose an image. D1 is in no pair.
    result = pair_list(pairs, excluded={a2, c1, LfwImage("D", 1)})

    assert (result.name, result.view, result.folds) == ("pairs", 2, 2)
    assert (result.matched, result.mismatched) == (2, 2)
    assert (result.identities, result.images) == (3, 6)
    assert result.pairs_with_excluded_image == 2


# --- detection stats ---------------------------------------------------------------------------


def face(x: float, y: float, side: float) -> Detection:
    """A square detection with a frontal face's landmarks, as `ryuk.eda.pose` models them."""

    def at(fx: float, fy: float) -> tuple[float, float]:
        return (x + side * fx, y + side * fy)

    return Detection(
        Box(x, y, side, side),
        Landmarks(at(0.3, 0.35), at(0.7, 0.35), at(0.5, 0.55), at(0.33, 0.75), at(0.67, 0.75)),
        0.95,
    )


def test_detection_stats_describe_the_centre_most_face() -> None:
    centre_20 = face(90, 90, 20)  # at the centre of a 200 px image, too small at 30
    corner_80 = face(0, 0, 80)
    scans = [
        ImageScan((200, 200), ()),  # nothing found
        ImageScan((200, 200), (face(75, 75, 50),)),  # one usable face
        ImageScan((200, 200), (corner_80, centre_20)),  # two: centre-most is the small one
        ImageScan((100, 100), (face(37.5, 37.5, 25),)),  # found, but too small
    ]

    result = detection_stats(scans, min_face_size=30)

    assert (result.images, result.detected, result.multiple_faces, result.usable) == (4, 3, 1, 2)
    # Centre-most faces 50, 20 and 25 px, the numbers of the distribution test above.
    assert result.face_short_side.n == 3
    assert result.face_short_side.mean == 31.67
    assert result.face_short_side.histogram.edges == list(FACE_SIZE_EDGES)
    counts = result.face_short_side.histogram.counts
    assert {i: c for i, c in enumerate(counts) if c} == {10: 1, 12: 1, 25: 1}
    # One pose per detected image, from the same centre-most faces.
    pose = result.head_pose
    assert (pose.yaw.n, pose.pitch.n, pose.roll.n) == (3, 3, 3)
    assert pose.roll.histogram.edges == list(POSE_EDGES)
    # The landmarks are level and symmetric and the faces centred, so neither roll nor yaw.
    for angle in (pose.yaw, pose.roll):
        assert angle.quantiles is not None
        assert max(abs(angle.quantiles.p1), abs(angle.quantiles.p99)) < 1


def test_a_face_with_no_pose_is_left_out_of_the_pose_only() -> None:
    point = (50.0, 50.0)
    collapsed = Detection(Box(25, 25, 50, 50), Landmarks(point, point, point, point, point), 0.9)

    result = detection_stats([ImageScan((100, 100), (collapsed,))], min_face_size=30)

    assert (result.detected, result.usable, result.face_short_side.n) == (1, 1, 1)
    assert result.head_pose.yaw.n == 0
