from collections.abc import Callable
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from ryuk.eda.summary import EdaSummary, HeadPose
from ryuk.plotting.eda import (
    EDA_FIGURES,
    _nice_top,
    _signed,
    eda_figure,
    save_eda_figures,
)
from ryuk.plotting.style import COLUMN_WIDTH
from summaries import POSE_EDGES, SIZE_EDGES, celeba_draw, eda_summary, empty_distribution

SLUGS = [
    "lfw-images-per-identity",
    "lfw-pair-composition",
    "celeba-images-per-identity",
    "celeba-attribute-prevalence",
    "yunet-detection-rate",
    "face-size",
    "head-pose",
    "exclusions",
]


def _texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if isinstance(t, Text) and t.get_text()]


def _render(slug: str, summary: EdaSummary) -> Figure:
    fig = eda_figure(slug, summary)
    fig.draw_without_rendering()  # lays out and draws every artist, without an output file
    return fig


def test_the_registry_names_every_figure_in_report_order() -> None:
    assert list(EDA_FIGURES) == SLUGS


@pytest.mark.parametrize("slug", SLUGS)
def test_every_figure_renders_a_report_column_wide_with_direct_labels(slug: str) -> None:
    fig = _render(slug, eda_summary())
    assert fig.get_size_inches()[0] == COLUMN_WIDTH
    assert fig.axes
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert fig.legends == []


@pytest.mark.parametrize("slug", SLUGS)
def test_no_figure_bakes_in_a_caption(slug: str) -> None:
    fig = _render(slug, eda_summary())
    assert not any(text.startswith(("Figure", "Fig.")) for text in _texts(fig))
    assert fig.get_suptitle() == ""


def _empty_tables(summary: EdaSummary) -> EdaSummary:
    lfw = summary.lfw.model_copy(
        update={
            "images_per_identity": summary.lfw.images_per_identity.model_copy(
                update={"table": [], "identities": 0, "images": 0, "median": None, "max": None}
            ),
            "pairs": [],
        }
    )
    draws = [
        d.model_copy(
            update={
                "images_per_identity": d.images_per_identity.model_copy(
                    update={"table": [], "identities": 0, "images": 0}
                ),
                "attributes": [],
                "groups": [],
            }
        )
        for d in summary.celeba
    ]
    return summary.model_copy(update={"lfw": lfw, "celeba": draws})


def _no_celeba(summary: EdaSummary) -> EdaSummary:
    return summary.model_copy(update={"celeba": []})


def _no_detections(summary: EdaSummary) -> EdaSummary:
    """Nothing detected anywhere: every distribution empty, as are the quantiles."""
    size = empty_distribution(SIZE_EDGES)
    angle = empty_distribution(POSE_EDGES)
    pose = HeadPose(yaw=angle, pitch=angle, roll=angle)

    lfw_detection = summary.lfw.detection.model_copy(
        update={"detected": 0, "usable": 0, "face_short_side": size, "head_pose": pose}
    )
    draws = [
        d.model_copy(
            update={
                "detection": d.detection.model_copy(
                    update={"detected": 0, "usable": 0, "face_short_side": size, "head_pose": pose}
                ),
                "groups": [g.model_copy(update={"detected": 0, "usable": 0}) for g in d.groups],
            }
        )
        for d in summary.celeba
    ]
    return summary.model_copy(
        update={"lfw": summary.lfw.model_copy(update={"detection": lfw_detection}), "celeba": draws}
    )


def _an_empty_group(summary: EdaSummary) -> EdaSummary:
    """One draw's hat wearers have no images at all, and the other draw lacks the group."""
    validation, test = summary.celeba
    emptied = [
        g.model_copy(update={"identities": 0, "images": 0, "detected": 0, "usable": 0})
        if (g.attribute, g.value) == ("Wearing_Hat", True)
        else g
        for g in validation.groups
    ]
    missing = [g for g in test.groups if (g.attribute, g.value) != ("Wearing_Hat", True)]
    return summary.model_copy(
        update={
            "celeba": [
                validation.model_copy(update={"groups": emptied}),
                test.model_copy(update={"groups": missing}),
            ]
        }
    )


def _one_draw(summary: EdaSummary) -> EdaSummary:
    return summary.model_copy(update={"celeba": [celeba_draw("test")]})


def _pose_without_quantiles(summary: EdaSummary) -> EdaSummary:
    angle = summary.lfw.detection.head_pose.yaw.model_copy(update={"quantiles": None})
    pose = HeadPose(yaw=angle, pitch=angle, roll=angle)
    detection = summary.lfw.detection.model_copy(update={"head_pose": pose})
    return summary.model_copy(
        update={"lfw": summary.lfw.model_copy(update={"detection": detection})}
    )


EDGE_CASES: dict[str, Callable[[EdaSummary], EdaSummary]] = {
    "empty tables": _empty_tables,
    "no CelebA draws": _no_celeba,
    "no detections": _no_detections,
    "a group with no images": _an_empty_group,
    "one draw": _one_draw,
    "pose without quantiles": _pose_without_quantiles,
}


@pytest.mark.parametrize("case", EDGE_CASES)
@pytest.mark.parametrize("slug", SLUGS)
def test_every_figure_survives_edge_cases(slug: str, case: str) -> None:
    fig = _render(slug, EDGE_CASES[case](eda_summary()))
    assert fig.axes


def test_empty_panels_say_so() -> None:
    texts = _texts(_render("lfw-images-per-identity", _empty_tables(eda_summary())))
    assert "No LFW identities in the summary" in texts
    assert "No detections in the summary" in _texts(
        _render("face-size", _no_detections(eda_summary()))
    )


def test_a_group_with_no_images_is_marked_not_dropped() -> None:
    fig = _render("yunet-detection-rate", _an_empty_group(eda_summary()))
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "Wearing hat" in labels
    assert "no images" in _texts(fig)
    assert "no images" not in _texts(_render("yunet-detection-rate", eda_summary()))


def test_photo_conditions_have_no_eligible_identities() -> None:
    texts = _texts(_render("exclusions", eda_summary()))
    assert "a photo condition groups images, not identities" in texts
    assert "LFW has no gallery" in texts


def test_detection_rates_near_100_percent_get_a_zoomed_axis() -> None:
    fig = _render("yunet-detection-rate", eda_summary())
    low, high = fig.axes[0].get_xlim()
    assert 50 < low < 95
    assert 100 <= high < 101


def test_face_size_marks_the_minimum_usable_face_size_and_the_rule() -> None:
    summary = eda_summary()
    texts = " ".join(_texts(_render("face-size", summary)))
    assert "40 px" in texts
    assert "keeps 99.5%" in texts
    assert "the 99% rule" in texts


def test_celeba_images_per_identity_marks_the_gallery_threshold_and_candidates() -> None:
    texts = _texts(_render("celeba-images-per-identity", eda_summary()))
    assert "Gallery threshold: 20 images" in texts
    # Validation: 3 identities with 20 images and 3 with 25; test: 4 and 2.
    assert "CelebA validation: 6 with 20 or more" in texts
    assert "CelebA test: 6 with 20 or more" in texts


def test_lfw_pairs_are_in_view_order_with_their_table() -> None:
    fig = _render("lfw-pair-composition", eda_summary())
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == [
        "pairsDevTrain\nView 1, 1 fold",
        "pairsDevTest\nView 1, 1 fold",
        "pairs\nView 2, 10 folds",
    ]
    texts = _texts(fig)
    assert "Pairs with an\nexcluded image" in texts
    assert "matched" in texts
    assert "mismatched" in texts


def test_lfw_long_tail_names_its_ends() -> None:
    texts = _texts(_render("lfw-images-per-identity", eda_summary()))
    assert "45 identities, 112 images, median 1 each" in texts
    assert "15 with 2 or more, enough for a matched pair" in texts
    assert "1 identity has 40" in texts


def test_head_pose_medians_use_a_true_minus_sign() -> None:
    texts = _texts(_render("head-pose", eda_summary()))
    assert "median \N{MINUS SIGN}4°" in texts
    assert "median 0°" in texts  # -0.2 rounds to an unsigned zero


def test_save_eda_figures_writes_one_svg_per_slug_in_order(tmp_path: Path) -> None:
    written = save_eda_figures(eda_summary(), tmp_path / "figures")
    assert written == [tmp_path / "figures" / f"{slug}.svg" for slug in SLUGS]
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == sorted(
        f"{slug}.svg" for slug in SLUGS
    )
    for path in written:
        svg = path.read_text()
        assert svg.lstrip().startswith("<?xml")
        assert "<dc:date>" not in svg


def test_saved_figures_are_byte_identical_across_runs(tmp_path: Path) -> None:
    first = save_eda_figures(eda_summary(), tmp_path / "first")
    second = save_eda_figures(eda_summary(), tmp_path / "second")
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name


@pytest.mark.parametrize(
    ("value", "top"), [(0, 1.0), (0.7, 0.8), (6.6, 8.0), (96, 100.0), (660, 800.0)]
)
def test_nice_top_rounds_up_to_a_tick(value: float, top: float) -> None:
    assert _nice_top(value) == top


@pytest.mark.parametrize(
    ("value", "text"), [(-4.2, "\N{MINUS SIGN}4"), (-0.3, "0"), (0, "0"), (12.6, "+13")]
)
def test_signed_numbers(value: float, text: str) -> None:
    assert _signed(value) == text
