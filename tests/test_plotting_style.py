import itertools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import colors, font_manager
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

from ryuk.plotting import style
from ryuk.plotting.style import (
    COLUMN_WIDTH,
    DATASET_STYLES,
    FONT_FILES,
    MODEL_STYLES,
    SeriesStyle,
    caption,
    direct_label,
    font_path,
    lab_style,
    new_figure,
    panel_title,
    register_fonts,
    save_figure,
    use_lab_style,
)


def _lightness(colour: str) -> float:
    """Relative luminance, as WCAG defines it: what tells colours apart without hue."""
    rgb = np.array(colors.to_rgb(colour))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ [0.2126, 0.7152, 0.0722])


def _dash_pattern(series: SeriesStyle) -> tuple[float, ...]:
    """The on-off lengths of a series' line: empty for solid."""
    if series.dashes == "-":
        return ()
    assert isinstance(series.dashes, tuple)
    _offset, pattern = series.dashes
    return tuple(float(length) for length in pattern)


def _captions(fig: Figure) -> list[Text]:
    return [t for t in fig.texts if t.get_gid() == "caption"]


# Palette and series styles, as #13 decided them.


def test_palette_is_13s() -> None:
    assert (style.PAPER, style.RAISED, style.INK, style.MUTED) == (
        "#F4F1E8",
        "#FBFAF6",
        "#1C1B19",
        "#5E5A52",
    )
    assert (style.HAIRLINE, style.INPUT_BORDER) == ("#D8D2C4", "#BDB5A3")
    assert (style.MATCH, style.NO_MATCH) == ("#1F4E8C", "#A8490F")
    assert (style.OK, style.WARNING, style.DESTRUCTIVE) == ("#2F6B3A", "#7A5600", "#8E3D0C")


def test_model_styles_are_13s_colours_names_and_dashes() -> None:
    assert list(MODEL_STYLES) == ["arcface", "facenet", "sface"]
    assert [(s.name, s.colour) for s in MODEL_STYLES.values()] == [
        ("ArcFace", "#1F4E8C"),
        ("FaceNet", "#8A4FA0"),
        ("SFace", "#C98A1B"),
    ]
    arcface, facenet, sface = (_dash_pattern(s) for s in MODEL_STYLES.values())
    assert arcface == ()  # solid
    assert facenet[0] > 2 * facenet[1]  # long dash: long strokes, short gaps
    assert sface[0] < sface[1]  # dotted: short strokes, longer gaps


@pytest.mark.parametrize("styles", [MODEL_STYLES, DATASET_STYLES], ids=["models", "datasets"])
def test_series_differ_in_lightness_and_dash_not_hue_alone(
    styles: Mapping[str, SeriesStyle],
) -> None:
    series = list(styles.values())
    lightness = sorted(_lightness(s.colour) for s in series)
    assert all(b - a > 0.05 for a, b in itertools.pairwise(lightness))
    assert len({_dash_pattern(s) for s in series}) == len(series)


def test_celeba_draws_also_differ_in_marker_fill() -> None:
    assert DATASET_STYLES["validation"].filled != DATASET_STYLES["test"].filled
    hollow = DATASET_STYLES["validation"].points()
    assert hollow["markerfacecolor"] == "none"
    assert DATASET_STYLES["test"].points()["markerfacecolor"] == DATASET_STYLES["test"].colour


def test_series_style_gives_plot_keywords() -> None:
    facenet = MODEL_STYLES["facenet"]
    with lab_style():
        fig = new_figure(2)
        ax = fig.add_subplot()
        (line,) = ax.plot([0, 1], [0, 1], **facenet.line())
        (dots,) = ax.plot([0, 1], [0, 1], **facenet.points())
    assert line.get_label() == "FaceNet"
    assert colors.same_color(line.get_color(), "#8A4FA0")
    assert dots.get_linestyle() == "None"
    assert dots.get_marker() == "s"


# Fonts.


def test_fonts_register_once() -> None:
    register_fonts()
    register_fonts()
    vendored = {str(font_path(name)) for name in FONT_FILES}
    entries = [e for e in font_manager.fontManager.ttflist if e.fname in vendored]
    assert {e.fname for e in entries} == vendored
    # Registering again adds nothing: each file is in the list once, under its family name and
    # any alternative names matplotlib derives from it.
    assert len(entries) == len({(e.fname, e.name, e.weight) for e in entries})


@pytest.mark.parametrize(
    ("family", "weight", "font_style", "filename"),
    [
        ("Public Sans", "normal", "normal", "PublicSans-Regular.ttf"),
        ("Public Sans", "semibold", "normal", "PublicSans-SemiBold.ttf"),
        ("Public Sans", "normal", "italic", "PublicSans-Italic.ttf"),
        ("Newsreader 16pt", "normal", "normal", "Newsreader16pt-Regular.ttf"),
        ("Newsreader 16pt", "semibold", "normal", "Newsreader16pt-SemiBold.ttf"),
        ("Newsreader 16pt", "normal", "italic", "Newsreader16pt-Italic.ttf"),
    ],
)
def test_fonts_resolve_to_the_vendored_files(
    family: str, weight: str, font_style: Literal["normal", "italic"], filename: str
) -> None:
    register_fonts()
    found = font_manager.findfont(
        FontProperties(family=family, weight=weight, style=font_style),
        fallback_to_default=False,
    )
    assert Path(found) == font_path(filename)
    assert font_path(filename).parent.name == "fonts"


def test_the_style_resolves_generic_families_to_the_vendored_fonts() -> None:
    with lab_style():
        sans = font_manager.findfont(
            FontProperties(family=["sans-serif"]), fallback_to_default=False
        )
        serif = font_manager.findfont(FontProperties(family=["serif"]), fallback_to_default=False)
    assert Path(sans) == font_path("PublicSans-Regular.ttf")
    assert Path(serif) == font_path("Newsreader16pt-Regular.ttf")


# The style's rcParams.


def test_lab_style_applies_within_the_block_only() -> None:
    before = mpl.rcParams["axes.spines.top"]
    with lab_style():
        assert mpl.rcParams["svg.fonttype"] == "path"
        assert mpl.rcParams["svg.hashsalt"] == "ryuk"
        assert not mpl.rcParams["axes.spines.top"]
        assert not mpl.rcParams["axes.spines.right"]
        assert mpl.rcParams["axes.grid"]
        assert colors.same_color(mpl.rcParams["axes.edgecolor"], style.INK)
        assert colors.same_color(mpl.rcParams["figure.facecolor"], style.PAPER)
        assert mpl.rcParams["font.sans-serif"][0] == "Public Sans"
        assert mpl.rcParams["font.serif"][0] == "Newsreader 16pt"
    assert mpl.rcParams["axes.spines.top"] == before


def test_use_lab_style_applies_for_the_session() -> None:
    with mpl.rc_context():
        use_lab_style()
        assert mpl.rcParams["svg.hashsalt"] == "ryuk"


def test_new_figure_is_a_report_column_wide_with_constrained_layout() -> None:
    with lab_style():
        fig = new_figure(2.5)
    assert tuple(fig.get_size_inches()) == (COLUMN_WIDTH, 2.5)
    assert fig.get_layout_engine() is not None
    assert type(fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"


def test_panel_title_is_serif_and_left_aligned() -> None:
    with lab_style():
        ax = new_figure(2).add_subplot()
        title = panel_title(ax, "Yaw")
    assert title.get_text() == "Yaw"
    assert title.get_fontfamily() == ["serif"]
    assert ax.get_title(loc="left") == "Yaw"


# Direct labels.


def test_direct_label_sits_at_a_lines_last_finite_point_in_its_colour() -> None:
    with lab_style():
        ax = new_figure(2).add_subplot()
        (line,) = ax.plot([0, 1, 2, 3], [5, 6, 7, np.nan], **MODEL_STYLES["sface"].line())
        label = direct_label(ax, line, "SFace")
    assert label.xy == (2.0, 7.0)
    assert colors.same_color(label.get_color(), "#C98A1B")
    assert label.get_horizontalalignment() == "left"
    assert ax.get_legend() is None


def test_direct_label_at_a_point_defaults_to_ink_and_takes_alignment() -> None:
    with lab_style():
        ax = new_figure(2).add_subplot()
        label = direct_label(ax, (1.0, 2.0), "note", offset=(0, 5), ha="center", va="bottom")
    assert label.xy == (1.0, 2.0)
    assert colors.same_color(label.get_color(), style.INK)
    assert label.get_horizontalalignment() == "center"
    assert label.get_verticalalignment() == "bottom"


def test_direct_label_refuses_a_line_with_nothing_to_label() -> None:
    with lab_style():
        ax = new_figure(2).add_subplot()
        (line,) = ax.plot([np.nan], [np.nan])
        with pytest.raises(ValueError, match="no finite points"):
            direct_label(ax, line, "empty")


# Captions.


def test_caption_is_a_serif_figure_number_and_text_below_the_plot() -> None:
    with lab_style():
        fig = new_figure(2)
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("x")
        caption(fig, 3, "Detected face size per dataset.")
        fig.draw_without_rendering()
        number, body = _captions(fig)
        plot_box = ax.get_tightbbox()
        assert plot_box is not None
        plot_bottom = plot_box.y0
        caption_top = max(number.get_window_extent().y1, body.get_window_extent().y1)

    assert number.get_text() == "Figure 3."
    assert number.get_fontweight() in {"semibold", 600}
    assert body.get_text() == "Detected face size per dataset."
    assert number.get_fontfamily() == body.get_fontfamily() == ["Newsreader 16pt"]
    assert caption_top < plot_bottom
    assert fig.get_size_inches()[1] > 2


def test_caption_keeps_the_plot_its_size() -> None:
    with lab_style():
        plain = new_figure(2)
        plain.add_subplot().plot([0, 1], [0, 1])
        captioned = new_figure(2)
        captioned.add_subplot().plot([0, 1], [0, 1])
        caption(captioned, 1, "A caption.")
        plain.draw_without_rendering()
        captioned.draw_without_rendering()
    plain_height = plain.axes[0].get_window_extent().height
    captioned_height = captioned.axes[0].get_window_extent().height
    assert captioned_height == pytest.approx(plain_height, rel=0.02)


def test_a_long_caption_wraps_under_its_first_word() -> None:
    text = " ".join(["Images per identity in each draw, with the gallery threshold marked."] * 4)
    with lab_style():
        fig = new_figure(2)
        fig.add_subplot()
        caption(fig, 2, text)
    number, body = _captions(fig)
    lines = body.get_text().split("\n")
    assert len(lines) > 1
    assert " ".join(lines) == text
    assert body.get_position()[0] > number.get_position()[0]


def test_a_figure_takes_one_caption() -> None:
    with lab_style():
        fig = new_figure(2)
        fig.add_subplot()
        caption(fig, 1, "First.")
        with pytest.raises(ValueError, match="already has a caption"):
            caption(fig, 2, "Second.")


def test_caption_moves_the_axes_of_a_figure_without_a_layout_engine() -> None:
    with lab_style():
        fig = Figure(figsize=(4, 2))
        ax = fig.add_subplot()
        # In inches: the axes' height, and their top's distance from the figure's top.
        height_before = ax.get_position().height * 2
        gap_before = (1 - ax.get_position().y1) * 2
        caption(fig, 1, "Caption.")
    height = fig.get_size_inches()[1]
    assert height > 2
    assert ax.get_position().height * height == pytest.approx(height_before)
    assert (1 - ax.get_position().y1) * height == pytest.approx(gap_before)


def test_caption_refuses_other_layout_engines() -> None:
    with lab_style():
        fig = Figure(figsize=(4, 2), layout="tight")
        fig.add_subplot()
        with pytest.raises(ValueError, match="constrained layout"):
            caption(fig, 1, "Caption.")


# Saving.


def _svg(tmp_path: Path, name: str) -> bytes:
    with lab_style():
        fig = new_figure(2)
        ax = fig.add_subplot()
        (line,) = ax.plot([0, 1, 2], [0, 1, 4], **MODEL_STYLES["arcface"].line())
        direct_label(ax, line, "ArcFace")
    return save_figure(fig, tmp_path / name).read_bytes()


def test_saved_svgs_are_reproducible_transparent_and_font_free(tmp_path: Path) -> None:
    first, second = _svg(tmp_path, "a.svg"), _svg(tmp_path, "b.svg")
    assert first == second
    svg = first.decode()
    assert "<dc:date>" not in svg
    assert "<text" not in svg  # glyphs are paths, so the SVG needs no fonts
    assert not re.search(r"fill:\s*#f4f1e8", svg, re.IGNORECASE)  # no paper background


@pytest.mark.parametrize("suffix", [".pdf", ".png"])
def test_other_formats_save_reproducibly(tmp_path: Path, suffix: str) -> None:
    first = save_figure(_figure(), tmp_path / f"a{suffix}").read_bytes()
    second = save_figure(_figure(), tmp_path / f"b{suffix}").read_bytes()
    assert first == second


def _figure() -> Figure:
    with lab_style():
        fig = new_figure(1.5)
        fig.add_subplot().plot([0, 1], [1, 0])
    return fig
