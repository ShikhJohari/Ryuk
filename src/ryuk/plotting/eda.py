"""The dataset analysis figures (#17), each drawn from the committed `eda/summary.json` alone.

`EDA_FIGURES` names every figure by a stable slug, which the notebook and the report use to
refer to it, and `save_eda_figures` writes them all as `<slug>.svg`. No figure has a caption
or a "Figure N" title: the report numbers and captions its own, and a notebook adds one with
`ryuk.plotting.caption`.

Validation and test draws are drawn at the same place wherever they share an axis, so their
series often coincide; rows of dots are set slightly apart by draw, and labels that would
collide are stacked.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    StrMethodFormatter,
)
from numpy.typing import NDArray

from ryuk.eda.summary import (
    ATTRIBUTES,
    IDENTITY_ATTRIBUTES,
    PHOTO_CONDITIONS,
    Attribute,
    AttributePrevalence,
    CelebaDraw,
    DetectionStats,
    Distribution,
    EdaSummary,
    GroupStats,
    ImagesPerIdentity,
    PairList,
)
from ryuk.plotting.style import (
    DATASET_STYLES,
    HAIRLINE,
    INK,
    MUTED,
    PAPER,
    Dataset,
    direct_label,
    lab_style,
    new_figure,
    panel_title,
    save_figure,
)

_COUNT: Final = StrMethodFormatter("{x:,.0f}")
_PERCENT: Final = StrMethodFormatter("{x:.0f}%")
_FINE_PERCENT: Final = StrMethodFormatter("{x:.1f}%")

_SMALL: Final = 7.0
"""Type size for in-plot annotations, in points; tick labels are 7.5 and axis labels 8.5."""

_PALE: Final = "#E6E0D2"
"""A bar segment that is the complement of an ink one: without the attribute, or mismatched."""

_GROUP_NAMES: Final[Mapping[Attribute, tuple[str, str]]] = MappingProxyType(
    {
        "Male": ("Male", "Not male"),
        "Young": ("Young", "Not young"),
        "Eyeglasses": ("Eyeglasses", "No eyeglasses"),
        "Wearing_Hat": ("Wearing hat", "No hat"),
        "Blurry": ("Blurry", "Not blurry"),
    }
)
"""Row labels for #10's groups: the attribute's name when true, then when false."""

_PAIR_LISTS: Final = ("pairsDevTrain", "pairsDevTest", "pairs")
"""LFW's pairs files in reading order: View 1's training and test lists, then View 2's."""

_ROW_OFFSET: Final[Mapping[Dataset, float]] = MappingProxyType(
    {"lfw": 0.0, "validation": -0.17, "test": 0.17}
)
"""How far each dataset's dot sits from its row's centre, in rows, so equal values stay apart."""


# Shared pieces ---------------------------------------------------------------------------------


def _share(part: int, whole: int) -> float | None:
    """`part` of `whole` as a percentage, or None when there is nothing to take a share of."""
    return 100 * part / whole if whole else None


def _short_name(dataset: Dataset) -> str:
    """A dataset's name where the context already says CelebA: "validation" or "test"."""
    return DATASET_STYLES[dataset].name.removeprefix("CelebA ")


def _no_data(ax: Axes, message: str) -> None:
    """Mark a panel that has nothing to draw, instead of leaving empty axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=MUTED)


def _rows_unruled(ax: Axes) -> None:
    """A dot or bar plot's look: rows unruled, a light grid along the value axis only."""
    ax.grid(visible=False)
    ax.grid(visible=True, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _nice_top(value: float) -> float:
    """The first round tick at or above `value`, for an axis that starts at zero."""
    if value <= 0:
        return 1.0
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, value)
    return float(next(t for t in ticks if t >= value))


def signed(value: float, _pos: int | None = None) -> str:
    """A whole number with its sign, a true minus sign (U+2212) when negative; zero unsigned.

    Also a tick formatter, hence `_pos`. The report's prose uses it for the same angles.
    """
    rounded = round(value)
    if rounded == 0:
        return "0"
    return f"{rounded:+d}".replace("-", "\N{MINUS SIGN}")


@dataclass(frozen=True, slots=True)
class _Label:
    """A direct label to place at a point, possibly nudged to clear its neighbours."""

    xy: tuple[float, float]
    text: str
    colour: str


def _stacked_labels(ax: Axes, labels: Sequence[_Label], *, gap: float = 9.0) -> None:
    """Label points such as curve peaks just above each, stacking upward any that would overlap.

    Points left of the middle of the group are labelled on their left, the rest on their right,
    so labels lean away from their neighbours' curves. Two labels on one side overlap when they
    are within `gap` points vertically and 90 points horizontally. Call after the axes limits are
    final: positions are compared on screen.
    """
    if not labels:
        return
    fig = ax.get_figure(root=True)
    if fig is None:
        raise ValueError("the axes must belong to a figure")
    to_points = 72 / fig.dpi
    xs = [label.xy[0] for label in labels]
    middle = (min(xs) + max(xs)) / 2
    placed: dict[bool, list[tuple[float, float]]] = {True: [], False: []}
    for label in sorted(labels, key=lambda lab: lab.xy[1]):
        x, y = (float(v) * to_points for v in ax.transData.transform(label.xy))
        on_left = label.xy[0] < middle
        lift = 3.0
        for px, py in placed[on_left]:
            if abs(px - x) < 90 and abs(py - (y + lift)) < gap:
                lift = py - y + gap
        placed[on_left].append((x, y + lift))
        direct_label(
            ax,
            label.xy,
            label.text,
            colour=label.colour,
            offset=(-3 if on_left else 3, lift),
            ha="right" if on_left else "left",
            va="bottom",
            fontsize=_SMALL,
        )


def _at_least(table: ImagesPerIdentity) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """For each image count k in the table, how many identities have at least k images."""
    ks = np.array([row.images for row in table.table], dtype=float)
    counts = np.array([row.identities for row in table.table], dtype=float)
    return ks, np.cumsum(counts[::-1])[::-1]


def _found[T, *Args](lookup: Callable[[*Args], T], *args: *Args) -> T | None:
    """What `lookup(*args)` finds in the summary, or None if it is not there: a figure also
    draws a summary missing some of its attributes or groups, leaving them out."""
    try:
        return lookup(*args)
    except KeyError:
        return None


# LFW ------------------------------------------------------------------------------------------


def lfw_images_per_identity(summary: EdaSummary) -> Figure:
    """LFW's long tail: how many identities have at least k images, on log axes."""
    ipi = summary.lfw.images_per_identity
    fig = new_figure(2.7)
    ax = fig.add_subplot()
    if not ipi.table:
        _no_data(ax, "No LFW identities in the summary")
        return fig

    ks, at_least = _at_least(ipi)
    # One series alone needs no dash pattern to tell it apart, so LFW is drawn solid here. The
    # last step runs on past the largest count so the most photographed identity has a width.
    ax.plot(
        np.append(ks, ks[-1] * 1.3),
        np.append(at_least, at_least[-1]),
        drawstyle="steps-post",
        color=DATASET_STYLES["lfw"].colour,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_major_formatter(_COUNT)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.8, ks[-1] * 1.7)
    ax.set_ylim(0.5, at_least[0] * 3)
    ax.set_xlabel("Images per identity, k")
    ax.set_ylabel("Identities with k or more images")

    first, total = int(ks[0]), int(at_least[0])
    median = f", median {ipi.median:g} each" if ipi.median is not None else ""
    marks = [((first, total), f"{total:,} identities, {ipi.images:,} images{median}", (5, 3))]
    if first < 2 <= ks[-1]:
        two = ipi.at_least(2)
        marks.append(((2, two), f"{two:,} with 2 or more, enough for a matched pair", (5, 3)))
    if ipi.max is not None and ipi.max > max(first, 2):
        top = ipi.at_least(ipi.max)
        noun = "identity has" if top == 1 else "identities have"
        # Left of the last point: the curve there is a step higher, so the space is free.
        marks.append(((ipi.max, top), f"{top:,} {noun} {ipi.max:,}", (-6, 0)))
    for xy, text, offset in marks:
        ax.plot(*xy, marker="o", color=INK, markersize=3.5, linestyle="none")
        leftward = offset[0] < 0
        direct_label(
            ax,
            (float(xy[0]), float(xy[1])),
            text,
            offset=offset,
            ha="right" if leftward else "left",
            va="center" if leftward else "bottom",
        )
    return fig


def lfw_pair_composition(summary: EdaSummary) -> Figure:
    """LFW's pairs files: matched and mismatched pairs, and the identities and images they use."""
    order = {name: i for i, name in enumerate(_PAIR_LISTS)}
    pairs = sorted(summary.lfw.pairs, key=lambda p: (p.view, order[p.name]))
    fig = new_figure(1.1 + 0.42 * max(len(pairs), 1))
    bars, table = fig.subplots(1, 2, sharey=True, width_ratios=(1.9, 1.2))
    if not pairs:
        _no_data(bars, "No LFW pairs files in the summary")
        _no_data(table, "")
        return fig

    rows = np.arange(len(pairs))
    matched = np.array([p.matched for p in pairs], dtype=float)
    mismatched = np.array([p.mismatched for p in pairs], dtype=float)
    widest = float(max((matched + mismatched).max(), 1.0))
    # Neutral, as the attribute bars' with and without are: a pair's ground truth is neither a
    # match nor a no match, which are a threshold's outcomes, so their colours stay for those.
    bars.barh(rows, matched, height=0.62, color=INK, linewidth=0)
    bars.barh(rows, mismatched, left=matched, height=0.62, color=_PALE, linewidth=0)
    for row, pair in enumerate(pairs):
        _segment_count(bars, row, (0, pair.matched), PAPER, widest)
        _segment_count(bars, row, (pair.matched, pair.mismatched), INK, widest)
    names = (
        (0.0, float(matched[0]), "matched"),
        (float(matched[0]), float(mismatched[0]), "mismatched"),
    )
    for left, width, name in names:
        if width:
            bars.text(
                left + width / 2,
                -0.4,
                name,
                ha="center",
                va="bottom",
                color=MUTED,
                fontsize=_SMALL,
            )
    bars.set_yticks(rows, [_pair_list_label(p) for p in pairs])
    bars.set_ylim(len(pairs) - 0.5, _TABLE_TOP - 0.05)
    bars.set_xlim(0, _nice_top(widest))
    bars.xaxis.set_major_formatter(_COUNT)
    bars.set_xlabel("Pairs")
    _rows_unruled(bars)
    _pair_table(table, pairs)
    return fig


_TABLE_TOP: Final = -1.3
"""Where the pairs table's top rule sits, in rows: above its two-line headings."""


def _pair_list_label(pair: PairList) -> str:
    folds = "1 fold" if pair.folds == 1 else f"{pair.folds} folds"
    return f"{pair.name}\nView {pair.view}, {folds}"


def _segment_count(
    ax: Axes, row: float, segment: tuple[float, float], colour: str, widest: float
) -> None:
    """Write a bar segment's count inside it, if it is wide enough to hold it.

    `segment` is where the segment starts and how wide it is.
    """
    left, width = segment
    if width >= widest * 0.07:
        ax.text(
            left + widest * 0.012,
            row,
            f"{width:,.0f}",
            ha="left",
            va="center",
            color=colour,
            fontsize=_SMALL,
        )


def _pair_table(ax: Axes, pairs: Sequence[PairList]) -> None:
    """A booktabs-style table beside the bars: what each pairs file draws on."""
    columns = (
        ("Identities", 1.0, [f"{p.identities:,}" for p in pairs]),
        ("Images", 1.0, [f"{p.images:,}" for p in pairs]),
        ("Pairs with an\nexcluded image", 1.5, [f"{p.pairs_with_excluded_image:,}" for p in pairs]),
    )
    ax.set_xlim(0, sum(width for _, width, _ in columns))
    ax.axis("off")
    right = 0.0
    for heading, width, cells in columns:
        right += width
        x = right - 0.08
        ax.text(x, -0.62, heading, ha="right", va="bottom", color=MUTED, fontsize=_SMALL)
        for row, cell in enumerate(cells):
            ax.text(x, row, cell, ha="right", va="center", color=INK)
    # Heavy top and bottom rules and a light one under the headings.
    ax.axhline(_TABLE_TOP, color=INK, linewidth=0.8)
    ax.axhline(-0.5, color=INK, linewidth=0.4)
    ax.axhline(len(pairs) - 0.5, color=INK, linewidth=0.8)


# CelebA ---------------------------------------------------------------------------------------


def celeba_images_per_identity(summary: EdaSummary) -> Figure:
    """Images per identity in each draw, with #9's gallery threshold marked."""
    threshold = summary.rules.min_gallery_images
    fig = new_figure(2.7)
    ax = fig.add_subplot()
    draws = [d for d in summary.celeba if d.images_per_identity.table]
    if not draws:
        _no_data(ax, "No CelebA identities in the summary")
        return fig

    top, largest = 0.0, 1.0
    for draw in draws:
        ks, at_least = _at_least(draw.images_per_identity)
        # Identities with at least k images, for every k from 1 to one past the largest count.
        grid = np.arange(1, int(ks[-1]) + 2)
        counts = np.array([at_least[ks >= k][0] if (ks >= k).any() else 0.0 for k in grid])
        ax.plot(grid, counts, drawstyle="steps-post", **DATASET_STYLES[draw.draw].line())
        top = max(top, float(counts[0]))
        largest = max(largest, float(grid[-1]))

    ax.axvline(threshold, color=INK, linewidth=0.8)
    direct_label(
        ax,
        (threshold, top * 1.12),
        f"Gallery threshold: {threshold} images",
        offset=(4, 0),
        va="top",
    )
    # Each draw's gallery candidates where its line meets the threshold. The labels go under
    # the curves, left of the line, highest first, so they never overlap.
    crossings = sorted(
        ((d, d.images_per_identity.at_least(threshold)) for d in draws),
        key=lambda item: item[1],
        reverse=True,
    )
    lowest = min(n for _, n in crossings)
    for i, (draw, n) in enumerate(crossings):
        style = DATASET_STYLES[draw.draw]
        # Hollow markers on top, so a filled one never hides them.
        ax.plot([threshold], [n], zorder=3 if style.filled else 4, **style.points())
        direct_label(
            ax,
            (threshold, lowest),
            f"{style.name}: {n:,} with {threshold} or more",
            colour=style.colour,
            offset=(-7, -8 - 11 * i),
            ha="right",
            va="top",
        )
    ax.set_xlim(0.5, largest + 0.5)
    ax.set_ylim(0, top * 1.15)
    ax.yaxis.set_major_formatter(_COUNT)
    ax.set_xlabel("Images per identity, k")
    ax.set_ylabel("Identities with k or more images")
    return fig


def celeba_attribute_prevalence(summary: EdaSummary) -> Figure:
    """#17's attributes counted per identity, per draw.

    Male and Young belong to the person, so each identity is split by its majority label, with
    the identities short of a majority as mixed. Eyeglasses, hats and blur belong to the photo,
    so each is shown as the share of images labelled with it against the share of identities
    with at least one such image.
    """
    fig = new_figure(3.3)
    left, right = fig.subplots(1, 2, width_ratios=(1.05, 1))
    draws = list(summary.celeba)
    if not draws:
        _no_data(left, "No CelebA draws in the summary")
        _no_data(right, "")
        return fig
    _identity_attribute_bars(left, draws, summary.rules.majority_agreement)
    _photo_condition_dumbbells(right, draws)
    return fig


@dataclass(frozen=True, slots=True)
class _AttributeRow:
    """One row of the prevalence small multiple: attributes in blocks, draws within them."""

    attribute: Attribute
    draw: CelebaDraw
    prevalence: AttributePrevalence


def _attribute_rows(
    draws: Sequence[CelebaDraw], attributes: Sequence[Attribute]
) -> list[_AttributeRow]:
    return [
        _AttributeRow(attribute, draw, prevalence)
        for attribute in attributes
        for draw in draws
        if (prevalence := _found(draw.prevalence, attribute)) is not None
    ]


def _block_positions(rows: Sequence[_AttributeRow]) -> list[float]:
    """Row positions from the top, with a gap between attribute blocks for their headings."""
    positions: list[float] = []
    y = 0.0
    for i, row in enumerate(rows):
        if i and row.attribute != rows[i - 1].attribute:
            y += 1.0
        positions.append(y)
        y += 1.0
    return positions


def _block_axes(ax: Axes, rows: Sequence[_AttributeRow], positions: Sequence[float]) -> None:
    """Label each row with its draw and each block of rows with its attribute."""
    ax.set_yticks(list(positions), [_short_name(row.draw.draw) for row in rows])
    ax.set_ylim(positions[-1] + 0.6, -1.3)
    _rows_unruled(ax)
    ax.tick_params(axis="y", labelsize=_SMALL)
    for attribute in dict.fromkeys(row.attribute for row in rows):
        top = min(y for row, y in zip(rows, positions, strict=True) if row.attribute == attribute)
        ax.text(
            0,
            top - 0.62,
            _GROUP_NAMES[attribute][0],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            color=INK,
            fontweight="semibold",
        )


def _identity_attribute_bars(ax: Axes, draws: Sequence[CelebaDraw], agreement: float) -> None:
    rows = _attribute_rows(draws, IDENTITY_ATTRIBUTES)
    panel_title(ax, "Identity attributes, by majority label")
    if not rows:
        _no_data(ax, "No identity attributes in the summary")
        return
    positions = _block_positions(rows)
    segments: tuple[tuple[str, Callable[[AttributePrevalence], int], str, str], ...] = (
        ("with", lambda p: p.majority_with, INK, PAPER),
        ("mixed", lambda p: p.mixed, "#B9B09E", INK),
        ("without", lambda p: p.majority_without, _PALE, INK),
    )
    named: set[str] = set()
    for row, y in zip(rows, positions, strict=True):
        counts = [count(row.prevalence) for _, count, _, _ in segments]
        total = sum(counts)
        left = 0.0
        for (name, _, fill, text_colour), count in zip(segments, counts, strict=True):
            share = _share(count, total) or 0.0
            ax.barh(y, share, left=left, height=0.7, color=fill, linewidth=0)
            if share >= 8:
                ax.text(
                    left + 1.5,
                    y,
                    f"{count:,}",
                    ha="left",
                    va="center",
                    color=text_colour,
                    fontsize=_SMALL,
                )
                # Each segment is named once, over the first bar wide enough to hold the name.
                if name not in named:
                    ax.text(
                        left + share / 2,
                        y - 0.4,
                        name,
                        ha="center",
                        va="bottom",
                        color=MUTED,
                        fontsize=_SMALL,
                    )
                    named.add(name)
            left += share
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.xaxis.set_major_formatter(_PERCENT)
    ax.set_xlabel(f"Identities (majority: {agreement:.0%} of images agree)")
    _block_axes(ax, rows, positions)


def _photo_condition_dumbbells(ax: Axes, draws: Sequence[CelebaDraw]) -> None:
    rows = _attribute_rows(draws, PHOTO_CONDITIONS)
    panel_title(ax, "Photo conditions, per image and per identity")
    if not rows:
        _no_data(ax, "No photo conditions in the summary")
        return
    positions = _block_positions(rows)
    highest = 0.0
    ends: list[tuple[float, float, float]] = []
    for row, y in zip(rows, positions, strict=True):
        per_image = _share(row.prevalence.images_with, row.draw.images_per_identity.images)
        per_identity = _share(
            row.prevalence.identities_with_any, row.draw.images_per_identity.identities
        )
        if per_image is None or per_identity is None:
            continue
        style = DATASET_STYLES[row.draw.draw]
        ax.plot(
            [per_image, per_identity],
            [y, y],
            color=style.colour,
            linewidth=1.2,
            linestyle=style.dashes,
        )
        ax.plot([per_image], [y], **{**style.points(), "marker": "|", "markersize": 7})
        ax.plot([per_identity], [y], **style.points())
        highest = max(highest, per_image, per_identity)
        ends.append((per_image, per_identity, y))
    if ends:
        # The two ends are named once, over the first row drawn.
        per_image, per_identity, y = ends[0]
        for x, text in ((per_image, "of images"), (per_identity, "of identities, any image")):
            direct_label(
                ax,
                (x, y),
                text,
                colour=MUTED,
                offset=(0, 5),
                ha="center",
                va="bottom",
                fontsize=_SMALL,
            )
    ax.set_xlim(0, _nice_top(highest * 1.08))
    ax.xaxis.set_major_formatter(_PERCENT)
    ax.set_xlabel("Share with the condition")
    _block_axes(ax, rows, positions)


# Detection and exclusion by group ---------------------------------------------------------------

type _Stats = DetectionStats | GroupStats


@dataclass(frozen=True, slots=True)
class _GroupRow:
    """One row of the per-group dot plots: a label and each dataset's numbers for it."""

    label: str
    block: str
    """Rows in the same block are drawn together; a hairline separates blocks."""
    series: tuple[tuple[Dataset, CelebaDraw | None, _Stats], ...]
    """Per dataset, the draw (None for LFW) and its numbers for this row."""


def _group_rows(summary: EdaSummary) -> list[_GroupRow]:
    rows = [
        _GroupRow("All LFW", "lfw", (("lfw", None, summary.lfw.detection),)),
        _GroupRow("All CelebA", "celeba", tuple((d.draw, d, d.detection) for d in summary.celeba)),
    ]
    for attribute in ATTRIBUTES:
        for value in (True, False):
            series = tuple(
                (draw.draw, draw, group)
                for draw in summary.celeba
                if (group := _found(draw.group, attribute, value)) is not None
            )
            if series:
                rows.append(_GroupRow(_GROUP_NAMES[attribute][not value], attribute, series))
    return rows


type _RowValue = Callable[[CelebaDraw | None, _Stats], float | None]


def _dot_rows(ax: Axes, rows: Sequence[_GroupRow], value: _RowValue) -> list[float]:
    """Draw one dot per dataset per row at `value`, and return every value drawn."""
    drawn: list[float] = []
    for y, row in enumerate(rows):
        for dataset, draw, stats in row.series:
            v = value(draw, stats)
            if v is not None:
                ax.plot([v], [y + _ROW_OFFSET[dataset]], **DATASET_STYLES[dataset].points())
                drawn.append(v)
    return drawn


def _dot_axes(ax: Axes, rows: Sequence[_GroupRow]) -> None:
    ax.set_yticks(range(len(rows)), [row.label for row in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    _rows_unruled(ax)
    for y in range(1, len(rows)):
        if rows[y].block != rows[y - 1].block:
            ax.axhline(y - 0.5, color=HAIRLINE, linewidth=0.6)


def _row_note(ax: Axes, y: float, text: str) -> None:
    ax.text(
        0.015,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        color=MUTED,
        fontsize=_SMALL,
        fontstyle="italic",
    )


def _label_draws(ax: Axes, rows: Sequence[_GroupRow], value: _RowValue) -> None:
    """Name the draws once, beside their dots on the first CelebA row, instead of a legend.

    A label goes on whichever side of its dot has more room. Call after the limits are final.
    """
    row = next((i for i, r in enumerate(rows) if r.block == "celeba"), None)
    if row is None:
        return
    low, high = ax.get_xlim()
    for dataset, draw, stats in rows[row].series:
        x = value(draw, stats)
        if x is None:
            continue
        room_left = (x - low) > (high - low) / 2
        direct_label(
            ax,
            (x, row + _ROW_OFFSET[dataset]),
            _short_name(dataset),
            colour=DATASET_STYLES[dataset].colour,
            offset=(-6, 0) if room_left else (6, 0),
            ha="right" if room_left else "left",
            fontsize=_SMALL,
        )


def _zoomed_percent_axis(ax: Axes, values: Sequence[float]) -> None:
    """A percentage axis ending at 100%, starting on a round step just below the lowest value.

    Re-detection rates sit close to 100%, where a full 0-100% axis would hide every difference.
    """
    lowest = min(values, default=0.0)
    step = next((s for s in (0.5, 1, 2, 5, 10, 20) if 100 - lowest <= 5 * s), 25)
    start = max(0.0, math.floor((lowest - step * 0.3) / step) * step)
    ax.set_xlim(start, 100 + step * 0.1)
    ax.xaxis.set_major_locator(FixedLocator(list(np.arange(start, 100 + step / 2, step))))
    ax.xaxis.set_major_formatter(_FINE_PERCENT if step < 1 else _PERCENT)


def _group_figure_height(rows: Sequence[_GroupRow]) -> float:
    return 0.85 + 0.24 * len(rows)


def yunet_detection_rate(summary: EdaSummary) -> Figure:
    """YuNet's re-detection rate, the share of images with any face found, per draw and group."""
    rows = _group_rows(summary)
    fig = new_figure(_group_figure_height(rows))
    ax = fig.add_subplot()

    def rate(_draw: CelebaDraw | None, stats: _Stats) -> float | None:
        return _share(stats.detected, stats.images)

    drawn = _dot_rows(ax, rows, rate)
    _dot_axes(ax, rows)
    for y, row in enumerate(rows):
        if all(stats.images == 0 for _, _, stats in row.series):
            _row_note(ax, y, "no images")
    _zoomed_percent_axis(ax, drawn)
    ax.set_xlabel("Images with a face detected")
    _label_draws(ax, rows, rate)
    return fig


def exclusions(summary: EdaSummary) -> Figure:
    """Images excluded for having no usable face, and the eligible identities that remain."""
    rows = _group_rows(summary)
    fig = new_figure(_group_figure_height(rows) + 0.2)
    left, right = fig.subplots(1, 2, sharey=True)

    def excluded(_draw: CelebaDraw | None, stats: _Stats) -> float | None:
        return _share(stats.images - stats.usable, stats.images)

    panel_title(left, "Images excluded")
    drawn = _dot_rows(left, rows, excluded)
    _dot_axes(left, rows)
    worst = max(drawn, default=0.0)
    left.set_xlim(0, _nice_top(worst * 1.1))
    left.xaxis.set_major_formatter(_FINE_PERCENT if worst < 2 else _PERCENT)
    left.set_xlabel("Images with no usable face")
    for y, row in enumerate(rows):
        if all(stats.images == 0 for _, _, stats in row.series):
            _row_note(left, y, "no images")
    _label_draws(left, rows, excluded)

    def eligible(draw: CelebaDraw | None, stats: _Stats) -> float | None:
        if isinstance(stats, GroupStats):
            return None if stats.eligible_identities is None else float(stats.eligible_identities)
        return None if draw is None else float(draw.eligible_identities)

    panel_title(right, "Eligible identities")
    counts = _dot_rows(right, rows, eligible)
    _dot_axes(right, rows)
    right.tick_params(axis="y", labelleft=False)
    right.set_xlim(0, _nice_top(max(counts, default=0.0) * 1.1))
    right.xaxis.set_major_formatter(_COUNT)
    right.set_xlabel(f"Identities with {summary.rules.min_gallery_images}+ usable images")
    for y, row in enumerate(rows):
        first_of_block = y == 0 or rows[y - 1].block != row.block
        if row.block == "lfw":
            _row_note(right, y, "LFW has no gallery")
        elif row.block in PHOTO_CONDITIONS and first_of_block:
            _row_note(right, y + 0.5, "a photo condition groups images, not identities")
    _label_draws(right, rows, eligible)
    return fig


# Face size and pose ----------------------------------------------------------------------------


def _datasets(summary: EdaSummary) -> list[tuple[Dataset, DetectionStats]]:
    return [("lfw", summary.lfw.detection), *((d.draw, d.detection) for d in summary.celeba)]


def _histogram(distribution: Distribution) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    edges = np.asarray(distribution.histogram.edges, dtype=float)
    counts = np.asarray(distribution.histogram.counts, dtype=float)
    if counts.size != edges.size - 1:
        raise ValueError(f"a histogram with {edges.size} edges needs {edges.size - 1} counts")
    return edges, counts


def _density(distribution: Distribution) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """A histogram as step-plot points: the percentage of all values per unit, at each edge."""
    if distribution.n == 0:
        return None
    edges, counts = _histogram(distribution)
    density = 100 * counts / (distribution.n * np.diff(edges))
    return edges, np.append(density, density[-1])


def _cdf(distribution: Distribution) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """At each histogram edge, the percentage of all values below it.

    Values below the first edge are not binned, so this reads low by their share; the face
    size histogram starts at zero, where there are none.
    """
    if distribution.n == 0:
        return None
    edges, counts = _histogram(distribution)
    return edges, 100 * np.concatenate(([0.0], np.cumsum(counts))) / distribution.n


def face_size(summary: EdaSummary) -> Figure:
    """Detected face size per dataset, and the low end where the minimum usable size is set."""
    rule = summary.min_usable_face_size
    fig = new_figure(2.8)
    whole, low = fig.subplots(1, 2, width_ratios=(1.25, 1))
    datasets = [(key, stats.face_short_side) for key, stats in _datasets(summary)]
    datasets = [(key, d) for key, d in datasets if d.n]
    if not datasets:
        _no_data(whole, "No detections in the summary")
        _no_data(low, "")
        return fig

    panel_title(whole, "Distribution")
    peaks: list[_Label] = []
    for key, distribution in datasets:
        points = _density(distribution)
        if points is None:  # pragma: no cover - datasets with no detections are filtered above
            continue
        style = DATASET_STYLES[key]
        edges, density = points
        whole.plot(edges, density, drawstyle="steps-post", **style.line())
        i = int(np.argmax(density[:-1]))
        peaks.append(
            _Label((float(edges[i : i + 2].mean()), float(density[i])), style.name, style.colour)
        )
    # The bulk of the faces: up to past the largest 99th percentile, so rare giants do not
    # squeeze the peaks. The rule is a muted line here, since the peaks' labels may cross it.
    right = max(d.quantiles.p99 for _, d in datasets if d.quantiles is not None)
    whole.set_xlim(0, max(right * 1.25, rule.value * 1.5))
    whole.set_ylim(0, whole.get_ylim()[1] * 1.25)
    whole.axvline(rule.value, color=MUTED, linewidth=0.7, zorder=1)
    direct_label(
        whole,
        (rule.value, whole.get_ylim()[1]),
        f"{rule.value} px",
        colour=MUTED,
        offset=(3, -2),
        va="top",
        fontsize=_SMALL,
    )
    _stacked_labels(whole, peaks)
    whole.set_xlabel("Face box short side (px)")
    whole.set_ylabel("Detections (% per px)")
    whole.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))

    panel_title(low, "Low end, cumulative")
    lost = 100 * (1 - rule.keep)
    ceiling = max(5.0, lost * 5)
    for key, distribution in datasets:
        points = _cdf(distribution)
        if points is not None:
            low.plot(*points, **DATASET_STYLES[key].line())
    low.axvline(rule.value, color=INK, linewidth=0.8)
    low.axhline(lost, color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)))
    direct_label(
        low,
        (rule.value, ceiling),
        f"Minimum usable face,\n{rule.value} px, keeps {rule.kept:.1%}\nof CelebA detections",
        offset=(-5, -3),
        ha="right",
        va="top",
        fontsize=_SMALL,
        linespacing=1.3,
    )
    # Around the rule: from half the minimum usable face size to half as much again above it.
    low.set_xlim(rule.value * 0.5, rule.value * 1.5)
    direct_label(
        low,
        (rule.value * 0.5, lost),
        f"{lost:g}%, the {rule.keep:.0%} rule",
        colour=MUTED,
        offset=(3, 2),
        va="bottom",
        fontsize=_SMALL,
    )
    low.set_ylim(0, ceiling)
    low.yaxis.set_major_formatter(_PERCENT)
    low.set_xlabel("Face box short side (px)")
    low.set_ylabel("Detections smaller")
    return fig


def head_pose(summary: EdaSummary) -> Figure:
    """The rough landmark head pose, yaw, pitch and roll, as small multiples per dataset."""
    datasets = _datasets(summary)
    fig = new_figure(0.75 + 0.95 * len(datasets))
    grid = fig.subplots(len(datasets), 3, sharex="col", squeeze=False)
    for r, (key, stats) in enumerate(datasets):
        style = DATASET_STYLES[key]
        for c, angle in enumerate(("yaw", "pitch", "roll")):
            ax: Axes = grid[r][c]
            if r == 0:
                panel_title(ax, angle.capitalize())
            if c == 0:
                ax.set_ylabel(
                    style.name.replace("CelebA ", "CelebA\n"),
                    rotation=0,
                    ha="right",
                    va="center",
                    color=style.colour,
                )
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            distribution: Distribution = getattr(stats.head_pose, angle)
            _pose_panel(ax, distribution, style.colour)
    for ax in grid[-1]:
        ax.set_xlabel("Degrees")
        ax.xaxis.set_major_formatter(FuncFormatter(signed))
    return fig


def _pose_panel(ax: Axes, distribution: Distribution, colour: str) -> None:
    points = _density(distribution)
    if points is None:
        ax.text(
            0.5,
            0.5,
            "no detections",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=_SMALL,
        )
        return
    edges, density = points
    peak = float(density.max()) or 1.0
    ax.fill_between(edges, density, step="post", color=colour, alpha=0.16, linewidth=0)
    ax.plot(edges, density, drawstyle="steps-post", color=colour, linewidth=1.0)
    ax.set_ylim(0, peak * 1.4)
    if distribution.quantiles is not None:
        median = distribution.quantiles.p50
        ax.axvline(median, color=INK, linewidth=0.7)
        direct_label(
            ax,
            (median, peak * 1.38),
            f"median {signed(median)}°",
            offset=(3, 0),
            va="top",
            fontsize=_SMALL,
        )


# Registry -------------------------------------------------------------------------------------

EDA_FIGURES: Final[Mapping[str, Callable[[EdaSummary], Figure]]] = MappingProxyType(
    {
        "lfw-images-per-identity": lfw_images_per_identity,
        "lfw-pair-composition": lfw_pair_composition,
        "celeba-images-per-identity": celeba_images_per_identity,
        "celeba-attribute-prevalence": celeba_attribute_prevalence,
        "yunet-detection-rate": yunet_detection_rate,
        "face-size": face_size,
        "head-pose": head_pose,
        "exclusions": exclusions,
    }
)
"""Every EDA figure by its slug, in the order the report shows them. The slugs are stable: the
notebook, the report and the committed file names refer to them."""


def eda_figure(slug: str, summary: EdaSummary) -> Figure:
    """The EDA figure `slug`, drawn in the lab-notebook style."""
    with lab_style():
        return EDA_FIGURES[slug](summary)


def save_eda_figures(summary: EdaSummary, directory: Path) -> list[Path]:
    """Draw every EDA figure and write it to `directory` as `<slug>.svg`, in registry order.

    The files are reproducible: the same summary always gives the same bytes. Each is renamed
    into place once whole, as `save_figure` writes it.
    """
    written = []
    for slug in EDA_FIGURES:
        fig = eda_figure(slug, summary)
        try:
            written.append(save_figure(fig, directory / f"{slug}.svg"))
        finally:
            # Figures made without pyplot are not tracked by it; clearing one is how it closes.
            fig.clear()
    return written
