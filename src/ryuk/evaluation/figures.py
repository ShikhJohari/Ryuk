"""The evaluation figures: the LFW ROC, every recognition model's TAR against FAR on View 2, and
the CelebA TPIR against FPIR on a draw, both on a log false-alarm axis; then learning on
embeddings (each method's gain, the learned rule's decision boundary, the gap to the runner-up)
and the bias breakdown's per-group FPIR.

Each network's colour and dash are #13's, from the shared `ryuk.plotting.MODEL_STYLES`
(ArcFace solid navy, FaceNet long-dash violet, SFace dotted ochre). Kinds of probe are not
models, so they take greys and one accent instead (`KIND_STYLES`).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from numpy.typing import NDArray

from ryuk.eda.summary import Draw
from ryuk.evaluation.names import model_name
from ryuk.evaluation.results import (
    AttributeBreakdown,
    BiasAttribute,
    Identification,
    Learning,
    LearningModel,
    Method,
    MethodResult,
    Results,
    TopGapSample,
    Verification,
)
from ryuk.evaluation.tables import METHOD_NAMES
from ryuk.plotting import MODEL_STYLES, direct_label, new_figure, panel_title
from ryuk.plotting.style import HAIRLINE, INK, INPUT_BORDER, MUTED, NO_MATCH, PAPER

MIN_FAR: Final = 1e-4
"""The left edge of the FAR axis; View 2's 3,000 negatives cannot resolve below 1/3,000."""
LABEL_FPIR: Final = 1e-3
"""Where the open-set figure labels each curve: at FPIR 0.1% the curves stand furthest apart."""


def lfw_roc(verification: Verification) -> Figure:
    figure = Figure(figsize=(6.4, 4.4), layout="constrained")
    axes = figure.add_subplot()
    lowest = 1.0
    for result in verification.models:
        style = MODEL_STYLES[result.model.network]
        # A log axis cannot show FAR 0; the curve starts where its first false accept does.
        points = [(f, t) for f, t in zip(result.roc.far, result.roc.tar, strict=True) if f > 0]
        far = [MIN_FAR, *(f for f, _ in points)]
        tar = [_tar_before(result.roc.far, result.roc.tar), *(t for _, t in points)]
        label = f"{model_name(result.model)}, {result.accuracy * 100:.2f}%"
        axes.step(
            far,
            tar,
            where="post",
            color=style.colour,
            linestyle=style.dashes,
            linewidth=1.6,
            label=label,
        )
        lowest = min(lowest, *tar)
    axes.set_xscale("log")
    axes.set_xlim(MIN_FAR, 1.0)
    axes.set_ylim(max(0.0, math.floor(lowest * 20) / 20), 1.0005)
    axes.set_xlabel("False accept rate (log scale)")
    axes.set_ylabel("True accept rate")
    axes.grid(visible=True, which="major", linewidth=0.4, alpha=0.5)
    axes.legend(loc="lower right", frameon=False)
    return figure


def _tar_before(far: list[float], tar: list[float]) -> float:
    """TAR at FAR 0: the highest TAR reached before the first false accept. The same reading
    gives TPIR at FPIR 0 on an open-set curve."""
    return max(t for f, t in zip(far, tar, strict=True) if f == 0)


def openset_curves(identification: Identification, draw: Draw = "test") -> Figure:
    """TPIR against FPIR on one CelebA draw, log FPIR axis, each model's frozen threshold marked.

    Each model is labelled on its curve rather than in a legend (#13).
    """
    figure = new_figure(height=4.2)
    axes = figure.add_subplot()
    selection = identification.selection(draw)
    # One false alarm is the smallest FPIR the draw's non-mated probes can show.
    floor = 10 ** math.floor(math.log10(1 / selection.non_mated_probes))
    lowest = 1.0
    for model in identification.models:
        result = model.on(draw)
        style = MODEL_STYLES[model.model.network]
        curve = result.curve
        points = [(f, t) for f, t in zip(curve.fpir, curve.tpir, strict=True) if f > 0]
        at_zero = _tar_before(curve.fpir, curve.tpir)
        fpir = [floor, *(f for f, _ in points)]
        tpir = [at_zero, *(t for _, t in points)]
        axes.step(fpir, tpir, where="post", linewidth=1.6, **style.line())
        point = result.at_threshold
        axes.plot(
            [max(point.fpir.value, floor)], [point.tpir.value], markersize=6, **style.points()
        )
        # Labelled just under its curve at FPIR 0.1%, where the curves stand furthest apart.
        at_label = max(t for f, t in zip(fpir, tpir, strict=True) if f <= LABEL_FPIR)
        direct_label(
            axes,
            (LABEL_FPIR, at_label),
            model_name(model.model),
            colour=style.colour,
            offset=(4.0, -4.0),
            va="top",
        )
        lowest = min(lowest, *tpir)
    axes.set_xscale("log")
    axes.set_xlim(floor, 1.0)
    axes.set_ylim(max(0.0, math.floor(lowest * 20) / 20), 1.0005)
    axes.set_xlabel("False positive identification rate (log scale)")
    axes.set_ylabel("True positive identification rate")
    axes.grid(visible=True, which="major", linewidth=0.4, alpha=0.5)
    return figure


type ProbeKind = Literal["right", "wrong", "non-mated"]
"""A probe by its best-photo top candidate: a mated probe's right or wrong, or a non-mated one."""


@dataclass(frozen=True, slots=True)
class KindStyle:
    """How one kind of probe is drawn: greys for the two common kinds, the accent for the rare
    wrong top candidate, so it stands out wherever it falls."""

    colour: str
    filled: bool
    size: float
    """Scatter marker area, in points squared."""


KIND_STYLES: Final[Mapping[ProbeKind, KindStyle]] = MappingProxyType(
    {
        "right": KindStyle(MUTED, filled=True, size=5.0),
        "non-mated": KindStyle(INPUT_BORDER, filled=False, size=7.0),
        "wrong": KindStyle(NO_MATCH, filled=True, size=11.0),
    }
)
"""Drawn in this order, so the wrong top candidates lie on top."""

_ROW: Final = 0.6
"""Inches per method in the gains figure: room for three models' intervals and their labels."""
_LABEL_WIDTH: Final = 0.25
"""About how much of a panel's width a kind's label takes."""
_LABEL_HEIGHT: Final = 0.09
"""About how much of a panel's height a line of label text takes."""
_TICK_PAD: Final = 3.5
"""Points between the axes and a tick label with no tick, matplotlib's default."""
_GROUP_ROW: Final = 0.2
"""Inches per group in the per-group FPIR figure."""
_ATTRIBUTE_GAP: Final = 0.7
"""Space between one attribute's groups and the next's, in rows."""


def learning_gains(learning: Learning) -> Figure:
    """Each method's test TPIR gain at the target FPIR over best photo, in points, with its 95%
    paired bootstrap interval: a row per method, a point per model within it.

    A method improves on best photo only if its interval lies wholly above the zero line; its
    marker is then filled and its interval heavier, and otherwise hollow and light. The rows of
    methods that need retraining whenever the watchlist changes, and so never run live, are
    shaded. Each model is labelled at the end of its first interval rather than in a legend.
    """
    methods = list(
        dict.fromkeys(
            result.method
            for compared in learning.models
            for result in compared.methods
            if result.gain is not None
        )
    )
    figure = new_figure(height=_ROW * len(methods) + 0.8)
    axes = figure.add_subplot()
    count = len(learning.models)
    offsets = np.linspace(-0.25, 0.25, count) if count > 1 else np.zeros(1)
    retraining = [
        any(r.needs_retraining for c in learning.models if (r := _method(c, method)) is not None)
        for method in methods
    ]
    for row, retrains in enumerate(retraining):
        if retrains:
            axes.axhspan(row - 0.5, row + 0.5, color=HAIRLINE, alpha=0.5, linewidth=0, zorder=0)
    lows: list[float] = []
    highs: list[float] = []
    for offset, compared in zip(offsets, learning.models, strict=True):
        style = MODEL_STYLES[compared.model.network]
        segments, widths = [], []
        points: dict[bool, tuple[list[float], list[float]]] = {True: ([], []), False: ([], [])}
        for row, method in enumerate(methods):
            result = _method(compared, method)
            if result is None or result.gain is None:
                continue
            gain, y = result.gain, row + float(offset)
            segments.append([(gain.ci.low * 100, y), (gain.ci.high * 100, y)])
            widths.append(2.2 if gain.improves else 1.0)
            points[gain.improves][0].append(gain.value * 100)
            points[gain.improves][1].append(y)
            lows.append(min(gain.ci.low * 100, 0.0))
            highs.append(max(gain.ci.high * 100, 0.0))
        axes.add_collection(LineCollection(segments, colors=style.colour, linewidths=widths))
        for improves, (xs, ys) in points.items():
            if xs:
                axes.plot(
                    xs,
                    ys,
                    markersize=5.5,
                    **{
                        **style.points(),
                        "markerfacecolor": style.colour if improves else PAPER,
                        "label": f"{style.name}, improves" if improves else style.name,
                    },
                )
        if segments:
            (_, _), (high, y) = segments[0]
            direct_label(axes, (high, y), model_name(compared.model), colour=style.colour)
    axes.axvline(0, color=INK, linewidth=0.8, label="no gain", zorder=1)
    for row, retrains in enumerate(retraining):
        if retrains:
            # Under the method's name, where no interval can reach.
            axes.annotate(
                "retrains, never live",
                xy=(0.0, row),
                xycoords=("axes fraction", "data"),
                xytext=(-_TICK_PAD, -6.0),
                textcoords="offset points",
                ha="right",
                va="top",
                color=MUTED,
                fontsize=7.0,
            )
    # Around every interval and zero, with a margin for the markers.
    low, high = min(lows, default=0.0), max(highs, default=0.0)
    margin = (high - low) * 0.06 or 0.5
    axes.set_xlim(low - margin, high + margin)
    axes.set_ylim(len(methods) - 0.5, -0.5)
    axes.set_yticks(range(len(methods)), labels=[METHOD_NAMES[m] for m in methods])
    axes.grid(visible=False, axis="y")
    axes.tick_params(axis="y", length=0, pad=_TICK_PAD)
    axes.set_xlabel(
        f"Test TPIR gain at FPIR {learning.target_fpir:.0%} over best photo (percentage points)"
    )
    return figure


def _method(compared: LearningModel, method: Method) -> MethodResult | None:
    return next((result for result in compared.methods if result.method == method), None)


def learned_rules(learning: Learning) -> Figure:
    """The learned decision rule of each model, a panel each, over a sample of validation probes
    as their best-photo top score against its gap to the runner-up.

    The solid line is the rule's decision boundary at its frozen cut-off c on P(match), where
    intercept + top_score * top + gap * gap = log(c / (1 - c)): a probe above and right of it is
    a match. The dashed line is best photo's own frozen threshold, which ignores the gap. The
    sample keeps every probe whose top candidate is wrong, so the kinds are not in proportion.
    The first panel names each kind on its own cloud, where it lies furthest from the others.
    """
    count = len(learning.models)
    figure = new_figure(height=2.9)
    panels = figure.subplots(1, count, sharex=True, sharey=True, squeeze=False)[0]
    tops = [v for c in learning.models for v in c.sample.top] + [
        c.methods[0].threshold for c in learning.models
    ]
    gaps = [v for c in learning.models for v in c.sample.gap]
    xlim = (min(tops) - 0.03, max(tops) + 0.03)
    # Headroom above the highest gap for the two lines' labels.
    ylim = (0.0, max(max(gaps, default=0.1) * 1.2, 0.05))
    for index, (axes, compared) in enumerate(zip(panels, learning.models, strict=True)):
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)
        top = np.asarray(compared.sample.top, dtype=float)
        gap = np.asarray(compared.sample.gap, dtype=float)
        kind = np.asarray(compared.sample.kind)
        for name, style in KIND_STYLES.items():
            chosen = kind == name
            axes.scatter(
                top[chosen],
                gap[chosen],
                s=style.size,
                c=style.colour if style.filled else "none",
                edgecolors=style.colour,
                linewidths=0.6,
                label=name,
                zorder=2 if name == "wrong" else 1,
            )
        boundary = _boundary(compared, xlim, ylim)
        axes.plot(*boundary, color=INK, linewidth=1.3, label="learned rule", zorder=3)
        baseline = compared.methods[0].threshold
        axes.plot(
            [baseline, baseline],
            list(ylim),
            color=INK,
            linewidth=0.9,
            linestyle=(0, (4.0, 2.0)),
            label="best photo",
            zorder=3,
        )
        panel_title(axes, model_name(compared.model))
        axes.set_xlabel("Top score (cosine)")
        if index == 0:
            axes.set_ylabel("Gap to the runner-up")
    if count:
        # Labels are placed by their size on the laid-out panel.
        figure.draw_without_rendering()
        first, compared = panels[0], learning.models[0]
        boundary, baseline = _boundary(compared, xlim, ylim), compared.methods[0].threshold
        view = _View(first, xlim, ylim)
        taken = _label_lines(view, boundary, baseline)
        _label_kinds(view, compared.sample, _line_samples(boundary, baseline, ylim), taken)
    return figure


def _boundary(
    compared: LearningModel, xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[list[float], list[float]]:
    """Two points of the learned rule's decision boundary spanning the view."""
    rule, cut = compared.learned_rule, compared.method("learned").threshold
    if not 0.0 < cut < 1.0:
        raise ValueError(f"the learned rule's cut-off must lie strictly between 0 and 1: {cut}")
    logit = math.log(cut / (1.0 - cut))
    if rule.gap != 0.0:
        xs = [xlim[0], xlim[1]]
        return xs, [(logit - rule.intercept - rule.top_score * x) / rule.gap for x in xs]
    if rule.top_score != 0.0:
        x = (logit - rule.intercept) / rule.top_score
        return [x, x], list(ylim)
    raise ValueError("a learned rule with no weight on either input has no boundary")


def _line_samples(
    boundary: tuple[list[float], list[float]], baseline: float, ylim: tuple[float, float]
) -> NDArray[np.float64]:
    """Points along both lines, as (x, y) rows, for labels to keep clear of."""
    along = np.linspace(0.0, 1.0, 60)
    (x0, x1), (y0, y1) = boundary
    return np.concatenate(
        [
            np.column_stack([x0 + along * (x1 - x0), y0 + along * (y1 - y0)]),
            np.column_stack([np.full(along.size, baseline), ylim[0] + along * np.diff(ylim)]),
        ]
    )


type _Box = tuple[float, float, float, float]
"""A label's extent in axes fractions: left, bottom, right, top."""


@dataclass(frozen=True, slots=True)
class _View:
    """A laid-out panel and its limits, for placing labels by their size."""

    axes: Axes
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def fraction(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]),
            (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]),
        )

    def data(self, fx: float, fy: float) -> tuple[float, float]:
        return (
            self.xlim[0] + fx * (self.xlim[1] - self.xlim[0]),
            self.ylim[0] + fy * (self.ylim[1] - self.ylim[0]),
        )

    def size(self, text: str) -> tuple[float, float]:
        """`text`'s width and height as fractions of the panel, in the current font."""
        width, height, _ = TextToPath().get_text_width_height_descent(
            text, FontProperties(), ismath=False
        )
        extent = self.axes.get_window_extent()
        points = 72 / self.axes.figure.dpi
        return width / (extent.width * points), height / (extent.height * points)


def _overlaps(box: _Box, other: _Box) -> bool:
    return box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]


def _inside(box: _Box) -> bool:
    return box[0] >= 0.0 and box[2] <= 1.0 and box[1] >= 0.0 and box[3] <= 1.0


_LINE_LABEL_PAD: Final = 0.015
"""Space between a line and its label, as a fraction of the panel."""
_LINE_LABEL_STEPS: Final = 6
"""How many label heights down its line a line's label may move to fit."""


def _label_lines(
    view: _View, boundary: tuple[list[float], list[float]], baseline: float
) -> list[_Box]:
    """Name both lines beside them, as near the top of the view as they fit: each label on
    either side of its line, a step down it at a time, until both lie inside the panel and
    apart. Returns the boxes they take."""
    (fx0, fy0), (fx1, fy1) = (view.fraction(x, y) for x, y in zip(*boundary, strict=True))
    at_baseline = view.fraction(baseline, view.ylim[1])[0]
    steps = range(_LINE_LABEL_STEPS)

    def beside(x: float, top: float, text: str, left: bool) -> _Box:
        width, height = view.size(text)
        start = x - _LINE_LABEL_PAD - width if left else x + _LINE_LABEL_PAD
        return (start, top - height, start + width, top)

    def level(text: str, step: int) -> float:
        return 1.0 - _LINE_LABEL_PAD - step * (view.size(text)[1] + _LINE_LABEL_PAD)

    def on_boundary(top: float) -> float | None:
        """The boundary's x at the middle of a label whose top is at `top`."""
        if math.isclose(fy0, fy1):
            return None
        middle = top - view.size("learned rule")[1] / 2
        return fx0 + (middle - fy0) * (fx1 - fx0) / (fy1 - fy0)

    learned: list[tuple[int, _Box | None]] = [(0, None)]
    if not (max(fy0, fy1) < 0 or min(fy0, fy1) > 1):
        learned = [
            (step, beside(x, level("learned rule", step), "learned rule", left))
            for step in steps
            if (x := on_boundary(level("learned rule", step))) is not None
            # Left of the boundary first: it usually leans left of best photo's threshold.
            for left in (True, False)
        ]
    best = [
        (step, beside(at_baseline, level("best photo", step), "best photo", left))
        for step in steps
        for left in (False, True)
    ]

    def crosses_boundary(box: _Box) -> bool:
        return any(
            box[0] - _LINE_LABEL_PAD <= x <= box[2] + _LINE_LABEL_PAD
            for y in (box[1], box[3])
            if (x := on_boundary(y + view.size("learned rule")[1] / 2)) is not None
        )

    def crosses_baseline(box: _Box) -> bool:
        return box[0] - _LINE_LABEL_PAD <= at_baseline <= box[2] + _LINE_LABEL_PAD

    def fits(rule: _Box | None, photo: _Box, *, strict: bool) -> bool:
        """Both inside and apart; strictly, each also clear of the other's line."""
        if not _inside(photo) or (strict and crosses_boundary(photo)):
            return False
        return rule is None or (
            _inside(rule) and not _overlaps(rule, photo) and not (strict and crosses_baseline(rule))
        )

    # Two lines close together may leave no room clear of both; a label then lies over the
    # other line, on its paper backing, but never over the other label.
    options = [(0, learned[0][1], best[0][1])]
    for strict in (True, False):
        if fitting := [
            (first + second, rule, photo)
            for first, rule in learned
            for second, photo in best
            if fits(rule, photo, strict=strict)
        ]:
            options = fitting
            break
    _, rule, photo = min(options, key=lambda option: option[0])
    taken = []
    for text, placed in (("learned rule", rule), ("best photo", photo)):
        if placed is None:
            continue
        taken.append(placed)
        direct_label(
            view.axes,
            view.data(placed[0], placed[3]),
            text,
            offset=(0.0, 0.0),
            va="top",
            bbox={"boxstyle": "square,pad=0.1", "facecolor": PAPER, "edgecolor": "none"},
            zorder=4,
        )
    return taken


def _label_kinds(
    view: _View, sample: TopGapSample, lines: NDArray[np.float64], taken: Sequence[_Box]
) -> None:
    """Name each kind of probe on its own cloud: centred on the probe of that kind furthest from
    every probe of another kind and from the lines, inside the panel and clear of the labels
    already placed."""
    scale = np.array([view.xlim[1] - view.xlim[0], view.ylim[1] - view.ylim[0]])
    origin = np.array([view.xlim[0], view.ylim[0]])
    points = (np.column_stack([sample.top, sample.gap]) - origin) / scale
    kind = np.asarray(sample.kind)
    boxes = list(taken)
    guides = (lines - origin) / scale
    for name, style in KIND_STYLES.items():
        own = points[kind == name]
        if not own.size:
            continue
        width, height = view.size(name)
        half = np.array([width / 2 + 0.01, height / 2 + 0.01])
        fits = [
            spot
            for spot in own
            if _inside((*(spot - half), *(spot + half)))
            and not any(_overlaps((*(spot - half), *(spot + half)), b) for b in boxes)
        ]
        candidates = np.array(fits) if fits else own
        others = np.concatenate([points[kind != name], guides])
        # Labels are wider than tall: weigh horizontal distance down.
        apart = (candidates[:, None, :] - others[None, :, :]) * np.array([0.6, 1.0])
        spot = candidates[np.argmax(np.min(np.hypot(apart[..., 0], apart[..., 1]), axis=1))]
        boxes.append((*(spot - half), *(spot + half)))
        direct_label(
            view.axes,
            view.data(float(spot[0]), float(spot[1])),
            name,
            colour=style.colour,
            offset=(0.0, 0.0),
            ha="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER, "edgecolor": "none"},
            zorder=4,
        )


def gap_distributions(learning: Learning) -> Figure:
    """How far the best-photo top candidate scores above the runner-up on the test draw, a panel
    per model: mated probes whose top candidate is right, mated probes whose top candidate is
    wrong, and non-mated probes.

    Each kind is a density, its counts over its total and the bin width, so each encloses an
    area of 1: the wrong top candidates are a small fraction of the mated probes and would not
    show on a count axis. The axis stops where 99% of every kind's probes have been counted.
    """
    count = len(learning.models)
    figure = new_figure(height=2.5)
    panels = figure.subplots(1, count, sharex=True, sharey=True, squeeze=False)[0]
    reach = 0.0
    labels: list[tuple[ProbeKind, float, float]] = []
    for index, (axes, compared) in enumerate(zip(panels, learning.models, strict=True)):
        histogram = compared.gaps
        edges = np.asarray(histogram.edges, dtype=float)
        widths = np.diff(edges)
        kinds: dict[ProbeKind, list[int]] = {
            "right": histogram.right,
            "non-mated": histogram.non_mated,
            "wrong": histogram.wrong,
        }
        peaks: list[tuple[ProbeKind, float, float]] = []
        for name, counts in kinds.items():
            counted = np.asarray(counts, dtype=float)
            if not counted.sum():
                continue
            density = counted / (counted.sum() * widths)
            style = KIND_STYLES[name]
            axes.stairs(
                density,
                edges,
                label=name,
                fill=name == "right",
                color=HAIRLINE if name == "right" else style.colour,
                edgecolor=style.colour,
                linewidth=0.9 if name == "right" else 1.3,
                linestyle="-" if style.filled else (0, (4.0, 2.0)),
            )
            covered = np.searchsorted(np.cumsum(counted) / counted.sum(), 0.99)
            reach = max(reach, float(edges[min(int(covered) + 1, edges.size - 1)]))
            peak = int(np.argmax(density))
            peaks.append((name, float(edges[peak + 1]), float(density[peak])))
        panel_title(axes, model_name(compared.model))
        axes.set_xlabel("Gap to the runner-up (cosine)")
        if index == 0:
            axes.set_ylabel("Density")
            labels = peaks
    for axes in panels:
        axes.set_xlim(0.0, reach or 1.0)
        axes.set_ylim(0.0, axes.get_ylim()[1] * 1.08)
    if count:
        _label_peaks(panels[0], labels)
    return figure


def _label_peaks(axes: Axes, peaks: Sequence[tuple[ProbeKind, float, float]]) -> None:
    """Name each kind just above and right of its highest bin, moving a label down where it
    would sit on a taller one's."""
    (left, right), (_, high) = axes.get_xlim(), axes.get_ylim()
    placed: list[tuple[float, float]] = []
    for name, x, y in sorted(peaks, key=lambda peak: -peak[2]):
        near = [
            below for across, below in placed if abs(across - x) < _LABEL_WIDTH * (right - left)
        ]
        level = min([y, *(below - _LABEL_HEIGHT * high for below in near)])
        placed.append((x, level))
        direct_label(
            axes,
            (x, level),
            name,
            colour=KIND_STYLES[name].colour,
            offset=(2.0, 1.0),
            va="bottom",
        )


def group_fpir(results: Results) -> Figure:
    """Each group's test FPIR at the model's single frozen threshold, with its 95% interval, a
    panel per model under best photo; the dashed line is the model's overall test FPIR.

    Every model is drawn under the same rule, so the panels compare models on equal terms;
    Table 4 gives the first active model's numbers under the rule it runs live. A group with
    too few held-out identities to estimate keeps its row, labelled with its count, and the
    indicative Male and Young cells are shaded.
    """
    if results.bias is None or results.identification is None:
        raise ValueError("the results have no bias breakdown; run `ryuk evaluate bias`")
    breakdowns = [b for b in results.bias.models if b.rule == "best-photo"]
    if not breakdowns:
        raise ValueError("the bias breakdown has no model under best-photo")
    overall = {m.model: m.test.at_threshold.fpir.value for m in results.identification.models}
    rows = _group_rows(breakdowns[0].attributes)
    bottom = rows[-1].y + 0.5 if rows else 0.5
    top = -1.3
    """Room above the first group for the overall line's label."""
    figure = new_figure(height=_GROUP_ROW * (bottom - top) + 0.8)
    panels = figure.subplots(1, len(breakdowns), sharex=True, sharey=True, squeeze=False)[0]
    measured = [
        g.fpir.ci.high for b in breakdowns for a in b.attributes for g in a.groups if g.fpir
    ]
    reach = max([*measured, *overall.values()]) * 100 * 1.08
    indicative = [row.y for row in rows if row.indicative]
    for index, (axes, breakdown) in enumerate(zip(panels, breakdowns, strict=True)):
        style = MODEL_STYLES[breakdown.model.network]
        groups = {(a.attribute, g.label): g for a in breakdown.attributes for g in a.groups}
        if indicative:
            axes.axhspan(
                min(indicative) - 0.5,
                max(indicative) + 0.5,
                color=HAIRLINE,
                alpha=0.5,
                linewidth=0,
                zorder=0,
            )
        segments, xs, ys = [], [], []
        for row in rows:
            group = groups.get((row.attribute, row.label))
            if group is None:
                continue
            if group.fpir is None:
                axes.annotate(
                    f"too few (n = {group.held_out_identities:,})",
                    xy=(0.0, row.y),
                    xycoords=("axes fraction", "data"),
                    xytext=(3.0, 0.0),
                    textcoords="offset points",
                    va="center",
                    color=MUTED,
                    fontsize=7.0,
                    # Over the overall line, should it pass behind.
                    bbox={"boxstyle": "square,pad=0.1", "facecolor": PAPER, "edgecolor": "none"},
                    zorder=4,
                )
                continue
            segments.append([(group.fpir.ci.low * 100, row.y), (group.fpir.ci.high * 100, row.y)])
            xs.append(group.fpir.value * 100)
            ys.append(row.y)
        axes.add_collection(LineCollection(segments, colors=style.colour, linewidths=1.0))
        axes.plot(xs, ys, markersize=4.0, **style.points())
        fpir = overall[breakdown.model] * 100
        axes.plot(
            [fpir, fpir],
            [-0.5, bottom],
            color=INK,
            linewidth=0.8,
            linestyle=(0, (4.0, 2.0)),
            label="overall",
            zorder=1,
        )
        panel_title(axes, model_name(breakdown.model))
        axes.set_xlabel("Test FPIR (%)")
        axes.grid(visible=False, axis="y")
        if index == 0:
            direct_label(axes, (fpir, -0.5), "overall", offset=(3.0, 1.0), va="bottom")
    if indicative:
        # Beside the last panel's shading, clear of every interval.
        panels[-1].annotate(
            "indicative",
            xy=(1.0, (min(indicative) + max(indicative)) / 2),
            xycoords=("axes fraction", "data"),
            xytext=(3.0, 0.0),
            textcoords="offset points",
            rotation=270,
            ha="left",
            va="center",
            color=MUTED,
            fontsize=7.0,
        )
    first = panels[0]
    # A little room left of 0, so a group with no false alarm is not hidden by the axis.
    first.set_xlim(-0.02 * reach, reach)
    first.set_ylim(bottom, top)
    first.set_yticks([row.y for row in rows], labels=[row.label for row in rows])
    first.tick_params(axis="y", length=0)
    return figure


@dataclass(frozen=True, slots=True)
class _GroupRow:
    y: float
    attribute: BiasAttribute
    label: str
    indicative: bool


def _group_rows(attributes: Sequence[AttributeBreakdown]) -> list[_GroupRow]:
    """Where each group goes, top to bottom, with a gap between one attribute and the next."""
    rows: list[_GroupRow] = []
    y = 0.0
    for position, attribute in enumerate(attributes):
        if position:
            y += _ATTRIBUTE_GAP
        for group in attribute.groups:
            rows.append(_GroupRow(y, attribute.attribute, group.label, attribute.indicative))
            y += 1.0
    return rows
