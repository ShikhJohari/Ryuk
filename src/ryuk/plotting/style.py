"""The lab-notebook figure style from #13: palette, fonts, series styles, direct labels, captions.

Every figure the package draws, for the notebooks and the report alike, is made inside
`lab_style()` on a figure from `new_figure()`. Lines are told apart by lightness and dash
pattern as well as hue, so the figures survive greyscale printing and colour-blind readers,
and they are labelled directly at their ends instead of in a legend.

Captions are opt-in: the report sets its own, so figure functions never draw a "Figure N."
title and a notebook adds one with `caption()`.
"""

import atexit
import contextlib
import functools
import importlib.resources
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text
from matplotlib.textpath import TextToPath
from matplotlib.typing import LineStyleType, MarkerType, RcKeyType

from ryuk.eda.summary import Draw

# The palette of #13's direction B. Match and no match differ in lightness as well as hue.
PAPER: Final = "#F4F1E8"
RAISED: Final = "#FBFAF6"
INK: Final = "#1C1B19"
MUTED: Final = "#5E5A52"
HAIRLINE: Final = "#D8D2C4"
INPUT_BORDER: Final = "#BDB5A3"
MATCH: Final = "#1F4E8C"
NO_MATCH: Final = "#A8490F"
OK: Final = "#2F6B3A"
WARNING: Final = "#7A5600"
DESTRUCTIVE: Final = "#8E3D0C"

SANS: Final = "Public Sans"
"""Body text, tick labels and axis labels.

#13 asks for tabular numerals, but matplotlib 3.11 measures text without OpenType features
and then draws it with them, so turning on `tnum` shifts every label off its anchor (a centred
tick label drifts right). Figures keep Public Sans's default proportional digits; numbers set
in a column are right-aligned instead."""
SERIF: Final = "Newsreader 16pt"
"""Titles, figure numbers and captions."""

COLUMN_WIDTH: Final = 6.3
"""A single-column A4 text block, in inches: the width every report figure is drawn at."""

FONT_FILES: Final = (
    "Newsreader16pt-Regular.ttf",
    "Newsreader16pt-Italic.ttf",
    "Newsreader16pt-SemiBold.ttf",
    "PublicSans-Regular.ttf",
    "PublicSans-Italic.ttf",
    "PublicSans-SemiBold.ttf",
)
"""The vendored fonts in `ryuk/plotting/fonts/`, which the report's Typst build also uses."""

type Network = Literal["arcface", "facenet", "sface"]
"""A recognition model's network, whatever its weights or execution provider."""

type Dataset = Literal["lfw"] | Draw
"""What an EDA series is drawn from: LFW, or one of CelebA's draws."""


@dataclass(frozen=True, slots=True)
class SeriesStyle:
    """How one series, such as a recognition model's ROC curve, is drawn and labelled.

    `dashes` is a matplotlib line style: "-" or an (offset, (on, off, ...)) pattern in points.
    `filled` says whether its markers are solid or hollow, another cue besides hue.
    """

    name: str
    colour: str
    dashes: LineStyleType
    marker: MarkerType
    filled: bool = True

    def line(self) -> dict[str, Any]:
        """Keyword arguments for `Axes.plot` drawing this series as a line."""
        return {"color": self.colour, "linestyle": self.dashes, "label": self.name}

    def points(self) -> dict[str, Any]:
        """Keyword arguments for `Axes.plot` drawing this series as unjoined markers."""
        return {
            "color": self.colour,
            "linestyle": "none",
            "marker": self.marker,
            "markerfacecolor": self.colour if self.filled else "none",
            "markeredgecolor": self.colour,
            "label": self.name,
        }


_LONG_DASH: Final = (0, (7.0, 2.5))
_DASH: Final = (0, (4.0, 2.0))
_DOTTED: Final = (0, (1.2, 1.8))

MODEL_STYLES: Final[Mapping[Network, SeriesStyle]] = MappingProxyType(
    {
        "arcface": SeriesStyle("ArcFace", "#1F4E8C", "-", "o"),
        "facenet": SeriesStyle("FaceNet", "#8A4FA0", _LONG_DASH, "s"),
        "sface": SeriesStyle("SFace", "#C98A1B", _DOTTED, "D"),
    }
)
"""#13's per-model style, keyed by network: each differs in lightness and dash, not hue alone."""

DATASET_STYLES: Final[Mapping[Dataset, SeriesStyle]] = MappingProxyType(
    {
        # Kept clear of the model hues so an EDA series is never read as a model. The two CelebA
        # draws share a hue and differ in lightness, dash and marker fill; LFW is the darkest.
        "lfw": SeriesStyle("LFW", INK, _DOTTED, "o"),
        "validation": SeriesStyle("CelebA validation", "#5E9C94", _DASH, "o", filled=False),
        "test": SeriesStyle("CelebA test", "#1D5750", "-", "o"),
    }
)
"""The EDA's per-dataset style: LFW, and CelebA's validation and test draws."""


def _rc() -> dict[RcKeyType, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [SANS],
        "font.serif": [SERIF],
        "font.size": 8.5,
        "text.color": INK,
        # On screen a figure sits on paper, as the client's pages do. `save_figure` writes files
        # transparent instead, so the report's page shows through: the PDF is meant for print,
        # where a tinted panel on white paper reads as a box and wastes toner, and a transparent
        # figure also sits right on the report if it keeps #13's paper tone.
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "figure.dpi": 150,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.labelsize": 8.5,
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=[INK, MUTED]),
        "grid.color": HAIRLINE,
        "grid.linewidth": 0.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 1.0,
        "lines.solid_capstyle": "butt",
        "patch.linewidth": 0.6,
        "hatch.color": INPUT_BORDER,
        "hatch.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 200,
        # Glyphs as paths, so an SVG looks the same without the fonts installed; a fixed salt so
        # the SVG's element ids, and so the committed files, do not change between runs.
        "svg.fonttype": "path",
        "svg.hashsalt": "ryuk",
        "pdf.fonttype": 42,
    }


STYLE: Final[Mapping[RcKeyType, Any]] = MappingProxyType(_rc())
"""The rcParams of the lab-notebook style."""

_open_fonts = contextlib.ExitStack()
# as_file() may extract a font to a temporary file, and matplotlib reopens fonts by path at
# render time, so the files are held open until the interpreter exits.
atexit.register(_open_fonts.close)


@functools.cache
def font_path(filename: str) -> Path:
    """The on-disk path of one of the vendored fonts, the same for the life of the process."""
    resource = importlib.resources.files("ryuk.plotting").joinpath("fonts", filename)
    return _open_fonts.enter_context(importlib.resources.as_file(resource))


def register_fonts() -> None:
    """Make the vendored fonts available to matplotlib. Safe to call any number of times."""
    registered = {entry.fname for entry in font_manager.fontManager.ttflist}
    for filename in FONT_FILES:
        path = font_path(filename)
        if str(path) not in registered:
            font_manager.fontManager.addfont(path)


@contextlib.contextmanager
def lab_style() -> Iterator[None]:
    """Draw and save figures in the lab-notebook style within this block."""
    register_fonts()
    with mpl.rc_context(dict(STYLE)):
        yield


def use_lab_style() -> None:
    """Apply the style for the rest of the session, as a notebook does in its first cell."""
    register_fonts()
    mpl.rcParams.update(STYLE)


def new_figure(height: float, width: float = COLUMN_WIDTH) -> Figure:
    """An empty figure, `width` by `height` inches, laid out by matplotlib's constrained layout.

    Call inside `lab_style()`: text and ticks take the style when they are made. The figure is
    not registered with pyplot, so nothing keeps it alive once the caller lets go of it.
    """
    return Figure(figsize=(width, height), layout="constrained")


def panel_title(ax: Axes, text: str) -> Text:
    """A small-multiple's title: serif, left-aligned over the axes. Not a figure caption."""
    return ax.set_title(text, fontfamily="serif", loc="left", color=INK)


def direct_label(
    ax: Axes,
    target: Line2D | tuple[float, float],
    text: str,
    *,
    colour: str | None = None,
    offset: tuple[float, float] = (4.0, 0.0),
    **kwargs: Any,  # noqa: ANN401 - passed through to Axes.annotate
) -> Annotation:
    """Label a series where it is drawn instead of in a legend (#13).

    Given a line, the label goes just past its last finite point, in the line's colour. Given
    an (x, y) in data coordinates, it goes there, in ink unless `colour` says otherwise.
    `offset` moves the label from that point, in points; it is left-aligned and vertically
    centred there unless `ha` or `va` say otherwise. The label is not clipped to the axes, and
    constrained layout makes room for it. Other keywords go to `Axes.annotate`.
    """
    if isinstance(target, Line2D):
        xy = _line_end(target)
        colour = colour if colour is not None else str(target.get_color())
    else:
        xy = target
        colour = colour if colour is not None else INK
    return ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        color=colour,
        annotation_clip=False,
        **{"ha": "left", "va": "center", **kwargs},
    )


def _line_end(line: Line2D) -> tuple[float, float]:
    xs = np.asarray(line.get_xdata(), dtype=float)
    ys = np.asarray(line.get_ydata(), dtype=float)
    finite = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
    if finite.size:
        return (float(xs[finite[-1]]), float(ys[finite[-1]]))
    raise ValueError("cannot label a line with no finite points")


_CAPTION_GID: Final = "caption"
_CAPTION_SIZE: Final = 9.5
_CAPTION_LINE_SPACING: Final = 1.35
_CAPTION_GAP: Final = 10.0
"""Space between the plot and its caption, in points."""
_CAPTION_MARGIN: Final = 4.0
"""Space around the caption at the figure's left, right and bottom edges, in points."""


def caption(fig: Figure, number: int, text: str) -> None:
    """Add a notebook-style caption under the figure: "**Figure N.** text", in the serif face.

    The figure grows taller to hold it, so the plot keeps its size. Text too long for one line
    wraps with a hanging indent under the caption's first word. Only for figures with
    constrained layout or none; a figure takes one caption.
    """
    if any(t.get_gid() == _CAPTION_GID for t in fig.texts):
        raise ValueError("the figure already has a caption")
    engine = fig.get_layout_engine()
    if engine is not None and not isinstance(engine, ConstrainedLayoutEngine):
        raise ValueError("caption() needs a figure with constrained layout or no layout engine")

    register_fonts()
    label = f"Figure {number}."
    label_font = FontProperties(family=SERIF, weight="semibold", size=_CAPTION_SIZE)
    body_font = FontProperties(family=SERIF, size=_CAPTION_SIZE)
    measure = TextToPath()
    label_width = measure.get_text_width_height_descent(label, label_font, ismath=False)[0]
    # A lone space has no ink to measure, so its width is what it adds between two letters.
    space = (
        measure.get_text_width_height_descent("a a", body_font, False)[0]
        - measure.get_text_width_height_descent("aa", body_font, False)[0]
    )

    width_in, height_in = fig.get_size_inches()
    fig_width, fig_height = width_in * 72, height_in * 72
    indent = _CAPTION_MARGIN + label_width + space
    lines = _wrap(text, body_font, fig_width - indent - _CAPTION_MARGIN, measure)
    extra = _CAPTION_GAP + len(lines) * _CAPTION_SIZE * _CAPTION_LINE_SPACING + _CAPTION_MARGIN
    new_height = fig_height + extra
    _make_room_below(fig, fig_height, extra)
    fig.set_figheight(new_height / 72)

    top = (extra - _CAPTION_GAP) / new_height
    common: dict[str, Any] = {"va": "top", "ha": "left", "color": INK, "gid": _CAPTION_GID}
    fig.text(_CAPTION_MARGIN / fig_width, top, label, fontproperties=label_font, **common)
    fig.text(
        indent / fig_width,
        top,
        "\n".join(lines),
        fontproperties=body_font,
        linespacing=_CAPTION_LINE_SPACING,
        **common,
    )


def _make_room_below(fig: Figure, height: float, extra: float) -> None:
    """Keep the plot where it is while the figure grows `extra` points taller at the bottom."""
    new_height = height + extra
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        bottom = extra / new_height
        engine.set(rect=(0.0, bottom, 1.0, 1.0 - bottom))
        return
    for ax in fig.axes:
        box = ax.get_position()
        ax.set_position(
            (
                box.x0,
                (box.y0 * height + extra) / new_height,
                box.width,
                box.height * height / new_height,
            )
        )


def _wrap(text: str, font: FontProperties, width: float, measure: TextToPath) -> list[str]:
    """`text` broken greedily at spaces into lines no wider than `width` points where possible."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}" if line else word
            fits = measure.get_text_width_height_descent(candidate, font, False)[0] <= width
            if fits or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def save_figure(fig: Figure, path: Path) -> Path:
    """Write `fig` to `path` in the format its suffix names, transparent and reproducible.

    The file has no creation date, so writing the same figure twice gives the same bytes.
    """
    with lab_style():
        fig.savefig(path, transparent=True, metadata=_metadata(path))
    return path


def _metadata(path: Path) -> dict[str, Any]:
    match path.suffix.lower():
        case ".svg":
            return {"Date": None}
        case ".pdf":
            return {"CreationDate": None, "ModDate": None}
        case ".png":
            return {"Software": None}
        case _:
            return {}
