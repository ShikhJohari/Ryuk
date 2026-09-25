"""The evaluation figures: the LFW ROC, every recognition model's TAR against FAR on View 2, and
the CelebA TPIR against FPIR on a draw, both on a log false-alarm axis.

Each network's colour and dash are #13's, from the shared `ryuk.plotting.MODEL_STYLES`
(ArcFace solid navy, FaceNet long-dash violet, SFace dotted ochre).
"""

import math
from typing import Final

from matplotlib.figure import Figure

from ryuk.eda.summary import Draw
from ryuk.evaluation.results import Identification, Verification
from ryuk.evaluation.tables import model_name
from ryuk.plotting import MODEL_STYLES, direct_label, new_figure

MIN_FAR: Final = 1e-4
"""The left edge of the FAR axis; View 2's 3,000 negatives cannot resolve below 1/3,000."""


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
    """TAR at FAR 0: the highest TAR reached before the first false accept."""
    return max(t for f, t in zip(far, tar, strict=True) if f == 0)


def openset_curves(identification: Identification, draw: Draw = "test") -> Figure:
    """TPIR against FPIR on one CelebA draw, log FPIR axis, each model's frozen threshold marked.

    Each model is labelled at its frozen operating point rather than in a legend (#13).
    """
    figure = new_figure(height=4.2)
    axes = figure.add_subplot()
    selection = next(d for d in identification.draws if d.draw == draw)
    # One false alarm is the smallest FPIR the draw's non-mated probes can show.
    floor = 10 ** math.floor(math.log10(1 / selection.non_mated_probes))
    lowest = 1.0
    for model in identification.models:
        result = model.test if draw == "test" else model.validation
        style = MODEL_STYLES[model.model.network]
        curve = result.curve
        points = [(f, t) for f, t in zip(curve.fpir, curve.tpir, strict=True) if f > 0]
        at_zero = max(t for f, t in zip(curve.fpir, curve.tpir, strict=True) if f == 0)
        fpir = [floor, *(f for f, _ in points)]
        tpir = [at_zero, *(t for _, t in points)]
        axes.step(fpir, tpir, where="post", linewidth=1.6, **style.line())
        point = result.at_threshold
        axes.plot(
            [max(point.fpir.value, floor)], [point.tpir.value], markersize=6, **style.points()
        )
        direct_label(
            axes,
            (max(point.fpir.value, floor), point.tpir.value),
            model_name(model.model),
            colour=style.colour,
            offset=(6.0, -8.0),
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
