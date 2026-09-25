"""The LFW ROC figure: every recognition model's TAR against FAR on View 2, log FAR axis.

The per-network colour and dash follow #13 (ArcFace solid navy, FaceNet long-dash violet, SFace
dotted ochre). Once the shared `ryuk.plotting` module from #25 is on main, this draws through
it instead of its own style table.
"""

import math
from typing import Final

from matplotlib.figure import Figure

from ryuk.evaluation.results import Verification
from ryuk.evaluation.tables import model_name
from ryuk.recognition import Network

STYLES: Final[dict[Network, tuple[str, str | tuple[int, tuple[int, int]]]]] = {
    "arcface": ("#1F4E8C", "-"),
    "facenet": ("#8A4FA0", (0, (8, 3))),
    "sface": ("#C98A1B", ":"),
}
MIN_FAR: Final = 1e-4
"""The left edge of the FAR axis; View 2's 3,000 negatives cannot resolve below 1/3,000."""


def lfw_roc(verification: Verification) -> Figure:
    figure = Figure(figsize=(6.4, 4.4), layout="constrained")
    axes = figure.add_subplot()
    lowest = 1.0
    for result in verification.models:
        colour, dash = STYLES[result.model.network]
        # A log axis cannot show FAR 0; the curve starts where its first false accept does.
        points = [(f, t) for f, t in zip(result.roc.far, result.roc.tar, strict=True) if f > 0]
        far = [MIN_FAR, *(f for f, _ in points)]
        tar = [_tar_before(result.roc.far, result.roc.tar), *(t for _, t in points)]
        label = f"{model_name(result.model)}, {result.accuracy * 100:.2f}%"
        axes.step(far, tar, where="post", color=colour, linestyle=dash, linewidth=1.6, label=label)
        lowest = min(lowest, *(t for f, t in zip(far, tar, strict=True) if f >= 1e-3))
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
