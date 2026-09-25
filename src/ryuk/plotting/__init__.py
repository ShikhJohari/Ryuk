"""Every figure Ryuk draws, in #13's lab-notebook style, for the notebooks and the report (#17).

`style` holds the palette, fonts, per-model and per-dataset series styles and the direct-label
and caption helpers; `eda` draws the dataset analysis from the committed summary.
"""

from ryuk.plotting.eda import EDA_FIGURES, eda_figure, save_eda_figures
from ryuk.plotting.style import (
    DATASET_STYLES,
    MODEL_STYLES,
    Dataset,
    Network,
    SeriesStyle,
    caption,
    direct_label,
    lab_style,
    new_figure,
    panel_title,
    register_fonts,
    save_figure,
    use_lab_style,
)

__all__ = [
    "DATASET_STYLES",
    "EDA_FIGURES",
    "MODEL_STYLES",
    "Dataset",
    "Network",
    "SeriesStyle",
    "caption",
    "direct_label",
    "eda_figure",
    "lab_style",
    "new_figure",
    "panel_title",
    "register_fonts",
    "save_eda_figures",
    "save_figure",
    "use_lab_style",
]
