"""The per-probe raw scores file (#10)."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from ryuk.evaluation.openset import ScoredProbes
from ryuk.evaluation.scores import scores_path, scores_table, write_scores
from ryuk.recognition import ModelKey


def _scored(shift: float = 0.0) -> ScoredProbes:
    return ScoredProbes(
        draw="test",
        mated_identity=np.array([1, 1, 2]),
        mated_score=np.array([0.9, 0.4, 0.7]) + shift,
        mated_correct=np.array([True, False, True]),
        non_mated_identity=np.array([7, 8]),
        non_mated_score=np.array([0.3, 0.1]) + shift,
    )


IMAGES = ["a.png", "b.png", "c.png", "d.png", "e.png"]


def test_each_probe_is_one_row_with_every_methods_score(tmp_path: Path) -> None:
    table = scores_table(
        IMAGES, {"best-photo": _scored(), "mean": _scored(0.05)}, np.linspace(0, 0.4, 5)
    )
    path = scores_path(tmp_path, ModelKey("sface", "f" * 64, "cpu"), "c" * 64)
    write_scores(path, table)

    read = pq.read_table(path).to_pydict()
    assert read["image"] == IMAGES
    assert read["identity"] == [1, 1, 2, 7, 8]
    assert read["mated"] == [True, True, True, False, False]
    assert read["mean"] == pytest.approx([0.95, 0.45, 0.75, 0.35, 0.15])
    # A non-mated probe has no right or wrong top candidate.
    assert read["best-photo_right"] == [True, False, True, None, None]
    assert path.parent.name.startswith("sface-cpu-")


def test_methods_scored_on_other_probes_are_refused() -> None:
    other = ScoredProbes(
        draw="test",
        mated_identity=np.array([1, 2, 2]),
        mated_score=np.zeros(3),
        mated_correct=np.ones(3, bool),
        non_mated_identity=np.array([7, 8]),
        non_mated_score=np.zeros(2),
    )

    with pytest.raises(ValueError, match="other probes"):
        scores_table(IMAGES, {"best-photo": _scored(), "mean": other}, np.zeros(5))


def test_one_image_per_probe_is_required() -> None:
    with pytest.raises(ValueError, match="one per probe"):
        scores_table(IMAGES[:4], {"best-photo": _scored()}, np.zeros(5))
