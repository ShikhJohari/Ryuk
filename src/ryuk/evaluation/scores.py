"""Per-probe raw scores (#10): every probe of a draw with each method's score for its top
candidate, kept as Parquet under the cache directory for re-plotting and fresh bootstraps.

Never committed and never read by the notebooks or the report, which read `results.json` only;
`ryuk evaluate learn` rewrites the file from the cached embeddings. One file per recognition model
and draw, named by the draw's `selection_sha256`, so a file from another draw is never mistaken
for this one.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ryuk.evaluation.openset import ScoredProbes
from ryuk.evaluation.results import Method
from ryuk.fetch.pinned import write_into_place
from ryuk.recognition import ModelKey


def scores_path(root: Path, model: ModelKey, selection_sha256: str) -> Path:
    """Where one model's scores on one draw live under `root`, the cache's `scores` folder."""
    return root / model.id / f"{selection_sha256}.parquet"


def scores_table(
    images: Sequence[str],
    scored: Mapping[Method, ScoredProbes],
    runner_up: NDArray[np.float64],
) -> pa.Table:
    """One row per probe, mated then non-mated as in `ScoredProbes`.

    Columns: `image`, `identity`, `mated`, the best-photo `runner_up` score, and per method
    `<method>` (its top candidate's score) and `<method>_right` (whether a mated probe's top
    candidate is right; null for a non-mated probe).
    """
    if not scored:
        raise ValueError("need at least one method's scores")
    first = next(iter(scored.values()))
    mated = first.mated_score.size
    probes = mated + first.non_mated_score.size
    if len(images) != probes or runner_up.shape != (probes,):
        raise ValueError(f"expected {probes} images and runner-up scores, one per probe")
    columns: dict[str, pa.Array[Any]] = {
        "image": pa.array(images, type=pa.string()),
        "identity": pa.array(
            np.concatenate([first.mated_identity, first.non_mated_identity]), type=pa.int64()
        ),
        "mated": pa.array(np.arange(probes) < mated),
        "runner_up": pa.array(runner_up, type=pa.float64()),
    }
    for method, result in scored.items():
        if not (
            np.array_equal(result.mated_identity, first.mated_identity)
            and np.array_equal(result.non_mated_identity, first.non_mated_identity)
        ):
            raise ValueError(f"{method} was scored on other probes")
        columns[method] = pa.array(
            np.concatenate([result.mated_score, result.non_mated_score]), type=pa.float64()
        )
        right = np.concatenate([result.mated_correct, np.zeros(result.non_mated_score.size, bool)])
        mask = np.arange(probes) >= mated
        columns[f"{method}_right"] = pa.array(right, mask=mask, type=pa.bool_())
    return pa.table(columns)


def write_scores(path: Path, table: pa.Table) -> None:
    write_into_place(path, lambda part: pq.write_table(table, part))
