"""Tiny CelebA fetch roots for tests: image shards and the label table, laid out as fetched."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ryuk.datasets.celeba import labels_path, shards_dir
from synthetic import face

ATTRIBUTES = ("Blurry", "Eyeglasses", "Male", "Wearing_Hat", "Young")


@dataclass(frozen=True)
class Row:
    celeb_id: int
    image: NDArray[np.uint8]
    attributes: Mapping[str, bool] = field(default_factory=dict)
    """Attributes not given are False."""


def png(image: NDArray[np.uint8]) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def write_celeba(
    root: Path, shards: Mapping[str, Sequence[Row]], *, row_group_size: int = 2
) -> pa.Table:
    """Write each shard, named `<split>-NNNNN-of-NNNNN.parquet`, and the label table for them.

    Image paths are `<shard stem>-<row>.png`, so every image's path is unique.
    """
    shards_dir(root).mkdir(parents=True, exist_ok=True)
    labels: list[pa.Table] = []
    for name, rows in sorted(shards.items()):
        paths = [f"{name.removesuffix('.parquet')}-{i}.png" for i in range(len(rows))]
        images = pa.array(
            [
                {"bytes": png(row.image), "path": path}
                for row, path in zip(rows, paths, strict=True)
            ],
            pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        )
        ids = pa.array([row.celeb_id for row in rows], pa.int64())
        flags = {a: pa.array([row.attributes.get(a, False) for row in rows]) for a in ATTRIBUTES}
        table = pa.table({"image": images, "celeb_id": ids, **flags})
        pq.write_table(table, shards_dir(root) / name, row_group_size=row_group_size)

        count = len(rows)
        labels.append(
            pa.table(
                {
                    "split": pa.array([name.split("-", 1)[0]] * count, pa.string()),
                    "shard": pa.array([name] * count, pa.string()),
                    "row_group": pa.array([i // row_group_size for i in range(count)], pa.int16()),
                    "row_in_file": pa.array(range(count), pa.int32()),
                    "path": pa.array(paths, pa.string()),
                    "celeb_id": ids,
                    **flags,
                }
            )
        )
    table = pa.concat_tables(labels)
    labels_path(root).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, labels_path(root))
    return table


def solid(value: int, height: int = 8, width: int = 6) -> NDArray[np.uint8]:
    """A small uniform BGR image, told apart from others by its value."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def rehearsal_split(base: int) -> list[Row]:
    """Identities base..base+4 in five looks: two can be enrolled (21 usable images each, one
    of them with a blank besides), and three are held out with 3, 12 and 1 usable images."""
    blank = np.full((250, 250, 3), 127, dtype=np.uint8)
    counts = {base: 21, base + 1: 21, base + 2: 3, base + 3: 12, base + 4: 1}
    rows = [
        Row(identity, face(look, shot))
        for look, (identity, count) in enumerate(counts.items())
        for shot in range(count)
    ]
    return [*rows, Row(base, blank), Row(base + 4, blank)]


def write_rehearsal(root: Path) -> None:
    """A CelebA fetch root for the watchlist rehearsal with a gallery of 2: identities 100-104
    in the validation split and 200-204 in the test split, each laid out by `rehearsal_split`."""
    write_celeba(
        root,
        {
            "valid-00000-of-00001.parquet": rehearsal_split(100),
            "test-00000-of-00001.parquet": rehearsal_split(200),
        },
        row_group_size=8,
    )
