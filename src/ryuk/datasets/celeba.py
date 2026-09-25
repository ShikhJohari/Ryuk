"""CelebA as fetched: the label table for every split, and the valid and test splits' images.

The label table (`celeba/metadata/celeba_meta.parquet`) has one row per image: its split, where
it sits (`shard`, `row_group`, `row_in_file`), its file name (`path`), its identity (`celeb_id`)
and the 40 attributes. The images are PNG bytes in the shards' `image` struct column. They are
read one row group at a time, about 100 images, so a whole split never sits in memory.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ryuk.datasets import DatasetError
from ryuk.detector import Image
from ryuk.fetch.celeba import CELEBA

type Split = Literal["train", "valid", "test"]

LOCATION_COLUMNS: Final = ("split", "shard", "row_group", "row_in_file", "path", "celeb_id")
"""The label table's columns that say which image a row is and where to find it."""


def labels_path(root: Path) -> Path:
    """The label table under the fetch root (`RYUK_DATA_DIR`)."""
    return root.joinpath(CELEBA.labels)


def shards_dir(root: Path) -> Path:
    """The folder holding the image shards under the fetch root."""
    return root.joinpath(CELEBA.shards[0].path.parent)


def read_labels(
    root: Path, split: Split | None = None, attributes: Sequence[str] | None = None
) -> pa.Table:
    """The label rows of `split` (every split if None), in shard and row order.

    The table has the location columns, then `attributes` (all 40 if None).
    """
    schema = pq.read_schema(labels_path(root))
    if missing := [name for name in LOCATION_COLUMNS if name not in schema.names]:
        raise DatasetError(f"{CELEBA.labels} has no {', '.join(missing)} column")
    if attributes is None:
        attributes = [name for name in schema.names if name not in LOCATION_COLUMNS]
    table = pq.read_table(
        labels_path(root),
        columns=[*LOCATION_COLUMNS, *attributes],
        filters=None if split is None else [("split", "=", split)],
    )
    return table.sort_by([("shard", "ascending"), ("row_in_file", "ascending")])


@dataclass(frozen=True, slots=True)
class CelebaImage:
    row: int
    """The row of the label table the image was read for."""
    path: str
    """CelebA's file name for the image, such as `000001.png`."""
    celeb_id: int
    png: bytes

    def decode(self) -> Image:
        """The image as BGR uint8, as the detector takes it."""
        decoded = cv2.imdecode(np.frombuffer(self.png, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise DatasetError(f"{self.path} does not decode as an image")
        return np.asarray(decoded, dtype=np.uint8)


def iter_images(root: Path, labels: pa.Table) -> Iterator[CelebaImage]:
    """The image for each row of `labels`, a selection of the label table's rows.

    Images come in shard and row order, whatever the order of `labels`; `CelebaImage.row` says
    which row of `labels` each one is. Only the row groups holding a selected image are read.
    """
    shards = [str(name) for name in labels.column("shard").to_pylist()]
    paths = [str(name) for name in labels.column("path").to_pylist()]
    groups: list[int] = labels.column("row_group").to_numpy().tolist()
    rows_in_file: list[int] = labels.column("row_in_file").to_numpy().tolist()
    celeb_ids: list[int] = labels.column("celeb_id").to_numpy().tolist()

    order = sorted(range(labels.num_rows), key=lambda i: (shards[i], rows_in_file[i]))
    for shard, in_shard in groupby(order, key=lambda i: shards[i]):
        parquet = pq.ParquetFile(shards_dir(root) / shard)
        sizes = [parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)]
        for group, in_group in groupby(in_shard, key=lambda i: groups[i]):
            images = parquet.read_row_group(group, columns=["image"]).column("image")
            structs = images.combine_chunks()
            group_start = sum(sizes[:group])
            for i in in_group:
                offset = rows_in_file[i] - group_start
                if not 0 <= offset < len(structs):
                    raise DatasetError(
                        f"{shard}: row {rows_in_file[i]} is not in row group {group}"
                    )
                image = structs[offset]
                if (path := image["path"].as_py()) != paths[i]:
                    raise DatasetError(
                        f"{shard}: row {rows_in_file[i]} holds {path}, labelled {paths[i]}"
                    )
                yield CelebaImage(
                    row=i, path=paths[i], celeb_id=celeb_ids[i], png=image["bytes"].as_py()
                )
