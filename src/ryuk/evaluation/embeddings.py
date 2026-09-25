"""The benchmark embedding cache: one embedding per image, per recognition model, per pipeline.

It lives under the cache directory, never in the app database (#22, user story 58). An image
is keyed by the sha256 of its bytes, so a changed file is never served a stale embedding. The
pipeline names everything between the image and the model that could change the face: the
detector's weights, the minimum usable face size and the crop. An image whose pipeline found
no usable face is cached too, as None, so it is not detected again.

Each (model, pipeline) is one Parquet file, rewritten whole and renamed into place on save.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ryuk.fetch.pinned import write_into_place
from ryuk.recognition import Embedding, ModelKey

type Cached = Embedding | None
"""An image's embedding, or None when the pipeline found no usable face in it."""


def image_key(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


class EmbeddingCache:
    def __init__(self, root: Path) -> None:
        self._root = root

    def path(self, model: ModelKey, pipeline: str) -> Path:
        return self._root / model.id / f"{pipeline}.parquet"

    def load(self, model: ModelKey, pipeline: str, dimension: int) -> dict[str, Cached]:
        """Everything cached for this model and pipeline, by image key."""
        path = self.path(model, pipeline)
        if not path.is_file():
            return {}
        table = pq.read_table(path)
        images = [str(image) for image in table.column("image").to_pylist()]
        usable = table.column("usable").to_numpy()
        column = table.column("embedding").combine_chunks()
        if not isinstance(column, pa.FixedSizeListArray) or column.type.list_size != dimension:
            raise ValueError(f"{path} holds {column.type} embeddings, expected {dimension}")
        rows = column.flatten().to_numpy().reshape(-1, dimension).astype(np.float32)
        return {
            image: rows[index].copy() if usable[index] else None
            for index, image in enumerate(images)
        }

    def save(
        self, model: ModelKey, pipeline: str, dimension: int, entries: Mapping[str, Cached]
    ) -> None:
        """Merge `entries` into what is cached for this model and pipeline."""
        merged = {**self.load(model, pipeline, dimension), **entries}
        images = sorted(merged)
        # Parquet does not round-trip null fixed-size lists, so a miss is a zero row flagged
        # as not usable.
        rows = np.zeros((len(images), dimension), dtype=np.float32)
        usable = np.zeros(len(images), dtype=np.bool_)
        for index, image in enumerate(images):
            embedding = merged[image]
            if embedding is not None:
                if embedding.shape != (dimension,):
                    raise ValueError(f"{image}: embedding of shape {embedding.shape}")
                rows[index] = embedding
                usable[index] = True
        table = pa.table(
            {
                "image": pa.array(images, type=pa.string()),
                "usable": pa.array(usable),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(rows.reshape(-1)), dimension
                ),
            }
        )
        write_into_place(self.path(model, pipeline), lambda part: pq.write_table(table, part))
