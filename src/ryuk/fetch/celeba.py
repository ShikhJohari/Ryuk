"""CelebA from `flwrlabs/celeba` on the Hugging Face Hub, at one pinned revision.

The valid and test splits are the evaluation's validation and test draws, so their shards are
downloaded whole and checked against the Hub's LFS sha256s. Labels (identity and the 40
attributes) are kept for every split, train included, but the train images are not: its label
columns are read from the remote shards with range requests, skipping the image bytes (#8).

The label table is derived, so it is pinned by a digest of its content rather than its bytes,
which depend on the Parquet writer.
"""

import hashlib
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ryuk.fetch.http import RangeFile
from ryuk.fetch.pinned import (
    Checksum,
    ChecksumMismatchError,
    Fetched,
    PinnedFile,
    fetch_pinned,
    part_path,
)

logger = logging.getLogger(__name__)

REVISION = "2d738f56e0e7f925ea36ae7c808ea925264aacec"
_CONFIG = "img_align+identity+attr"
_IMAGE_BYTES_COLUMN = "image.bytes"
_DOWNLOADS = 3
_RANGE_READS = 16


@dataclass(frozen=True)
class CelebaSource:
    shards: tuple[PinnedFile, ...]
    """Every split's Parquet shards, named `<split>-NNNNN-of-NNNNN.parquet`."""
    image_splits: frozenset[str]
    """The splits whose shards are downloaded whole, images included."""
    license: PinnedFile
    labels: PurePosixPath
    labels_rows: int
    labels_digest: str
    """`labels_digest()` of the label table the dataset fetch recorded."""


def _hub(path: str, size: int, sha256: str) -> PinnedFile:
    return PinnedFile(
        url=f"https://huggingface.co/datasets/flwrlabs/celeba/resolve/{REVISION}/{path}",
        path=PurePosixPath("celeba", "parquet" if path != "LICENSE" else "", path),
        size=size,
        checksum=Checksum("sha256", sha256),
    )


# From https://huggingface.co/api/datasets/flwrlabs/celeba/revision/<REVISION>?blobs=true
_SHARDS = {
    "test-00000-of-00003": (
        391_177_633,
        "c22a47cdec9e5b36b004c6af906ab6d49066e52a27e74440b55eae3dd9c8d44a",
    ),
    "test-00001-of-00003": (
        383_959_773,
        "205df6b286ccbdf9b2e67205cc86380fcdbe7673bf87c796a629bd7fafeda1b3",
    ),
    "test-00002-of-00003": (
        383_093_383,
        "db9c0833ab64e6a156530f87b54b1da931ce5d10731ecb9789238503b009ef97",
    ),
    "train-00000-of-00019": (
        500_402_070,
        "9f349da30bb755c3af41929270d6c5560121dd09121616980f05cb4e7b00b988",
    ),
    "train-00001-of-00019": (
        498_251_284,
        "5e6f2800d1ba6653072340daea433a9eb09b1c25d6a60d16158396e45f9f67ee",
    ),
    "train-00002-of-00019": (
        494_123_817,
        "57728860cfa53a153f90d5ef9e0af96d946209c3add1e484ace6560922d92551",
    ),
    "train-00003-of-00019": (
        490_434_458,
        "c676ff8b9290037f8228a58e69276bd5886cf5d5ca22926cdc7d39fdbbc3fac4",
    ),
    "train-00004-of-00019": (
        494_342_449,
        "94fdfb5f8f95011680eac2c454a18035411678a9e143ef476160e68771cf32bb",
    ),
    "train-00005-of-00019": (
        503_422_722,
        "0d83c60365c00cf466f2f55a8d36092c7982526218bf6884dd9334f41f96100c",
    ),
    "train-00006-of-00019": (
        494_063_518,
        "0c80fe536cb1d4239c2f7ada134da9858968c66023bc6a2976a8bc05b04fcf36",
    ),
    "train-00007-of-00019": (
        492_511_619,
        "8d0761963dfa3040d79daa2f7f30268da922e0d6fb5c1aed82d796ed0f718f2e",
    ),
    "train-00008-of-00019": (
        497_445_734,
        "265747c6c3cf8a87f18fba27c00dd7ae7e6c39287e7fffc885918d1a714ab592",
    ),
    "train-00009-of-00019": (
        503_059_054,
        "9f675cc97f7227e5ecba5b999b4599a46460d77cf0a9c3f9f3949213093ef3e5",
    ),
    "train-00010-of-00019": (
        498_101_294,
        "64974dcedcba2b76b521af639d9b6d5dba4df8e36ff9aa7a11853f82ecc6be8f",
    ),
    "train-00011-of-00019": (
        501_186_917,
        "011dca62f5569645a12d80d4ea49f01660de32a0d650bafe3fb3e4f9db272bbf",
    ),
    "train-00012-of-00019": (
        493_796_347,
        "20ede3eda874dfe3ef42d553bd4b395d3197a07c7651f2ed9f741084e7498985",
    ),
    "train-00013-of-00019": (
        503_616_679,
        "d73541045d8db00ff48593c81c76e915d3adf7e9ae3048b66d4c02ca01a6488e",
    ),
    "train-00014-of-00019": (
        490_108_044,
        "01dc8362d841d325ae9543c83eea4eb508c2a3f17122167baf0c04d242fd68af",
    ),
    "train-00015-of-00019": (
        489_097_029,
        "0611aba2a5381a728407fbc5368aab63b53d319155d9c867fe99b234219b9efc",
    ),
    "train-00016-of-00019": (
        497_714_156,
        "b0fb099f260d4b6cb582880cddbc92ff11dc68cb4ca72870e24542810961b157",
    ),
    "train-00017-of-00019": (
        489_160_281,
        "cb4fd9c4121c3b5e807562005786ae486ddc7e1cc9a91619736e4359533cb811",
    ),
    "train-00018-of-00019": (
        489_124_107,
        "feb6cf9d6a28e836f113098670a8b26d4c31d4f683a3e1a21a58371f68cfb6bb",
    ),
    "valid-00000-of-00003": (
        387_827_665,
        "d4bbce4bdd8c2bff49a3546cbe2e18bd2bd7312d2ebd95efd29aed6ba5df05db",
    ),
    "valid-00001-of-00003": (
        385_032_335,
        "76ddf9a3d15728dd00172f365eb76d0f852038fa08bfdc6b72e732e671c2daca",
    ),
    "valid-00002-of-00003": (
        383_642_321,
        "94da82da5405e159594a61b6fa6477aa7b5103d442d89dd9428841192c187b05",
    ),
}

CELEBA = CelebaSource(
    shards=tuple(
        _hub(f"{_CONFIG}/{name}.parquet", size, sha256) for name, (size, sha256) in _SHARDS.items()
    ),
    image_splits=frozenset({"valid", "test"}),
    license=_hub(
        "LICENSE", 859, "f13d0af838cb5e89a77a9fd2860f2f8dd531c10a123acd288ec24b7bee36c8d4"
    ),
    labels=PurePosixPath("celeba/metadata/celeba_meta.parquet"),
    labels_rows=202_599,
    labels_digest="cef70c2c20d9397b0c0d928988aeab1365fa639289fb2094c70f2475cff32646",
)


def fetch_celeba(root: Path, source: CelebaSource = CELEBA) -> list[Fetched]:
    downloads = [source.license, *(s for s in source.shards if _split(s) in source.image_splits)]
    with ThreadPoolExecutor(_DOWNLOADS) as pool:
        fetched = list(pool.map(lambda pinned: fetch_pinned(pinned, root), downloads))
    fetched.append(_fetch_labels(root, source))
    return fetched


def labels_digest(table: pa.Table) -> str:
    """A sha256 over the table's column names, types and values, independent of file layout."""
    hasher = hashlib.sha256()
    for field, column in zip(table.schema, table.columns, strict=True):
        hasher.update(f"{field.name}\0{field.type}\0".encode())
        if column.null_count:
            raise ValueError(f"label column {field.name} has nulls")
        if pa.types.is_string(field.type):
            hasher.update("\0".join(str(value) for value in column.to_pylist()).encode())
        else:
            hasher.update(column.to_numpy().astype("<i8").tobytes())
    return hasher.hexdigest()


def _split(shard: PinnedFile) -> str:
    return shard.path.name.split("-", 1)[0]


def _fetch_labels(root: Path, source: CelebaSource) -> Fetched:
    target = root.joinpath(source.labels)
    if target.is_file() and labels_digest(pq.read_table(target)) == source.labels_digest:
        logger.info("%s already verified", source.labels)
        return Fetched(target, updated=False)

    shards = sorted(source.shards, key=lambda shard: shard.path.name)
    table = pa.concat_tables([_shard_labels(root, shard, source) for shard in shards])
    digest = labels_digest(table)
    if table.num_rows != source.labels_rows or digest != source.labels_digest:
        raise ChecksumMismatchError(
            f"{source.labels}: built {table.num_rows} rows with digest {digest}, "
            f"pinned at {source.labels_rows} rows with digest {source.labels_digest}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(target)
    try:
        pq.write_table(table, part)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    logger.info("%s verified (%d rows)", source.labels, table.num_rows)
    return Fetched(target, updated=True)


def _shard_labels(root: Path, shard: PinnedFile, source: CelebaSource) -> pa.Table:
    """Every column but the image bytes, with where each row sits in its shard."""
    if _split(shard) in source.image_splits:
        parquet = pq.ParquetFile(root.joinpath(shard.path))
    else:
        logger.info("reading labels from %s", shard.url)
        remote = RangeFile(shard.url, shard.size)
        parquet = pq.ParquetFile(pa.PythonFile(remote, mode="r"))
        _prefetch(remote, parquet, _label_columns(parquet))
    columns = _label_columns(parquet)
    table = parquet.read(columns=[parquet.metadata.schema.column(i).path for i in columns])

    metadata = parquet.metadata
    rows_per_group = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    labels = {
        "split": pa.array([_split(shard)] * table.num_rows, pa.string()),
        "shard": pa.array([shard.path.name] * table.num_rows, pa.string()),
        "row_group": pa.array(
            np.repeat(np.arange(len(rows_per_group)), rows_per_group), pa.int16()
        ),
        "row_in_file": pa.array(np.arange(table.num_rows), pa.int32()),
        "path": pc.struct_field(table.column("image"), "path"),
    }
    for name in table.column_names:
        if name != "image":
            labels[name] = table.column(name)
    return pa.table(labels)


def _label_columns(parquet: pq.ParquetFile) -> list[int]:
    schema = parquet.metadata.schema
    return [i for i in range(len(schema)) if schema.column(i).path != _IMAGE_BYTES_COLUMN]


def _prefetch(remote: RangeFile, parquet: pq.ParquetFile, columns: Sequence[int]) -> None:
    """Read each row group's label columns in parallel, one span per row group.

    Parquet stores a row group's columns back to back, so its label columns form one short span
    between image blobs. Fetched one by one they cost a CDN round trip each (#8).
    """
    metadata = parquet.metadata
    spans = []
    for group in range(metadata.num_row_groups):
        bounds = [_chunk_bounds(metadata.row_group(group).column(i)) for i in columns]
        start = min(first for first, _ in bounds)
        spans.append((start, max(end for _, end in bounds) - start))
    with ThreadPoolExecutor(_RANGE_READS) as pool:
        spans_read = pool.map(lambda span: remote.fetch(*span), spans)
        for (start, _), data in zip(spans, spans_read, strict=True):
            remote.prefetched(start, data)


def _chunk_bounds(chunk: pq.ColumnChunkMetaData) -> tuple[int, int]:
    """Where a column chunk's bytes start and end, dictionary page included."""
    dictionary = chunk.dictionary_page_offset
    start = chunk.data_page_offset if dictionary is None else dictionary
    return start, start + chunk.total_compressed_size
