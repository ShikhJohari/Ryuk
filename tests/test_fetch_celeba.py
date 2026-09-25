"""CelebA: the valid and test shards in full, and every split's labels without the images."""

import hashlib
import io
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from file_server import FileServer
from ryuk.fetch import FetchError
from ryuk.fetch.celeba import CELEBA, CelebaSource, fetch_celeba, labels_digest
from ryuk.fetch.pinned import Checksum, ChecksumMismatchError, PinnedFile

IMAGE_BYTES = 100_000  # incompressible stand-ins for PNGs, which dominate each shard on the Hub


def shard(first_id: int, rows: int) -> bytes:
    table = pa.table(
        {
            "image": pa.array(
                [
                    {
                        "bytes": np.random.default_rng(row).bytes(IMAGE_BYTES),
                        "path": f"{first_id + row}.png",
                    }
                    for row in range(rows)
                ],
                pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "celeb_id": pa.array([(first_id + row) // 3 for row in range(rows)], pa.int64()),
            "Male": pa.array([row % 2 == 0 for row in range(rows)]),
            "Young": pa.array([row % 3 == 0 for row in range(rows)]),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, row_group_size=2)
    return buffer.getvalue()


def pin(server: FileServer, path: str, content: bytes) -> PinnedFile:
    return PinnedFile(
        url=server.serve(path, content),
        path=PurePosixPath("celeba", path),
        size=len(content),
        checksum=Checksum("sha256", hashlib.sha256(content).hexdigest()),
    )


SHARDS = {
    "test-00000-of-00001.parquet": shard(100, 3),
    "train-00000-of-00001.parquet": shard(200, 5),
    "valid-00000-of-00001.parquet": shard(300, 2),
}
EXPECTED_ROWS = [
    ("test", "test-00000-of-00001.parquet", 0, 0, "100.png", 33, True, True),
    ("test", "test-00000-of-00001.parquet", 0, 1, "101.png", 33, False, False),
    ("test", "test-00000-of-00001.parquet", 1, 2, "102.png", 34, True, False),
    ("train", "train-00000-of-00001.parquet", 0, 0, "200.png", 66, True, True),
    ("train", "train-00000-of-00001.parquet", 0, 1, "201.png", 67, False, False),
    ("train", "train-00000-of-00001.parquet", 1, 2, "202.png", 67, True, False),
    ("train", "train-00000-of-00001.parquet", 1, 3, "203.png", 67, False, True),
    ("train", "train-00000-of-00001.parquet", 2, 4, "204.png", 68, True, False),
    ("valid", "valid-00000-of-00001.parquet", 0, 0, "300.png", 100, True, True),
    ("valid", "valid-00000-of-00001.parquet", 0, 1, "301.png", 100, False, False),
]
COLUMNS = ["split", "shard", "row_group", "row_in_file", "path", "celeb_id", "Male", "Young"]


def expected_labels() -> pa.Table:
    columns = list(zip(*EXPECTED_ROWS, strict=True))
    types = [pa.string(), pa.string(), pa.int16(), pa.int32(), pa.string(), pa.int64()]
    types += [pa.bool_(), pa.bool_()]
    return pa.table(
        [pa.array(values, kind) for values, kind in zip(columns, types, strict=True)],
        names=COLUMNS,
    )


def source(server: FileServer, *, digest: str | None = None) -> CelebaSource:
    return CelebaSource(
        shards=tuple(
            pin(server, f"parquet/img_align+identity+attr/{name}", content)
            for name, content in SHARDS.items()
        ),
        image_splits=frozenset({"valid", "test"}),
        license=pin(server, "LICENSE", b"CelebA Dataset Release Agreement"),
        labels=PurePosixPath("celeba/metadata/celeba_meta.parquet"),
        labels_rows=len(EXPECTED_ROWS),
        labels_digest=digest or labels_digest(expected_labels()),
    )


def shard_path(name: str) -> str:
    return f"/parquet/img_align+identity+attr/{name}"


def test_the_image_splits_are_downloaded_whole_and_verified(
    file_server: FileServer, tmp_path: Path
) -> None:
    fetch_celeba(tmp_path, source(file_server))

    folder = tmp_path / "celeba" / "parquet" / "img_align+identity+attr"
    assert sorted(path.name for path in folder.iterdir()) == [
        "test-00000-of-00001.parquet",
        "valid-00000-of-00001.parquet",
    ]
    assert (tmp_path / "celeba" / "LICENSE").read_bytes() == b"CelebA Dataset Release Agreement"


def test_the_labels_cover_every_split_with_each_row_located_in_its_shard(
    file_server: FileServer, tmp_path: Path
) -> None:
    fetch_celeba(tmp_path, source(file_server))

    labels = pq.read_table(tmp_path / "celeba" / "metadata" / "celeba_meta.parquet")
    assert labels.equals(expected_labels())


def test_labels_of_a_split_without_images_are_read_without_its_images(
    file_server: FileServer, tmp_path: Path
) -> None:
    fetch_celeba(tmp_path, source(file_server))

    train = "train-00000-of-00001.parquet"
    # Reading the footer costs one 64 KiB tail read; the five images are 500 kB.
    assert file_server.served_bytes[shard_path(train)] < len(SHARDS[train]) / 4
    # The downloaded shards were read from disk, not fetched a second time.
    assert file_server.requests.count(shard_path("test-00000-of-00001.parquet")) == 1


def test_a_second_run_fetches_nothing(file_server: FileServer, tmp_path: Path) -> None:
    celeba = source(file_server)
    fetch_celeba(tmp_path, celeba)
    file_server.requests.clear()

    fetched = fetch_celeba(tmp_path, celeba)

    assert not any(item.updated for item in fetched)
    assert file_server.requests == []


def test_labels_that_do_not_match_their_pin_are_never_written(
    file_server: FileServer, tmp_path: Path
) -> None:
    with pytest.raises(ChecksumMismatchError, match=r"celeba_meta\.parquet"):
        fetch_celeba(tmp_path, source(file_server, digest="0" * 64))

    assert list(tmp_path.rglob("celeba_meta*")) == []


def test_a_labels_file_that_no_longer_matches_is_rebuilt(
    file_server: FileServer, tmp_path: Path
) -> None:
    celeba = source(file_server)
    fetch_celeba(tmp_path, celeba)
    labels = tmp_path / "celeba" / "metadata" / "celeba_meta.parquet"
    pq.write_table(expected_labels().slice(1), labels)

    fetch_celeba(tmp_path, celeba)

    assert pq.read_table(labels).equals(expected_labels())


def test_an_unreadable_labels_file_is_rebuilt(file_server: FileServer, tmp_path: Path) -> None:
    celeba = source(file_server)
    fetch_celeba(tmp_path, celeba)
    labels = tmp_path / "celeba" / "metadata" / "celeba_meta.parquet"
    labels.write_bytes(b"not parquet")

    fetch_celeba(tmp_path, celeba)

    assert pq.read_table(labels).equals(expected_labels())


def test_a_shard_that_is_not_parquet_fails_as_a_fetch_error(
    file_server: FileServer, tmp_path: Path
) -> None:
    celeba = source(file_server)
    train = "train-00000-of-00001.parquet"
    file_server.serve(shard_path(train)[1:], b"x" * len(SHARDS[train]))

    with pytest.raises(FetchError, match=r"train-00000-of-00001\.parquet"):
        fetch_celeba(tmp_path, celeba)


def test_the_digest_covers_values_names_and_types() -> None:
    labels = expected_labels()
    flipped = labels.set_column(6, "Male", pa.array([not value for value in labels["Male"]]))
    renamed = labels.rename_columns([*COLUMNS[:-1], "Old"])
    widened = labels.set_column(2, "row_group", labels["row_group"].cast(pa.int64()))

    missing = labels.set_column(6, "Male", pa.array([None, *labels["Male"].to_pylist()[1:]]))
    tables = (labels, flipped, renamed, widened, missing)

    assert len({labels_digest(table) for table in tables}) == len(tables)


def test_the_pinned_source_is_the_revision_the_dataset_fetch_recorded() -> None:
    splits = [pinned.path.name.split("-")[0] for pinned in CELEBA.shards]
    assert (splits.count("train"), splits.count("valid"), splits.count("test")) == (19, 3, 3)
    assert all(
        "/resolve/2d738f56e0e7f925ea36ae7c808ea925264aacec/" in pinned.url
        for pinned in (*CELEBA.shards, CELEBA.license)
    )
    assert CELEBA.labels_rows == 202_599
    downloaded = [pinned for pinned in CELEBA.shards if pinned.path.name.startswith("test-00000")]
    assert downloaded[0].checksum.hexdigest.startswith("c22a47cd")
