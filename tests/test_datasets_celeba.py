"""CelebA as fetched: the label table, and each labelled image read from its shard's row group."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from celeba_files import Row, png, solid, write_celeba
from ryuk.datasets import DatasetError
from ryuk.datasets.celeba import CelebaImage, iter_images, labels_path, read_labels, shards_dir

SHARDS = {
    "valid-00000-of-00002.parquet": [Row(7, solid(10)), Row(7, solid(11)), Row(8, solid(12))],
    "valid-00001-of-00002.parquet": [Row(9, solid(20), {"Male": True})],
    "test-00000-of-00001.parquet": [Row(30, solid(30)), Row(31, solid(31))],
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_celeba(tmp_path, SHARDS)
    return tmp_path


def test_the_files_live_where_the_fetch_puts_them() -> None:
    assert labels_path(Path("d")) == Path("d/celeba/metadata/celeba_meta.parquet")
    assert shards_dir(Path("d")) == Path("d/celeba/parquet/img_align+identity+attr")


def test_labels_are_read_for_one_split_in_shard_order(root: Path) -> None:
    labels = read_labels(root, "valid", attributes=["Male"])

    assert labels.column_names == [
        "split",
        "shard",
        "row_group",
        "row_in_file",
        "path",
        "celeb_id",
        "Male",
    ]
    assert labels.column("celeb_id").to_pylist() == [7, 7, 8, 9]
    assert labels.column("Male").to_pylist() == [False, False, False, True]
    assert read_labels(root).num_rows == 6


def test_every_labelled_image_is_read_with_its_bytes(root: Path) -> None:
    labels = read_labels(root, "valid")

    images = list(iter_images(root, labels))

    assert [(image.row, image.celeb_id, image.path) for image in images] == [
        (0, 7, "valid-00000-of-00002-0.png"),
        (1, 7, "valid-00000-of-00002-1.png"),
        (2, 8, "valid-00000-of-00002-2.png"),
        (3, 9, "valid-00001-of-00002-0.png"),
    ]
    assert images[2].png == png(solid(12))
    assert int(images[3].decode()[0, 0, 0]) == 20


def test_a_subset_of_labels_reads_only_those_images_in_shard_order(root: Path) -> None:
    labels = read_labels(root, "valid")
    # Rows in reverse: the images still come in shard order, each naming its labels row.
    subset = labels.take(pa.array([3, 2, 0]))

    images = list(iter_images(root, subset))

    assert [(image.row, image.path) for image in images] == [
        (2, "valid-00000-of-00002-0.png"),
        (1, "valid-00000-of-00002-2.png"),
        (0, "valid-00001-of-00002-0.png"),
    ]


def test_an_image_decodes_to_bgr() -> None:
    image = CelebaImage(row=0, path="x.png", celeb_id=1, png=png(solid(5, 9, 4)))

    decoded = image.decode()

    assert decoded.shape == (9, 4, 3)
    assert decoded.dtype == np.uint8


def test_bytes_that_are_not_an_image_are_an_error() -> None:
    image = CelebaImage(row=0, path="x.png", celeb_id=1, png=b"not a png")

    with pytest.raises(DatasetError, match=r"x\.png"):
        image.decode()


def test_labels_that_do_not_match_the_shard_are_an_error(root: Path) -> None:
    labels = read_labels(root, "test")
    wrong = labels.set_column(4, "path", pc.utf8_upper(labels.column("path")))

    with pytest.raises(DatasetError, match=r"TEST-00000-OF-00001-0\.PNG"):
        list(iter_images(root, wrong))


def test_a_row_past_the_end_of_its_row_group_is_an_error(root: Path) -> None:
    labels = read_labels(root, "test")
    wrong = labels.set_column(3, "row_in_file", pa.array([0, 5], pa.int32()))

    with pytest.raises(DatasetError, match="row 5"):
        list(iter_images(root, wrong))


def test_a_missing_shard_is_reported(root: Path) -> None:
    (shards_dir(root) / "test-00000-of-00001.parquet").unlink()

    with pytest.raises(FileNotFoundError):
        list(iter_images(root, read_labels(root, "test")))


def test_a_label_table_without_the_location_columns_is_an_error(root: Path) -> None:
    pq.write_table(pa.table({"celeb_id": [1]}), labels_path(root))

    with pytest.raises(DatasetError, match="split"):
        read_labels(root)
