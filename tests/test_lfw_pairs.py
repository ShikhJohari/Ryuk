from pathlib import Path, PurePosixPath

import pytest

from ryuk.evaluation.lfw import Pair, PairsFormatError, image_path, read_pairs


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pairs.txt"
    path.write_text(text)
    return path


def test_an_image_path_zero_pads_the_number() -> None:
    assert image_path("Aaron_Peirsol", 4) == PurePosixPath("Aaron_Peirsol/Aaron_Peirsol_0004.jpg")


def test_view_2_has_folds_of_matched_then_mismatched_pairs(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "2\t1\nAnn\t1\t2\nAnn\t1\tBob\t3\nCy\t2\t10\nCy\t1\tDee\t1\n",
    )

    assert read_pairs(path) == [
        Pair(image_path("Ann", 1), image_path("Ann", 2), same=True, fold=0),
        Pair(image_path("Ann", 1), image_path("Bob", 3), same=False, fold=0),
        Pair(image_path("Cy", 2), image_path("Cy", 10), same=True, fold=1),
        Pair(image_path("Cy", 1), image_path("Dee", 1), same=False, fold=1),
    ]


def test_view_1_is_one_fold(tmp_path: Path) -> None:
    path = write(tmp_path, "2\nAnn\t1\t2\nBob\t1\t3\nAnn\t1\tBob\t3\nCy\t1\tDee\t1\n")

    pairs = read_pairs(path)

    assert [pair.same for pair in pairs] == [True, True, False, False]
    assert {pair.fold for pair in pairs} == {0}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("ten\n", "bad header"),
        ("1\t1\t1\nAnn\t1\t2\nAnn\t1\tBob\t3\n", "bad header"),
        ("1\t2\nAnn\t1\t2\nAnn\t1\tBob\t3\n", "promises 4 pairs"),
        ("1\nAnn\t1\tBob\t2\nAnn\t1\tBob\t3\n", "line 2: expected a matched pair"),
        ("1\nAnn\t1\t2\nAnn\t1\t3\n", "line 3: expected a mismatched pair"),
        ("1\nAnn\tone\t2\nAnn\t1\tBob\t3\n", "line 2"),
    ],
)
def test_a_malformed_file_is_refused(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(PairsFormatError, match=message):
        read_pairs(write(tmp_path, text))
