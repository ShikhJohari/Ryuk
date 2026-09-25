"""LFW as fetched: identities listed by folder, and the three pairs lists parsed and checked."""

from pathlib import Path

import pytest

from ryuk.datasets import DatasetError
from ryuk.datasets.lfw import (
    PAIRS_FILES,
    LfwImage,
    Pair,
    images_dir,
    list_identities,
    parse_pairs,
    read_pairs,
)


def lfw_root(tmp_path: Path, identities: dict[str, int]) -> Path:
    """A fetch root holding `lfw/lfw_funneled/` with `count` empty JPEGs per identity."""
    folder = tmp_path / "lfw" / "lfw_funneled"
    for identity, count in identities.items():
        (folder / identity).mkdir(parents=True)
        for number in range(1, count + 1):
            (folder / identity / f"{identity}_{number:04d}.jpg").write_bytes(b"")
    return tmp_path


def test_an_image_knows_its_path_under_the_funneled_folder() -> None:
    image = LfwImage("Aaron_Eckhart", 7)

    assert image.path.as_posix() == "Aaron_Eckhart/Aaron_Eckhart_0007.jpg"
    assert images_dir(Path("data/raw")) == Path("data/raw/lfw/lfw_funneled")


def test_identities_are_the_folders_and_their_images_in_number_order(tmp_path: Path) -> None:
    root = lfw_root(tmp_path, {"Zico": 1, "Abel_Pacheco": 3})
    # The fetched folder also holds the pairs files, which are not identities.
    (images_dir(root) / "pairs_01.txt").write_text("")
    # Out of order on disk, and a stray non-JPEG inside a folder.
    (images_dir(root) / "Abel_Pacheco" / "Abel_Pacheco_0010.jpg").write_bytes(b"")
    (images_dir(root) / "Abel_Pacheco" / "Thumbs.db").write_bytes(b"")

    identities = list_identities(root)

    assert list(identities) == ["Abel_Pacheco", "Zico"]
    assert [image.number for image in identities["Abel_Pacheco"]] == [1, 2, 3, 10]
    assert identities["Zico"] == [LfwImage("Zico", 1)]


def test_a_folder_without_images_is_not_an_identity(tmp_path: Path) -> None:
    root = lfw_root(tmp_path, {"Zico": 1})
    (images_dir(root) / "Empty_Person").mkdir()

    assert list(list_identities(root)) == ["Zico"]


def test_an_image_named_for_another_identity_is_an_error(tmp_path: Path) -> None:
    root = lfw_root(tmp_path, {"Zico": 1})
    (images_dir(root) / "Zico" / "Pele_0001.jpg").write_bytes(b"")

    with pytest.raises(DatasetError, match=r"Zico/Pele_0001\.jpg"):
        list_identities(root)


def test_a_missing_lfw_folder_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_identities(tmp_path)


VIEW_1 = """2
Aaron_Peirsol\t1\t2
Aaron_Peirsol\t3\t4
AJ_Cook\t1\tMarsha_Thomason\t1
Aaron_Peirsol\t2\tZico\t3
"""

VIEW_2 = """2\t1
Abel_Pacheco\t1\t4
Abel_Pacheco\t1\tZico\t2
Zico\t1\t2
Zico\t3\tAbel_Pacheco\t2
"""


def test_a_view_1_list_is_one_fold_of_matched_then_mismatched_pairs() -> None:
    pairs = parse_pairs("pairsDevTrain", VIEW_1)

    assert pairs.name == "pairsDevTrain"
    assert pairs.view == 1
    assert pairs.folds == 1
    assert pairs.pairs == (
        Pair(LfwImage("Aaron_Peirsol", 1), LfwImage("Aaron_Peirsol", 2), fold=0),
        Pair(LfwImage("Aaron_Peirsol", 3), LfwImage("Aaron_Peirsol", 4), fold=0),
        Pair(LfwImage("AJ_Cook", 1), LfwImage("Marsha_Thomason", 1), fold=0),
        Pair(LfwImage("Aaron_Peirsol", 2), LfwImage("Zico", 3), fold=0),
    )
    assert [pair.matched for pair in pairs.pairs] == [True, True, False, False]


def test_a_view_2_list_has_its_folds_in_file_order() -> None:
    pairs = parse_pairs("pairs", VIEW_2)

    assert pairs.view == 2
    assert pairs.folds == 2
    assert [(pair.fold, pair.matched) for pair in pairs.pairs] == [
        (0, True),
        (0, False),
        (1, True),
        (1, False),
    ]
    assert pairs.pairs[3] == Pair(LfwImage("Zico", 3), LfwImage("Abel_Pacheco", 2), fold=1)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", "empty"),
        ("two\n", "header"),
        ("1\t2\t3\n", "header"),
        ("0\n", "header"),
        ("1\nZico\t1\t2\n", "expected 2 pairs"),
        ("1\nZico\t1\t2\nZico\t1\tPele\t1\nZico\t2\t3\n", "expected 2 pairs"),
        ("1\nZico\t1\tPele\t1\nZico\t1\t2\n", "line 2: expected a matched pair"),
        ("1\nZico\t1\t2\nZico\t1\t2\n", "line 3: expected a mismatched pair"),
        ("1\nZico\t1\t2\nZico\t1\tZico\t2\n", "line 3: a mismatched pair of one identity"),
        ("1\nZico\tone\t2\nZico\t1\tPele\t1\n", "line 2: image numbers"),
        ("1\nZico\t0\t2\nZico\t1\tPele\t1\n", "line 2: image numbers"),
    ],
)
def test_a_malformed_pairs_list_is_an_error_naming_the_problem(text: str, problem: str) -> None:
    with pytest.raises(DatasetError, match=f"pairsDevTest: {problem}"):
        parse_pairs("pairsDevTest", text)


def test_trailing_blank_lines_are_ignored() -> None:
    pairs = parse_pairs("pairsDevTest", "1\nZico\t1\t2\nZico\t1\tPele\t1\n\n\n")

    assert len(pairs.pairs) == 2


def test_the_three_lists_are_read_from_the_fetch_root(tmp_path: Path) -> None:
    (tmp_path / "lfw").mkdir()
    for name in PAIRS_FILES:
        (tmp_path / "lfw" / f"{name}.txt").write_text(VIEW_2 if name == "pairs" else VIEW_1)

    lists = [read_pairs(tmp_path, name) for name in PAIRS_FILES]

    assert [(p.name, p.view, len(p.pairs)) for p in lists] == [
        ("pairsDevTrain", 1, 4),
        ("pairsDevTest", 1, 4),
        ("pairs", 2, 4),
    ]
