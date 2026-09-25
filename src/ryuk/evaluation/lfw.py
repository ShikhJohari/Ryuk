"""LFW's pairs files and where each pair's images live.

View 1 is `pairsDevTrain.txt` (1,100 matched and 1,100 mismatched pairs) and `pairsDevTest.txt`
(500 and 500); it is for choosing pipeline options. View 2 is `pairs.txt`: 10 folds of 300
matched then 300 mismatched pairs, in file order; it is run once per model for the reported
number (#9).
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

IMAGES = PurePosixPath("lfw/lfw_funneled")
"""Where the funneled images are, relative to the data directory."""
VIEW_2 = PurePosixPath("lfw/pairs.txt")
VIEW_1_TRAIN = PurePosixPath("lfw/pairsDevTrain.txt")
VIEW_1_TEST = PurePosixPath("lfw/pairsDevTest.txt")


@dataclass(frozen=True, slots=True)
class Pair:
    """Two LFW images, as paths relative to the funneled image folder, and their fold."""

    first: PurePosixPath
    second: PurePosixPath
    same: bool
    fold: int


class PairsFormatError(ValueError):
    """A pairs file does not follow LFW's layout."""


def image_path(name: str, number: int) -> PurePosixPath:
    return PurePosixPath(name, f"{name}_{number:04d}.jpg")


def read_pairs(path: Path) -> list[Pair]:
    """Every pair in an LFW pairs file, in file order.

    The header is either `folds<TAB>n` (View 2) or `n` (View 1, one fold). Each fold is `n`
    matched lines, `name i j`, then `n` mismatched lines, `name1 i name2 j`.
    """
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise PairsFormatError(f"{path.name} is empty")
    header, rows = lines[0], lines[1:]
    try:
        numbers = [int(value) for value in header]
    except ValueError as error:
        raise PairsFormatError(f"{path.name}: bad header {header!r}") from error
    match numbers:
        case [folds, per_side]:
            pass
        case [per_side]:
            folds = 1
        case _:
            raise PairsFormatError(f"{path.name}: bad header {header!r}")
    if len(rows) != folds * 2 * per_side:
        raise PairsFormatError(
            f"{path.name}: header {header!r} promises {folds * 2 * per_side} pairs, "
            f"the file has {len(rows)}"
        )

    pairs: list[Pair] = []
    for index, row in enumerate(rows):
        fold, position = divmod(index, 2 * per_side)
        same = position < per_side
        pairs.append(_pair(row, same, fold, path.name, index + 2))
    return pairs


def _pair(row: list[str], same: bool, fold: int, file: str, line: int) -> Pair:
    try:
        if same and len(row) == 3:
            name, first, second = row
            return Pair(image_path(name, int(first)), image_path(name, int(second)), True, fold)
        if not same and len(row) == 4:
            name1, first, name2, second = row
            return Pair(image_path(name1, int(first)), image_path(name2, int(second)), False, fold)
    except ValueError:
        pass
    kind = "matched" if same else "mismatched"
    raise PairsFormatError(f"{file} line {line}: expected a {kind} pair, got {row!r}")
