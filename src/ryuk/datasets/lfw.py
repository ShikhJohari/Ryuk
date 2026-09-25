"""LFW as fetched: one folder of JPEGs per identity, and the View 1 and View 2 pairs lists.

View 1 (`pairsDevTrain`, `pairsDevTest`) is for tuning the pipeline; View 2 (`pairs`) is the
ten-fold list the reported verification numbers come from (#9).
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, NamedTuple

from ryuk.datasets import DatasetError
from ryuk.fetch.lfw import LFW

type PairsName = Literal["pairsDevTrain", "pairsDevTest", "pairs"]
type View = Literal[1, 2]

PAIRS_FILES: Final[tuple[PairsName, ...]] = ("pairsDevTrain", "pairsDevTest", "pairs")
VIEWS: Final[Mapping[PairsName, View]] = MappingProxyType(
    {"pairsDevTrain": 1, "pairsDevTest": 1, "pairs": 2}
)


class LfwImage(NamedTuple):
    """One LFW image: an identity and the image's 1-based number, as the pairs lists name it."""

    identity: str
    number: int

    @property
    def path(self) -> PurePosixPath:
        """Where the image lives under the funneled folder."""
        return PurePosixPath(self.identity, f"{self.identity}_{self.number:04d}.jpg")


@dataclass(frozen=True, slots=True)
class Pair:
    first: LfwImage
    second: LfwImage
    fold: int
    """0-based, in file order. View 1 lists have a single fold."""

    @property
    def matched(self) -> bool:
        """Whether both images show the same identity."""
        return self.first.identity == self.second.identity


@dataclass(frozen=True, slots=True)
class PairsFile:
    """One pairs list: per fold, its matched pairs then as many mismatched ones."""

    name: PairsName
    folds: int
    pairs: tuple[Pair, ...]

    @property
    def view(self) -> View:
        return VIEWS[self.name]


def images_dir(root: Path) -> Path:
    """The funneled images' folder under the fetch root (`RYUK_DATA_DIR`)."""
    return root.joinpath(LFW.extracted)


def list_identities(root: Path) -> dict[str, list[LfwImage]]:
    """Every identity with its images, both in name and number order.

    Identities are the folders holding at least one image. The folder also holds the dataset's
    own `pairs_*.txt` files, which are skipped along with anything else that is not a folder.
    """
    identities: dict[str, list[LfwImage]] = {}
    for folder in sorted(images_dir(root).iterdir()):
        if not folder.is_dir():
            continue
        name = re.compile(rf"{re.escape(folder.name)}_(\d{{4}})\.jpg")
        images = []
        for path in folder.glob("*.jpg"):
            if (match := name.fullmatch(path.name)) is None:
                raise DatasetError(f"{folder.name}/{path.name} is not named for its identity")
            images.append(LfwImage(folder.name, int(match.group(1))))
        if images:
            identities[folder.name] = sorted(images)
    return identities


def read_pairs(root: Path, name: PairsName) -> PairsFile:
    """The pairs list `name` from the fetch root."""
    return parse_pairs(name, root.joinpath("lfw", f"{name}.txt").read_text())


def parse_pairs(name: PairsName, text: str) -> PairsFile:
    """A pairs list from its text.

    The first line is the pairs per class, preceded by the number of folds when there is more
    than one (`pairs.txt` starts `10\\t300`). Each fold then lists that many matched pairs,
    `name n1 n2`, followed by as many mismatched ones, `name1 n1 name2 n2`.
    """
    lines = text.rstrip("\n").split("\n")
    if not lines[0]:
        raise DatasetError(f"{name}: empty")
    folds, per_class = _header(name, lines[0])
    rows = lines[1:]
    expected = folds * 2 * per_class
    if len(rows) != expected:
        raise DatasetError(f"{name}: expected {expected} pairs, found {len(rows)}")

    pairs = []
    for index, row in enumerate(rows):
        fold, position = divmod(index, 2 * per_class)
        pairs.append(_pair(name, index + 2, row, fold, matched=position < per_class))
    return PairsFile(name, folds, tuple(pairs))


def _header(name: PairsName, line: str) -> tuple[int, int]:
    fields = line.split()
    if not 1 <= len(fields) <= 2 or not all(field.isdigit() for field in fields):
        raise DatasetError(f"{name}: header {line!r} is not '[folds] pairs-per-class'")
    counts = [int(field) for field in fields]
    folds, per_class = (1, counts[0]) if len(counts) == 1 else (counts[0], counts[1])
    if folds < 1 or per_class < 1:
        raise DatasetError(f"{name}: header {line!r} needs at least one fold and pair")
    return folds, per_class


def _pair(name: PairsName, line: int, row: str, fold: int, *, matched: bool) -> Pair:
    fields = row.split()
    if matched and len(fields) != 3:
        raise DatasetError(f"{name}: line {line}: expected a matched pair, got {row!r}")
    if not matched and len(fields) != 4:
        raise DatasetError(f"{name}: line {line}: expected a mismatched pair, got {row!r}")
    identities, numbers = (
        ([fields[0], fields[0]], fields[1:]) if matched else (fields[0::2], fields[1::2])
    )
    if not all(number.isdigit() and int(number) >= 1 for number in numbers):
        raise DatasetError(f"{name}: line {line}: image numbers must be positive, got {row!r}")
    first, second = (
        LfwImage(identity, int(number))
        for identity, number in zip(identities, numbers, strict=True)
    )
    pair = Pair(first, second, fold)
    if not matched and pair.matched:
        raise DatasetError(f"{name}: line {line}: a mismatched pair of one identity, {row!r}")
    return pair
