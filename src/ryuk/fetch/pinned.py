"""Pinned files: a source URL, a place under a root, and the size and checksum it must have.

A download streams into a `.part` file beside its target and is renamed into place only after
its size and checksum pass, so a corrupted or interrupted download never looks like a real file.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from ryuk.fetch.http import FetchError, open_url

__all__ = [
    "Algorithm",
    "Checksum",
    "ChecksumMismatchError",
    "FetchError",
    "Fetched",
    "PinnedFile",
    "fetch_pinned",
    "file_checksum",
    "is_verified",
    "part_path",
    "verify",
]

logger = logging.getLogger(__name__)

Algorithm = Literal["md5", "sha256"]
_CHUNK_BYTES = 1 << 20


class ChecksumMismatchError(FetchError):
    """A file's bytes are not the pinned bytes."""


@dataclass(frozen=True)
class Checksum:
    # LFW's mirrors only publish MD5; it checks integrity here, not authenticity.
    algorithm: Algorithm
    hexdigest: str

    def __str__(self) -> str:
        return f"{self.algorithm} {self.hexdigest}"


@dataclass(frozen=True)
class PinnedFile:
    url: str
    path: PurePosixPath
    """Where the file lives, relative to the root it is fetched into."""
    size: int
    checksum: Checksum


@dataclass(frozen=True)
class Fetched:
    path: Path
    updated: bool
    """False when a verified copy was already in place and nothing was written."""


def fetch_pinned(pinned: PinnedFile, root: Path) -> Fetched:
    """Make `root / pinned.path` hold the pinned bytes, downloading only if it does not yet."""
    target = root.joinpath(pinned.path)
    if is_verified(target, pinned):
        logger.info("%s already verified", pinned.path)
        return Fetched(target, updated=False)

    logger.info("downloading %s (%.1f MB)", pinned.path, pinned.size / 1e6)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(target)
    try:
        with open_url(pinned.url) as response, part.open("wb") as out:
            while chunk := response.read(_CHUNK_BYTES):
                out.write(chunk)
        verify(part, pinned)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    logger.info("%s verified (%s)", pinned.path, pinned.checksum)
    return Fetched(target, updated=True)


def file_checksum(path: Path, algorithm: Algorithm) -> str:
    hasher = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def part_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.part")


def is_verified(path: Path, pinned: PinnedFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == pinned.size
        and file_checksum(path, pinned.checksum.algorithm) == pinned.checksum.hexdigest
    )


def verify(path: Path, pinned: PinnedFile) -> None:
    """Raise `ChecksumMismatchError` unless `path` holds the pinned bytes."""
    size = path.stat().st_size
    if size != pinned.size:
        raise ChecksumMismatchError(
            f"{pinned.path}: downloaded {size} bytes from {pinned.url}, pinned at {pinned.size}"
        )
    actual = file_checksum(path, pinned.checksum.algorithm)
    if actual != pinned.checksum.hexdigest:
        raise ChecksumMismatchError(
            f"{pinned.path}: {pinned.checksum.algorithm} of the download from {pinned.url} "
            f"is {actual}, pinned at {pinned.checksum.hexdigest}"
        )
