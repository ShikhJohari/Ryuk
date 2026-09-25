"""LFW from scikit-learn's figshare mirror: the funneled images and the three pairs lists.

The MD5s are the ones the dataset fetch recorded (#8), which match torchvision's published values
and the files' S3 ETags. The archive is kept after extraction, so a rerun checks it and skips.
"""

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ryuk.fetch.pinned import Checksum, Fetched, FetchError, PinnedFile, fetch_pinned

logger = logging.getLogger(__name__)

_STAMP = ".extracted-from"


@dataclass(frozen=True)
class LfwSource:
    archive: PinnedFile
    pairs: tuple[PinnedFile, ...]
    extracted: PurePosixPath
    """The folder the archive unpacks to, relative to the fetch root; the archive's top level."""


def _figshare(file_id: int, path: str, size: int, md5: str) -> PinnedFile:
    return PinnedFile(
        url=f"https://ndownloader.figshare.com/files/{file_id}",
        path=PurePosixPath("lfw", path),
        size=size,
        checksum=Checksum("md5", md5),
    )


LFW = LfwSource(
    archive=_figshare(
        5976015, "archives/lfw-funneled.tgz", 243_346_528, "1b42dfed7d15c9b2dd63d5e5840c86ad"
    ),
    pairs=(
        _figshare(5976006, "pairs.txt", 155_335, "9f1ba174e4e1c508ff7cdf10ac338a7d"),
        _figshare(5976012, "pairsDevTrain.txt", 56_579, "4f27cbf15b2da4a85c1907eb4181ad21"),
        _figshare(5976009, "pairsDevTest.txt", 26_002, "5132f7440eb68cf58910c8a45a2ac10b"),
    ),
    extracted=PurePosixPath("lfw/lfw_funneled"),
)


def fetch_lfw(root: Path, source: LfwSource = LFW) -> list[Fetched]:
    fetched = [fetch_pinned(pinned, root) for pinned in source.pairs]
    archive = fetch_pinned(source.archive, root)
    fetched.append(archive)
    fetched.append(_extract(archive.path, source, root))
    return fetched


def _extract(archive: Path, source: LfwSource, root: Path) -> Fetched:
    """Unpack the verified archive, unless this archive's extraction is already in place.

    Extraction happens in a scratch folder that is renamed into place once complete, with a
    stamp naming the archive's checksum, so a half-extracted folder is never mistaken for LFW.
    """
    target = root.joinpath(source.extracted)
    stamp = str(source.archive.checksum)
    if (target / _STAMP).is_file() and (target / _STAMP).read_text() == stamp:
        logger.info("%s already extracted", source.extracted)
        return Fetched(target, updated=False)

    logger.info("extracting %s", source.archive.path)
    scratch = target.with_name(f".{target.name}.partial")
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(scratch, filter="data")
        except tarfile.TarError as error:
            raise FetchError(f"{source.archive.path}: {error}") from error
        unpacked = scratch / target.name
        if not unpacked.is_dir():
            raise FetchError(f"{source.archive.path} has no top-level {target.name} folder")
        (unpacked / _STAMP).write_text(stamp)
        _replace_folder(unpacked, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return Fetched(target, updated=True)


def _replace_folder(new: Path, target: Path) -> None:
    stale = target.with_name(f".{target.name}.stale")
    shutil.rmtree(stale, ignore_errors=True)
    if target.exists():
        target.replace(stale)
    new.replace(target)
    shutil.rmtree(stale, ignore_errors=True)
