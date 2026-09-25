"""LFW from scikit-learn's figshare mirror: the funneled images and the three pairs lists.

The MD5s are the ones the dataset fetch recorded (#8), which match torchvision's published values
and the files' S3 ETags. The archive is kept after extraction, so a rerun checks it and skips.
"""

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ryuk.fetch import FetchError
from ryuk.fetch.pinned import Checksum, Fetched, PinnedFile, fetch_pinned

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LfwSource:
    archive: PinnedFile
    pairs: tuple[PinnedFile, ...]
    extracted: PurePosixPath
    """The folder the archive unpacks to, relative to the fetch root; the archive's top level."""
    images: int
    """How many JPEGs the extraction holds, so a folder damaged since is extracted again."""


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
    images=13_233,
)


def fetch_lfw(root: Path, source: LfwSource = LFW) -> list[Fetched]:
    fetched = [fetch_pinned(pinned, root) for pinned in source.pairs]
    archive = fetch_pinned(source.archive, root)
    fetched.append(archive)
    fetched.append(_extract(archive.path, source, root))
    return fetched


def _extract(archive: Path, source: LfwSource, root: Path) -> Fetched:
    """Unpack the verified archive, unless this archive's extraction is already in place.

    Extraction happens in a scratch folder renamed into place once complete. A stamp beside it,
    written last, names the archive's checksum, so a half-extracted folder is never taken for LFW.
    """
    target = root.joinpath(source.extracted)
    stamp = target.with_name(f".{target.name}.extracted-from")
    checksum = str(source.archive.checksum)
    if stamp.is_file() and stamp.read_text() == checksum and _images(target) == source.images:
        logger.info("%s already extracted", source.extracted)
        return Fetched(target, updated=False)

    logger.info("extracting %s", source.archive.path)
    stamp.unlink(missing_ok=True)
    scratch = target.with_name(f".{target.name}.partial")
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(scratch, filter="data")
        except (tarfile.TarError, OSError) as error:
            raise FetchError(f"extracting {source.archive.path} failed: {error}") from error
        unpacked = scratch / target.name
        if not unpacked.is_dir():
            raise FetchError(f"{source.archive.path} has no top-level {target.name} folder")
        if (found := _images(unpacked)) != source.images:
            raise FetchError(f"{source.archive.path} holds {found} images, not {source.images}")
        _replace_folder(unpacked, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    stamp.write_text(checksum)
    return Fetched(target, updated=True)


def _images(folder: Path) -> int:
    return sum(1 for _ in folder.rglob("*.jpg"))


def _replace_folder(new: Path, target: Path) -> None:
    stale = target.with_name(f".{target.name}.stale")
    shutil.rmtree(stale, ignore_errors=True)
    if target.exists():
        target.replace(stale)
    new.replace(target)
    shutil.rmtree(stale, ignore_errors=True)
