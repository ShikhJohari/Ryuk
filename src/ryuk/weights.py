"""The pretrained weights Ryuk runs, each pinned to a source, a path and a sha256 (#7).

They live under the weights directory (`RYUK_WEIGHTS_DIR`), never in `~/.insightface` or the
torch cache, so the files the service verifies at startup are the files that were fetched.
"""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ryuk.fetch.pinned import (
    Checksum,
    Fetched,
    FetchError,
    PinnedFile,
    fetch_pinned,
    is_verified,
    part_path,
    verify,
)

logger = logging.getLogger(__name__)

_OPENCV_ZOO = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "47534e27c9851bb1128ccc0102f1145e27f23f98/models"
)


@dataclass(frozen=True)
class ExtractedFile:
    """A file taken out of a pinned zip archive, which is discarded once it is extracted."""

    archive: PinnedFile
    member: str
    path: PurePosixPath
    size: int
    checksum: Checksum

    @property
    def pinned(self) -> PinnedFile:
        return PinnedFile(
            url=f"{self.archive.url}#{self.member}",
            path=self.path,
            size=self.size,
            checksum=self.checksum,
        )


@dataclass(frozen=True)
class Weights:
    name: str
    file: PinnedFile | ExtractedFile

    def path(self, weights_dir: Path) -> Path:
        return weights_dir.joinpath(self.file.path)


YUNET = Weights(
    name="yunet",
    file=PinnedFile(
        url=f"{_OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2026may.onnx",
        path=PurePosixPath("face_detection_yunet_2026may.onnx"),
        size=229_738,
        checksum=Checksum(
            "sha256", "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
        ),
    ),
)
SFACE = Weights(
    name="sface",
    file=PinnedFile(
        url=f"{_OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        path=PurePosixPath("face_recognition_sface_2021dec.onnx"),
        size=38_696_353,
        checksum=Checksum(
            "sha256", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
        ),
    ),
)
ARCFACE = Weights(
    name="arcface",
    file=ExtractedFile(
        archive=PinnedFile(
            url="https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
            path=PurePosixPath("buffalo_l.zip"),
            size=288_621_354,
            checksum=Checksum(
                "sha256", "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
            ),
        ),
        member="w600k_r50.onnx",
        path=PurePosixPath("w600k_r50.onnx"),
        size=174_383_860,
        checksum=Checksum(
            "sha256", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
        ),
    ),
)
FACENET = Weights(
    name="facenet",
    file=PinnedFile(
        url="https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt",
        path=PurePosixPath("20180402-114759-vggface2.pt"),
        size=111_898_327,
        checksum=Checksum(
            "sha256", "281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b"
        ),
    ),
)
WEIGHTS = (YUNET, SFACE, ARCFACE, FACENET)


def fetch_weights(weights_dir: Path, weights: tuple[Weights, ...] = WEIGHTS) -> list[Fetched]:
    return [
        fetch_pinned(entry.file, weights_dir)
        if isinstance(entry.file, PinnedFile)
        else _fetch_extracted(entry.file, weights_dir)
        for entry in weights
    ]


def _fetch_extracted(extracted: ExtractedFile, weights_dir: Path) -> Fetched:
    pinned = extracted.pinned
    target = weights_dir.joinpath(pinned.path)
    if is_verified(target, pinned):
        logger.info("%s already verified", pinned.path)
        return Fetched(target, updated=False)

    archive = fetch_pinned(extracted.archive, weights_dir).path
    part = part_path(target)
    try:
        with zipfile.ZipFile(archive) as zipped:
            try:
                member = zipped.open(extracted.member)
            except KeyError as error:
                raise FetchError(f"{extracted.archive.path} has no {extracted.member}") from error
            with member, part.open("wb") as out:
                while chunk := member.read(1 << 20):
                    out.write(chunk)
        verify(part, pinned)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        archive.unlink(missing_ok=True)
    part.replace(target)
    logger.info("%s verified (%s)", pinned.path, pinned.checksum)
    return Fetched(target, updated=True)
