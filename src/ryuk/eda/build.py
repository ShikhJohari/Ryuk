"""Building the dataset summary from the fetched data: scan with YuNet, measure the minimum usable
face size, then aggregate under it (#17, #25).

The minimum comes first because every usable-face count depends on it: it is measured over the
centre-most detection of every detected CelebA image in both draws, then applied to LFW and
CelebA alike through `benchmark_face`.
"""

import datetime
import logging
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from ryuk.datasets import DatasetError
from ryuk.datasets.celeba import CelebaImage, Split, iter_images, read_labels
from ryuk.datasets.lfw import PAIRS_FILES, LfwImage, PairsFile, images_dir, list_identities
from ryuk.datasets.lfw import read_pairs as read_lfw_pairs
from ryuk.detector import NMS_THRESHOLD, SCORE_THRESHOLD, Image
from ryuk.eda.aggregate import (
    LabelledImage,
    celeba_draw,
    detection_stats,
    images_per_identity,
    min_usable_face_size,
    pair_list,
)
from ryuk.eda.scan import ImageScan, Scanner
from ryuk.eda.summary import (
    ATTRIBUTES,
    SCHEMA_VERSION,
    Attribute,
    Draw,
    EdaSummary,
    Lfw,
    Provenance,
    Rules,
)
from ryuk.fetch.celeba import CELEBA, REVISION
from ryuk.fetch.lfw import LFW
from ryuk.weights import YUNET

logger = logging.getLogger(__name__)

RULES: Final = Rules(min_gallery_images=20, majority_agreement=0.8)
"""#9's gallery eligibility and #10's majority label."""
DRAWS: Final[tuple[tuple[Draw, Split], ...]] = (("validation", "valid"), ("test", "test"))


class ProvenanceError(Exception):
    """The git state the summary records could not be read."""


@dataclass(frozen=True)
class LfwScan:
    identities: Mapping[str, Sequence[LfwImage]]
    scans: Mapping[LfwImage, ImageScan]
    pairs: Sequence[PairsFile]


@dataclass(frozen=True)
class DrawImage:
    identity: int
    labels: Mapping[Attribute, bool]
    scan: ImageScan


@dataclass(frozen=True)
class DrawScan:
    draw: Draw
    images: Sequence[DrawImage]


def build_summary(data_dir: Path, weights_dir: Path, *, workers: int, repo: Path) -> EdaSummary:
    """Scan LFW and both CelebA draws and summarise them. `repo` is the git checkout whose
    commit the summary records."""
    scanner = Scanner(YUNET.path(weights_dir), workers)
    lfw = scan_lfw(data_dir, scanner)
    draws = [scan_draw(data_dir, draw, split, scanner) for draw, split in DRAWS]
    return summarise(lfw, draws, provenance(repo, datetime.date.today()))


def scan_lfw(root: Path, scanner: Scanner) -> LfwScan:
    """Every LFW image scanned, and the three pairs lists."""
    identities = list_identities(root)
    folder = images_dir(root)
    images = [image for images in identities.values() for image in images]
    scans = scanner.scan(
        images, lambda image: _read_jpeg(folder / image.path), total=len(images), name="LFW"
    )
    pairs = [read_lfw_pairs(root, name) for name in PAIRS_FILES]
    return LfwScan(identities, dict(zip(images, scans, strict=True)), pairs)


def scan_draw(root: Path, draw: Draw, split: Split, scanner: Scanner) -> DrawScan:
    """Every image of one CelebA split scanned, with its identity and the EDA's attributes."""
    labels = read_labels(root, split, attributes=ATTRIBUTES)
    rows: list[int] = []

    def images() -> Iterator[CelebaImage]:
        for image in iter_images(root, labels):
            rows.append(image.row)
            yield image

    scans = scanner.scan(
        images(), CelebaImage.decode, total=labels.num_rows, name=f"CelebA {split}"
    )
    identities: list[int] = labels.column("celeb_id").to_numpy().tolist()
    flags: dict[Attribute, list[bool]] = {
        attribute: labels.column(attribute).to_numpy().tolist() for attribute in ATTRIBUTES
    }
    return DrawScan(
        draw,
        [
            DrawImage(identities[row], {a: flags[a][row] for a in ATTRIBUTES}, scan)
            for row, scan in zip(rows, scans, strict=True)
        ],
    )


def summarise(
    lfw: LfwScan, draws: Sequence[DrawScan], provenance: Provenance, rules: Rules = RULES
) -> EdaSummary:
    """The summary of scanned data. `draws` are the validation draw, then the test draw."""
    minimum = min_usable_face_size(
        [
            face.box.short_side
            for draw in draws
            for image in draw.images
            if (face := image.scan.centre_most) is not None
        ]
    )
    size = minimum.value
    logger.info(
        "minimum usable face size %d px keeps %.2f%% of %d CelebA detections",
        size,
        100 * minimum.kept,
        minimum.detections,
    )
    return EdaSummary(
        schema_version=SCHEMA_VERSION,
        provenance=provenance,
        rules=rules,
        min_usable_face_size=minimum,
        lfw=_lfw(lfw, size),
        celeba=[
            celeba_draw(
                draw.draw,
                [
                    LabelledImage(
                        identity=image.identity,
                        labels=image.labels,
                        detected=bool(image.scan.detections),
                        usable=image.scan.usable(size),
                    )
                    for image in draw.images
                ],
                detection_stats([image.scan for image in draw.images], min_face_size=size),
                rules,
            )
            for draw in draws
        ],
    )


def _lfw(lfw: LfwScan, min_face_size: int) -> Lfw:
    for pairs in lfw.pairs:
        for pair in pairs.pairs:
            for image in (pair.first, pair.second):
                if image not in lfw.scans:
                    raise DatasetError(f"{pairs.name} names {image.path}, which is not in LFW")
    excluded = {image for image, scan in lfw.scans.items() if not scan.usable(min_face_size)}
    return Lfw(
        images_per_identity=images_per_identity(len(images) for images in lfw.identities.values()),
        pairs=[pair_list(pairs, excluded) for pairs in lfw.pairs],
        detection=detection_stats(list(lfw.scans.values()), min_face_size=min_face_size),
    )


def provenance(repo: Path, generated: datetime.date) -> Provenance:
    """Where the summary came from: the code's commit and the pinned data and detector."""
    return Provenance(
        generated=generated,
        git_commit=_git(repo, "rev-parse", "HEAD").strip(),
        # Untracked files don't count: the summary's own output folder is one until committed,
        # and code nothing tracked imports can't change the numbers.
        git_dirty=bool(_git(repo, "status", "--porcelain", "--untracked-files=no").strip()),
        celeba_revision=REVISION,
        celeba_labels_digest=CELEBA.labels_digest,
        lfw_archive=str(LFW.archive.checksum),
        yunet_sha256=YUNET.file.checksum.hexdigest,
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )


def _git(repo: Path, *args: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise ProvenanceError("git is not on PATH; the summary records the commit it came from")
    # Fixed arguments to the git found on PATH, no shell.
    result = subprocess.run(  # noqa: S603
        [git, "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ProvenanceError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
    return result.stdout


def _read_jpeg(path: Path) -> Image:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"{path} does not decode as an image")
    return np.asarray(image, dtype=np.uint8)
