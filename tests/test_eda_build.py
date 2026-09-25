"""`ryuk eda` end to end on a tiny fetch root, with the real YuNet and composed faces."""

import datetime
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from celeba_files import Row, write_celeba
from ryuk.datasets import DatasetError
from ryuk.datasets.lfw import LfwImage, Pair, PairsFile, images_dir
from ryuk.detector import NMS_THRESHOLD, SCORE_THRESHOLD, Box, Detection, Detector, Image, Landmarks
from ryuk.eda.build import (
    RULES,
    DrawImage,
    DrawScan,
    LfwScan,
    ProvenanceError,
    build_summary,
    provenance,
    scan_draw,
    scan_lfw,
    summarise,
)
from ryuk.eda.files import read_summary, write_summary
from ryuk.eda.scan import ImageScan, Scanner
from ryuk.eda.summary import ATTRIBUTES, Rules
from ryuk.fetch.celeba import CELEBA, REVISION
from ryuk.fetch.lfw import LFW
from ryuk.weights import YUNET

FIXTURES = Path(__file__).parent / "fixtures"
ASTRONAUT = cv2.imread(str(FIXTURES / "astronaut.jpg"), cv2.IMREAD_COLOR)
assert ASTRONAUT is not None
HEAD = ASTRONAUT[0:260, 120:340]
GREY = 127


def centred_head(height: int, width: int, head_width: int | None) -> Image:
    """A grey image with the astronaut's head `head_width` px wide at its centre, or none.

    In a 178 x 218 or 250 x 250 image YuNet boxes a head 170, 150, 130, 110 or 84 px wide at
    about 69, 62, 54, 46 and 34 px on the short side.
    """
    canvas = np.full((height, width, 3), GREY, dtype=np.uint8)
    if head_width is not None:
        head_height = round(HEAD.shape[0] * head_width / HEAD.shape[1])
        head = cv2.resize(HEAD, (head_width, head_height), interpolation=cv2.INTER_AREA)
        y, x = (height - head_height) // 2, (width - head_width) // 2
        canvas[y : y + head_height, x : x + head_width] = head
    return canvas


def celeba(head_width: int | None) -> Image:
    return centred_head(218, 178, head_width)


LFW_IMAGES = {
    LfwImage("Ann_Face", 1): 150,
    LfwImage("Ann_Face", 2): 130,
    LfwImage("Bob_Blank", 1): None,
    LfwImage("Cy_Small", 1): 84,  # found, but under the minimum measured on CelebA
}
PAIRS = {
    "pairsDevTrain": "1\nAnn_Face\t1\t2\nAnn_Face\t1\tBob_Blank\t1\n",
    "pairsDevTest": "1\nAnn_Face\t2\t1\nAnn_Face\t2\tCy_Small\t1\n",
    "pairs": (
        "2\t1\nAnn_Face\t1\t2\nAnn_Face\t2\tCy_Small\t1\n"
        "Ann_Face\t2\t1\nCy_Small\t1\tBob_Blank\t1\n"
    ),
}
CELEBA_SHARDS = {
    "valid-00000-of-00001.parquet": [
        Row(1, celeba(170), {"Male": True, "Eyeglasses": True}),
        Row(1, celeba(150), {"Male": True}),
        Row(1, celeba(130), {"Male": True}),
        Row(2, celeba(110), {"Young": True}),
        Row(2, celeba(None), {"Young": True, "Blurry": True}),
    ],
    "test-00000-of-00001.parquet": [
        Row(3, celeba(150)),
        Row(3, celeba(130), {"Wearing_Hat": True}),
        Row(4, celeba(None)),
    ],
}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    for image, head_width in LFW_IMAGES.items():
        path = images_dir(root) / image.path
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), centred_head(250, 250, head_width))
    for name, text in PAIRS.items():
        (root / "lfw" / f"{name}.txt").write_text(text)
    write_celeba(root, CELEBA_SHARDS)
    return root


def expected_minimum() -> int:
    """With fewer than 100 detections keeping 99% means keeping all, so the minimum is the
    smallest CelebA face rounded down to 10 px, found here by detecting directly."""
    detector = Detector(FIXTURES / YUNET.file.path.name)
    sides = [
        face.box.short_side
        for rows in CELEBA_SHARDS.values()
        for row in rows
        for face in detector.detect(row.image)
    ]
    return math.floor(min(sides) / 10) * 10


def test_the_summary_is_built_from_the_fetch_root(data_dir: Path) -> None:
    summary = build_summary(data_dir, FIXTURES, workers=2, repo=Path(__file__).parent)

    # Six CelebA faces, the smallest the 110 px head boxed at about 46 px.
    minimum = summary.min_usable_face_size
    assert minimum.value == expected_minimum() == 40
    assert (minimum.detections, minimum.kept, minimum.kept_at_next_step) == (6, 1.0, 0.833333)
    assert summary.rules == RULES
    assert summary.provenance.generated == datetime.date.today()

    lfw = summary.lfw
    assert (lfw.images_per_identity.identities, lfw.images_per_identity.images) == (3, 4)
    detection = lfw.detection
    assert (detection.images, detection.detected, detection.usable) == (4, 3, 2)
    # Bob has no face and Cy's is too small: every pair with either loses an image.
    assert [(p.name, p.view, p.folds, p.pairs_with_excluded_image) for p in lfw.pairs] == [
        ("pairsDevTrain", 1, 1, 1),
        ("pairsDevTest", 1, 1, 1),
        ("pairs", 2, 2, 2),
    ]

    validation, test = summary.celeba
    assert (validation.draw, validation.split, test.draw, test.split) == (
        "validation",
        "valid",
        "test",
        "test",
    )
    assert (validation.detection.images, validation.detection.usable) == (5, 4)
    assert (test.detection.images, test.detection.usable) == (3, 2)
    # Nobody has 20 images, the gallery minimum.
    assert (validation.gallery_candidates, validation.eligible_identities) == (0, 0)
    male = {g.value: g for g in validation.groups if g.attribute == "Male"}
    assert (male[True].identities, male[True].images, male[True].usable) == (1, 3, 3)
    assert (male[False].identities, male[False].images, male[False].usable) == (1, 2, 1)
    blurry = {g.value: g for g in validation.groups if g.attribute == "Blurry"}
    assert (blurry[True].images, blurry[True].detected) == (1, 0)


def test_the_summary_survives_a_round_trip_through_its_file(data_dir: Path, tmp_path: Path) -> None:
    summary = build_summary(data_dir, FIXTURES, workers=1, repo=Path(__file__).parent)

    written = write_summary(summary, tmp_path / "eda")

    assert [path.name for path in written] == ["summary.json", "summary.schema.json"]
    assert read_summary(tmp_path / "eda") == summary
    assert not list((tmp_path / "eda").glob("*.part"))


def test_rules_can_be_changed(data_dir: Path) -> None:
    scanner = Scanner(FIXTURES / YUNET.file.path.name, 1)
    lfw = scan_lfw(data_dir, scanner)
    draws = [scan_draw(data_dir, "validation", scanner)]
    rules = Rules(min_gallery_images=2, majority_agreement=0.8)

    summary = summarise(lfw, draws, provenance(Path(__file__).parent, datetime.date.today()), rules)

    [validation] = summary.celeba
    # Identity 1 has three usable images; identity 2 two images, one usable.
    assert (validation.gallery_candidates, validation.eligible_identities) == (2, 1)


def test_a_pair_naming_an_unknown_image_is_an_error() -> None:
    point = (50.0, 50.0)
    face = ImageScan((100, 100), (Detection(Box(25, 25, 50, 50), Landmarks(*[point] * 5), 0.95),))
    known, stranger = LfwImage("Ann_Face", 1), LfwImage("Nobody", 1)
    lfw = LfwScan(
        identities={"Ann_Face": [known]},
        scans={known: face},
        pairs=[PairsFile("pairs", 1, (Pair(known, stranger, 0),))],
    )
    draws = [DrawScan("validation", [DrawImage(1, dict.fromkeys(ATTRIBUTES, False), face)])]

    with pytest.raises(DatasetError, match=r"pairs names Nobody/Nobody_0001\.jpg"):
        summarise(lfw, draws, provenance(Path(__file__).parent, datetime.date.today()))


def test_a_damaged_pairs_list_fails_before_the_scan(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    shutil.copytree(data_dir / "lfw", root / "lfw")
    (root / "lfw" / "pairsDevTest.txt").write_text("1\nAnn_Face\t2\t9\nAnn_Face\t2\tCy_Small\t1\n")

    def scan(*_args: object, **_kwargs: object) -> list[ImageScan]:
        raise AssertionError("LFW was scanned before its pairs lists were checked")

    scanner = Scanner(FIXTURES / YUNET.file.path.name, 1)
    monkeypatch.setattr(Scanner, "scan", scan)

    # Ann_Face has images 1 and 2 only.
    with pytest.raises(DatasetError, match=r"pairsDevTest names Ann_Face/Ann_Face_0009\.jpg"):
        scan_lfw(root, scanner)


def test_a_malformed_pairs_list_fails_before_the_scan(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    shutil.copytree(data_dir / "lfw", root / "lfw")
    (root / "lfw" / "pairs.txt").write_text("2\t1\nAnn_Face\t1\t2\n")

    def scan(*_args: object, **_kwargs: object) -> list[ImageScan]:
        raise AssertionError("LFW was scanned before its pairs lists were read")

    scanner = Scanner(FIXTURES / YUNET.file.path.name, 1)
    monkeypatch.setattr(Scanner, "scan", scan)

    with pytest.raises(DatasetError, match="pairs: expected 4 pairs, found 1"):
        scan_lfw(root, scanner)


def test_an_unreadable_lfw_image_is_an_error(tmp_path: Path) -> None:
    folder = images_dir(tmp_path) / "Zico"
    folder.mkdir(parents=True)
    (folder / "Zico_0001.jpg").write_bytes(b"not a jpeg")
    (images_dir(tmp_path) / "Ann").mkdir()
    cv2.imwrite(str(images_dir(tmp_path) / "Ann" / "Ann_0001.jpg"), celeba(None))
    for name in PAIRS:
        (tmp_path / "lfw" / f"{name}.txt").write_text("1\nAnn\t1\t1\nAnn\t1\tZico\t1\n")

    with pytest.raises(DatasetError, match=r"Zico_0001\.jpg"):
        scan_lfw(tmp_path, Scanner(FIXTURES / YUNET.file.path.name, 1))


GIT = shutil.which("git") or "git"


def git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    return subprocess.run(  # noqa: S603
        [GIT, "-C", str(repo), *identity, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_provenance_records_the_commit_and_the_pins(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "file.txt").write_text("one")
    git(tmp_path, "add", "file.txt")
    git(tmp_path, "commit", "-q", "-m", "one")
    today = datetime.date(2026, 9, 25)

    (tmp_path / "eda").mkdir()
    (tmp_path / "eda" / "summary.json").write_text("{}")  # untracked output: still clean
    clean = provenance(tmp_path, today)
    (tmp_path / "file.txt").write_text("two")
    dirty = provenance(tmp_path, today)

    assert clean.git_commit == git(tmp_path, "rev-parse", "HEAD").strip()
    assert (clean.git_dirty, dirty.git_dirty) == (False, True)
    assert clean.generated == today
    assert clean.celeba_revision == REVISION
    assert clean.celeba_labels_digest == CELEBA.labels_digest
    assert clean.lfw_archive == f"md5 {LFW.archive.checksum.hexdigest}"
    assert clean.yunet_sha256 == YUNET.file.checksum.hexdigest
    assert (clean.score_threshold, clean.nms_threshold) == (SCORE_THRESHOLD, NMS_THRESHOLD)


def test_provenance_needs_a_git_checkout(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="rev-parse"):
        provenance(tmp_path, datetime.date(2026, 9, 25))


def test_provenance_needs_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(ProvenanceError, match="not on PATH"):
        provenance(tmp_path, datetime.date(2026, 9, 25))
