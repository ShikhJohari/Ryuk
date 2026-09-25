"""The parallel YuNet scan: every image in order, all its detections, and failures surfaced."""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from ryuk.detector import Box, Detection, Image, Landmarks
from ryuk.eda import scan as scan_module
from ryuk.eda.scan import ImageScan, Scanner, default_workers

FIXTURES = Path(__file__).parent / "fixtures"
YUNET = FIXTURES / "face_detection_yunet_2026may.onnx"


def _astronaut(width: int) -> Image:
    image = cv2.imread(str(FIXTURES / "astronaut.jpg"), cv2.IMREAD_COLOR)
    assert image is not None
    return np.asarray(cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA), np.uint8)


FACE = _astronaut(256)
BLANK = np.full((120, 160, 3), 127, dtype=np.uint8)


def _load(item: str) -> Image:
    return FACE if item == "face" else BLANK


def test_every_image_is_scanned_in_order(caplog: pytest.LogCaptureFixture) -> None:
    # More images than the queue holds, so results arrive while images are still submitted.
    items = ["face" if i % 3 == 0 else "blank" for i in range(60)]
    caplog.set_level(logging.INFO, logger="ryuk.eda.scan")

    scans = Scanner(YUNET, 3).scan(iter(items), _load, total=60, name="synthetic")

    assert [len(s.detections) for s in scans] == [1 if i == "face" else 0 for i in items]
    assert [s.shape for s in scans] == [(256, 256) if i == "face" else (120, 160) for i in items]
    assert "synthetic: scanned 60 images" in caplog.text


def test_progress_is_logged(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scan_module, "_LOG_EVERY", 2)
    caplog.set_level(logging.INFO, logger="ryuk.eda.scan")

    Scanner(YUNET, 1).scan(["blank"] * 4, _load, total=4, name="tiny")

    assert "tiny: scanned 2 of 4 images" in caplog.text
    assert "tiny: scanned 4 of 4 images" in caplog.text


def test_opencv_threads_are_restored() -> None:
    before = cv2.getNumThreads()

    Scanner(YUNET, 2).scan(["blank"], _load, total=1, name="one")

    assert cv2.getNumThreads() == before


def test_a_failed_load_stops_the_scan() -> None:
    def load(item: str) -> Image:
        if item == "bad":
            raise ValueError("cannot decode bad")
        return BLANK

    before = cv2.getNumThreads()
    with pytest.raises(ValueError, match="cannot decode bad"):
        Scanner(YUNET, 2).scan(["ok", "bad", "ok"], load, total=3, name="broken")
    assert cv2.getNumThreads() == before


def test_missing_weights_fail_before_any_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Scanner(tmp_path / "none.onnx")


def test_at_least_one_worker_is_needed() -> None:
    with pytest.raises(ValueError, match="workers"):
        Scanner(YUNET, workers=0)
    assert Scanner(YUNET).workers == default_workers() >= 1


def _detection(x: float, y: float, side: float) -> Detection:
    point = (x + side / 2, y + side / 2)
    return Detection(Box(x, y, side, side), Landmarks(point, point, point, point, point), 0.95)


def test_a_scan_knows_its_centre_most_and_usable_faces() -> None:
    small_at_centre = _detection(90, 90, 20)
    large_in_corner = _detection(0, 0, 60)
    scan = ImageScan((200, 200), (large_in_corner, small_at_centre))

    assert scan.centre_most is small_at_centre
    assert scan.usable(60)
    assert not scan.usable(61)
    assert ImageScan((200, 200), ()).centre_most is None
    assert not ImageScan((200, 200), ()).usable(1)
