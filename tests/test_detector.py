from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from ryuk.detector import (
    MIN_USABLE_FACE_SIZE,
    PHOTO_DETECTION_SIDE,
    Box,
    Detection,
    Detector,
    Image,
    Landmarks,
    Point,
    benchmark_face,
    centre_most,
    usable_faces,
)

FIXTURES = Path(__file__).parent / "fixtures"
YUNET = FIXTURES / "face_detection_yunet_2026may.onnx"


def _read(path: Path) -> Image:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, path
    return np.asarray(image, dtype=np.uint8)


ASTRONAUT = _read(FIXTURES / "astronaut.jpg")
# The astronaut's head and hair with some margin, for composing test images.
HEAD = ASTRONAUT[0:260, 120:340]
GREY = 127


def _head(width: int) -> Image:
    height = round(HEAD.shape[0] * width / HEAD.shape[1])
    return np.asarray(cv2.resize(HEAD, (width, height), interpolation=cv2.INTER_AREA), np.uint8)


def _canvas(height: int, width: int, heads: Sequence[tuple[int, int, int]]) -> Image:
    """A grey image with a copy of the astronaut's head, `w` pixels wide, at each (x, y, w)."""
    canvas = np.full((height, width, 3), GREY, dtype=np.uint8)
    for x, y, w in heads:
        head = _head(w)
        canvas[y : y + head.shape[0], x : x + head.shape[1]] = head
    return canvas


def _contains(box: Box, point: Point) -> bool:
    x, y = point
    return box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector(YUNET)


def _detection(x: float, y: float, width: float, height: float) -> Detection:
    """A detection with plausible landmarks; the rules only look at the box."""

    def at(fx: float, fy: float) -> Point:
        return (x + fx * width, y + fy * height)

    return Detection(
        box=Box(x, y, width, height),
        landmarks=Landmarks(at(0.3, 0.4), at(0.7, 0.4), at(0.5, 0.6), at(0.35, 0.8), at(0.65, 0.8)),
        score=0.95,
    )


def test_a_box_knows_its_short_side_and_centre() -> None:
    box = Box(10.0, 20.0, 30.0, 50.0)

    assert box.short_side == 30.0
    assert box.centre == (25.0, 45.0)


@pytest.mark.parametrize(
    ("width", "height", "usable"),
    [
        (MIN_USABLE_FACE_SIZE, MIN_USABLE_FACE_SIZE, True),
        (MIN_USABLE_FACE_SIZE, 200.0, True),
        (MIN_USABLE_FACE_SIZE - 0.1, 200.0, False),
        (200.0, MIN_USABLE_FACE_SIZE - 0.1, False),
    ],
)
def test_a_face_is_usable_when_its_short_side_reaches_the_minimum(
    width: float, height: float, usable: bool
) -> None:
    detection = _detection(0.0, 0.0, width, height)

    assert usable_faces([detection]) == ([detection] if usable else [])


def test_the_minimum_face_size_can_be_overridden() -> None:
    detection = _detection(0.0, 0.0, 30.0, 30.0)

    assert usable_faces([detection], min_face_size=30) == [detection]
    assert usable_faces([detection], min_face_size=31) == []


def test_usable_faces_keep_their_order() -> None:
    big, small, medium = (
        _detection(0.0, 0.0, 120.0, 150.0),
        _detection(200.0, 0.0, 20.0, 25.0),
        _detection(400.0, 0.0, 80.0, 95.0),
    )

    assert usable_faces([big, small, medium]) == [big, medium]


def test_no_detections_give_no_face() -> None:
    assert usable_faces([]) == []
    assert centre_most([], (100, 100, 3)) is None
    assert benchmark_face([], (100, 100, 3)) is None


def test_the_centre_most_detection_is_nearest_the_image_centre() -> None:
    # A wide image (H=200, W=600): the centre is (300, 100), so x and y must not be swapped.
    near_left = _detection(0.0, 50.0, 100.0, 100.0)  # centre (50, 100): 250 away
    near_centre = _detection(170.0, 20.0, 100.0, 100.0)  # centre (220, 70): about 85 away
    below = _detection(250.0, 150.0, 100.0, 100.0)  # centre (300, 200): 100 away

    assert centre_most([near_left, below, near_centre], (200, 600, 3)) is near_centre


def test_a_tie_for_centre_most_goes_to_the_first_detection() -> None:
    left = _detection(0.0, 50.0, 100.0, 100.0)  # centre (50, 100)
    right = _detection(200.0, 50.0, 100.0, 100.0)  # centre (250, 100), mirror image of left

    assert centre_most([left, right], (200, 300)) is left
    assert centre_most([right, left], (200, 300)) is right


def test_the_benchmark_face_is_the_centre_most_usable_face() -> None:
    too_small_at_centre = _detection(90.0, 90.0, 20.0, 20.0)
    usable_off_centre = _detection(0.0, 0.0, 80.0, 80.0)

    detections = [too_small_at_centre, usable_off_centre]
    assert centre_most(detections, (200, 200, 3)) is too_small_at_centre
    assert benchmark_face(detections, (200, 200, 3)) is usable_off_centre
    assert benchmark_face(detections, (200, 200, 3), min_face_size=20) is too_small_at_centre


def _gradient() -> Image:
    ramp = np.linspace(0, 255, 320, dtype=np.uint8)
    return np.ascontiguousarray(np.broadcast_to(ramp[np.newaxis, :, np.newaxis], (240, 320, 3)))


def _noise() -> Image:
    return np.random.default_rng(0).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "image",
    [np.full((240, 320, 3), GREY, dtype=np.uint8), _gradient(), _noise()],
    ids=["blank", "gradient", "noise"],
)
def test_an_image_without_a_face_has_no_detections(detector: Detector, image: Image) -> None:
    assert detector.detect(image) == []
    assert benchmark_face(detector.detect(image), image.shape) is None


def test_the_astronaut_is_one_face(detector: Detector) -> None:
    [face] = detector.detect(ASTRONAUT)

    assert face.score >= 0.9
    assert face.box.short_side >= MIN_USABLE_FACE_SIZE
    assert all(_contains(face.box, point) for point in face.landmarks)
    lm = face.landmarks
    # The subject's right eye and mouth corner appear on the image's left.
    assert lm.right_eye[0] < lm.nose_tip[0] < lm.left_eye[0]
    assert lm.right_mouth_corner[0] < lm.left_mouth_corner[0]
    # Upright: eyes above the nose, the nose above the mouth.
    assert max(lm.right_eye[1], lm.left_eye[1]) < lm.nose_tip[1]
    assert lm.nose_tip[1] < min(lm.right_mouth_corner[1], lm.left_mouth_corner[1])
    assert benchmark_face([face], ASTRONAUT.shape) == face


def test_a_large_close_up_is_missed_at_full_size_and_found_within_the_bound(
    detector: Detector,
) -> None:
    # Six times the astronaut: a 3072 px portrait with a face over 500 px on a side, the shape
    # of a phone close-up. YuNet finds nothing at that size.
    large = np.asarray(
        cv2.resize(ASTRONAUT, (3072, 3072), interpolation=cv2.INTER_CUBIC), dtype=np.uint8
    )
    [small_face] = detector.detect(ASTRONAUT)

    assert detector.detect(large) == []
    [face] = detector.detect(large, max_side=PHOTO_DETECTION_SIDE)

    # The result is in the large image's pixels: six times the astronaut's, give or take the
    # difference between detecting at 512 and at 640 px (about 1% of the image).
    assert face.box.x == pytest.approx(small_face.box.x * 6, abs=40)
    assert face.box.width == pytest.approx(small_face.box.width * 6, abs=40)
    assert face.landmarks.nose_tip[0] == pytest.approx(small_face.landmarks.nose_tip[0] * 6, abs=40)
    assert face.landmarks.nose_tip[1] == pytest.approx(small_face.landmarks.nose_tip[1] * 6, abs=40)
    assert face.score >= 0.9


def test_the_bound_leaves_a_small_image_alone(detector: Detector) -> None:
    assert detector.detect(ASTRONAUT, max_side=PHOTO_DETECTION_SIDE) == detector.detect(ASTRONAUT)


def test_a_face_cut_off_by_the_edge_keeps_its_unclipped_box(detector: Detector) -> None:
    cut = np.ascontiguousarray(ASTRONAUT[:, 185:])

    [face] = detector.detect(cut)

    assert face.box.x < 0


def test_the_benchmark_face_of_several_is_the_one_nearest_the_centre(detector: Detector) -> None:
    # Two large heads in the corners and a smaller, still usable one at the centre of an 800x600
    # image. YuNet boxes them at about 98, 81 and 99 pixels on the short side.
    image = _canvas(600, 800, [(10, 10, 240), (300, 190, 200), (550, 306, 240)])

    detections = detector.detect(image)

    assert len(detections) == 3
    chosen = benchmark_face(detections, image.shape)
    assert chosen is not None
    assert chosen is centre_most(detections, image.shape)
    assert _contains(chosen.box, (400, 300))
    assert chosen.box.short_side == min(d.box.short_side for d in detections)


def test_a_face_below_the_minimum_size_is_not_usable(detector: Detector) -> None:
    # A large head on the left, and a small one at the centre that YuNet still finds: at the
    # default 0.9 score threshold it finds heads whose box is down to about 30 pixels wide.
    image = _canvas(480, 640, [(20, 120, 220), (280, 190, 84)])

    detections = detector.detect(image)

    assert len(detections) == 2
    [usable] = usable_faces(detections)
    [too_small] = [d for d in detections if d is not usable]
    assert too_small.box.short_side < MIN_USABLE_FACE_SIZE
    assert centre_most(detections, image.shape) is too_small
    assert benchmark_face(detections, image.shape) is usable


def test_an_image_whose_only_face_is_too_small_has_no_benchmark_face(detector: Detector) -> None:
    image = _canvas(240, 320, [(120, 70, 84)])

    detections = detector.detect(image)

    assert len(detections) == 1
    assert benchmark_face(detections, image.shape) is None


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((240, 320), dtype=np.uint8),
        np.zeros((240, 320, 1), dtype=np.uint8),
        np.zeros((240, 320, 4), dtype=np.uint8),
        np.zeros((240, 320, 3), dtype=np.float32),
    ],
    ids=["grayscale", "one-channel", "bgra", "float"],
)
def test_only_bgr_uint8_images_are_accepted(detector: Detector, image: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError, match="BGR uint8"):
        detector.detect(image)  # type: ignore[arg-type]


def test_missing_weights_are_reported_by_path(tmp_path: Path) -> None:
    missing = tmp_path / "yunet.onnx"

    with pytest.raises(FileNotFoundError, match=r"yunet\.onnx"):
        Detector(missing)


# Where alignCrop puts the five landmarks in its 112x112 output: the ArcFace template.
TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ]
)


@pytest.mark.parametrize("size", [112, 160])
def test_an_aligned_face_is_a_square_bgr_crop(detector: Detector, size: Literal[112, 160]) -> None:
    [face] = detector.detect(ASTRONAUT)

    aligned = detector.align(ASTRONAUT, face, size)

    assert aligned.shape == (size, size, 3)
    assert aligned.dtype == np.uint8


@pytest.mark.parametrize("size", [112, 160])
def test_alignment_puts_the_landmarks_on_the_template(
    detector: Detector, size: Literal[112, 160]
) -> None:
    [face] = detector.detect(ASTRONAUT)
    aligned = detector.align(ASTRONAUT, face, size)
    # YuNet needs some context around a face, so pad the tight crop before detecting again.
    pad = size // 2
    padded = cv2.copyMakeBorder(aligned, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=GREY)

    [again] = detector.detect(np.asarray(padded, dtype=np.uint8))

    landmarks = np.array(again.landmarks) - pad
    template = TEMPLATE_112 * size / 112
    # YuNet's landmarks are not quite the template's points; a misaligned crop is off by far more.
    assert np.linalg.norm(landmarks - template, axis=1).max() < size / 10


def test_alignment_undoes_rotation_and_scale(detector: Detector) -> None:
    [face] = detector.detect(ASTRONAUT)
    # Rotate by 35 degrees, shrink and shift the image, and move the landmarks with it.
    warp = np.asarray(cv2.getRotationMatrix2D((256.0, 256.0), -35.0, 0.8), dtype=np.float64)
    warp[:, 2] += np.array([20.0, -15.0])
    warped = cv2.warpAffine(ASTRONAUT, warp, (512, 512), flags=cv2.INTER_LINEAR)
    points = np.array(face.landmarks) @ warp[:, :2].T + warp[:, 2]
    moved = Detection(face.box, Landmarks(*((float(x), float(y)) for x, y in points)), face.score)

    upright = detector.align(ASTRONAUT, face).astype(np.int16)
    realigned = detector.align(np.asarray(warped, dtype=np.uint8), moved).astype(np.int16)

    # Only interpolation separates the two; an unaligned crop differs by about 80 levels.
    assert np.abs(realigned - upright).mean() < 5


def test_alignment_accepts_only_bgr_uint8_images(detector: Detector) -> None:
    [face] = detector.detect(ASTRONAUT)

    with pytest.raises(ValueError, match="BGR uint8"):
        detector.align(ASTRONAUT.astype(np.float32), face)
