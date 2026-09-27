"""Face detection with YuNet, the usable-face rule, and alignment for the recognition models."""

import errno
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

# Measured by #17's rule and recorded in eda/summary.json (`min_usable_face_size`): the largest
# multiple of 10 px, on the box's short side, that keeps at least 99% of CelebA detections (the
# centre-most detection of every detected image in both draws; 70 px keeps 99.5%, 80 px 84.7%).
# It governs enrollment and the live monitor as well as evaluation. `ryuk eda` remeasures it;
# a test holds this constant to the committed summary.
MIN_USABLE_FACE_SIZE: Final = 70

# YuNet's face score and non-maximum suppression thresholds, the detector's defaults everywhere.
SCORE_THRESHOLD: Final = 0.9
NMS_THRESHOLD: Final = 0.3

# YuNet misses faces much over about 400 px on a side, so a close-up phone portrait at full
# resolution finds no face at all. Enrollment detects on a copy bounded to this long side and
# scales the result back; benchmark images and camera frames are smaller and are unaffected.
PHOTO_DETECTION_SIDE: Final = 640

type Point = tuple[float, float]
"""An (x, y) position in image pixels."""

type Image = NDArray[np.uint8]
"""A BGR image, height x width x 3, as OpenCV decodes it."""


@dataclass(frozen=True, slots=True)
class Box:
    """A face's bounding box as YuNet reports it, in image pixels.

    Not clipped: a face cut off by the frame gets a box that extends past the image edge,
    so `x` or `y` can be negative and `x + width` can exceed the image width.
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def short_side(self) -> float:
        return min(self.width, self.height)

    @property
    def centre(self) -> Point:
        return (self.x + self.width / 2, self.y + self.height / 2)


class Landmarks(NamedTuple):
    """YuNet's five facial landmarks, in YuNet's order.

    "Right" and "left" are the person's: for an upright face looking at the camera, the right
    eye and right mouth corner are the ones on the image's left (smaller x). YuNet labels them
    by that position, not by anatomy, so in a mirrored image such as a selfie view `right_eye`
    is still the eye on the image's left.
    """

    right_eye: Point
    left_eye: Point
    nose_tip: Point
    right_mouth_corner: Point
    left_mouth_corner: Point


@dataclass(frozen=True, slots=True)
class Detection:
    """A face found in an image, in that image's pixels. Says nothing about who it is.

    `score` is YuNet's face score, from 0 to 1.
    """

    box: Box
    landmarks: Landmarks
    score: float


class Detector:
    """The one YuNet face detector shared by enrollment, the live monitor and evaluation.

    `weights` is the YuNet ONNX file; the threshold and limits are YuNet's own, passed through.
    Not thread-safe: each call resizes the network's input to the image, so give every thread
    its own instance.
    """

    def __init__(
        self,
        weights: Path,
        *,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = 5000,
    ) -> None:
        # OpenCV's own error for a missing file is an opaque "Can't read ONNX file" cv2.error.
        if not weights.is_file():
            raise FileNotFoundError(errno.ENOENT, "YuNet weights not found", str(weights))
        # The input size is a placeholder; detect() sets the real one per image.
        self._yunet = cv2.FaceDetectorYN.create(
            str(weights), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        # alignCrop is an instance method of the SFace recognizer, and create() refuses an empty
        # model path because it calls dnn::readNet. alignCrop never runs that network, so
        # YuNet's own ONNX stands in and alignment needs no SFace weights.
        self._aligner = cv2.FaceRecognizerSF.create(str(weights), "")

    def detect(self, image: Image, *, max_side: int | None = None) -> list[Detection]:
        """Every face YuNet finds in a BGR uint8 image, best score first.

        With `max_side`, an image whose long side is larger is detected on a copy shrunk to that
        side, and every box and landmark is scaled back to `image`'s pixels. Scores are the
        copy's.
        """
        _check_bgr(image)
        height, width = image.shape[:2]
        factor = 1.0
        if max_side is not None and max(height, width) > max_side:
            factor = max(height, width) / max_side
            image = np.asarray(
                cv2.resize(
                    image,
                    (max(1, round(width / factor)), max(1, round(height / factor))),
                    interpolation=cv2.INTER_AREA,
                ),
                dtype=np.uint8,
            )
            height, width = image.shape[:2]
        self._yunet.setInputSize((width, height))
        # The stubs say detect() always returns an array, but it returns None for no faces.
        faces: cv2.typing.MatLike | None
        _, faces = self._yunet.detect(image)
        if faces is None:
            return []
        return [_detection(row, factor) for row in np.asarray(faces, dtype=np.float32)]

    def align(self, image: Image, detection: Detection, size: Literal[112, 160] = 112) -> Image:
        """The detected face as a size x size BGR crop, aligned for a recognition model.

        OpenCV's alignCrop maps the five landmarks onto the ArcFace template at 112x112, which
        SFace and ArcFace take as is. For FaceNet, 160, that crop is resized (#7).
        """
        _check_bgr(image)
        aligned = self._aligner.alignCrop(image, _yunet_row(detection))
        if size == 160:
            aligned = cv2.resize(aligned, (160, 160), interpolation=cv2.INTER_LINEAR)
        return np.asarray(aligned, dtype=np.uint8)


def _check_bgr(image: Image) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"expected a BGR uint8 image of shape (height, width, 3), "
            f"got {image.dtype} of shape {image.shape}"
        )


def _detection(row: NDArray[np.float32], factor: float = 1.0) -> Detection:
    """A Detection from one YuNet output row, its coordinates multiplied by `factor`.

    The row holds 15 floats: box (x, y, width, height), five landmarks as x, y pairs in
    `Landmarks` order, then the score.
    """
    x, y, width, height = (float(v) * factor for v in row[:4])
    points = [(float(row[i]) * factor, float(row[i + 1]) * factor) for i in range(4, 14, 2)]
    return Detection(Box(x, y, width, height), Landmarks(*points), float(row[14]))


def _yunet_row(detection: Detection) -> NDArray[np.float32]:
    """The 15-float YuNet row for a Detection, the form alignCrop takes."""
    box = detection.box
    coords = [c for point in detection.landmarks for c in point]
    return np.array(
        [box.x, box.y, box.width, box.height, *coords, detection.score], dtype=np.float32
    )


def is_usable(detection: Detection, min_face_size: int = MIN_USABLE_FACE_SIZE) -> bool:
    """Whether a detection is large enough to use: box short side at least `min_face_size`."""
    return detection.box.short_side >= min_face_size


def usable_faces(
    detections: Iterable[Detection], min_face_size: int = MIN_USABLE_FACE_SIZE
) -> list[Detection]:
    """The detections large enough to use, in order."""
    return [d for d in detections if is_usable(d, min_face_size)]


def centre_most(detections: Iterable[Detection], image_shape: Sequence[int]) -> Detection | None:
    """The detection whose box centre is nearest the image centre, or None if there are none.

    `image_shape` is the image array's shape, (height, width) first. A tie goes to the earlier
    detection.
    """
    height, width = image_shape[0], image_shape[1]
    image_centre = (width / 2, height / 2)
    return min(detections, key=lambda d: math.dist(d.box.centre, image_centre), default=None)


def benchmark_face(
    detections: Iterable[Detection],
    image_shape: Sequence[int],
    min_face_size: int = MIN_USABLE_FACE_SIZE,
) -> Detection | None:
    """The one face a benchmark image contributes: its centre-most usable face (#17).

    None means the image has no usable face and is excluded. The rule is the same for LFW and
    CelebA.
    """
    return centre_most(usable_faces(detections, min_face_size), image_shape)
