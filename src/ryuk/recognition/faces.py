"""Cutting a detected face out of an image in the form a recognition model takes.

`five-point` is YuNet's landmarks mapped onto the ArcFace template by OpenCV's alignCrop, which
every network can take. `box-margin-14` is FaceNet's native crop, the YuNet box widened by a
14-pixel margin at the output scale as facenet-pytorch's `extract_face` does, which LFW View 1
weighs against the five-point crop for FaceNet (#7, #9).
"""

from typing import Literal

import cv2
import numpy as np

from ryuk.detector import Detection, Detector, Image
from ryuk.recognition import AlignedSize

type Crop = Literal["five-point", "box-margin-14"]
CROPS: tuple[Crop, ...] = ("five-point", "box-margin-14")

_MARGIN = 14


def face_crop(
    detector: Detector, image: Image, detection: Detection, crop: Crop, size: AlignedSize
) -> Image:
    """The detected face as a `size` x `size` BGR crop."""
    match crop:
        case "five-point":
            return detector.align(image, detection, size)
        case "box-margin-14":
            return box_crop(image, detection, size, _MARGIN)


def box_crop(image: Image, detection: Detection, size: int, margin: int) -> Image:
    """The detection's box, widened by `margin` pixels of the output, clipped to the image.

    The rule is facenet-pytorch's: the margin is given at the output size, so it is scaled by
    box / (size - margin) to the image, half on each side, and the result is squashed to
    `size` x `size` without keeping the aspect ratio.
    """
    height, width = image.shape[:2]
    box = detection.box
    margin_x = margin * box.width / (size - margin)
    margin_y = margin * box.height / (size - margin)
    left = int(max(box.x - margin_x / 2, 0))
    top = int(max(box.y - margin_y / 2, 0))
    right = int(min(box.x + box.width + margin_x / 2, width))
    bottom = int(min(box.y + box.height + margin_y / 2, height))
    if right <= left or bottom <= top:
        raise ValueError(f"the detection's box {box} lies outside the {width}x{height} image")
    face = cv2.resize(image[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA)
    return np.asarray(face, dtype=np.uint8)
