import numpy as np
import pytest

from ryuk.detector import Box, Detection, Landmarks
from ryuk.recognition.faces import box_crop


def _detection(x: float, y: float, width: float, height: float) -> Detection:
    centre = (x + width / 2, y + height / 2)
    return Detection(Box(x, y, width, height), Landmarks(*[centre] * 5), 0.95)


def _image() -> np.ndarray:
    # Each pixel's value is its column, so a crop's columns show where it was cut.
    columns = np.tile(np.arange(200, dtype=np.uint8), (200, 1))
    return np.repeat(columns[:, :, np.newaxis], 3, axis=2)


def test_the_margin_is_given_at_the_output_size_and_split_between_the_sides() -> None:
    # A 146-pixel box from column 27 to 173, cropped to 160 with margin 14:
    # 14 * 146 / (160 - 14) = 14 source pixels, 7 each side, so columns 20 to 179 are kept,
    # 160 of them, and the resize to 160 leaves them as they are.
    crop = box_crop(_image(), _detection(27, 27, 146, 146), 160, 14)

    assert crop.shape == (160, 160, 3)
    assert crop.dtype == np.uint8
    assert int(crop[0, 0, 0]) == pytest.approx(20, abs=1)
    assert int(crop[0, -1, 0]) == pytest.approx(179, abs=1)


def test_a_box_near_the_edge_is_clipped_to_the_image() -> None:
    crop = box_crop(_image(), _detection(-10, -10, 60, 60), 112, 32)

    assert crop.shape == (112, 112, 3)
    assert crop[0, 0, 0] == 0


def test_a_box_outside_the_image_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        box_crop(_image(), _detection(300, 300, 50, 50), 160, 14)
