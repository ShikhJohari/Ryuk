"""Uploaded photos: checked, turned upright, and re-encoded with every metadata field stripped.

A photo is kept only as its pixels. EXIF (GPS included), ICC profiles, comments and every other
field are dropped by building a fresh image from the decoded pixels before encoding it (#12).
"""

import io
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

from ryuk.detector import Image
from ryuk.watchlist.errors import WatchlistError

MAX_PHOTO_BYTES: Final = 10 * 1024 * 1024
MEDIA_TYPES: Final = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_QUALITY: Final = 95


@dataclass(frozen=True, slots=True)
class Photo:
    """A photo as it is stored: encoded bytes, upright, with no metadata."""

    data: bytes
    media_type: str
    width: int
    height: int

    def pixels(self) -> Image:
        """The stored photo decoded to BGR, the form the detector takes."""
        with PILImage.open(io.BytesIO(self.data)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])


def prepare_photo(upload: bytes) -> Photo:
    """The upload as it will be stored, or a WatchlistError saying why it cannot be.

    JPEG, PNG and WebP are accepted, up to 10 MB, and keep their format; orientation from EXIF
    is applied to the pixels, since the tag that carried it is dropped.
    """
    if len(upload) > MAX_PHOTO_BYTES:
        raise WatchlistError(
            413, "photo_too_large", f"The photo is over {MAX_PHOTO_BYTES // (1024 * 1024)} MB."
        )
    unsupported = WatchlistError(
        422, "unsupported_image", "The file is not a JPEG, PNG or WebP image that can be read."
    )
    try:
        # A decompression bomb is refused outright rather than only warned about.
        with (
            warnings.catch_warnings(action="error", category=PILImage.DecompressionBombWarning),
            PILImage.open(io.BytesIO(upload), formats=list(MEDIA_TYPES)) as opened,
        ):
            image_format = opened.format or ""
            upright = ImageOps.exif_transpose(opened).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        PILImage.DecompressionBombError,
        PILImage.DecompressionBombWarning,
    ) as error:
        raise unsupported from error
    if image_format not in MEDIA_TYPES:
        raise unsupported

    # A fresh image carries none of the original's info: no EXIF, ICC profile or comments.
    clean = PILImage.frombytes("RGB", upright.size, upright.tobytes())
    encoded = io.BytesIO()
    if image_format == "PNG":
        clean.save(encoded, format=image_format)
    else:
        clean.save(encoded, format=image_format, quality=_QUALITY)
    return Photo(encoded.getvalue(), MEDIA_TYPES[image_format], clean.width, clean.height)
