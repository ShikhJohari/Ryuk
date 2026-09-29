"""Uploaded photos: checked, turned upright, bounded in size, and re-encoded with every metadata
field stripped.

A photo is kept only as its pixels. EXIF (GPS included), ICC profiles, comments and every other
field are dropped by building a fresh image from the decoded pixels before encoding it (#12).

A photo is stored with its long side at most MAX_PHOTO_SIDE px, shrunk if it arrives larger: a
12 MP phone photo is kept at 2048x1536. This bounds the stored photo, the memory the startup
rebuild takes to decode every photo, and the detector's input; recognition works on 112 px face
crops, so matching loses nothing. The photo as uploaded is not kept.
"""

import io
import threading
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from ryuk.detector import Image
from ryuk.watchlist.errors import WatchlistError

MAX_PHOTO_BYTES: Final = 10 * 1024 * 1024
MAX_PHOTO_PIXELS: Final = 40_000_000
"""More than any phone camera takes, and under Pillow's decompression bomb limit (about 89 MP)."""
MAX_PHOTO_SIDE: Final = 2048
MEDIA_TYPES: Final = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_STORED_AS: Final = {"MPO": "JPEG"}
"""Formats Pillow reports that are stored as another. A JPEG with a multi-picture (MPF) segment,
as some cameras and phones write, opens as MPO; its first picture is kept, as a plain JPEG."""
_QUALITY: Final = 95

# Decoding a photo near the pixel limit takes hundreds of MB, and it runs before the watchlist
# lock is taken, on any of the server's worker threads; two at a time bounds that memory.
_DECODING: Final = threading.BoundedSemaphore(2)


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

    JPEG, PNG and WebP are accepted, up to 10 MB and 40 megapixels, and keep their format (a
    multi-picture JPEG is kept as a JPEG of its first picture); orientation from EXIF is applied to
    the pixels, since the tag that carried it is dropped.
    """
    if len(upload) > MAX_PHOTO_BYTES:
        raise WatchlistError(
            413, "photo_too_large", f"The photo is over {MAX_PHOTO_BYTES // (1024 * 1024)} MB."
        )
    with _DECODING:
        image_format, upright = _decode(upload)
    # A fresh image carries none of the original's info: no EXIF, ICC profile or comments.
    clean = PILImage.frombytes("RGB", upright.size, upright.tobytes())
    encoded = io.BytesIO()
    if image_format == "PNG":
        clean.save(encoded, format=image_format)
    else:
        clean.save(encoded, format=image_format, quality=_QUALITY)
    return Photo(encoded.getvalue(), MEDIA_TYPES[image_format], clean.width, clean.height)


def _decode(upload: bytes) -> tuple[str, PILImage.Image]:
    """The upload's format and its pixels as RGB, upright, long side at most MAX_PHOTO_SIDE."""
    unsupported = WatchlistError(
        422, "unsupported_image", "The file is not a JPEG, PNG or WebP image that can be read."
    )
    too_large = WatchlistError(
        413, "photo_too_large", f"The photo is over {MAX_PHOTO_PIXELS // 1_000_000} megapixels."
    )
    # Pillow's decoders fail in many ways on a damaged file: OSError, SyntaxError (a PNG with a
    # corrupt chunk), ValueError, EOFError, struct.error, IndexError and more. Every one means
    # the file cannot be read, so each is caught here; only running out of memory is not.
    try:
        # Reads only the header. Pillow's own bomb check refuses a size far over the pixel limit
        # here (and warns about one just over its limit), before any pixel is decoded.
        opened = PILImage.open(io.BytesIO(upload), formats=list(MEDIA_TYPES))
    except (PILImage.DecompressionBombError, PILImage.DecompressionBombWarning) as error:
        raise too_large from error
    except MemoryError:
        raise
    except Exception as error:
        raise unsupported from error
    with opened:
        image_format = opened.format or ""
        image_format = _STORED_AS.get(image_format, image_format)
        if image_format not in MEDIA_TYPES:
            raise unsupported
        if opened.width * opened.height > MAX_PHOTO_PIXELS:
            raise too_large
        try:
            # A large JPEG is decoded at a reduced scale, both sides still at least the bound.
            opened.draft(None, (MAX_PHOTO_SIDE, MAX_PHOTO_SIDE))
            ImageOps.exif_transpose(opened, in_place=True)
            upright = opened.convert("RGB")
        except MemoryError:
            raise
        except Exception as error:
            raise unsupported from error
    upright.thumbnail((MAX_PHOTO_SIDE, MAX_PHOTO_SIDE), PILImage.Resampling.LANCZOS)
    return image_format, upright
