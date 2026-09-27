"""The live monitor's binary frame message (#5): a 17-byte big-endian header, then a JPEG.

| Bytes | Field                                  |
|-------|----------------------------------------|
| 0     | message type, 1 for a frame (uint8)    |
| 1-4   | sequence number (uint32)               |
| 5-12  | capture time, client ms epoch (uint64) |
| 13-14 | width (uint16)                         |
| 15-16 | height (uint16)                        |
| 17..  | the frame as JPEG                      |

Boxes in the result are in the pixels of the frame the header describes, so the client can scale
them to whatever size it draws the video at.
"""

import io
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image as PILImage

from ryuk.detector import Image

FRAME_MESSAGE: Final = 0x01
MAX_FRAME_BYTES: Final = 2 * 1024 * 1024
"""Far more than a 640x480 JPEG at quality 0.7 (about 30 KB) or a full HD one."""
MAX_FRAME_SIDE: Final = 1920

_HEADER: Final = struct.Struct(">BIQHH")


@dataclass(frozen=True, slots=True)
class Frame:
    seq: int
    captured_at: int
    width: int
    height: int
    jpeg: bytes


class FrameError(Exception):
    """A message that is not a frame the live monitor can use, with a stable code."""

    def __init__(self, code: str, detail: str, seq: int | None = None) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.seq = seq


def parse_frame(message: bytes) -> Frame:
    """The frame in a binary message, checked against the header alone; the JPEG is decoded by
    `decode_frame`, off the event loop."""
    if len(message) <= _HEADER.size:
        raise FrameError("invalid_frame", "A frame is a 17-byte header followed by a JPEG.")
    kind, seq, captured_at, width, height = _HEADER.unpack_from(message)
    if kind != FRAME_MESSAGE:
        raise FrameError("invalid_frame", f"Message type {kind} is not a frame.", seq)
    if len(message) - _HEADER.size > MAX_FRAME_BYTES:
        raise FrameError(
            "frame_too_large", f"A frame is at most {MAX_FRAME_BYTES // (1024 * 1024)} MB.", seq
        )
    if not (0 < width <= MAX_FRAME_SIDE and 0 < height <= MAX_FRAME_SIDE):
        raise FrameError(
            "frame_too_large",
            f"A frame is at most {MAX_FRAME_SIDE} px on a side; this one is {width}x{height}.",
            seq,
        )
    return Frame(seq, captured_at, width, height, message[_HEADER.size :])


def decode_frame(frame: Frame) -> Image:
    """The frame's JPEG as a BGR image, the size its header says."""
    unreadable = FrameError("invalid_frame", "The frame is not a JPEG that can be read.", frame.seq)
    # Pillow fails in many ways on a damaged file; each means the frame cannot be read.
    try:
        opened = PILImage.open(io.BytesIO(frame.jpeg), formats=["JPEG"])  # the header only
    except MemoryError:
        raise
    except Exception as error:
        raise unreadable from error
    with opened:
        # The header was held to the limits; the JPEG must be the size it claims before any
        # pixel is decoded.
        if opened.size != (frame.width, frame.height):
            raise FrameError(
                "invalid_frame",
                f"The header says {frame.width}x{frame.height}, but the JPEG is "
                f"{opened.width}x{opened.height}.",
                frame.seq,
            )
        try:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        except MemoryError:
            raise
        except Exception as error:
            raise unreadable from error
    return np.ascontiguousarray(rgb[:, :, ::-1])
