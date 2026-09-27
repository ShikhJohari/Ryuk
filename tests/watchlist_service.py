"""Running the service in-process on a temporary database, with the real YuNet detector from the
fixtures and fake recognition models standing in for the networks."""

import io
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from ryuk.api import create_app
from ryuk.detector import Detector, Image
from ryuk.recognition import ModelKey, RecognitionModel
from ryuk.watchlist.database import open_database
from ryuk.watchlist.registry import Evaluated, Evaluation, Unavailable
from ryuk.watchlist.service import Watchlist, start_watchlist
from synthetic import YUNET, face

THRESHOLD = 0.9
"""The fake models' frozen threshold: two shots of one look score about 0.9997 under a fake,
two different looks at most about 0.78."""
MS_PER_FACE = 7.5


def evaluated(*keys: ModelKey, first_active: ModelKey | None = None) -> Evaluation:
    """An evaluation that froze THRESHOLD for each key and chose `first_active`."""
    return Evaluation(
        {key: Evaluated(THRESHOLD, "five-point", MS_PER_FACE) for key in keys}, first_active
    )


@contextmanager
def serve(
    database: Path,
    models: Sequence[RecognitionModel | Unavailable],
    evaluation: Evaluation,
    *,
    detector: bool = True,
) -> Iterator[TestClient]:
    """The service on `database`, as `ryuk serve` starts it but with these models."""

    def start() -> Watchlist:
        return start_watchlist(
            open_database(database), Detector(YUNET) if detector else None, models, evaluation
        )

    with TestClient(
        create_app(start), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as client:
        yield client


def portrait(look: int, shot: int = 0, size: int = 400) -> Image:
    """One large face, `size` px square; the same look is the same person."""
    return np.asarray(cv2.resize(face(look, shot), (size, size)), dtype=np.uint8)


def group(*looks: int) -> Image:
    """Several people side by side, each face large enough to use."""
    return np.hstack([portrait(look) for look in looks])


def with_bystander(look: int, bystander: int) -> Image:
    """A large face with a much smaller one in the background, too small to use."""
    canvas = np.full((700, 700, 3), 127, dtype=np.uint8)
    canvas[:400, :400] = portrait(look)
    canvas[520:640, 520:640] = cv2.resize(face(bystander, 0), (120, 120))
    return canvas


def distant(look: int) -> Image:
    """A lone face that is detected but too small to use."""
    canvas = np.full((400, 400, 3), 127, dtype=np.uint8)
    canvas[140:260, 140:260] = cv2.resize(face(look, 0), (120, 120))
    return canvas


def blank() -> Image:
    return np.full((300, 300, 3), 127, dtype=np.uint8)


def encode(image: Image, image_format: str = "JPEG", **options: object) -> bytes:
    """A BGR image as the bytes of an uploaded file."""
    buffer = io.BytesIO()
    rgb = PILImage.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
    rgb.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def upload(image: Image | bytes) -> dict[str, tuple[str, bytes, str]]:
    data = image if isinstance(image, bytes) else encode(image)
    return {"photo": ("photo.jpg", data, "image/jpeg")}
