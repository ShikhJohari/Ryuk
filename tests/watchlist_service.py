"""Running the service in-process on a temporary database, with the real YuNet detector from the
fixtures and fake recognition models standing in for the networks."""

import datetime
import io
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from ryuk.api import create_app
from ryuk.detector import Detector, Image
from ryuk.evaluation.active import Contender
from ryuk.evaluation.results import Interval, LfwGate, Rate, RecognitionModelId
from ryuk.recognition import ModelKey, RecognitionModel
from ryuk.watchlist.database import open_database, sqlite_engine
from ryuk.watchlist.registry import Evaluated, Evaluation, Unavailable
from ryuk.watchlist.service import Clock, Watchlist, start_watchlist, utc_now
from synthetic import YUNET, face

THRESHOLD = 0.9
"""The fake models' frozen threshold: two shots of one look score about 0.9997 under a fake,
two different looks at most about 0.78."""
MS_PER_FACE = 7.5
LFW_GATE = LfwGate(accuracy="scored-pairs", scored_pairs=5917, pairs=6000, tolerance_points=0.5)


def evaluated(
    *keys: ModelKey, first_active: ModelKey | None = None, contenders: Sequence[Contender] = ()
) -> Evaluation:
    """An evaluation that froze THRESHOLD for each key and chose `first_active`, judging
    `contenders` for that choice."""
    return Evaluation(
        {key: Evaluated(THRESHOLD, "best-photo", "five-point", MS_PER_FACE) for key in keys},
        first_active,
        contenders,
        LFW_GATE,
    )


def contender(model: RecognitionModel) -> Contender:
    """`model` as a contender #9's rule judges eligible, so it can be chosen as the fallback."""
    network = model.key.network
    if network == "fake":
        raise ValueError("a contender stands in for a real network")
    return Contender(
        model=RecognitionModelId(
            network=network,
            provider=model.key.provider,
            weights_sha256=model.key.weights_sha256,
            dimension=model.dimension,
        ),
        lfw_gap_points=0.1,
        test_tpir=Rate(value=0.9, ci=Interval(low=0.88, high=0.92)),
        test_fpir=0.01,
        ms_per_face=MS_PER_FACE,
    )


@contextmanager
def serve(
    database: Path,
    models: Sequence[RecognitionModel | Unavailable],
    evaluation: Evaluation,
    *,
    detector: bool = True,
    clock: Clock = utc_now,
) -> Iterator[TestClient]:
    """The service on `database`, as `ryuk serve` starts it but with these models."""

    def start() -> Watchlist:
        return start_watchlist(
            open_database(database),
            Detector(YUNET) if detector else None,
            models,
            evaluation,
            clock,
        )

    with TestClient(
        create_app(start), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as client:
        yield client


class FakeClock:
    """A clock that stands still until the test moves it."""

    def __init__(self, now: datetime.datetime) -> None:
        self.now = now

    def __call__(self) -> datetime.datetime:
        return self.now

    def advance(self, seconds: float) -> datetime.datetime:
        self.now += datetime.timedelta(seconds=seconds)
        return self.now


@contextmanager
def writing(database: Path) -> Iterator[Session]:
    """A transaction on the service's database from beside it, as the live monitor's tracker
    writes sightings; committed on leaving."""
    engine = sqlite_engine(f"sqlite:///{database}")
    try:
        with Session(engine) as session, session.begin():
            yield session
    finally:
        engine.dispose()


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


def frame(image: Image, seq: int = 1, captured_at: int = 0) -> bytes:
    """A live monitor frame message as the client sends it: the 17-byte header (message type
    1, sequence number, capture time in ms, width, height; big-endian) then the JPEG."""
    height, width = image.shape[:2]
    header = struct.pack(">BIQHH", 1, seq, captured_at, width, height)
    return header + encode(image, quality=70)
