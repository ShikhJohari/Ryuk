"""Running YuNet over every image of a dataset, keeping each image's shape and detections.

The scan keeps all detections, not only the benchmark face, because the minimum usable face size
is measured before the usable-face rule can be applied (#17). A `Detector` is not thread-safe,
so each worker thread has its own; OpenCV releases the GIL while decoding and detecting.
"""

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, NamedTuple

import cv2

from ryuk.detector import Detection, Detector, Image, benchmark_face, centre_most

logger = logging.getLogger(__name__)

# Images in flight per worker. Bounds memory when the source is a lazy iterator of CelebA PNGs.
_QUEUED_PER_WORKER: Final = 16
_LOG_EVERY: Final = 2000


class ImageScan(NamedTuple):
    """What YuNet found in one image: every detection, best score first, and the image's size."""

    shape: tuple[int, int]
    """(height, width), in the order of an image array's shape."""
    detections: tuple[Detection, ...]

    @property
    def centre_most(self) -> Detection | None:
        """The detection nearest the image centre, usable or not."""
        return centre_most(self.detections, self.shape)

    def usable(self, min_face_size: int) -> bool:
        """Whether the image has a usable face, so is not excluded from evaluation."""
        return benchmark_face(self.detections, self.shape, min_face_size) is not None


def default_workers() -> int:
    """One worker per CPU core. On an M4 (4 performance and 6 efficiency cores) throughput still
    rises from 8 to 10 workers; at 10 a full `ryuk eda` scans about 1,700 LFW and 1,900 CelebA
    images a second, 53,000 images in about 30 s.
    """
    return os.cpu_count() or 1


@dataclass(frozen=True)
class Scanner:
    """Scans images with one YuNet per worker thread, at the detector's default settings."""

    weights: Path
    """The YuNet ONNX file."""
    workers: int = field(default_factory=default_workers)

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")
        # Fail on missing weights here, not on the first worker thread.
        Detector(self.weights)

    def scan[T](
        self, items: Iterable[T], load: Callable[[T], Image], *, total: int, name: str
    ) -> list[ImageScan]:
        """Scan each item's image, loaded by `load` on a worker thread, in the order of `items`.

        `total` and `name` only label the progress log. OpenCV's own thread pool is turned off
        for the scan and restored after: with a worker per core its threads only contend
        (measured about 15% slower on an M4).
        """
        weights, workers = self.weights, self.workers
        local = threading.local()

        def scan_one(item: T) -> ImageScan:
            detector: Detector | None = getattr(local, "detector", None)
            if detector is None:
                detector = local.detector = Detector(weights)
            image = load(item)
            height, width = image.shape[:2]
            return ImageScan((height, width), tuple(detector.detect(image)))

        started = time.perf_counter()
        scans: list[ImageScan] = []

        def collect(future: Future[ImageScan]) -> None:
            scans.append(future.result())
            if len(scans) % _LOG_EVERY == 0:
                logger.info("%s: scanned %d of %d images", name, len(scans), total)

        threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(workers, thread_name_prefix="yunet") as pool:
                pending: deque[Future[ImageScan]] = deque()
                try:
                    for item in items:
                        pending.append(pool.submit(scan_one, item))
                        if len(pending) >= workers * _QUEUED_PER_WORKER:
                            collect(pending.popleft())
                    while pending:
                        collect(pending.popleft())
                except BaseException:
                    # Don't finish the queued images after a failure or Ctrl-C.
                    pool.shutdown(cancel_futures=True)
                    raise
        finally:
            cv2.setNumThreads(threads)

        elapsed = time.perf_counter() - started
        logger.info(
            "%s: scanned %d images in %.1f s (%.0f a second, %d workers)",
            name,
            len(scans),
            elapsed,
            len(scans) / elapsed if elapsed else 0.0,
            workers,
        )
        return scans
