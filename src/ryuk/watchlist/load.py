"""Starting the watchlist from the settings: the database, the weights present and the committed
thresholds. Missing weights are not an error; the models they belong to are unavailable.

Missing or mismatched evaluation results are: the thresholds, and the detector and minimum face
size they were measured with, are what makes a match mean anything, so the service refuses to
start without them rather than run unmeasured.
"""

import logging
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from ryuk.detector import MIN_USABLE_FACE_SIZE, Detector
from ryuk.evaluation.results import DetectorId, Results, read_results
from ryuk.fetch.pinned import file_checksum
from ryuk.recognition import ModelKey, Network, RecognitionModel
from ryuk.recognition.arcface import default_provider
from ryuk.recognition.load import NETWORKS, load_model
from ryuk.settings import Settings
from ryuk.watchlist.database import open_database
from ryuk.watchlist.registry import Evaluation, Unavailable
from ryuk.watchlist.service import Watchlist, start_watchlist
from ryuk.weights import ARCFACE, FACENET, SFACE, YUNET, Weights

logger = logging.getLogger(__name__)

_PINNED: Final[dict[Network, tuple[Weights, int]]] = {
    "sface": (SFACE, 128),
    "arcface": (ARCFACE, 512),
    "facenet": (FACENET, 512),
}
"""Each network's pinned weights and embedding dimension, to name it while it is unavailable."""


class StartupError(RuntimeError):
    """The service cannot start as configured. The message says why and what to do, in one or
    two sentences an operator can act on."""


def open_watchlist(settings: Settings) -> Watchlist:
    """The watchlist as `ryuk serve` runs it, or a StartupError before the database is touched."""
    results = _committed_results(settings.results)
    detector = _detector(settings.weights_dir, results)
    engine = open_database(settings.database)
    models = [_load(network, settings) for network in NETWORKS]
    return start_watchlist(engine, detector, models, Evaluation.from_results(results))


def _committed_results(path: Path) -> Results:
    try:
        results = read_results(path)
    except ValidationError as error:
        raise StartupError(
            f"The evaluation results at {path} do not match their schema: {error}"
        ) from None
    if results is None:
        raise StartupError(
            f"No evaluation results at {path}, so no recognition model has a threshold. Run "
            "Ryuk from the repository root, or point RYUK_RESULTS at evaluation/results.json."
        )
    return results


def _detector(weights_dir: Path, results: Results) -> Detector | None:
    """The detector, checked against the one every block of the results was measured with; None
    when its weights are missing."""
    blocks = {"verification": results.verification.detector}
    if results.identification is not None:
        blocks["identification"] = results.identification.detector
    for block, measured in blocks.items():
        if measured.min_face_size != MIN_USABLE_FACE_SIZE:
            raise StartupError(
                f"Evaluation's {block} results were measured with a minimum usable face size of "
                f"{measured.min_face_size} px, but this service uses {MIN_USABLE_FACE_SIZE} px; "
                "evaluate again or run the version of Ryuk they came from."
            )
    path = YUNET.path(weights_dir)
    try:
        detector = Detector(path)
    except FileNotFoundError:
        logger.warning("No detector weights: enrollment is unavailable until they are fetched")
        return None
    _check_detector_weights(path, blocks)
    return detector


def _check_detector_weights(path: Path, blocks: dict[str, DetectorId]) -> None:
    sha256 = file_checksum(path, "sha256")
    for block, measured in blocks.items():
        if sha256 != measured.weights_sha256:
            raise StartupError(
                f"The detector weights at {path} (sha256 {sha256}) are not the ones evaluation's "
                f"{block} results were measured with ({measured.weights_sha256}); fetch them again "
                "with `ryuk weights fetch`, or evaluate again with these."
            )


def _load(network: Network, settings: Settings) -> RecognitionModel | Unavailable:
    weights, dimension = _PINNED[network]
    # Checked before loading, which for FaceNet would import torch just to find no file.
    if not weights.path(settings.weights_dir).is_file():
        provider = default_provider() if network == "arcface" else "cpu"
        logger.warning("No %s weights: that recognition model is unavailable", network)
        return Unavailable(ModelKey(network, weights.file.checksum.hexdigest, provider), dimension)
    return load_model(network, settings.weights_dir)
