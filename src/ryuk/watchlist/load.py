"""Starting the watchlist from the settings: the database, the weights present and the committed
thresholds. Missing weights are not an error; the models they belong to are unavailable."""

import logging
from typing import Final

from ryuk.detector import Detector
from ryuk.evaluation.results import read_results
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


def open_watchlist(settings: Settings) -> Watchlist:
    engine = open_database(settings.database)
    try:
        detector: Detector | None = Detector(YUNET.path(settings.weights_dir))
    except FileNotFoundError:
        logger.warning("No detector weights: enrollment is unavailable until they are fetched")
        detector = None
    models = [_load(network, settings) for network in NETWORKS]
    evaluation = Evaluation.from_results(read_results(settings.results))
    return start_watchlist(engine, detector, models, evaluation)


def _load(network: Network, settings: Settings) -> RecognitionModel | Unavailable:
    weights, dimension = _PINNED[network]
    # Checked before loading, which for FaceNet would import torch just to find no file.
    if not weights.path(settings.weights_dir).is_file():
        provider = default_provider() if network == "arcface" else "cpu"
        logger.warning("No %s weights: that recognition model is unavailable", network)
        return Unavailable(ModelKey(network, weights.file.checksum.hexdigest, provider), dimension)
    return load_model(network, settings.weights_dir)
