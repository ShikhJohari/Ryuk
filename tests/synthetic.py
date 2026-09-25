"""Fake recognition models and synthetic faces shared by the evaluation harness tests."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ryuk.detector import Image
from ryuk.recognition import AlignedSize, Embedding, ModelKey, Network
from ryuk.recognition.fake import FakeRecognitionModel

FIXTURES = Path(__file__).parent / "fixtures"
YUNET = FIXTURES / "face_detection_yunet_2026may.onnx"
ASTRONAUT = np.asarray(cv2.imread(str(FIXTURES / "astronaut.jpg"), cv2.IMREAD_COLOR), np.uint8)


class Counting:
    """A fake standing in for a real network: counts its embeddings, to show what the cache
    saves, and presents its key under that network, as the results only take real ones."""

    def __init__(self, model: FakeRecognitionModel, network: Network) -> None:
        self._model = model
        self._network: Network = network
        self.calls = 0

    @property
    def key(self) -> ModelKey:
        return replace(self._model.key, network=self._network)

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def input_size(self) -> AlignedSize:
        return self._model.input_size

    def embed(self, face: Image) -> Embedding:
        self.calls += 1
        return self._model.embed(face)


def fake(network: Network, seed: int = 0) -> Counting:
    size: AlignedSize = 160 if network == "facenet" else 112
    return Counting(FakeRecognitionModel(input_size=size, seed=seed), network)


LOOKS = 6
"""How many identities `face` can tell apart: mirrored or not, times three channel orders."""


def face(look: int, shot: int) -> Image:
    """A 250x250 image of one large face. Looks differ in pixels; shots of one differ slightly."""
    head = cv2.resize(np.asarray(ASTRONAUT[0:300, 100:350]), (200, 240))
    if look % 2:
        head = np.ascontiguousarray(head[:, ::-1])
    head = np.roll(head, (look // 2) % 3, axis=2)
    canvas = np.full((250, 250, 3), 127, dtype=np.uint8)
    canvas[5:245, 25:225] = head
    return np.clip(canvas.astype(np.int16) + 2 * shot, 0, 255).astype(np.uint8)
