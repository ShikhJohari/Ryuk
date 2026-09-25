"""A fake recognition model for tests: deterministic, instant, and needs no weights.

Its embedding is a fixed random projection of the face's downscaled pixels, so the same face
always gives the same embedding, and similar faces give similar ones.
"""

import hashlib
from typing import Literal

import cv2
import numpy as np

from ryuk.detector import Image
from ryuk.recognition import (
    AlignedSize,
    Embedding,
    ModelKey,
    Network,
    check_aligned,
    l2_normalise,
)

_THUMBNAIL = 16


class FakeRecognitionModel:
    def __init__(
        self,
        *,
        dimension: int = 32,
        input_size: AlignedSize = 112,
        seed: int = 0,
        network: Network | Literal["fake"] = "fake",
    ) -> None:
        """`network` lets the fake stand in for a real network where one is required, as in
        the evaluation harness's tests; its key's weights hash still marks it as fake."""
        self.dimension = dimension
        self.input_size: AlignedSize = input_size
        # The seed plays the part of the weights: another seed is another recognition model.
        digest = hashlib.sha256(f"fake recognition model {seed}".encode()).hexdigest()
        self._key = ModelKey(network, digest, "cpu")
        rng = np.random.default_rng(seed)
        # One extra row for a constant input, so a flat face still has a direction.
        self._projection = rng.standard_normal((_THUMBNAIL * _THUMBNAIL * 3 + 1, dimension))

    @property
    def key(self) -> ModelKey:
        return self._key

    def embed(self, face: Image) -> Embedding:
        check_aligned(face, self.input_size)
        thumbnail = cv2.resize(face, (_THUMBNAIL, _THUMBNAIL), interpolation=cv2.INTER_AREA)
        pixels = thumbnail.astype(np.float64).reshape(-1)
        # Centring makes it blind to overall brightness, as a real model roughly is.
        pixels -= pixels.mean()
        return l2_normalise(np.append(pixels, 1.0) @ self._projection)
