"""SFace (opencv_zoo, 2021dec) through OpenCV's DNN module: 128-d embeddings from 112x112 faces."""

from pathlib import Path

import cv2
import numpy as np

from ryuk.detector import Image
from ryuk.recognition import (
    AlignedSize,
    Embedding,
    ModelKey,
    check_aligned,
    l2_normalise,
    weights_key,
)


class SFace:
    """SFace fp32 or, for the int8 footnote only, the quantised int8 weights (#9)."""

    dimension = 128
    input_size: AlignedSize = 112

    def __init__(self, weights: Path) -> None:
        self._key = weights_key("sface", weights, "cpu")
        self._net = cv2.FaceRecognizerSF.create(str(weights), "")

    @property
    def key(self) -> ModelKey:
        return self._key

    def embed(self, face: Image) -> Embedding:
        check_aligned(face, self.input_size)
        # feature() returns a 1 x 128 row that is not normalised (#7).
        return l2_normalise(np.asarray(self._net.feature(face), dtype=np.float32))
