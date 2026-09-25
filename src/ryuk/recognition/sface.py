"""SFace (opencv_zoo, 2021dec) through OpenCV's DNN module: 128-d embeddings from 112x112 faces."""

import errno
from pathlib import Path

import cv2
import numpy as np

from ryuk.detector import Image
from ryuk.fetch.pinned import file_checksum
from ryuk.recognition import AlignedSize, Embedding, ModelKey, check_aligned, l2_normalise


class SFace:
    """SFace fp32 or, for the int8 footnote only, the quantised int8 weights (#9)."""

    dimension = 128
    input_size: AlignedSize = 112

    def __init__(self, weights: Path) -> None:
        if not weights.is_file():
            raise FileNotFoundError(errno.ENOENT, "SFace weights not found", str(weights))
        self._key = ModelKey("sface", file_checksum(weights, "sha256"), "cpu")
        self._net = cv2.FaceRecognizerSF.create(str(weights), "")

    @property
    def key(self) -> ModelKey:
        return self._key

    def embed(self, face: Image) -> Embedding:
        check_aligned(face, self.input_size)
        # feature() returns a 1 x 128 row that is not normalised (#7).
        return l2_normalise(np.asarray(self._net.feature(face), dtype=np.float32))
