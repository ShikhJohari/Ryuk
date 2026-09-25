"""ArcFace `w600k_r50` from InsightFace's buffalo_l, run directly on onnxruntime (#18)."""

import errno
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ryuk.detector import Image
from ryuk.fetch.pinned import file_checksum
from ryuk.recognition import (
    AlignedSize,
    Embedding,
    ModelKey,
    Provider,
    check_aligned,
    l2_normalise,
)

_COREML = "CoreMLExecutionProvider"
_CPU = "CPUExecutionProvider"
# MLProgram matches CPU embeddings; NeuralNetwork is faster but shifts them slightly (#7).
_COREML_OPTIONS = {"ModelFormat": "MLProgram"}
# onnxruntime logs a warning for every model CoreML only partly covers; errors still surface.
_LOG_ERRORS_ONLY = 3


def default_provider() -> Provider:
    """CoreML on Apple Silicon when onnxruntime offers it, CPU everywhere else."""
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    return "coreml" if apple_silicon and _COREML in ort.get_available_providers() else "cpu"


class ArcFace:
    """512-d embeddings from 112x112 faces. On CoreML and on CPU it is two recognition models."""

    dimension = 512
    input_size: AlignedSize = 112

    def __init__(self, weights: Path, provider: Provider | None = None) -> None:
        if not weights.is_file():
            raise FileNotFoundError(errno.ENOENT, "ArcFace weights not found", str(weights))
        provider = default_provider() if provider is None else provider
        if provider == "coreml" and _COREML not in ort.get_available_providers():
            raise RuntimeError("onnxruntime on this machine has no CoreML execution provider")
        self._key = ModelKey("arcface", file_checksum(weights, "sha256"), provider)

        options = ort.SessionOptions()
        options.log_severity_level = _LOG_ERRORS_ONLY
        providers: list[str | tuple[str, dict[str, str]]] = (
            [(_COREML, _COREML_OPTIONS), _CPU] if provider == "coreml" else [_CPU]
        )
        self._session = ort.InferenceSession(str(weights), options, providers=providers)
        self._input = self._session.get_inputs()[0].name

    @property
    def key(self) -> ModelKey:
        return self._key

    def embed(self, face: Image) -> Embedding:
        check_aligned(face, self.input_size)
        # InsightFace's preprocessing: RGB, (x - 127.5) / 127.5, NCHW.
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis]
        (output,) = self._session.run(None, {self._input: blob})
        return l2_normalise(np.asarray(output, dtype=np.float32))
