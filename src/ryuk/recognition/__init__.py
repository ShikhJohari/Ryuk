"""Recognition models: an aligned face in, an L2-normalised float32 embedding out.

A recognition model is identified by its network, the sha256 of the weights it loaded and the
execution provider it runs on (`ModelKey`). The same network with other weights, or on another
provider, is a different recognition model, and its embeddings must never be compared with
this one's.
"""

import errno
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from ryuk.detector import Image
from ryuk.fetch.pinned import file_checksum

type Network = Literal["arcface", "facenet", "sface"]
"""The pretrained networks Ryuk compares (#9). The fake model used in tests is not one."""

type Provider = Literal["cpu", "coreml"]
"""Where a network runs. `coreml` is onnxruntime's CoreML provider in MLProgram format (#7)."""

type AlignedSize = Literal[112, 160]
"""The side of the square aligned face a network takes: 112 (SFace, ArcFace) or 160 (FaceNet)."""

type Embedding = NDArray[np.float32]
"""A 1-D, L2-normalised float32 vector of the model's dimension."""


@dataclass(frozen=True, slots=True)
class ModelKey:
    """What identifies a recognition model: network, exact weights and execution provider."""

    network: Network | Literal["fake"]
    weights_sha256: str
    provider: Provider

    @property
    def id(self) -> str:
        """The key as one string, safe as a file or directory name."""
        return f"{self.network}-{self.provider}-{self.weights_sha256}"


def weights_key(network: Network, weights: Path, provider: Provider) -> ModelKey:
    """The key for `network` loaded from `weights`, hashing the file it will actually load.

    Raises FileNotFoundError naming the path if the weights are missing, since each runtime's
    own error for a missing file is opaque.
    """
    if not weights.is_file():
        raise FileNotFoundError(errno.ENOENT, f"{network} weights not found", str(weights))
    return ModelKey(network, file_checksum(weights, "sha256"), provider)


class RecognitionModel(Protocol):
    """A frozen network that maps one aligned face to an embedding.

    Implementations embed one face per call, so an image's embedding never depends on which
    other faces shared a batch. None is thread-safe; give each thread its own instance.
    """

    @property
    def key(self) -> ModelKey: ...

    @property
    def dimension(self) -> int: ...

    @property
    def input_size(self) -> AlignedSize: ...

    def embed(self, face: Image) -> Embedding:
        """The embedding of a BGR uint8 face aligned to `input_size` x `input_size`."""
        ...


def check_aligned(face: Image, size: AlignedSize) -> None:
    if face.dtype != np.uint8 or face.shape != (size, size, 3):
        raise ValueError(
            f"expected an aligned BGR uint8 face of shape ({size}, {size}, 3), "
            f"got {face.dtype} of shape {face.shape}"
        )


def l2_normalise(vector: NDArray[np.floating]) -> Embedding:
    """`vector` flattened, as float32, scaled to unit length."""
    flat = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalise a zero or non-finite embedding")
    return np.asarray(flat / norm, dtype=np.float32)
