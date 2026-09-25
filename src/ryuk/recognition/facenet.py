"""FaceNet (Inception-ResNet-v1 trained on VGGFace2) through facenet-pytorch 2.5.3 on CPU."""

from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

from ryuk.detector import Image
from ryuk.recognition import (
    AlignedSize,
    Embedding,
    ModelKey,
    check_aligned,
    l2_normalise,
    weights_key,
)


class FaceNet:
    """512-d embeddings from 160x160 faces."""

    dimension = 512
    input_size: AlignedSize = 160

    def __init__(self, weights: Path) -> None:
        self._key = weights_key("facenet", weights, "cpu")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # The checkpoint also holds the VGGFace2 classifier head, which embeddings never use.
        # Loading from a path here keeps facenet-pytorch from downloading into the torch cache.
        network = InceptionResnetV1()
        network.load_state_dict({k: v for k, v in state.items() if not k.startswith("logits.")})
        self._network = network.eval()

    @property
    def key(self) -> ModelKey:
        return self._key

    def embed(self, face: Image) -> Embedding:
        check_aligned(face, self.input_size)
        # fixed_image_standardization, (x - 127.5) / 128, on RGB; raw pixels collapse
        # discrimination (#7).
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = torch.from_numpy(((rgb - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis].copy())
        with torch.inference_mode():
            output = self._network(tensor)
        return l2_normalise(output.numpy())
