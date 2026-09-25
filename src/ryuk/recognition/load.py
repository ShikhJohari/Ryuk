"""Loading the recognition models Ryuk compares from the weights directory."""

from pathlib import Path

from ryuk.recognition import Network, Provider, RecognitionModel
from ryuk.weights import ARCFACE, FACENET, SFACE

NETWORKS: tuple[Network, ...] = ("sface", "arcface", "facenet")


def load_model(
    network: Network, weights_dir: Path, provider: Provider | None = None
) -> RecognitionModel:
    """The named network with its pinned weights. `provider` only applies to ArcFace.

    For ArcFace, None picks the machine's default provider: CoreML on Apple Silicon, else CPU.
    SFace and FaceNet only run on CPU. The imports are deferred so that loading one network
    never imports the others' runtimes; torch alone takes about a second.
    """
    if provider == "coreml" and network != "arcface":
        raise ValueError(f"{network} runs on cpu only, not coreml")
    match network:
        case "sface":
            from ryuk.recognition.sface import SFace  # noqa: PLC0415

            return SFace(SFACE.path(weights_dir))
        case "arcface":
            from ryuk.recognition.arcface import ArcFace  # noqa: PLC0415

            return ArcFace(ARCFACE.path(weights_dir), provider)
        case "facenet":
            from ryuk.recognition.facenet import FaceNet  # noqa: PLC0415

            return FaceNet(FACENET.path(weights_dir))
