"""The names tables, figures and prose give recognition models."""

from ryuk.evaluation.results import RecognitionModelId

NAMES = {"sface": "SFace", "arcface": "ArcFace", "facenet": "FaceNet"}
PROVIDERS = {"cpu": "CPU", "coreml": "CoreML"}


def model_name(model: RecognitionModelId) -> str:
    """The name a table or figure shows: the network, and the provider where it matters."""
    name = NAMES[model.network]
    return f"{name} ({PROVIDERS[model.provider]})" if model.network == "arcface" else name
