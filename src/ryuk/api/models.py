"""The recognition models and their states (#12)."""

from fastapi import APIRouter

from ryuk.api.dependencies import WatchlistDep
from ryuk.api.schema import ApiModel
from ryuk.recognition import Network, Provider
from ryuk.watchlist.registry import ModelState

router = APIRouter(tags=["models"])


class RecognitionModelInfo(ApiModel):
    id: str
    """The model key as one string: network, provider and weights sha256."""
    network: Network
    provider: Provider
    weights_sha256: str
    name: str
    state: ModelState
    threshold: float | None
    """The threshold frozen by evaluation; None when the model was not evaluated."""
    dimension: int
    ms_per_face: float | None
    """Evaluation's measured time per face, on the machine it ran on."""


@router.get("/models")
def get_models(watchlist: WatchlistDep) -> list[RecognitionModelInfo]:
    registry = watchlist.registry
    return [
        RecognitionModelInfo(
            id=model.key.id,
            network=model.identity.network,
            provider=model.identity.provider,
            weights_sha256=model.identity.weights_sha256,
            name=model.name,
            state=registry.state(model),
            threshold=None if model.evaluated is None else model.evaluated.threshold,
            dimension=model.identity.dimension,
            ms_per_face=None if model.evaluated is None else model.evaluated.ms_per_face,
        )
        for model in registry.models
    ]
