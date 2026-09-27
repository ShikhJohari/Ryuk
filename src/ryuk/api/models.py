"""The recognition models, their states and the active model (#12)."""

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from ryuk.api.dependencies import LiveMonitorDep, WatchlistDep
from ryuk.api.monitor import ActiveModelChanged
from ryuk.api.schema import ApiModel
from ryuk.recognition import Network, Provider
from ryuk.watchlist.registry import ModelRegistry, ModelState, RegisteredModel

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


class ActiveModelChoice(ApiModel):
    model_key: str
    """The `id` of an evaluated model whose weights are present."""


@router.get("/models")
def get_models(watchlist: WatchlistDep) -> list[RecognitionModelInfo]:
    registry = watchlist.registry
    return [_info(registry, model) for model in registry.models]


@router.put("/active-model")
async def set_active_model(
    watchlist: WatchlistDep, live_monitor: LiveMonitorDep, choice: ActiveModelChoice
) -> RecognitionModelInfo:
    """Switch the active model: `409 cannot_be_active` for a model that is unavailable or not
    evaluated. The live monitor is told, and its next frame is judged by the new model."""
    model = await run_in_threadpool(watchlist.activate, choice.model_key)
    assert model.evaluated is not None  # noqa: S101 - only an evaluated model can be active
    await live_monitor.announce(
        ActiveModelChanged(
            type="active_model_changed",
            model_key=model.key.id,
            threshold=model.evaluated.threshold,
        )
    )
    return _info(watchlist.registry, model)


def _info(registry: ModelRegistry, model: RegisteredModel) -> RecognitionModelInfo:
    return RecognitionModelInfo(
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
