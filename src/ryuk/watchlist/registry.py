"""The recognition models the service knows, what state each is in, and the active model.

A model is `unavailable` without its weights, `not_evaluated` without a threshold from the
committed evaluation for its exact key, and otherwise `available`, or `active` if it is the one
the live monitor uses (#12). On Linux, ArcFace runs on CPU, a different recognition model from
the CoreML one that was evaluated, so it is not evaluated there.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ryuk.evaluation.active import Contender, first_active_model
from ryuk.evaluation.names import model_name
from ryuk.evaluation.results import MatchRule, RecognitionModelId, Results
from ryuk.recognition import ModelKey, Network, RecognitionModel
from ryuk.recognition.faces import Crop

type ModelState = Literal["active", "available", "not_evaluated", "unavailable"]

LIVE_RULES: frozenset[MatchRule] = frozenset({"best-photo"})
"""The match rules the service can compute. A threshold for any other rule is not used."""


@dataclass(frozen=True, slots=True)
class Evaluated:
    """What evaluation measured for one recognition model, as far as the service needs it."""

    threshold: float
    crop: Crop
    """The crop the threshold was measured with; enrollment must cut faces the same way."""
    ms_per_face: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The thresholds the committed evaluation froze, by exact model key, the first active
    model it chose, and every model it judged for that choice."""

    models: Mapping[ModelKey, Evaluated]
    first_active: ModelKey | None
    contenders: Sequence[Contender] = ()

    def first_active_for(self, runnable: Iterable[ModelKey]) -> ModelKey | None:
        """The first active model among the models this machine can run.

        Evaluation's choice when it can run here; otherwise #9's rule applied again to the models
        that can, so that ArcFace on CoreML being absent on Linux leaves the best eligible model
        there rather than none.
        """
        runnable = set(runnable)
        if self.first_active in runnable:
            return self.first_active
        chosen = first_active_model([c for c in self.contenders if _key(c.model) in runnable]).model
        return None if chosen is None else _key(chosen)

    @classmethod
    def from_results(cls, results: Results | None) -> "Evaluation":
        if results is None or results.identification is None:
            return cls({}, None)
        measured = {model.model: model for model in results.identification.models}
        models = {
            _key(threshold.model): Evaluated(
                threshold.threshold,
                measured[threshold.model].crop,
                measured[threshold.model].ms_per_face,
            )
            for threshold in results.thresholds
            if threshold.rule in LIVE_RULES
        }
        first = results.first_active_model
        return cls(
            models,
            None if first is None or first.model is None else _key(first.model),
            ()
            if first is None
            else tuple(
                Contender(
                    model=judged.model,
                    lfw_gap_points=judged.lfw_gap_points,
                    test_tpir=judged.test_tpir,
                    test_fpir=judged.test_fpir,
                    ms_per_face=judged.ms_per_face,
                )
                for judged in first.eligibility
                if _key(judged.model) in models
            ),
        )


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A recognition model whose weights are missing, under the key its pinned weights give."""

    key: ModelKey
    dimension: int


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    identity: RecognitionModelId
    model: RecognitionModel | None
    """None when the weights are missing."""
    evaluated: Evaluated | None
    """None when evaluation froze no threshold for this exact key."""

    @property
    def key(self) -> ModelKey:
        return _key(self.identity)

    @property
    def name(self) -> str:
        return model_name(self.identity)

    @property
    def crop(self) -> Crop:
        return "five-point" if self.evaluated is None else self.evaluated.crop

    @property
    def can_be_active(self) -> bool:
        return self.model is not None and self.evaluated is not None


def register(
    models: Sequence[RecognitionModel | Unavailable], evaluation: Evaluation
) -> list[RegisteredModel]:
    """Each model with what evaluation measured for its exact key."""
    return [
        RegisteredModel(
            _identity(model.key, model.dimension),
            None if isinstance(model, Unavailable) else model,
            evaluation.models.get(model.key),
        )
        for model in models
    ]


class ModelRegistry:
    def __init__(self, models: Sequence[RegisteredModel], active: ModelKey | None) -> None:
        if active is not None and not any(m.key == active and m.can_be_active for m in models):
            raise ValueError(f"{active.id} cannot be active: it is unavailable or not evaluated")
        self._models = tuple(models)
        self._active = active

    @property
    def models(self) -> tuple[RegisteredModel, ...]:
        return self._models

    @property
    def active(self) -> RegisteredModel | None:
        return next((m for m in self._models if m.key == self._active), None)

    def loaded(self) -> list[tuple[RegisteredModel, RecognitionModel]]:
        """Every model whose weights are present, with the loaded network."""
        return [(m, m.model) for m in self._models if m.model is not None]

    def state(self, model: RegisteredModel) -> ModelState:
        if model.model is None:
            return "unavailable"
        if model.evaluated is None:
            return "not_evaluated"
        return "active" if model.key == self._active else "available"


def _key(identity: RecognitionModelId) -> ModelKey:
    return ModelKey(identity.network, identity.weights_sha256, identity.provider)


def _identity(key: ModelKey, dimension: int) -> RecognitionModelId:
    network: Network | Literal["fake"] = key.network
    if network == "fake":
        raise ValueError("register a fake recognition model under the network it stands in for")
    return RecognitionModelId(
        network=network,
        provider=key.provider,
        weights_sha256=key.weights_sha256,
        dimension=dimension,
    )
