"""The evaluation page's numbers (#32): the committed dataset summary and results, read once at
startup."""

from fastapi import APIRouter

from ryuk.api.dependencies import EvaluationDep
from ryuk.api.evaluation_models import EvaluationReport

router = APIRouter(tags=["evaluation"])


@router.get("/evaluation")
def get_evaluation(evaluation: EvaluationDep) -> EvaluationReport:
    """The datasets, LFW verification, CelebA identification, the first active model, learning,
    the bias breakdown and the live operating points. `503 evaluation_unavailable` when the
    service started without them."""
    return evaluation
