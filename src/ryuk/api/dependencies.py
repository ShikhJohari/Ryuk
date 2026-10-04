"""What routes take from the running service."""

from typing import Annotated

from fastapi import Depends, Request

from ryuk.api.evaluation_models import EvaluationReport
from ryuk.api.monitor import LiveMonitor
from ryuk.api.problems import ProblemError
from ryuk.watchlist.service import Watchlist


def get_watchlist(request: Request) -> Watchlist:
    watchlist: Watchlist | None = request.app.state.watchlist
    if watchlist is None:
        raise ProblemError(
            status=503, code="watchlist_unavailable", detail="The watchlist is not running."
        )
    return watchlist


def get_evaluation(request: Request) -> EvaluationReport:
    evaluation: EvaluationReport | None = request.app.state.evaluation
    if evaluation is None:
        raise ProblemError(
            status=503,
            code="evaluation_unavailable",
            detail="The service was started without the evaluation results.",
        )
    return evaluation


def get_live_monitor(request: Request) -> LiveMonitor:
    live_monitor: LiveMonitor = request.app.state.monitor
    return live_monitor


WatchlistDep = Annotated[Watchlist, Depends(get_watchlist)]
EvaluationDep = Annotated[EvaluationReport, Depends(get_evaluation)]
LiveMonitorDep = Annotated[LiveMonitor, Depends(get_live_monitor)]
