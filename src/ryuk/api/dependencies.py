"""What routes take from the running service."""

from typing import Annotated

from fastapi import Depends, Request

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


def get_live_monitor(request: Request) -> LiveMonitor:
    live_monitor: LiveMonitor = request.app.state.monitor
    return live_monitor


WatchlistDep = Annotated[Watchlist, Depends(get_watchlist)]
LiveMonitorDep = Annotated[LiveMonitor, Depends(get_live_monitor)]
