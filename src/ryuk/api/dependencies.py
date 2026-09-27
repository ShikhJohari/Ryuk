"""What routes take from the running service."""

from typing import Annotated

from fastapi import Depends, Request

from ryuk.api.problems import ProblemError
from ryuk.watchlist.service import Watchlist


def get_watchlist(request: Request) -> Watchlist:
    watchlist: Watchlist | None = request.app.state.watchlist
    if watchlist is None:
        raise ProblemError(
            status=503, code="watchlist_unavailable", detail="The watchlist is not running."
        )
    return watchlist


WatchlistDep = Annotated[Watchlist, Depends(get_watchlist)]
