"""The sightings history: newest first a page at a time, one sighting, and its face crop
(#12, #31)."""

from typing import Annotated

from fastapi import APIRouter, Query, Response

from ryuk.api.dependencies import WatchlistDep
from ryuk.api.schema import ApiModel
from ryuk.api.sighting_models import Sighting, SightingSummary, sighting, sighting_summary
from ryuk.watchlist.sightings import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/sightings", tags=["sightings"])


class SightingPage(ApiModel):
    items: list[SightingSummary]
    next_cursor: str | None
    """Pass as `cursor` for the next page; None on the last page."""


@router.get("")
def list_sightings(
    watchlist: WatchlistDep,
    person_id: Annotated[str | None, Query(alias="personId")] = None,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
) -> SightingPage:
    """Sightings newest first, of one person of interest if `personId` is given. `cursor` is
    the previous page's `nextCursor`; any other value is `422 invalid_cursor`."""
    found = watchlist.sightings(person_id, cursor, limit)
    return SightingPage(
        items=[sighting_summary(item) for item in found.items], next_cursor=found.next_cursor
    )


@router.get("/{sighting_id}")
def get_sighting(watchlist: WatchlistDep, sighting_id: str) -> Sighting:
    return sighting(watchlist.sighting(sighting_id))


@router.get(
    "/{sighting_id}/crop",
    response_class=Response,
    responses={
        200: {
            "description": "The face of the sighting's best match, never the whole frame",
            "content": {"image/jpeg": {}},
        }
    },
)
def get_sighting_crop(watchlist: WatchlistDep, sighting_id: str) -> Response:
    crop = watchlist.sighting_crop(sighting_id)
    # Face crops are never kept in the browser's cache, where a purge could not reach them.
    return Response(crop, media_type="image/jpeg", headers={"cache-control": "no-store"})
