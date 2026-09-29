"""Persons of interest and their enrolled photos; removal, restore and purge (#12, #29, #31)."""

import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Form, Response, UploadFile, status
from pydantic import ConfigDict, field_validator
from pydantic.json_schema import SkipJsonSchema
from starlette.concurrency import run_in_threadpool

from ryuk.api.contract import problem_response_doc
from ryuk.api.dependencies import LiveMonitorDep, WatchlistDep
from ryuk.api.problems import WarningsProblem
from ryuk.api.schema import ApiModel
from ryuk.watchlist import service
from ryuk.watchlist.errors import WarningCode
from ryuk.watchlist.photos import MAX_PHOTO_BYTES, MEDIA_TYPES
from ryuk.watchlist.service import StatusFilter
from ryuk.watchlist.tables import PersonStatus

router = APIRouter(prefix="/persons", tags=["persons"])

_WARNINGS_RESPONSE: dict[int | str, dict[str, Any]] = {
    409: problem_response_doc(WarningsProblem, "Warnings to acknowledge")
}


class EnrolledPhoto(ApiModel):
    id: str
    media_type: str
    width: int
    height: int
    created_at: datetime.datetime


class PersonOfInterestSummary(ApiModel):
    id: str
    name: str
    status: PersonStatus
    photo_count: int
    cover_photo_id: str
    """The first enrolled photo, for a thumbnail."""
    created_at: datetime.datetime
    status_changed_at: datetime.datetime


class PersonOfInterest(ApiModel):
    id: str
    name: str
    status: PersonStatus
    created_at: datetime.datetime
    status_changed_at: datetime.datetime
    photos: list[EnrolledPhoto]


_Acknowledged = Annotated[
    list[WarningCode], Form(alias="acknowledgedWarnings", default_factory=list)
]
"""Warning codes the operator confirmed, one form field each."""


class PersonOfInterestChanges(ApiModel):
    """What PATCH changes: the name, the status, both or neither. A field left out is left as it
    is; null, or any other field such as `statusChangedAt`, is refused with
    `422 invalid_request` rather than ignored."""

    # Merged with ApiModel's config: camelCase aliases and the rest still apply.
    model_config = ConfigDict(extra="forbid")

    # Optional but never null: None stands for "left out", and the contract says so.
    name: str | SkipJsonSchema[None] = None
    status: PersonStatus | SkipJsonSchema[None] = None
    """`removed` takes the person off the watchlist, keeping their photos and sightings;
    `on_watchlist` restores them."""

    @field_validator("name", "status", mode="before")
    @classmethod
    def _not_null(cls, value: object) -> object:
        # Only a value sent is validated, so None here is an explicit null.
        if value is None:
            raise ValueError("leave the field out to keep it; it cannot be null")
        return value


@router.get("")
def list_persons(
    watchlist: WatchlistDep, status: StatusFilter = "on_watchlist"
) -> list[PersonOfInterestSummary]:
    return [
        PersonOfInterestSummary(
            id=person.id,
            name=person.name,
            status=person.status,
            photo_count=len(person.photos),
            cover_photo_id=person.photos[0].id,
            created_at=person.created_at,
            status_changed_at=person.status_changed_at,
        )
        for person in watchlist.persons(status)
    ]


@router.post("", status_code=status.HTTP_201_CREATED, responses=_WARNINGS_RESPONSE)
def create_person(
    watchlist: WatchlistDep,
    name: Annotated[str, Form()],
    photo: UploadFile,
    acknowledged_warnings: _Acknowledged,
) -> PersonOfInterest:
    return _person(watchlist.enroll(name, _read(photo), acknowledged_warnings))


@router.get("/{person_id}")
def get_person(watchlist: WatchlistDep, person_id: str) -> PersonOfInterest:
    return _person(watchlist.person(person_id))


@router.patch("/{person_id}")
async def update_person(
    watchlist: WatchlistDep,
    live_monitor: LiveMonitorDep,
    person_id: str,
    changes: PersonOfInterestChanges,
) -> PersonOfInterest:
    """Rename, remove or restore a person of interest; an empty body changes nothing. Removal
    takes them off the watchlist the live monitor matches against from its next frame."""
    # Removal also ends the person's open sighting, announced to the live monitor (#16).
    change = await run_in_threadpool(
        watchlist.update_person, person_id, name=changes.name, status=changes.status
    )
    await live_monitor.announce_sightings(change.sightings)
    return _person(change.person)


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def purge_person(
    watchlist: WatchlistDep, live_monitor: LiveMonitorDep, person_id: str
) -> None:
    """Purge a person of interest, on the watchlist or removed: their enrolled photos,
    embeddings and sightings are erased, and they are cleared as runner-up on other sightings.
    Not reversible."""
    # Their open sighting is announced to the live monitor as ended, though not written (#16).
    ended = await run_in_threadpool(watchlist.purge, person_id)
    await live_monitor.announce_sightings(ended)


@router.post(
    "/{person_id}/photos", status_code=status.HTTP_201_CREATED, responses=_WARNINGS_RESPONSE
)
def add_photo(
    watchlist: WatchlistDep,
    person_id: str,
    photo: UploadFile,
    acknowledged_warnings: _Acknowledged,
) -> EnrolledPhoto:
    return _photo(watchlist.add_photo(person_id, _read(photo), acknowledged_warnings))


@router.get(
    "/{person_id}/photos/{photo_id}/image",
    response_class=Response,
    responses={
        200: {
            "description": "The enrolled photo, upright and without metadata",
            "content": {media_type: {} for media_type in MEDIA_TYPES.values()},
        }
    },
)
def get_photo_image(watchlist: WatchlistDep, person_id: str, photo_id: str) -> Response:
    photo = watchlist.photo_image(person_id, photo_id)
    # Face photos are never kept in the browser's cache, where a purge could not reach them.
    return Response(photo.data, media_type=photo.media_type, headers={"cache-control": "no-store"})


@router.delete("/{person_id}/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_photo(watchlist: WatchlistDep, person_id: str, photo_id: str) -> None:
    watchlist.delete_photo(person_id, photo_id)


def _read(upload: UploadFile) -> bytes:
    # PhotoUploadLimitMiddleware already refused a body far over the limit; this holds the photo
    # itself to it. One byte over is enough to know the photo is too large.
    return upload.file.read(MAX_PHOTO_BYTES + 1)


def _person(person: service.PersonOfInterest) -> PersonOfInterest:
    return PersonOfInterest(
        id=person.id,
        name=person.name,
        status=person.status,
        created_at=person.created_at,
        status_changed_at=person.status_changed_at,
        photos=[_photo(photo) for photo in person.photos],
    )


def _photo(photo: service.EnrolledPhoto) -> EnrolledPhoto:
    return EnrolledPhoto(
        id=photo.id,
        media_type=photo.media_type,
        width=photo.width,
        height=photo.height,
        created_at=photo.created_at,
    )
