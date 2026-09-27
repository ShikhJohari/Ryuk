"""Persons of interest and their enrolled photos (#12, #29)."""

import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Form, Response, UploadFile, status

from ryuk.api.contract import problem_response_doc
from ryuk.api.dependencies import WatchlistDep
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
    name: str


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
def update_person(
    watchlist: WatchlistDep, person_id: str, changes: PersonOfInterestChanges
) -> PersonOfInterest:
    return _person(watchlist.rename(person_id, changes.name))


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
    # One byte over the limit is enough to know the photo is too large.
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
