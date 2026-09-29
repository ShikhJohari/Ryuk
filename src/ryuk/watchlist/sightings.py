"""Sightings in the app database: the writes the live monitor's tracker makes, and the reads
behind the sightings history (#12, #16, #31).

The tracker holds each open sighting in memory and writes it through `open_sighting`,
`update_sighting` and `end_sighting`, each returning the summary it then announces on the live
monitor. Every function takes the caller's session, so a write joins the caller's transaction.
"""

import base64
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast

from sqlalchemy import CursorResult, and_, or_, select, update
from sqlalchemy.orm import Session, aliased

from ryuk.watchlist.errors import WatchlistError, not_found
from ryuk.watchlist.tables import PersonOfInterestRow, PersonStatus, SightingRow

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE: Final = 50
MAX_PAGE_SIZE: Final = 100


@dataclass(frozen=True, slots=True)
class SightingPerson:
    """The person of interest a sighting is of, or its runner-up, as they are now."""

    id: str
    name: str
    status: PersonStatus


@dataclass(frozen=True, slots=True)
class SightingSummary:
    """A sighting as the history lists it and the live monitor announces it: never the
    runner-up, which is only shown on the sighting's own page."""

    id: str
    person: SightingPerson
    model_key: str
    threshold: float
    started_at: datetime.datetime
    last_seen_at: datetime.datetime
    ended_at: datetime.datetime | None
    """None while the sighting is open; `last_seen_at` once it ends."""
    best_score: float


@dataclass(frozen=True, slots=True)
class RunnerUp:
    person: SightingPerson | None
    """None once the runner-up was purged; their score is kept."""
    score: float


@dataclass(frozen=True, slots=True)
class Sighting:
    summary: SightingSummary
    runner_up: RunnerUp | None
    """The second-ranked candidate at the best match; None when nobody else was on the
    watchlist to rank."""


@dataclass(frozen=True, slots=True)
class SightingPage:
    items: tuple[SightingSummary, ...]
    next_cursor: str | None
    """Where the next page starts; None on the last page."""


@dataclass(frozen=True, slots=True)
class BestMatch:
    """A sighting's best match so far, and the runner-up from the same frame."""

    score: float
    crop: bytes
    """The matched face as JPEG, cut with a margin and clipped to the frame."""
    runner_up_person_id: str | None
    runner_up_score: float | None

    def __post_init__(self) -> None:
        if self.runner_up_person_id is not None and self.runner_up_score is None:
            raise ValueError("a runner-up needs its score")


@dataclass(frozen=True, slots=True)
class NewSighting:
    """A sighting confirmation just opened."""

    id: str
    """The live tracker's own ID for it, so the ID it announces is the stored one."""
    person_id: str
    model_key: str
    threshold: float
    """The active model's threshold at confirmation."""
    started_at: datetime.datetime
    """The first matched frame in the confirming window."""
    last_seen_at: datetime.datetime
    best: BestMatch


def open_sighting(session: Session, new: NewSighting) -> SightingSummary:
    """Store a sighting that confirmation just opened; not found if the person is gone."""
    person = session.get(PersonOfInterestRow, new.person_id)
    if person is None:
        raise not_found("person of interest")
    row = SightingRow(
        id=new.id,
        person_id=new.person_id,
        model_key=new.model_key,
        threshold=new.threshold,
        started_at=new.started_at,
        last_seen_at=new.last_seen_at,
        ended_at=None,
    )
    _set_best(row, new.best)
    session.add(row)
    session.flush()
    return _summary(row, person)


def update_sighting(
    session: Session,
    sighting_id: str,
    *,
    last_seen_at: datetime.datetime,
    best: BestMatch | None = None,
) -> SightingSummary | None:
    """Write an open sighting's progress: when the person was last seen and, if it changed, the
    best match, whose crop is otherwise left as stored. None when the sighting is gone, as after
    a purge; a ValueError if it already ended."""
    return _write(session, sighting_id, last_seen_at, best, end=False)


def end_sighting(
    session: Session,
    sighting_id: str,
    *,
    last_seen_at: datetime.datetime,
    best: BestMatch | None = None,
) -> SightingSummary | None:
    """Write a sighting's last progress and end it when the person was last seen, the span they
    were actually seen. None when the sighting is gone, as after a purge; a ValueError if it
    already ended."""
    return _write(session, sighting_id, last_seen_at, best, end=True)


def _write(
    session: Session,
    sighting_id: str,
    last_seen_at: datetime.datetime,
    best: BestMatch | None,
    *,
    end: bool,
) -> SightingSummary | None:
    row = session.get(SightingRow, sighting_id)
    if row is None:
        return None
    if row.ended_at is not None:
        raise ValueError(f"sighting {sighting_id} already ended")
    row.last_seen_at = last_seen_at
    if best is not None:
        _set_best(row, best)
    if end:
        row.ended_at = last_seen_at
    session.flush()
    return _summary(row, _person_of(session, row))


def end_open_sightings(session: Session) -> int:
    """End every sighting still open when the person was last seen in it, and return how many.

    At startup no live monitor runs yet, so a sighting still open was left so by a run that
    stopped without ending it.
    """
    result = session.execute(
        update(SightingRow)
        .where(SightingRow.ended_at.is_(None))
        .values(ended_at=SightingRow.last_seen_at)
    )
    ended = cast(CursorResult[Any], result).rowcount
    if ended:
        logger.info("Ended %d sightings the last run left open", ended)
    return ended


def sighting_page(
    session: Session, *, person_id: str | None, cursor: str | None, limit: int
) -> SightingPage:
    """Up to `limit` sightings, newest first, after `cursor` if given and of `person_id` if
    given; 422 `invalid_cursor` for a cursor this service did not give out."""
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"a page size is 1 to {MAX_PAGE_SIZE}, not {limit}")
    query = (
        select(SightingRow, PersonOfInterestRow)
        .join(PersonOfInterestRow, SightingRow.person_id == PersonOfInterestRow.id)
        .order_by(SightingRow.started_at.desc(), SightingRow.id.desc())
        .limit(limit + 1)
    )
    if person_id is not None:
        query = query.where(SightingRow.person_id == person_id)
    if cursor is not None:
        started_at, last_id = _decode_cursor(cursor)
        query = query.where(
            or_(
                SightingRow.started_at < started_at,
                and_(SightingRow.started_at == started_at, SightingRow.id < last_id),
            )
        )
    rows = session.execute(query).all()
    items = tuple(_summary(sighting, person) for sighting, person in rows[:limit])
    more = len(rows) > limit
    return SightingPage(items, _encode_cursor(items[-1]) if more else None)


def get_sighting(session: Session, sighting_id: str) -> Sighting:
    runner_up = aliased(PersonOfInterestRow)
    found = session.execute(
        select(SightingRow, PersonOfInterestRow, runner_up)
        .join(PersonOfInterestRow, SightingRow.person_id == PersonOfInterestRow.id)
        .outerjoin(runner_up, SightingRow.runner_up_person_id == runner_up.id)
        .where(SightingRow.id == sighting_id)
    ).one_or_none()
    if found is None:
        raise not_found("sighting")
    row, person, runner_up_person = found
    return Sighting(
        _summary(row, person),
        None
        if row.runner_up_score is None
        else RunnerUp(
            None if runner_up_person is None else sighting_person(runner_up_person),
            row.runner_up_score,
        ),
    )


def sighting_crop(session: Session, sighting_id: str) -> bytes:
    """The JPEG of a sighting's best match."""
    crop = session.scalar(select(SightingRow.best_crop).where(SightingRow.id == sighting_id))
    if crop is None:
        raise not_found("sighting")
    return crop


def _set_best(row: SightingRow, best: BestMatch) -> None:
    row.best_score = best.score
    row.best_crop = best.crop
    row.runner_up_person_id = best.runner_up_person_id
    row.runner_up_score = best.runner_up_score


def _person_of(session: Session, row: SightingRow) -> PersonOfInterestRow:
    # The foreign key cascades, so a sighting's person is always there.
    return session.get_one(PersonOfInterestRow, row.person_id)


class Summarised(Protocol):
    """What a summary is made from: a stored sighting, or the live tracker's own."""

    @property
    def id(self) -> str: ...
    @property
    def model_key(self) -> str: ...
    @property
    def threshold(self) -> float: ...
    @property
    def started_at(self) -> datetime.datetime: ...
    @property
    def last_seen_at(self) -> datetime.datetime: ...
    @property
    def ended_at(self) -> datetime.datetime | None: ...
    @property
    def best_score(self) -> float: ...


def summarise(sighting: Summarised, person: SightingPerson) -> SightingSummary:
    """`sighting` as the history lists it and the live monitor announces it, of `person`."""
    return SightingSummary(
        id=sighting.id,
        person=person,
        model_key=sighting.model_key,
        threshold=sighting.threshold,
        started_at=sighting.started_at,
        last_seen_at=sighting.last_seen_at,
        ended_at=sighting.ended_at,
        best_score=sighting.best_score,
    )


def _summary(row: SightingRow, person: PersonOfInterestRow) -> SightingSummary:
    return summarise(row, sighting_person(person))


def sighting_person(row: PersonOfInterestRow) -> SightingPerson:
    """A person of interest as a sighting shows them, as they are now."""
    # The table's check constraint holds the status to these.
    return SightingPerson(row.id, row.name, cast(PersonStatus, row.status))


_CURSOR_SEPARATOR: Final = "|"


def _encode_cursor(last: SightingSummary) -> str:
    """The page's last sighting, opaque to the client: base64url of its start and ID."""
    raw = f"{last.started_at.isoformat()}{_CURSOR_SEPARATOR}{last.id}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime.datetime, str]:
    try:
        padded = cursor.encode("ascii") + b"=" * (-len(cursor) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True).decode()
        started, separator, last_id = raw.partition(_CURSOR_SEPARATOR)
        started_at = datetime.datetime.fromisoformat(started)
    except ValueError:  # bad base64, bytes that are not UTF-8, or no timestamp
        started_at, separator, last_id = None, "", ""
    if started_at is None or started_at.tzinfo is None or not separator or not last_id:
        raise WatchlistError(
            422, "invalid_cursor", "That cursor is not one the sightings list gave out."
        )
    return started_at, last_id
