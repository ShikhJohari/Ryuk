"""The live monitor's sightings written down: each change the tracker decides becomes a write to
the sightings store and an announcement for the live monitor (#16, #31).

Every live monitor connection has its own `MonitoringSession`, and with it its own tracker, so
ending one connection ends only the sightings it opened, never those of the connection that took
over from it. `MonitoringSessions` keeps the trackers and the store in step: every change is
all-or-nothing between the two.
"""

import datetime
import io
import itertools
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from PIL import Image as PILImage
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from ryuk.detector import Image
from ryuk.watchlist import sightings
from ryuk.watchlist.live import Match, Recognition
from ryuk.watchlist.sightings import BestMatch, NewSighting, SightingPerson, SightingSummary
from ryuk.watchlist.tracker import (
    Ended,
    LiveSighting,
    Opened,
    SightingChange,
    SightingTracker,
    Updated,
)

logger = logging.getLogger(__name__)

CROP_QUALITY: Final = 95
"""The JPEG quality a sighting's crop is stored at: the same policy as an enrolled photo's
(`ryuk.watchlist.photos`, #47 Q9), since both are the lasting record of a face."""

type SightingAnnouncementType = Literal["sighting_opened", "sighting_updated", "sighting_ended"]


@dataclass(frozen=True, slots=True)
class SightingAnnouncement:
    """A sighting write, as the live monitor announces it: never with the runner-up."""

    type: SightingAnnouncementType
    sighting: SightingSummary


@dataclass(frozen=True, eq=False, slots=True)
class MonitoringSession:
    """One live monitor connection's sightings, from `Watchlist.begin_monitoring` until
    `Watchlist.end_monitoring`. Compared by identity."""

    number: int


@dataclass(frozen=True, slots=True)
class LiveFrame:
    """A frame's faces, with the sighting writes it caused, in order, and the open sighting of
    each person matched in it."""

    recognition: Recognition
    sightings: tuple[SightingAnnouncement, ...] = ()
    sighting_ids: Mapping[str, str] = field(default_factory=dict)
    """By person ID; a person matched but not yet confirmed has none."""


class MonitoringSessions:
    """Every live monitor connection's sighting tracker, kept in step with the sightings store.

    A change is all-or-nothing: each tracker it touches is snapshotted first and restored if the
    transaction that writes the change fails, so a sighting is never ended in memory but open in
    its row, or announced but never stored, and the next frame or tick simply tries again. Not
    thread-safe: the watchlist calls it under its lock.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._trackers: dict[MonitoringSession, SightingTracker] = {}
        """Usually one; two only while a connection that was taken over is still tearing down."""
        self._numbers = itertools.count(1)

    def begin(self) -> MonitoringSession:
        monitor_session = MonitoringSession(next(self._numbers))
        self._trackers[monitor_session] = SightingTracker()
        return monitor_session

    def end(self, monitor_session: MonitoringSession) -> tuple[SightingAnnouncement, ...]:
        """End every sighting `monitor_session` has open, and forget it once they are written.
        If the write fails, it is kept, its sightings open, for a later end to write."""
        tracker = self._trackers.get(monitor_session)
        if tracker is None:
            return ()
        announcements = self._write_now(tracker, SightingTracker.end_all)
        del self._trackers[monitor_session]
        return announcements

    def observe(
        self,
        monitor_session: MonitoringSession,
        frame: Image,
        recognition: Recognition,
        now: datetime.datetime,
    ) -> LiveFrame:
        """`recognition` of `frame` tracked in `monitor_session`, with what it wrote."""
        tracker = self._trackers.get(monitor_session)
        if tracker is None:  # the session already ended: nothing is tracked for it
            return LiveFrame(recognition)
        announcements = self._write_now(
            tracker, lambda tracking: tracking.observe(frame, recognition, now)
        )
        open_ids = {
            face.candidate.person_id: sighting_id
            for face in recognition.faces
            if isinstance(face, Match)
            and (sighting_id := tracker.sighting_id(face.candidate.person_id)) is not None
        }
        return LiveFrame(recognition, announcements, open_ids)

    def tick(
        self, monitor_session: MonitoringSession, now: datetime.datetime
    ) -> tuple[SightingAnnouncement, ...]:
        tracker = self._trackers.get(monitor_session)
        if tracker is None:
            return ()
        return self._write_now(tracker, lambda tracking: tracking.tick(now))

    @contextmanager
    def all_or_nothing(self) -> Iterator[None]:
        """Undo every tracker change made in the block if it raises, as when the transaction
        the block wraps fails to commit. `end_person`, `end_all` and `purge` are made inside
        one, around the transaction they write in."""
        saved = [(tracker, tracker.snapshot()) for tracker in self._trackers.values()]
        try:
            yield
        except BaseException:
            for tracker, snapshot in saved:
                tracker.restore(snapshot)
            raise

    def end_person(self, session: Session, person_id: str) -> tuple[SightingAnnouncement, ...]:
        """End `person_id`'s open sightings, on their removal, written in `session`."""
        return tuple(write(session, self._ended(lambda tracker: tracker.end_person(person_id))))

    def end_all(self, session: Session) -> tuple[SightingAnnouncement, ...]:
        """End every open sighting, on a switch of the active model, written in `session`."""
        return tuple(write(session, self._ended(SightingTracker.end_all)))

    def purge(self, person: SightingPerson) -> tuple[SightingAnnouncement, ...]:
        """End `person`'s open sightings, whose rows their purge erased: announced with their
        last state, not written. They are forgotten as runner-up too, their scores kept."""
        ended = self._ended(lambda tracker: tracker.end_person(person.id))
        for tracker in self._trackers.values():
            tracker.clear_runner_up(person.id)
        return tuple(unwritten_end(change, person) for change in ended)

    def _ended(self, end: Callable[[SightingTracker], list[Ended]]) -> list[Ended]:
        return [change for tracker in self._trackers.values() for change in end(tracker)]

    def _write_now(
        self, tracker: SightingTracker, act: Callable[[SightingTracker], Sequence[SightingChange]]
    ) -> tuple[SightingAnnouncement, ...]:
        """`act` on `tracker` and write what it decides in a transaction of its own, or, if the
        write fails, neither."""
        snapshot = tracker.snapshot()
        try:
            changes = act(tracker)
            if not changes:  # most frames and ticks
                return ()
            with Session(self._engine) as session, session.begin():
                return tuple(write(session, changes))
        except BaseException:
            tracker.restore(snapshot)
            raise


def write(session: Session, changes: Iterable[SightingChange]) -> list[SightingAnnouncement]:
    """Write the tracker's changes in `session`'s transaction, each as its announcement.

    A crop is encoded only when a write carries a new best match. An update or end whose row is
    gone writes and announces nothing: the one way that happens is a purge, which announces
    the end itself (`unwritten_end`).
    """
    announcements: list[SightingAnnouncement] = []
    for change in changes:
        match change:
            case Opened(sighting=live):
                new = NewSighting(
                    person_id=live.person_id,
                    model_key=live.model.id,
                    threshold=live.threshold,
                    started_at=live.started_at,
                    last_seen_at=live.last_seen_at,
                    best=_best(live),
                    id=live.id,
                )
                announcements.append(
                    SightingAnnouncement("sighting_opened", sightings.open_sighting(session, new))
                )
            case Updated(sighting=live, new_best=new_best):
                summary = sightings.update_sighting(
                    session,
                    live.id,
                    last_seen_at=live.last_seen_at,
                    best=_best(live) if new_best else None,
                )
                announcements += _announced("sighting_updated", live, summary)
            case Ended(sighting=live, new_best=new_best):
                summary = sightings.end_sighting(
                    session,
                    live.id,
                    last_seen_at=live.last_seen_at,
                    best=_best(live) if new_best else None,
                )
                announcements += _announced("sighting_ended", live, summary)
    return announcements


def unwritten_end(change: Ended, person: SightingPerson) -> SightingAnnouncement:
    """The end of a sighting a purge already erased: announced with its last state, not
    written. `person` is the purged person as they were."""
    return SightingAnnouncement("sighting_ended", sightings.summarise(change.sighting, person))


def encode_crop(crop: Image) -> bytes:
    """A BGR crop as the JPEG a sighting keeps: pixels only, no metadata."""
    rgb = PILImage.fromarray(np.ascontiguousarray(crop[:, :, ::-1]))
    encoded = io.BytesIO()
    rgb.save(encoded, format="JPEG", quality=CROP_QUALITY)
    return encoded.getvalue()


def _best(live: LiveSighting) -> BestMatch:
    return BestMatch(
        live.best_score,
        encode_crop(live.best_crop),
        live.runner_up_person_id,
        live.runner_up_score,
    )


def _announced(
    announcement: SightingAnnouncementType, live: LiveSighting, summary: SightingSummary | None
) -> list[SightingAnnouncement]:
    if summary is None:
        logger.warning(
            "Sighting %s is no longer stored; its %s was not written", live.id, announcement
        )
        return []
    return [SightingAnnouncement(announcement, summary)]
