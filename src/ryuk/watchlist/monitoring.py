"""The live monitor's sightings written down: each change the tracker decides becomes a write to
the sightings store and an announcement for the live monitor (#16, #31).

Every live monitor connection has its own `MonitoringSession`, and with it its own tracker, so
ending one connection ends only the sightings it opened, never those of the connection that took
over from it.
"""

import io
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from ryuk.detector import Image
from ryuk.watchlist import sightings
from ryuk.watchlist.live import Recognition
from ryuk.watchlist.sightings import BestMatch, NewSighting, SightingPerson, SightingSummary
from ryuk.watchlist.tracker import Ended, LiveSighting, Opened, SightingChange, Updated

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
