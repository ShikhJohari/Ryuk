"""The sighting models the REST API and the live monitor's WebSocket share (#12).

A module of their own, so that the monitor's messages can carry a `SightingSummary` without the
monitor importing the sightings routes, which depend on it.
"""

import datetime

from ryuk.api.schema import ApiModel
from ryuk.watchlist import sightings
from ryuk.watchlist.tables import PersonStatus


class SightingPerson(ApiModel):
    """A sighting's person of interest, or its runner-up, as they are now."""

    id: str
    name: str
    status: PersonStatus


class SightingSummary(ApiModel):
    """A sighting as the history lists it and the live monitor announces it. Never carries the
    runner-up."""

    id: str
    person: SightingPerson
    model_key: str
    threshold: float
    """The active model's threshold when the sighting opened."""
    started_at: datetime.datetime
    last_seen_at: datetime.datetime
    ended_at: datetime.datetime | None
    """None while the sighting is open; `lastSeenAt` once it ends."""
    best_score: float


class RunnerUp(ApiModel):
    person: SightingPerson | None
    """None once the runner-up was purged; their score is kept."""
    score: float


class Sighting(SightingSummary):
    """One sighting with the runner-up at its best match."""

    runner_up: RunnerUp | None
    """None when nobody else was on the watchlist to rank second."""


def sighting_summary(summary: sightings.SightingSummary) -> SightingSummary:
    return SightingSummary(
        id=summary.id,
        person=_person(summary.person),
        model_key=summary.model_key,
        threshold=summary.threshold,
        started_at=summary.started_at,
        last_seen_at=summary.last_seen_at,
        ended_at=summary.ended_at,
        best_score=summary.best_score,
    )


def sighting(found: sightings.Sighting) -> Sighting:
    runner_up = found.runner_up
    return Sighting(
        **sighting_summary(found.summary).model_dump(),
        runner_up=None
        if runner_up is None
        else RunnerUp(
            person=None if runner_up.person is None else _person(runner_up.person),
            score=runner_up.score,
        ),
    )


def _person(person: sightings.SightingPerson) -> SightingPerson:
    return SightingPerson(id=person.id, name=person.name, status=person.status)
