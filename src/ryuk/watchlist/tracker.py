"""Turning the live monitor's per-frame matches into sightings (#16, #31).

A match is shown live the moment it happens, but only a confirmed one is logged: a person of
interest's sighting opens once they are matched in at least half the frames processed over the
last 500 ms, with a minimum of three, so a single fluke frame never opens one. It keeps the best
match's face crop and that frame's runner-up, and ends 3 s after the person's last match, with
`ended_at` the last time they were seen. Removal, purge, switching the active model and closing
the live monitor end sightings too, through the explicit end operations.

`SightingTracker` is pure: no database, no I/O, no threads, and no clock of its own. Time is
passed in with every call, and the tracker answers with the changes to make, `Opened`,
`Updated` and `Ended`, each carrying the sighting as it should now be written and announced. The
caller owns the lock that serialises calls, the database and the socket.
"""

import datetime
import math
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Final

from ryuk.detector import Box, Image
from ryuk.recognition import ModelKey
from ryuk.watchlist.live import Match, Recognition

CONFIRMATION_WINDOW: Final = datetime.timedelta(milliseconds=500)
"""The frames that confirm a sighting at a frame at time t are those in (t - window, t]."""
MIN_CONFIRMING_FRAMES: Final = 3
"""Matched frames a confirmation needs however few frames the window holds."""
GAP: Final = datetime.timedelta(seconds=3)
"""A sighting ends once this long has passed since its last match."""
WRITE_INTERVAL: Final = datetime.timedelta(seconds=1)
"""The least time between two writes of one open sighting; changes in between are held."""
CROP_MARGIN: Final = 0.25
"""How far the best crop reaches past the detection's box on each side, as a share of the box's
width and height, so the stored face keeps its outline and a little context."""


@dataclass(frozen=True, slots=True)
class LiveSighting:
    """A sighting as the tracker holds it, to be written and announced."""

    id: str
    person_id: str
    model: ModelKey
    threshold: float
    started_at: datetime.datetime
    """The first matched frame in the window that confirmed the sighting."""
    last_seen_at: datetime.datetime
    """The last frame that matched the person."""
    ended_at: datetime.datetime | None
    """`last_seen_at` once ended; None while the sighting is open."""
    best_score: float
    best_crop: Image = field(compare=False, repr=False)
    """The best match's face, cut from its frame with a margin: read-only pixels, BGR, to be
    JPEG-encoded when written. Never the whole frame."""
    runner_up_person_id: str | None
    """The runner-up at the best match; None if there was none or they have since been purged."""
    runner_up_score: float | None
    """The runner-up's score at the best match, kept when the runner-up is purged."""


@dataclass(frozen=True, slots=True)
class Opened:
    """A sighting was confirmed: insert it and announce it before the frame's result."""

    sighting: LiveSighting


@dataclass(frozen=True, slots=True)
class Updated:
    """An open sighting's held changes are due: write them and announce the update."""

    sighting: LiveSighting
    new_best: bool
    """Whether the best match changed since the last write, so the crop needs writing again."""


@dataclass(frozen=True, slots=True)
class Ended:
    """A sighting ended: write its final state, unless its row is already gone, and announce
    it."""

    sighting: LiveSighting
    new_best: bool
    """Whether the best match changed since the last write, so the crop needs writing again."""


type SightingChange = Opened | Updated | Ended


def cut_crop(frame: Image, box: Box, margin: float = CROP_MARGIN) -> Image:
    """The face in `box`, widened by `margin` of its size on each side and clipped to `frame`, as a
    read-only copy, so the frame's buffer is not kept alive and nothing can change the crop."""
    height, width = frame.shape[:2]
    left = max(math.floor(box.x - margin * box.width), 0)
    top = max(math.floor(box.y - margin * box.height), 0)
    right = min(math.ceil(box.x + box.width * (1 + margin)), width)
    bottom = min(math.ceil(box.y + box.height * (1 + margin)), height)
    if right <= left or bottom <= top:
        raise ValueError(f"the box {box} lies outside the {width}x{height} frame")
    crop = frame[top:bottom, left:right].copy()
    crop.flags.writeable = False
    return crop


def new_sighting_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class _Observation:
    """A person's best match in one frame: several faces of one person count once."""

    score: float
    crop: Image = field(compare=False, repr=False)
    runner_up_person_id: str | None
    runner_up_score: float | None


@dataclass(slots=True)
class _Frame:
    """A processed frame in the confirmation window, with the matches of each person who has no
    open sighting."""

    at: datetime.datetime
    matches: dict[str, _Observation]


@dataclass(slots=True)
class _Open:
    """An open sighting, with what it has not written yet."""

    sighting: LiveSighting
    written_at: datetime.datetime
    held: bool = False
    """Whether it has changed since `written_at`."""
    new_best: bool = False
    """Whether its best match has changed since `written_at`."""


@dataclass(frozen=True, slots=True)
class TrackerSnapshot:
    """A tracker's state at one moment, to go back to. Opaque to the caller."""

    window: tuple[_Frame, ...]
    open: tuple[tuple[str, _Open], ...]
    judged_by: tuple[ModelKey, float] | None
    now: datetime.datetime | None


class SightingTracker:
    """The sightings of one live monitor connection under one active model at a time."""

    def __init__(self, new_id: Callable[[], str] = new_sighting_id) -> None:
        self._new_id = new_id
        self._window: deque[_Frame] = deque()
        self._open: dict[str, _Open] = {}
        """By person ID; at most one open sighting per person."""
        self._judged_by: tuple[ModelKey, float] | None = None
        """The model and threshold the window's matches and open sightings were judged by."""
        self._now: datetime.datetime | None = None

    def snapshot(self) -> TrackerSnapshot:
        """The tracker's state now, for `restore` to go back to if the writes it decides next
        fail. Cheap: the window's few frames and the open sightings are copied, while the crops
        they hold are read-only and shared."""
        return TrackerSnapshot(
            window=tuple(_Frame(frame.at, dict(frame.matches)) for frame in self._window),
            open=tuple((person_id, replace(opened)) for person_id, opened in self._open.items()),
            judged_by=self._judged_by,
            now=self._now,
        )

    def restore(self, snapshot: TrackerSnapshot) -> None:
        """Go back to `snapshot`, undoing every change since. It stays usable: copies are
        restored, not the snapshot's own state."""
        self._window = deque(_Frame(frame.at, dict(frame.matches)) for frame in snapshot.window)
        self._open = {person_id: replace(opened) for person_id, opened in snapshot.open}
        self._judged_by = snapshot.judged_by
        self._now = snapshot.now

    def sighting_id(self, person_id: str) -> str | None:
        """The ID of `person_id`'s open sighting, if they have one."""
        opened = self._open.get(person_id)
        return None if opened is None else opened.sighting.id

    def observe(
        self, frame: Image, recognition: Recognition, now: datetime.datetime
    ) -> list[SightingChange]:
        """Take a processed frame, matched or not, and return what changed, in order: sightings
        that ended with the gap, then sightings the frame confirmed, then held changes due."""
        now = self._advance(now)
        changes: list[SightingChange] = []
        judged_by = (recognition.model, recognition.threshold)
        if self._judged_by != judged_by:
            # A sighting is under one active model; matches under another never confirm one.
            changes += self.end_all()
            self._judged_by = judged_by
        changes += self._end_gaps(now)
        waiting = _Frame(now, {})
        for person_id, face in _best_per_person(recognition).items():
            opened = self._open.get(person_id)
            if opened is None:
                waiting.matches[person_id] = _observe(frame, face)
            else:
                _extend(opened, frame, face, now)
        self._window.append(waiting)
        self._forget_before(now)
        changes += self._confirm(recognition, now)
        changes += self._write_held(now)
        return changes

    def tick(self, now: datetime.datetime) -> list[SightingChange]:
        """Apply the gap and the write interval with no frame, as when frames stop because the
        client's tab is hidden."""
        now = self._advance(now)
        changes = self._end_gaps(now)
        self._forget_before(now)
        return changes + self._write_held(now)

    def end_person(self, person_id: str) -> list[Ended]:
        """End `person_id`'s open sighting, on their removal or purge, and forget their matches
        still awaiting confirmation."""
        for waiting in self._window:
            waiting.matches.pop(person_id, None)
        opened = self._open.pop(person_id, None)
        return [] if opened is None else [_ended(opened)]

    def end_all(self) -> list[Ended]:
        """End every open sighting, when the live monitor closes or the active model changes, and
        forget every match awaiting confirmation."""
        changes = [_ended(opened) for opened in self._open.values()]
        self._open.clear()
        self._window.clear()
        self._judged_by = None
        return changes

    def clear_runner_up(self, person_id: str) -> None:
        """Forget `person_id` as anyone's runner-up, keeping the score, after their purge.

        Held, not written: the purge has already cleared the stored runner-up, and the runner-up
        is never announced.
        """
        for opened in self._open.values():
            if opened.sighting.runner_up_person_id == person_id:
                opened.sighting = replace(opened.sighting, runner_up_person_id=None)
        for waiting in self._window:
            for other, observation in waiting.matches.items():
                if observation.runner_up_person_id == person_id:
                    waiting.matches[other] = replace(observation, runner_up_person_id=None)

    def _advance(self, now: datetime.datetime) -> datetime.datetime:
        """`now`, never earlier than a time already seen, so a wall clock stepping back cannot
        end a sighting before it started."""
        if self._now is not None and now < self._now:
            now = self._now
        self._now = now
        return now

    def _end_gaps(self, now: datetime.datetime) -> list[SightingChange]:
        gone = [p for p, opened in self._open.items() if now - opened.sighting.last_seen_at >= GAP]
        return [_ended(self._open.pop(person_id)) for person_id in gone]

    def _forget_before(self, now: datetime.datetime) -> None:
        while self._window and self._window[0].at <= now - CONFIRMATION_WINDOW:
            self._window.popleft()

    def _confirm(self, recognition: Recognition, now: datetime.datetime) -> list[SightingChange]:
        """Open a sighting for everyone the window at `now` confirms."""
        seen: dict[str, list[tuple[datetime.datetime, _Observation]]] = {}
        for waiting in self._window:
            for person_id, observation in waiting.matches.items():
                seen.setdefault(person_id, []).append((waiting.at, observation))
        changes: list[SightingChange] = []
        for person_id, matched in seen.items():
            if not _confirmed(len(matched), len(self._window)):
                continue
            # The plain maximum; `max` keeps the first of equal scores, so only a strictly
            # higher one replaces an earlier best.
            best = max((observation for _, observation in matched), key=lambda o: o.score)
            sighting = LiveSighting(
                id=self._new_id(),
                person_id=person_id,
                model=recognition.model,
                threshold=recognition.threshold,
                started_at=matched[0][0],
                last_seen_at=matched[-1][0],
                ended_at=None,
                best_score=best.score,
                best_crop=best.crop,
                runner_up_person_id=best.runner_up_person_id,
                runner_up_score=best.runner_up_score,
            )
            # Inserted now; the window's matches are the sighting's from here on.
            self._open[person_id] = _Open(sighting, written_at=now)
            for waiting in self._window:
                waiting.matches.pop(person_id, None)
            changes.append(Opened(sighting))
        return changes

    def _write_held(self, now: datetime.datetime) -> list[SightingChange]:
        changes: list[SightingChange] = []
        for opened in self._open.values():
            if opened.held and now - opened.written_at >= WRITE_INTERVAL:
                changes.append(Updated(opened.sighting, opened.new_best))
                opened.held, opened.new_best, opened.written_at = False, False, now
        return changes


def _confirmed(matched: int, frames: int) -> bool:
    return matched >= MIN_CONFIRMING_FRAMES and 2 * matched >= frames


def _best_per_person(recognition: Recognition) -> dict[str, Match]:
    """Each matched person's highest-scoring face in the frame, the first on a tie."""
    best: dict[str, Match] = {}
    for face in recognition.faces:
        if isinstance(face, Match):
            person_id = face.candidate.person_id
            if person_id not in best or face.candidate.score > best[person_id].candidate.score:
                best[person_id] = face
    return best


def _observe(frame: Image, face: Match) -> _Observation:
    runner_up = face.runner_up
    return _Observation(
        score=face.candidate.score,
        crop=cut_crop(frame, face.box),
        runner_up_person_id=None if runner_up is None else runner_up.person_id,
        runner_up_score=None if runner_up is None else runner_up.score,
    )


def _extend(opened: _Open, frame: Image, face: Match, now: datetime.datetime) -> None:
    """Hold a new match of an open sighting's person until the next write. The crop is cut only
    when the match beats the best so far."""
    sighting = replace(opened.sighting, last_seen_at=now)
    if face.candidate.score > sighting.best_score:
        observation = _observe(frame, face)
        sighting = replace(
            sighting,
            best_score=observation.score,
            best_crop=observation.crop,
            runner_up_person_id=observation.runner_up_person_id,
            runner_up_score=observation.runner_up_score,
        )
        opened.new_best = True
    opened.sighting, opened.held = sighting, True


def _ended(opened: _Open) -> Ended:
    sighting = opened.sighting
    return Ended(replace(sighting, ended_at=sighting.last_seen_at), opened.new_best)
