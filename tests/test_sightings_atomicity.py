"""The live monitor's sightings and the database stay in step when a write fails (#31).

Every path that changes a sighting is all-or-nothing: if its transaction fails to commit, or the
watchlist's embeddings fail to reload after it, nothing is announced, the tracker is as it was, the
stored sighting is as it was, and trying again works.
"""

import datetime
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from ryuk.detector import Detector, Image
from ryuk.recognition import ModelKey, RecognitionModel
from ryuk.watchlist.database import open_database
from ryuk.watchlist.live import WatchlistEmbeddings
from ryuk.watchlist.monitoring import LiveFrame, MonitoringSession
from ryuk.watchlist.service import PersonOfInterest, Watchlist, start_watchlist
from synthetic import YUNET, fake
from test_live_sightings import BROWSER, MONITOR, confirm, enroll, serving
from watchlist_service import FakeClock, blank, encode, evaluated, portrait

T0 = datetime.datetime(2026, 9, 29, 12, 0, tzinfo=datetime.UTC)
ADA = portrait(0, shot=0)
"""Ada, sent as raw pixels rather than a JPEG frame: matched at about 0.9962."""
ADA_BETTER = portrait(0, shot=3)
"""Ada matched higher, at about 0.9972."""
SFACE, FACENET = fake("sface"), fake("facenet", seed=1)


class InjectedError(Exception):
    """The fault a test injects."""


@contextmanager
def failing_commit(*, armed: bool = True) -> Iterator[threading.Event]:
    """The next transaction to commit fails to, and sets the event it yields. Unless `armed`,
    only once the test clears the event."""
    fired = threading.Event()
    if not armed:
        fired.set()

    def fail(session: Session) -> None:
        if fired.is_set():
            return
        fired.set()
        raise InjectedError("the commit failed")

    event.listen(Session, "before_commit", fail)
    try:
        yield fired
    finally:
        event.remove(Session, "before_commit", fail)


def failing_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The next reload of the watchlist's embeddings fails, as after a change."""
    load = WatchlistEmbeddings.load
    failed = False

    def once(session: Session, model: ModelKey | None) -> WatchlistEmbeddings:
        nonlocal failed
        if failed:
            return load(session, model)
        failed = True
        raise InjectedError("the reload failed")

    monkeypatch.setattr(WatchlistEmbeddings, "load", once)


class Monitor:
    """A watchlist with Ada and Grace enrolled, and one live monitor session on it."""

    def __init__(self, tmp_path: Path, clock: FakeClock, *models: RecognitionModel) -> None:
        models = models or (SFACE,)
        self.clock = clock
        self.watchlist: Watchlist = start_watchlist(
            open_database(tmp_path / "ryuk.sqlite3"),
            Detector(YUNET),
            models,
            evaluated(*(model.key for model in models), first_active=models[0].key),
            clock,
        )
        self.ada: PersonOfInterest = self.watchlist.enroll("Ada Lovelace", encode(portrait(0)))
        self.grace: PersonOfInterest = self.watchlist.enroll("Grace Hopper", encode(portrait(3)))
        self.session: MonitoringSession = self.watchlist.begin_monitoring()

    def frame(self, image: Image, after: float = 0.1) -> LiveFrame:
        self.clock.advance(after)
        return self.watchlist.recognise(image, self.session)

    def confirm(self) -> str:
        """Ada's sighting, opened by three matched frames: its ID."""
        [opened] = [a for _ in range(3) for a in self.frame(ADA).sightings]
        assert opened.type == "sighting_opened"
        return opened.sighting.id

    def stored(self, sighting_id: str) -> Any:
        return self.watchlist.sighting(sighting_id)

    def still_open(self, sighting_id: str) -> None:
        """The sighting is open both in the tracker and in its row."""
        assert self.stored(sighting_id).summary.ended_at is None
        live = self.frame(ADA)
        assert [a.type for a in live.sightings if a.type != "sighting_updated"] == []
        assert live.sighting_ids[self.ada.id] == sighting_id


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(T0)


@pytest.fixture
def monitor(tmp_path: Path, clock: FakeClock) -> Iterator[Monitor]:
    opened = Monitor(tmp_path, clock)
    yield opened
    opened.watchlist.close()


def test_a_frame_whose_opening_fails_to_store_opens_nothing_and_the_next_frame_opens_it(
    monitor: Monitor,
) -> None:
    monitor.frame(ADA)
    monitor.frame(ADA)

    with failing_commit(), pytest.raises(InjectedError):
        monitor.frame(ADA)
    stored = monitor.watchlist.sightings().items
    retried = monitor.frame(ADA)

    assert stored == ()
    [opened] = retried.sightings
    assert opened.type == "sighting_opened"
    assert retried.sighting_ids == {monitor.ada.id: opened.sighting.id}
    assert monitor.stored(opened.sighting.id).summary.ended_at is None


def test_a_frame_whose_end_fails_to_store_keeps_the_sighting_open_until_the_next_frame(
    monitor: Monitor,
) -> None:
    sighting_id = monitor.confirm()
    last_seen = monitor.clock.now

    with failing_commit(), pytest.raises(InjectedError):
        monitor.frame(blank(), after=3)
    stored = monitor.stored(sighting_id).summary
    retried = monitor.frame(blank())

    assert stored.ended_at is None
    [ended] = retried.sightings
    assert (ended.type, ended.sighting.id) == ("sighting_ended", sighting_id)
    assert monitor.stored(sighting_id).summary.ended_at == last_seen


def test_a_tick_whose_write_fails_changes_nothing_and_the_next_tick_writes_it(
    monitor: Monitor,
) -> None:
    sighting_id = monitor.confirm()
    monitor.frame(ADA_BETTER)
    monitor.clock.advance(1)

    with failing_commit(), pytest.raises(InjectedError):
        monitor.watchlist.tick(monitor.session)
    stored = monitor.stored(sighting_id).summary
    [updated] = monitor.watchlist.tick(monitor.session)

    assert updated.type == "sighting_updated"
    assert updated.sighting.best_score > stored.best_score
    assert monitor.stored(sighting_id).summary.best_score == updated.sighting.best_score


def test_ending_a_monitoring_session_that_fails_to_store_keeps_its_sightings_open(
    monitor: Monitor,
) -> None:
    sighting_id = monitor.confirm()

    with failing_commit(), pytest.raises(InjectedError):
        monitor.watchlist.end_monitoring(monitor.session)
    monitor.still_open(sighting_id)
    [ended] = monitor.watchlist.end_monitoring(monitor.session)

    assert (ended.type, ended.sighting.id) == ("sighting_ended", sighting_id)
    assert monitor.stored(sighting_id).summary.ended_at is not None
    assert monitor.watchlist.end_monitoring(monitor.session) == ()


def test_a_removal_that_fails_to_reload_keeps_the_person_and_their_sighting(
    monitor: Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    sighting_id = monitor.confirm()

    failing_reload(monkeypatch)
    with pytest.raises(InjectedError):
        monitor.watchlist.update_person(monitor.ada.id, status="removed")
    assert monitor.watchlist.person(monitor.ada.id).status == "on_watchlist"
    monitor.still_open(sighting_id)
    removed = monitor.watchlist.update_person(monitor.ada.id, status="removed")

    [ended] = removed.sightings
    assert (ended.type, ended.sighting.id) == ("sighting_ended", sighting_id)
    assert monitor.stored(sighting_id).summary.ended_at is not None


def test_a_purge_that_fails_to_commit_keeps_the_person_and_their_sighting(
    monitor: Monitor,
) -> None:
    sighting_id = monitor.confirm()

    with failing_commit(), pytest.raises(InjectedError):
        monitor.watchlist.purge(monitor.ada.id)
    assert monitor.watchlist.person(monitor.ada.id).name == "Ada Lovelace"
    monitor.still_open(sighting_id)
    [ended] = monitor.watchlist.purge(monitor.ada.id)

    assert (ended.type, ended.sighting.id) == ("sighting_ended", sighting_id)
    assert monitor.watchlist.sightings().items == ()


def test_a_runner_up_purge_that_fails_keeps_them_as_runner_up(monitor: Monitor) -> None:
    sighting_id = monitor.confirm()
    opened_best = monitor.stored(sighting_id).summary.best_score
    monitor.frame(ADA_BETTER)  # a new best match held, Grace its runner-up

    with failing_commit(), pytest.raises(InjectedError):
        monitor.watchlist.purge(monitor.grace.id)
    [ended] = monitor.watchlist.end_monitoring(monitor.session)

    assert ended.sighting.id == sighting_id
    assert ended.sighting.best_score > opened_best
    runner_up = monitor.stored(sighting_id).runner_up
    assert runner_up is not None
    assert runner_up.person is not None
    assert runner_up.person.id == monitor.grace.id


def test_a_model_switch_that_fails_to_commit_keeps_the_model_and_every_sighting(
    tmp_path: Path, clock: FakeClock
) -> None:
    monitor = Monitor(tmp_path, clock, SFACE, FACENET)
    try:
        sighting_id = monitor.confirm()

        with failing_commit(), pytest.raises(InjectedError):
            monitor.watchlist.activate(FACENET.key.id)
        active = monitor.watchlist.registry.active
        monitor.still_open(sighting_id)
        switched = monitor.watchlist.activate(FACENET.key.id)
    finally:
        monitor.watchlist.close()

    assert active is not None
    assert active.key == SFACE.key
    [ended] = switched.sightings
    assert (ended.type, ended.sighting.id) == ("sighting_ended", sighting_id)


def test_a_socket_whose_sightings_fail_to_end_closes_cleanly_and_leaves_them_open(
    tmp_path: Path, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with failing_commit(armed=False) as fired:
            with client.websocket_connect(MONITOR, headers=BROWSER) as socket:
                opened = confirm(socket, clock)["sighting"]
                fired.clear()  # armed now, for the end the close makes
            assert fired.wait(timeout=10)
        stored = client.get(f"/api/sightings/{opened['id']}").json()
        health = client.get("/api/health")

    assert stored["endedAt"] is None
    assert health.status_code == 200
    [record] = [r for r in caplog.records if r.name == "ryuk.api.monitor"]
    assert record.levelname == "ERROR"
    assert "sightings" in record.getMessage()
