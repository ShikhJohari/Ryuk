"""The sightings store at the service seam: the writes the live monitor's tracker makes, the
reads the history pages make, and the startup rule for sightings a previous run left open."""

import datetime
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from ryuk.watchlist import sightings
from ryuk.watchlist.database import open_database
from ryuk.watchlist.errors import WatchlistError
from ryuk.watchlist.service import start_watchlist
from ryuk.watchlist.sightings import BestMatch, NewSighting, SightingPerson
from ryuk.watchlist.tables import PersonOfInterestRow, SightingRow
from synthetic import fake
from watchlist_service import evaluated

T0 = datetime.datetime(2026, 9, 29, 12, 0, 0, 250_000, tzinfo=datetime.UTC)
MODEL = "sface-cpu-aa"


def at(seconds: float) -> datetime.datetime:
    return T0 + datetime.timedelta(seconds=seconds)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    opened = open_database(tmp_path / "ryuk.sqlite3")
    with Session(opened) as session, session.begin():
        add_person(session, "ada", "Ada Lovelace")
        add_person(session, "grace", "Grace Hopper", status="removed")
    yield opened
    opened.dispose()


def add_person(session: Session, person_id: str, name: str, status: str = "on_watchlist") -> None:
    session.add(
        PersonOfInterestRow(
            id=person_id,
            name=name,
            name_key=name.casefold(),
            status=status,
            created_at=T0,
            status_changed_at=T0,
        )
    )


def best(score: float, crop: bytes = b"\xff\xd8 crop", runner_up: str | None = None) -> BestMatch:
    return BestMatch(
        score=score,
        crop=crop,
        runner_up_person_id=runner_up,
        runner_up_score=None if runner_up is None else score - 0.3,
    )


def open_one(session: Session, person_id: str = "ada", started: float = 0) -> str:
    opened = sightings.open_sighting(
        session,
        NewSighting(
            person_id=person_id,
            model_key=MODEL,
            threshold=0.9,
            started_at=at(started),
            last_seen_at=at(started + 0.4),
            best=best(0.95, runner_up="grace"),
        ),
    )
    return opened.id


def stored(engine: Engine, sighting_id: str) -> SightingRow:
    with Session(engine, expire_on_commit=False) as session:
        row = session.get(SightingRow, sighting_id)
        assert row is not None
        _ = row.best_crop  # deferred: loaded before the session closes
        return row


def test_opening_a_sighting_stores_it_open_and_returns_its_summary(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        opened = sightings.open_sighting(
            session,
            NewSighting(
                person_id="ada",
                model_key=MODEL,
                threshold=0.9,
                started_at=at(0),
                last_seen_at=at(0.4),
                best=best(0.95, runner_up="grace"),
            ),
        )

    assert opened.person == SightingPerson("ada", "Ada Lovelace", "on_watchlist")
    assert (opened.model_key, opened.threshold, opened.best_score) == (MODEL, 0.9, 0.95)
    assert (opened.started_at, opened.last_seen_at, opened.ended_at) == (at(0), at(0.4), None)
    row = stored(engine, opened.id)
    assert (row.best_crop, row.runner_up_person_id) == (b"\xff\xd8 crop", "grace")
    assert row.runner_up_score == pytest.approx(0.65)


def test_opening_a_sighting_for_nobody_is_refused(engine: Engine) -> None:
    with Session(engine) as session, pytest.raises(WatchlistError) as refused:
        open_one(session, person_id="nobody")

    assert refused.value.code == "not_found"


def test_an_update_moves_last_seen_and_keeps_the_crop_unless_the_best_match_changed(
    engine: Engine,
) -> None:
    with Session(engine) as session, session.begin():
        sighting_id = open_one(session)

    with Session(engine) as session, session.begin():
        updated = sightings.update_sighting(session, sighting_id, last_seen_at=at(1.5))
    assert updated is not None
    assert (updated.last_seen_at, updated.best_score, updated.ended_at) == (at(1.5), 0.95, None)
    assert stored(engine, sighting_id).best_crop == b"\xff\xd8 crop"

    with Session(engine) as session, session.begin():
        better = sightings.update_sighting(
            session, sighting_id, last_seen_at=at(2), best=best(0.97, b"better")
        )
    assert better is not None
    assert better.best_score == 0.97
    row = stored(engine, sighting_id)
    # The runner-up comes from the same frame as the best match, so it changes with it.
    assert (row.best_crop, row.runner_up_person_id, row.runner_up_score) == (b"better", None, None)


def test_ending_a_sighting_sets_its_end_to_when_the_person_was_last_seen(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        sighting_id = open_one(session)

    with Session(engine) as session, session.begin():
        ended = sightings.end_sighting(session, sighting_id, last_seen_at=at(3))

    assert ended is not None
    assert (ended.last_seen_at, ended.ended_at) == (at(3), at(3))
    with Session(engine) as session, pytest.raises(ValueError, match="already ended"):
        sightings.update_sighting(session, sighting_id, last_seen_at=at(4))


def test_writing_to_a_sighting_that_is_gone_writes_nothing(engine: Engine) -> None:
    # As after a purge, which takes the person's sightings with them.
    with Session(engine) as session, session.begin():
        assert sightings.update_sighting(session, "gone", last_seen_at=at(1)) is None
        assert sightings.end_sighting(session, "gone", last_seen_at=at(1)) is None


def test_a_runner_up_needs_a_score() -> None:
    with pytest.raises(ValueError, match="runner-up"):
        BestMatch(score=0.95, crop=b"", runner_up_person_id="grace", runner_up_score=None)


def test_ending_every_open_sighting_ends_each_at_its_own_last_sighting(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        first = open_one(session, started=0)
        second = open_one(session, person_id="grace", started=10)
        done = open_one(session, started=20)
        sightings.update_sighting(session, second, last_seen_at=at(14))
        sightings.end_sighting(session, done, last_seen_at=at(21))

    with Session(engine) as session, session.begin():
        assert sightings.end_open_sightings(session) == 2

    assert stored(engine, first).ended_at == at(0.4)
    assert stored(engine, second).ended_at == at(14)
    assert stored(engine, done).ended_at == at(21)


def test_sightings_a_previous_run_left_open_are_ended_at_startup(tmp_path: Path) -> None:
    database = tmp_path / "ryuk.sqlite3"
    engine = open_database(database)
    with Session(engine) as session, session.begin():
        add_person(session, "ada", "Ada Lovelace")
        add_person(session, "grace", "Grace Hopper")
        sighting_id = open_one(session)
        sightings.update_sighting(session, sighting_id, last_seen_at=at(7))
    engine.dispose()

    model = fake("sface")
    watchlist = start_watchlist(
        open_database(database), None, [model], evaluated(model.key, first_active=model.key)
    )
    try:
        [summary] = watchlist.sightings().items
    finally:
        watchlist.close()

    assert (summary.id, summary.last_seen_at, summary.ended_at) == (sighting_id, at(7), at(7))


def test_a_page_is_newest_first_and_its_cursor_resumes_after_its_last_sighting(
    engine: Engine,
) -> None:
    with Session(engine) as session, session.begin():
        ids = [open_one(session, started=seconds) for seconds in (0, 5, 5, 5, 9)]
    newest_first = [ids[4], *sorted(ids[1:4], reverse=True), ids[0]]

    pages: list[list[str]] = []
    cursor = None
    with Session(engine) as session:
        while True:
            page = sightings.sighting_page(session, person_id=None, cursor=cursor, limit=2)
            pages.append([s.id for s in page.items])
            cursor = page.next_cursor
            if cursor is None:
                break

    assert pages == [newest_first[:2], newest_first[2:4], newest_first[4:]]


def test_a_full_last_page_has_no_cursor(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        open_one(session, started=0)
        open_one(session, started=1)

    with Session(engine) as session:
        page = sightings.sighting_page(session, person_id=None, cursor=None, limit=2)

    assert (len(page.items), page.next_cursor) == (2, None)


@pytest.mark.parametrize("limit", [0, sightings.MAX_PAGE_SIZE + 1])
def test_a_page_size_out_of_bounds_is_a_programming_error(engine: Engine, limit: int) -> None:
    with Session(engine) as session, pytest.raises(ValueError, match="page size"):
        sightings.sighting_page(session, person_id=None, cursor=None, limit=limit)


def test_a_sighting_is_read_with_its_runner_up_and_crop(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        sighting_id = open_one(session)

    with Session(engine) as session:
        sighting = sightings.get_sighting(session, sighting_id)
        crop = sightings.sighting_crop(session, sighting_id)

    assert sighting.summary.id == sighting_id
    assert sighting.runner_up is not None
    assert sighting.runner_up.person == SightingPerson("grace", "Grace Hopper", "removed")
    assert sighting.runner_up.score == pytest.approx(0.65)
    assert crop == b"\xff\xd8 crop"


def test_the_crop_is_not_read_with_a_page(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        open_one(session)

    with Session(engine) as session:
        sightings.sighting_page(session, person_id=None, cursor=None, limit=10)
        [row] = session.scalars(select(SightingRow)).all()
        assert "best_crop" not in row.__dict__
