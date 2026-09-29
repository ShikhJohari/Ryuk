"""The tracker's changes written to the sightings store: what each write carries, and the end a
purge announces without writing (#16, #31)."""

import datetime
import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from ryuk.recognition import ModelKey
from ryuk.watchlist import sightings
from ryuk.watchlist.database import open_database
from ryuk.watchlist.monitoring import SightingEvent, encode_crop, unwritten_end, write
from ryuk.watchlist.sightings import SightingPerson
from ryuk.watchlist.tables import PersonOfInterestRow, SightingRow
from ryuk.watchlist.tracker import Ended, LiveSighting, Opened, Updated

T0 = datetime.datetime(2026, 9, 29, 12, 0, tzinfo=datetime.UTC)
MODEL = ModelKey(network="sface", provider="cpu", weights_sha256="a" * 64)


def at(seconds: float) -> datetime.datetime:
    return T0 + datetime.timedelta(seconds=seconds)


def pixels(fill: int) -> np.ndarray:
    crop = np.full((40, 30, 3), fill, dtype=np.uint8)
    crop.flags.writeable = False
    return crop


def live(
    *, last_seen: float = 0.4, score: float = 0.95, fill: int = 90, ended: bool = False
) -> LiveSighting:
    return LiveSighting(
        id="0123456789abcdef0123456789abcdef",
        person_id="ada",
        model=MODEL,
        threshold=0.9,
        started_at=at(0),
        last_seen_at=at(last_seen),
        ended_at=at(last_seen) if ended else None,
        best_score=score,
        best_crop=pixels(fill),
        runner_up_person_id="grace",
        runner_up_score=0.4,
    )


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    opened = open_database(tmp_path / "ryuk.sqlite3")
    with Session(opened) as session, session.begin():
        for person_id, name in (("ada", "Ada Lovelace"), ("grace", "Grace Hopper")):
            session.add(
                PersonOfInterestRow(
                    id=person_id,
                    name=name,
                    name_key=name.casefold(),
                    status="on_watchlist",
                    created_at=T0,
                    status_changed_at=T0,
                )
            )
    yield opened
    opened.dispose()


def written(engine: Engine, *changes: Opened | Updated | Ended) -> list[SightingEvent]:
    with Session(engine) as session, session.begin():
        return write(session, changes)


def stored(engine: Engine) -> SightingRow:
    with Session(engine, expire_on_commit=False) as session:
        row = session.get(SightingRow, live().id)
        assert row is not None
        _ = row.best_crop  # deferred: loaded before the session closes
        return row


def test_an_opening_is_stored_under_the_trackers_id_with_its_crop_as_jpeg(engine: Engine) -> None:
    [event] = written(engine, Opened(live()))

    assert event.type == "sighting_opened"
    assert event.sighting.id == live().id
    assert event.sighting.person == SightingPerson("ada", "Ada Lovelace", "on_watchlist")
    row = stored(engine)
    assert (row.model_key, row.runner_up_person_id, row.runner_up_score) == (MODEL.id, "grace", 0.4)
    decoded = np.asarray(PILImage.open(io.BytesIO(row.best_crop)))
    assert decoded.shape == (40, 30, 3)
    assert abs(int(decoded.mean()) - 90) <= 2


def test_an_update_without_a_new_best_match_leaves_the_stored_crop(engine: Engine) -> None:
    written(engine, Opened(live()))
    crop = stored(engine).best_crop

    [event] = written(engine, Updated(live(last_seen=1.4, fill=200), new_best=False))

    assert event.type == "sighting_updated"
    assert event.sighting.last_seen_at == at(1.4)
    assert stored(engine).best_crop == crop


def test_an_update_with_a_new_best_match_writes_its_crop_and_runner_up(engine: Engine) -> None:
    written(engine, Opened(live()))

    written(engine, Updated(live(last_seen=1.4, score=0.97, fill=200), new_best=True))

    row = stored(engine)
    assert row.best_score == 0.97
    assert row.best_crop == encode_crop(pixels(200))


def test_an_end_is_written_at_the_last_seen_time(engine: Engine) -> None:
    written(engine, Opened(live()))

    [event] = written(engine, Ended(live(last_seen=2, ended=True), new_best=False))

    assert event.type == "sighting_ended"
    assert event.sighting.ended_at == at(2)
    assert stored(engine).ended_at == at(2)


def test_a_write_to_a_sighting_no_longer_stored_announces_nothing(
    engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    events = written(engine, Updated(live(), new_best=False), Ended(live(), new_best=True))

    assert events == []
    assert "no longer stored" in caplog.text


def test_a_purged_persons_end_is_announced_from_its_last_state_without_a_write(
    engine: Engine,
) -> None:
    ada = SightingPerson("ada", "Ada Lovelace", "on_watchlist")

    event = unwritten_end(Ended(live(last_seen=2, ended=True), new_best=False), ada)

    assert event.type == "sighting_ended"
    assert event.sighting.person == ada
    assert (event.sighting.last_seen_at, event.sighting.ended_at) == (at(2), at(2))
    assert (event.sighting.model_key, event.sighting.best_score) == (MODEL.id, 0.95)
    with Session(engine) as session:
        assert sightings.sighting_page(session, person_id=None, cursor=None, limit=10).items == ()
