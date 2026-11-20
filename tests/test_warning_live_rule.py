"""The looks-like-other warning scores a new photo by the active model's live rule, the match score
the live monitor would give that face, against the threshold frozen under that rule (#73).

Ada is enrolled from two different looks, so her best photo and her mean disagree: a new shot of
one look is close to that photo but far from her mean, and a blend of both looks is close to her
mean but to neither photo.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import pytest

from ryuk.detector import Detector, Image
from ryuk.evaluation.results import MatchRule
from ryuk.recognition import Embedding
from ryuk.watchlist.database import open_database
from ryuk.watchlist.errors import EnrollmentWarning, UnacknowledgedWarningsError
from ryuk.watchlist.service import PersonOfInterest, Watchlist, start_watchlist
from ryuk.watchlist.tables import decode_embedding
from synthetic import YUNET, fake
from watchlist_service import THRESHOLD, encode, evaluated, portrait

SFACE = fake("sface")


@contextmanager
def watchlist_under(database: Path, rule: MatchRule) -> Iterator[Watchlist]:
    """The watchlist with one fake model active, its threshold frozen under `rule`."""
    watchlist = start_watchlist(
        open_database(database),
        Detector(YUNET),
        [SFACE],
        evaluated(SFACE.key, first_active=SFACE.key, rule=rule),
    )
    try:
        yield watchlist
    finally:
        watchlist.close()


def ada(watchlist: Watchlist, first: int, second: int) -> PersonOfInterest:
    """Ada enrolled from look `first`, with a photo of look `second` added: two looks far enough
    apart that the same-person warning is acknowledged."""
    person = watchlist.enroll("Ada", encode(portrait(first)))
    watchlist.add_photo(person.id, encode(portrait(second)), ["may_not_be_same_person"])
    return person


def blend(first: int, second: int) -> Image:
    """One face, half of each look."""
    halves = (portrait(first).astype(np.float64) + portrait(second).astype(np.float64)) / 2
    return halves.astype(np.uint8)


def embeddings(database: Path, person_id: str) -> list[Embedding]:
    """A person's enrolled embeddings under the active model, the only model there is."""
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            "SELECT embedding.vector FROM embedding"
            " JOIN enrolled_photo ON enrolled_photo.id = embedding.photo_id"
            " WHERE enrolled_photo.person_id = ?",
            (person_id,),
        ).fetchall()
    return [decode_embedding(vector) for (vector,) in rows]


def warnings(watchlist: Watchlist, image: Image) -> list[EnrollmentWarning]:
    """The warnings enrolling Grace from `image` raises; none if she is enrolled."""
    try:
        watchlist.enroll("Grace", encode(image))
    except UnacknowledgedWarningsError as refused:
        return refused.warnings
    return []


@pytest.mark.parametrize(("rule", "warns"), [("best-photo", True), ("mean", False)])
def test_a_photo_like_one_of_a_persons_photos_warns_only_where_that_photo_is_the_score(
    tmp_path: Path, rule: MatchRule, warns: bool
) -> None:
    # Looks 0 and 3 score about 0.28 against each other: another shot of look 0 scores about
    # 0.9997 against that photo, but about 0.80 against Ada's mean.
    with watchlist_under(tmp_path / "ryuk.sqlite3", rule) as watchlist:
        person = ada(watchlist, 0, 3)

        raised = warnings(watchlist, portrait(0, shot=3))

    assert [(w.code, w.person_id) for w in raised] == (
        [("looks_like_other", person.id)] if warns else []
    )


@pytest.mark.parametrize(("rule", "warns"), [("best-photo", False), ("mean", True)])
def test_a_photo_like_a_persons_mean_warns_only_where_the_mean_is_the_score(
    tmp_path: Path, rule: MatchRule, warns: bool
) -> None:
    # A blend of looks 0 and 5 scores about 0.80 against either photo, but about 0.95 against
    # Ada's mean.
    database = tmp_path / "ryuk.sqlite3"
    with watchlist_under(database, rule) as watchlist:
        person = ada(watchlist, 0, 5)

        raised = warnings(watchlist, blend(0, 5))

        assert [(w.code, w.person_id) for w in raised] == (
            [("looks_like_other", person.id)] if warns else []
        )
        if warns:
            grace = watchlist.enroll("Grace", encode(blend(0, 5)), ["looks_like_other"])
            # The warning reports the mean-rule match score: Ada's renormalised mean against
            # Grace's embedding.
            [probe] = embeddings(database, grace.id)
            mean = np.sum(embeddings(database, person.id), axis=0)
            score = float(mean @ probe / np.linalg.norm(mean))
            assert score >= THRESHOLD
            assert f"(score {score:.3f}, threshold {THRESHOLD:.3f})" in raised[0].detail
