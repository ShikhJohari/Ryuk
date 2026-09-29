"""Ranking the watchlist against one live face: the top candidate, who can become a match, and the
runner-up a sighting records beside its best match (#16, #31)."""

from pathlib import Path

import numpy as np
import pytest

from ryuk.detector import Detector
from ryuk.recognition import Embedding
from ryuk.watchlist.database import open_database
from ryuk.watchlist.live import Candidate, Match, WatchlistEmbeddings
from ryuk.watchlist.service import start_watchlist
from synthetic import YUNET, fake
from watchlist_service import THRESHOLD, encode, evaluated, portrait


def unit(*values: float) -> Embedding:
    vector = np.asarray(values, dtype=np.float32)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def watchlist(*persons: tuple[str, str, list[Embedding]]) -> WatchlistEmbeddings:
    """A watchlist of `(id, name, enrolled embeddings)`, each person's rows together."""
    vectors = [vector for _, _, photos in persons for vector in photos]
    counts = [len(photos) for _, _, photos in persons]
    starts = np.cumsum([0, *counts[:-1]], dtype=np.intp) if persons else np.empty(0, np.intp)
    return WatchlistEmbeddings(
        np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32),
        tuple((person_id, name) for person_id, name, _ in persons),
        starts,
    )


def test_with_nobody_on_the_watchlist_there_is_no_ranking() -> None:
    assert watchlist().ranking(unit(1, 0, 0), "best-photo") is None


def test_with_one_person_they_are_the_top_candidate_with_no_runner_up() -> None:
    ranking = watchlist(("a", "Ada", [unit(1, 0, 0)])).ranking(unit(1, 0, 0), "best-photo")

    assert ranking is not None
    assert ranking.top == Candidate("a", "Ada", ranking.top.score)
    assert ranking.top.score == pytest.approx(1.0)
    assert ranking.runner_up is None


def test_the_runner_up_is_the_second_highest_scoring_person() -> None:
    embeddings = watchlist(
        ("a", "Ada", [unit(0, 0, 1)]),
        ("b", "Barbara", [unit(1, 0, 0)]),
        ("c", "Charles", [unit(1, 1, 0)]),
    )

    ranking = embeddings.ranking(unit(1, 0.2, 0), "best-photo")

    assert ranking is not None
    assert ranking.top.person_id == "b"
    assert ranking.runner_up is not None
    assert ranking.runner_up.person_id == "c"
    assert ranking.runner_up.name == "Charles"
    assert ranking.top.score > ranking.runner_up.score > 0.0


def test_each_person_is_ranked_by_their_best_enrolled_photo_so_they_rank_only_once() -> None:
    # Ada's two photos both beat Barbara's: the runner-up is still Barbara, never Ada again.
    embeddings = watchlist(
        ("a", "Ada", [unit(1, 0, 0), unit(1, 0.1, 0)]),
        ("b", "Barbara", [unit(0, 1, 0)]),
    )

    ranking = embeddings.ranking(unit(1, 0, 0), "best-photo")

    assert ranking is not None
    assert ranking.top.person_id == "a"
    assert ranking.top.score == pytest.approx(1.0)
    assert ranking.runner_up is not None
    assert (ranking.runner_up.person_id, ranking.runner_up.name) == ("b", "Barbara")
    assert ranking.runner_up.score == pytest.approx(0.0, abs=1e-6)


def test_a_tie_ranks_the_person_listed_first_on_top_and_the_other_as_runner_up() -> None:
    embeddings = watchlist(
        ("a", "Ada", [unit(1, 0, 0)]),
        ("b", "Barbara", [unit(1, 0, 0)]),
        ("c", "Charles", [unit(0, 1, 0)]),
    )

    ranking = embeddings.ranking(unit(1, 0, 0), "best-photo")

    assert ranking is not None
    assert ranking.top.person_id == "a"
    assert ranking.runner_up is not None
    assert ranking.runner_up.person_id == "b"
    assert ranking.runner_up.score == ranking.top.score


def test_a_live_match_carries_the_runner_up_whatever_their_score(tmp_path: Path) -> None:
    sface = fake("sface")
    watchlist = start_watchlist(
        open_database(tmp_path / "ryuk.sqlite3"),
        Detector(YUNET),
        [sface],
        evaluated(sface.key, first_active=sface.key),
    )
    try:
        ada = watchlist.enroll("Ada Lovelace", encode(portrait(0)))
        grace = watchlist.enroll("Grace Hopper", encode(portrait(3)))
        [face] = watchlist.recognise(portrait(0, shot=1)).faces
    finally:
        watchlist.close()

    assert isinstance(face, Match)
    assert face.candidate.person_id == ada.id
    assert face.runner_up is not None
    assert (face.runner_up.person_id, face.runner_up.name) == (grace.id, "Grace Hopper")
    assert face.runner_up.score < THRESHOLD <= face.candidate.score
