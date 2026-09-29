"""Recognition on live frames: every face boxed, and every usable face scored against the
watchlist under the active model's live rule (#12, #16).

The active model's embeddings of every person on the watchlist sit in one in-memory matrix,
`WatchlistEmbeddings`, so a frame is scored by brute-force cosine in NumPy without touching the
database.
"""

from dataclasses import dataclass
from typing import assert_never

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from ryuk.detector import MAX_DETECTION_SIDE, Box, Detector, Image, is_usable
from ryuk.evaluation.results import MatchRule
from ryuk.recognition import Embedding, ModelKey
from ryuk.recognition.faces import face_crop
from ryuk.watchlist.registry import ActiveModel
from ryuk.watchlist.tables import (
    EmbeddingRow,
    EnrolledPhotoRow,
    PersonOfInterestRow,
    decode_embedding,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """The person of interest ranked first by similarity to a detection, with their match score."""

    person_id: str
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class Match:
    """A usable face whose top candidate scores at or above the threshold."""

    box: Box
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class NoMatch:
    """A usable face whose top candidate scores below the threshold. Never names anyone."""

    box: Box
    score: float | None
    """None when nobody is on the watchlist, so there is no candidate to score."""


@dataclass(frozen=True, slots=True)
class TooSmall:
    """A detection smaller than a usable face: boxed, never scored."""

    box: Box


type LiveFace = Match | NoMatch | TooSmall


@dataclass(frozen=True, slots=True)
class Recognition:
    """One frame's faces, with the model and threshold that judged them."""

    model: ModelKey
    threshold: float
    faces: tuple[LiveFace, ...]


class WatchlistEmbeddings:
    """One recognition model's embeddings of every person on the watchlist, a row per enrolled
    photo, with each person's rows together."""

    def __init__(
        self,
        vectors: NDArray[np.float32],
        persons: tuple[tuple[str, str], ...],
        starts: NDArray[np.intp],
    ) -> None:
        self._vectors = vectors
        self._persons = persons
        """Each person's ID and name, in the order of their rows."""
        self._starts = starts
        """The first row of each person's photos."""

    @classmethod
    def load(cls, session: Session, model: ModelKey | None) -> "WatchlistEmbeddings":
        """`model`'s embeddings of the watchlist from the app database; none when no model is
        active."""
        rows = (
            []
            if model is None
            else session.execute(
                select(PersonOfInterestRow.id, PersonOfInterestRow.name, EmbeddingRow.vector)
                .join(EnrolledPhotoRow, EnrolledPhotoRow.person_id == PersonOfInterestRow.id)
                .join(EmbeddingRow, EmbeddingRow.photo_id == EnrolledPhotoRow.id)
                .where(
                    PersonOfInterestRow.status == "on_watchlist",
                    EmbeddingRow.model_key == model.id,
                )
                .order_by(PersonOfInterestRow.id)
            ).all()
        )
        persons: list[tuple[str, str]] = []
        starts: list[int] = []
        for index, (person_id, name, _) in enumerate(rows):
            if not persons or persons[-1][0] != person_id:
                persons.append((person_id, name))
                starts.append(index)
        vectors = (
            np.stack([decode_embedding(blob) for _, _, blob in rows])
            if rows
            else np.empty((0, 0), dtype=np.float32)
        )
        return cls(vectors, tuple(persons), np.asarray(starts, dtype=np.intp))

    def top_candidate(self, probe: Embedding, rule: MatchRule) -> Candidate | None:
        """The person of interest who scores highest against `probe` under `rule`, or None when
        nobody is on the watchlist."""
        if not self._persons:
            return None
        match rule:
            case "best-photo":
                # Each person's score is the cosine to their best enrolled photo.
                scores = np.maximum.reduceat(self._vectors @ probe, self._starts)
            case "mean":
                # Each person's score is the cosine to their renormalised mean embedding.
                means = np.add.reduceat(self._vectors, self._starts, axis=0)
                means /= np.linalg.norm(means, axis=1, keepdims=True)
                scores = means @ probe
            case "learned":
                # Scores a person against the runner-up, so it needs the whole ranking.
                raise ValueError("the learned rule is not computed live yet")
            case _:
                assert_never(rule)
        best = int(np.argmax(scores))
        person_id, name = self._persons[best]
        return Candidate(person_id, name, float(scores[best]))


def recognise(
    detector: Detector, active: ActiveModel, watchlist: WatchlistEmbeddings, frame: Image
) -> Recognition:
    """Every face YuNet finds in a BGR frame, each usable one judged by the active model against
    the watchlist."""
    model, evaluated = active.model, active.evaluated
    faces: list[LiveFace] = []
    for detection in detector.detect(frame, max_side=MAX_DETECTION_SIDE):
        if not is_usable(detection):
            faces.append(TooSmall(detection.box))
            continue
        probe = model.embed(face_crop(detector, frame, detection, evaluated.crop, model.input_size))
        candidate = watchlist.top_candidate(probe, evaluated.rule)
        if candidate is not None and candidate.score >= evaluated.threshold:
            faces.append(Match(detection.box, candidate))
        else:
            faces.append(NoMatch(detection.box, None if candidate is None else candidate.score))
    return Recognition(active.key, evaluated.threshold, tuple(faces))
