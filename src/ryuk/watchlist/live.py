"""Recognition on live frames: every face boxed, and every usable face scored against the
watchlist under the active model's live rule (#12, #16).

The active model's embeddings of every person on the watchlist sit in one in-memory matrix, the
`Gallery`, so a frame is scored by brute-force cosine in NumPy without touching the database.
"""

from dataclasses import dataclass
from typing import assert_never

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from ryuk.detector import PHOTO_DETECTION_SIDE, Box, Detector, Image, is_usable
from ryuk.evaluation.results import MatchRule
from ryuk.recognition import Embedding, ModelKey, RecognitionModel
from ryuk.recognition.faces import face_crop
from ryuk.watchlist.registry import RegisteredModel
from ryuk.watchlist.tables import (
    EmbeddingRow,
    EnrolledPhotoRow,
    PersonOfInterestRow,
    decode_embedding,
)


@dataclass(frozen=True, slots=True)
class MatchedPerson:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class Match:
    """A usable face whose top candidate scores at or above the threshold."""

    box: Box
    score: float
    person: MatchedPerson


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


@dataclass(frozen=True, slots=True)
class Candidate:
    person: MatchedPerson
    score: float


class Gallery:
    """One recognition model's embeddings of every person on the watchlist, a row per enrolled
    photo, with each person's rows together."""

    def __init__(
        self,
        vectors: NDArray[np.float32],
        persons: tuple[MatchedPerson, ...],
        starts: NDArray[np.intp],
    ) -> None:
        self._vectors = vectors
        self._persons = persons
        self._starts = starts

    @classmethod
    def load(cls, session: Session, model: ModelKey | None) -> "Gallery":
        """`model`'s gallery from the app database; empty when no model is active."""
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
        persons: list[MatchedPerson] = []
        starts: list[int] = []
        for index, (person_id, name, _) in enumerate(rows):
            if not persons or persons[-1].id != person_id:
                persons.append(MatchedPerson(person_id, name))
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
            case _:
                assert_never(rule)
        best = int(np.argmax(scores))
        return Candidate(self._persons[best], float(scores[best]))


def recognise(
    detector: Detector,
    active: RegisteredModel,
    loaded: RecognitionModel,
    gallery: Gallery,
    image: Image,
) -> Recognition:
    """Every face YuNet finds in a BGR frame, judged by the active model against `gallery`."""
    evaluated = active.evaluated
    if evaluated is None:
        raise ValueError(f"{active.name} has no threshold, so it cannot be active")
    faces: list[LiveFace] = []
    for detection in detector.detect(image, max_side=PHOTO_DETECTION_SIDE):
        if not is_usable(detection):
            faces.append(TooSmall(detection.box))
            continue
        probe = loaded.embed(face_crop(detector, image, detection, active.crop, loaded.input_size))
        candidate = gallery.top_candidate(probe, evaluated.rule)
        if candidate is not None and candidate.score >= evaluated.threshold:
            faces.append(Match(detection.box, candidate.score, candidate.person))
        else:
            faces.append(NoMatch(detection.box, None if candidate is None else candidate.score))
    return Recognition(active.key, evaluated.threshold, tuple(faces))
