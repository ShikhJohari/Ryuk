"""Managing persons of interest: their enrolled photos, and the embeddings made from them.

Enrolled photos are the source of truth (ADR 0003): every photo is embedded under every
recognition model whose weights are present, and at startup embeddings made with other weights,
another crop or another `PIPELINE_VERSION` are rebuilt from the photos and missing ones filled in.
"""

import datetime
import logging
import threading
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Literal, cast

from sqlalchemy import Engine, delete, exists, func, select, update
from sqlalchemy.orm import Session, selectinload

from ryuk.detector import (
    MAX_DETECTION_SIDE,
    MIN_USABLE_FACE_SIZE,
    Box,
    Detection,
    Detector,
    Image,
    Landmarks,
)
from ryuk.detector import usable_faces as usable
from ryuk.pipeline import PIPELINE_VERSION
from ryuk.recognition import Embedding, ModelKey, RecognitionModel
from ryuk.recognition.faces import face_crop
from ryuk.watchlist import sightings
from ryuk.watchlist.errors import (
    EnrollmentWarning,
    UnacknowledgedWarningsError,
    WarningCode,
    WatchlistError,
    not_found,
)
from ryuk.watchlist.live import WatchlistEmbeddings, recognise
from ryuk.watchlist.monitoring import (
    LiveFrame,
    MonitoringSession,
    MonitoringSessions,
    SightingAnnouncement,
)
from ryuk.watchlist.photos import Photo, prepare_photo
from ryuk.watchlist.registry import (
    ActiveModel,
    Evaluation,
    ModelRegistry,
    RegisteredModel,
    Unavailable,
    register,
)
from ryuk.watchlist.sightings import DEFAULT_PAGE_SIZE, Sighting, SightingPage
from ryuk.watchlist.tables import (
    EmbeddingRow,
    EnrolledPhotoRow,
    PersonOfInterestRow,
    PersonStatus,
    RecognitionModelRow,
    SettingRow,
    SightingRow,
    decode_embedding,
    encode_embedding,
)

logger = logging.getLogger(__name__)

type StatusFilter = Literal["on_watchlist", "removed", "all"]
type Clock = Callable[[], datetime.datetime]

ACTIVE_MODEL_SETTING: Final = "active_model"
MAX_NAME_LENGTH: Final = 200


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


@dataclass(frozen=True, slots=True)
class EnrolledPhoto:
    id: str
    media_type: str
    width: int
    height: int
    created_at: datetime.datetime


@dataclass(frozen=True, slots=True)
class PersonOfInterest:
    id: str
    name: str
    status: PersonStatus
    created_at: datetime.datetime
    status_changed_at: datetime.datetime
    photos: tuple[EnrolledPhoto, ...]


@dataclass(frozen=True, slots=True)
class PersonChange:
    """A person of interest as a change left them, with the sighting a removal ended."""

    person: PersonOfInterest
    sightings: tuple[SightingAnnouncement, ...] = ()


@dataclass(frozen=True, slots=True)
class Activation:
    """The active model a switch chose, with the sightings the switch ended."""

    active: ActiveModel
    sightings: tuple[SightingAnnouncement, ...] = ()


def start_watchlist(
    engine: Engine,
    detector: Detector | None,
    models: Sequence[RecognitionModel | Unavailable],
    evaluation: Evaluation,
    clock: Clock = utc_now,
) -> "Watchlist":
    """The watchlist on a migrated database, with its embeddings brought up to date, the active
    model chosen and any sighting a previous run left open ended, ready before the service
    accepts a request."""
    registered = register(models, evaluation)
    with Session(engine) as session, session.begin():
        sightings.end_open_sightings(session)
        for model in registered:
            if model.model is not None:
                _sync_embeddings(session, detector, model, model.model)
        active = _choose_active(session, registered, evaluation)
    return Watchlist(engine, detector, ModelRegistry(registered, active), clock)


class Watchlist:
    def __init__(
        self, engine: Engine, detector: Detector | None, registry: ModelRegistry, clock: Clock
    ) -> None:
        self._engine = engine
        self._detector = detector
        self._clock = clock
        self.registry = registry
        # The detector and models are not thread-safe, and the warnings and the last-photo rule
        # read what they then write, so changes and live frames are handled one at a time.
        self._lock = threading.Lock()
        with Session(engine) as session:
            self._embeddings = self._watchlist_embeddings(session)
        # Each live monitor connection's sightings, changed only under the lock.
        self._monitoring = MonitoringSessions(engine)

    def close(self) -> None:
        self._engine.dispose()

    def monitor_refusal(self) -> str | None:
        """Why live frames cannot be recognised, in one sentence, or None when they can."""
        if self.registry.active is None:
            return "No evaluated recognition model can be active."
        if self._detector is None:
            return "The face detector's weights are missing."
        return None

    def activate(self, model_id: str) -> Activation:
        """Make the model with key `model_id` the active model, and keep that choice.

        Only an evaluated model whose weights are present can be active. The live monitor's next
        frame is judged by it, against the embeddings enrollment already made under it. A switch
        to another model ends every open sighting, since a sighting is under one model.
        """
        with self._lock:
            model = self.registry.model(model_id)
            if model is None:
                raise not_found("recognition model")
            if not model.can_be_active:
                why = (
                    "its weights are not on this machine"
                    if model.model is None
                    else "evaluation froze no threshold for it"
                )
                raise WatchlistError(
                    409, "cannot_be_active", f"{model.name} cannot be active: {why}."
                )
            current = self.registry.active
            with (
                self._monitoring.all_or_nothing(),
                Session(self._engine) as session,
                session.begin(),
            ):
                session.merge(SettingRow(key=ACTIVE_MODEL_SETTING, value=model.key.id))
                embeddings = WatchlistEmbeddings.load(session, model.key)
                ended = (
                    ()
                    if current is not None and current.key == model.key
                    else self._monitoring.end_all(session)
                )
            self._embeddings = embeddings
            return Activation(self.registry.activate(model.key), ended)

    def begin_monitoring(self) -> MonitoringSession:
        """Start tracking sightings for a live monitor connection."""
        with self._lock:
            return self._monitoring.begin()

    def end_monitoring(
        self, monitor_session: MonitoringSession
    ) -> tuple[SightingAnnouncement, ...]:
        """End every sighting `monitor_session` has open, when its connection closes, however it
        closes; the sightings of any other session are left open."""
        with self._lock:
            return self._monitoring.end(monitor_session)

    def recognise(self, frame: Image, monitor_session: MonitoringSession) -> LiveFrame:
        """Every face in a live frame, each usable one scored against the watchlist by the
        active model under its live rule, and the sightings the frame opened, updated or ended
        in `monitor_session`."""
        with self._lock:
            active = self.registry.active
            if self._detector is None or active is None:
                # The live monitor refuses to start in this state (`monitor_refusal`).
                raise RuntimeError("live frames need the detector and an active model")
            recognition = recognise(self._detector, active, self._embeddings, frame)
            return self._monitoring.observe(monitor_session, frame, recognition, self._clock())

    def tick(self, monitor_session: MonitoringSession) -> tuple[SightingAnnouncement, ...]:
        """End `monitor_session`'s sightings whose gap has passed and write its held changes that
        are due, with no frame: the live monitor's frames stop while its tab is hidden."""
        with self._lock:
            return self._monitoring.tick(monitor_session, self._clock())

    def persons(self, status: StatusFilter) -> list[PersonOfInterest]:
        """Persons of interest with `status`, by name."""
        query = (
            select(PersonOfInterestRow)
            .options(selectinload(PersonOfInterestRow.photos))
            .order_by(PersonOfInterestRow.name_key, PersonOfInterestRow.created_at)
        )
        if status != "all":
            query = query.where(PersonOfInterestRow.status == status)
        with Session(self._engine) as session:
            return [_person(row) for row in session.scalars(query)]

    def person(self, person_id: str) -> PersonOfInterest:
        with Session(self._engine) as session:
            return _person(_get_person(session, person_id))

    def enroll(
        self, name: str, upload: bytes, acknowledged: Iterable[WarningCode] = ()
    ) -> PersonOfInterest:
        """Create a person of interest from a name and one photo; they exist only if the photo
        passes enrollment and every warning raised was acknowledged."""
        name = clean_name(name)
        photo = prepare_photo(upload)
        with self._change() as session:
            detection, embeddings = self._enrollable(photo)
            _require_acknowledged(
                [
                    *_duplicate_name(session, name),
                    *self._looks_like_other(session, embeddings, person_id=None),
                ],
                acknowledged,
            )
            now = self._clock()
            person = PersonOfInterestRow(
                id=_new_id(),
                name=name,
                name_key=name_key(name),
                status="on_watchlist",
                created_at=now,
                status_changed_at=now,
            )
            session.add(person)
            self._add_photo(session, person, photo, detection, embeddings)
            session.flush()
            return _person(person)

    def add_photo(
        self, person_id: str, upload: bytes, acknowledged: Iterable[WarningCode] = ()
    ) -> EnrolledPhoto:
        photo = prepare_photo(upload)
        with self._change() as session:
            person = _get_person(session, person_id)
            detection, embeddings = self._enrollable(photo)
            _require_acknowledged(
                [
                    *self._not_same_person(session, embeddings, person),
                    *self._looks_like_other(session, embeddings, person_id=person.id),
                ],
                acknowledged,
            )
            row = self._add_photo(session, person, photo, detection, embeddings)
            session.flush()
            return _photo(row)

    def photo_image(self, person_id: str, photo_id: str) -> Photo:
        with Session(self._engine) as session:
            return _stored_photo(_get_photo(session, person_id, photo_id))

    def delete_photo(self, person_id: str, photo_id: str) -> None:
        """Erase one enrolled photo and its embeddings; never a person's last photo."""
        with self._change() as session:
            row = _get_photo(session, person_id, photo_id)
            remaining = session.scalar(
                select(func.count()).where(EnrolledPhotoRow.person_id == person_id)
            )
            if remaining == 1:
                raise WatchlistError(
                    409,
                    "last_photo",
                    "A person of interest's last enrolled photo cannot be deleted.",
                )
            session.delete(row)

    def update_person(
        self, person_id: str, *, name: str | None = None, status: PersonStatus | None = None
    ) -> PersonChange:
        """Change a person of interest's name, status or both in one transaction; with neither,
        the person as they are. Removal ends their open sighting."""
        if name is None and status is None:
            return PersonChange(self.person(person_id))
        name = None if name is None else clean_name(name)
        with self._change() as session:
            person = _get_person(session, person_id)
            if name is not None:
                person.name = name
                person.name_key = name_key(name)
            ended: tuple[SightingAnnouncement, ...] = ()
            if status is not None and status != person.status:
                person.status = status
                person.status_changed_at = self._clock()
                if status == "removed":
                    ended = self._monitoring.end_person(session, person_id)
            session.flush()
            return PersonChange(_person(person), ended)

    def purge(self, person_id: str) -> tuple[SightingAnnouncement, ...]:
        """Erase a person of interest, on the watchlist or removed, for good.

        One transaction takes their enrolled photos, embeddings and sightings, and clears them
        as runner-up on everyone else's sightings, whose runner-up scores are kept. The foreign
        keys would do both; the runner-up is cleared here too so the rule does not rest on them
        alone. `secure_delete` overwrites the freed pages (ADR 0004).

        Their open sighting ends with it, announced with its last state but not written, as its
        row is gone; and the live monitor forgets them as runner-up, keeping the score.
        """
        with self._change() as session:
            person = sightings.sighting_person(_get_person(session, person_id))
            session.execute(
                update(SightingRow)
                .where(SightingRow.runner_up_person_id == person_id)
                .values(runner_up_person_id=None)
            )
            session.execute(delete(PersonOfInterestRow).where(PersonOfInterestRow.id == person_id))
            return self._monitoring.purge(person)

    def sightings(
        self,
        person_id: str | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> SightingPage:
        """Sightings newest first, a page at a time, of one person of interest if given; an
        unknown person has none."""
        with Session(self._engine) as session:
            return sightings.sighting_page(session, person_id=person_id, cursor=cursor, limit=limit)

    def sighting(self, sighting_id: str) -> Sighting:
        with Session(self._engine) as session:
            return sightings.get_sighting(session, sighting_id)

    def sighting_crop(self, sighting_id: str) -> bytes:
        """The JPEG face crop of a sighting's best match."""
        with Session(self._engine) as session:
            return sightings.sighting_crop(session, sighting_id)

    @contextmanager
    def _change(self) -> Iterator[Session]:
        """A transaction that changes the watchlist, made one at a time. The live monitor's
        embeddings of the watchlist are reloaded within it and replaced once it commits, so the
        next frame sees the change, and a failed reload rolls the change back, the live monitor's
        sightings with it."""
        with self._lock, self._monitoring.all_or_nothing():
            with Session(self._engine) as session, session.begin():
                yield session
                session.flush()
                embeddings = self._watchlist_embeddings(session)
            self._embeddings = embeddings

    def _watchlist_embeddings(self, session: Session) -> WatchlistEmbeddings:
        active = self.registry.active
        return WatchlistEmbeddings.load(session, None if active is None else active.key)

    def _enrollable(self, photo: Photo) -> tuple[Detection, dict[str, Embedding]]:
        """The photo's one usable face and its embedding under every loaded model, by key."""
        if self._detector is None:
            raise WatchlistError(
                503,
                "no_detector",
                "Enrollment needs the face detector's weights; fetch them with "
                "`ryuk weights fetch` and restart the service.",
            )
        pixels = photo.pixels()
        # Detected on a bounded copy, since YuNet misses a close-up's face at full resolution; the
        # box and landmarks come back in the stored photo's pixels, which the crop is cut from.
        detection = enrollable_face(self._detector.detect(pixels, max_side=MAX_DETECTION_SIDE))
        embeddings = {
            model.key.id: _embed(self._detector, pixels, detection, model, loaded)
            for model, loaded in self.registry.loaded()
        }
        return detection, embeddings

    def _add_photo(
        self,
        session: Session,
        person: PersonOfInterestRow,
        photo: Photo,
        detection: Detection,
        embeddings: dict[str, Embedding],
    ) -> EnrolledPhotoRow:
        row = EnrolledPhotoRow(
            id=_new_id(),
            image=photo.data,
            media_type=photo.media_type,
            width=photo.width,
            height=photo.height,
            face_box=[detection.box.x, detection.box.y, detection.box.width, detection.box.height],
            face_landmarks=[list(point) for point in detection.landmarks],
            face_score=detection.score,
            created_at=self._clock(),
        )
        row.embeddings = [_embedding_row(row.id, key, vector) for key, vector in embeddings.items()]
        person.photos.append(row)
        return row

    def _looks_like_other(
        self, session: Session, embeddings: dict[str, Embedding], person_id: str | None
    ) -> list[EnrollmentWarning]:
        """The warning when the photo's top candidate among other persons of interest, removed
        ones included, scores at or above the active model's threshold. Skipped when no model is
        active."""
        active = self.registry.active
        if active is None:
            return []
        query = (
            select(
                PersonOfInterestRow.id,
                PersonOfInterestRow.name,
                PersonOfInterestRow.status,
                EmbeddingRow.vector,
            )
            .join(EnrolledPhotoRow, EnrolledPhotoRow.person_id == PersonOfInterestRow.id)
            .join(EmbeddingRow, EmbeddingRow.photo_id == EnrolledPhotoRow.id)
            .where(EmbeddingRow.model_key == active.key.id)
        )
        if person_id is not None:
            query = query.where(PersonOfInterestRow.id != person_id)
        probe = embeddings[active.key.id]
        best: dict[str, tuple[float, str, str]] = {}
        for other_id, other_name, other_status, vector in session.execute(query):
            score = float(decode_embedding(vector) @ probe)
            if other_id not in best or score > best[other_id][0]:
                best[other_id] = (score, other_name, other_status)
        if not best:
            return []
        top_id, (score, top_name, top_status) = max(best.items(), key=lambda item: item[1][0])
        if score < active.evaluated.threshold:
            return []
        # Removed persons are compared too, as the duplicate-name warning does: a removal can be
        # undone, and the operator should know the face is already enrolled.
        removed = " (removed from the watchlist)" if top_status == "removed" else ""
        return [
            EnrollmentWarning(
                "looks_like_other",
                f"This photo looks like {top_name}{removed}, another person of interest "
                f"(score {score:.3f}, threshold {active.evaluated.threshold:.3f}).",
                top_id,
            )
        ]

    def _not_same_person(
        self, session: Session, embeddings: dict[str, Embedding], person: PersonOfInterestRow
    ) -> list[EnrollmentWarning]:
        """The warning when the photo scores below the threshold against all of the person's
        photos under the active model. Skipped when no model is active."""
        active = self.registry.active
        if active is None:
            return []
        vectors = session.scalars(
            select(EmbeddingRow.vector)
            .join(EnrolledPhotoRow, EmbeddingRow.photo_id == EnrolledPhotoRow.id)
            .where(
                EnrolledPhotoRow.person_id == person.id,
                EmbeddingRow.model_key == active.key.id,
            )
        ).all()
        if not vectors:
            return []
        probe = embeddings[active.key.id]
        score = max(float(decode_embedding(vector) @ probe) for vector in vectors)
        if score >= active.evaluated.threshold:
            return []
        return [
            EnrollmentWarning(
                "may_not_be_same_person",
                f"This photo may not be {person.name}: its best score against their photos is "
                f"{score:.3f}, under the threshold {active.evaluated.threshold:.3f}.",
                None,
            )
        ]


def enrollable_face(detections: Sequence[Detection]) -> Detection:
    """The one usable face an enrolled photo must have. Smaller faces, such as a bystander in
    the background, are ignored."""
    if not detections:
        raise WatchlistError(422, "no_face", "No face was found in the photo.")
    faces = usable(detections)
    if not faces:
        raise WatchlistError(
            422,
            "face_too_small",
            f"The face is too small to use: it must be at least {MIN_USABLE_FACE_SIZE} px "
            "across. Use a closer photo.",
        )
    if len(faces) > 1:
        raise WatchlistError(
            422,
            "multiple_faces",
            f"The photo has {len(faces)} faces large enough to use; enroll a photo of only this "
            "person.",
        )
    return faces[0]


def clean_name(name: str) -> str:
    """The name with surrounding space trimmed and inner runs of space collapsed.

    Whitespace, tabs and line breaks included, collapses to one space. Any other control
    character, and the Unicode controls that reorder text, are refused: they could hide or
    disguise a name wherever it is shown or logged. So is a name with nothing visible, such as
    only zero-width spaces; a direction mark inside a real name is kept.
    """
    cleaned = " ".join(name.split())
    if any(_is_forbidden_in_name(character) for character in cleaned):
        raise WatchlistError(
            422, "invalid_name", "A name cannot contain control or text-direction characters."
        )
    visible = "".join(c for c in cleaned if unicodedata.category(c) != "Cf")
    if not visible.strip():
        raise WatchlistError(422, "invalid_name", "A person of interest needs a name.")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise WatchlistError(
            422, "invalid_name", f"A name can be at most {MAX_NAME_LENGTH} characters."
        )
    return cleaned


_BIDI_CONTROLS: Final = frozenset(
    chr(code) for code in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))
)
"""The embeddings, overrides and isolates (LRE to RLO, LRI to PDI) that reorder displayed text."""


def _is_forbidden_in_name(character: str) -> bool:
    # Cc is exactly C0 (U+0000 to U+001F), DEL and C1 (U+0080 to U+009F); those that are
    # whitespace were already collapsed to spaces.
    return unicodedata.category(character) == "Cc" or character in _BIDI_CONTROLS


def name_key(name: str) -> str:
    """The name as the duplicate-name warning compares it: case and spaces ignored."""
    return "".join(unicodedata.normalize("NFKC", name).casefold().split())


def _duplicate_name(session: Session, name: str) -> list[EnrollmentWarning]:
    """The warning when another person of interest, removed ones included, has this name."""
    other = session.scalars(
        select(PersonOfInterestRow)
        .where(PersonOfInterestRow.name_key == name_key(name))
        .order_by(PersonOfInterestRow.created_at)
    ).first()
    if other is None:
        return []
    removed = " (removed from the watchlist)" if other.status == "removed" else ""
    return [
        EnrollmentWarning(
            "duplicate_name",
            f"A person of interest named {other.name}{removed} already exists.",
            other.id,
        )
    ]


def _require_acknowledged(
    warnings: list[EnrollmentWarning], acknowledged: Iterable[WarningCode]
) -> None:
    """Refuse with every warning raised unless the operator acknowledged each of them."""
    if {warning.code for warning in warnings} - set(acknowledged):
        raise UnacknowledgedWarningsError(warnings)


def _sync_embeddings(
    session: Session, detector: Detector | None, model: RegisteredModel, loaded: RecognitionModel
) -> None:
    """Drop embeddings made with other weights for this network and provider, with another crop
    or by another pipeline version, and embed every enrolled photo that has no embedding under
    this model yet."""
    key = model.key
    recorded = session.scalars(
        select(RecognitionModelRow).where(
            RecognitionModelRow.network == key.network,
            RecognitionModelRow.provider == key.provider,
        )
    ).all()
    for row in recorded:
        change = _stale_because(row, model)
        if change is not None:
            logger.warning("%s %s; rebuilding its embeddings from photos", model.name, change)
            session.delete(row)  # its embeddings go with it, by the foreign key's cascade
    # Deleted before the row is added again: re-adding a key in the same flush would update the
    # row in place, and its embeddings would survive.
    session.flush()
    if session.get(RecognitionModelRow, key.id) is None:
        session.add(
            RecognitionModelRow(
                model_key=key.id,
                network=key.network,
                weights_sha256=key.weights_sha256,
                provider=key.provider,
                dim=loaded.dimension,
                crop=model.crop,
                pipeline_version=PIPELINE_VERSION,
            )
        )
    session.flush()

    missing = session.scalars(
        select(EnrolledPhotoRow).where(
            ~exists().where(
                EmbeddingRow.photo_id == EnrolledPhotoRow.id, EmbeddingRow.model_key == key.id
            )
        )
    ).all()
    if not missing:
        return
    if detector is None:
        logger.warning(
            "%d photos have no %s embedding, but the detector's weights are missing",
            len(missing),
            model.name,
        )
        return
    for photo in missing:
        detection = Detection(
            Box(*photo.face_box),
            Landmarks(*((x, y) for x, y in photo.face_landmarks)),
            photo.face_score,
        )
        pixels = _stored_photo(photo).pixels()
        vector = _embed(detector, pixels, detection, model, loaded)
        session.add(_embedding_row(photo.id, key.id, vector))
    logger.info("Embedded %d photos under %s", len(missing), model.name)


def _stale_because(row: RecognitionModelRow, model: RegisteredModel) -> str | None:
    """What changed since the embeddings recorded under `row` were made for this network and
    provider, or None when `model` would make them the same way."""
    if row.model_key != model.key.id:
        return "weights changed"
    if row.crop != model.crop:
        return f"crop changed from {row.crop} to {model.crop}"
    if row.pipeline_version != PIPELINE_VERSION:
        return f"pipeline changed from version {row.pipeline_version} to {PIPELINE_VERSION}"
    return None


def _choose_active(
    session: Session, models: Sequence[RegisteredModel], evaluation: Evaluation
) -> ModelKey | None:
    """The persisted active model if it can still be active, else evaluation's first active
    model among those that can; None when no model can be active.

    Only evaluation's own first active model is persisted, on the first start it can run. A
    fallback, such as SFace where ArcFace on CoreML cannot run or before its weights are
    fetched, is active for this run only, so a later start with those weights applies
    evaluation's choice. A persisted choice that cannot run now is kept, not overwritten, so it
    applies again once its weights are back.
    """
    usable_keys = {model.key.id: model.key for model in models if model.can_be_active}
    stored = session.get(SettingRow, ACTIVE_MODEL_SETTING)
    if stored is not None and stored.value in usable_keys:
        return usable_keys[stored.value]
    first = evaluation.first_active_for(usable_keys.values())
    if stored is None and first is not None and first == evaluation.first_active:
        session.add(SettingRow(key=ACTIVE_MODEL_SETTING, value=first.id))
    return first


def _embed(
    detector: Detector,
    pixels: Image,
    detection: Detection,
    model: RegisteredModel,
    loaded: RecognitionModel,
) -> Embedding:
    return loaded.embed(face_crop(detector, pixels, detection, model.crop, loaded.input_size))


def _embedding_row(photo_id: str, model_key: str, vector: Embedding) -> EmbeddingRow:
    return EmbeddingRow(
        photo_id=photo_id,
        model_key=model_key,
        dim=int(vector.shape[0]),
        vector=encode_embedding(vector),
    )


def _get_person(session: Session, person_id: str) -> PersonOfInterestRow:
    person = session.get(PersonOfInterestRow, person_id)
    if person is None:
        raise not_found("person of interest")
    return person


def _get_photo(session: Session, person_id: str, photo_id: str) -> EnrolledPhotoRow:
    photo = session.get(EnrolledPhotoRow, photo_id)
    if photo is None or photo.person_id != person_id:
        raise not_found("enrolled photo of this person of interest")
    return photo


def _stored_photo(row: EnrolledPhotoRow) -> Photo:
    return Photo(row.image, row.media_type, row.width, row.height)


def _person(row: PersonOfInterestRow) -> PersonOfInterest:
    return PersonOfInterest(
        id=row.id,
        name=row.name,
        status=cast(PersonStatus, row.status),  # the table's check constraint holds it to these
        created_at=row.created_at,
        status_changed_at=row.status_changed_at,
        photos=tuple(_photo(photo) for photo in row.photos),
    )


def _photo(row: EnrolledPhotoRow) -> EnrolledPhoto:
    return EnrolledPhoto(row.id, row.media_type, row.width, row.height, row.created_at)


def _new_id() -> str:
    return uuid.uuid4().hex
