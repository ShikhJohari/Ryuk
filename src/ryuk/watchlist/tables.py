"""The app database's tables (#12). Alembic migrations create them; nothing else does.

Enrolled photos and embeddings are BLOBs in the rows that own them, so a purge is one cascading
transaction (ADR 0004). IDs are opaque strings; timestamps are UTC.
"""

import datetime
from typing import Final, Literal

import numpy as np
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Dialect,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    String,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ryuk.recognition import Embedding

type PersonStatus = Literal["on_watchlist", "removed"]

_VECTOR_DTYPE: Final = np.dtype("<f4")


class UtcDateTime(TypeDecorator[datetime.datetime]):
    """A timezone-aware UTC datetime. SQLite has no timezones, so it is stored naive in UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: datetime.datetime | None, dialect: Dialect
    ) -> datetime.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(datetime.UTC).replace(tzinfo=None)

    def process_result_value(
        self, value: datetime.datetime | None, dialect: Dialect
    ) -> datetime.datetime | None:
        return None if value is None else value.replace(tzinfo=datetime.UTC)


class Base(DeclarativeBase):
    # Named constraints, so a later migration can drop or alter them by name.
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class PersonOfInterestRow(Base):
    __tablename__ = "person_of_interest"
    __table_args__ = (CheckConstraint("status IN ('on_watchlist', 'removed')", name="status"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    name_key: Mapped[str] = mapped_column(String, index=True)
    """The name folded for the duplicate-name warning: case and spaces ignored."""
    status: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime.datetime] = mapped_column(UtcDateTime)
    status_changed_at: Mapped[datetime.datetime] = mapped_column(UtcDateTime)

    photos: Mapped[list["EnrolledPhotoRow"]] = relationship(
        back_populates="person",
        order_by="EnrolledPhotoRow.created_at",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class EnrolledPhotoRow(Base):
    __tablename__ = "enrolled_photo"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    person_id: Mapped[str] = mapped_column(
        ForeignKey("person_of_interest.id", ondelete="CASCADE"), index=True
    )
    image: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    """Encoded with orientation applied and every metadata field stripped."""
    media_type: Mapped[str] = mapped_column(String)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    face_box: Mapped[list[float]] = mapped_column(JSON)
    """The enrolled face's box as [x, y, width, height], so rebuilding reuses the detection."""
    face_landmarks: Mapped[list[list[float]]] = mapped_column(JSON)
    """Its five landmarks as [x, y] pairs, in YuNet's order."""
    face_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(UtcDateTime)

    person: Mapped[PersonOfInterestRow] = relationship(back_populates="photos")
    embeddings: Mapped[list["EmbeddingRow"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class RecognitionModelRow(Base):
    """A recognition model some embeddings were made with (network, weights, provider), the crop
    they were cut with and the pipeline version that made them."""

    __tablename__ = "recognition_model"

    model_key: Mapped[str] = mapped_column(String, primary_key=True)
    network: Mapped[str] = mapped_column(String)
    weights_sha256: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    dim: Mapped[int] = mapped_column(Integer)
    crop: Mapped[str] = mapped_column(String)
    """The crop its embeddings were cut with; another crop means they are rebuilt."""
    pipeline_version: Mapped[int] = mapped_column(Integer)
    """`PIPELINE_VERSION` when its embeddings were made; another version means they are rebuilt."""


class EmbeddingRow(Base):
    """One enrolled photo's embedding under one recognition model: float32, L2-normalised."""

    __tablename__ = "embedding"

    photo_id: Mapped[str] = mapped_column(
        ForeignKey("enrolled_photo.id", ondelete="CASCADE"), primary_key=True
    )
    model_key: Mapped[str] = mapped_column(
        ForeignKey("recognition_model.model_key", ondelete="CASCADE"), primary_key=True
    )
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    """The embedding as little-endian float32; see `encode_embedding`."""


class SettingRow(Base):
    __tablename__ = "setting"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String)


def encode_embedding(vector: Embedding) -> bytes:
    """An embedding as `EmbeddingRow.vector` stores it."""
    return np.asarray(vector, dtype=_VECTOR_DTYPE).tobytes()


def decode_embedding(blob: bytes) -> Embedding:
    return np.frombuffer(blob, dtype=_VECTOR_DTYPE).astype(np.float32)
