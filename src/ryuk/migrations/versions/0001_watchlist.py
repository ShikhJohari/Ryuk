"""Persons of interest, enrolled photos, embeddings, recognition models, settings.

Revision ID: 0001
Revises:
Create Date: 2026-09-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "person_of_interest",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("name_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("status_changed_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "status IN ('on_watchlist', 'removed')", name=op.f("ck_person_of_interest_status")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_person_of_interest")),
    )
    op.create_index(
        op.f("ix_person_of_interest_name_key"), "person_of_interest", ["name_key"], unique=False
    )
    op.create_table(
        "recognition_model",
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("network", sa.String(), nullable=False),
        sa.Column("weights_sha256", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("crop", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("model_key", name=op.f("pk_recognition_model")),
    )
    op.create_table(
        "setting",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_setting")),
    )
    op.create_table(
        "enrolled_photo",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("person_id", sa.String(), nullable=False),
        sa.Column("image", sa.LargeBinary(), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("face_box", sa.JSON(), nullable=False),
        sa.Column("face_landmarks", sa.JSON(), nullable=False),
        sa.Column("face_score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["person_of_interest.id"],
            name=op.f("fk_enrolled_photo_person_id_person_of_interest"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_enrolled_photo")),
    )
    op.create_index(
        op.f("ix_enrolled_photo_person_id"), "enrolled_photo", ["person_id"], unique=False
    )
    op.create_table(
        "embedding",
        sa.Column("photo_id", sa.String(), nullable=False),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_key"],
            ["recognition_model.model_key"],
            name=op.f("fk_embedding_model_key_recognition_model"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["photo_id"],
            ["enrolled_photo.id"],
            name=op.f("fk_embedding_photo_id_enrolled_photo"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("photo_id", "model_key", name=op.f("pk_embedding")),
    )


def downgrade() -> None:
    op.drop_table("embedding")
    op.drop_index(op.f("ix_enrolled_photo_person_id"), table_name="enrolled_photo")
    op.drop_table("enrolled_photo")
    op.drop_table("setting")
    op.drop_table("recognition_model")
    op.drop_index(op.f("ix_person_of_interest_name_key"), table_name="person_of_interest")
    op.drop_table("person_of_interest")
