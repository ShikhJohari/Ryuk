"""Sightings: a person of interest seen in the live monitor, with the best match's face crop.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # model_key is deliberately not a foreign key: the recognition_model row is deleted when its
    # embeddings are rebuilt, and sightings must outlive that.
    op.create_table(
        "sighting",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("person_id", sa.String(), nullable=False),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("best_score", sa.Float(), nullable=False),
        sa.Column("best_crop", sa.LargeBinary(), nullable=False),
        sa.Column("runner_up_person_id", sa.String(), nullable=True),
        sa.Column("runner_up_score", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["person_of_interest.id"],
            name=op.f("fk_sighting_person_id_person_of_interest"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["runner_up_person_id"],
            ["person_of_interest.id"],
            name=op.f("fk_sighting_runner_up_person_id_person_of_interest"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sighting")),
    )
    op.create_index(op.f("ix_sighting_person_id"), "sighting", ["person_id"], unique=False)
    op.create_index(
        op.f("ix_sighting_runner_up_person_id"), "sighting", ["runner_up_person_id"], unique=False
    )
    op.create_index("ix_sighting_started_at", "sighting", ["started_at", "id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_sighting_started_at", table_name="sighting")
    op.drop_index(op.f("ix_sighting_runner_up_person_id"), table_name="sighting")
    op.drop_index(op.f("ix_sighting_person_id"), table_name="sighting")
    op.drop_table("sighting")
