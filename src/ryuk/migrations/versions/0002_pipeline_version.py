"""Record the pipeline version each recognition model's embeddings were made by.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FIRST_VERSION = "1"
"""The pipeline every embedding stored before this migration was made by. A literal, never
`PIPELINE_VERSION`: a migration must do the same thing whenever it runs."""


def upgrade() -> None:
    with op.batch_alter_table("recognition_model", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "pipeline_version",
                sa.Integer(),
                nullable=False,
                server_default=_FIRST_VERSION,
            )
        )
    # The default only fills the rows already there; the service names the version of every row
    # it adds, so the table is rebuilt without it.
    with op.batch_alter_table("recognition_model", schema=None, recreate="always") as batch_op:
        batch_op.alter_column("pipeline_version", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("recognition_model", schema=None) as batch_op:
        batch_op.drop_column("pipeline_version")
