"""Alembic's entry point. The service passes its own connection; the `alembic` command line
(run from the repository root, see `alembic.ini`) opens the database named by `RYUK_DATABASE`
and migrates it the same way, in one transaction with foreign keys off (ADR 0004)."""

from alembic import context
from sqlalchemy import Connection

from ryuk.settings import Settings
from ryuk.watchlist.database import migration_transaction, sqlite_engine
from ryuk.watchlist.tables import Base


def run_migrations(connection: Connection) -> None:
    # SQLite cannot alter most of a table in place, so autogenerate writes batch operations,
    # which rebuild the table instead.
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


passed = context.config.attributes.get("connection")
if passed is not None:
    run_migrations(passed)
else:
    path = Settings().database
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqlite_engine(f"sqlite:///{path}")
    try:
        with migration_transaction(engine) as opened:
            run_migrations(opened)
    finally:
        engine.dispose()
