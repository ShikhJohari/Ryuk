"""Alembic's entry point. The service passes its own connection; the `alembic` command line
(run from the repository root, see `alembic.ini`) opens the database named by `RYUK_DATABASE`."""

from alembic import context

from ryuk.settings import Settings
from ryuk.watchlist.database import sqlite_engine
from ryuk.watchlist.tables import Base

connection = context.config.attributes.get("connection")
if connection is not None:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
else:
    path = Settings().database
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite_engine(f"sqlite:///{path}").begin() as opened:
        context.configure(connection=opened, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
