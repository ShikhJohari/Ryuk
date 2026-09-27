"""Opening the app database and bringing its schema to the latest migration."""

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event

MIGRATIONS = "ryuk:migrations"


def open_database(path: Path) -> Engine:
    """An engine on the SQLite file at `path`, created if missing, migrated to the latest schema.

    Every connection runs with foreign keys enforced, so deletes cascade, and with
    `secure_delete`, so erased face data is overwritten rather than left in free pages (ADR 0004).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqlite_engine(f"sqlite:///{path}")
    migrate(engine)
    return engine


def sqlite_engine(url: str) -> Engine:
    # Requests run on worker threads; each checks a connection out of the pool for itself.
    engine = create_engine(url, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _enable_pragmas)
    return engine


def migrate(engine: Engine) -> None:
    """Upgrade the schema to the latest migration, the only way the schema is created."""
    config = Config()
    config.set_main_option("script_location", MIGRATIONS)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


def _enable_pragmas(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN401 - DB-API objects
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA secure_delete = ON")
    cursor.close()
