"""Opening the app database and bringing its schema to the latest migration."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, event

MIGRATIONS = "ryuk:migrations"


class ForeignKeyViolationError(RuntimeError):
    """A migration left rows whose foreign keys point at nothing; it was rolled back."""


def open_database(path: Path) -> Engine:
    """An engine on the SQLite file at `path`, created if missing, migrated to the latest schema.

    Every connection runs with foreign keys enforced, so deletes cascade, with `secure_delete`,
    so erased face data is overwritten rather than left in free pages (ADR 0004), and with a
    rollback journal that is emptied at commit, so it keeps no copy of what a delete erased.

    The file holds face photos, so only its owner may read it; SQLite gives its journal the
    same mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600)
    path.chmod(0o600)  # touch leaves an existing file's mode as it was
    engine = sqlite_engine(f"sqlite:///{path}")
    migrate(engine)
    return engine


def sqlite_engine(url: str) -> Engine:
    """An engine whose transactions are real SQLite transactions, DDL included.

    Left to itself, Python's sqlite3 begins a transaction only before a data change, so a
    `CREATE TABLE` runs outside it and survives a rollback. SQLAlchemy's documented recipe
    takes that job over: the driver never begins on its own and every SQLAlchemy transaction
    starts with an explicit `BEGIN`.
    """
    # Requests run on worker threads; each checks a connection out of the pool for itself. A
    # failed statement's parameters stay out of its error, which is logged: they can be a name or
    # a photo's bytes.
    engine = create_engine(url, connect_args={"check_same_thread": False}, hide_parameters=True)
    event.listen(engine, "connect", _on_connect)
    event.listen(engine, "begin", _begin)
    return engine


def migrate(engine: Engine) -> None:
    """Upgrade the schema to the latest migration, the only way the schema is created."""
    config = Config()
    config.set_main_option("script_location", MIGRATIONS)
    with migration_transaction(engine) as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


@contextmanager
def migration_transaction(engine: Engine) -> Iterator[Connection]:
    """A connection in one transaction with foreign keys off, for migrations to run in.

    SQLite rebuilds a table to alter it (Alembic's batch mode), and dropping the old table is a
    delete that, with foreign keys on, cascades to every enrolled photo and embedding beneath
    it; `secure_delete` then makes the loss unrecoverable (ADR 0004). So foreign keys are off
    for the whole migration, as SQLite's documentation prescribes for a table rebuild, and
    `PRAGMA foreign_key_check` stands in for them before commit: any row left pointing at
    nothing rolls the whole migration back. The connection goes back to the pool with foreign
    keys on, or not at all.
    """
    with engine.connect() as connection:
        # The pragma is a no-op inside a transaction, so it goes first.
        if _set_foreign_keys(connection, enabled=False):
            connection.invalidate()
            raise RuntimeError("could not turn foreign keys off to migrate the database")
        try:
            with connection.begin():
                yield connection
                _check_foreign_keys(connection)
        finally:
            restored = False
            try:
                restored = _set_foreign_keys(connection, enabled=True)
            finally:
                if not restored:
                    # A connection that is not enforcing them must never serve a request.
                    connection.invalidate()


def _check_foreign_keys(connection: Connection) -> None:
    violations = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
    if violations:
        table, rowid, parent, _ = violations[0]
        raise ForeignKeyViolationError(
            f"the migration left {len(violations)} row(s) referring to rows that do not exist"
            f" (first: {table} row {rowid} -> {parent}); it was rolled back"
        )


def _set_foreign_keys(connection: Connection, *, enabled: bool) -> bool:
    """Set `PRAGMA foreign_keys` and return whether they are now on.

    Straight on the driver's connection: through SQLAlchemy the statement would begin a
    transaction first, and inside one SQLite ignores it.
    """
    cursor = connection.connection.cursor()
    try:
        cursor.execute(f"PRAGMA foreign_keys = {'ON' if enabled else 'OFF'}")
        cursor.execute("PRAGMA foreign_keys")
        row = cursor.fetchone()
    finally:
        cursor.close()
    return row is not None and row[0] == 1


def _on_connect(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN401 - DB-API objects
    # sqlite3 never begins a transaction by itself; `_begin` does. With no transaction open at
    # connect time, every pragma takes effect.
    dbapi_connection.isolation_level = None
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA secure_delete = ON")
    # A delete copies the pages it zeroes, a deleted photo's included, to the rollback journal.
    # TRUNCATE empties the journal at commit, so no file keeps them; PERSIST would leave them in
    # it, and WAL keeps every photo's inserted pages in the WAL until they are overwritten. The
    # disk blocks the journal freed are the filesystem's to reuse, beyond SQLite's reach.
    cursor.execute("PRAGMA journal_mode = TRUNCATE")
    cursor.close()


def _begin(connection: Connection) -> None:
    connection.exec_driver_sql("BEGIN")
