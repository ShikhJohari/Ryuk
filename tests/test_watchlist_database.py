"""The app database's connections and migrations: pragmas on every connection, DDL that rolls
back, and migrations run with foreign keys off so a table rebuild cannot cascade (ADR 0004)."""

import os
import stat
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Connection, Engine, event, inspect
from sqlalchemy.exc import IntegrityError

import ryuk.migrations
from ryuk.watchlist.database import (
    MIGRATIONS,
    ForeignKeyViolationError,
    migrate,
    migration_transaction,
    open_database,
    sqlite_engine,
)

TABLES = {
    "alembic_version",
    "person_of_interest",
    "enrolled_photo",
    "embedding",
    "recognition_model",
    "setting",
}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    opened = open_database(tmp_path / "ryuk.sqlite3")
    yield opened
    opened.dispose()


ENFORCED = (1, 1, "truncate")
"""Foreign keys and secure_delete on, and a rollback journal truncated at commit."""


def pragmas(connection: Connection) -> tuple[Any, Any, Any]:
    return (
        connection.exec_driver_sql("PRAGMA foreign_keys").scalar(),
        connection.exec_driver_sql("PRAGMA secure_delete").scalar(),
        connection.exec_driver_sql("PRAGMA journal_mode").scalar(),
    )


def add_person(connection: Connection, person_id: str) -> None:
    connection.exec_driver_sql(
        "INSERT INTO person_of_interest (id, name, name_key, status, created_at,"
        " status_changed_at) VALUES (?, 'Ada', 'ada', 'on_watchlist', '2026-09-26 00:00:00',"
        " '2026-09-26 00:00:00')",
        (person_id,),
    )


def add_photo(
    connection: Connection, photo_id: str, person_id: str, image: bytes = b"\x00"
) -> None:
    connection.exec_driver_sql(
        "INSERT INTO enrolled_photo (id, person_id, image, media_type, width, height, face_box,"
        " face_landmarks, face_score, created_at) VALUES (?, ?, ?, 'image/jpeg', 1, 1,"
        " '[0, 0, 1, 1]', '[]', 0.9, '2026-09-26 00:00:00')",
        (photo_id, person_id, image),
    )


def fail_halfway(connection: Connection, *statements: str) -> None:
    for statement in statements:
        connection.exec_driver_sql(statement)
    raise RuntimeError("halfway")


def count(connection: Connection, table: str) -> Any:
    return connection.exec_driver_sql(f"SELECT count(*) FROM {table}").scalar()  # noqa: S608


def test_a_fresh_database_is_migrated_to_every_table(tmp_path: Path) -> None:
    engine = open_database(tmp_path / "nested" / "ryuk.sqlite3")

    with engine.connect() as connection:
        names = connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'table'")
        assert set(names.scalars()) == TABLES
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar()
    engine.dispose()


def test_every_connection_enforces_foreign_keys_secure_delete_and_the_journal_mode(
    engine: Engine,
) -> None:
    with engine.connect() as first, engine.connect() as second:
        assert pragmas(first) == ENFORCED
        assert pragmas(second) == ENFORCED

    seen: list[tuple[Any, Any, Any]] = []

    def check() -> None:
        with engine.connect() as connection:
            seen.append(pragmas(connection))

    worker = threading.Thread(target=check)
    worker.start()
    worker.join()
    assert seen == [ENFORCED]


def test_the_journal_keeps_no_copy_of_a_deleted_photo_once_the_delete_commits(
    engine: Engine, tmp_path: Path
) -> None:
    image = os.urandom(200_000)  # spans many overflow pages, and appears nowhere else
    journal = tmp_path / "ryuk.sqlite3-journal"

    def holding_it() -> list[str]:
        piece = image[100_000:100_064]
        return sorted(p.name for p in tmp_path.glob("ryuk.sqlite3*") if piece in p.read_bytes())

    with engine.begin() as connection:
        add_person(connection, "p1")
        add_photo(connection, "ph1", "p1", image)
    assert holding_it() == ["ryuk.sqlite3"]

    with engine.begin() as connection:
        connection.exec_driver_sql("DELETE FROM enrolled_photo WHERE id = 'ph1'")
        # To zero the photo's pages, the delete first copies them to the rollback journal.
        assert "ryuk.sqlite3-journal" in holding_it()

    # At commit the journal is emptied in place: not kept with the pages in it (PERSIST), and
    # not unlinked (DELETE, the default), which leaves them just as unreachable to SQLite.
    assert journal.is_file()
    assert journal.stat().st_size == 0
    assert holding_it() == []


def test_the_database_and_its_journal_are_readable_by_their_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "ryuk.sqlite3"
    engine = open_database(path)
    with engine.begin() as connection:
        add_person(connection, "p1")
    engine.dispose()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "ryuk.sqlite3-journal").stat().st_mode) == 0o600


def test_a_database_made_readable_to_others_is_made_private_again(tmp_path: Path) -> None:
    path = tmp_path / "ryuk.sqlite3"
    open_database(path).dispose()
    path.chmod(0o644)

    open_database(path).dispose()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_failed_statement_does_not_log_its_parameters(engine: Engine) -> None:
    with engine.begin() as connection:
        add_person(connection, "p1")
        add_photo(connection, "ph1", "p1", b"face bytes")

    # A second photo with the same ID fails; the error SQLAlchemy raises, which the service logs,
    # must not carry the photo bytes, or a name, sent with the statement.
    with engine.connect() as connection, pytest.raises(IntegrityError) as failed:
        add_photo(connection, "ph1", "p1", b"face bytes")

    assert "face bytes" not in str(failed.value)
    assert "parameters hidden" in str(failed.value)


def test_the_connection_that_ran_the_migration_goes_back_enforcing_foreign_keys(
    tmp_path: Path,
) -> None:
    engine = sqlite_engine(f"sqlite:///{tmp_path / 'ryuk.sqlite3'}")
    checked_out: list[object] = []
    event.listen(engine, "checkout", lambda dbapi, *_: checked_out.append(dbapi))
    migrate(engine)
    [migrating] = set(map(id, checked_out))

    with engine.connect() as connection:
        # Reused from the pool, not a new connection.
        assert id(connection.connection.dbapi_connection) == migrating
        assert pragmas(connection) == ENFORCED
    engine.dispose()


def test_ddl_rolls_back_with_its_transaction(tmp_path: Path) -> None:
    engine = sqlite_engine(f"sqlite:///{tmp_path / 'ryuk.sqlite3'}")

    with pytest.raises(RuntimeError, match="halfway"), engine.begin() as connection:
        fail_halfway(connection, "CREATE TABLE t (x)")

    with engine.connect() as connection:
        assert not inspect(connection).has_table("t")
    engine.dispose()


def test_a_migration_that_fails_halfway_leaves_nothing_behind(engine: Engine) -> None:
    with pytest.raises(RuntimeError, match="halfway"), migration_transaction(engine) as c:
        fail_halfway(
            c, "CREATE TABLE half_done (x)", "UPDATE alembic_version SET version_num = 'next'"
        )

    with engine.connect() as connection:
        assert not inspect(connection).has_table("half_done")
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar() != (
            "next"
        )
        assert pragmas(connection) == ENFORCED


def test_migrations_run_with_foreign_keys_off_and_restore_them(engine: Engine) -> None:
    with migration_transaction(engine) as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 0

    with engine.connect() as connection:
        assert pragmas(connection) == ENFORCED


def test_rebuilding_a_table_during_a_migration_keeps_the_rows_beneath_it(engine: Engine) -> None:
    with engine.begin() as connection:
        add_person(connection, "p1")
        add_photo(connection, "f1", "p1")

    # The rebuild Alembic's batch mode does to alter a table on SQLite: copy, drop, rename.
    with migration_transaction(engine) as connection:
        operations = Operations(MigrationContext.configure(connection))
        with operations.batch_alter_table("person_of_interest", recreate="always"):
            pass

    with engine.connect() as connection:
        assert (count(connection, "person_of_interest"), count(connection, "enrolled_photo")) == (
            1,
            1,
        )


def test_a_migration_that_breaks_a_foreign_key_is_rolled_back(engine: Engine) -> None:
    with (
        pytest.raises(
            ForeignKeyViolationError, match=r"enrolled_photo row \d+ -> person_of_interest"
        ),
        migration_transaction(engine) as connection,
    ):
        add_photo(connection, "orphan", "nobody")

    with engine.connect() as connection:
        assert count(connection, "enrolled_photo") == 0
        assert pragmas(connection) == ENFORCED


def test_the_alembic_command_line_migrates_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "cli" / "ryuk.sqlite3"
    monkeypatch.setenv("RYUK_DATABASE", str(database))
    config = Config()
    config.set_main_option("script_location", MIGRATIONS)

    command.upgrade(config, "head")

    engine = sqlite_engine(f"sqlite:///{database}")
    with engine.connect() as connection:
        assert set(inspect(connection).get_table_names()) == TABLES
    engine.dispose()


def test_autogenerate_writes_batch_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "ryuk.sqlite3"
    open_database(database).dispose()
    engine = sqlite_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX ix_person_of_interest_name_key")
    engine.dispose()
    monkeypatch.setenv("RYUK_DATABASE", str(database))
    versions = tmp_path / "versions"
    versions.mkdir()
    config = Config()
    config.set_main_option("script_location", MIGRATIONS)
    config.set_main_option("path_separator", "os")
    # The committed migrations, and a scratch directory for the one written here.
    committed = Path(ryuk.migrations.__file__).parent / "versions"
    config.set_main_option("version_locations", os.pathsep.join([str(committed), str(versions)]))

    command.revision(config, "restore the index", autogenerate=True, version_path=str(versions))

    [written] = versions.glob("*.py")
    assert "with op.batch_alter_table('person_of_interest'" in written.read_text()
