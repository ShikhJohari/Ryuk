"""The committed `eda/` files: a current schema, a summary that passes it, and a detector that
uses the minimum face size the summary measured."""

import errno
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from ryuk.detector import MIN_USABLE_FACE_SIZE
from ryuk.eda.files import (
    FIGURES_DIR,
    SCHEMA_FILE,
    SUMMARY_FILE,
    read_summary,
    schema_json,
    summary_json,
    write_eda,
    write_from_summary,
    write_schema,
)
from ryuk.fetch import FetchError
from ryuk.plotting.eda import EDA_FIGURES

EDA = Path(__file__).parents[1] / "eda"


def test_the_committed_schema_is_generated_from_the_model() -> None:
    # Stale after any change to ryuk.eda.summary, docstrings included: rerun `ryuk eda`, or
    # `ryuk eda --figures-only` if the summary itself still parses.
    assert (EDA / SCHEMA_FILE).read_text() == schema_json()


def test_the_committed_summary_passes_the_committed_schema() -> None:
    schema = json.loads((EDA / SCHEMA_FILE).read_text())
    Draft202012Validator.check_schema(schema)

    errors = list(
        Draft202012Validator(schema).iter_errors(json.loads((EDA / SUMMARY_FILE).read_text()))
    )

    assert errors == []


def test_the_committed_summary_is_as_ryuk_eda_writes_it() -> None:
    summary = read_summary(EDA)

    assert (EDA / SUMMARY_FILE).read_text() == summary_json(summary)
    assert [draw.draw for draw in summary.celeba] == ["validation", "test"]


def test_the_detector_uses_the_measured_minimum_face_size() -> None:
    assert read_summary(EDA).min_usable_face_size.value == MIN_USABLE_FACE_SIZE


def test_the_schema_rejects_what_the_model_rejects() -> None:
    validator = Draft202012Validator(json.loads(schema_json()))
    summary = json.loads((EDA / SUMMARY_FILE).read_text())
    summary["schema_version"] = 2
    summary["lfw"]["detection"]["unexpected"] = 1
    del summary["min_usable_face_size"]["kept"]

    problems = {error.message for error in validator.iter_errors(summary)}

    assert len(problems) == 3


def test_the_schema_can_be_written_alone(tmp_path: Path) -> None:
    path = write_schema(tmp_path / "eda")

    assert path == tmp_path / "eda" / SCHEMA_FILE
    assert path.read_text() == schema_json()


def test_a_failed_write_leaves_the_file_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / SCHEMA_FILE).write_text("the schema before")
    write_text = Path.write_text

    def disk_full(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
        write_text(path, data[:10], *args, **kwargs)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_text", disk_full)

    with pytest.raises(FetchError, match=f"writing {SCHEMA_FILE} failed"):
        write_schema(tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == [SCHEMA_FILE]
    assert (tmp_path / SCHEMA_FILE).read_text() == "the schema before"


def _figure_files(directory: Path) -> list[Path]:
    return [directory / FIGURES_DIR / f"{slug}.svg" for slug in EDA_FIGURES]


def test_ryuk_eda_writes_the_summary_its_schema_and_the_figures(tmp_path: Path) -> None:
    summary = read_summary(EDA)

    written = write_eda(summary, tmp_path / "eda")

    directory = tmp_path / "eda"
    assert written == [directory / SUMMARY_FILE, directory / SCHEMA_FILE, *_figure_files(directory)]
    assert read_summary(directory) == summary
    assert not list(directory.rglob("*.part"))


def test_figures_only_rewrites_the_schema_and_figures_from_the_summary(tmp_path: Path) -> None:
    # A summary on its own, as after editing a model's docstring or a figure: the stale schema
    # and figures are rewritten, the summary is read and left as it was.
    directory = tmp_path / "eda"
    directory.mkdir()
    shutil.copy(EDA / SUMMARY_FILE, directory / SUMMARY_FILE)
    (directory / SCHEMA_FILE).write_text("stale")
    before = (directory / SUMMARY_FILE).read_bytes()

    written = write_from_summary(directory)

    assert written == [directory / SCHEMA_FILE, *_figure_files(directory)]
    assert (directory / SCHEMA_FILE).read_text() == schema_json()
    assert (directory / SUMMARY_FILE).read_bytes() == before
    assert all(path.read_text().lstrip().startswith("<?xml") for path in written[1:])
    assert not list(directory.rglob("*.part"))

    # Drawn from the same summary, the figures are those `ryuk eda` writes, byte for byte.
    again = write_eda(read_summary(directory), tmp_path / "again")
    assert [path.read_bytes() for path in again[2:]] == [path.read_bytes() for path in written[1:]]


def test_figures_only_needs_a_summary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_from_summary(tmp_path)

    assert list(tmp_path.iterdir()) == []
