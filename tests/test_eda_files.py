"""The committed `eda/` files: a current schema, a summary that passes it, and a detector that
uses the minimum face size the summary measured."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ryuk.detector import MIN_USABLE_FACE_SIZE
from ryuk.eda.files import (
    SCHEMA_FILE,
    SUMMARY_FILE,
    read_summary,
    schema_json,
    summary_json,
    write_schema,
)

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
