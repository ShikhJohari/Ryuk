"""The committed EDA files: `summary.json` and the JSON schema CI validates it against."""

import json
from pathlib import Path
from typing import Final

from ryuk.eda.summary import EdaSummary

SUMMARY_FILE: Final = "summary.json"
SCHEMA_FILE: Final = "summary.schema.json"
FIGURES_DIR: Final = "figures"


def summary_json(summary: EdaSummary) -> str:
    return summary.model_dump_json(indent=2) + "\n"


def schema_json() -> str:
    """The summary's JSON schema, generated from `EdaSummary`; the same text on every run."""
    return json.dumps(EdaSummary.model_json_schema(), indent=2) + "\n"


def write_summary(summary: EdaSummary, directory: Path) -> list[Path]:
    """Write the summary and its schema into `directory`, each renamed into place when whole."""
    directory.mkdir(parents=True, exist_ok=True)
    return [_write(directory / SUMMARY_FILE, summary_json(summary)), write_schema(directory)]


def write_schema(directory: Path) -> Path:
    """Write the schema alone. It comes from the code, not the data: the models' docstrings are
    its descriptions, so editing one makes the committed schema stale."""
    directory.mkdir(parents=True, exist_ok=True)
    return _write(directory / SCHEMA_FILE, schema_json())


def read_summary(directory: Path) -> EdaSummary:
    """The summary written into `directory`, validated."""
    return EdaSummary.model_validate_json((directory / SUMMARY_FILE).read_bytes())


def _write(path: Path, text: str) -> Path:
    part = path.with_name(f"{path.name}.part")
    try:
        part.write_text(text)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path
