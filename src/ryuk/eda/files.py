"""The committed EDA files: `summary.json`, the JSON schema CI validates it against, and the
figures drawn from it.

Every file is written beside its target and renamed into place once whole, as fetched files
are (`ryuk.fetch.pinned.write_into_place`), so a failed run never leaves a half-written file.
"""

import json
from pathlib import Path
from typing import Final

from ryuk.eda.summary import EdaSummary
from ryuk.fetch.pinned import write_into_place
from ryuk.plotting.eda import save_eda_figures

SUMMARY_FILE: Final = "summary.json"
SCHEMA_FILE: Final = "summary.schema.json"
FIGURES_DIR: Final = "figures"


def summary_json(summary: EdaSummary) -> str:
    return summary.model_dump_json(indent=2) + "\n"


def schema_json() -> str:
    """The summary's JSON schema, generated from `EdaSummary`; the same text on every run."""
    return json.dumps(EdaSummary.model_json_schema(), indent=2) + "\n"


def write_eda(summary: EdaSummary, directory: Path) -> list[Path]:
    """Write everything `ryuk eda` commits into `directory`: the summary, its schema, and the
    figures in `FIGURES_DIR`."""
    return [*write_summary(summary, directory), *save_eda_figures(summary, directory / FIGURES_DIR)]


def write_from_summary(directory: Path) -> list[Path]:
    """Rewrite the schema and the figures from the summary already in `directory`, which is
    left as it is. Needs no data: for when the models' docstrings or the figures change."""
    summary = read_summary(directory)
    return [write_schema(directory), *save_eda_figures(summary, directory / FIGURES_DIR)]


def write_summary(summary: EdaSummary, directory: Path) -> list[Path]:
    """Write the summary and its schema into `directory`."""
    return [_write(directory / SUMMARY_FILE, summary_json(summary)), write_schema(directory)]


def write_schema(directory: Path) -> Path:
    """Write the schema alone. It comes from the code, not the data: the models' docstrings are
    its descriptions, so editing one makes the committed schema stale."""
    return _write(directory / SCHEMA_FILE, schema_json())


def read_summary(directory: Path) -> EdaSummary:
    """The summary written into `directory`, validated."""
    return EdaSummary.model_validate_json((directory / SUMMARY_FILE).read_bytes())


def _write(path: Path, text: str) -> Path:
    def write(part: Path) -> None:
        part.write_text(text)

    write_into_place(path, write)
    return path
