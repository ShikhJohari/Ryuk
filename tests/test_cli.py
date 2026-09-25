"""`ryuk eda` as the command line runs it: failures are reported, not raised."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ryuk import cli
from ryuk.eda.files import FIGURES_DIR, SCHEMA_FILE, SUMMARY_FILE
from ryuk.plotting.eda import EDA_FIGURES

EDA = Path(__file__).parents[1] / "eda"


@pytest.fixture(autouse=True)
def _keep_test_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    # The command points the root logger at the runner's stderr, which closes after the run.
    monkeypatch.setattr(cli, "configure_logging", lambda: None)


@pytest.fixture
def output(tmp_path: Path) -> Path:
    directory = tmp_path / "eda"
    directory.mkdir()
    shutil.copy(EDA / SUMMARY_FILE, directory / SUMMARY_FILE)
    return directory


def test_figures_only_rewrites_the_schema_and_figures(output: Path) -> None:
    result = CliRunner().invoke(cli.app, ["eda", "--figures-only", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"Wrote {1 + len(EDA_FIGURES)} files to {output}\n"
    assert (output / SCHEMA_FILE).is_file()
    assert len(list((output / FIGURES_DIR).glob("*.svg"))) == len(EDA_FIGURES)


def test_a_figure_that_cannot_be_written_is_an_error_not_a_traceback(output: Path) -> None:
    (output / FIGURES_DIR).write_text("a file where the figures folder goes")

    result = CliRunner().invoke(cli.app, ["eda", "--figures-only", "--output", str(output)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.output


def test_a_missing_summary_is_an_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["eda", "--figures-only", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert result.stderr.startswith("error: ")
    assert SUMMARY_FILE in result.stderr
