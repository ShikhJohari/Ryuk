"""`ryuk eda` and `ryuk evaluate` as the command line runs them: failures are reported, not
raised, and a failed evaluation writes nothing."""

import json
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from celeba_files import write_rehearsal
from lfw_files import write_lfw
from ryuk import cli
from ryuk.eda.files import FIGURES_DIR, SCHEMA_FILE, SUMMARY_FILE
from ryuk.evaluation import learning
from ryuk.evaluation.celeba import CelebaEvaluation
from ryuk.evaluation.results import read_results
from ryuk.plotting.eda import EDA_FIGURES
from ryuk.recognition import Network
from ryuk.recognition.load import NETWORKS
from ryuk.weights import YUNET as YUNET_WEIGHTS
from synthetic import YUNET, Counting, fake

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


@dataclass(frozen=True)
class Evaluating:
    """`ryuk evaluate` pointed by RYUK_* at a temporary machine: YuNet from the fixtures, fake
    models standing in for the networks, and the results file at `results`."""

    results: Path
    data: Path
    models: dict[Network, Counting]

    def calls(self) -> list[int]:
        return [model.calls for model in self.models.values()]

    def run(self, *arguments: str) -> Any:
        return CliRunner().invoke(cli.app, ["evaluate", *arguments])


@pytest.fixture
def evaluating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Evaluating:
    weights = tmp_path / "weights"
    YUNET_WEIGHTS.path(weights).parent.mkdir(parents=True)
    shutil.copy(YUNET, YUNET_WEIGHTS.path(weights))
    machine = Evaluating(
        results=tmp_path / "evaluation" / "results.json",
        data=tmp_path / "raw",
        models={network: fake(network, seed=i) for i, network in enumerate(NETWORKS)},
    )
    monkeypatch.setenv("RYUK_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("RYUK_DATA_DIR", str(machine.data))
    monkeypatch.setenv("RYUK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RYUK_RESULTS", str(machine.results))
    monkeypatch.setattr(cli, "load_model", lambda network, _: machine.models[network])
    monkeypatch.setattr(cli, "SFace", lambda _: fake("sface", seed=9))
    # The synthetic CelebA has two identities that can be enrolled, not 500.
    monkeypatch.setattr(cli, "CelebaEvaluation", partial(CelebaEvaluation, gallery_size=2))
    return machine


def _edit(path: Path, change: Any) -> bytes:
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path.read_bytes()


def test_evaluate_reads_the_results_ryuk_results_names(evaluating: Evaluating) -> None:
    result = evaluating.run("celeba")

    assert result.exit_code == 1
    assert result.stderr == f"error: no {evaluating.results}; run `ryuk evaluate lfw` first\n"


@pytest.fixture
def rehearsed(evaluating: Evaluating) -> Evaluating:
    """LFW then CelebA run through the command line on the fake models."""
    write_lfw(evaluating.data)
    write_rehearsal(evaluating.data)
    lfw = evaluating.run("lfw")
    assert lfw.exit_code == 0, lfw.output
    celeba = evaluating.run("celeba", "--workers", "2")
    assert celeba.exit_code == 0, celeba.output
    assert celeba.stdout.endswith(f"Wrote {evaluating.results}\n")
    return evaluating


@pytest.fixture
def moved(rehearsed: Evaluating) -> Evaluating:
    """The rehearsal's results, now evaluated with other weights for every network, as on a
    machine whose models are not the ones the results were measured on."""
    before = {network: model.key.id for network, model in rehearsed.models.items()}
    for i, network in enumerate(NETWORKS):
        rehearsed.models[network] = fake(network, seed=10 + i)
    assert all(before[n] != m.key.id for n, m in rehearsed.models.items())
    return rehearsed


def _measured(machine: Evaluating, network: Network) -> str:
    """The model key the results at RYUK_RESULTS record for `network`."""
    results = read_results(machine.results)
    assert results is not None
    model = next(m.model for m in results.verification.models if m.model.network == network)
    return f"{network}-{model.provider}-{model.weights_sha256}"


def test_lfw_refuses_before_embedding_when_celebas_models_differ(moved: Evaluating) -> None:
    recorded = moved.results.read_bytes()
    arcface = moved.models["arcface"].key.id

    result = moved.run("lfw")

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stderr.startswith(
        f"error: the CelebA results in {moved.results} were measured on other models"
    )
    assert (
        f"  arcface loads as {arcface}, but the results measured {_measured(moved, 'arcface')}"
        in result.stderr
    )
    assert "Pass --replace-identification" in result.stderr
    assert moved.calls() == [0, 0, 0]
    assert moved.results.read_bytes() == recorded


def test_lfw_drops_celebas_results_only_when_told_to(moved: Evaluating) -> None:
    result = moved.run("lfw", "--replace-identification")

    assert result.exit_code == 0, result.output
    assert result.stderr.startswith(f"warning: sface loads as {moved.models['sface'].key.id}")
    assert "run `ryuk evaluate celeba` again" in result.stderr
    assert result.stdout.endswith(f"Wrote {moved.results}\n")
    written = read_results(moved.results)
    assert written is not None
    assert [m.model.network for m in written.verification.models] == list(NETWORKS)
    assert (written.identification, written.thresholds, written.first_active_model) == (
        None,
        [],
        None,
    )


def test_celeba_refuses_before_anything_runs_when_lfw_scored_other_models(
    moved: Evaluating,
) -> None:
    recorded = moved.results.read_bytes()

    result = moved.run("celeba", "--workers", "2")

    assert result.exit_code == 1
    assert result.stderr.startswith(
        f"error: the recognition models are not the ones LFW scored in {moved.results}:"
    )
    for network in NETWORKS:
        assert f"  {network} loads as {moved.models[network].key.id}" in result.stderr
    assert "Nothing was run or written." in result.stderr
    assert moved.calls() == [0, 0, 0]
    assert moved.results.read_bytes() == recorded


def test_a_rerun_rebuilds_the_committed_draws(rehearsed: Evaluating) -> None:
    first = read_results(rehearsed.results)

    again = rehearsed.run("celeba", "--workers", "2")

    assert again.exit_code == 0, again.output
    written = read_results(rehearsed.results)
    assert first is not None
    assert first.identification is not None
    assert written is not None
    assert written.identification is not None
    assert written.identification.draws == first.identification.draws
    assert [t.model for t in written.thresholds] == [t.model for t in first.thresholds]


def test_a_draw_that_differs_from_its_committed_digest_fails_before_embedding(
    rehearsed: Evaluating,
) -> None:
    def tamper(document: Any) -> None:
        document["identification"]["draws"][1]["selection_sha256"] = "0" * 64

    tampered = _edit(rehearsed.results, tamper)
    calls = rehearsed.calls()

    refused = rehearsed.run("celeba", "--workers", "2")

    assert refused.exit_code == 1
    assert refused.stderr.startswith("error: the rebuilt test draw is not the committed one")
    assert "Pass --redraw" in refused.stderr
    assert rehearsed.calls() == calls
    assert rehearsed.results.read_bytes() == tampered

    redrawn = rehearsed.run("celeba", "--workers", "2", "--redraw")

    assert redrawn.exit_code == 0, redrawn.output
    written = read_results(rehearsed.results)
    assert written is not None
    assert written.identification is not None
    assert written.identification.selection("test").selection_sha256 != "0" * 64


def test_lfw_choosing_another_crop_writes_nothing_unless_told_to_drop_celeba(
    rehearsed: Evaluating,
) -> None:
    def other_crop(document: Any) -> None:
        # As if CelebA and LFW had run with SFace on a box crop; LFW now chooses five-point.
        document["verification"]["models"][0]["crop"] = "box-margin-14"
        document["identification"]["models"][0]["crop"] = "box-margin-14"

    edited = _edit(rehearsed.results, other_crop)

    refused = rehearsed.run("lfw")

    assert refused.exit_code == 1
    assert (
        "  sface on CelebA used the box-margin-14 crop, but LFW now chooses five-point"
        in refused.stderr
    )
    assert rehearsed.results.read_bytes() == edited

    replaced = rehearsed.run("lfw", "--replace-identification")

    assert replaced.exit_code == 0, replaced.output
    written = read_results(rehearsed.results)
    assert written is not None
    assert written.identification is None
    assert written.verification.models[0].crop == "five-point"


def test_learn_warns_that_it_drops_the_bias_breakdown(
    rehearsed: Evaluating, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rehearsal's gallery of 2 leaves no classifier a threshold (tests/test_celeba_evaluation).
    monkeypatch.setattr(learning, "CLASSIFIERS", ())
    first = rehearsed.run("learn", "--workers", "2")
    assert first.exit_code == 0, first.output
    assert "warning" not in first.stderr
    bias = rehearsed.run("bias", "--workers", "2")
    assert bias.exit_code == 0, bias.output

    again = rehearsed.run("learn", "--workers", "2")

    assert again.exit_code == 0, again.output
    assert again.stderr.startswith("warning: the bias breakdown is dropped")
    assert "run `ryuk evaluate bias` again" in again.stderr
    written = read_results(rehearsed.results)
    assert written is not None
    assert written.learning is not None
    assert written.bias is None
