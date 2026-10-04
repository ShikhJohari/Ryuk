"""Starting the watchlist from the settings, as `ryuk serve` does."""

import json
import shutil
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ryuk import cli
from ryuk.api import create_app
from ryuk.api.evaluation_models import evaluation_report
from ryuk.eda.files import SUMMARY_FILE, read_summary
from ryuk.evaluation.results import read_results
from ryuk.settings import Settings
from ryuk.watchlist.errors import StartupError
from ryuk.watchlist.load import committed_results, committed_summary, open_watchlist
from ryuk.weights import YUNET
from synthetic import YUNET as YUNET_FIXTURE
from watchlist_service import portrait, upload

RESULTS = Path(__file__).parents[1] / "evaluation" / "results.json"
EDA = Path(__file__).parents[1] / "eda"


def settings(tmp_path: Path) -> Settings:
    return Settings(
        weights_dir=tmp_path / "weights",
        database=tmp_path / "app" / "ryuk.sqlite3",
        results=RESULTS,
    )


def test_the_service_starts_with_no_weights_and_every_model_unavailable(tmp_path: Path) -> None:
    app = create_app(lambda: open_watchlist(settings(tmp_path)))

    with TestClient(app, base_url="http://127.0.0.1") as client:
        models = client.get("/api/models").json()
        enrolled = client.post("/api/persons", data={"name": "Ada"}, files=upload(portrait(0)))

    assert [(m["network"], m["state"]) for m in models] == [
        ("sface", "unavailable"),
        ("arcface", "unavailable"),
        ("facenet", "unavailable"),
    ]
    # Unavailable models are named by their pinned weights, and keep their frozen thresholds.
    assert models[0]["weightsSha256"] == PINNED_SFACE_SHA256
    assert models[0]["threshold"] is not None
    assert enrolled.json()["code"] == "no_detector"


def test_with_only_the_detector_a_person_is_enrolled_with_no_embeddings(tmp_path: Path) -> None:
    config = settings(tmp_path)
    config.weights_dir.mkdir()
    shutil.copy(YUNET_FIXTURE, YUNET.path(config.weights_dir))

    with TestClient(create_app(lambda: open_watchlist(config)), base_url="http://127.0.0.1") as c:
        enrolled = c.post("/api/persons", data={"name": "Ada"}, files=upload(portrait(0)))

        assert enrolled.status_code == 201
        # A second photo of the same face raises no face warning: no model can be active.
        again = c.post("/api/persons", data={"name": "Grace"}, files=upload(portrait(0, 3)))
        assert again.status_code == 201


def with_detector(config: Settings) -> Settings:
    config.weights_dir.mkdir()
    shutil.copy(YUNET_FIXTURE, YUNET.path(config.weights_dir))
    return config


def test_the_service_refuses_to_start_without_the_evaluation_results(tmp_path: Path) -> None:
    # As when `ryuk serve` runs outside the repository root.
    config = with_detector(settings(tmp_path).model_copy(update={"results": tmp_path / "none"}))

    with (
        pytest.raises(StartupError, match=r"No evaluation results at .*none"),
        TestClient(create_app(lambda: open_watchlist(config)), base_url="http://127.0.0.1"),
    ):
        pass

    # Refused before the database, which holds face photos, is created there.
    assert not config.database.parent.exists()


def test_the_service_refuses_to_start_on_results_that_do_not_match_their_schema(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "results.json"
    broken.write_text('{"schema_version": 1}')
    config = settings(tmp_path).model_copy(update={"results": broken})

    with pytest.raises(StartupError, match="do not match their schema") as stopped:
        open_watchlist(config)
    assert "\n" not in str(stopped.value)
    assert str(stopped.value).endswith("1 error, the first at verification: Field required.")


@pytest.mark.parametrize("block", ["verification", "identification"])
@pytest.mark.parametrize(
    ("field", "value", "said"),
    [
        ("weights_sha256", "0" * 64, "weights"),
        ("min_face_size", 80, "minimum usable face size"),
    ],
)
def test_the_service_refuses_to_start_with_a_detector_evaluation_did_not_measure(
    tmp_path: Path, block: str, field: str, value: object, said: str
) -> None:
    results = json.loads(RESULTS.read_text())
    results[block]["detector"][field] = value
    tampered = tmp_path / "results.json"
    tampered.write_text(json.dumps(results))
    config = with_detector(settings(tmp_path).model_copy(update={"results": tampered}))

    with pytest.raises(StartupError, match=said):
        open_watchlist(config)
    assert not config.database.parent.exists()


def test_ryuk_serve_reports_a_refused_start_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda: None)

    def serving(*_: object, **__: object) -> None:
        pytest.fail("the service started")

    monkeypatch.setattr(uvicorn, "run", serving)
    monkeypatch.setenv("RYUK_RESULTS", str(tmp_path / "missing.json"))
    monkeypatch.setenv("RYUK_DATABASE", str(tmp_path / "app" / "ryuk.sqlite3"))
    monkeypatch.setenv("RYUK_WEIGHTS_DIR", str(tmp_path / "weights"))

    result = CliRunner().invoke(cli.app, ["serve"])

    assert result.exit_code == 1
    assert result.stderr.startswith("error: No evaluation results at ")
    assert "Traceback" not in result.output
    assert not (tmp_path / "app").exists()


def test_the_watchlist_takes_results_already_read_without_reading_them_again(
    tmp_path: Path,
) -> None:
    config = settings(tmp_path).model_copy(update={"results": tmp_path / "none"})

    app = create_app(lambda: open_watchlist(config, committed_results(RESULTS)))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        models = client.get("/api/models").json()

    # Thresholds come from the results handed in, not from the missing file.
    assert models[0]["threshold"] is not None


def test_a_missing_dataset_summary_is_a_startup_error(tmp_path: Path) -> None:
    with pytest.raises(StartupError, match=rf"No dataset summary at .*{SUMMARY_FILE}.*RYUK_EDA"):
        committed_summary(tmp_path)


def test_a_dataset_summary_that_does_not_match_its_schema_is_a_startup_error(
    tmp_path: Path,
) -> None:
    (tmp_path / SUMMARY_FILE).write_text(
        (EDA / SUMMARY_FILE).read_text().replace('"schema_version": 1', '"schema_version": 2')
    )

    with pytest.raises(StartupError, match="does not match its schema") as stopped:
        committed_summary(tmp_path)
    assert "\n" not in str(stopped.value)
    assert "1 error, the first at schema_version: " in str(stopped.value)


def ryuk_serve_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **paths: Path) -> None:
    """Point `ryuk serve` at a temporary machine with the committed outputs, or at `paths`."""
    monkeypatch.setattr(cli, "configure_logging", lambda: None)
    monkeypatch.setenv("RYUK_RESULTS", str(paths.get("results", RESULTS)))
    monkeypatch.setenv("RYUK_EDA", str(paths.get("eda", EDA)))
    monkeypatch.setenv("RYUK_DATABASE", str(tmp_path / "app" / "ryuk.sqlite3"))
    monkeypatch.setenv("RYUK_WEIGHTS_DIR", str(tmp_path / "weights"))


def test_ryuk_serve_hands_the_service_the_committed_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ryuk_serve_env(monkeypatch, tmp_path)
    served: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: served.append(app))

    result = CliRunner().invoke(cli.app, ["serve"])

    assert result.exit_code == 0, result.output
    [app] = served
    with TestClient(app, base_url="http://127.0.0.1") as client:  # type: ignore[arg-type]
        evaluation = client.get("/api/evaluation")
        models = client.get("/api/models")
    results = read_results(RESULTS)
    assert results is not None
    expected = evaluation_report(results, read_summary(EDA))
    assert evaluation.status_code == 200
    assert evaluation.json() == expected.model_dump(mode="json", by_alias=True)
    assert len(models.json()) == 3


def test_ryuk_serve_refuses_to_start_without_the_dataset_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ryuk_serve_env(monkeypatch, tmp_path, eda=tmp_path / "eda")

    def serving(*_: object, **__: object) -> None:
        pytest.fail("the service started")

    monkeypatch.setattr(uvicorn, "run", serving)

    result = CliRunner().invoke(cli.app, ["serve"])

    assert result.exit_code == 1
    assert result.stderr.startswith("error: No dataset summary at ")
    assert "Traceback" not in result.output
    assert not (tmp_path / "app").exists()


PINNED_SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
