"""Starting the watchlist from the settings, as `ryuk serve` does."""

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from ryuk.api import create_app
from ryuk.settings import Settings
from ryuk.watchlist.load import open_watchlist
from ryuk.weights import YUNET
from synthetic import YUNET as YUNET_FIXTURE
from watchlist_service import portrait, upload

RESULTS = Path(__file__).parents[1] / "evaluation" / "results.json"


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


PINNED_SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
