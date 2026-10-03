"""The model registry and startup through the HTTP seam: model states, the active model, and
embeddings rebuilt or filled in from enrolled photos."""

from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from results_files import synthetic_identification, synthetic_live, synthetic_verification
from ryuk.api import create_app
from ryuk.evaluation.active import assemble, frozen_thresholds
from ryuk.evaluation.results import read_results
from ryuk.pipeline import PIPELINE_VERSION
from ryuk.recognition import ModelKey
from ryuk.watchlist import service
from ryuk.watchlist.database import MIGRATIONS, migrate, open_database
from ryuk.watchlist.registry import Evaluated, Evaluation, Unavailable
from ryuk.watchlist.tables import Base
from synthetic import fake
from watchlist_service import (
    MS_PER_FACE,
    THRESHOLD,
    contender,
    evaluated,
    portrait,
    serve,
    upload,
)

RESULTS = Path(__file__).parents[1] / "evaluation" / "results.json"
ARCFACE_KEY = ModelKey("arcface", "a" * 64, "cpu")


def enroll(client: TestClient, name: str, look: int, shot: int = 0) -> Any:
    return client.post("/api/persons", data={"name": name}, files=upload(portrait(look, shot)))


def states(client: TestClient) -> dict[str, str]:
    return {model["network"]: model["state"] for model in client.get("/api/models").json()}


def test_every_model_is_listed_with_its_state(tmp_path: Path) -> None:
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    arcface = Unavailable(ARCFACE_KEY, 512)
    evaluation = evaluated(sface.key, facenet.key, ARCFACE_KEY, first_active=sface.key)
    with serve(tmp_path / "ryuk.sqlite3", [sface, arcface, facenet], evaluation) as client:
        models = client.get("/api/models").json()

    assert [(m["network"], m["state"]) for m in models] == [
        ("sface", "active"),
        ("arcface", "unavailable"),
        ("facenet", "available"),
    ]
    assert models[0] == {
        "id": sface.key.id,
        "network": "sface",
        "provider": "cpu",
        "weightsSha256": sface.key.weights_sha256,
        "name": "SFace",
        "state": "active",
        "threshold": THRESHOLD,
        "dimension": 32,
        "msPerFace": MS_PER_FACE,
    }
    assert models[1]["name"] == "ArcFace (CPU)"


def test_a_model_without_a_threshold_for_its_exact_weights_is_not_evaluated(
    tmp_path: Path,
) -> None:
    sface = fake("sface")
    other_weights = fake("sface", seed=9).key
    with serve(tmp_path / "ryuk.sqlite3", [sface], evaluated(other_weights)) as client:
        [model] = client.get("/api/models").json()

    assert model["state"] == "not_evaluated"
    assert model["threshold"] is None
    assert model["msPerFace"] is None


def test_the_first_active_model_is_persisted_and_kept_across_restarts(tmp_path: Path) -> None:
    database = tmp_path / "ryuk.sqlite3"
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    with serve(
        database, [sface, facenet], evaluated(sface.key, facenet.key, first_active=sface.key)
    ) as client:
        assert states(client) == {"sface": "active", "facenet": "available"}

    # Evaluation now prefers FaceNet, but the operator's persisted choice stands.
    with serve(
        database, [sface, facenet], evaluated(sface.key, facenet.key, first_active=facenet.key)
    ) as client:
        assert states(client) == {"sface": "active", "facenet": "available"}


def test_a_persisted_model_that_lost_its_weights_gives_way_to_the_first_active(
    tmp_path: Path,
) -> None:
    database = tmp_path / "ryuk.sqlite3"
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    evaluation = evaluated(sface.key, facenet.key, first_active=sface.key)
    with serve(database, [sface, facenet], evaluation):
        pass

    gone = Unavailable(sface.key, 32)
    moved = evaluated(sface.key, facenet.key, first_active=facenet.key)
    with serve(database, [gone, facenet], moved) as client:
        assert states(client) == {"sface": "unavailable", "facenet": "active"}
    # Its weights are back: the persisted choice applies again.
    with serve(database, [sface, facenet], moved) as client:
        assert states(client) == {"sface": "active", "facenet": "available"}


def test_a_fallback_active_model_is_not_persisted_so_evaluations_choice_applies_later(
    tmp_path: Path,
) -> None:
    database = tmp_path / "ryuk.sqlite3"
    sface, arcface = fake("sface"), fake("arcface", seed=1)
    evaluation = evaluated(
        sface.key,
        arcface.key,
        first_active=arcface.key,
        contenders=[contender(sface), contender(arcface)],
    )
    # ArcFace's weights are not fetched yet, so #9's rule picks SFace for this run.
    with serve(database, [sface, Unavailable(arcface.key, 32)], evaluation) as client:
        assert states(client) == {"sface": "active", "arcface": "unavailable"}

    # They arrive: evaluation's own choice applies, as the fallback was not persisted.
    with serve(database, [sface, arcface], evaluation) as client:
        assert states(client) == {"sface": "available", "arcface": "active"}

    # Evaluation's choice was persisted then: it stands though evaluation now prefers SFace.
    prefers_sface = evaluated(sface.key, arcface.key, first_active=sface.key)
    with serve(database, [sface, arcface], prefers_sface) as client:
        assert states(client) == {"sface": "available", "arcface": "active"}


def test_no_model_is_active_when_none_that_can_run_was_judged_eligible(tmp_path: Path) -> None:
    sface = fake("sface")
    evaluation = evaluated(sface.key, ARCFACE_KEY, first_active=ARCFACE_KEY)
    with serve(tmp_path / "ryuk.sqlite3", [sface, Unavailable(ARCFACE_KEY, 512)], evaluation) as c:
        assert states(c) == {"sface": "available", "arcface": "unavailable"}


def test_a_weights_change_rebuilds_that_models_embeddings_from_the_photos(
    tmp_path: Path,
) -> None:
    database = tmp_path / "ryuk.sqlite3"
    before, after = fake("sface", seed=0), fake("sface", seed=1)
    evaluation = Evaluation(
        {**evaluated(before.key).models, **evaluated(after.key).models}, before.key
    )
    with serve(database, [before], evaluation) as client:
        ada = enroll(client, "Ada", 0).json()
        enroll(client, "Grace", 1)
    assert before.calls == 2

    changed = Evaluation(evaluation.models, after.key)
    with serve(database, [after], changed) as client:
        # Both enrolled photos were embedded again, before any request was served.
        assert after.calls == 2
        [model] = client.get("/api/models").json()
        assert model["id"] == after.key.id
        # The rebuilt embeddings are what the new model compares against.
        warned = enroll(client, "Someone", 0, shot=3)
        assert [(w["code"], w["personId"]) for w in warned.json()["warnings"]] == [
            ("looks_like_other", ada["id"])
        ]

    # Nothing changed this time, so nothing is embedded again.
    with serve(database, [after], changed):
        pass
    assert after.calls == 3


def test_a_crop_change_rebuilds_that_models_embeddings_from_the_photos(tmp_path: Path) -> None:
    database = tmp_path / "ryuk.sqlite3"
    facenet = fake("facenet")
    # Not evaluated, so FaceNet cuts faces five-point.
    with serve(database, [facenet], evaluated()) as client:
        ada = enroll(client, "Ada", 0).json()
        enroll(client, "Grace", 1)
    assert facenet.calls == 2

    # Evaluation measured FaceNet's threshold on its own box crop.
    box = Evaluation(
        {facenet.key: Evaluated(THRESHOLD, "best-photo", "box-margin-32", MS_PER_FACE)}, facenet.key
    )
    with serve(database, [facenet], box) as client:
        # Both enrolled photos were embedded again, before any request was served.
        assert facenet.calls == 4
        assert states(client) == {"facenet": "active"}
        # A box-cut probe is compared against box-cut enrolled embeddings.
        warned = enroll(client, "Someone", 0, shot=3)
        assert [(w["code"], w["personId"]) for w in warned.json()["warnings"]] == [
            ("looks_like_other", ada["id"])
        ]
    assert facenet.calls == 5


def test_a_pipeline_version_change_rebuilds_every_models_embeddings_from_the_photos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "ryuk.sqlite3"
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    evaluation = evaluated(sface.key, facenet.key, first_active=sface.key)
    with serve(database, [sface, facenet], evaluation) as client:
        ada = enroll(client, "Ada", 0).json()
        enroll(client, "Grace", 1)
    assert (sface.calls, facenet.calls) == (2, 2)

    # Same weights, same crop: only the pipeline between photo and embedding changed.
    monkeypatch.setattr(service, "PIPELINE_VERSION", PIPELINE_VERSION + 1)
    with serve(database, [sface, facenet], evaluation) as client:
        # Both enrolled photos were embedded again under both models before any request.
        assert (sface.calls, facenet.calls) == (4, 4)
        warned = enroll(client, "Someone", 0, shot=3)
        assert [(w["code"], w["personId"]) for w in warned.json()["warnings"]] == [
            ("looks_like_other", ada["id"])
        ]

    # The new version is recorded, so the next start rebuilds nothing.
    with serve(database, [sface, facenet], evaluation):
        pass
    assert (sface.calls, facenet.calls) == (5, 5)


def test_an_unchanged_crop_rebuilds_nothing(tmp_path: Path) -> None:
    database = tmp_path / "ryuk.sqlite3"
    facenet = fake("facenet")
    box = Evaluation(
        {facenet.key: Evaluated(THRESHOLD, "best-photo", "box-margin-32", MS_PER_FACE)}, facenet.key
    )
    with serve(database, [facenet], box) as client:
        enroll(client, "Ada", 0)
    assert facenet.calls == 1

    with serve(database, [facenet], box) as client:
        assert states(client) == {"facenet": "active"}
    assert facenet.calls == 1


def test_embeddings_are_filled_in_when_a_models_weights_appear(tmp_path: Path) -> None:
    database = tmp_path / "ryuk.sqlite3"
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    with serve(database, [sface, Unavailable(facenet.key, 32)], evaluated(sface.key)) as client:
        enroll(client, "Ada", 0)
    assert facenet.calls == 0

    evaluation = evaluated(sface.key, facenet.key, first_active=facenet.key)
    with serve(database, [sface, facenet], evaluation) as client:
        assert facenet.calls == 1
        assert states(client)["facenet"] == "active"
        assert enroll(client, "Grace", 0, shot=3).json()["warnings"][0]["code"] == (
            "looks_like_other"
        )


def test_the_committed_thresholds_name_the_first_active_model() -> None:
    evaluation = Evaluation.from_results(read_results(RESULTS))

    assert evaluation.first_active is not None
    assert evaluation.first_active.network == "arcface"
    assert {key.network for key in evaluation.models} == {"sface", "arcface", "facenet"}
    # Enrollment cuts faces the way each threshold was measured: FaceNet on its own box crop.
    crops = {key.network: e.crop for key, e in evaluation.models.items()}
    assert crops == {"sface": "five-point", "arcface": "five-point", "facenet": "box-margin-32"}


def test_where_the_first_active_model_cannot_run_the_rule_picks_among_those_that_can() -> None:
    evaluation = Evaluation.from_results(read_results(RESULTS))
    keys = {key.network: key for key in evaluation.models}

    # On Linux ArcFace runs on CPU, a model with no threshold, so the CoreML one never loads.
    # Under their live mean rules, SFace's test TPIR (96.2%) leads FaceNet's (88.4%) with no
    # overlap, so #9's rule picks it.
    assert evaluation.first_active_for([keys["sface"], keys["facenet"]]) == keys["sface"]
    assert evaluation.first_active_for([keys["facenet"]]) == keys["facenet"]
    assert evaluation.first_active_for(keys.values()) == keys["arcface"]
    assert evaluation.first_active_for([]) is None


def test_each_model_reads_its_same_person_threshold_where_live_froze_one() -> None:
    identification = synthetic_identification()
    live = synthetic_live(identification, frozen_thresholds(identification, None))
    verification = synthetic_verification()

    measured = Evaluation.from_results(assemble(verification, identification, None, None, live))
    unmeasured = Evaluation.from_results(assemble(verification, identification))

    assert [e.same_person_threshold for e in measured.models.values()] == [
        m.same_person.threshold for m in live.models
    ]
    assert [e.same_person_threshold for e in unmeasured.models.values()] == [None, None, None]


def test_contenders_without_the_gate_they_were_judged_by_are_refused() -> None:
    sface = fake("sface")

    with pytest.raises(ValueError, match="contenders come with the LFW gate"):
        Evaluation({}, None, [contender(sface)])


def test_no_results_means_nothing_is_evaluated(tmp_path: Path) -> None:
    evaluation = Evaluation.from_results(read_results(tmp_path / "missing.json"))

    assert evaluation == Evaluation({}, None)


def test_migrations_create_exactly_the_schema_the_tables_describe(tmp_path: Path) -> None:
    engine = open_database(tmp_path / "nested" / "ryuk.sqlite3")

    with engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
        assert set(inspect(connection).get_table_names()) == {
            "alembic_version",
            "person_of_interest",
            "enrolled_photo",
            "embedding",
            "recognition_model",
            "setting",
            "sighting",
        }

    config = Config()
    config.set_main_option("script_location", MIGRATIONS)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
    with engine.connect() as connection:
        assert set(inspect(connection).get_table_names()) == {"alembic_version"}
    migrate(engine)
    engine.dispose()


def test_the_database_enforces_foreign_keys_and_secure_delete(tmp_path: Path) -> None:
    engine = open_database(tmp_path / "ryuk.sqlite3")

    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert connection.exec_driver_sql("PRAGMA secure_delete").scalar() == 1
    engine.dispose()


@pytest.mark.parametrize("path", ["/api/models", "/api/persons"])
def test_without_a_watchlist_its_routes_are_unavailable(path: str) -> None:
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.get(path)

    assert response.status_code == 503
    assert response.json()["code"] == "watchlist_unavailable"


def test_a_fake_model_must_stand_in_for_a_real_network(tmp_path: Path) -> None:
    from ryuk.recognition.fake import FakeRecognitionModel  # noqa: PLC0415

    with (
        pytest.raises(ValueError, match="network"),
        serve(tmp_path / "ryuk.sqlite3", [FakeRecognitionModel()], evaluated()),
    ):
        pass


def test_the_operator_switches_the_active_model_and_the_choice_is_kept(tmp_path: Path) -> None:
    database = tmp_path / "ryuk.sqlite3"
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    evaluation = evaluated(sface.key, facenet.key, first_active=sface.key)
    with serve(database, [sface, facenet], evaluation) as client:
        response = client.put("/api/active-model", json={"modelKey": facenet.key.id})
        assert response.status_code == 200
        assert states(client) == {"sface": "available", "facenet": "active"}

    with serve(database, [sface, facenet], evaluation) as client:
        assert states(client) == {"sface": "available", "facenet": "active"}


@pytest.mark.parametrize(
    ("model", "status", "code"),
    [
        pytest.param("unknown", 404, "not_found", id="unknown"),
        pytest.param("not_evaluated", 409, "cannot_be_active", id="not evaluated"),
        pytest.param("unavailable", 409, "cannot_be_active", id="unavailable"),
    ],
)
def test_only_an_evaluated_model_with_its_weights_can_be_made_active(
    tmp_path: Path, model: str, status: int, code: str
) -> None:
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    arcface = Unavailable(ARCFACE_KEY, 512)
    keys = {
        "unknown": "sface-cpu-" + "0" * 64,
        "not_evaluated": facenet.key.id,
        "unavailable": ARCFACE_KEY.id,
    }
    evaluation = evaluated(sface.key, ARCFACE_KEY, first_active=sface.key)
    with serve(tmp_path / "ryuk.sqlite3", [sface, arcface, facenet], evaluation) as client:
        response = client.put("/api/active-model", json={"modelKey": keys[model]})
        after = states(client)

    assert response.status_code == status
    assert response.json()["code"] == code
    assert after["sface"] == "active"


def test_a_threshold_sent_with_the_active_model_is_refused(tmp_path: Path) -> None:
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    evaluation = evaluated(sface.key, facenet.key, first_active=sface.key)
    with serve(tmp_path / "ryuk.sqlite3", [sface, facenet], evaluation) as client:
        response = client.put(
            "/api/active-model", json={"modelKey": facenet.key.id, "threshold": 0.1}
        )
        after = states(client)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert after == {"sface": "active", "facenet": "available"}
