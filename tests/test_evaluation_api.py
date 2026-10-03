"""`GET /api/evaluation`: the committed dataset summary and results as the evaluation page reads
them, trimmed to what it draws."""

from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from results_files import (
    synthetic_bias,
    synthetic_identification,
    synthetic_learning,
    synthetic_live,
    synthetic_results,
    synthetic_verification,
)
from ryuk.api import create_app
from ryuk.api.evaluation_models import EvaluationReport, evaluation_report
from ryuk.api.problems import PROBLEM_MEDIA_TYPE
from ryuk.eda.files import read_summary
from ryuk.evaluation.active import MAX_MS_PER_FACE, MAX_TEST_FPIR, assemble, frozen_thresholds
from ryuk.evaluation.results import Results, read_results
from ryuk.evaluation.verification import TOLERANCE_POINTS
from summaries import eda_summary

REPOSITORY = Path(__file__).parents[1]
BUDGET_BYTES = 128 * 1024
"""The committed outputs' view is about 113 KB of JSON; the full results file is about 1 MB."""


@cache
def complete_results() -> Results:
    """Every section present, with FaceNet running the mean rule live so the bias breakdown
    covers it under two rules."""
    verification, identification = synthetic_verification(), synthetic_identification()
    learning = synthetic_learning(identification, {"facenet": "mean"})
    live = synthetic_live(identification, frozen_thresholds(identification, learning))
    bias = synthetic_bias(identification, learning)
    return assemble(verification, identification, learning, bias, live)


@cache
def complete_view() -> Any:
    with serving(evaluation_report(complete_results(), eda_summary())) as client:
        response = client.get("/api/evaluation")
    assert response.status_code == 200
    return response.json()


def serving(evaluation: EvaluationReport | None) -> TestClient:
    return TestClient(create_app(evaluation=evaluation), base_url="http://127.0.0.1")


def keys(value: object) -> Iterator[str]:
    """Every object key anywhere in a JSON value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys(item)


def test_without_an_evaluation_the_route_answers_503(client: TestClient) -> None:
    response = client.get("/api/evaluation")

    assert response.status_code == 503
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == "evaluation_unavailable"


def test_the_dataset_summary_is_served_as_counts() -> None:
    summary = eda_summary()
    dataset = complete_view()["dataset"]

    assert dataset["minUsableFaceSize"] == summary.min_usable_face_size.value
    assert dataset["minGalleryImages"] == summary.rules.min_gallery_images
    assert dataset["lfw"]["identities"] == summary.lfw.images_per_identity.identities
    assert [pairs["name"] for pairs in dataset["lfw"]["pairs"]] == [
        pairs.name for pairs in summary.lfw.pairs
    ]
    assert set(dataset["lfw"]["pairs"][0]) == {
        "name",
        "view",
        "folds",
        "matched",
        "mismatched",
        "identities",
        "images",
        "pairsWithExcludedImage",
    }
    test = summary.draw("test")
    assert dataset["celeba"][1] == {
        "draw": "test",
        "split": "test",
        "identities": test.images_per_identity.identities,
        "images": test.images_per_identity.images,
        "galleryCandidates": test.gallery_candidates,
        "eligibleIdentities": test.eligible_identities,
        "detection": {
            "images": test.detection.images,
            "detected": test.detection.detected,
            "multipleFaces": test.detection.multiple_faces,
            "usable": test.detection.usable,
        },
    }


def test_verification_carries_table_1_and_the_roc_curves() -> None:
    verification = complete_results().verification
    served = complete_view()["verification"]
    facenet = verification.models[2]

    assert served["pairs"] == verification.pairs == 203
    assert served["scoredPairs"] == verification.scored_pairs == 200
    assert served["tolerancePoints"] == TOLERANCE_POINTS
    assert served["sfaceInt8"]["msPerFaceInt8"] == verification.sface_int8.ms_per_face_int8
    model = served["models"][2]
    assert model["model"] == {
        "id": f"facenet-cpu-{'a' * 64}",
        "network": "facenet",
        "provider": "cpu",
        "weightsSha256": "a" * 64,
        "dimension": 512,
        "name": "FaceNet",
    }
    assert served["models"][1]["model"]["name"] == "ArcFace (CoreML)"
    assert model["crop"] == "box-margin-14"
    assert model["accuracy"] == facenet.accuracy
    assert model["standardError"] == facenet.standard_error
    assert model["accuracyIfExcludedWereErrors"] == facenet.accuracy_if_excluded_were_errors
    assert model["auc"] == facenet.auc
    assert model["gapPoints"] == facenet.gap_points
    assert model["reproducesPublished"] is facenet.reproduces_published
    # The spread FaceNet's figure is quoted with is the source's own words, not a number.
    assert model["published"]["note"].startswith("99.65 ± 0.25")
    assert served["models"][0]["published"]["note"] is None
    assert [point["indicative"] for point in model["operatingPoints"]] == [False, True]
    assert model["roc"] == {"far": facenet.roc.far, "tar": facenet.roc.tar}
    assert "folds" not in model


def test_identification_carries_table_2_and_the_test_curve_only() -> None:
    identification = complete_results().identification
    assert identification is not None
    served = complete_view()["identification"]
    sface = identification.models[0]

    assert served["bootstrapResamples"] == identification.bootstrap.resamples
    assert served["draws"][1] == {
        "draw": "test",
        "split": "test",
        "galleryIdentities": 100,
        "heldOutIdentities": 300,
        "enrolledPhotos": 500,
        "matedProbes": 1500,
        "nonMatedProbes": 3000,
    }
    model = served["models"][0]
    assert model["threshold"] == sface.threshold
    assert model["targetFpir"] == sface.target_fpir
    assert model["rule"] == "best-photo"
    assert model["msPerFace"] == sface.ms_per_face
    assert model["rank1"]["value"] == sface.test.rank_1.value
    fpir = sface.test.at_threshold.fpir
    assert model["atThreshold"]["fpir"] == {
        "value": fpir.value,
        "ci": {"low": fpir.ci.low, "high": fpir.ci.high},
        "adjustedWilson": None
        if fpir.adjusted_wilson is None
        else {"low": fpir.adjusted_wilson.low, "high": fpir.adjusted_wilson.high},
    }
    assert model["curve"] == {"fpir": sface.test.curve.fpir, "tpir": sface.test.curve.tpir}
    assert model["curve"] != {
        "fpir": sface.validation.curve.fpir,
        "tpir": sface.validation.curve.tpir,
    }
    assert [point["targetFpir"] for point in model["operatingPoints"]] == [
        point.target_fpir for point in sface.test.operating_points
    ]


def test_the_first_active_model_carries_the_rule_and_every_judgement() -> None:
    chosen = complete_results().first_active_model
    assert chosen is not None
    assert chosen.model is not None
    served = complete_view()["firstActiveModel"]

    assert served["model"]["network"] == chosen.model.network
    assert served["reason"] == chosen.reason
    assert served["lfwGate"] == {
        "accuracy": "scored-pairs",
        "scoredPairs": 200,
        "pairs": 203,
        "tolerancePoints": TOLERANCE_POINTS,
    }
    assert (served["maxTestFpir"], served["maxMsPerFace"]) == (MAX_TEST_FPIR, MAX_MS_PER_FACE)
    assert [judged["eligible"] for judged in served["eligibility"]] == [
        judged.eligible for judged in chosen.eligibility
    ]
    assert set(served["eligibility"][0]) == {
        "model",
        "lfwGapPoints",
        "reproducesLfw",
        "testTpir",
        "testFpir",
        "fpirWithinLimit",
        "msPerFace",
        "fastEnough",
        "eligible",
    }


def test_learning_carries_each_methods_test_rates_and_gain() -> None:
    learning = complete_results().learning
    assert learning is not None
    served = complete_view()["learning"]
    facenet = learning.models[2]

    assert served["targetFpir"] == learning.target_fpir
    compared = served["models"][2]
    assert compared["liveRule"] == "mean"
    assert compared["liveReason"] == facenet.live_reason
    assert [m["method"] for m in compared["methods"]] == ["best-photo", "mean", "learned"]
    baseline, mean, _ = compared["methods"]
    assert baseline["gain"] is None
    assert mean["gain"] == {
        "targetFpir": 0.01,
        "value": 0.02,
        "ci": {"low": 0.01, "high": 0.03},
        "improves": True,
    }
    assert mean["family"] == "scoring-rule"
    assert mean["needsRetraining"] is False
    assert mean["hyperparameter"] is None
    measured = facenet.method("mean")
    assert mean["threshold"] == measured.threshold
    assert mean["atThreshold"]["tpir"]["value"] == measured.test.at_threshold.tpir.value


def test_the_bias_breakdown_keeps_every_rule_and_group() -> None:
    bias = complete_results().bias
    assert bias is not None
    served = complete_view()["bias"]

    assert served["minIdentities"] == bias.min_identities
    assert served["agreement"] == bias.agreement
    assert [(m["model"]["network"], m["rule"]) for m in served["models"]] == [
        ("sface", "best-photo"),
        ("arcface", "best-photo"),
        ("facenet", "best-photo"),
        ("facenet", "mean"),
    ]
    attributes = {a["attribute"]: a for a in served["models"][0]["attributes"]}
    assert attributes["Male"]["basis"] == "identity"
    assert attributes["Eyeglasses"]["basis"] == "photo"
    # Too few identities behind the hat's non-mated probes for an FPIR, so no ratio either.
    assert attributes["Wearing_Hat"]["fpirRatio"] is None
    assert any(group["fpir"] is None for group in attributes["Wearing_Hat"]["groups"])
    hat = next(a for a in bias.models[0].attributes if a.attribute == "Wearing_Hat")
    assert [g["label"] for g in attributes["Wearing_Hat"]["groups"]] == [
        group.label for group in hat.groups
    ]
    assert "values" not in attributes["Male"]["groups"][0]


def test_the_live_section_carries_the_same_person_threshold_and_small_galleries() -> None:
    live = complete_results().live
    assert live is not None
    served = complete_view()["live"]["models"][0]
    measured = live.models[0]

    assert served["rule"] == measured.rule
    assert served["threshold"] == measured.threshold
    same_person = served["samePerson"]
    assert same_person["threshold"] == measured.same_person.threshold
    assert same_person["targetFar"] == 0.001
    assert same_person["validationImpostorPairs"] == 600_000
    assert [rates["enrolledPhotos"] for rates in same_person["test"]] == [1, 5]
    assert same_person["test"][0]["warningRateAtLiveThreshold"]["value"] == pytest.approx(0.2)
    assert [(g["identities"], g["enrolledPhotos"]) for g in served["smallGalleries"]] == [
        (100, 5),
        (100, 1),
        (20, 5),
        (20, 1),
        (5, 5),
        (5, 1),
    ]
    assert served["smallGalleries"][2]["galleries"] == 5


def test_the_view_leaves_out_what_the_page_does_not_draw() -> None:
    served = set(keys(complete_view()))

    # Identity lists, the validation draw, learning's curves, histograms and samples, provenance
    # and the learned rule's coefficients. LFW's fold results are checked with verification.
    assert served.isdisjoint(
        {
            "gallery",
            "heldOut",
            "excludedImages",
            "validation",
            "gaps",
            "sample",
            "learnedRule",
            "provenance",
            "candidates",
            "values",
            "view1",
        }
    )
    methods = complete_view()["learning"]["models"][0]["methods"]
    assert all("curve" not in method for method in methods)


def test_a_section_not_yet_measured_is_null() -> None:
    with serving(evaluation_report(synthetic_results(), eda_summary())) as client:
        identified = client.get("/api/evaluation").json()
    with serving(
        evaluation_report(Results(verification=synthetic_verification()), eda_summary())
    ) as client:
        verified = client.get("/api/evaluation").json()

    assert identified["identification"] is not None
    assert identified["firstActiveModel"] is not None
    assert (identified["learning"], identified["bias"], identified["live"]) == (None, None, None)
    assert verified["verification"]["models"]
    assert [verified[section] for section in ("identification", "firstActiveModel")] == [
        None,
        None,
    ]


def test_the_committed_outputs_are_served_within_budget() -> None:
    results = read_results(REPOSITORY / "evaluation" / "results.json")
    assert results is not None
    evaluation = evaluation_report(results, read_summary(REPOSITORY / "eda"))

    with serving(evaluation) as client:
        response = client.get("/api/evaluation")

    assert response.status_code == 200
    assert len(response.content) < BUDGET_BYTES
    served = EvaluationReport.model_validate_json(response.content)
    assert served == evaluation
    assert all(
        section is not None
        for section in (
            served.identification,
            served.first_active_model,
            served.learning,
            served.bias,
            served.live,
        )
    )
    # The classifiers' hyperparameters come through without the candidates tried.
    assert served.learning is not None
    chosen = [
        method.hyperparameter
        for compared in served.learning.models
        for method in compared.methods
        if method.hyperparameter is not None
    ]
    assert {hyperparameter.name for hyperparameter in chosen} == {"k", "C"}
