"""The CelebA rehearsal end to end on a tiny synthetic CelebA: real YuNet, fake models."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from celeba_files import write_rehearsal
from ryuk.detector import Detector
from ryuk.eda.scan import Scanner
from ryuk.eda.summary import Draw
from ryuk.evaluation import learning
from ryuk.evaluation.active import assemble
from ryuk.evaluation.celeba import CelebaEvaluation
from ryuk.evaluation.draws import DrawMismatchError
from ryuk.evaluation.embeddings import EmbeddingCache
from ryuk.evaluation.results import (
    Identification,
    Provenance,
    Results,
    Verification,
    identification_mismatch,
    model_changes,
)
from ryuk.evaluation.verification import PUBLISHED, Pipeline, ScoredPairs, lfw_result, model_id
from ryuk.recognition import Network, RecognitionModel
from ryuk.recognition.faces import Crop
from synthetic import YUNET, Counting, fake

PROVENANCE = Provenance(
    commit="1" * 40, dirty=False, generated_at=datetime(2026, 9, 26, tzinfo=UTC), machine="test"
)
NETWORKS: tuple[Network, ...] = ("sface", "arcface", "facenet")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_rehearsal(tmp_path)
    return tmp_path


@pytest.fixture
def fakes() -> dict[Network, Counting]:
    return {network: fake(network, seed=i) for i, network in enumerate(NETWORKS)}


def _evaluation(
    root: Path, cache: Path, selections: dict[Draw, str] | None = None
) -> CelebaEvaluation:
    return CelebaEvaluation(
        root=root,
        pipeline=Pipeline(Detector(YUNET), "e" * 64, min_face_size=70, crop="five-point"),
        cache=EmbeddingCache(cache),
        scanner=Scanner(YUNET, workers=2),
        gallery_size=2,
        selections=selections or {},
    )


def _run(evaluation: CelebaEvaluation, fakes: dict[Network, Counting]) -> Identification:
    def loader(network: Network) -> Callable[[], RecognitionModel]:
        return lambda: fakes[network]

    crops: dict[Network, Crop] = {
        "sface": "five-point",
        "arcface": "five-point",
        "facenet": "box-margin-32",
    }
    return evaluation.run({n: loader(n) for n in NETWORKS}, crops, PROVENANCE)


def test_each_draw_excludes_unusable_images_then_enrols_and_holds_out(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    identification = _run(_evaluation(root, tmp_path / "cache"), fakes)

    validation, test = identification.draws
    assert (validation.draw, validation.split, test.draw, test.split) == (
        "validation",
        "valid",
        "test",
        "test",
    )
    assert (validation.images, validation.usable_images) == (60, 58)
    assert validation.gallery_candidates == 2
    assert (validation.gallery, validation.held_out) == ([100, 101], [102, 103, 104])
    assert test.gallery == [200, 201]
    assert (validation.enrolled_photos, validation.mated_probes) == (10, 30)
    # Held out: 3, 10 of 12, and the one usable image.
    assert validation.non_mated_probes == 14


def test_a_rebuilt_draw_that_matches_its_committed_digest_is_used(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    first = _run(_evaluation(root, tmp_path / "cache"), fakes)
    committed: dict[Draw, str] = {d.draw: d.selection_sha256 for d in first.draws}

    again = _run(_evaluation(root, tmp_path / "cache", committed), fakes)

    assert again.draws == first.draws


def test_a_tampered_selection_digest_fails_the_rebuild_before_anything_is_embedded(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    evaluation = _evaluation(root, tmp_path / "cache", {"test": "0" * 64})

    with pytest.raises(DrawMismatchError, match="the rebuilt test draw is not the committed one"):
        _run(evaluation, fakes)
    assert [model.calls for model in fakes.values()] == [0, 0, 0]


def test_every_model_is_scored_on_the_test_draw_at_its_validation_threshold(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    identification = _run(_evaluation(root, tmp_path / "cache"), fakes)

    assert [m.model.network for m in identification.models] == list(NETWORKS)
    for result in identification.models:
        assert result.rule == "best-photo"
        assert result.target_fpir == 0.01
        assert result.validation.at_threshold.threshold == result.threshold
        assert result.test.at_threshold.threshold == result.threshold
        assert result.validation.at_threshold.fpir.value <= 0.01
        # Five looks a fake model tells apart: every mated probe finds its own identity.
        assert result.validation.rank_1.value == result.test.rank_1.value == 1.0
        assert result.ms_per_face > 0
    assert identification.models[2].crop == "box-margin-32"


def test_a_rerun_embeds_only_the_faces_it_times(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    evaluation = _evaluation(root, tmp_path / "cache")
    first = _run(evaluation, fakes)
    for model in fakes.values():
        model.calls = 0

    again = _run(evaluation, fakes)

    # 10 warm-up searches, then the test draw's 30 mated probes timed.
    assert [model.calls for model in fakes.values()] == [40, 40, 40]
    assert [m.test for m in again.models] == [m.test for m in first.models]


def _verification(identification: Identification) -> Verification:
    """LFW results for the same models: perfect on 100 pairs, so SFace lands 0.6 points above
    its published 99.40 and is flagged, while ArcFace and FaceNet reproduce theirs."""
    scores = np.array([*np.linspace(0.6, 0.9, 50), *np.linspace(-0.2, 0.3, 50)] * 2)
    matched = np.array(([True] * 50 + [False] * 50) * 2)
    scored = ScoredPairs(scores, matched, np.repeat([0, 1], 100), np.array([0, 0]))
    return Verification(
        provenance=PROVENANCE,
        detector=identification.detector,
        pairs=200,
        excluded_images=[],
        view_1=[],
        models=[
            lfw_result(m.model, m.crop, scored, PUBLISHED[m.model.network])
            for m in identification.models
        ],
        sface_int8={  # type: ignore[arg-type]
            "model": identification.models[0].model,
            "accuracy": 0.99,
            "standard_error": 0.001,
            "cosine_to_fp32_mean": 0.96,
            "cosine_to_fp32_min": 0.93,
            "faces_compared": 12,
            "ms_per_face_int8": 11.0,
            "ms_per_face_fp32": 4.0,
        },
    )


def test_the_results_carry_a_threshold_for_every_model_and_the_first_active_one(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    identification = _run(_evaluation(root, tmp_path / "cache"), fakes)

    results = assemble(_verification(identification), identification)

    assert [(t.model.network, t.threshold) for t in results.thresholds] == [
        (m.model.network, m.threshold) for m in identification.models
    ]
    assert {(t.commit, t.date.isoformat(), t.rule) for t in results.thresholds} == {
        ("1" * 40, "2026-09-26", "best-photo")
    }
    active = results.first_active_model
    assert active is not None
    assert [c.eligible for c in active.eligibility] == [False, True, True]
    assert active.model is not None
    assert active.model.network in {"arcface", "facenet"}
    assert Results.model_validate_json(results.model_dump_json()) == results


def test_thresholds_without_identification_are_refused(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    identification = _run(_evaluation(root, tmp_path / "cache"), fakes)
    results = assemble(_verification(identification), identification)
    document = results.model_dump(mode="json")
    document["identification"] = None

    with pytest.raises(ValueError, match="thresholds must list every model"):
        Results.model_validate(document)


def test_identification_from_another_pipeline_than_lfw_is_refused(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    identification = _run(_evaluation(root, tmp_path / "cache"), fakes)
    verification = _verification(identification)
    # LFW re-run chose another crop for FaceNet: CelebA's thresholds no longer apply.
    models = [
        m.model_copy(update={"crop": "five-point"}) if m.model.network == "facenet" else m
        for m in verification.models
    ]
    rerun = verification.model_copy(update={"models": models})

    assert identification_mismatch(verification, identification) is None
    assert identification_mismatch(rerun, identification) == (
        "facenet on CelebA used the box-margin-32 crop, but LFW now chooses five-point"
    )
    with pytest.raises(ValueError, match="facenet on CelebA used the box-margin-32 crop"):
        assemble(rerun, identification)


def test_a_loaded_model_is_compared_with_the_one_recorded_for_its_network(
    fakes: dict[Network, Counting],
) -> None:
    recorded = [model_id(fakes["sface"]), model_id(fakes["arcface"])]
    other_weights = fake("arcface", seed=7)

    assert (
        model_changes({"sface": fakes["sface"].key, "arcface": fakes["arcface"].key}, recorded)
        == []
    )
    # FaceNet has nothing recorded, so there is nothing to compare it with.
    assert model_changes({"facenet": fakes["facenet"].key}, recorded) == []
    assert model_changes({"arcface": other_weights.key}, recorded) == [
        f"arcface loads as {other_weights.key.id}, but the results measured "
        f"{fakes['arcface'].key.id}"
    ]


def _loaders(fakes: dict[Network, Counting]) -> dict[Network, Callable[[], RecognitionModel]]:
    return {network: (lambda network=network: fakes[network]) for network in NETWORKS}  # type: ignore[misc]


@pytest.fixture
def without_classifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rehearsal's 14 non-mated probes leave FPIR 1% no false alarm at all, and a two-class
    softmax scores faces far from both classes higher than genuine ones, so no classifier can be
    frozen there. These tests cover the wiring; tests/test_learning.py covers the classifiers."""
    monkeypatch.setattr(learning, "CLASSIFIERS", ())


@pytest.mark.usefixtures("without_classifiers")
def test_learning_compares_every_method_on_identifications_draws_and_keeps_the_scores(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    identification = _run(_evaluation(root, tmp_path / "cache"), fakes)
    committed: dict[Draw, str] = {d.draw: d.selection_sha256 for d in identification.draws}
    evaluation = _evaluation(root, tmp_path / "cache", committed)

    learning = evaluation.learn(_loaders(fakes), identification, PROVENANCE, tmp_path / "scores")

    results = assemble(_verification(identification), identification, learning)
    assert results.learning == learning
    assert [m.model for m in learning.models] == [m.model for m in identification.models]
    assert [m.methods[0].threshold for m in learning.models] == [
        m.threshold for m in identification.models
    ]
    written = sorted(path.name for path in (tmp_path / "scores").rglob("*.parquet"))
    assert written == sorted(d.selection_sha256 + ".parquet" for d in identification.draws * 3)


@pytest.mark.usefixtures("without_classifiers")
def test_the_bias_breakdown_covers_each_model_under_best_photo_and_its_live_rule(
    root: Path, tmp_path: Path, fakes: dict[Network, Counting]
) -> None:
    identification = _run(_evaluation(root, tmp_path / "cache"), fakes)
    committed: dict[Draw, str] = {d.draw: d.selection_sha256 for d in identification.draws}
    evaluation = _evaluation(root, tmp_path / "cache", committed)
    learning = evaluation.learn(_loaders(fakes), identification, PROVENANCE, tmp_path / "scores")

    bias = evaluation.bias(_loaders(fakes), identification, learning, PROVENANCE)

    results = assemble(_verification(identification), identification, learning, bias)
    assert results.bias == bias
    # Two gallery identities: every rate is too few to estimate, yet every group is reported.
    groups = [g for m in bias.models for a in m.attributes for g in a.groups]
    assert groups
    assert all(g.tpir is None and g.fpir is None for g in groups)
