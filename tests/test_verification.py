"""The LFW harness end to end on a tiny synthetic LFW: real YuNet, fake recognition models."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pytest

from ryuk.detector import Detector, Image
from ryuk.evaluation.embeddings import EmbeddingCache
from ryuk.evaluation.lfw import Pair
from ryuk.evaluation.results import CropTrial, Provenance, Published, RecognitionModelId, Results
from ryuk.evaluation.verification import (
    PUBLISHED,
    LfwData,
    LfwEvaluation,
    Models,
    Pipeline,
    ScoredPairs,
    choose,
    lfw_result,
    model_id,
    score_pairs,
)
from ryuk.recognition import AlignedSize, Embedding, ModelKey, Network, RecognitionModel
from ryuk.recognition.fake import FakeRecognitionModel

FIXTURES = Path(__file__).parent / "fixtures"
YUNET = FIXTURES / "face_detection_yunet_2026may.onnx"
ASTRONAUT = np.asarray(cv2.imread(str(FIXTURES / "astronaut.jpg"), cv2.IMREAD_COLOR), np.uint8)
PROVENANCE = Provenance(
    commit="0" * 40, dirty=False, generated_at=datetime(2026, 9, 25, tzinfo=UTC), machine="test"
)


def _person(identity: int, shot: int) -> Image:
    """A 250x250 LFW-like image. Identities differ in pixels; shots of one differ slightly."""
    head = cv2.resize(np.asarray(ASTRONAUT[0:300, 100:350]), (200, 240))
    if identity % 2:
        head = np.ascontiguousarray(head[:, ::-1])
    head = np.roll(head, identity * 40, axis=2) if identity >= 2 else head
    canvas = np.full((250, 250, 3), 127, dtype=np.uint8)
    canvas[5:245, 25:225] = head
    return np.clip(canvas.astype(np.int16) + 4 * shot, 0, 255).astype(np.uint8)


def _write(folder: Path, name: str, number: int, image: Image) -> None:
    path = folder / name / f"{name}_{number:04d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _pairs_file(path: Path, header: str, folds: list[tuple[list[str], list[str]]]) -> None:
    lines = [header]
    for matched, mismatched in folds:
        lines += matched + mismatched
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def lfw(tmp_path: Path) -> LfwData:
    """Four identities of three shots each, plus Blank_Wall, whose one image has no face."""
    root = tmp_path / "lfw"
    images = root / "lfw_funneled"
    names = ["Ann", "Bob", "Cy", "Dee"]
    for identity, name in enumerate(names):
        for shot in (1, 2, 3):
            _write(images, name, shot, _person(identity, shot))
    _write(images, "Blank_Wall", 1, np.full((250, 250, 3), 127, dtype=np.uint8))

    fold = (
        ["Ann\t1\t2", "Bob\t1\t3", "Blank_Wall\t1\t1"],
        ["Ann\t1\tBob\t2", "Cy\t1\tDee\t1", "Ann\t2\tBlank_Wall\t1"],
    )
    other = (
        ["Cy\t1\t2", "Dee\t2\t3", "Ann\t2\t3"],
        ["Bob\t1\tCy\t3", "Dee\t1\tAnn\t3", "Bob\t2\tDee\t2"],
    )
    _pairs_file(root / "pairs.txt", "2\t3", [fold, other])
    _pairs_file(root / "pairsDevTrain.txt", "3", [other])
    _pairs_file(root / "pairsDevTest.txt", "3", [fold])
    return LfwData.read(tmp_path)


class Counting:
    """A recognition model that counts its embeddings, to show what the cache saves."""

    def __init__(self, model: FakeRecognitionModel) -> None:
        self._model = model
        self.calls = 0

    @property
    def key(self) -> ModelKey:
        return self._model.key

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def input_size(self) -> AlignedSize:
        return self._model.input_size

    def embed(self, face: Image) -> Embedding:
        self.calls += 1
        return self._model.embed(face)


def _fake(network: Network, seed: int = 0) -> Counting:
    size: AlignedSize = 160 if network == "facenet" else 112
    return Counting(FakeRecognitionModel(network=network, input_size=size, seed=seed))


def _models(fakes: dict[str, Counting]) -> Models:
    def loader(name: str) -> Callable[[], RecognitionModel]:
        return lambda: fakes[name]

    return Models(
        compared={network: loader(network) for network in ("sface", "arcface", "facenet")},
        sface_int8=loader("sface-int8"),
    )


@pytest.fixture
def fakes() -> dict[str, Counting]:
    return {
        "sface": _fake("sface"),
        "arcface": _fake("arcface", seed=1),
        "facenet": _fake("facenet", seed=2),
        "sface-int8": _fake("sface", seed=3),
    }


@pytest.fixture
def evaluation(lfw: LfwData, tmp_path: Path) -> LfwEvaluation:
    pipeline = Pipeline(Detector(YUNET), "e" * 64, min_face_size=40, crop="five-point")
    return LfwEvaluation(lfw, pipeline, EmbeddingCache(tmp_path / "cache"))


def test_every_model_is_scored_on_view_2_and_the_result_is_valid_json(
    evaluation: LfwEvaluation, fakes: dict[str, Counting]
) -> None:
    verification = evaluation.run(_models(fakes), PROVENANCE)

    assert [m.model.network for m in verification.models] == ["sface", "arcface", "facenet"]
    assert verification.pairs == 12
    results = Results(verification=verification)
    assert Results.model_validate_json(results.model_dump_json()) == results


def test_images_without_a_usable_face_are_listed_and_their_pairs_not_scored(
    evaluation: LfwEvaluation, fakes: dict[str, Counting]
) -> None:
    verification = evaluation.run(_models(fakes), PROVENANCE)

    assert verification.excluded_images == ["Blank_Wall/Blank_Wall_0001.jpg"]
    # Fold 0 loses its two Blank_Wall pairs; fold 1 keeps all six.
    for model in verification.models:
        assert [fold.pairs for fold in model.folds] == [4, 6]


def test_only_facenet_tries_its_crops_on_view_1_and_one_is_chosen(
    evaluation: LfwEvaluation, fakes: dict[str, Counting]
) -> None:
    verification = evaluation.run(_models(fakes), PROVENANCE)

    trials = verification.view_1
    assert {(t.network, t.crop) for t in trials} == {
        ("facenet", "five-point"),
        ("facenet", "box-margin-14"),
        ("facenet", "box-margin-32"),
    }
    assert sum(t.chosen for t in trials) == 1
    chosen = next(t.crop for t in trials if t.chosen)
    facenet = next(m for m in verification.models if m.model.network == "facenet")
    assert facenet.crop == chosen


def test_a_rerun_embeds_nothing_new(evaluation: LfwEvaluation, fakes: dict[str, Counting]) -> None:
    evaluation.run(_models(fakes), PROVENANCE)
    first = {name: fake.calls for name, fake in fakes.items()}
    assert all(first.values())

    evaluation.run(_models(fakes), PROVENANCE)

    assert {name: fake.calls for name, fake in fakes.items()} == {
        # The int8 footnote times fp32 and int8 afresh: 10 warm-up calls, then all 12 faces.
        name: calls + (10 + 12 if name in {"sface", "sface-int8"} else 0)
        for name, calls in first.items()
    }


def test_the_int8_footnote_compares_int8_with_fp32(
    evaluation: LfwEvaluation, fakes: dict[str, Counting]
) -> None:
    footnote = evaluation.run(_models(fakes), PROVENANCE).sface_int8

    # 12 distinct faces with a usable face across the View 2 pairs.
    assert footnote.faces_compared == 12
    assert footnote.model.weights_sha256 == fakes["sface-int8"].key.weights_sha256
    assert -1.0 <= footnote.cosine_to_fp32_min <= footnote.cosine_to_fp32_mean <= 1.0
    assert footnote.ms_per_face_int8 > 0
    assert footnote.ms_per_face_fp32 > 0


def _scored(scores: list[float], same: list[bool], folds: list[int]) -> ScoredPairs:
    return ScoredPairs(
        np.array(scores, dtype=np.float64), np.array(same), np.array(folds, dtype=np.int_)
    )


def _id(network: Network = "sface") -> RecognitionModelId:
    return RecognitionModelId(network=network, provider="cpu", weights_sha256="a" * 64, dimension=4)


def test_a_model_within_half_a_point_of_its_published_figure_reproduces_it() -> None:
    # Perfectly separable: every fold is 100% accurate, 0.60 points above SFace's 99.40.
    perfect = _scored([0.9, 0.8, 0.1, 0.0] * 2, [True, True, False, False] * 2, [0] * 4 + [1] * 4)

    result = lfw_result(_id(), "five-point", perfect, PUBLISHED["sface"])

    assert result.accuracy == 1.0
    assert result.standard_error == 0.0
    assert result.gap_points == pytest.approx(0.6)
    assert not result.reproduces_published
    close = Published(accuracy=0.996, source="https://example.org")
    assert lfw_result(_id(), "five-point", perfect, close).reproduces_published


def test_a_result_carries_both_operating_points_and_the_roc() -> None:
    scored = _scored([0.9, 0.8, 0.1, 0.0] * 2, [True, True, False, False] * 2, [0] * 4 + [1] * 4)

    result = lfw_result(_id(), "five-point", scored, PUBLISHED["sface"])

    assert [(p.target_far, p.indicative) for p in result.operating_points] == [
        (1e-2, False),
        (1e-3, True),
    ]
    assert all(p.tar == 1.0 and p.far == 0.0 for p in result.operating_points)
    assert result.auc == 1.0
    assert (result.roc.far[0], result.roc.tar[0]) == (0.0, 0.0)
    assert (result.roc.far[-1], result.roc.tar[-1]) == (1.0, 1.0)


def test_pairs_with_an_unusable_image_are_dropped_from_the_scores() -> None:
    a, b = PurePosixPath("a.jpg"), PurePosixPath("b.jpg")
    blank = PurePosixPath("blank.jpg")
    unit = np.array([1.0, 0.0], dtype=np.float32)
    pairs = [Pair(a, b, True, 0), Pair(a, blank, False, 0), Pair(b, a, False, 1)]

    scored = score_pairs(pairs, {a: unit, b: unit, blank: None})

    assert scored.scores.tolist() == [1.0, 1.0]
    assert scored.same.tolist() == [True, False]
    assert scored.folds.tolist() == [0, 1]
    with pytest.raises(ValueError, match="no pair"):
        score_pairs(pairs[1:2], {a: unit, blank: None})


def test_the_best_crop_is_chosen_and_a_tie_goes_to_the_first() -> None:
    def trial(crop: str, accuracy: float) -> CropTrial:
        return CropTrial.model_validate(
            {
                "network": "facenet",
                "crop": crop,
                "threshold": 0.3,
                "accuracy": accuracy,
                "chosen": False,
            }
        )

    assert [t.chosen for t in choose([trial("five-point", 0.9), trial("box-margin-14", 0.95)])] == [
        False,
        True,
    ]
    assert [t.chosen for t in choose([trial("five-point", 0.9), trial("box-margin-14", 0.9)])] == [
        True,
        False,
    ]


def test_the_fake_model_never_reaches_the_results() -> None:
    with pytest.raises(ValueError, match="fake"):
        model_id(FakeRecognitionModel())
