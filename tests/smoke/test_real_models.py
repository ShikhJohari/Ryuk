"""Weights smoke test: the real recognition models on real LFW faces (#18).

Not a benchmark. It checks shapes, normalisation, stable output, model keys, and that a few
known View 2 pairs are ordered right: every matched pair scores above every mismatched pair.
Runs in CI's weights smoke job on Linux CPU (`pytest -m smoke`) after `ryuk weights fetch` and
`ryuk data fetch --dataset lfw`, and locally against RYUK_WEIGHTS_DIR and RYUK_DATA_DIR. Missing
weights or images fail the test; they are never skipped. LFW is fetched, never committed.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from ryuk.datasets.lfw import LfwImage, Pair, images_dir, read_pairs
from ryuk.detector import Detector, Image, benchmark_face
from ryuk.recognition import RecognitionModel
from ryuk.recognition.arcface import ArcFace, default_provider
from ryuk.recognition.faces import face_crop
from ryuk.recognition.load import load_model
from ryuk.recognition.sface import SFace
from ryuk.settings import Settings
from ryuk.weights import ARCFACE, FACENET, SFACE, SFACE_INT8, YUNET

pytestmark = pytest.mark.smoke

# The first two matched and first two mismatched pairs of View 2's first fold, which are
# Abel_Pacheco 1-4, Akhmed_Zakayev 1-3, and Abdel_Madi_Shabneh 1 against Dean_Barker 1 and
# Giancarlo_Fisichella 1.
KNOWN_PAIRS = 2
PINNED_SHA256 = {
    "sface": SFACE.file.checksum.hexdigest,
    "arcface": ARCFACE.file.checksum.hexdigest,
    "facenet": FACENET.file.checksum.hexdigest,
}


@pytest.fixture(scope="module")
def weights_dir() -> Path:
    return Settings().weights_dir


@pytest.fixture(scope="module")
def lfw() -> Path:
    return images_dir(Settings().data_dir)


@pytest.fixture(scope="module")
def view_2() -> tuple[Pair, ...]:
    return read_pairs(Settings().data_dir, "pairs").pairs


@pytest.fixture(scope="module")
def matched(view_2: tuple[Pair, ...]) -> list[Pair]:
    return [pair for pair in view_2 if pair.matched][:KNOWN_PAIRS]


@pytest.fixture(scope="module")
def mismatched(view_2: tuple[Pair, ...]) -> list[Pair]:
    return [pair for pair in view_2 if not pair.matched][:KNOWN_PAIRS]


@pytest.fixture(scope="module")
def detector(weights_dir: Path) -> Detector:
    return Detector(YUNET.path(weights_dir))


def _loaders(weights_dir: Path) -> dict[str, Callable[[], RecognitionModel]]:
    loaders: dict[str, Callable[[], RecognitionModel]] = {
        "sface": lambda: load_model("sface", weights_dir),
        "arcface-cpu": lambda: load_model("arcface", weights_dir, "cpu"),
        "facenet": lambda: load_model("facenet", weights_dir),
    }
    if default_provider() == "coreml":
        loaders["arcface-coreml"] = lambda: load_model("arcface", weights_dir, "coreml")
    return loaders


@pytest.fixture(scope="module", params=["sface", "arcface-cpu", "arcface-coreml", "facenet"])
def model(request: pytest.FixtureRequest, weights_dir: Path) -> RecognitionModel:
    loaders = _loaders(weights_dir)
    if request.param not in loaders:
        pytest.skip("CoreML is only offered on Apple Silicon")
    return loaders[request.param]()


def _face(lfw: Path, detector: Detector, model: RecognitionModel, lfw_image: LfwImage) -> Image:
    path = lfw / lfw_image.path
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert decoded is not None, f"{path} is missing: run `ryuk data fetch --dataset lfw`"
    image = np.asarray(decoded, dtype=np.uint8)
    detection = benchmark_face(detector.detect(image), image.shape)
    assert detection is not None, lfw_image.path
    return face_crop(detector, image, detection, "five-point", model.input_size)


def test_an_embedding_is_unit_length_float32_of_the_model_dimension(
    model: RecognitionModel, detector: Detector, lfw: Path, matched: list[Pair]
) -> None:
    embedding = model.embed(_face(lfw, detector, model, matched[0].first))

    assert embedding.shape == (model.dimension,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
    assert model.dimension == {"sface": 128, "arcface": 512, "facenet": 512}[model.key.network]


def test_the_key_names_the_pinned_weights(model: RecognitionModel) -> None:
    assert model.key.weights_sha256 == PINNED_SHA256[model.key.network]


def test_the_same_face_gives_the_same_embedding_across_runs_and_instances(
    model: RecognitionModel, detector: Detector, weights_dir: Path, lfw: Path, matched: list[Pair]
) -> None:
    face = _face(lfw, detector, model, matched[0].first)
    first = model.embed(face)
    model.embed(_face(lfw, detector, model, matched[1].first))

    again = model.embed(face)
    fresh = _loaders(weights_dir)[_param(model)]().embed(face)

    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, fresh)


def test_known_matched_pairs_outscore_known_mismatched_pairs(
    model: RecognitionModel,
    detector: Detector,
    lfw: Path,
    matched: list[Pair],
    mismatched: list[Pair],
) -> None:
    def score(pair: Pair) -> float:
        first, second = (
            model.embed(_face(lfw, detector, model, image)) for image in (pair.first, pair.second)
        )
        return float(first @ second)

    matched_scores = [score(pair) for pair in matched]
    mismatched_scores = [score(pair) for pair in mismatched]

    assert min(matched_scores) > max(mismatched_scores) + 0.2, (matched_scores, mismatched_scores)


def test_other_weights_or_another_provider_make_another_model_key(weights_dir: Path) -> None:
    fp32 = SFace(SFACE.path(weights_dir))
    int8 = SFace(SFACE_INT8.path(weights_dir))
    assert fp32.key != int8.key
    assert int8.key.weights_sha256 == SFACE_INT8.file.checksum.hexdigest

    if default_provider() == "coreml":
        on_cpu = ArcFace(ARCFACE.path(weights_dir), "cpu")
        on_coreml = ArcFace(ARCFACE.path(weights_dir), "coreml")
        assert on_cpu.key.weights_sha256 == on_coreml.key.weights_sha256
        assert on_cpu.key != on_coreml.key


def _param(model: RecognitionModel) -> str:
    key = model.key
    return f"arcface-{key.provider}" if key.network == "arcface" else key.network
