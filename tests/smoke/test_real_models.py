"""Weights smoke test: the real recognition models on real LFW faces (#18).

Not a benchmark. It checks shapes, normalisation, stable output, model keys, and that a few
known View 2 pairs are ordered right: every matched pair scores above every mismatched pair.
Runs in CI's weights smoke job on Linux CPU (`pytest -m smoke`) after `ryuk weights fetch`, and
locally against RYUK_WEIGHTS_DIR. Missing weights fail the test; they are never skipped.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from ryuk.detector import Detector, Image, benchmark_face
from ryuk.recognition import RecognitionModel
from ryuk.recognition.arcface import ArcFace, default_provider
from ryuk.recognition.faces import face_crop
from ryuk.recognition.load import load_model
from ryuk.recognition.sface import SFace
from ryuk.settings import Settings
from ryuk.weights import ARCFACE, FACENET, SFACE, SFACE_INT8, YUNET

pytestmark = pytest.mark.smoke

LFW = Path(__file__).parents[1] / "fixtures" / "lfw"
# The first two matched and first two mismatched pairs of View 2's first fold.
MATCHED = [
    ("Abel_Pacheco/Abel_Pacheco_0001.jpg", "Abel_Pacheco/Abel_Pacheco_0004.jpg"),
    ("Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg", "Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg"),
]
MISMATCHED = [
    ("Abdel_Madi_Shabneh/Abdel_Madi_Shabneh_0001.jpg", "Dean_Barker/Dean_Barker_0001.jpg"),
    (
        "Abdel_Madi_Shabneh/Abdel_Madi_Shabneh_0001.jpg",
        "Giancarlo_Fisichella/Giancarlo_Fisichella_0001.jpg",
    ),
]
PINNED_SHA256 = {
    "sface": SFACE.file.checksum.hexdigest,
    "arcface": ARCFACE.file.checksum.hexdigest,
    "facenet": FACENET.file.checksum.hexdigest,
}


@pytest.fixture(scope="module")
def weights_dir() -> Path:
    return Settings().weights_dir


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


def _face(detector: Detector, model: RecognitionModel, relative: str) -> Image:
    image = np.asarray(cv2.imread(str(LFW / relative), cv2.IMREAD_COLOR), dtype=np.uint8)
    detection = benchmark_face(detector.detect(image), image.shape)
    assert detection is not None, relative
    return face_crop(detector, image, detection, "five-point", model.input_size)


def test_an_embedding_is_unit_length_float32_of_the_model_dimension(
    model: RecognitionModel, detector: Detector
) -> None:
    embedding = model.embed(_face(detector, model, MATCHED[0][0]))

    assert embedding.shape == (model.dimension,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
    assert model.dimension == {"sface": 128, "arcface": 512, "facenet": 512}[model.key.network]


def test_the_key_names_the_pinned_weights(model: RecognitionModel) -> None:
    assert model.key.weights_sha256 == PINNED_SHA256[model.key.network]


def test_the_same_face_gives_the_same_embedding_across_runs_and_instances(
    model: RecognitionModel, detector: Detector, weights_dir: Path
) -> None:
    face = _face(detector, model, MATCHED[0][0])
    first = model.embed(face)
    model.embed(_face(detector, model, MISMATCHED[0][1]))

    again = model.embed(face)
    fresh = _loaders(weights_dir)[_param(model)]().embed(face)

    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, fresh)


def test_known_matched_pairs_outscore_known_mismatched_pairs(
    model: RecognitionModel, detector: Detector
) -> None:
    def score(pair: tuple[str, str]) -> float:
        first, second = (model.embed(_face(detector, model, path)) for path in pair)
        return float(first @ second)

    matched = [score(pair) for pair in MATCHED]
    mismatched = [score(pair) for pair in MISMATCHED]

    assert min(matched) > max(mismatched) + 0.2, (matched, mismatched)


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
