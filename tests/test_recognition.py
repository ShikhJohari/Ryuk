"""The recognition model interface, through the fake model and the loader. Real networks are
exercised by the weights smoke tests (tests/smoke)."""

from pathlib import Path

import numpy as np
import pytest

from ryuk.detector import Image
from ryuk.recognition import ModelKey, RecognitionModel, check_aligned, l2_normalise
from ryuk.recognition.fake import FakeRecognitionModel
from ryuk.recognition.load import load_model


def face(seed: int, size: int = 112) -> Image:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_the_fake_meets_the_interface() -> None:
    model: RecognitionModel = FakeRecognitionModel(dimension=16)

    embedding = model.embed(face(1))

    assert embedding.shape == (16,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-6)
    assert model.dimension == 16
    assert model.input_size == 112


def test_the_same_face_always_gives_the_same_embedding() -> None:
    np.testing.assert_array_equal(
        FakeRecognitionModel().embed(face(1)), FakeRecognitionModel().embed(face(1))
    )


def test_similar_faces_score_higher_than_different_ones() -> None:
    model = FakeRecognitionModel()
    original = face(1)
    brighter = np.clip(original.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    same = float(model.embed(original) @ model.embed(brighter))
    different = float(model.embed(original) @ model.embed(face(2)))

    assert same > 0.9 > different


def test_another_seed_is_another_recognition_model() -> None:
    assert FakeRecognitionModel(seed=0).key != FakeRecognitionModel(seed=1).key
    assert FakeRecognitionModel(seed=0).key == FakeRecognitionModel(seed=0).key


def test_a_face_of_the_wrong_size_or_type_is_refused() -> None:
    model = FakeRecognitionModel(input_size=160)

    with pytest.raises(ValueError, match=r"\(160, 160, 3\)"):
        model.embed(face(1, size=112))
    with pytest.raises(ValueError, match="uint8"):
        check_aligned(face(1).astype(np.float32), 112)


def test_a_model_key_is_one_file_safe_string() -> None:
    key = ModelKey("arcface", "4c06" * 16, "coreml")

    assert key.id == f"arcface-coreml-{'4c06' * 16}"
    assert "/" not in key.id


def test_normalising_flattens_and_scales_to_unit_length() -> None:
    normalised = l2_normalise(np.array([[3.0, 4.0]]))

    np.testing.assert_allclose(normalised, [0.6, 0.8])
    assert normalised.dtype == np.float32


@pytest.mark.parametrize("vector", [[0.0, 0.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_a_degenerate_embedding_is_refused(vector: list[float]) -> None:
    with pytest.raises(ValueError, match="zero or non-finite"):
        l2_normalise(np.array(vector))


@pytest.mark.parametrize("network", ["sface", "facenet"])
def test_only_arcface_runs_on_coreml(network: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cpu only"):
        load_model(network, tmp_path, "coreml")  # type: ignore[arg-type]


@pytest.mark.parametrize("network", ["sface", "arcface", "facenet"])
def test_missing_weights_fail_with_the_path(network: str, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_model(network, tmp_path, "cpu")  # type: ignore[arg-type]


def test_a_flat_face_still_has_an_embedding() -> None:
    flat = np.full((112, 112, 3), 128, dtype=np.uint8)

    assert np.linalg.norm(FakeRecognitionModel().embed(flat)) == pytest.approx(1.0, abs=1e-6)
