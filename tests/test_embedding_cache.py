import platform
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pytest

from ryuk.evaluation.embeddings import EmbeddingCache, current_runtime, image_key
from ryuk.recognition import ModelKey

MODEL = ModelKey("sface", "ab" * 32, "cpu")
PIPELINE = "yunet-ebafce4e3c11-min40-five-point"


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_nothing_is_cached_at_first(tmp_path: Path) -> None:
    assert EmbeddingCache(tmp_path).load(MODEL, PIPELINE, 3) == {}


def test_embeddings_and_misses_come_back_as_saved(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    first, third = unit(1, 2, 3), unit(-1, 0, 4)

    cache.save(MODEL, PIPELINE, 3, {"a": first, "b": None, "c": third})
    loaded = EmbeddingCache(tmp_path).load(MODEL, PIPELINE, 3)

    assert set(loaded) == {"a", "b", "c"}
    assert loaded["b"] is None
    np.testing.assert_array_equal(loaded["a"], first)
    np.testing.assert_array_equal(loaded["c"], third)
    assert loaded["a"] is not None
    assert loaded["a"].dtype == np.float32


def test_a_save_merges_with_what_is_cached(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.save(MODEL, PIPELINE, 3, {"a": unit(1, 0, 0), "b": None})

    cache.save(MODEL, PIPELINE, 3, {"b": unit(0, 1, 0), "c": None})

    loaded = cache.load(MODEL, PIPELINE, 3)
    assert set(loaded) == {"a", "b", "c"}
    assert loaded["c"] is None
    np.testing.assert_array_equal(loaded["b"], unit(0, 1, 0))


def test_models_and_pipelines_never_share_entries(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.save(MODEL, PIPELINE, 3, {"a": unit(1, 0, 0)})

    on_coreml = ModelKey("sface", MODEL.weights_sha256, "coreml")
    other_weights = ModelKey("sface", "cd" * 32, "cpu")
    assert cache.load(on_coreml, PIPELINE, 3) == {}
    assert cache.load(other_weights, PIPELINE, 3) == {}
    assert cache.load(MODEL, "yunet-ebafce4e3c11-min40-box-margin-14", 3) == {}


def test_a_file_of_another_dimension_is_refused(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.save(MODEL, PIPELINE, 3, {"a": unit(1, 0, 0)})

    with pytest.raises(ValueError, match="expected 4"):
        cache.load(MODEL, PIPELINE, 4)


def test_an_embedding_of_the_wrong_shape_is_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    cache = EmbeddingCache(tmp_path)

    with pytest.raises(ValueError, match="shape"):
        cache.save(MODEL, PIPELINE, 3, {"a": unit(1, 0, 0, 0)})

    assert not cache.path(MODEL, PIPELINE).exists()


def test_images_are_keyed_by_their_bytes() -> None:
    assert image_key(b"jpeg") == image_key(b"jpeg")
    assert image_key(b"jpeg") != image_key(b"jpeg!")


def test_the_runtime_names_the_platform_and_every_embedding_library() -> None:
    runtime = current_runtime()

    assert runtime.startswith(f"{platform.system()}-{platform.machine()}-".lower())
    assert f"onnxruntime{version('onnxruntime')}" in runtime
    assert f"torch{version('torch')}".lower() in runtime
    assert f"opencv{cv2.__version__}" in runtime
    assert "/" not in runtime
