"""Table 1, the LFW ROC figure and the results file, from a small synthetic verification."""

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from ryuk.evaluation.figures import lfw_roc
from ryuk.evaluation.results import (
    Int8Footnote,
    Provenance,
    RecognitionModelId,
    Results,
    Verification,
    json_schema,
    read_results,
    write_results,
)
from ryuk.evaluation.tables import lfw_markdown, lfw_rows
from ryuk.evaluation.verification import PUBLISHED, ScoredPairs, lfw_result
from ryuk.recognition import Network, Provider

REPOSITORY = Path(__file__).parents[1]
RESULTS = REPOSITORY / "evaluation" / "results.json"
SCHEMA = REPOSITORY / "evaluation" / "results.schema.json"


def _id(network: Network, provider: Provider = "cpu", dimension: int = 512) -> RecognitionModelId:
    return RecognitionModelId(
        network=network, provider=provider, weights_sha256="a" * 64, dimension=dimension
    )


def _scored(errors: int) -> ScoredPairs:
    """Two identical folds of 50 matched and 50 mismatched pairs; `errors` negatives per fold
    outscore every positive. The recipe's threshold lands just under the lowest positive, 0.6,
    so each fold misclassifies exactly `errors` pairs."""
    scores, same, folds = [], [], []
    for fold in (0, 1):
        positives = np.linspace(0.6, 0.9, 50)
        negatives = np.linspace(-0.2, 0.3, 50)
        negatives[:errors] = 0.95
        scores += [*positives, *negatives]
        same += [True] * 50 + [False] * 50
        folds += [fold] * 100
    return ScoredPairs(np.array(scores), np.array(same), np.array(folds, dtype=np.int_))


def _verification() -> Verification:
    return Verification(
        provenance=Provenance(
            commit="0" * 40,
            dirty=False,
            generated_at=datetime(2026, 9, 25, tzinfo=UTC),
            machine="Test machine",
        ),
        detector={"weights_sha256": "e" * 64, "min_face_size": 40},  # type: ignore[arg-type]
        pairs=200,
        excluded_images=["Blank_Wall/Blank_Wall_0001.jpg"],
        view_1=[],
        models=[
            lfw_result(_id("sface", dimension=128), "five-point", _scored(0), PUBLISHED["sface"]),
            lfw_result(_id("arcface", "coreml"), "five-point", _scored(0), PUBLISHED["arcface"]),
            lfw_result(_id("facenet"), "box-margin-14", _scored(5), PUBLISHED["facenet"]),
        ],
        sface_int8=Int8Footnote(
            model=_id("sface", dimension=128),
            accuracy=0.99,
            standard_error=0.001,
            cosine_to_fp32_mean=0.96,
            cosine_to_fp32_min=0.93,
            faces_compared=12,
            ms_per_face_int8=11.1,
            ms_per_face_fp32=3.9,
        ),
    )


def test_table_1_has_one_row_per_model_with_the_protocol_columns() -> None:
    rows = lfw_rows(_verification())

    assert [row.model for row in rows] == ["SFace", "ArcFace (CoreML)", "FaceNet"]
    assert [row.embedding_size for row in rows] == ["128", "512", "512"]
    assert [row.published for row in rows] == ["99.40", "99.83", "99.65"]
    # SFace and ArcFace separate perfectly: 100.00, 0.60 and 0.17 points above published.
    assert rows[0].ours == "100.00 ± 0.00 ⚑"
    assert rows[1].ours == "100.00 ± 0.00"
    assert rows[0].tar_at_far_1e2 == "100.00"


def test_a_model_off_its_published_figure_is_flagged_not_hidden() -> None:
    rows = lfw_rows(_verification())
    table = lfw_markdown(_verification())

    # FaceNet: 5 of 50 negatives per fold outscore every positive, so accuracy is 95%.
    assert not rows[2].reproduces_published
    assert rows[2].ours.endswith("⚑")
    assert "a pipeline bug to find, not a result" in table


def test_the_table_notes_cover_exclusions_published_notes_and_int8() -> None:
    table = lfw_markdown(_verification())

    assert table.startswith("| Model | Embedding size | Published LFW | Our LFW (± SE) |")
    assert "1 image had no usable face" in table
    assert "ArcFace (CoreML): The buffalo_l pack's figure." in table
    assert "SFace int8 is not a row. On Test machine it scores 99.00 ± 0.10" in table
    assert "11.1 ms per face against fp32's 3.9 ms" in table


def test_the_roc_figure_draws_one_curve_per_model() -> None:
    figure = lfw_roc(_verification())

    (axes,) = figure.axes
    assert [line.get_label() for line in axes.get_lines()] == [
        "SFace, 100.00%",
        "ArcFace (CoreML), 100.00%",
        "FaceNet, 95.00%",
    ]
    assert axes.get_xscale() == "log"


def test_results_round_trip_through_the_file(tmp_path: Path) -> None:
    results = Results(verification=_verification())
    path = tmp_path / "results.json"

    write_results(path, results)

    assert read_results(path) == results
    assert path.read_text().endswith("}\n")
    assert read_results(tmp_path / "missing.json") is None


def test_the_committed_schema_is_generated_from_the_model() -> None:
    assert SCHEMA.read_text() == json_schema(), "run `uv run ryuk evaluate schema`"


def test_the_committed_results_match_the_schema() -> None:
    document = json.loads(RESULTS.read_text())

    jsonschema.validate(document, json.loads(SCHEMA.read_text()))
    Results.model_validate(document)


def test_the_committed_results_reproduce_every_published_figure_or_flag_it() -> None:
    results = Results.model_validate_json(RESULTS.read_bytes())

    for model in results.verification.models:
        gap = abs(model.accuracy - model.published.accuracy) * 100
        assert model.reproduces_published == (gap <= 0.5)
    assert {m.model.network for m in results.verification.models} == {"sface", "arcface", "facenet"}


@pytest.mark.parametrize("field", ["far", "tar"])
def test_a_roc_with_unpaired_points_is_refused(field: str) -> None:
    document = json.loads(Results(verification=_verification()).model_dump_json())
    document["verification"]["models"][0]["roc"][field].append(1.0)

    with pytest.raises(ValueError, match="paired"):
        Results.model_validate(document)
