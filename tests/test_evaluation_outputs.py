"""Table 1, the LFW ROC figure and the results file, from a small synthetic verification."""

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from ryuk.eda.summary import Draw
from ryuk.evaluation.active import assemble
from ryuk.evaluation.figures import lfw_roc, openset_curves
from ryuk.evaluation.openset import ScoredProbes, draw_result, freeze
from ryuk.evaluation.results import (
    Bootstrap,
    DetectorId,
    DrawSelection,
    Identification,
    Int8Footnote,
    OpenSetModel,
    Provenance,
    RecognitionModelId,
    Results,
    Verification,
    json_schema,
    read_results,
    write_results,
)
from ryuk.evaluation.tables import lfw_markdown, lfw_rows, openset_markdown, openset_rows
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
    scores, matched, folds = [], [], []
    for fold in (0, 1):
        positives = np.linspace(0.6, 0.9, 50)
        negatives = np.linspace(-0.2, 0.3, 50)
        negatives[:errors] = 0.95
        scores += [*positives, *negatives]
        matched += [True] * 50 + [False] * 50
        folds += [fold] * 100
    return ScoredPairs(
        np.array(scores), np.array(matched), np.array(folds, dtype=np.int_), np.array([3, 0])
    )


def _verification() -> Verification:
    return Verification(
        provenance=Provenance(
            commit="0" * 40,
            dirty=False,
            generated_at=datetime(2026, 9, 25, tzinfo=UTC),
            machine="Test machine",
        ),
        detector={"weights_sha256": "e" * 64, "min_face_size": 40},  # type: ignore[arg-type]
        # 200 scored and 3 excluded, all in fold 0.
        pairs=203,
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
    assert rows[0].tar_at_far == ("100.00", "100.00")


def test_table_1_is_one_contiguous_markdown_table_then_its_notes() -> None:
    lines = lfw_markdown(_verification()).split("\n")

    assert lines[:2] == [
        "| Model | Embedding size | Published LFW | Our LFW (± SE) | AUC "
        "| TAR @ FAR 1% | TAR @ FAR 0.1% (indicative) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    assert [line.split(" | ")[0] for line in lines[2:5]] == [
        "| SFace",
        "| ArcFace (CoreML)",
        "| FaceNet",
    ]
    assert lines[5] == ""
    assert lines[6].startswith("LFW View 2, 200 of 203 pairs scored")


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
    assert "200 of 203 pairs scored" in table
    assert "1 image had no usable face, so 3 pairs using them are not scored" in table
    # SFace is perfect on 100 scored pairs per fold; fold 0 lost 3: (100/103 + 1) / 2 = 98.54%.
    assert "Counting each unscored pair as an error: SFace 98.54" in table
    assert "crimdet shipped int8" in table
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


def _probes(draw: Draw, rng: np.random.Generator, overlap: float) -> ScoredProbes:
    """100 gallery identities x 15 mated probes, 1 in 50 misidentified, and 300 held-out
    identities x 10 non-mated probes; `overlap` pushes non-mated scores up into the mated ones."""
    mated = np.repeat(np.arange(100), 15)
    return ScoredProbes(
        draw=draw,
        mated_identity=mated,
        mated_score=rng.uniform(0.45, 0.95, mated.size),
        mated_correct=np.arange(mated.size) % 50 != 0,
        non_mated_identity=np.repeat(np.arange(1000, 1300), 10),
        non_mated_score=rng.uniform(-0.2, 0.3 + overlap, 3000),
    )


def _selection(draw: Draw) -> DrawSelection:
    return DrawSelection(
        draw=draw,
        split="valid" if draw == "validation" else "test",
        seed=27,
        images=5000,
        usable_images=4800,
        gallery_candidates=150,
        gallery=list(range(100)),
        held_out=list(range(1000, 1300)),
        enrolled_photos=500,
        mated_probes=1500,
        non_mated_probes=3000,
        selection_sha256="c" * 64,
    )


def _identification() -> Identification:
    rng = np.random.default_rng(27)
    verification = _verification()
    models = []
    for lfw, overlap, ms in zip(
        verification.models, (0.3, 0.2, 0.4), (4.0, 12.0, 25.0), strict=True
    ):
        frozen = freeze(_probes("validation", rng, overlap))
        models.append(
            OpenSetModel(
                model=lfw.model,
                crop=lfw.crop,
                rule="best-photo",
                threshold=frozen.value,
                target_fpir=frozen.target_fpir,
                validation=draw_result(_probes("validation", rng, overlap), frozen, seed=1),
                test=draw_result(_probes("test", rng, overlap), frozen, seed=1),
                ms_per_face=ms,
            )
        )
    return Identification(
        provenance=verification.provenance,
        detector=DetectorId(weights_sha256="e" * 64, min_face_size=70),
        bootstrap=Bootstrap(resamples=2000, seed=1, confidence=0.95),
        draws=[_selection("validation"), _selection("test")],
        models=models,
    )


def _results() -> Results:
    return assemble(_verification(), _identification())


def test_table_2_has_one_row_per_model_on_the_test_draw_with_the_active_one_starred() -> None:
    results = _results()
    rows = openset_rows(results)
    active = results.first_active_model

    assert active is not None
    assert active.model is not None
    # SFace is off its published LFW figure, so ArcFace or FaceNet starts live.
    assert active.model.network in {"arcface", "facenet"}
    assert [row.active for row in rows].count(True) == 1
    assert [row.model.removesuffix(" ★") for row in rows] == [
        "SFace",
        "ArcFace (CoreML)",
        "FaceNet",
    ]
    assert results.identification is not None
    assert rows[0].threshold == f"{results.identification.models[0].threshold:.3f}"
    assert rows[0].ms_per_face == "4.0"


def test_table_2_is_one_contiguous_markdown_table_then_its_notes() -> None:
    lines = openset_markdown(_results()).split("\n")

    assert lines[:2] == [
        "| Model | Rank-1 [95% CI] | Frozen threshold | TPIR at threshold [CI] "
        "| FPIR at threshold [CI] | Misidentification [CI] | TPIR @ FPIR 0.1% (indicative) "
        "| ms per face |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    assert lines[5] == ""
    assert lines[6].startswith(
        "CelebA test draw: 100 gallery identities with 1,500 mated probes, 300 held-out "
        "identities with 3,000 non-mated probes."
    )


def test_table_2_notes_the_wilson_check_the_active_model_and_who_is_not_eligible() -> None:
    table = openset_markdown(_results())

    assert "dependence-adjusted Wilson interval" in table
    assert "★ First active model." in table
    assert "SFace is not eligible: its LFW accuracy is +0.60 points from published." in table


def test_a_rate_is_shown_with_its_interval() -> None:
    row = openset_rows(_results())[1]

    value, interval = row.tpir.split(" ", 1)
    low, high = (float(v) for v in interval.strip("[]").split(", "))
    assert low <= float(value) <= high


def test_the_openset_figure_draws_one_curve_and_one_marked_threshold_per_model() -> None:
    figure = openset_curves(_identification())

    (axes,) = figure.axes
    assert [line.get_label() for line in axes.get_lines()] == [
        "SFace",
        "SFace",
        "ArcFace",
        "ArcFace",
        "FaceNet",
        "FaceNet",
    ]
    assert [text.get_text() for text in axes.texts] == ["SFace", "ArcFace (CoreML)", "FaceNet"]
    assert axes.get_xscale() == "log"
    assert axes.get_legend() is None


def test_results_with_identification_round_trip_through_the_file(tmp_path: Path) -> None:
    results = _results()
    path = tmp_path / "results.json"

    write_results(path, results)

    assert read_results(path) == results


def test_the_committed_results_carry_a_threshold_for_every_evaluated_model() -> None:
    results = Results.model_validate_json(RESULTS.read_bytes())

    assert results.identification is not None
    evaluated = [m.model for m in results.identification.models]
    assert {m.network for m in evaluated} == {"sface", "arcface", "facenet"}
    assert [t.model for t in results.thresholds] == evaluated
    assert results.first_active_model is not None
    assert results.first_active_model.reason
