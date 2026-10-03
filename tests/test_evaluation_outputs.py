"""Table 1, the LFW ROC figure and the results file, from a small synthetic verification."""

import json
import re
from pathlib import Path

import jsonschema
import pytest

from results_files import synthetic_identification, synthetic_results, synthetic_verification
from ryuk.evaluation.figures import lfw_roc, openset_curves
from ryuk.evaluation.results import (
    Results,
    Verification,
    json_schema,
    read_results,
    write_results,
)
from ryuk.evaluation.tables import lfw_markdown, lfw_rows, openset_markdown, openset_rows

REPOSITORY = Path(__file__).parents[1]
RESULTS = REPOSITORY / "evaluation" / "results.json"
SCHEMA = REPOSITORY / "evaluation" / "results.schema.json"


def test_table_1_has_one_row_per_model_with_the_protocol_columns() -> None:
    rows = lfw_rows(synthetic_verification())

    assert [row.model for row in rows] == ["SFace", "ArcFace (CoreML)", "FaceNet"]
    assert [row.embedding_size for row in rows] == ["128", "512", "512"]
    assert [row.published for row in rows] == ["99.40", "99.83", "99.65"]
    # SFace and ArcFace separate perfectly: 100.00, 0.60 and 0.17 points above published.
    assert rows[0].ours == "100.00 ± 0.00 ⚑"
    assert rows[1].ours == "100.00 ± 0.00"
    assert rows[0].tar_at_far == ("100.00", "100.00")


def test_table_1_is_one_contiguous_markdown_table_then_its_notes() -> None:
    lines = lfw_markdown(synthetic_verification()).split("\n")

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
    rows = lfw_rows(synthetic_verification())
    table = lfw_markdown(synthetic_verification())

    # FaceNet: 5 of 50 negatives per fold outscore every positive, so accuracy is 95%.
    assert not rows[2].reproduces_published
    assert rows[2].ours.endswith("⚑")
    assert "a pipeline bug to find, not a result" in table


def test_the_table_notes_cover_exclusions_published_notes_and_int8() -> None:
    table = lfw_markdown(synthetic_verification())

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
    figure = lfw_roc(synthetic_verification())

    (axes,) = figure.axes
    assert [line.get_label() for line in axes.get_lines()] == [
        "SFace, 100.00%",
        "ArcFace (CoreML), 100.00%",
        "FaceNet, 95.00%",
    ]
    assert axes.get_xscale() == "log"


def test_results_round_trip_through_the_file(tmp_path: Path) -> None:
    results = Results(verification=synthetic_verification())
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
    document = json.loads(Results(verification=synthetic_verification()).model_dump_json())
    document["verification"]["models"][0]["roc"][field].append(1.0)

    with pytest.raises(ValueError, match="paired"):
        Results.model_validate(document)


def test_table_2_has_one_row_per_model_on_the_test_draw_with_the_active_one_starred() -> None:
    results = synthetic_results()
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
    lines = openset_markdown(synthetic_results()).split("\n")

    assert lines[0].startswith("|  | SFace | ArcFace (CoreML)")
    assert re.fullmatch(r"\|:-+(\|-+:){3}\|", lines[1])
    assert [line.split(" | ")[0] for line in lines[2:9]] == [
        "| Rank-1",
        "| Frozen threshold",
        "| TPIR",
        "| FPIR",
        "| Misidentification",
        "| TPIR @ FPIR 0.1% (indicative)",
        "| ms per face",
    ]
    assert lines[9] == ""
    assert lines[10].startswith(
        "CelebA test draw: 100 gallery identities with 1,500 mated probes, 300 held-out "
        "identities with 3,000 non-mated probes."
    )


def test_table_2_notes_the_wilson_check_the_active_model_and_who_is_not_eligible() -> None:
    table = openset_markdown(synthetic_results())

    assert "dependence-adjusted Wilson interval" in table
    assert "★ First active model." in table
    assert "SFace is not eligible: its LFW accuracy is +0.60 points from published." in table


def test_a_rate_is_shown_with_its_interval() -> None:
    row = openset_rows(synthetic_results())[1]

    value, interval = row.tpir.split(" ", 1)
    low, high = (float(v) for v in interval.strip("[]").split("\N{EN DASH}"))
    assert low <= float(value) <= high


def test_the_openset_figure_draws_one_curve_and_one_marked_threshold_per_model() -> None:
    figure = openset_curves(synthetic_identification())

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
    results = synthetic_results()
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


def test_the_committed_results_name_the_lfw_gates_input() -> None:
    results = Results.model_validate_json(RESULTS.read_bytes())

    assert results.first_active_model is not None
    gate = results.first_active_model.lfw_gate
    assert (gate.accuracy, gate.scored_pairs, gate.pairs) == ("scored-pairs", 5917, 6000)


def test_verification_whose_scored_and_excluded_pairs_miss_view_2s_is_refused() -> None:
    document = synthetic_verification().model_dump(mode="json")
    document["pairs"] = 204

    with pytest.raises(ValueError, match="scored 200 and excluded 3 pairs, but View 2 has 204"):
        Verification.model_validate(document)


def test_verification_whose_models_score_different_pairs_is_refused() -> None:
    document = synthetic_verification().model_dump(mode="json")
    # ArcFace loses one more pair to exclusion than SFace and FaceNet do.
    document["models"][1]["folds"][0] |= {"pairs": 99, "excluded": 4}

    with pytest.raises(ValueError, match="every model scores the same pairs"):
        Verification.model_validate(document)


def test_rates_under_10_percent_get_two_decimals_and_the_rest_one() -> None:
    row = openset_rows(synthetic_results())[1]

    assert re.fullmatch("\\d+\\.\\d \\[\\d+\\.\\d\N{EN DASH}\\d+\\.\\d\\]", row.tpir)
    assert re.fullmatch("\\d\\.\\d\\d \\[\\d\\.\\d\\d\N{EN DASH}\\d\\.\\d\\d\\]", row.fpir)
