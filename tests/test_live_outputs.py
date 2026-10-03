"""The report's tables of the operating points enrollment and the live monitor run at (#49)."""

from functools import cache

import pytest

from results_files import (
    synthetic_identification,
    synthetic_live,
    synthetic_results,
    synthetic_verification,
)
from ryuk.evaluation.active import assemble, frozen_thresholds
from ryuk.evaluation.names import model_name
from ryuk.evaluation.results import Results
from ryuk.evaluation.tables import (
    same_person_notes,
    same_person_table,
    small_gallery_notes,
    small_gallery_table,
)

NAMES = [model_name(m.model) for m in synthetic_identification().models]


@cache
def results() -> Results:
    identification = synthetic_identification()
    live = synthetic_live(identification, frozen_thresholds(identification, None))
    return assemble(synthetic_verification(), identification, None, None, live)


def _cell(value: str, low: str, high: str) -> str:
    return f"{value} [{low}\N{EN DASH}{high}]"


def _rows(table: str) -> list[list[str]]:
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in table.splitlines()]


def test_the_same_person_table_has_a_column_per_model_and_a_row_per_measure() -> None:
    rows = _rows(same_person_table(results()))

    assert rows[0] == ["", *NAMES]
    assert [row[0] for row in rows[2:]] == [
        "Same-person threshold",
        "1:N threshold",
        "Own photos warned, 1 photo",
        "Own photos warned, 5 photos",
        "At the 1:N threshold, 1 photo",
        "At the 1:N threshold, 5 photos",
        "FAR, 1 photo",
        "FAR, 5 photos",
    ]
    sface = [row[1] for row in rows[2:]]
    live = results().thresholds[0]
    assert sface[:2] == ["0.300", f"{live.threshold:.3f} (best photo)"]
    # Rates in percent with their intervals, at the value's precision.
    assert sface[2:] == [
        _cell("4.00", "2.00", "8.00"),
        _cell("0.80", "0.40", "1.60"),
        _cell("20.0", "10.0", "40.0"),
        _cell("4.00", "2.00", "8.00"),
        _cell("0.10", "0.05", "0.20"),
        _cell("0.50", "0.25", "1.00"),
    ]


def test_a_learned_rule_has_no_warning_rate_at_its_one_to_many_threshold() -> None:
    identification = synthetic_identification()
    thresholds = frozen_thresholds(identification, None)
    learned = [thresholds[0].model_copy(update={"rule": "learned"}), *thresholds[1:]]
    live = synthetic_live(identification, learned)
    measured = results().model_copy(update={"live": live})

    rows = _rows(same_person_table(measured))

    assert [row[1] for row in rows[6:8]] == ["\N{EM DASH}", "\N{EM DASH}"]


def test_the_same_person_notes_state_how_each_threshold_was_chosen() -> None:
    notes = " ".join(same_person_notes(results()))

    assert "frozen at FAR 0.1% on the validation draw's 600,000 impostor pairs" in notes
    assert "1,500 mated pairs and 600,000 impostor pairs" in notes


def test_the_small_gallery_table_has_a_row_per_model_and_gallery_size() -> None:
    rows = _rows(small_gallery_table(results()))

    assert rows[0] == [
        "Model",
        "Gallery",
        "TPIR, 5 photos",
        "FPIR, 5 photos",
        "TPIR, 1 photo",
        "FPIR, 1 photo",
    ]
    body = rows[2:]
    assert [(row[0], row[1]) for row in body] == [
        (name, size) for model in NAMES for name, size in ((model, "100"), ("", "20"), ("", "5"))
    ]
    assert body[0][2:] == [
        _cell("90.0", "45.0", "100.0"),
        _cell("1.00", "0.50", "2.00"),
        _cell("85.0", "42.5", "100.0"),
        _cell("0.20", "0.10", "0.40"),
    ]


def test_a_small_watchlists_false_alarms_get_a_third_decimal() -> None:
    rows = _rows(small_gallery_table(results()))

    # FPIR 0.05% with every photo and 0.01% with one, on galleries of 5 identities.
    assert [rows[4][3], rows[4][5]] == [
        _cell("0.050", "0.025", "0.100"),
        _cell("0.010", "0.005", "0.020"),
    ]
    assert rows[4][2] == _cell("90.0", "45.0", "100.0")


def test_the_small_gallery_notes_say_how_the_gallery_was_split() -> None:
    notes = " ".join(small_gallery_notes(results()))

    assert "split at random (seed 49) into disjoint galleries" in notes
    assert "each identity's first enrolled photo" in notes


def test_the_live_tables_need_the_live_operating_points() -> None:
    with pytest.raises(ValueError, match="ryuk evaluate live"):
        same_person_table(synthetic_results())
    with pytest.raises(ValueError, match="ryuk evaluate live"):
        small_gallery_table(synthetic_results())
