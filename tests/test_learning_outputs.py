"""Tables 3 to 5 and Figures 3 to 6, learning and bias, from small synthetic results."""

import math
import re
from functools import cache

import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch
from matplotlib.text import Text

from results_files import (
    synthetic_bias,
    synthetic_identification,
    synthetic_learning,
    synthetic_verification,
)
from ryuk.evaluation.active import assemble
from ryuk.evaluation.figures import (
    KIND_STYLES,
    gap_distributions,
    group_fpir,
    learned_rules,
    learning_gains,
)
from ryuk.evaluation.results import (
    Gain,
    GapHistogram,
    Hyperparameter,
    Learning,
    LearningModel,
    Results,
    SignedInterval,
    TopGapSample,
)
from ryuk.evaluation.tables import (
    bias_breakdown,
    bias_markdown,
    bias_notes,
    bias_table,
    breakdown_name,
    fpir_extremes,
    fpir_ratio_markdown,
    fpir_ratio_table,
    headline_point,
    learning_markdown,
    learning_notes,
    learning_table,
    method_rates_table,
    method_result,
)
from ryuk.plotting.style import MATCH, NO_MATCH

MINUS = "\N{MINUS SIGN}"
DASH = "\N{EN DASH}"
RATE = re.compile(rf"(\d+\.\d+) \[(\d+\.\d+){DASH}(\d+\.\d+)\]")


def _with_knn(compared: LearningModel, k: int, rng: np.random.Generator) -> LearningModel:
    """`compared` with a kNN classifier after its mean rule, improving only on ArcFace, and a
    sample and gap histogram big enough to draw."""
    mean = compared.method("mean")
    improves = compared.model.network == "arcface"
    knn = mean.model_copy(
        update={
            "method": "knn",
            "family": "classifier",
            "needs_retraining": True,
            "hyperparameter": Hyperparameter(name="k", value=k, candidates=[3, 5, 7]),
            "gain": Gain(
                target_fpir=0.01,
                value=0.03 if improves else -0.004,
                ci=SignedInterval(low=0.01, high=0.05)
                if improves
                else SignedInterval(low=-0.02, high=0.012),
                improves=improves,
            ),
        }
    )
    methods = [compared.methods[0], mean, knn, *compared.methods[2:]]
    right = 300
    kinds = ["right"] * right + ["wrong"] * 20 + ["non-mated"] * 300
    sample = TopGapSample(
        top=[
            *rng.uniform(0.4, 0.95, right),
            *rng.uniform(0.3, 0.6, 20),
            *rng.uniform(0.0, 0.5, 300),
        ],
        gap=[*rng.uniform(0.05, 0.6, right), *rng.uniform(0.0, 0.05, 320)],
        kind=kinds,  # type: ignore[arg-type]
    )
    edges = [i / 50 for i in range(31)]
    gaps = GapHistogram(
        edges=edges,
        right=[int(400 * math.exp(-((i - 8) ** 2) / 40)) for i in range(30)],
        wrong=[12, 6, 3, 1] + [0] * 26,
        non_mated=[int(900 * math.exp(-i / 2)) for i in range(30)],
    )
    return compared.model_copy(update={"methods": methods, "sample": sample, "gaps": gaps})


@cache
def _learning() -> Learning:
    learning = synthetic_learning(synthetic_identification(), {"facenet": "mean"})
    rng = np.random.default_rng(3)
    models = [
        _with_knn(compared, k, rng) for compared, k in zip(learning.models, (3, 5, 7), strict=True)
    ]
    return learning.model_copy(update={"models": models})


@cache
def _results() -> Results:
    identification = synthetic_identification()
    learning = _learning()
    return assemble(
        synthetic_verification(),
        identification,
        learning,
        synthetic_bias(identification, learning),
    )


def _cells(table: str) -> dict[str, list[list[str]]]:
    """Each body row's cells by its label; a label that repeats keeps every row."""
    rows: dict[str, list[list[str]]] = {}
    for line in table.split("\n")[2:]:
        label, *cells = (cell.strip() for cell in line.strip("|").split("|"))
        rows.setdefault(label, []).append(cells)
    return rows


# Table 3.


def test_table_3_is_one_contiguous_markdown_table_then_its_notes() -> None:
    lines = learning_markdown(_results()).split("\n")

    assert lines[0] == "|  | SFace | ArcFace (CoreML) | FaceNet |"
    assert re.fullmatch(r"\|:-+(\|-+:){3}\|", lines[1])
    assert [line.split(" | ")[0] for line in lines[2:9]] == [
        "| Best photo",
        "| Mean",
        "| *Gain*",
        "| kNN †",
        "| *Gain*",
        "| Learned rule",
        "| *Gain*",
    ]
    assert lines[9] == ""
    assert lines[10].startswith("CelebA test draw.")


def test_table_3_reads_tpir_at_fpir_1_percent_off_each_methods_own_curve() -> None:
    results = _results()
    assert results.learning is not None
    mean = results.learning.models[0].method("mean")
    point = headline_point(mean.test, 0.01)

    assert point.target_fpir == 0.01
    assert not point.indicative
    cell = _cells(learning_table(results))["Mean"][0][0]
    match = RATE.fullmatch(cell)
    assert match is not None
    value, low, high = (float(v) for v in match.groups())
    assert value == pytest.approx(point.tpir.value * 100, abs=0.05)
    assert low <= value <= high


def test_a_gain_is_signed_in_points_and_bold_only_where_it_improves() -> None:
    gains = _cells(learning_table(_results()))["*Gain*"]

    mean, knn, _ = gains
    # SFace's mean: 0.0 [-1.0, +1.0]; ArcFace's kNN improves; FaceNet's mean is its live rule.
    assert mean[0] == f"0.0 [{MINUS}1.0, +1.0]"
    assert knn[0] == f"{MINUS}0.4 [{MINUS}2.0, +1.2]"
    assert knn[1] == "**+3.0 [+1.0, +5.0]**"
    assert mean[2] == "**+2.0 [+1.0, +3.0]**"
    assert sum(cell.startswith("**") for row in gains for cell in row) == 2


def test_each_models_live_rule_is_starred_once() -> None:
    rows = _cells(learning_table(_results()))

    assert [cell.endswith(" ★") for cell in rows["Best photo"][0]] == [True, True, False]
    assert [cell.endswith(" ★") for cell in rows["Mean"][0]] == [False, False, True]
    assert not any("★" in cell for cell in rows["kNN †"][0] + rows["Learned rule"][0])


def test_table_3_notes_explain_every_marker_and_the_fitting() -> None:
    notes = learning_notes(_results())
    text = "\n".join(notes)

    assert "read off each method's own test curve" in text
    assert "frozen on the validation draw" in text
    assert "**Bold**" in text
    assert "No method measurably improves on best photo for SFace." in text
    # The synthetic reasons are not sentences; the real ones end with a full stop.
    assert "★ The rule each model runs live. SFace: best-photo for the test" in text
    assert "FaceNet: mean for the test" in text
    assert "† Needs retraining whenever the watchlist changes, so never runs live" in text
    assert "kNN k = 3 (SFace), 5 (ArcFace (CoreML)), 7 (FaceNet), of 3, 5, 7" in text
    assert "Each classifier is trained on its draw's own gallery." in text
    assert "5-fold identity-grouped cross-fitting" in text
    assert f"SFace intercept {MINUS}20.00, top score 30.00, gap 15.00" in text


def test_a_hyperparameter_every_model_chose_is_given_once_in_running_text() -> None:
    results = _results()
    learning = _learning()
    same = [
        compared.model_copy(
            update={
                "methods": [
                    m.model_copy(
                        update={
                            "method": "logistic-regression",
                            "hyperparameter": Hyperparameter(
                                name="C", value=10.0, candidates=[1.0, 10.0]
                            ),
                        }
                    )
                    if m.method == "knn"
                    else m
                    for m in compared.methods
                ]
            }
        )
        for compared in learning.models
    ]
    notes = learning_notes(
        results.model_copy(update={"learning": learning.model_copy(update={"models": same})})
    )

    assert "frozen: logistic regression C = 10 for every model, of 1, 10." in "\n".join(notes)


def test_the_frozen_threshold_table_has_a_row_per_method_for_one_model() -> None:
    results = _results()
    assert results.learning is not None
    compared = results.learning.models[2]
    lines = method_rates_table(results, compared.model).split("\n")

    assert lines[0] == "| Method | Rank-1 | Frozen threshold | TPIR | FPIR | Misidentification |"
    rows = _cells("\n".join(lines))
    assert list(rows) == ["Best photo", "Mean ★", "kNN †", "Learned rule"]
    assert rows["Mean ★"][0][1] == f"{compared.method('mean').threshold:.3f}"
    assert RATE.fullmatch(rows["Mean ★"][0][2])


def test_a_methods_frozen_threshold_rates_can_be_quoted() -> None:
    results = _results()
    assert results.learning is not None
    compared = results.learning.models[2]

    mean = method_result(results, compared.model, "mean")

    assert mean == compared.method("mean")
    assert mean.test.at_threshold.threshold == mean.threshold


def test_the_learning_tables_need_learning() -> None:
    with pytest.raises(ValueError, match="ryuk evaluate learn"):
        learning_table(assemble(synthetic_verification(), synthetic_identification()))


# Table 4.


def test_the_bias_breakdown_defaults_to_the_first_active_model_under_its_live_rule() -> None:
    results = _results()
    active = results.first_active_model
    assert active is not None
    assert active.model is not None

    breakdown = bias_breakdown(results)

    assert breakdown.model == active.model
    live = next(t.rule for t in results.thresholds if t.model == active.model)
    assert breakdown.rule == live


def test_table_4_is_one_contiguous_table_of_groups_under_their_attributes() -> None:
    results = _results()
    lines = bias_markdown(results).split("\n")

    assert lines[0] == (
        "| Group | Gallery identities | Mated probes | TPIR | Misidentification "
        "| Held-out identities | Non-mated probes | FPIR |"
    )
    assert lines[1] == "|:---" + "|---:" * 7 + "|"
    rows = [[cell.strip() for cell in line.strip("|").split("|")][:2] for line in lines[2:22]]
    assert rows == [
        ["*Male*", ""],
        ["Male", "50"],
        ["Not male", "40"],
        ["*Young*", ""],
        ["Young", "77"],
        ["Not young", "19"],
        ["*Male and young*", "*indicative*"],
        ["Male, young", "38"],
        ["Male, not young", "10"],
        ["Not male, young", "32"],
        ["Not male, not young", "8"],
        ["*Eyeglasses*", "*per photo*"],
        ["Eyeglasses", "100"],
        ["No eyeglasses", "100"],
        ["*Wearing a hat*", "*per photo*"],
        ["Hat", "75"],
        ["No hat", "100"],
        ["*Blurry*", "*per photo*"],
        ["Blurry", "75"],
        ["Not blurry", "100"],
    ]
    assert lines[22] == ""
    assert lines[23].startswith("CelebA test draw, ")


def test_table_4_shows_counts_rates_with_intervals_and_too_few_in_words() -> None:
    results = _results()
    table = _cells(bias_table(results, synthetic_identification().models[0].model, "best-photo"))

    assert table["*Male*"] == [[""] * 7]
    gallery, mated, tpir, misidentification, held_out, non_mated, fpir = table["Male"][0]
    assert (gallery, held_out) == ("50", "98")
    assert (mated, non_mated) == ("750", "980")
    assert all(RATE.fullmatch(cell) for cell in (tpir, misidentification, fpir))
    # 19 gallery identities are not young: too few for a TPIR; 60 held out are enough.
    not_young = table["Not young"][0]
    assert not_young[:4] == ["19", "285", "too few", "too few"]
    assert RATE.fullmatch(not_young[6])
    # Only 20 held-out identities wear a hat.
    assert table["Hat"][0][4:] == ["20", "200", "too few"]


def test_each_side_of_table_4_is_a_table_of_its_own() -> None:
    results = _results()
    model = synthetic_identification().models[0].model

    gallery = bias_table(results, model, "best-photo", side="gallery").split("\n")
    held_out = bias_table(results, model, "best-photo", side="held-out").split("\n")

    assert gallery[0] == "| Group | Gallery identities | Mated probes | TPIR | Misidentification |"
    assert held_out[0] == "| Group | Held-out identities | Non-mated probes | FPIR |"
    assert len(gallery) == len(held_out) == 22
    assert _cells("\n".join(held_out))["*Eyeglasses*"] == [["*per photo*", "", ""]]
    # The gallery side's dashes share out the page; the held-out side fits at its own width.
    assert re.fullmatch(r"\|:-+(\|-+:){4}\|", gallery[1])
    assert sum(len(dashes) for dashes in re.findall("-+", gallery[1])) in range(95, 106)
    assert held_out[1] == "|:---|---:|---:|---:|"


def test_table_4_notes_the_rules_behind_the_groups() -> None:
    results = _results()
    model = synthetic_identification().models[0].model
    text = "\n".join(bias_notes(results, model, "best-photo"))

    assert "SFace under best photo at its single threshold" in text
    assert "at least 80% of them agree" in text
    assert "Male 10 gallery and 6 held-out, Young 4 gallery and 0 held-out" in text
    assert "each probe's own label" in text
    assert "Too few: fewer than 30 identities" in text
    assert "drawn at random, not stratified" in text
    assert "dependence-adjusted Wilson interval" in text


def test_prose_can_name_the_groups_behind_an_fpir_ratio() -> None:
    breakdown = bias_breakdown(_results(), synthetic_identification().models[0].model, "best-photo")
    male, _, _, _, hat, _ = breakdown.attributes

    extremes = fpir_extremes(male)

    assert extremes is not None
    worst, best = extremes
    assert worst.fpir is not None
    assert best.fpir is not None
    assert male.fpir_ratio == pytest.approx(worst.fpir.value / best.fpir.value)
    assert fpir_extremes(hat) is None
    assert breakdown_name(breakdown) == "SFace"


def test_a_breakdown_that_was_not_run_is_refused() -> None:
    model = synthetic_identification().models[0].model

    with pytest.raises(KeyError, match="SFace under mean"):
        bias_breakdown(_results(), model, "mean")


# Table 5.


def test_table_5_has_an_fpir_ratio_per_attribute_and_breakdown() -> None:
    results = _results()
    lines = fpir_ratio_table(results).split("\n")

    assert lines[0] == "|  | SFace | ArcFace (CoreML) | FaceNet | FaceNet, mean |"
    rows = _cells("\n".join(lines))
    assert list(rows) == [
        "Male",
        "Young",
        "Male and young (indicative)",
        "Eyeglasses",
        "Wearing a hat",
        "Blurry",
    ]
    assert all(re.fullmatch(r"\d+\.\d\d\N{MULTIPLICATION SIGN}", cell) for cell in rows["Male"][0])
    # Too few held-out identities wear a hat for a second FPIR.
    assert rows["Wearing a hat"][0] == ["\N{EM DASH}"] * 4


def test_table_5_notes_why_a_ratio_is_undefined() -> None:
    assert "\N{EM DASH} Undefined: fewer than two groups" in fpir_ratio_markdown(_results())


# Figures.


def test_the_gains_figure_draws_every_models_gain_with_its_interval_and_a_zero_line() -> None:
    figure = learning_gains(_learning())

    (axes,) = figure.axes
    labels = [line.get_label() for line in axes.get_lines()]
    assert labels.count("no gain") == 1
    zero = next(line for line in axes.get_lines() if line.get_label() == "no gain")
    assert list(np.asarray(zero.get_xdata())) == [0, 0]
    points = {
        str(line.get_label()): line
        for line in axes.get_lines()
        if str(line.get_label()).startswith(("SFace", "ArcFace", "FaceNet"))
    }
    # Three methods for each of three models; ArcFace's kNN and FaceNet's mean improve.
    assert {label: np.asarray(line.get_xdata()).size for label, line in points.items()} == {
        "SFace": 3,
        "ArcFace": 2,
        "ArcFace, improves": 1,
        "FaceNet": 2,
        "FaceNet, improves": 1,
    }
    assert [t.get_text() for t in axes.get_yticklabels()] == ["Mean", "kNN", "Learned rule"]
    assert len(axes.collections) == 3
    assert [t.get_text() for t in axes.texts] == [
        "SFace",
        "ArcFace (CoreML)",
        "FaceNet",
        "retrains, never live",
    ]
    assert axes.get_legend() is None


def test_the_learned_rule_figure_draws_the_sample_and_both_decision_lines_per_model() -> None:
    learning = _learning()
    figure = learned_rules(learning)

    assert len(figure.axes) == 3
    for axes, compared in zip(figure.axes, learning.models, strict=True):
        scattered = [c for c in axes.collections if isinstance(c, PathCollection)]
        assert [np.asarray(c.get_offsets()).shape[0] for c in scattered] == [300, 300, 20]
        lines = {str(line.get_label()): line for line in axes.get_lines()}
        assert set(lines) == {"learned rule", "best photo"}
        baseline = compared.method("best-photo").threshold
        assert list(np.asarray(lines["best photo"].get_xdata())) == [baseline, baseline]
        # Every point of the boundary sits where P(match) equals the learned rule's cut-off.
        rule, cut = compared.learned_rule, compared.method("learned").threshold
        x, y = (np.asarray(v, dtype=float) for v in lines["learned rule"].get_data())
        logit = rule.intercept + rule.top_score * x + rule.gap * y
        assert np.allclose(logit, math.log(cut / (1 - cut)))
    assert sorted(t.get_text() for t in figure.axes[0].texts) == [
        "best photo",
        "learned rule",
        "non-mated",
        "right",
        "wrong",
    ]
    assert not figure.axes[1].texts


def test_the_gap_figure_draws_each_kind_as_a_density() -> None:
    figure = gap_distributions(_learning())

    assert len(figure.axes) == 3
    for axes in figure.axes:
        steps = [patch for patch in axes.patches if isinstance(patch, StepPatch)]
        assert [step.get_label() for step in steps] == ["right", "non-mated", "wrong"]
        for step in steps:
            density, edges, _ = step.get_data()
            assert float(np.sum(density * np.diff(edges))) == pytest.approx(1.0)
    assert sorted(t.get_text() for t in figure.axes[0].texts) == ["non-mated", "right", "wrong"]


def test_kinds_of_probe_are_not_coloured_as_match_outcomes() -> None:
    # Match and no match are the live monitor's outcomes; a top candidate being right or wrong
    # is ground truth, which is neither.
    outcomes = {to_hex(MATCH), to_hex(NO_MATCH)}
    for figure in (learned_rules(_learning()), gap_distributions(_learning())):
        used = {
            to_hex(colour)
            for axes in figure.axes
            for collection in axes.collections
            for colour in (*collection.get_facecolor(), *collection.get_edgecolor())
        }
        used |= {to_hex(p.get_edgecolor()) for axes in figure.axes for p in axes.patches}
        used |= {to_hex(t.get_color()) for t in figure.findobj(Text) if isinstance(t, Text)}
        assert used.isdisjoint(outcomes)
    assert {to_hex(style.colour) for style in KIND_STYLES.values()}.isdisjoint(outcomes)
    # The rare wrong top candidate stands out by its marker, not by an outcome's colour.
    assert KIND_STYLES["wrong"].marker != KIND_STYLES["right"].marker


def test_the_group_fpir_figure_keeps_too_few_groups_as_labelled_gaps() -> None:
    results = _results()
    figure = group_fpir(results)

    assert len(figure.axes) == 3
    first = figure.axes[0]
    (points,) = [line for line in first.get_lines() if line.get_label() == "SFace"]
    # 14 groups, of which Male, not young and Hat have no FPIR.
    assert np.asarray(points.get_xdata()).size == 12
    too_few = [t.get_text() for t in first.texts if t.get_text().startswith("too few")]
    assert too_few == ["too few (n = 20)", "too few (n = 20)"]
    (overall,) = [line for line in first.get_lines() if line.get_label() == "overall"]
    assert results.identification is not None
    fpir = results.identification.models[0].test.at_threshold.fpir.value
    assert list(np.asarray(overall.get_xdata())) == [pytest.approx(fpir * 100)] * 2
    assert len([t for t in first.get_yticklabels() if t.get_text()]) == 14
    assert all(isinstance(line, Line2D) for line in first.get_lines())
