"""The report's tables, built from the committed results (#9, #10): Table 1, verification on LFW;
Table 2, watchlist search on CelebA; Table 3, learning on embeddings; Table 4, one model's bias
breakdown; and Table 5, the worst-to-best FPIR ratio of every breakdown."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from ryuk.eda.summary import Draw
from ryuk.evaluation.active import failures
from ryuk.evaluation.bootstrap import disagree
from ryuk.evaluation.names import model_name
from ryuk.evaluation.openset import FPIR_TARGETS
from ryuk.evaluation.results import (
    AttributeBreakdown,
    Bias,
    BiasAttribute,
    BiasModel,
    DrawResult,
    Gain,
    GroupRates,
    Hyperparameter,
    Identification,
    Interval,
    Learning,
    LearningModel,
    LfwModel,
    MatchRule,
    Method,
    MethodResult,
    OpenSetPoint,
    Rate,
    RecognitionModelId,
    Results,
    SignedInterval,
    Verification,
)

METHOD_NAMES: Final[Mapping[Method, str]] = MappingProxyType(
    {
        "best-photo": "Best photo",
        "mean": "Mean",
        "knn": "kNN",
        "logistic-regression": "Logistic regression",
        "linear-svm": "Linear SVM",
        "learned": "Learned rule",
    }
)
"""The name a table or figure gives each of #10's methods."""

ATTRIBUTE_NAMES: Final[Mapping[BiasAttribute, str]] = MappingProxyType(
    {
        "Male": "Male",
        "Young": "Young",
        "Male_and_Young": "Male and young",
        "Eyeglasses": "Eyeglasses",
        "Wearing_Hat": "Wearing a hat",
        "Blurry": "Blurry",
    }
)
"""The name a table or figure gives each attribute of the bias breakdown."""

TOO_FEW: Final = "too few"
"""A group's rate with too few identities behind it to estimate (#10)."""
UNDEFINED: Final = "\N{EM DASH}"
"""An FPIR ratio with fewer than two groups' FPIR behind it, or a best FPIR of 0."""


@dataclass(frozen=True)
class LfwRow:
    model: str
    embedding_size: str
    published: str
    ours: str
    auc: str
    tar_at_far: tuple[str, ...]
    """TAR at each operating point, in the order the results list them."""
    reproduces_published: bool


def lfw_header(verification: Verification) -> tuple[str, ...]:
    """Table 1's columns; the TAR columns follow the operating points in the results."""
    points = verification.models[0].operating_points if verification.models else []
    return (
        "Model",
        "Embedding size",
        "Published LFW",
        "Our LFW (± SE)",
        "AUC",
        *(
            f"TAR @ FAR {point.target_far * 100:g}%{' (indicative)' if point.indicative else ''}"
            for point in points
        ),
    )


def lfw_rows(verification: Verification) -> list[LfwRow]:
    return [_row(model) for model in verification.models]


def _row(result: LfwModel) -> LfwRow:
    ours = f"{result.accuracy * 100:.2f} ± {result.standard_error * 100:.2f}"
    return LfwRow(
        model=model_name(result.model),
        embedding_size=str(result.model.dimension),
        published=f"{result.published.accuracy * 100:.2f}",
        ours=ours if result.reproduces_published else f"{ours} ⚑",
        auc=f"{result.auc:.4f}",
        tar_at_far=tuple(f"{point.tar * 100:.2f}" for point in result.operating_points),
        reproduces_published=result.reproduces_published,
    )


def lfw_markdown(verification: Verification) -> str:
    """Table 1 as Markdown, then its notes: flags, published-figure notes and the int8 footnote."""
    header = lfw_header(verification)
    rows = lfw_rows(verification)
    table = [
        _markdown_row(header),
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
        *(
            _markdown_row(
                (row.model, row.embedding_size, row.published, row.ours, row.auc, *row.tar_at_far)
            )
            for row in rows
        ),
    ]
    scored = sum(fold.pairs for fold in verification.models[0].folds) if rows else 0
    notes = [
        f"LFW View 2, {scored:,} of {verification.pairs:,} pairs scored, in 10 folds, threshold "
        "per fold chosen on the other nine. Accuracy and TAR in percent."
    ]
    if not all(row.reproduces_published for row in rows):
        notes.append(
            "⚑ More than 0.5 points from the published figure: a pipeline bug to find, "
            "not a result."
        )
    if verification.excluded_images:
        count = len(verification.excluded_images)
        worst = ", ".join(
            f"{model_name(model.model)} {model.accuracy_if_excluded_were_errors * 100:.2f}"
            for model in verification.models
        )
        notes.append(
            f"{count} image{'s' if count != 1 else ''} had no usable face, so "
            f"{verification.pairs - scored:,} pairs using them are not scored; the published "
            f"recipes score every pair. Counting each unscored pair as an error: {worst}."
        )
    notes += [
        f"{model_name(model.model)}: {model.published.note}"
        for model in verification.models
        if model.published.note
    ]
    notes.append(int8_footnote(verification))
    # A Markdown table's rows must be contiguous; each note is its own paragraph.
    return "\n".join(table) + "\n\n" + "\n\n".join(notes)


def _markdown_row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(cells) + " |"


def int8_footnote(verification: Verification) -> str:
    note = verification.sface_int8
    return (
        f"SFace int8 is not a row. On {verification.provenance.machine} it scores "
        f"{note.accuracy * 100:.2f} ± {note.standard_error * 100:.2f}, takes "
        f"{note.ms_per_face_int8:.1f} ms per face against fp32's {note.ms_per_face_fp32:.1f} ms, "
        f"and its embeddings drift from fp32's (cosine {note.cosine_to_fp32_mean:.3f} on "
        f"average, {note.cosine_to_fp32_min:.3f} at worst, over {note.faces_compared:,} faces). "
        "It is built for memory-constrained, integer-accelerated hardware this machine is not. "
        "crimdet shipped int8; Ryuk does not. The block-quantised int8bq variant does not load "
        "on OpenCV 5's default engine at all (#7)."
    )


@dataclass(frozen=True)
class OpenSetRow:
    model: str
    rank_1: str
    threshold: str
    tpir: str
    fpir: str
    misidentification: str
    tpir_at_low_fpir: str
    ms_per_face: str
    active: bool


def openset_measures(identification: Identification) -> tuple[str, ...]:
    """Table 2's measures (#9), with #10's misidentification rate beside FPIR, one row each.

    Every rate but rank-1 is at the frozen threshold. Models are the columns: three of them fit
    a page with an interval in every cell, where seven measures across would not.
    """
    target = next(target for target, indicative in FPIR_TARGETS if indicative)
    return (
        "Rank-1",
        "Frozen threshold",
        "TPIR",
        "FPIR",
        "Misidentification",
        f"TPIR @ FPIR {target * 100:g}% (indicative)",
        "ms per face",
    )


def openset_rows(results: Results, draw: Draw = "test") -> list[OpenSetRow]:
    identification = _identification(results)
    active = results.first_active_model.model if results.first_active_model else None
    rows = []
    for model in identification.models:
        result = model.on(draw)
        low_fpir = result.indicative_point
        rows.append(
            OpenSetRow(
                model=model_name(model.model) + (" ★" if model.model == active else ""),
                rank_1=_rate(result.rank_1),
                threshold=f"{model.threshold:.3f}",
                tpir=_rate(result.at_threshold.tpir),
                fpir=_rate(result.at_threshold.fpir),
                misidentification=_rate(result.at_threshold.misidentification),
                tpir_at_low_fpir=""
                if low_fpir is None
                else _scaled(low_fpir.tpir.value, _digits(low_fpir.tpir.value)),
                ms_per_face=f"{model.ms_per_face:.1f}",
                active=model.model == active,
            )
        )
    return rows


def openset_table(results: Results, draw: Draw = "test") -> str:
    """Table 2 alone, as Markdown, on one draw: a row per measure, a column per model."""
    rows = openset_rows(results, draw)
    cells = [
        (
            r.rank_1,
            r.threshold,
            r.tpir,
            r.fpir,
            r.misidentification,
            r.tpir_at_low_fpir,
            r.ms_per_face,
        )
        for r in rows
    ]
    return "\n".join(
        [
            _markdown_row(("", *(row.model for row in rows))),
            # Pandoc sizes a long table's columns by these dashes: the measures need the room.
            "|:" + "-" * 16 + "|" + "|".join(["-" * 12 + ":"] * len(rows)) + "|",
            *(
                _markdown_row((measure, *(model[i] for model in cells)))
                for i, measure in enumerate(openset_measures(_identification(results)))
            ),
        ]
    )


def openset_notes(results: Results, draw: Draw = "test") -> list[str]:
    """Table 2's notes: the draw, the adjusted Wilson check and the first active model."""
    identification = _identification(results)
    selection = identification.selection(draw)
    notes = [
        f"CelebA {draw} draw: {len(selection.gallery):,} gallery identities with "
        f"{selection.mated_probes:,} mated probes, {len(selection.held_out):,} held-out "
        f"identities with {selection.non_mated_probes:,} non-mated probes. Each model's "
        f"threshold was frozen at FPIR {identification.models[0].target_fpir:.0%} on the "
        "validation draw. Rates in percent with 95% "
        "identity-level bootstrap intervals in brackets "
        f"({identification.bootstrap.resamples:,} resamples); every rate but rank-1 is at the "
        "frozen threshold."
    ]
    notes += wilson_notes(identification, draw)
    if results.first_active_model is not None:
        notes.append(f"★ First active model. {results.first_active_model.reason}")
        notes += [
            f"{model_name(c.model)} is not eligible: {', '.join(failures(c))}."
            for c in results.first_active_model.eligibility
            if not c.eligible
        ]
    return notes


def openset_markdown(results: Results, draw: Draw = "test") -> str:
    """Table 2 as Markdown on one draw, then each of its notes as a paragraph."""
    # A Markdown table's rows must be contiguous; each note is its own paragraph.
    return openset_table(results, draw) + "\n\n" + "\n\n".join(openset_notes(results, draw))


def wilson_notes(identification: Identification, draw: Draw = "test") -> list[str]:
    """What the dependence-adjusted Wilson check found for rates in the 1% tails (#9)."""
    checked: list[tuple[str, str, Rate]] = []
    for model in identification.models:
        result = model.on(draw)
        named = {
            "rank-1": result.rank_1,
            "TPIR": result.at_threshold.tpir,
            "FPIR": result.at_threshold.fpir,
            "misidentification": result.at_threshold.misidentification,
        }
        checked += [
            (model_name(model.model), name, rate)
            for name, rate in named.items()
            if rate.adjusted_wilson is not None
        ]
    return _wilson_checked(checked)


def _wilson_checked(checked: Sequence[tuple[str, str, Rate]]) -> list[str]:
    """The adjusted Wilson note for rates named (whose, which rate), each with its check."""
    if not checked:
        return []
    differing = [
        f"{whose} {name} {_interval(wilson, _digits(wilson.low))}"
        for whose, name, rate in checked
        if (wilson := rate.adjusted_wilson) is not None and disagree(rate.ci, wilson)
    ]
    if not differing:
        return [
            f"The {len(checked)} rates within 1% of 0 or 100% were also checked with the "
            "dependence-adjusted Wilson interval (Fogliato et al.); it agrees with the bootstrap."
        ]
    return [
        "Where the dependence-adjusted Wilson interval (Fogliato et al.) differs meaningfully "
        f"from the bootstrap for a rate within 1% of 0 or 100%, it is: {'; '.join(differing)}."
    ]


def headline_point(result: DrawResult, target_fpir: float) -> OpenSetPoint:
    """The operating point #10 compares methods at: TPIR at `target_fpir` read off the draw's own
    curve, not the frozen threshold's rates."""
    for point in result.operating_points:
        if not point.indicative and math.isclose(point.target_fpir, target_fpir):
            return point
    raise KeyError(f"no operating point at FPIR {target_fpir:.2%}")


def method_result(results: Results, model: RecognitionModelId, method: Method) -> MethodResult:
    """One method's result on one model, for prose to quote its rates at the frozen threshold."""
    return _compared(_learning(results), model).method(method)


def learning_table(results: Results) -> str:
    """Table 3 alone, as Markdown, on the test draw: a row per method with its TPIR at the target
    FPIR, a row under each but the baseline with its gain on it, and a column per model."""
    learning = _learning(results)
    methods = list(dict.fromkeys(m.method for c in learning.models for m in c.methods))
    rows: list[tuple[str, ...]] = []
    for method in methods:
        found = [compared.find(method) for compared in learning.models]
        retrains = any(result.needs_retraining for result in found if result is not None)
        rows.append(
            (
                METHOD_NAMES[method] + (" †" if retrains else ""),
                *(
                    _headline(compared, result, learning.target_fpir)
                    for compared, result in zip(learning.models, found, strict=True)
                ),
            )
        )
        if any(result is not None and result.gain is not None for result in found):
            rows.append(
                (
                    "*Gain*",
                    *(
                        _gain(compared, result, learning.target_fpir)
                        for compared, result in zip(learning.models, found, strict=True)
                    ),
                )
            )
    return "\n".join(
        [
            _markdown_row(("", *(model_name(compared.model) for compared in learning.models))),
            # Pandoc sizes a long table's columns by these dashes, as shares of the page: a rate,
            # its interval and a star on one line, and the longest method's name on one.
            "|:" + "-" * 27 + "|" + "|".join(["-" * 24 + ":"] * len(learning.models)) + "|",
            *(_markdown_row(row) for row in rows),
        ]
    )


def learning_notes(results: Results) -> list[str]:
    """Table 3's notes: what its TPIR is, what each marker means, and how each method was fitted."""
    learning = _learning(results)
    target = f"{learning.target_fpir:.0%}"
    notes = [
        f"CelebA test draw. TPIR at FPIR {target} is read off each method's own test curve, at "
        f"the threshold that gives FPIR {target} on the test draw itself, so every method is "
        "compared at the same FPIR. At each method's threshold frozen on the validation draw, "
        "which is what the live monitor would run, test FPIR lands either side of "
        f"{target}. Rates in percent with 95% identity-level bootstrap intervals in brackets "
        f"({learning.bootstrap.resamples:,} resamples).",
        "*Gain*: the method's TPIR minus best photo's, in percentage points, with the 95% paired "
        "bootstrap interval of the difference: both methods are read in the same resamples of "
        "the same identities. **Bold**: the interval lies wholly above zero, a measurable "
        "improvement; anything else is no measurable improvement.",
    ]
    unimproved = [
        model_name(compared.model)
        for compared in learning.models
        if not any(m.gain is not None and m.gain.improves for m in compared.methods)
    ]
    if unimproved:
        whom = "any model" if len(unimproved) == len(learning.models) else _listed(unimproved)
        notes.append(f"No method measurably improves on best photo for {whom}.")
    notes.append(
        "★ The rule each model runs live. "
        + " ".join(f"{model_name(c.model)}: {c.live_reason}" for c in learning.models)
    )
    if any(m.needs_retraining for compared in learning.models for m in compared.methods):
        notes.append(
            "† Needs retraining whenever the watchlist changes, so never runs live, whatever "
            "its gain."
        )
    notes += _hyperparameter_notes(learning)
    notes.append(_learned_rule_note(learning))
    return notes


def learning_markdown(results: Results) -> str:
    """Table 3 as Markdown, then each of its notes as a paragraph."""
    # A Markdown table's rows must be contiguous; each note is its own paragraph.
    return learning_table(results) + "\n\n" + "\n\n".join(learning_notes(results))


def method_rates_table(results: Results, model: RecognitionModelId, draw: Draw = "test") -> str:
    """Every method on one model and draw at its own frozen threshold, as the live monitor would
    run it, with rank-1: a row per method. Six columns, for the notebook; the report quotes these
    rates from `method_result` instead."""
    learning = _learning(results)
    compared = _compared(learning, model)
    rows = []
    for result in compared.methods:
        on = result.test if draw == "test" else result.validation
        live = " ★" if result.method == compared.live_rule else ""
        rows.append(
            (
                METHOD_NAMES[result.method] + live + (" †" if result.needs_retraining else ""),
                _rate(on.rank_1),
                f"{result.threshold:.3f}",
                _rate(on.at_threshold.tpir),
                _rate(on.at_threshold.fpir),
                _rate(on.at_threshold.misidentification),
            )
        )
    header = ("Method", "Rank-1", "Frozen threshold", "TPIR", "FPIR", "Misidentification")
    return "\n".join(
        [
            _markdown_row(header),
            "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
            *(_markdown_row(row) for row in rows),
        ]
    )


def _headline(compared: LearningModel, result: MethodResult | None, target_fpir: float) -> str:
    if result is None:
        return ""
    cell = _rate(headline_point(result.test, target_fpir).tpir)
    return cell + (" ★" if result.method == compared.live_rule else "")


def _gain(compared: LearningModel, result: MethodResult | None, target_fpir: float) -> str:
    """A gain in points, at the precision its model's baseline TPIR is shown at."""
    if result is None or result.gain is None:
        return ""
    digits = _digits(headline_point(compared.method("best-photo").test, target_fpir).tpir.value)
    cell = _signed_interval(result.gain, digits)
    return f"**{cell}**" if result.gain.improves else cell


def _signed_interval(gain: Gain, digits: int) -> str:
    """A difference and its interval in points, each signed: "+1.2 [+0.4, +2.0]"."""
    interval: SignedInterval = gain.ci
    return (
        f"{_signed(gain.value, digits)} "
        f"[{_signed(interval.low, digits)}, {_signed(interval.high, digits)}]"
    )


def _signed(difference: float, digits: int) -> str:
    """A difference of rates in points with its sign, a true minus; no sign on a rounded zero."""
    text = f"{difference * 100:+.{digits}f}"
    if float(text) == 0:
        return f"{0:.{digits}f}"
    return text.replace("-", "\N{MINUS SIGN}")


def _hyperparameter_notes(learning: Learning) -> list[str]:
    """The hyperparameter each classifier chose per model, and the candidates it chose from."""
    chosen: dict[Method, list[tuple[str, Hyperparameter]]] = {}
    for compared in learning.models:
        for result in compared.methods:
            if result.hyperparameter is not None:
                chosen.setdefault(result.method, []).append(
                    (model_name(compared.model), result.hyperparameter)
                )
    if not chosen:
        return []
    parts = []
    for method, picks in chosen.items():
        first = picks[0][1]
        if len({picked.value for _, picked in picks}) == 1 and len(picks) > 1:
            values = f"{first.value:g} for every model"
        else:
            values = ", ".join(f"{picked.value:g} ({name})" for name, picked in picks)
        candidates = ", ".join(f"{value:g}" for value in first.candidates)
        parts.append(
            f"{_in_sentence(METHOD_NAMES[method])} {first.name} = {values}, of {candidates}"
        )
    return [
        "Hyperparameters chosen on the validation draw, each candidate trained on its gallery "
        f"and scored on its probes, then frozen: {'; '.join(parts)}. Each classifier is "
        "trained on its draw's own gallery."
    ]


def _in_sentence(name: str) -> str:
    """A table's name for a method as running text has it: "Linear SVM" as "linear SVM", but
    "kNN" as it is."""
    return name if name[:1].islower() or name[1:2].isupper() else name[:1].lower() + name[1:]


def _learned_rule_note(learning: Learning) -> str:
    folds = sorted({compared.learned_rule.folds for compared in learning.models})
    fitted = "-, ".join(map(str, folds)) + "-fold"
    coefficients = "; ".join(
        f"{model_name(c.model)} intercept {_coefficient(c.learned_rule.intercept)}, top score "
        f"{_coefficient(c.learned_rule.top_score)}, gap {_coefficient(c.learned_rule.gap)}"
        for c in learning.models
    )
    return (
        "The learned rule is a logistic regression on the top candidate's best-photo score and "
        "its gap to the runner-up, fitted on the validation draw's probes with "
        f"{fitted} identity-grouped cross-fitting; its cut-off on P(match) comes from the "
        f"out-of-fold output. Coefficients: {coefficients}."
    )


def _coefficient(value: float) -> str:
    return f"{value:.2f}".replace("-", "\N{MINUS SIGN}")


def bias_breakdown(
    results: Results, model: RecognitionModelId | None = None, rule: MatchRule | None = None
) -> BiasModel:
    """One model's breakdown under one rule: by default the first active model (or the first
    model, when none is) under the rule it runs live."""
    bias = _bias(results)
    if model is None:
        active = results.first_active_model
        model = active.model if active is not None and active.model is not None else None
        model = model if model is not None else bias.models[0].model
    if rule is None:
        rule = next((t.rule for t in results.thresholds if t.model == model), "best-photo")
    for breakdown in bias.models:
        if breakdown.model == model and breakdown.rule == rule:
            return breakdown
    raise KeyError(f"no bias breakdown of {model_name(model)} under {rule}")


type Side = Literal["gallery", "held-out"]
"""Half of the bias breakdown: the gallery identities' TPIR and misidentification, or the held-out
identities' FPIR."""


def bias_table(
    results: Results,
    model: RecognitionModelId | None = None,
    rule: MatchRule | None = None,
    *,
    side: Side | None = None,
) -> str:
    """Table 4 alone, as Markdown: one model's per-group rates on the test draw at its single
    frozen threshold, the groups of each attribute under a row naming it.

    With no `side`, every column, for the notebook. A page cannot hold three rates with their
    intervals beside a group's label and its counts, so the report shows each side as its own
    table: `gallery`, the rates over the mated probes of the group's gallery identities, and
    `held-out`, FPIR over the non-mated probes of its held-out identities.
    """
    breakdown = bias_breakdown(results, model, rule)
    columns = [column for column in _BIAS_COLUMNS if side is None or column.side in (None, side)]
    rows: list[tuple[str, ...]] = []
    for attribute in breakdown.attributes:
        rows.append(
            (
                f"*{ATTRIBUTE_NAMES[attribute.attribute]}*",
                _qualifier(attribute),
                *[""] * (len(columns) - 2),
            )
        )
        rows += [
            (group.label, *(column.cell(group) for column in columns[1:]))
            for group in attribute.groups
        ]
    # The gallery side needs Pandoc to size its columns by their dashes, as shares of the page.
    # The held-out side fits at its natural width; sized to the page, its headers would wrap.
    sized = side == "gallery"
    return "\n".join(
        [
            _markdown_row(tuple(column.header for column in columns)),
            "|" + "|".join(column.rule(sized=sized) for column in columns) + "|",
            *(_markdown_row(row) for row in rows),
        ]
    )


@dataclass(frozen=True)
class _BiasColumn:
    header: str
    width: int
    """Dashes under the header: about a percentage of the page."""
    side: Side | None
    """The side of the breakdown it belongs to; None for the group's label."""
    cell: Callable[[GroupRates], str]

    def rule(self, *, sized: bool) -> str:
        """The column's part of the separator row, its dashes giving its width if `sized`."""
        dashes = "-" * (self.width if sized else 3)
        return ":" + dashes if self.side is None else dashes + ":"


_BIAS_COLUMNS: Final = (
    _BiasColumn("Group", 25, None, lambda group: group.label),
    _BiasColumn("Gallery identities", 14, "gallery", lambda g: f"{g.gallery_identities:,}"),
    _BiasColumn("Mated probes", 12, "gallery", lambda group: f"{group.mated_probes:,}"),
    _BiasColumn("TPIR", 24, "gallery", lambda group: _estimate(group.tpir)),
    _BiasColumn(
        "Misidentification", 24, "gallery", lambda group: _estimate(group.misidentification)
    ),
    _BiasColumn("Held-out identities", 14, "held-out", lambda g: f"{g.held_out_identities:,}"),
    _BiasColumn("Non-mated probes", 14, "held-out", lambda g: f"{g.non_mated_probes:,}"),
    _BiasColumn("FPIR", 24, "held-out", lambda group: _estimate(group.fpir)),
)
"""Table 4's columns, the group's label first. Each rate's column holds a rate and its interval
on one line in the report's tabular digits, and either side's columns fit a page."""


def bias_notes(
    results: Results, model: RecognitionModelId | None = None, rule: MatchRule | None = None
) -> list[str]:
    """Table 4's notes: the threshold, how the groups are made, what "too few" means and the
    adjusted Wilson check."""
    bias = _bias(results)
    breakdown = bias_breakdown(results, model, rule)
    notes = [
        f"CelebA test draw, {model_name(breakdown.model)} under "
        f"{METHOD_NAMES[breakdown.rule].lower()} at its single threshold, "
        f"{breakdown.threshold:.3f}, frozen on the validation draw: no group has its own. TPIR "
        "and misidentification are over the mated probes of each group's gallery identities, "
        "FPIR over the non-mated probes of its held-out identities. Rates in percent with 95% "
        "identity-level bootstrap intervals in brackets "
        f"({bias.bootstrap.resamples:,} resamples).",
    ]
    by_identity = [a for a in breakdown.attributes if a.basis == "identity"]
    if by_identity:
        note = (
            f"{_listed([ATTRIBUTE_NAMES[a.attribute] for a in by_identity if not a.indicative])} "
            "go by each identity's majority label over all its images in the split, usable or "
            f"not, where at least {bias.agreement:.0%} of them agree."
        )
        mixed = [
            f"{ATTRIBUTE_NAMES[a.attribute]} {a.mixed_gallery_identities:,} gallery and "
            f"{a.mixed_held_out_identities:,} held-out"
            for a in by_identity
            if a.mixed_gallery_identities or a.mixed_held_out_identities
        ]
        if mixed:
            note += f" Identities with no such majority are left out: {', '.join(mixed)}."
        notes.append(note)
    by_photo = [a for a in breakdown.attributes if a.basis == "photo"]
    if by_photo:
        notes.append(
            f"{_listed([ATTRIBUTE_NAMES[a.attribute] for a in by_photo])} go by each probe's own "
            "label, so one identity's probes can fall in both groups; its identities are those "
            "with a probe in the group."
        )
    if any(
        rate is None
        for a in breakdown.attributes
        for g in a.groups
        for rate in (g.tpir, g.misidentification, g.fpir)
    ):
        notes.append(
            f"Too few: fewer than {bias.min_identities} identities stand behind the rate, so it "
            "is not estimated; the group is still reported with its counts."
        )
    notes.append(
        "The gallery was drawn at random, not stratified; its identity counts are its composition."
    )
    notes += _wilson_checked(
        [
            (group.label, name, rate)
            for attribute in breakdown.attributes
            for group in attribute.groups
            for name, rate in (
                ("TPIR", group.tpir),
                ("misidentification", group.misidentification),
                ("FPIR", group.fpir),
            )
            if rate is not None and rate.adjusted_wilson is not None
        ]
    )
    return notes


def bias_markdown(
    results: Results, model: RecognitionModelId | None = None, rule: MatchRule | None = None
) -> str:
    """Table 4 as Markdown with every column, then each of its notes as a paragraph."""
    # A Markdown table's rows must be contiguous; each note is its own paragraph.
    notes = bias_notes(results, model, rule)
    return bias_table(results, model, rule) + "\n\n" + "\n\n".join(notes)


def _qualifier(attribute: AttributeBreakdown) -> str:
    """What sets an attribute's groups apart, beside its name in the next column, where a long
    heading would otherwise wrap."""
    qualifiers = [
        *(["indicative"] if attribute.indicative else []),
        *(["per photo"] if attribute.basis == "photo" else []),
    ]
    return f"*{', '.join(qualifiers)}*" if qualifiers else ""


def _estimate(rate: Rate | None) -> str:
    return TOO_FEW if rate is None else _rate(rate)


def fpir_extremes(attribute: AttributeBreakdown) -> tuple[GroupRates, GroupRates] | None:
    """The groups with the highest and the lowest FPIR, among those with one, for prose to name
    behind a ratio; None when fewer than two have one."""
    measured = [(group.fpir.value, group) for group in attribute.groups if group.fpir is not None]
    if len(measured) < 2:
        return None
    return (
        max(measured, key=lambda pair: pair[0])[1],
        min(measured, key=lambda pair: pair[0])[1],
    )


def breakdown_name(breakdown: BiasModel) -> str:
    """A breakdown's model, and its rule where that is not best-photo: "FaceNet, mean"."""
    name = model_name(breakdown.model)
    if breakdown.rule == "best-photo":
        return name
    return f"{name}, {METHOD_NAMES[breakdown.rule].lower()}"


def fpir_ratio_table(results: Results) -> str:
    """Table 5 alone, as Markdown: the worst-to-best FPIR ratio (NIST FRVT style), a row per
    attribute and a column per breakdown, each model under best-photo and then its live rule
    where that differs."""
    bias = _bias(results)
    attributes = list(dict.fromkeys(a.attribute for m in bias.models for a in m.attributes))
    rows = []
    for attribute in attributes:
        found = [
            next((a for a in breakdown.attributes if a.attribute == attribute), None)
            for breakdown in bias.models
        ]
        indicative = any(a is not None and a.indicative for a in found)
        rows.append(
            (
                ATTRIBUTE_NAMES[attribute] + (" (indicative)" if indicative else ""),
                *(_ratio(a) for a in found),
            )
        )
    return "\n".join(
        [
            _markdown_row(("", *(breakdown_name(breakdown) for breakdown in bias.models))),
            # Pandoc sizes a long table's columns by these dashes, as shares of the page: the
            # attributes' names on one line; a breakdown's name may wrap over its ratios.
            "|:" + "-" * 36 + "|" + "|".join(["-" * 14 + ":"] * len(bias.models)) + "|",
            *(_markdown_row(row) for row in rows),
        ]
    )


def fpir_ratio_notes(results: Results) -> list[str]:
    """Table 5's notes: what the ratio is over, and why one can be undefined."""
    bias = _bias(results)
    notes = [
        "The highest group FPIR over the lowest, per attribute, on the CelebA test draw at each "
        "model's single frozen threshold: under best photo, and under the rule the model runs "
        f"live where that differs. Only groups of at least {bias.min_identities} held-out "
        "identities have an FPIR."
    ]
    if any(a.fpir_ratio is None for m in bias.models for a in m.attributes):
        notes.append("— Undefined: fewer than two groups have an FPIR, or the lowest FPIR is 0.")
    return notes


def fpir_ratio_markdown(results: Results) -> str:
    """Table 5 as Markdown, then each of its notes as a paragraph."""
    # A Markdown table's rows must be contiguous; each note is its own paragraph.
    return fpir_ratio_table(results) + "\n\n" + "\n\n".join(fpir_ratio_notes(results))


def _ratio(attribute: AttributeBreakdown | None) -> str:
    if attribute is None:
        return ""
    return (
        UNDEFINED
        if attribute.fpir_ratio is None
        else f"{attribute.fpir_ratio:.2f}\N{MULTIPLICATION SIGN}"
    )


def _learning(results: Results) -> Learning:
    if results.learning is None:
        raise ValueError("the results have no learning; run `ryuk evaluate learn`")
    return results.learning


def _compared(learning: Learning, model: RecognitionModelId) -> LearningModel:
    if (compared := learning.model(model)) is None:
        raise KeyError(f"no learning result for {model_name(model)}")
    return compared


def _bias(results: Results) -> Bias:
    if results.bias is None:
        raise ValueError("the results have no bias breakdown; run `ryuk evaluate bias`")
    return results.bias


def _listed(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _identification(results: Results) -> Identification:
    if results.identification is None:
        raise ValueError("the results have no identification; run `ryuk evaluate celeba`")
    return results.identification


def _digits(value: float) -> int:
    """Decimals for a rate in percent: one from 10% up, two below, where a tenth is a big step."""
    return 1 if value >= 0.1 else 2


def _scaled(value: float, digits: int) -> str:
    return f"{value * 100:.{digits}f}"


def _interval(interval: Interval, digits: int) -> str:
    return f"[{_scaled(interval.low, digits)}\N{EN DASH}{_scaled(interval.high, digits)}]"


def _rate(rate: Rate) -> str:
    """A rate and its interval, all at the value's precision."""
    digits = _digits(rate.value)
    return f"{_scaled(rate.value, digits)} {_interval(rate.ci, digits)}"
