"""The report's tables, built from the committed results (#9): Table 1, verification on LFW, and
Table 2, watchlist search on CelebA."""

from dataclasses import dataclass

from ryuk.eda.summary import Draw
from ryuk.evaluation.active import failures
from ryuk.evaluation.bootstrap import disagree
from ryuk.evaluation.names import model_name
from ryuk.evaluation.openset import FPIR_TARGETS
from ryuk.evaluation.results import (
    Identification,
    Interval,
    LfwModel,
    Rate,
    Results,
    Verification,
)


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
    if not checked:
        return []
    differing = [
        f"{model} {name} {_interval(wilson, _digits(wilson.low))}"
        for model, name, rate in checked
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
