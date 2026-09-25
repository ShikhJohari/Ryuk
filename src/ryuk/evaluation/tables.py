"""The report's tables, built from the committed results (#9): Table 1, verification on LFW, and
Table 2, watchlist search on CelebA."""

from dataclasses import dataclass

from ryuk.eda.summary import Draw
from ryuk.evaluation.bootstrap import Interval, disagree
from ryuk.evaluation.results import (
    Eligibility,
    Identification,
    LfwModel,
    Rate,
    RecognitionModelId,
    Results,
    Verification,
)
from ryuk.evaluation.results import Interval as IntervalRecord

NAMES = {"sface": "SFace", "arcface": "ArcFace", "facenet": "FaceNet"}
PROVIDERS = {"cpu": "CPU", "coreml": "CoreML"}


def model_name(model: RecognitionModelId) -> str:
    """The name a table or figure shows: the network, and the provider where it matters."""
    name = NAMES[model.network]
    return f"{name} ({PROVIDERS[model.provider]})" if model.network == "arcface" else name


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


def openset_header(identification: Identification) -> tuple[str, ...]:
    """Table 2's columns (#9), with #10's misidentification rate beside FPIR."""
    points = identification.models[0].test.operating_points if identification.models else []
    target = next((p.target_fpir for p in points if p.indicative), 0.001)
    return (
        "Model",
        "Rank-1 [95% CI]",
        "Frozen threshold",
        "TPIR at threshold [CI]",
        "FPIR at threshold [CI]",
        "Misidentification [CI]",
        f"TPIR @ FPIR {target * 100:g}% (indicative)",
        "ms per face",
    )


def openset_rows(results: Results, draw: Draw = "test") -> list[OpenSetRow]:
    identification = _identification(results)
    active = results.first_active_model.model if results.first_active_model else None
    rows = []
    for model in identification.models:
        result = model.test if draw == "test" else model.validation
        low_fpir = next((p for p in result.operating_points if p.indicative), None)
        rows.append(
            OpenSetRow(
                model=model_name(model.model) + (" ★" if model.model == active else ""),
                rank_1=_rate(result.rank_1),
                threshold=f"{model.threshold:.3f}",
                tpir=_rate(result.at_threshold.tpir),
                fpir=_rate(result.at_threshold.fpir),
                misidentification=_rate(result.at_threshold.misidentification),
                tpir_at_low_fpir="" if low_fpir is None else _percent(low_fpir.tpir.value),
                ms_per_face=f"{model.ms_per_face:.1f}",
                active=model.model == active,
            )
        )
    return rows


def openset_markdown(results: Results, draw: Draw = "test") -> str:
    """Table 2 as Markdown on one draw, then its notes: the draw, the adjusted Wilson check and
    the first active model."""
    identification = _identification(results)
    header = openset_header(identification)
    table = [
        _markdown_row(header),
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
        *(
            _markdown_row(
                (
                    row.model,
                    row.rank_1,
                    row.threshold,
                    row.tpir,
                    row.fpir,
                    row.misidentification,
                    row.tpir_at_low_fpir,
                    row.ms_per_face,
                )
            )
            for row in openset_rows(results, draw)
        ),
    ]
    selection = next(d for d in identification.draws if d.draw == draw)
    notes = [
        f"CelebA {draw} draw: {len(selection.gallery):,} gallery identities with "
        f"{selection.mated_probes:,} mated probes, {len(selection.held_out):,} held-out "
        f"identities with {selection.non_mated_probes:,} non-mated probes. Each model's "
        "threshold was frozen at FPIR 1% on the validation draw. Rates in percent with 95% "
        f"identity-level bootstrap intervals ({identification.bootstrap.resamples:,} resamples)."
    ]
    notes += wilson_notes(identification, draw)
    if results.first_active_model is not None:
        notes.append(f"★ First active model. {results.first_active_model.reason}")
        notes += [
            f"{model_name(c.model)} is not eligible: {', '.join(_failures(c))}."
            for c in results.first_active_model.candidates
            if not c.eligible
        ]
    return "\n".join(table) + "\n\n" + "\n\n".join(notes)


def wilson_notes(identification: Identification, draw: Draw = "test") -> list[str]:
    """What the dependence-adjusted Wilson check found for rates in the 1% tails (#9)."""
    checked: list[tuple[str, str, Rate]] = []
    for model in identification.models:
        result = model.test if draw == "test" else model.validation
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
        f"{model} {name} {_interval(wilson)}"
        for model, name, rate in checked
        if (wilson := rate.adjusted_wilson) is not None
        and disagree(Interval(rate.ci.low, rate.ci.high), Interval(wilson.low, wilson.high))
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


def _failures(candidate: Eligibility) -> list[str]:
    failures = []
    if not candidate.reproduces_lfw:
        failures.append(
            "LFW did not score it"
            if candidate.lfw_gap_points is None
            else f"its LFW accuracy is {candidate.lfw_gap_points:+.2f} points from published"
        )
    if not candidate.fpir_within_limit:
        failures.append(f"its test FPIR is {_percent(candidate.test_fpir)}%, over 2%")
    if not candidate.fast_enough:
        failures.append(f"it takes {candidate.ms_per_face:.1f} ms per face, over 30")
    return failures


def _identification(results: Results) -> Identification:
    if results.identification is None:
        raise ValueError("the results have no identification; run `ryuk evaluate celeba`")
    return results.identification


def _percent(value: float) -> str:
    return f"{value * 100:.2f}"


def _interval(interval: IntervalRecord) -> str:
    return f"[{_percent(interval.low)}, {_percent(interval.high)}]"


def _rate(rate: Rate) -> str:
    return f"{_percent(rate.value)} {_interval(rate.ci)}"
