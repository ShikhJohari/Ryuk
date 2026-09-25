"""Table 1 of the report, verification on LFW, built from the committed results (#9)."""

from dataclasses import dataclass

from ryuk.evaluation.results import LfwModel, RecognitionModelId, Verification

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
