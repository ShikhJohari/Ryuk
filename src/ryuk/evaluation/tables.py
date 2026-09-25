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
    tar_at_far_1e2: str
    tar_at_far_1e3: str
    reproduces_published: bool


HEADER = (
    "Model",
    "Embedding size",
    "Published LFW",
    "Our LFW (± SE)",
    "AUC",
    "TAR @ FAR 1%",
    "TAR @ FAR 0.1% (indicative)",
)


def lfw_rows(verification: Verification) -> list[LfwRow]:
    return [_row(model) for model in verification.models]


def _row(result: LfwModel) -> LfwRow:
    points = {point.target_far: point.tar for point in result.operating_points}
    ours = f"{result.accuracy * 100:.2f} ± {result.standard_error * 100:.2f}"
    return LfwRow(
        model=model_name(result.model),
        embedding_size=str(result.model.dimension),
        published=f"{result.published.accuracy * 100:.2f}",
        ours=ours if result.reproduces_published else f"{ours} ⚑",
        auc=f"{result.auc:.4f}",
        tar_at_far_1e2=f"{points[1e-2] * 100:.2f}",
        tar_at_far_1e3=f"{points[1e-3] * 100:.2f}",
        reproduces_published=result.reproduces_published,
    )


def lfw_markdown(verification: Verification) -> str:
    """Table 1 as Markdown, then its notes: flags, published-figure notes and the int8 footnote."""
    rows = lfw_rows(verification)
    lines = [
        "| " + " | ".join(HEADER) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(HEADER) - 1)) + "|",
        *(
            "| "
            + " | ".join(
                (
                    row.model,
                    row.embedding_size,
                    row.published,
                    row.ours,
                    row.auc,
                    row.tar_at_far_1e2,
                    row.tar_at_far_1e3,
                )
            )
            + " |"
            for row in rows
        ),
        "",
        f"LFW View 2, {verification.pairs:,} pairs in 10 folds, threshold per fold chosen on the "
        f"other nine. Accuracy and TAR in percent.",
    ]
    if not all(row.reproduces_published for row in rows):
        lines.append(
            "⚑ More than 0.5 points from the published figure: a pipeline bug to find, "
            "not a result."
        )
    if verification.excluded_images:
        count = len(verification.excluded_images)
        lines.append(
            f"{count} image{'s' if count != 1 else ''} had no usable face; "
            "pairs using them are not scored."
        )
    for model in verification.models:
        if model.published.note:
            lines.append(f"{model_name(model.model)}: {model.published.note}")
    lines.append(int8_footnote(verification))
    return "\n\n".join(lines[:-1]) + "\n\n" + lines[-1] if len(lines) > 1 else lines[0]


def int8_footnote(verification: Verification) -> str:
    note = verification.sface_int8
    return (
        f"SFace int8 is not a row. On {verification.provenance.machine} it scores "
        f"{note.accuracy * 100:.2f} ± {note.standard_error * 100:.2f}, takes "
        f"{note.ms_per_face_int8:.1f} ms per face against fp32's {note.ms_per_face_fp32:.1f} ms, "
        f"and its embeddings drift from fp32's (cosine {note.cosine_to_fp32_mean:.3f} on "
        f"average, {note.cosine_to_fp32_min:.3f} at worst, over {note.faces_compared:,} faces). "
        "It is built for integer-accelerated hardware this machine is not."
    )
