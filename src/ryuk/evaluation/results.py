"""`evaluation/results.json`: every measured number the notebooks, service, client and report read.

The model here is the schema. `evaluation/results.schema.json` is generated from it and committed,
and CI checks that the committed results validate against it (#18). Only the evaluation commands
write the file; it is never edited by hand. Rates and accuracies are fractions in [0, 1]; the
gap to a published figure is in percentage points, the unit those figures are quoted in.
"""

import datetime
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from ryuk.eda.summary import Draw, Split
from ryuk.fetch.pinned import write_into_place
from ryuk.recognition import ModelKey, Network, Provider
from ryuk.recognition.faces import Crop

type Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
type Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type Cosine = Annotated[float, Field(ge=-1.0, le=1.0)]


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Provenance(_Record):
    """Where and when a section was computed."""

    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    dirty: bool
    """True if the working tree had uncommitted changes, so `commit` alone cannot reproduce it."""
    generated_at: AwareDatetime
    machine: str
    """The platform the run was timed on, since latencies only hold for that machine."""


class RecognitionModelId(_Record):
    """A recognition model's identity: embeddings from two different ones never mix."""

    network: Network
    provider: Provider
    weights_sha256: Sha256
    dimension: Annotated[int, Field(gt=0)]


class DetectorId(_Record):
    weights_sha256: Sha256
    min_face_size: Annotated[int, Field(gt=0)]


class Curve(_Record):
    """A ROC curve as FAR against TAR, from (0, 0) to (1, 1), collinear points dropped."""

    far: list[Fraction]
    tar: list[Fraction]

    @model_validator(mode="after")
    def _paired(self) -> Self:
        if len(self.far) != len(self.tar) or len(self.far) < 2:
            raise ValueError("far and tar must be paired lists of at least two points")
        return self


class OperatingPoint(_Record):
    """TAR at the lowest threshold whose FAR is at or below the target."""

    target_far: Fraction
    far: Fraction
    tar: Fraction
    threshold: Cosine | None
    """None when no threshold meets the target except accepting nothing."""
    indicative: bool
    """True when the target is too small for the negatives to pin it down (#9: FAR 1e-3)."""


class Fold(_Record):
    accuracy: Fraction
    threshold: Cosine
    """The threshold chosen on the other nine folds."""
    pairs: Annotated[int, Field(gt=0)]
    """Pairs scored; a pair with an excluded image is not."""
    excluded: Annotated[int, Field(ge=0)]
    """The fold's pairs not scored because an image had no usable face."""


class Published(_Record):
    accuracy: Fraction
    source: str
    note: str | None = None


class LfwModel(_Record):
    """One recognition model's LFW View 2 result under the 10-fold recipe (#9)."""

    model: RecognitionModelId
    crop: Crop
    accuracy: Fraction
    """Mean of the fold accuracies, over the pairs that could be scored."""
    accuracy_if_excluded_were_errors: Fraction
    """The same mean with every unscored pair counted as an error: the worst the exclusions
    could be hiding. The published recipes score every pair."""
    standard_error: Annotated[float, Field(ge=0.0)]
    """Sample standard deviation of the fold accuracies over the square root of the fold count."""
    folds: list[Fold]
    auc: Fraction
    operating_points: list[OperatingPoint]
    roc: Curve
    published: Published
    gap_points: float
    """Our accuracy minus the published one, in percentage points."""
    reproduces_published: bool
    """False flags a gap over 0.5 points: a pipeline bug to fix, not a result (#9)."""


class CropTrial(_Record):
    """A pipeline choice tried on LFW View 1: threshold from DevTrain, accuracy on DevTest."""

    network: Network
    crop: Crop
    threshold: Cosine
    accuracy: Fraction
    chosen: bool


class Int8Footnote(_Record):
    """Why Ryuk runs SFace fp32, not int8 (#9): measured on the same View 2 pairs and faces."""

    model: RecognitionModelId
    accuracy: Fraction
    standard_error: Annotated[float, Field(ge=0.0)]
    cosine_to_fp32_mean: Cosine
    cosine_to_fp32_min: Cosine
    faces_compared: Annotated[int, Field(gt=0)]
    ms_per_face_int8: Annotated[float, Field(gt=0.0)]
    ms_per_face_fp32: Annotated[float, Field(gt=0.0)]
    """Warm median embedding time for an aligned face on `provenance.machine`."""


class Verification(_Record):
    """LFW View 2, scored once per model, plus the View 1 choices that shaped the pipeline."""

    provenance: Provenance
    detector: DetectorId
    pairs: Annotated[int, Field(gt=0)]
    excluded_images: list[str]
    """LFW images with no usable face, relative to `lfw_funneled`; their pairs are not scored."""
    view_1: list[CropTrial]
    models: list[LfwModel]
    sface_int8: Int8Footnote


class Interval(_Record):
    low: Fraction
    high: Fraction


class Rate(_Record):
    """A rate over probes with its 95% identity-level percentile bootstrap interval (#9)."""

    value: Fraction
    ci: Interval
    adjusted_wilson: Interval | None = None
    """Fogliato et al.'s dependence-adjusted Wilson interval, the check #9 asks for when an error
    rate is below 1%: FPIR or misidentification under 1%, or rank-1 or TPIR over 99%."""


class OpenSetPoint(_Record):
    """TPIR at the lowest threshold whose FPIR on this draw is at or below the target.

    The threshold is the draw's own, so on the test draw this is a curve reading, not the frozen
    operating point; its interval re-chooses the threshold in every resample.
    """

    target_fpir: Fraction
    fpir: Fraction
    tpir: Rate
    threshold: Cosine | None
    """None when only accepting nothing meets the target."""
    indicative: bool
    """True when the target rests on a handful of false alarms (#9: FPIR 0.1%)."""


class AtThreshold(_Record):
    """The rates at a model's frozen threshold: a match iff the top candidate scores at or above
    it. TPIR counts a mated probe whose top candidate is right, misidentification one whose top
    candidate is wrong; FPIR counts non-mated probes."""

    threshold: Cosine
    tpir: Rate
    fpir: Rate
    misidentification: Rate


class OpenSetCurve(_Record):
    """TPIR against FPIR from accepting nothing to accepting every probe, about 200 points
    spaced evenly in log FPIR; each is a point of the full curve."""

    fpir: list[Fraction]
    tpir: list[Fraction]

    @model_validator(mode="after")
    def _paired(self) -> Self:
        if len(self.fpir) != len(self.tpir) or len(self.fpir) < 2:
            raise ValueError("fpir and tpir must be paired lists of at least two points")
        return self


class DrawResult(_Record):
    """One recognition model on one draw, under the best-photo rule."""

    draw: Draw
    mated_probes: Annotated[int, Field(gt=0)]
    non_mated_probes: Annotated[int, Field(gt=0)]
    rank_1: Rate
    at_threshold: AtThreshold
    operating_points: list[OpenSetPoint]
    curve: OpenSetCurve

    @property
    def indicative_point(self) -> OpenSetPoint | None:
        """The operating point too far out in the tail to report as more than indicative."""
        return next((point for point in self.operating_points if point.indicative), None)


type MatchRule = Literal["best-photo"]
"""How a match score is computed from a probe and a candidate's enrolled photos (#10): the cosine
to the best enrolled photo. #10's comparison may add rules; a model runs live under one."""


class DrawSelection(_Record):
    """How one CelebA draw was made: its split, seed, counts and the identities in it (#9)."""

    draw: Draw
    split: Split
    seed: int
    images: Annotated[int, Field(gt=0)]
    """Images in the split."""
    usable_images: Annotated[int, Field(ge=0)]
    """Images with a usable face; the rest are excluded before the draw is made."""
    gallery_candidates: Annotated[int, Field(ge=0)]
    """Identities with enough usable images to be enrolled (5 enrolled photos and 15 probes)."""
    gallery: list[int]
    """The CelebA identities enrolled, ascending."""
    held_out: list[int]
    """Every other identity with a usable image, ascending."""
    enrolled_photos: Annotated[int, Field(gt=0)]
    mated_probes: Annotated[int, Field(gt=0)]
    non_mated_probes: Annotated[int, Field(gt=0)]
    selection_sha256: Sha256
    """A digest of every image the draw chose and its role, to check a rebuilt draw."""


class Bootstrap(_Record):
    resamples: Annotated[int, Field(gt=0)]
    seed: int
    confidence: Fraction


class OpenSetModel(_Record):
    """One recognition model's watchlist rehearsal: threshold frozen on validation, then the test
    draw scored once at it."""

    model: RecognitionModelId
    crop: Crop
    """The crop LFW View 1 chose for the network, reused unchanged."""
    rule: MatchRule
    threshold: Cosine
    target_fpir: Fraction
    validation: DrawResult
    test: DrawResult
    ms_per_face: Annotated[float, Field(gt=0.0)]
    """Warm median time to take one CelebA probe from pixels to its top candidate: detection,
    crop, embedding and the gallery search, on `provenance.machine`."""

    def on(self, draw: Draw) -> DrawResult:
        return self.test if draw == "test" else self.validation


class Identification(_Record):
    """Open-set identification on CelebA's validation and test draws (#9, #27)."""

    provenance: Provenance
    detector: DetectorId
    bootstrap: Bootstrap
    draws: list[DrawSelection]
    models: list[OpenSetModel]

    def selection(self, draw: Draw) -> DrawSelection:
        """How `draw` was made."""
        for selection in self.draws:
            if selection.draw == draw:
                return selection
        raise KeyError(f"no {draw} draw")


class ModelThreshold(_Record):
    """A recognition model's frozen threshold, as the service reads it at startup (#12)."""

    model: RecognitionModelId
    rule: MatchRule
    threshold: Cosine
    target_fpir: Fraction
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    date: datetime.date


class Eligibility(_Record):
    """#9's three tests for the first active model, with the numbers each was judged on."""

    model: RecognitionModelId
    lfw_gap_points: float | None
    """LFW accuracy minus the published figure, in points; None if LFW did not score it."""
    reproduces_lfw: bool
    test_tpir: Rate
    test_fpir: Fraction
    fpir_within_limit: bool
    ms_per_face: Annotated[float, Field(gt=0.0)]
    fast_enough: bool
    eligible: bool


class FirstActiveModel(_Record):
    """The model the live monitor starts with, and why (#9)."""

    model: RecognitionModelId | None
    """None when no model is eligible."""
    reason: str
    eligibility: list[Eligibility]
    """Every model evaluated, judged by the rule."""


class Results(_Record):
    schema_version: Literal[1] = 1
    verification: Verification
    identification: Identification | None = None
    """Absent until `ryuk evaluate celeba` has run."""
    thresholds: list[ModelThreshold] = []
    """One per model in `identification`: the block the service reads."""
    first_active_model: FirstActiveModel | None = None

    @model_validator(mode="after")
    def _thresholds_follow_identification(self) -> Self:
        evaluated = (
            [] if self.identification is None else [m.model for m in self.identification.models]
        )
        if [t.model for t in self.thresholds] != evaluated:
            raise ValueError("thresholds must list every model identification evaluated, in order")
        if (self.first_active_model is None) != (self.identification is None):
            raise ValueError("the first active model comes with identification, and only with it")
        if self.identification is not None and (
            mismatch := identification_mismatch(self.verification, self.identification)
        ):
            raise ValueError(mismatch)
        return self


def model_changes(
    loaded: Mapping[Network, ModelKey], recorded: Iterable[RecognitionModelId]
) -> list[str]:
    """How each loaded recognition model differs from the one `recorded` for its network, one
    sentence each; empty when every loaded model is one the results measured. A network with no
    recorded model is not compared."""
    by_network = {model.network: model for model in recorded}
    changes = []
    for network, key in loaded.items():
        was = by_network.get(network)
        if was is None:
            continue
        measured = ModelKey(was.network, was.weights_sha256, was.provider)
        if key != measured:
            changes.append(f"{network} loads as {key.id}, but the results measured {measured.id}")
    return changes


def identification_mismatch(
    verification: Verification, identification: Identification
) -> str | None:
    """Why CelebA's results no longer apply to what LFW scored, or None when they do."""
    crops = {result.model: result.crop for result in verification.models}
    for result in identification.models:
        if result.model not in crops:
            return f"{result.model.network} on CelebA is not a model LFW scored"
        if crops[result.model] != result.crop:
            return (
                f"{result.model.network} on CelebA used the {result.crop} crop, but LFW now "
                f"chooses {crops[result.model]}"
            )
    return None


def json_schema() -> str:
    return _dump(Results.model_json_schema())


def read_results(path: Path) -> Results | None:
    return Results.model_validate_json(path.read_bytes()) if path.is_file() else None


def write_results(path: Path, results: Results) -> None:
    text = _dump(results.model_dump(mode="json"))

    def write(part: Path) -> None:
        part.write_text(text)

    write_into_place(path, write)


def _dump(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
