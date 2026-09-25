"""`evaluation/results.json`: every measured number the notebooks, service, client and report read.

The model here is the schema. `evaluation/results.schema.json` is generated from it and committed,
and CI checks that the committed results validate against it (#18). Only the evaluation commands
write the file; it is never edited by hand. Rates and accuracies are fractions in [0, 1]; the
gap to a published figure is in percentage points, the unit those figures are quoted in.
"""

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from ryuk.fetch.pinned import write_into_place
from ryuk.recognition import Network, Provider
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


class Published(_Record):
    accuracy: Fraction
    source: str
    note: str | None = None


class LfwModel(_Record):
    """One recognition model's LFW View 2 result under the 10-fold recipe (#9)."""

    model: RecognitionModelId
    crop: Crop
    accuracy: Fraction
    """Mean of the fold accuracies."""
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


class Results(_Record):
    schema_version: Literal[1] = 1
    verification: Verification


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
