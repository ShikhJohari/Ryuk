"""The committed dataset summary, `eda/summary.json`: every number the EDA reports (#17).

`ryuk eda` writes it from the raw data; the notebook, the report and the figures read only this
file, so none of them needs the datasets. `eda/summary.schema.json` is generated from these
models and CI validates the committed summary against it.
"""

import datetime
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Final = 1

type Draw = Literal["validation", "test"]
"""A CelebA draw. The validation draw is CelebA's `valid` split, the test draw its `test` split."""

type Split = Literal["valid", "test"]
"""A CelebA split that one of the draws is."""

DRAW_SPLITS: Final[Mapping[Draw, Split]] = MappingProxyType({"validation": "valid", "test": "test"})
"""The CelebA split each draw is, in draw order: the validation draw, then the test draw."""

type IdentityAttribute = Literal["Male", "Young"]
"""An attribute of the person, given to an identity by a majority of its images' labels (#10)."""

type PhotoCondition = Literal["Eyeglasses", "Wearing_Hat", "Blurry"]
"""An attribute of the photo, which varies between one identity's images (#10)."""

type Attribute = IdentityAttribute | PhotoCondition

IDENTITY_ATTRIBUTES: Final[tuple[IdentityAttribute, ...]] = ("Male", "Young")
PHOTO_CONDITIONS: Final[tuple[PhotoCondition, ...]] = ("Eyeglasses", "Wearing_Hat", "Blurry")
ATTRIBUTES: Final[tuple[Attribute, ...]] = (*IDENTITY_ATTRIBUTES, *PHOTO_CONDITIONS)


class SummaryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Histogram(SummaryModel):
    """Counts over bins: `counts[i]` values fell in `[edges[i], edges[i + 1])`.

    The last bin is closed on the right. Values outside the edges are not counted; the enclosing
    `Distribution.n` still includes them.
    """

    edges: list[float] = Field(min_length=2)
    counts: list[int]


class Quantiles(SummaryModel):
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float


class Distribution(SummaryModel):
    """One measured quantity over a population, such as face size over a draw's detections."""

    n: int = Field(ge=0)
    mean: float | None
    """None when `n` is 0, as are the quantiles."""
    quantiles: Quantiles | None
    histogram: Histogram


class ImagesPerIdentityRow(SummaryModel):
    images: int = Field(ge=1)
    identities: int = Field(ge=1)
    """How many identities have exactly `images` images."""


class ImagesPerIdentity(SummaryModel):
    identities: int = Field(ge=0)
    images: int = Field(ge=0)
    median: float | None
    max: int | None
    table: list[ImagesPerIdentityRow]
    """The frequency table, ascending by `images`; image counts no identity has are omitted."""

    def at_least(self, images: int) -> int:
        """How many identities have `images` images or more."""
        return sum(row.identities for row in self.table if row.images >= images)


class HeadPose(SummaryModel):
    """A rough head pose fitted from the five landmarks, in degrees (see `ryuk.eda.pose`).

    YuNet pulls landmarks towards upright on rotated faces, so these are biased towards zero.
    The generic face model is fitted to YuNet's typical landmark layout, so zero pitch means the
    typical portrait in these datasets rather than a measured level head.
    """

    yaw: Distribution
    pitch: Distribution
    roll: Distribution


class DetectionStats(SummaryModel):
    """How YuNet did on one dataset or draw, at the detector's default settings."""

    images: int = Field(ge=0)
    detected: int = Field(ge=0)
    """Images with at least one detection."""
    multiple_faces: int = Field(ge=0)
    """Images with more than one detection."""
    usable: int = Field(ge=0)
    """Images with a usable face at the minimum usable face size; the rest are excluded."""
    face_short_side: Distribution
    """Short side of the centre-most detection's box, in pixels, over detected images."""
    head_pose: HeadPose
    """Pose of the centre-most detection, over detected images whose landmarks fit a pose."""


class PairList(SummaryModel):
    """One LFW pairs file."""

    name: Literal["pairsDevTrain", "pairsDevTest", "pairs"]
    view: Literal[1, 2]
    folds: int = Field(ge=1)
    matched: int = Field(ge=0)
    mismatched: int = Field(ge=0)
    identities: int = Field(ge=0)
    """Distinct identities appearing in any pair."""
    images: int = Field(ge=0)
    """Distinct images appearing in any pair."""
    pairs_with_excluded_image: int = Field(ge=0)
    """Pairs where at least one image has no usable face."""


class Lfw(SummaryModel):
    images_per_identity: ImagesPerIdentity
    pairs: list[PairList]
    detection: DetectionStats


class AttributePrevalence(SummaryModel):
    """How common one attribute is in a draw, counted per identity as well as per image (#17)."""

    attribute: Attribute
    images_with: int = Field(ge=0)
    """Images labelled with the attribute."""
    identities_with_any: int = Field(ge=0)
    """Identities with at least one image labelled with it."""
    identity_share: Histogram
    """Per identity, the share of its images labelled with it, binned over [0, 1]."""
    majority_with: int = Field(ge=0)
    """Identities with the attribute on at least `majority_agreement` of their images."""
    majority_without: int = Field(ge=0)
    """Identities without it on at least `majority_agreement` of their images."""
    mixed: int = Field(ge=0)
    """Identities in neither majority. For an identity attribute they get no group (#10)."""


class GroupStats(SummaryModel):
    """Detection and exclusion for the images of one group in one draw (#10's bias groups).

    For an identity attribute the group is the identities whose majority label is `value`, with
    all their images. For a photo condition it is the images labelled `value`, from any identity.
    """

    attribute: Attribute
    value: bool
    identities: int = Field(ge=0)
    images: int = Field(ge=0)
    detected: int = Field(ge=0)
    usable: int = Field(ge=0)
    eligible_identities: int | None = Field(ge=0)
    """Identities in the group with at least `min_gallery_images` usable images. None for a photo
    condition, which groups images rather than identities."""


class CelebaDraw(SummaryModel):
    draw: Draw
    split: Split
    images_per_identity: ImagesPerIdentity
    gallery_candidates: int = Field(ge=0)
    """Identities with at least `min_gallery_images` images before exclusion."""
    eligible_identities: int = Field(ge=0)
    """Identities with at least `min_gallery_images` usable images: those a gallery can use."""
    attributes: list[AttributePrevalence]
    detection: DetectionStats
    groups: list[GroupStats]

    def prevalence(self, attribute: Attribute) -> AttributePrevalence:
        """The draw's prevalence of `attribute`. Raises `KeyError` if the draw lacks it."""
        found = next((a for a in self.attributes if a.attribute == attribute), None)
        if found is None:
            raise KeyError(f"the {self.draw} draw has no prevalence for {attribute}")
        return found

    def group(self, attribute: Attribute, value: bool) -> GroupStats:
        """The draw's group with `attribute` equal to `value`. Raises `KeyError` if it lacks it."""
        found = next((g for g in self.groups if (g.attribute, g.value) == (attribute, value)), None)
        if found is None:
            raise KeyError(f"the {self.draw} draw has no {attribute}={value} group")
        return found


class MinUsableFaceSize(SummaryModel):
    """#17's rule: the largest round pixel value, on the box's short side, that keeps at least
    `keep` of CelebA detections (the centre-most detection of every detected image, both draws).
    """

    value: int = Field(gt=0)
    step: int = Field(gt=0)
    """What counts as round: `value` is a multiple of it."""
    keep: float = Field(gt=0, le=1)
    detections: int = Field(gt=0)
    kept: float = Field(ge=0, le=1)
    """The share of detections kept at `value`."""
    kept_at_next_step: float = Field(ge=0, le=1)
    """The share kept at `value + step`, below `keep` by construction."""


class Provenance(SummaryModel):
    generated: datetime.date
    git_commit: str
    git_dirty: bool
    """True if the working tree had uncommitted changes when the summary was written."""
    celeba_revision: str
    celeba_labels_digest: str
    lfw_archive: str
    """The LFW archive's pinned checksum, as `<algorithm> <hexdigest>`."""
    yunet_sha256: str
    score_threshold: float
    nms_threshold: float


class Rules(SummaryModel):
    """The fixed rules the summary was computed under."""

    min_gallery_images: int = Field(gt=0)
    """#9's gallery eligibility: at least this many images, counted after exclusion (#17)."""
    majority_agreement: float = Field(gt=0.5, le=1)
    """#10's majority label: an identity needs this share of its images to agree."""


class EdaSummary(SummaryModel):
    schema_version: Literal[1]
    provenance: Provenance
    rules: Rules
    min_usable_face_size: MinUsableFaceSize
    lfw: Lfw
    celeba: list[CelebaDraw]
    """The validation draw, then the test draw."""

    def draw(self, draw: Draw) -> CelebaDraw:
        """The CelebA draw `draw`. Raises `KeyError` if the summary lacks it."""
        found = next((d for d in self.celeba if d.draw == draw), None)
        if found is None:
            raise KeyError(f"the summary has no {draw} draw")
        return found
