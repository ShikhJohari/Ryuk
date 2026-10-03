"""`evaluation/results.json`: every measured number the notebooks, service, client and report read.

The model here is the schema. `evaluation/results.schema.json` is generated from it and committed,
and CI checks that the committed results validate against it (#18). Only the evaluation commands
write the file; it is never edited by hand. Rates and accuracies are fractions in [0, 1]; the
gap to a published figure is in percentage points, the unit those figures are quoted in.
"""

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self, assert_never

import numpy as np
from numpy.typing import NDArray
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from ryuk.eda.summary import Draw, Split
from ryuk.fetch.pinned import write_into_place
from ryuk.recognition import ModelKey, Network, Provider
from ryuk.recognition.faces import Crop

type FloatArray = NDArray[np.float64]
type Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
type Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type Cosine = Annotated[float, Field(ge=-1.0, le=1.0)]
type MatchScore = Annotated[float, Field(ge=-1.0, le=1.0)]
"""A live rule's match score: a cosine under best-photo and mean, and under learned the logistic
regression's output, P(match) in `LearnedRule`'s formula."""
type Difference = Annotated[float, Field(ge=-1.0, le=1.0)]
"""One rate minus another."""


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
    threshold: float | None
    """On the method's own score; None when only accepting nothing meets the target."""
    indicative: bool
    """True when the target rests on a handful of false alarms (#9: FPIR 0.1%)."""


class AtThreshold(_Record):
    """The rates at a model's frozen threshold: a match iff the top candidate scores at or above
    it. TPIR counts a mated probe whose top candidate is right, misidentification one whose top
    candidate is wrong; FPIR counts non-mated probes."""

    threshold: float
    """On the method's own score: a cosine for a scoring rule, a classifier's class score, or
    the learned rule's match score."""
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
    """One recognition model on one draw under one method: the best-photo rule in
    identification, each method of #10's comparison in learning."""

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


type MatchRule = Literal["best-photo", "mean", "learned"]
"""How a match score is computed from a probe and the watchlist, for the rules that can run live
because they need no retraining when the watchlist changes (#10):

- `best-photo`: the cosine to the candidate's best enrolled photo, the baseline;
- `mean`: the cosine to the renormalised mean of the candidate's enrolled embeddings;
- `learned`: the learned decision rule's match score for the best-photo top candidate, a logistic
  regression on its cosine and its gap to the runner-up (`LearnedRule`).

A model runs live under one, recorded in `thresholds`."""

type Method = Literal["best-photo", "mean", "knn", "logistic-regression", "linear-svm", "learned"]
"""Every method #10 compares: the two scoring rules, the three gallery-trained classifiers and
the learned decision rule."""

type MethodFamily = Literal["scoring-rule", "classifier", "learned-rule"]


def match_rule(method: Method) -> MatchRule | None:
    """The rule `method` runs as live, or None for a classifier: it needs retraining whenever the
    watchlist changes, so it can never run live (#10)."""
    match method:
        case "best-photo" | "mean" | "learned":
            return method
        case "knn" | "logistic-regression" | "linear-svm":
            return None
        case _:
            assert_never(method)


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


class LearnedRule(_Record):
    """#10's learned decision rule (c) for one recognition model: a logistic regression on the
    best-photo top candidate's score and its gap to the runner-up, fitted on the validation
    draw's probes. Its match score is the regression's output,

        P(match) = 1 / (1 + exp(-(intercept + top_score * top + gap * (top - runner_up)))),

    and its threshold is a cut-off on that match score.

    The rule must never score the runner-up above the best-photo top candidate it judges, or
    evaluation and the live monitor would disagree about who the top candidate is. The live
    monitor scores the runner-up with the negative of the top's gap, so the top's logit exceeds
    the runner-up's by (top - runner_up) * (top_score + 2 * gap): coefficients with
    top_score + 2 * gap below 0 are refused."""

    intercept: float
    top_score: float
    """The coefficient of the top candidate's best-photo cosine."""
    gap: float
    """The coefficient of the top candidate's cosine minus the runner-up's."""
    folds: Annotated[int, Field(ge=2)]
    """Identity-grouped cross-fitting folds; the cut-off comes from their out-of-fold output."""

    @model_validator(mode="after")
    def _keeps_the_top_on_top(self) -> Self:
        if self.top_score + 2 * self.gap < 0:
            raise ValueError(
                "the learned rule would score the runner-up above the top candidate: "
                f"top_score + 2 * gap is {self.top_score + 2 * self.gap:g}, below 0"
            )
        return self

    def probability(self, top: FloatArray, gap: FloatArray) -> FloatArray:
        """The match score, P(match), for each candidate's best-photo cosine and its margin over
        the best other candidate: the one formula evaluation and the live monitor both use."""
        z = self.intercept + self.top_score * np.asarray(top) + self.gap * np.asarray(gap)
        # 1 / (1 + exp(-z)), without overflowing for a very negative z.
        return np.asarray(np.exp(-np.logaddexp(0.0, -z)), dtype=np.float64)


class ModelThreshold(_Record):
    """A recognition model's frozen threshold under its live rule, as the service reads it at
    startup (#12)."""

    model: RecognitionModelId
    rule: MatchRule
    threshold: MatchScore
    """A cut-off on `rule`'s match score."""
    target_fpir: Fraction
    learned_rule: LearnedRule | None = None
    """The learned rule's coefficients, when `rule` is `learned`, and only then."""
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    date: datetime.date

    @model_validator(mode="after")
    def _learned_rule_iff_learned(self) -> Self:
        if (self.learned_rule is None) == (self.rule == "learned"):
            raise ValueError("a learned threshold carries its rule's coefficients, and only it")
        return self


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


class SignedInterval(_Record):
    low: Difference
    high: Difference


class Gain(_Record):
    """A method's test TPIR at the target FPIR minus the baseline's, with the 95% paired
    identity-level bootstrap interval of that difference: both methods are read in the same
    resamples of the same identities (#10)."""

    target_fpir: Fraction
    value: Difference
    ci: SignedInterval
    improves: bool
    """True iff the interval lies entirely above zero; otherwise "no measurable improvement"."""

    @model_validator(mode="after")
    def _improves_iff_above_zero(self) -> Self:
        if self.improves != (self.ci.low > 0):
            raise ValueError("a method improves on the baseline iff its gain's interval is above 0")
        return self


class Hyperparameter(_Record):
    """A classifier's hyperparameter, chosen on the validation draw and frozen (#10)."""

    name: Literal["k", "C"]
    value: Annotated[float, Field(gt=0.0)]
    candidates: list[Annotated[float, Field(gt=0.0)]]
    """Every value tried on the validation draw."""


class MethodResult(_Record):
    """One method on both draws: fitted and its threshold frozen on validation, then the test
    draw scored once."""

    method: Method
    family: MethodFamily
    needs_retraining: bool
    """True when a change to the watchlist means fitting again, so it can never go live (#10)."""
    hyperparameter: Hyperparameter | None
    """None for a method with nothing to choose."""
    threshold: float
    """Frozen at `Learning.target_fpir` on the validation draw, on the method's own score."""
    validation: DrawResult
    test: DrawResult
    gain: Gain | None
    """None for the baseline, which every other method is compared against."""

    @model_validator(mode="after")
    def _retraining_by_method(self) -> Self:
        if self.needs_retraining != (match_rule(self.method) is None):
            raise ValueError(
                f"{self.method} {'does not need' if self.needs_retraining else 'needs'} retraining "
                "when the watchlist changes"
            )
        return self


def winning_rule(methods: Sequence[MethodResult]) -> MatchRule:
    """The rule a model runs live: of the methods that need no retraining and whose gain's
    interval lies above zero, the one with the largest gain (the first of a tie); otherwise the
    baseline, best-photo."""
    improving = [
        (rule, m.gain.value)
        for m in methods
        if m.gain is not None and m.gain.improves and (rule := match_rule(m.method)) is not None
    ]
    if not improving:
        return "best-photo"
    return max(improving, key=lambda pair: pair[1])[0]


class GapHistogram(_Record):
    """How far each probe's best-photo top candidate scores above the runner-up on the test
    draw: mated probes whose top candidate is right, mated probes whose top candidate is wrong,
    and non-mated probes. `edges` has one more entry than each count list."""

    edges: list[float]
    right: list[Annotated[int, Field(ge=0)]]
    wrong: list[Annotated[int, Field(ge=0)]]
    non_mated: list[Annotated[int, Field(ge=0)]]

    @model_validator(mode="after")
    def _binned(self) -> Self:
        bins = len(self.edges) - 1
        if bins < 1 or any(len(c) != bins for c in (self.right, self.wrong, self.non_mated)):
            raise ValueError("a histogram has one count per bin and one more edge than bins")
        return self


class TopGapSample(_Record):
    """A seeded sample of validation probes as (top score, gap to runner-up), for drawing the
    learned rule's decision boundary; every probe whose top candidate is wrong is kept."""

    top: list[Cosine]
    gap: list[Annotated[float, Field(ge=0.0, le=2.0)]]
    kind: list[Literal["right", "wrong", "non-mated"]]

    @model_validator(mode="after")
    def _paired(self) -> Self:
        if not len(self.top) == len(self.gap) == len(self.kind):
            raise ValueError("top, gap and kind must be paired lists")
        return self


class LearningModel(_Record):
    """#10's comparison on one recognition model, and the rule it runs live as a result."""

    model: RecognitionModelId
    methods: list[MethodResult]
    """The baseline first, then every other method."""
    learned_rule: LearnedRule
    gaps: GapHistogram
    sample: TopGapSample
    live_rule: MatchRule
    """The baseline unless a method that needs no retraining improves on it; of several, the one
    with the largest gain (`winning_rule`)."""
    live_reason: str

    @model_validator(mode="after")
    def _baseline_first(self) -> Self:
        if not self.methods or self.methods[0].method != "best-photo":
            raise ValueError("the baseline, best-photo, comes first")
        if any(m.gain is None for m in self.methods[1:]) or self.methods[0].gain is not None:
            raise ValueError("every method but the baseline carries its gain over the baseline")
        if self.live_rule != (winner := winning_rule(self.methods)):
            raise ValueError(
                f"the live rule is {winner} by the gains measured, not {self.live_rule}"
            )
        return self

    @property
    def bias_rules(self) -> tuple[MatchRule, ...]:
        """The rules the bias breakdown covers: best-photo, then the live rule where it differs."""
        rules: tuple[MatchRule, MatchRule] = ("best-photo", self.live_rule)
        return tuple(dict.fromkeys(rules))

    def find(self, method: Method) -> MethodResult | None:
        """`method`'s result, or None if it was not compared on this model."""
        return next((result for result in self.methods if result.method == method), None)

    def method(self, method: Method) -> MethodResult:
        if (result := self.find(method)) is None:
            raise KeyError(f"no {method} result")
        return result


class DrawDigest(_Record):
    draw: Draw
    selection_sha256: Sha256


class Learning(_Record):
    """Learning on frozen embeddings (#10, #28): every method on the same two draws as
    identification, fitted on validation only, compared on the test draw."""

    provenance: Provenance
    bootstrap: Bootstrap
    draws: list[DrawDigest]
    """Identification's draws, which these must be."""
    target_fpir: Fraction
    folds: Annotated[int, Field(ge=2)]
    """Identity-grouped folds for cross-fitting the learned rule; hyperparameters are chosen on
    the whole validation draw."""
    models: list[LearningModel]

    def model(self, model: RecognitionModelId) -> LearningModel | None:
        """The comparison on `model`, or None if learning did not compare it."""
        return next((compared for compared in self.models if compared.model == model), None)


type BiasAttribute = Literal[
    "Male", "Young", "Male_and_Young", "Eyeglasses", "Wearing_Hat", "Blurry"
]

type GroupBasis = Literal["identity", "photo"]
"""`identity`: an identity's majority label over all its images, with at least the agreement
`Bias.agreement` asks for; `photo`: each probe's own label."""


class GroupRates(_Record):
    """One group's rates at the frozen threshold, each with its identity-level interval.

    TPIR and misidentification are over the mated probes of the group's gallery identities, FPIR
    over the non-mated probes of its held-out identities; for a photo condition, over the probes
    with that label. A rate is None when fewer than `Bias.min_identities` identities stand
    behind it: too few to estimate, reported rather than dropped (#10)."""

    label: str
    values: dict[str, bool]
    """The attribute values that define the group, for example {"Male": true, "Young": false}."""
    gallery_identities: Annotated[int, Field(ge=0)]
    held_out_identities: Annotated[int, Field(ge=0)]
    mated_probes: Annotated[int, Field(ge=0)]
    non_mated_probes: Annotated[int, Field(ge=0)]
    tpir: Rate | None
    misidentification: Rate | None
    fpir: Rate | None


class AttributeBreakdown(_Record):
    attribute: BiasAttribute
    basis: GroupBasis
    indicative: bool
    """True for the Male_and_Young cells (#10)."""
    mixed_gallery_identities: Annotated[int, Field(ge=0)]
    """Gallery identities left out because no label reaches the agreement; 0 for photos."""
    mixed_held_out_identities: Annotated[int, Field(ge=0)]
    groups: list[GroupRates]
    fpir_ratio: Annotated[float, Field(ge=1.0)] | None
    """The worst group's FPIR over the best's, among groups with an FPIR; None when fewer than
    two have one or the best is 0."""


class BiasModel(_Record):
    model: RecognitionModelId
    rule: MatchRule
    threshold: float
    """The rule's single frozen threshold; no group gets its own (#10)."""
    attributes: list[AttributeBreakdown]


class Bias(_Record):
    """Who the system fails (#10, #28): per-group rates on the test draw at each model's single
    frozen threshold, under the baseline rule and under its live rule where that differs."""

    provenance: Provenance
    bootstrap: Bootstrap
    draw: DrawDigest
    min_identities: Annotated[int, Field(gt=0)]
    agreement: Fraction
    models: list[BiasModel]


class Results(_Record):
    schema_version: Literal[1] = 1
    verification: Verification
    identification: Identification | None = None
    """Absent until `ryuk evaluate celeba` has run."""
    thresholds: list[ModelThreshold] = []
    """One per model in `identification`: the block the service reads."""
    first_active_model: FirstActiveModel | None = None
    learning: Learning | None = None
    """Absent until `ryuk evaluate learn` has run."""
    bias: Bias | None = None
    """Absent until `ryuk evaluate bias` has run."""

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
        if mismatch := learning_mismatch(self.identification, self.learning):
            raise ValueError(mismatch)
        if mismatch := bias_mismatch(self.identification, self.learning, self.bias):
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


def learning_mismatch(
    identification: Identification | None, learning: Learning | None
) -> str | None:
    """Why learning no longer applies to identification, or None when it does."""
    if learning is None:
        return None
    if identification is None:
        return "learning compares methods on identification's draws, and there are none"
    drawn = [
        DrawDigest(draw=d.draw, selection_sha256=d.selection_sha256) for d in identification.draws
    ]
    if learning.draws != drawn:
        return "learning was scored on other draws than identification's"
    if [m.model for m in learning.models] != [m.model for m in identification.models]:
        return "learning must cover every model identification evaluated, in order"
    for compared, rehearsed in zip(learning.models, identification.models, strict=True):
        if compared.method("best-photo").threshold != rehearsed.threshold:
            return f"{rehearsed.model.network}'s baseline threshold differs from identification's"
    return None


def bias_mismatch(
    identification: Identification | None, learning: Learning | None, bias: Bias | None
) -> str | None:
    """Why the bias breakdown no longer applies, or None when it does."""
    if bias is None:
        return None
    if identification is None or learning is None:
        return "the bias breakdown needs identification and learning"
    test = identification.selection("test")
    if bias.draw != DrawDigest(draw="test", selection_sha256=test.selection_sha256):
        return "the bias breakdown was scored on another test draw than identification's"
    expected = [(m.model, rule) for m in learning.models for rule in m.bias_rules]
    if [(m.model, m.rule) for m in bias.models] != expected:
        return "the bias breakdown must cover each model under best-photo and its live rule"
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
