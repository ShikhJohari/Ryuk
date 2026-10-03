"""The evaluation page's view of the committed outputs (#32): the dataset summary and the results.

A module of their own, so that the route's dependency can name `EvaluationReport` without
importing the route. The view is mapped once at startup and trimmed to what the page draws: no
method curves, gap histograms, decision-boundary samples or identity lists, and only the test
draw's curve. Rates and accuracies are fractions in [0, 1], as in the results; gaps to a
published figure are in percentage points.
"""

from collections.abc import Callable
from typing import Literal

from ryuk.api.schema import ApiModel
from ryuk.eda import summary as eda
from ryuk.eda.summary import Draw, Split
from ryuk.evaluation import results
from ryuk.evaluation.active import MAX_MS_PER_FACE, MAX_TEST_FPIR
from ryuk.evaluation.names import model_name
from ryuk.evaluation.results import BiasAttribute, GroupBasis, MatchRule, Method, MethodFamily
from ryuk.evaluation.verification import TOLERANCE_POINTS
from ryuk.recognition import ModelKey, Network, Provider
from ryuk.recognition.faces import Crop


class Interval(ApiModel):
    """A 95% interval. On a gain, a difference of rates, so `low` may be negative."""

    low: float
    high: float


class Rate(ApiModel):
    """A rate over probes or pairs with its 95% identity-level percentile bootstrap interval."""

    value: float
    ci: Interval
    adjusted_wilson: Interval | None
    """The dependence-adjusted Wilson interval, computed only for a rate within 1% of 0 or 100%;
    None otherwise."""


class ModelRef(ApiModel):
    """A recognition model as evaluation measured it."""

    id: str
    """The model key as one string, as `GET /api/models` lists it."""
    network: Network
    provider: Provider
    weights_sha256: str
    dimension: int
    """The embedding's length."""
    name: str
    """The name the report's tables give it."""


# The datasets.


class DetectionCounts(ApiModel):
    """How YuNet did on one dataset or draw, in images."""

    images: int
    detected: int
    """Images with at least one detection."""
    multiple_faces: int
    """Images with more than one detection."""
    usable: int
    """Images with a usable face; the rest are excluded from evaluation."""


class LfwPairList(ApiModel):
    name: Literal["pairsDevTrain", "pairsDevTest", "pairs"]
    view: int
    folds: int
    matched: int
    mismatched: int
    identities: int
    """Distinct identities appearing in any pair."""
    images: int
    """Distinct images appearing in any pair."""
    pairs_with_excluded_image: int
    """Pairs where at least one image has no usable face; they are not scored."""


class LfwDataset(ApiModel):
    identities: int
    images: int
    pairs: list[LfwPairList]
    detection: DetectionCounts


class CelebaDataset(ApiModel):
    """One CelebA draw before any gallery is chosen."""

    draw: Draw
    split: Split
    identities: int
    images: int
    gallery_candidates: int
    """Identities with at least `minGalleryImages` images before exclusion."""
    eligible_identities: int
    """Identities with at least `minGalleryImages` usable images: those a gallery can use."""
    detection: DetectionCounts


class DatasetSummary(ApiModel):
    min_usable_face_size: int
    """Pixels on the detection box's short side."""
    min_gallery_images: int
    lfw: LfwDataset
    celeba: list[CelebaDataset]
    """The validation draw, then the test draw."""


# LFW verification: Table 1 and the ROC figure.


class TarAtFar(ApiModel):
    """TAR at the lowest threshold whose FAR is at or below the target."""

    target_far: float
    far: float
    tar: float
    threshold: float | None
    """A cosine; None when only accepting nothing meets the target."""
    indicative: bool
    """True when the target is too small for the negative pairs to pin it down."""


class RocCurve(ApiModel):
    """FAR against TAR from (0, 0) to (1, 1), paired by index."""

    far: list[float]
    tar: list[float]


class PublishedAccuracy(ApiModel):
    accuracy: float
    source: str
    """Where the figure is quoted from, a URL."""
    note: str | None
    """What the figure was measured under, where it differs; any spread quoted with it, such as
    FaceNet's ± 0.25, is here as the source states it."""


class LfwResult(ApiModel):
    model: ModelRef
    crop: Crop
    accuracy: float
    """Mean of the 10 fold accuracies, over the pairs that could be scored."""
    standard_error: float
    """Of `accuracy`: the fold accuracies' sample standard deviation over the root of the fold
    count."""
    accuracy_if_excluded_were_errors: float
    """The same mean with every unscored pair counted as an error; reported, never decides."""
    auc: float
    published: PublishedAccuracy
    gap_points: float
    """`accuracy` minus the published accuracy, in percentage points."""
    reproduces_published: bool
    """False when the gap exceeds `tolerancePoints`."""
    operating_points: list[TarAtFar]
    roc: RocCurve


class SfaceInt8(ApiModel):
    """Why Ryuk runs SFace fp32, not int8: the int8 weights on the same View 2 pairs and faces."""

    accuracy: float
    standard_error: float
    cosine_to_fp32_mean: float
    """Mean cosine between a face's int8 and fp32 embeddings."""
    cosine_to_fp32_min: float
    faces_compared: int
    ms_per_face_int8: float
    ms_per_face_fp32: float


class VerificationReport(ApiModel):
    """LFW View 2 under the 10-fold recipe, one result per model."""

    pairs: int
    """All of View 2's pairs."""
    scored_pairs: int
    """The pairs whose images both have a usable face; every model scores the same ones."""
    tolerance_points: float
    """The largest gap to the published accuracy, in percentage points, that reproduces it."""
    models: list[LfwResult]
    sface_int8: SfaceInt8


# CelebA open-set identification: Table 2 and the TPIR-against-FPIR figure.


class OpenSetDraw(ApiModel):
    """How one CelebA draw was split into gallery and held-out identities."""

    draw: Draw
    split: Split
    gallery_identities: int
    held_out_identities: int
    enrolled_photos: int
    mated_probes: int
    non_mated_probes: int
    """The smallest FPIR the draw can measure is 1 over this."""


class TpirAtFpir(ApiModel):
    """TPIR at the lowest threshold whose FPIR on the test draw itself is at or below the target:
    a reading of the test curve, not the frozen threshold's rates."""

    target_fpir: float
    fpir: float
    tpir: Rate
    threshold: float | None
    """On the method's own score; None when only accepting nothing meets the target."""
    indicative: bool
    """True when the target rests on a handful of false alarms."""


class OpenSetCurve(ApiModel):
    """TPIR against FPIR on the test draw, paired by index, about 200 points spaced evenly in log
    FPIR; the first may have FPIR 0."""

    fpir: list[float]
    tpir: list[float]


class AtThreshold(ApiModel):
    """The test draw's rates at the threshold frozen on the validation draw."""

    tpir: Rate
    fpir: Rate
    misidentification: Rate


class OpenSetResult(ApiModel):
    """One model's watchlist rehearsal, on the test draw."""

    model: ModelRef
    crop: Crop
    rule: MatchRule
    threshold: float
    """Frozen at `targetFpir` on the validation draw."""
    target_fpir: float
    rank_1: Rate
    at_threshold: AtThreshold
    operating_points: list[TpirAtFpir]
    curve: OpenSetCurve
    ms_per_face: float
    """Warm median milliseconds from pixels to top candidate, on the machine evaluation ran on."""


class IdentificationReport(ApiModel):
    bootstrap_resamples: int
    draws: list[OpenSetDraw]
    """The validation draw, then the test draw."""
    models: list[OpenSetResult]


# The first active model.


class LfwGate(ApiModel):
    """The LFW test as the rule applied it."""

    accuracy: Literal["scored-pairs"]
    """Which accuracy is held against the published one: over the scored pairs only."""
    scored_pairs: int
    pairs: int
    tolerance_points: float


class Eligibility(ApiModel):
    """One model judged by the rule, with the numbers each test was judged on."""

    model: ModelRef
    lfw_gap_points: float | None
    """None if LFW did not score this model."""
    reproduces_lfw: bool
    test_tpir: Rate
    test_fpir: float
    fpir_within_limit: bool
    ms_per_face: float
    fast_enough: bool
    eligible: bool


class FirstActiveModel(ApiModel):
    model: ModelRef | None
    """None when no model is eligible."""
    reason: str
    lfw_gate: LfwGate
    max_test_fpir: float
    """The rule's limit on test FPIR at the frozen threshold."""
    max_ms_per_face: float
    """The rule's limit on milliseconds per face."""
    eligibility: list[Eligibility]


# Learning on embeddings.


class Hyperparameter(ApiModel):
    """A classifier's hyperparameter, chosen on the validation draw."""

    name: Literal["k", "C"]
    value: float


class Gain(ApiModel):
    """A method's test TPIR at `targetFpir` minus the baseline's, with its paired interval."""

    target_fpir: float
    value: float
    ci: Interval
    improves: bool
    """True iff the interval lies wholly above zero."""


class MethodComparison(ApiModel):
    """One method fitted and frozen on the validation draw, then scored on the test draw."""

    method: Method
    family: MethodFamily
    needs_retraining: bool
    """True when a change to the watchlist means fitting again, so it can never run live."""
    hyperparameter: Hyperparameter | None
    threshold: float
    """On the method's own score."""
    rank_1: Rate
    at_threshold: AtThreshold
    operating_points: list[TpirAtFpir]
    """The point at the learning's `targetFpir` is the TPIR methods are compared at."""
    gain: Gain | None
    """None for the baseline, best-photo."""


class LearningComparison(ApiModel):
    model: ModelRef
    live_rule: MatchRule
    live_reason: str
    methods: list[MethodComparison]
    """The baseline first."""


class LearningReport(ApiModel):
    target_fpir: float
    models: list[LearningComparison]


# The bias breakdown.


class GroupRates(ApiModel):
    """One group's test-draw rates at the model's single frozen threshold. A rate is None when
    fewer than `minIdentities` identities stand behind it."""

    label: str
    gallery_identities: int
    held_out_identities: int
    mated_probes: int
    non_mated_probes: int
    tpir: Rate | None
    misidentification: Rate | None
    fpir: Rate | None


class AttributeBreakdown(ApiModel):
    attribute: BiasAttribute
    basis: GroupBasis
    """`identity`: each identity's majority label; `photo`: each probe's own label."""
    indicative: bool
    mixed_gallery_identities: int
    """Gallery identities left out because no label reaches the agreement; 0 for photos."""
    mixed_held_out_identities: int
    fpir_ratio: float | None
    """The worst group's FPIR over the best's; None when fewer than two groups have one or the
    best is 0."""
    groups: list[GroupRates]


class BiasBreakdown(ApiModel):
    model: ModelRef
    rule: MatchRule
    threshold: float
    attributes: list[AttributeBreakdown]


class BiasReport(ApiModel):
    min_identities: int
    """The fewest identities a group's rate is estimated from."""
    agreement: float
    """The share of an identity's images that must agree for its majority label."""
    models: list[BiasBreakdown]
    """Each model under best-photo, then under its live rule where that differs."""


# The live operating points.


class SamePersonRates(ApiModel):
    """The same-person warning on the test draw with `enrolledPhotos` photos per identity."""

    enrolled_photos: int
    mated_pairs: int
    impostor_pairs: int
    warning_rate: Rate
    """Mated pairs under the threshold: a person's own photos that warn."""
    far: Rate
    """Impostor pairs at or above it: another person's photos that would not warn."""
    warning_rate_at_live_threshold: Rate | None
    """The warning rate at the model's 1:N threshold; None when that is not a cosine."""


class SamePersonThreshold(ApiModel):
    threshold: float
    """A cosine, frozen on the validation draw's impostor pairs at `targetFar`."""
    target_far: float
    validation_far: float
    validation_impostor_pairs: int
    test: list[SamePersonRates]
    """One enrolled photo first, then more, ascending."""


class SmallGallery(ApiModel):
    """The live rule at its frozen threshold on the test draw split into galleries of
    `identities` each; non-mated probes count once per gallery."""

    identities: int
    enrolled_photos: int
    """Per identity."""
    galleries: int
    mated_probes: int
    non_mated_probes: int
    tpir: Rate
    fpir: Rate
    misidentification: Rate


class LiveOperatingPoints(ApiModel):
    model: ModelRef
    rule: MatchRule
    threshold: float
    """The live rule's frozen threshold, on its own match score."""
    same_person: SamePersonThreshold
    small_galleries: list[SmallGallery]
    """The rehearsal's own gallery first, then smaller ones."""


class LiveReport(ApiModel):
    models: list[LiveOperatingPoints]


class EvaluationReport(ApiModel):
    """Everything the evaluation page shows, from `eda/summary.json` and `evaluation/results.json`.
    A section is None until the command that measures it has run."""

    dataset: DatasetSummary
    verification: VerificationReport
    identification: IdentificationReport | None
    first_active_model: FirstActiveModel | None
    learning: LearningReport | None
    bias: BiasReport | None
    live: LiveReport | None


def evaluation_report(measured: results.Results, summary: eda.EdaSummary) -> EvaluationReport:
    """The page's view of the committed results and dataset summary."""
    return EvaluationReport(
        dataset=_dataset(summary),
        verification=_verification(measured.verification),
        identification=_optional(_identification, measured.identification),
        first_active_model=_optional(_first_active_model, measured.first_active_model),
        learning=_optional(_learning, measured.learning),
        bias=_optional(_bias, measured.bias),
        live=_optional(_live, measured.live),
    )


def _optional[T, U](mapper: Callable[[T], U], section: T | None) -> U | None:
    return None if section is None else mapper(section)


def _model(model: results.RecognitionModelId) -> ModelRef:
    return ModelRef(
        id=ModelKey(model.network, model.weights_sha256, model.provider).id,
        network=model.network,
        provider=model.provider,
        weights_sha256=model.weights_sha256,
        dimension=model.dimension,
        name=model_name(model),
    )


def _interval(interval: results.Interval | results.SignedInterval) -> Interval:
    return Interval(low=interval.low, high=interval.high)


def _rate(rate: results.Rate) -> Rate:
    wilson = rate.adjusted_wilson
    return Rate(
        value=rate.value,
        ci=_interval(rate.ci),
        adjusted_wilson=None if wilson is None else _interval(wilson),
    )


def _maybe_rate(rate: results.Rate | None) -> Rate | None:
    return None if rate is None else _rate(rate)


def _detection(stats: eda.DetectionStats) -> DetectionCounts:
    return DetectionCounts(
        images=stats.images,
        detected=stats.detected,
        multiple_faces=stats.multiple_faces,
        usable=stats.usable,
    )


def _dataset(summary: eda.EdaSummary) -> DatasetSummary:
    lfw = summary.lfw
    return DatasetSummary(
        min_usable_face_size=summary.min_usable_face_size.value,
        min_gallery_images=summary.rules.min_gallery_images,
        lfw=LfwDataset(
            identities=lfw.images_per_identity.identities,
            images=lfw.images_per_identity.images,
            pairs=[
                LfwPairList(
                    name=pairs.name,
                    view=pairs.view,
                    folds=pairs.folds,
                    matched=pairs.matched,
                    mismatched=pairs.mismatched,
                    identities=pairs.identities,
                    images=pairs.images,
                    pairs_with_excluded_image=pairs.pairs_with_excluded_image,
                )
                for pairs in lfw.pairs
            ],
            detection=_detection(lfw.detection),
        ),
        celeba=[
            CelebaDataset(
                draw=draw.draw,
                split=draw.split,
                identities=draw.images_per_identity.identities,
                images=draw.images_per_identity.images,
                gallery_candidates=draw.gallery_candidates,
                eligible_identities=draw.eligible_identities,
                detection=_detection(draw.detection),
            )
            for draw in summary.celeba
        ],
    )


def _verification(verification: results.Verification) -> VerificationReport:
    int8 = verification.sface_int8
    return VerificationReport(
        pairs=verification.pairs,
        scored_pairs=verification.scored_pairs,
        tolerance_points=TOLERANCE_POINTS,
        models=[_lfw_result(model) for model in verification.models],
        sface_int8=SfaceInt8(
            accuracy=int8.accuracy,
            standard_error=int8.standard_error,
            cosine_to_fp32_mean=int8.cosine_to_fp32_mean,
            cosine_to_fp32_min=int8.cosine_to_fp32_min,
            faces_compared=int8.faces_compared,
            ms_per_face_int8=int8.ms_per_face_int8,
            ms_per_face_fp32=int8.ms_per_face_fp32,
        ),
    )


def _lfw_result(result: results.LfwModel) -> LfwResult:
    return LfwResult(
        model=_model(result.model),
        crop=result.crop,
        accuracy=result.accuracy,
        standard_error=result.standard_error,
        accuracy_if_excluded_were_errors=result.accuracy_if_excluded_were_errors,
        auc=result.auc,
        published=PublishedAccuracy(
            accuracy=result.published.accuracy,
            source=result.published.source,
            note=result.published.note,
        ),
        gap_points=result.gap_points,
        reproduces_published=result.reproduces_published,
        operating_points=[
            TarAtFar(
                target_far=point.target_far,
                far=point.far,
                tar=point.tar,
                threshold=point.threshold,
                indicative=point.indicative,
            )
            for point in result.operating_points
        ],
        roc=RocCurve(far=result.roc.far, tar=result.roc.tar),
    )


def _identification(identification: results.Identification) -> IdentificationReport:
    return IdentificationReport(
        bootstrap_resamples=identification.bootstrap.resamples,
        draws=[
            OpenSetDraw(
                draw=selection.draw,
                split=selection.split,
                gallery_identities=len(selection.gallery),
                held_out_identities=len(selection.held_out),
                enrolled_photos=selection.enrolled_photos,
                mated_probes=selection.mated_probes,
                non_mated_probes=selection.non_mated_probes,
            )
            for selection in identification.draws
        ],
        models=[
            OpenSetResult(
                model=_model(rehearsed.model),
                crop=rehearsed.crop,
                rule=rehearsed.rule,
                threshold=rehearsed.threshold,
                target_fpir=rehearsed.target_fpir,
                rank_1=_rate(rehearsed.test.rank_1),
                at_threshold=_at_threshold(rehearsed.test.at_threshold),
                operating_points=_operating_points(rehearsed.test),
                curve=OpenSetCurve(fpir=rehearsed.test.curve.fpir, tpir=rehearsed.test.curve.tpir),
                ms_per_face=rehearsed.ms_per_face,
            )
            for rehearsed in identification.models
        ],
    )


def _at_threshold(rates: results.AtThreshold) -> AtThreshold:
    return AtThreshold(
        tpir=_rate(rates.tpir),
        fpir=_rate(rates.fpir),
        misidentification=_rate(rates.misidentification),
    )


def _operating_points(result: results.DrawResult) -> list[TpirAtFpir]:
    return [
        TpirAtFpir(
            target_fpir=point.target_fpir,
            fpir=point.fpir,
            tpir=_rate(point.tpir),
            threshold=point.threshold,
            indicative=point.indicative,
        )
        for point in result.operating_points
    ]


def _first_active_model(chosen: results.FirstActiveModel) -> FirstActiveModel:
    gate = chosen.lfw_gate
    return FirstActiveModel(
        model=None if chosen.model is None else _model(chosen.model),
        reason=chosen.reason,
        lfw_gate=LfwGate(
            accuracy=gate.accuracy,
            scored_pairs=gate.scored_pairs,
            pairs=gate.pairs,
            tolerance_points=gate.tolerance_points,
        ),
        max_test_fpir=MAX_TEST_FPIR,
        max_ms_per_face=MAX_MS_PER_FACE,
        eligibility=[
            Eligibility(
                model=_model(judged.model),
                lfw_gap_points=judged.lfw_gap_points,
                reproduces_lfw=judged.reproduces_lfw,
                test_tpir=_rate(judged.test_tpir),
                test_fpir=judged.test_fpir,
                fpir_within_limit=judged.fpir_within_limit,
                ms_per_face=judged.ms_per_face,
                fast_enough=judged.fast_enough,
                eligible=judged.eligible,
            )
            for judged in chosen.eligibility
        ],
    )


def _learning(learning: results.Learning) -> LearningReport:
    return LearningReport(
        target_fpir=learning.target_fpir,
        models=[
            LearningComparison(
                model=_model(compared.model),
                live_rule=compared.live_rule,
                live_reason=compared.live_reason,
                methods=[_method(result) for result in compared.methods],
            )
            for compared in learning.models
        ],
    )


def _method(result: results.MethodResult) -> MethodComparison:
    hyperparameter = result.hyperparameter
    gain = result.gain
    return MethodComparison(
        method=result.method,
        family=result.family,
        needs_retraining=result.needs_retraining,
        hyperparameter=None
        if hyperparameter is None
        else Hyperparameter(name=hyperparameter.name, value=hyperparameter.value),
        threshold=result.threshold,
        rank_1=_rate(result.test.rank_1),
        at_threshold=_at_threshold(result.test.at_threshold),
        operating_points=_operating_points(result.test),
        gain=None
        if gain is None
        else Gain(
            target_fpir=gain.target_fpir,
            value=gain.value,
            ci=_interval(gain.ci),
            improves=gain.improves,
        ),
    )


def _bias(bias: results.Bias) -> BiasReport:
    return BiasReport(
        min_identities=bias.min_identities,
        agreement=bias.agreement,
        models=[
            BiasBreakdown(
                model=_model(broken_down.model),
                rule=broken_down.rule,
                threshold=broken_down.threshold,
                attributes=[_attribute(attribute) for attribute in broken_down.attributes],
            )
            for broken_down in bias.models
        ],
    )


def _attribute(breakdown: results.AttributeBreakdown) -> AttributeBreakdown:
    return AttributeBreakdown(
        attribute=breakdown.attribute,
        basis=breakdown.basis,
        indicative=breakdown.indicative,
        mixed_gallery_identities=breakdown.mixed_gallery_identities,
        mixed_held_out_identities=breakdown.mixed_held_out_identities,
        fpir_ratio=breakdown.fpir_ratio,
        groups=[
            GroupRates(
                label=group.label,
                gallery_identities=group.gallery_identities,
                held_out_identities=group.held_out_identities,
                mated_probes=group.mated_probes,
                non_mated_probes=group.non_mated_probes,
                tpir=_maybe_rate(group.tpir),
                misidentification=_maybe_rate(group.misidentification),
                fpir=_maybe_rate(group.fpir),
            )
            for group in breakdown.groups
        ],
    )


def _live(live: results.Live) -> LiveReport:
    return LiveReport(
        models=[
            LiveOperatingPoints(
                model=_model(measured.model),
                rule=measured.rule,
                threshold=measured.threshold,
                same_person=_same_person(measured.same_person),
                small_galleries=[
                    SmallGallery(
                        identities=gallery.identities,
                        enrolled_photos=gallery.enrolled_photos,
                        galleries=gallery.galleries,
                        mated_probes=gallery.mated_probes,
                        non_mated_probes=gallery.non_mated_probes,
                        tpir=_rate(gallery.tpir),
                        fpir=_rate(gallery.fpir),
                        misidentification=_rate(gallery.misidentification),
                    )
                    for gallery in measured.small_galleries
                ],
            )
            for measured in live.models
        ]
    )


def _same_person(same_person: results.SamePerson) -> SamePersonThreshold:
    return SamePersonThreshold(
        threshold=same_person.threshold,
        target_far=same_person.target_far,
        validation_far=same_person.validation_far,
        validation_impostor_pairs=same_person.validation_impostor_pairs,
        test=[
            SamePersonRates(
                enrolled_photos=rates.enrolled_photos,
                mated_pairs=rates.mated_pairs,
                impostor_pairs=rates.impostor_pairs,
                warning_rate=_rate(rates.warning_rate),
                far=_rate(rates.far),
                warning_rate_at_live_threshold=_maybe_rate(rates.warning_rate_at_live_threshold),
            )
            for rates in same_person.test
        ],
    )
