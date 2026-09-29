"""Small synthetic results files, built without data or weights: an LFW verification of three
models, a CelebA identification on 100 gallery and 300 held-out identities, learning on it and a
bias breakdown of it. Verification and identification are built once; every record is frozen."""

from collections.abc import Mapping
from datetime import UTC, datetime
from functools import cache

import numpy as np

from ryuk.eda.build import RULES
from ryuk.eda.summary import Draw, IdentityAttribute, PhotoCondition
from ryuk.evaluation.active import assemble
from ryuk.evaluation.bias import MIN_IDENTITIES, GroupLabels, LabelledProbes, bias_model
from ryuk.evaluation.openset import ScoredProbes, draw_result, freeze
from ryuk.evaluation.results import (
    Bias,
    Bootstrap,
    DetectorId,
    DrawDigest,
    DrawSelection,
    Gain,
    GapHistogram,
    Identification,
    Int8Footnote,
    LearnedRule,
    Learning,
    LearningModel,
    MatchRule,
    Method,
    MethodFamily,
    MethodResult,
    OpenSetModel,
    Provenance,
    RecognitionModelId,
    Results,
    SignedInterval,
    TopGapSample,
    Verification,
)
from ryuk.evaluation.verification import PUBLISHED, ScoredPairs, lfw_result
from ryuk.recognition import Network, Provider


def model_id(
    network: Network, provider: Provider = "cpu", dimension: int = 512
) -> RecognitionModelId:
    return RecognitionModelId(
        network=network, provider=provider, weights_sha256="a" * 64, dimension=dimension
    )


def scored_pairs(errors: int) -> ScoredPairs:
    """Two identical folds of 50 matched and 50 mismatched pairs; `errors` negatives per fold
    outscore every positive. The recipe's threshold lands just under the lowest positive, 0.6,
    so each fold misclassifies exactly `errors` pairs."""
    scores, matched, folds = [], [], []
    for fold in (0, 1):
        positives = np.linspace(0.6, 0.9, 50)
        negatives = np.linspace(-0.2, 0.3, 50)
        negatives[:errors] = 0.95
        scores += [*positives, *negatives]
        matched += [True] * 50 + [False] * 50
        folds += [fold] * 100
    return ScoredPairs(
        np.array(scores), np.array(matched), np.array(folds, dtype=np.int_), np.array([3, 0])
    )


@cache
def synthetic_verification() -> Verification:
    return Verification(
        provenance=Provenance(
            commit="0" * 40,
            dirty=False,
            generated_at=datetime(2026, 9, 25, tzinfo=UTC),
            machine="Test machine",
        ),
        detector={"weights_sha256": "e" * 64, "min_face_size": 40},  # type: ignore[arg-type]
        # 200 scored and 3 excluded, all in fold 0.
        pairs=203,
        excluded_images=["Blank_Wall/Blank_Wall_0001.jpg"],
        view_1=[],
        models=[
            lfw_result(
                model_id("sface", dimension=128), "five-point", scored_pairs(0), PUBLISHED["sface"]
            ),
            lfw_result(
                model_id("arcface", "coreml"), "five-point", scored_pairs(0), PUBLISHED["arcface"]
            ),
            lfw_result(model_id("facenet"), "box-margin-14", scored_pairs(5), PUBLISHED["facenet"]),
        ],
        sface_int8=Int8Footnote(
            model=model_id("sface", dimension=128),
            accuracy=0.99,
            standard_error=0.001,
            cosine_to_fp32_mean=0.96,
            cosine_to_fp32_min=0.93,
            faces_compared=12,
            ms_per_face_int8=11.1,
            ms_per_face_fp32=3.9,
        ),
    )


def scored_probes(draw: Draw, rng: np.random.Generator, overlap: float) -> ScoredProbes:
    """100 gallery identities x 15 mated probes, 1 in 50 misidentified, and 300 held-out
    identities x 10 non-mated probes; `overlap` pushes non-mated scores up into the mated ones."""
    mated = np.repeat(np.arange(100), 15)
    return ScoredProbes(
        draw=draw,
        mated_identity=mated,
        mated_score=rng.uniform(0.45, 0.95, mated.size),
        mated_correct=np.arange(mated.size) % 50 != 0,
        non_mated_identity=np.repeat(np.arange(1000, 1300), 10),
        non_mated_score=rng.uniform(-0.2, 0.3 + overlap, 3000),
    )


def draw_selection(draw: Draw) -> DrawSelection:
    return DrawSelection(
        draw=draw,
        split="valid" if draw == "validation" else "test",
        seed=27,
        images=5000,
        usable_images=4800,
        gallery_candidates=150,
        gallery=list(range(100)),
        held_out=list(range(1000, 1300)),
        enrolled_photos=500,
        mated_probes=1500,
        non_mated_probes=3000,
        selection_sha256="c" * 64,
    )


@cache
def synthetic_identification() -> Identification:
    rng = np.random.default_rng(27)
    verification = synthetic_verification()
    models = []
    for lfw, overlap, ms in zip(
        verification.models, (0.3, 0.2, 0.4), (4.0, 12.0, 25.0), strict=True
    ):
        frozen = freeze(scored_probes("validation", rng, overlap))
        models.append(
            OpenSetModel(
                model=lfw.model,
                crop=lfw.crop,
                rule="best-photo",
                threshold=frozen.value,
                target_fpir=frozen.target_fpir,
                validation=draw_result(scored_probes("validation", rng, overlap), frozen, seed=1),
                test=draw_result(scored_probes("test", rng, overlap), frozen, seed=1),
                ms_per_face=ms,
            )
        )
    return Identification(
        provenance=verification.provenance,
        detector=DetectorId(weights_sha256="e" * 64, min_face_size=70),
        bootstrap=Bootstrap(resamples=2000, seed=1, confidence=0.95),
        draws=[draw_selection("validation"), draw_selection("test")],
        models=models,
    )


@cache
def synthetic_results() -> Results:
    return assemble(synthetic_verification(), synthetic_identification())


def synthetic_learning(
    identification: Identification, live: Mapping[Network, MatchRule] | None = None
) -> Learning:
    """Learning on `identification`'s draws: the baseline is identification's own result, and
    the mean and learned rules are other synthetic results; each model's live rule is `live`'s,
    best-photo by default, and a live rule other than best-photo is made to improve."""
    live = live or {}
    rng = np.random.default_rng(28)
    models = []
    for rehearsed in identification.models:
        rule = live.get(rehearsed.model.network, "best-photo")
        baseline = MethodResult(
            method="best-photo",
            family="scoring-rule",
            needs_retraining=False,
            hyperparameter=None,
            threshold=rehearsed.threshold,
            validation=rehearsed.validation,
            test=rehearsed.test,
            gain=None,
        )
        others = []
        compared: tuple[tuple[Method, MethodFamily], ...] = (
            ("mean", "scoring-rule"),
            ("learned", "learned-rule"),
        )
        for method, family in compared:
            frozen = freeze(scored_probes("validation", rng, 0.1))
            improves = method == rule
            others.append(
                MethodResult(
                    method=method,
                    family=family,
                    needs_retraining=False,
                    hyperparameter=None,
                    threshold=frozen.value,
                    validation=draw_result(scored_probes("validation", rng, 0.1), frozen, seed=1),
                    test=draw_result(scored_probes("test", rng, 0.1), frozen, seed=1),
                    gain=Gain(
                        target_fpir=0.01,
                        value=0.02 if improves else 0.0,
                        ci=SignedInterval(low=0.01, high=0.03)
                        if improves
                        else SignedInterval(low=-0.01, high=0.01),
                        improves=improves,
                    ),
                )
            )
        models.append(
            LearningModel(
                model=rehearsed.model,
                methods=[baseline, *others],
                learned_rule=LearnedRule(intercept=-20.0, top_score=30.0, gap=15.0, folds=5),
                gaps=GapHistogram(
                    edges=[0.0, 0.1, 0.2], right=[3, 9], wrong=[4, 1], non_mated=[8, 2]
                ),
                sample=TopGapSample(top=[0.7, 0.3], gap=[0.2, 0.01], kind=["right", "non-mated"]),
                live_rule=rule,
                live_reason=f"{rule} for the test",
            )
        )
    return Learning(
        provenance=identification.provenance.model_copy(update={"commit": "2" * 40}),
        bootstrap=identification.bootstrap,
        draws=[
            DrawDigest(draw=d.draw, selection_sha256=d.selection_sha256)
            for d in identification.draws
        ],
        target_fpir=0.01,
        folds=5,
        models=models,
    )


def synthetic_bias(identification: Identification, learning: Learning) -> Bias:
    """A bias breakdown of each of `learning`'s models under best-photo and its live rule, at
    that rule's frozen threshold, on fresh synthetic test probes (`scored_probes`).

    The labels are made to reach every case a report shows:

    - Male: gallery identities alternate, 1 in 10 mixed; a third of held-out identities are
      male, 1 in 50 mixed.
    - Young: 4 gallery identities mixed, and only 20 not young, too few for a TPIR.
    - Eyeglasses on 1 probe image in 10 and blurry on 1 in 20, on both sides.
    - A hat on 1 mated probe in 20, but only on the non-mated probes of 20 held-out identities,
      too few for an FPIR, so Wearing_Hat has no FPIR ratio.
    """
    rng = np.random.default_rng(10)
    identities: dict[int, dict[IdentityAttribute, bool | None]] = {
        **{
            i: {
                "Male": None if i % 10 == 9 else i % 2 == 0,
                "Young": None if i % 25 == 24 else i < 80,
            }
            for i in range(100)
        },
        **{
            1000 + k: {"Male": None if k % 50 == 49 else k % 3 == 0, "Young": k % 5 != 0}
            for k in range(300)
        },
    }
    models = []
    for compared in learning.models:
        overlap = {"sface": 0.3, "arcface": 0.2, "facenet": 0.4}[compared.model.network]
        for rule in dict.fromkeys(("best-photo", compared.live_rule)):
            scored = scored_probes("test", rng, overlap)
            mated = [f"test-mated-{n}.jpg" for n in range(scored.mated_score.size)]
            non_mated = [f"test-non-mated-{n}.jpg" for n in range(scored.non_mated_score.size)]
            photos: dict[str, dict[PhotoCondition, bool]] = {
                **{
                    image: {
                        "Eyeglasses": n % 10 == 0,
                        "Wearing_Hat": n % 20 == 7,
                        "Blurry": n % 20 == 3,
                    }
                    for n, image in enumerate(mated)
                },
                **{
                    image: {
                        "Eyeglasses": n % 10 == 0,
                        "Wearing_Hat": int(scored.non_mated_identity[n]) < 1020,
                        "Blurry": n % 20 == 3,
                    }
                    for n, image in enumerate(non_mated)
                },
            }
            probes = LabelledProbes(
                scored, mated, non_mated, GroupLabels(identities=identities, photos=photos)
            )
            models.append(
                bias_model(
                    probes,
                    compared.method(rule).threshold,
                    model=compared.model,
                    rule=rule,
                    seed=identification.bootstrap.seed,
                )
            )
    test = identification.selection("test")
    return Bias(
        provenance=learning.provenance,
        bootstrap=identification.bootstrap,
        draw=DrawDigest(draw="test", selection_sha256=test.selection_sha256),
        min_identities=MIN_IDENTITIES,
        agreement=RULES.majority_agreement,
        models=models,
    )
