"""#10's comparison on one recognition model: does anything learned on frozen embeddings beat the
best-photo rule? (#28)

Six methods score the same validation and test draws:

- two scoring rules, nothing fitted: best-photo, the baseline, and the mean rule;
- three classifiers trained on a draw's gallery, kNN, multinomial logistic regression and a
  linear SVM, open set by thresholding the top class's score. None has an "unknown" class: any
  source of unknown faces would leak test identities or be arbitrary;
- the learned decision rule, a logistic regression on the best-photo top score and its gap to the
  runner-up, which never sees the gallery.

Nothing is fitted on the test draw, and the code enforces it. A classifier's hyperparameter is
chosen by `choose_hyperparameter`, which refuses the test draw; the sealed token it returns is the
only way to train a classifier, and training takes a gallery's enrolled embeddings and nothing
else. On the test draw each classifier is retrained, with its frozen hyperparameter, on that
draw's own gallery: that is enrolment, as it would be live, and why a classifier can never go
live. The learned rule is fitted by `fit_learned_rule`, which also refuses the test draw, and its
cut-off is frozen on its identity-grouped out-of-fold output. Every threshold comes from `freeze`.

A method beats the baseline only if the paired bootstrap interval of its test TPIR gain at FPIR 1%
lies above zero. The live rule is the method with the largest such gain among those that need no
retraining when the watchlist changes, else best-photo.
"""

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, Self, assert_never

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from ryuk.eda.summary import Draw
from ryuk.evaluation.bootstrap import CONFIDENCE
from ryuk.evaluation.openset import (
    TARGET_FPIR,
    EmbeddedDraw,
    FrozenThreshold,
    Gallery,
    ScoredProbes,
    TopTwo,
    draw_result,
    freeze,
    paired_gain,
    score_probes,
    tpir_at_fpir,
)
from ryuk.evaluation.results import (
    Gain,
    GapHistogram,
    Hyperparameter,
    LearnedRule,
    LearningModel,
    MatchRule,
    Method,
    MethodFamily,
    MethodResult,
    RecognitionModelId,
    TopGapSample,
)
from ryuk.recognition import Embedding

logger = logging.getLogger(__name__)

type Classifier = Literal["knn", "logistic-regression", "linear-svm"]

METHODS: Final[tuple[Method, ...]] = (
    "best-photo",
    "mean",
    "knn",
    "logistic-regression",
    "linear-svm",
    "learned",
)
"""Every method compared, the baseline first."""
CLASSIFIERS: Final[tuple[Classifier, ...]] = ("knn", "logistic-regression", "linear-svm")

KNN_K: Final = (3, 5, 7, 9, 15)
"""Neighbours that vote, each weighted by its inverse distance so the top class's share is
rarely tied; on unit embeddings Euclidean distance ranks as cosine does."""
LOGISTIC_C: Final = (0.1, 1.0, 10.0, 100.0)
SVM_C: Final = (0.01, 0.1, 1.0, 10.0)
FOLDS: Final = 5
"""Identity-grouped folds for cross-fitting the learned rule."""
SAMPLE_SIZE: Final = 400
"""Validation probes sampled of each kind for drawing the learned rule's decision boundary; every
probe whose top candidate is wrong is kept, as there are few."""
SAMPLE_SEED: Final = 28
GAP_BINS_PER_UNIT: Final = 50
"""The gap histogram's bins are 0.02 wide, from 0, so every model's bins line up."""

_TOL: Final = 1e-8
"""lbfgs stops when every gradient component is under `tol`. scikit-learn minimises the mean
loss, whose gradient on 512-dimension unit embeddings starts near its default 1e-4, so the
default stops at the first step; at 1e-8 coefficients move under 0.1% if it is tightened."""
_MAX_ITER: Final = 10_000
_RANDOM_STATE: Final = 0
"""The linear SVM's dual solver visits samples in a random order."""


class _Estimator(Protocol):
    """The part of a scikit-learn classifier used here."""

    classes_: NDArray[np.int_]

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> Self: ...

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class _Spec:
    hyperparameter: Literal["k", "C"]
    candidates: tuple[float, ...]
    larger_is_simpler: bool
    """Which way a tie between candidates is broken: towards the most regularised."""
    build: Callable[[float], _Estimator]
    decision: bool
    """Scored by the top class's decision function, not its probability."""


_SPECS: Final[Mapping[Classifier, _Spec]] = {
    "knn": _Spec(
        "k",
        KNN_K,
        larger_is_simpler=True,
        build=lambda k: KNeighborsClassifier(n_neighbors=int(k), weights="distance"),
        decision=False,
    ),
    "logistic-regression": _Spec(
        "C",
        LOGISTIC_C,
        larger_is_simpler=False,
        build=lambda c: LogisticRegression(C=c, tol=_TOL, max_iter=_MAX_ITER),
        decision=False,
    ),
    "linear-svm": _Spec(
        "C",
        SVM_C,
        larger_is_simpler=False,
        # One-vs-rest. The dual solver fits a 500-identity gallery several times faster here.
        build=lambda c: LinearSVC(C=c, dual=True, max_iter=_MAX_ITER, random_state=_RANDOM_STATE),
        decision=True,
    ),
}

_FAMILIES: Final[Mapping[Method, MethodFamily]] = {
    "best-photo": "scoring-rule",
    "mean": "scoring-rule",
    "knn": "classifier",
    "logistic-regression": "classifier",
    "linear-svm": "classifier",
    "learned": "learned-rule",
}
_LIVE: Final[Mapping[Method, MatchRule]] = {
    "best-photo": "best-photo",
    "mean": "mean",
    "learned": "learned",
}
"""The methods that need no retraining when the watchlist changes, as the rule each runs live."""
_NAMES: Final[Mapping[Method, str]] = {
    "best-photo": "best-photo",
    "mean": "the mean rule",
    "knn": "kNN",
    "logistic-regression": "logistic regression",
    "linear-svm": "the linear SVM",
    "learned": "the learned rule",
}

_SEAL: Final = object()


@dataclass(frozen=True, eq=False)
class TrainedClassifier:
    """A classifier trained on one gallery's enrolled embeddings. Its top candidate for a probe
    is the top-scoring class, and its match score that class's score."""

    method: Classifier
    identities: NDArray[np.int_]
    """The gallery it was trained on, ascending."""
    estimator: _Estimator
    decision: bool

    def score(self, draw: EmbeddedDraw) -> ScoredProbes:
        """The draw whose gallery this was trained on, scored."""
        if not np.array_equal(self.identities, sorted(draw.enrolled)):
            raise ValueError("a classifier scores only the draw whose gallery it was trained on")
        mated_top, mated_score = self._top(draw.mated.embeddings)
        _, non_mated_score = self._top(draw.non_mated.embeddings)
        return ScoredProbes(
            draw=draw.draw,
            mated_identity=np.asarray(draw.mated.identities, dtype=np.int_),
            mated_score=mated_score,
            mated_correct=mated_top == draw.mated.identities,
            non_mated_identity=np.asarray(draw.non_mated.identities, dtype=np.int_),
            non_mated_score=non_mated_score,
        )

    def _top(self, probes: NDArray[np.float32]) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        x = probes.astype(np.float64)
        if self.decision:
            scores = self.estimator.decision_function(x)
            if scores.ndim == 1:
                # With two classes scikit-learn returns the second class's score alone.
                scores = np.column_stack([-scores, scores])
        else:
            scores = self.estimator.predict_proba(x)
        top = np.argmax(scores, axis=1)
        return (
            np.asarray(self.estimator.classes_[top], dtype=np.int_),
            scores[np.arange(top.size), top].astype(np.float64),
        )


class ChosenHyperparameter:
    """A classifier's hyperparameter, chosen on the validation draw. Made only by
    `choose_hyperparameter`, and the only way to train a classifier.

    Not a dataclass, so `dataclasses.replace` cannot copy one with another value, and its
    attributes are read-only.
    """

    __slots__ = ("_hyperparameter", "_method", "_tpirs", "_validation")

    def __init__(
        self,
        method: Classifier,
        hyperparameter: Hyperparameter,
        tpirs: tuple[float, ...],
        validation: ScoredProbes,
        seal: object = None,
    ) -> None:
        if seal is not _SEAL:
            raise TypeError("a hyperparameter is only chosen by choose_hyperparameter()")
        self._method = method
        self._hyperparameter = hyperparameter
        self._tpirs = tpirs
        self._validation = validation

    @property
    def method(self) -> Classifier:
        return self._method

    @property
    def hyperparameter(self) -> Hyperparameter:
        return self._hyperparameter

    @property
    def tpirs(self) -> tuple[float, ...]:
        """Validation TPIR at FPIR 1% of each candidate, in `hyperparameter.candidates` order."""
        return self._tpirs

    @property
    def validation(self) -> ScoredProbes:
        """The validation draw scored by the chosen classifier, trained on its gallery."""
        return self._validation

    def __repr__(self) -> str:
        return f"ChosenHyperparameter({self._method!r}, {self._hyperparameter!r})"


def choose_hyperparameter(method: Classifier, validation: EmbeddedDraw) -> ChosenHyperparameter:
    """The candidate whose classifier, trained on the validation gallery, has the highest TPIR at
    FPIR 1% on the validation probes; a tie goes to the most regularised, the smallest C or the
    largest k.

    A candidate that keeps FPIR at 1% only by accepting nothing, as a small k can when many
    non-mated probes get every vote, scores TPIR 0. If the chosen one still does, no threshold
    can be frozen for the method, and it is an error. A k larger than the gallery's enrolled
    photos is not a neighbourhood and is not tried.
    """
    if validation.draw != "validation":
        raise ValueError(
            f"only the validation draw chooses a hyperparameter, not the {validation.draw}"
        )
    spec = _SPECS[method]
    photos = sum(len(embeddings) for embeddings in validation.enrolled.values())
    candidates = tuple(
        value for value in spec.candidates if spec.hyperparameter != "k" or value <= photos
    )
    if not candidates:
        raise ValueError(f"no {method} candidate fits a gallery of {photos} enrolled photos")
    scored = [_train(method, value, validation.enrolled).score(validation) for value in candidates]
    points = [tpir_at_fpir(s, TARGET_FPIR) for s in scored]
    order = 1 if spec.larger_is_simpler else -1
    best = max(
        range(len(points)),
        key=lambda i: (points[i].tpir, np.isfinite(points[i].threshold), order * i),
    )
    if not np.isfinite(points[best].threshold):
        raise ValueError(
            f"{method} cannot keep FPIR at or below {TARGET_FPIR:.0%} on validation "
            "with any candidate except by accepting nothing"
        )
    value = candidates[best]
    logger.info("%s: %s = %g chosen on validation", method, spec.hyperparameter, value)
    return ChosenHyperparameter(
        method,
        Hyperparameter(name=spec.hyperparameter, value=value, candidates=list(candidates)),
        tuple(point.tpir for point in points),
        scored[best],
        _SEAL,
    )


def train_classifier(
    chosen: ChosenHyperparameter, enrolled: Mapping[int, Sequence[Embedding]]
) -> TrainedClassifier:
    """The chosen classifier trained on a gallery's enrolled embeddings, which is all it sees."""
    return _train(chosen.method, chosen.hyperparameter.value, enrolled)


def _train(
    method: Classifier, value: float, enrolled: Mapping[int, Sequence[Embedding]]
) -> TrainedClassifier:
    spec = _SPECS[method]
    identities = sorted(enrolled)
    if len(identities) < 2 or any(not enrolled[i] for i in identities):
        raise ValueError("a classifier needs two identities, each with an enrolled embedding")
    x = np.stack([photo for i in identities for photo in enrolled[i]]).astype(np.float64)
    y = np.array([i for i in identities for _ in enrolled[i]], dtype=np.int_)
    return TrainedClassifier(
        method=method,
        identities=np.array(identities, dtype=np.int_),
        estimator=_fitted(spec.build(value), x, y),
        decision=spec.decision,
    )


def _fitted(estimator: _Estimator, x: NDArray[np.float64], y: NDArray[np.int_]) -> _Estimator:
    """`estimator` fitted, or an error if it did not converge: its output would not then be the
    method's, so it must not be reported as the method's."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            return estimator.fit(x, y)
        except ConvergenceWarning as warning:
            raise RuntimeError(
                f"{type(estimator).__name__} did not converge: {warning}"
            ) from warning


@dataclass(frozen=True, eq=False)
class TopTwoDraw:
    """A draw's probes reduced to their best-photo top two, all the learned rule sees of it."""

    draw: Draw
    mated_identity: NDArray[np.int_]
    mated: TopTwo
    non_mated_identity: NDArray[np.int_]
    non_mated: TopTwo

    @classmethod
    def of(cls, embedded: EmbeddedDraw) -> "TopTwoDraw":
        gallery = Gallery.enrol(embedded.enrolled)
        return cls(
            draw=embedded.draw,
            mated_identity=np.asarray(embedded.mated.identities, dtype=np.int_),
            mated=gallery.top_two(embedded.mated.embeddings),
            non_mated_identity=np.asarray(embedded.non_mated.identities, dtype=np.int_),
            non_mated=gallery.top_two(embedded.non_mated.embeddings),
        )

    @property
    def mated_correct(self) -> NDArray[np.bool_]:
        return np.asarray(self.mated.identities == self.mated_identity)

    def scored(
        self, mated_score: NDArray[np.float64], non_mated_score: NDArray[np.float64]
    ) -> ScoredProbes:
        """The draw with each probe's best-photo top candidate given a new match score."""
        return ScoredProbes(
            draw=self.draw,
            mated_identity=self.mated_identity,
            mated_score=mated_score,
            mated_correct=self.mated_correct,
            non_mated_identity=self.non_mated_identity,
            non_mated_score=non_mated_score,
        )

    def best_photo(self) -> ScoredProbes:
        return self.scored(self.mated.scores, self.non_mated.scores)


def match_probability(rule: LearnedRule, top_two: TopTwo) -> NDArray[np.float64]:
    """The learned rule's P(match) for each probe's best-photo top candidate."""
    return _probability(rule, _features(top_two))


def _features(top_two: TopTwo) -> NDArray[np.float64]:
    """The learned rule's two inputs: the top score and its gap to the runner-up."""
    return np.column_stack([top_two.scores, top_two.gaps])


def _probability(rule: LearnedRule, features: NDArray[np.float64]) -> NDArray[np.float64]:
    z = rule.intercept + rule.top_score * features[:, 0] + rule.gap * features[:, 1]
    # 1 / (1 + exp(-z)), without overflowing for a very negative z.
    return np.asarray(np.exp(-np.logaddexp(0.0, -z)), dtype=np.float64)


class FittedRule:
    """The learned decision rule fitted on the validation draw, with its cut-off frozen on its
    out-of-fold output there. Made only by `fit_learned_rule`.

    Not a dataclass, so `dataclasses.replace` cannot copy one with another value, and its
    attributes are read-only.
    """

    __slots__ = ("_out_of_fold", "_rule", "_threshold")

    def __init__(
        self,
        rule: LearnedRule,
        threshold: FrozenThreshold,
        out_of_fold: ScoredProbes,
        seal: object = None,
    ) -> None:
        if seal is not _SEAL:
            raise TypeError("a learned rule is only fitted by fit_learned_rule()")
        self._rule = rule
        self._threshold = threshold
        self._out_of_fold = out_of_fold

    @property
    def rule(self) -> LearnedRule:
        """The coefficients fitted on every validation probe."""
        return self._rule

    @property
    def threshold(self) -> FrozenThreshold:
        return self._threshold

    @property
    def out_of_fold(self) -> ScoredProbes:
        """Each validation probe scored by the fold fit that never saw its identity."""
        return self._out_of_fold

    def score(self, draw: TopTwoDraw) -> ScoredProbes:
        return draw.scored(
            match_probability(self._rule, draw.mated), match_probability(self._rule, draw.non_mated)
        )

    def __repr__(self) -> str:
        return f"FittedRule({self._rule!r}, threshold={self._threshold.value!r})"


def fit_learned_rule(validation: TopTwoDraw, folds: int = FOLDS) -> FittedRule:
    """A logistic regression of "the top candidate is a match" on the top score and its gap, over
    every validation probe: 1 for a mated probe whose top candidate is right, 0 for any other.

    Unpenalised, so there is no hyperparameter to choose. Its cut-off is frozen at FPIR 1% on
    out-of-fold probabilities, each probe scored by a fit on the folds without its identity, and
    mated and non-mated identities are different people; the coefficients kept are the fit on
    every probe.
    """
    if validation.draw != "validation":
        raise ValueError(
            f"only the validation draw fits the learned rule, not the {validation.draw}"
        )
    mated = validation.mated_identity.size
    x = np.concatenate([_features(validation.mated), _features(validation.non_mated)])
    y = np.concatenate([validation.mated_correct, np.zeros(x.shape[0] - mated, dtype=np.bool_)])
    groups = np.concatenate([validation.mated_identity, validation.non_mated_identity])
    out_of_fold = np.empty(x.shape[0])
    for train, held_out in GroupKFold(n_splits=folds).split(x, y, groups):
        out_of_fold[held_out] = _probability(_fit_rule(x[train], y[train], folds), x[held_out])
    rule = _fit_rule(x, y, folds)
    scored = validation.scored(out_of_fold[:mated], out_of_fold[mated:])
    return FittedRule(rule, _freeze("learned", scored), scored, _SEAL)


def _fit_rule(x: NDArray[np.float64], y: NDArray[np.bool_], folds: int) -> LearnedRule:
    estimator = LogisticRegression(C=np.inf, tol=_TOL, max_iter=_MAX_ITER)
    _fitted(estimator, x, y.astype(np.int_))
    intercept = np.asarray(estimator.intercept_, dtype=np.float64)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    return LearnedRule(
        intercept=float(intercept[0]),
        top_score=float(coefficients[0, 0]),
        gap=float(coefficients[0, 1]),
        folds=folds,
    )


def _freeze(method: Method, validation: ScoredProbes) -> FrozenThreshold:
    try:
        return freeze(validation)
    except ValueError as error:
        raise ValueError(f"{method}: {error}") from error


@dataclass(frozen=True, slots=True)
class _Scored:
    """One method's scores on both draws and its threshold, frozen on validation."""

    method: Method
    validation: ScoredProbes
    threshold: FrozenThreshold
    test: ScoredProbes
    hyperparameter: Hyperparameter | None = None


@dataclass(frozen=True, eq=False)
class Comparison:
    """#10's comparison on one recognition model, and the per-probe scores behind it."""

    result: LearningModel
    scores: Mapping[Draw, Mapping[Method, ScoredProbes]]
    """Every method's scores on each draw; the learned rule's on validation are out of fold."""
    runner_up: Mapping[Draw, NDArray[np.float64]]
    """Each probe's best-photo runner-up score on each draw, mated probes then non-mated."""


def compare(
    validation: EmbeddedDraw,
    test: EmbeddedDraw,
    model: RecognitionModelId,
    *,
    seed: int,
    sample_seed: int = SAMPLE_SEED,
) -> Comparison:
    """Every method fitted and its threshold frozen on `validation`, then `test` scored once and
    each method's TPIR at FPIR 1% compared with the baseline's in the same resamples; `seed` is
    the bootstrap's."""
    if (validation.draw, test.draw) != ("validation", "test"):
        raise ValueError("compare takes the validation draw, then the test draw")
    scored = [_scoring_rule("best-photo", validation.score(), test.score())]
    scored.append(_scoring_rule("mean", _mean(validation), _mean(test)))
    for method in CLASSIFIERS:
        chosen = choose_hyperparameter(method, validation)
        scored.append(
            _Scored(
                method=method,
                validation=chosen.validation,
                threshold=_freeze(method, chosen.validation),
                test=train_classifier(chosen, test.enrolled).score(test),
                hyperparameter=chosen.hyperparameter,
            )
        )
    validation_top_two, test_top_two = TopTwoDraw.of(validation), TopTwoDraw.of(test)
    fitted = fit_learned_rule(validation_top_two)
    scored.append(
        _Scored("learned", fitted.out_of_fold, fitted.threshold, fitted.score(test_top_two))
    )

    baseline = scored[0].test
    methods = [
        MethodResult(
            method=s.method,
            family=_FAMILIES[s.method],
            needs_retraining=s.method not in _LIVE,
            hyperparameter=s.hyperparameter,
            threshold=s.threshold.value,
            validation=draw_result(s.validation, s.threshold, seed=seed),
            test=draw_result(s.test, s.threshold, seed=seed),
            gain=None
            if s.method == "best-photo"
            else paired_gain(baseline, s.test, TARGET_FPIR, seed=seed),
        )
        for s in scored
    ]
    rule, reason = live_rule(methods)
    result = LearningModel(
        model=model,
        methods=methods,
        learned_rule=fitted.rule,
        gaps=gap_histogram(test_top_two),
        sample=top_gap_sample(validation_top_two, seed=sample_seed),
        live_rule=rule,
        live_reason=reason,
    )
    return Comparison(
        result=result,
        scores={
            "validation": {s.method: s.validation for s in scored},
            "test": {s.method: s.test for s in scored},
        },
        runner_up={
            draw.draw: np.concatenate(
                [draw.mated.runner_up_scores, draw.non_mated.runner_up_scores]
            )
            for draw in (validation_top_two, test_top_two)
        },
    )


def score_rule(
    rule: MatchRule, draw: EmbeddedDraw, learned: LearnedRule | None = None
) -> ScoredProbes:
    """`draw` scored under a live rule, exactly as `compare` scores it; the learned rule needs
    its coefficients."""
    match rule:
        case "best-photo":
            return draw.score()
        case "mean":
            return _mean(draw)
        case "learned":
            if learned is None:
                raise ValueError("the learned rule needs its coefficients")
            top_two = TopTwoDraw.of(draw)
            return top_two.scored(
                match_probability(learned, top_two.mated),
                match_probability(learned, top_two.non_mated),
            )
        case _:
            assert_never(rule)


def _scoring_rule(method: Method, validation: ScoredProbes, test: ScoredProbes) -> _Scored:
    return _Scored(method, validation, _freeze(method, validation), test)


def _mean(draw: EmbeddedDraw) -> ScoredProbes:
    gallery = Gallery.enrol(draw.enrolled).averaged()
    return score_probes(draw.draw, gallery, draw.mated, draw.non_mated)


def live_rule(methods: Sequence[MethodResult]) -> tuple[MatchRule, str]:
    """The rule a model runs live, and why: of the methods that need no retraining and whose
    gain's interval lies above zero, the one with the largest gain; otherwise best-photo."""
    compared = [(m, m.gain) for m in methods if m.gain is not None]
    capable = [(m, gain) for m, gain in compared if not m.needs_retraining]
    improving = [(m, gain) for m, gain in capable if gain.improves]
    if improving:
        winner, gain = max(improving, key=lambda pair: pair[1].value)
        reason = (
            f"{_capitalised(_NAMES[winner.method])} improves test TPIR at FPIR "
            f"{gain.target_fpir:.0%} on best-photo by {_points(gain)}, the largest measurable "
            "gain of the methods that need no retraining."
        )
        return _LIVE[winner.method], reason
    target = compared[0][1].target_fpir if compared else TARGET_FPIR
    reason = (
        "No method that needs no retraining measurably improves test TPIR at FPIR "
        f"{target:.0%} on best-photo"
    )
    if capable:
        reason += ": " + ", ".join(f"{_NAMES[m.method]} {_points(gain)}" for m, gain in capable)
    retrained = [_NAMES[m.method] for m, gain in compared if m.needs_retraining and gain.improves]
    if retrained:
        one = len(retrained) == 1
        reason += (
            f"; {_listed(retrained)} {'improves' if one else 'improve'} on it but "
            f"{'needs' if one else 'need'} retraining whenever the watchlist changes"
        )
    return "best-photo", reason + "."


def _points(gain: Gain) -> str:
    return (
        f"{gain.value * 100:+.2f} points "
        f"({CONFIDENCE:.0%} CI {gain.ci.low * 100:+.2f} to {gain.ci.high * 100:+.2f})"
    )


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]


def _listed(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def gap_histogram(draw: TopTwoDraw) -> GapHistogram:
    """How far each probe's top candidate scores above the runner-up, by kind of probe, in bins
    0.02 wide from 0 up to the first edge at or above the largest gap."""
    correct = draw.mated_correct
    largest = float(max(draw.mated.gaps.max(), draw.non_mated.gaps.max()))
    bins = max(1, int(np.ceil(largest * GAP_BINS_PER_UNIT)))
    if bins / GAP_BINS_PER_UNIT < largest:
        bins += 1
    edges = np.arange(bins + 1) / GAP_BINS_PER_UNIT

    def counts(gaps: NDArray[np.float64]) -> list[int]:
        return [int(c) for c in np.histogram(gaps, edges)[0]]

    return GapHistogram(
        edges=[float(e) for e in edges],
        right=counts(draw.mated.gaps[correct]),
        wrong=counts(draw.mated.gaps[~correct]),
        non_mated=counts(draw.non_mated.gaps),
    )


def top_gap_sample(draw: TopTwoDraw, *, seed: int, size: int = SAMPLE_SIZE) -> TopGapSample:
    """Up to `size` probes whose top candidate is right and `size` non-mated ones, drawn with
    `seed`, and every probe whose top candidate is wrong, as (top score, gap)."""
    rng = np.random.default_rng(seed)
    correct = draw.mated_correct

    def pick(rows: NDArray[np.int_]) -> NDArray[np.int_]:
        return np.sort(rng.choice(rows, size, replace=False)) if rows.size > size else rows

    kinds: list[tuple[Literal["right", "wrong", "non-mated"], TopTwo, NDArray[np.int_]]] = [
        ("right", draw.mated, pick(np.flatnonzero(correct))),
        ("wrong", draw.mated, np.flatnonzero(~correct)),
        ("non-mated", draw.non_mated, pick(np.arange(draw.non_mated_identity.size))),
    ]
    return TopGapSample(
        top=[float(v) for _, top_two, rows in kinds for v in top_two.scores[rows]],
        gap=[float(v) for _, top_two, rows in kinds for v in top_two.gaps[rows]],
        kind=[kind for kind, _, rows in kinds for _ in rows],
    )
