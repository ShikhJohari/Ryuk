"""#10's comparison on one recognition model: every method fitted on validation only (#28)."""

import dataclasses
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from embedded_draws import embedded_draw, unit_rows
from ryuk.evaluation import learning
from ryuk.evaluation.learning import (
    CLASSIFIERS,
    KNN_K,
    METHODS,
    ChosenHyperparameter,
    Classifier,
    Comparison,
    FittedRule,
    TopTwoDraw,
    choose_hyperparameter,
    compare,
    fit_learned_rule,
    gap_histogram,
    live_rule,
    score_rule,
    top_gap_sample,
    train_classifier,
)
from ryuk.evaluation.openset import (
    EmbeddedDraw,
    Gallery,
    Probes,
    ScoredProbes,
    TopTwo,
    draw_result,
    freeze,
    paired_gain,
    score_probes,
)
from ryuk.evaluation.results import (
    Gain,
    Hyperparameter,
    LearningModel,
    MatchRule,
    Method,
    MethodResult,
    RecognitionModelId,
    SignedInterval,
)

MODEL = RecognitionModelId(network="arcface", provider="cpu", weights_sha256="a" * 64, dimension=16)


@pytest.fixture(scope="module")
def draws() -> tuple[EmbeddedDraw, EmbeddedDraw]:
    return embedded_draw("validation", 1), embedded_draw("test", 2)


@pytest.fixture(scope="module")
def comparison(draws: tuple[EmbeddedDraw, EmbeddedDraw]) -> Comparison:
    return compare(*draws, MODEL, seed=7)


@pytest.fixture(scope="module")
def compared(comparison: Comparison) -> LearningModel:
    return comparison.result


@pytest.mark.parametrize("rule", ["best-photo", "mean", "learned"])
def test_a_live_rule_scores_the_test_draw_exactly_as_the_comparison_did(
    draws: tuple[EmbeddedDraw, EmbeddedDraw], comparison: Comparison, rule: MatchRule
) -> None:
    measured = comparison.scores["test"][rule]

    scored = score_rule(rule, draws[1], comparison.result.learned_rule)

    np.testing.assert_array_equal(scored.mated_score, measured.mated_score)
    np.testing.assert_array_equal(scored.non_mated_score, measured.non_mated_score)
    np.testing.assert_array_equal(scored.mated_correct, measured.mated_correct)


def test_the_learned_rule_cannot_score_without_its_coefficients(
    draws: tuple[EmbeddedDraw, EmbeddedDraw],
) -> None:
    with pytest.raises(ValueError, match="coefficients"):
        score_rule("learned", draws[1])


def test_the_comparison_keeps_every_methods_scores_and_the_runner_up(
    draws: tuple[EmbeddedDraw, EmbeddedDraw], comparison: Comparison
) -> None:
    for embedded in draws:
        probes = embedded.mated.identities.size + embedded.non_mated.identities.size
        assert list(comparison.scores[embedded.draw]) == list(learning.METHODS)
        assert comparison.runner_up[embedded.draw].shape == (probes,)


def test_the_top_two_draw_scores_best_photo_exactly_as_the_draw_does() -> None:
    embedded = embedded_draw("validation", 3)

    ours, theirs = TopTwoDraw.of(embedded).best_photo(), embedded.score()

    for field in dataclasses.fields(ScoredProbes):
        assert np.array_equal(getattr(ours, field.name), getattr(theirs, field.name))


@pytest.mark.parametrize("method", CLASSIFIERS)
def test_a_hyperparameter_is_never_chosen_on_the_test_draw(method: Classifier) -> None:
    with pytest.raises(ValueError, match="only the validation draw"):
        choose_hyperparameter(method, embedded_draw("test", 3))


def test_the_learned_rule_is_never_fitted_on_the_test_draw() -> None:
    with pytest.raises(ValueError, match="only the validation draw"):
        fit_learned_rule(TopTwoDraw.of(embedded_draw("test", 3)))


def test_the_fitted_tokens_cannot_be_made_or_copied_with_other_values() -> None:
    embedded = embedded_draw("validation", 3)
    chosen = choose_hyperparameter("logistic-regression", embedded)
    fitted = fit_learned_rule(TopTwoDraw.of(embedded))
    record = Hyperparameter(name="C", value=1.0, candidates=[1.0])

    with pytest.raises(TypeError, match="choose_hyperparameter"):
        ChosenHyperparameter("knn", record, (0.5,), embedded.score())
    with pytest.raises(TypeError, match="fit_learned_rule"):
        FittedRule(fitted.rule, fitted.threshold, fitted.out_of_fold)
    for token in (chosen, fitted):
        # Neither is a dataclass, so neither can be copied with another value.
        with pytest.raises(TypeError):
            dataclasses.replace(token)  # type: ignore[type-var]
    with pytest.raises(AttributeError):
        chosen.hyperparameter = record  # type: ignore[misc]
    with pytest.raises(AttributeError):
        fitted.rule = fitted.rule  # type: ignore[misc]


def test_every_k_votes_among_several_neighbours() -> None:
    assert all(k > 1 for k in KNN_K)


def test_ties_go_to_the_simplest_classifier_the_smallest_c_and_the_largest_k() -> None:
    # Nearly noiseless: most candidates find every mated probe at FPIR 1%. A small k cannot keep
    # FPIR at 1% at all, as it gives too many non-mated probes every vote, so it scores 0.
    separable = embedded_draw("validation", 4, noise=0.01)

    chosen = {method: choose_hyperparameter(method, separable) for method in CLASSIFIERS}

    for method, token in chosen.items():
        record = token.hyperparameter
        tied = [c for c, t in zip(record.candidates, token.tpirs, strict=True) if t == 1.0]
        assert len(tied) > 1
        assert record.value == (max(tied) if method == "knn" else min(tied))
    assert dict(zip(KNN_K, chosen["knn"].tpirs, strict=True))[3] == 0.0


@pytest.mark.parametrize("method", CLASSIFIERS)
def test_the_hyperparameter_with_the_best_validation_tpir_is_chosen(method: Classifier) -> None:
    chosen = choose_hyperparameter(method, embedded_draw("validation", 5))

    record = chosen.hyperparameter
    assert record.name == ("k" if method == "knn" else "C")
    assert sorted(record.candidates) == record.candidates
    assert len(chosen.tpirs) == len(record.candidates)
    assert max(chosen.tpirs) > 0
    assert chosen.tpirs[record.candidates.index(record.value)] == max(chosen.tpirs)
    # Its validation scores are the chosen classifier's, trained on the validation gallery.
    retrained = train_classifier(chosen, embedded_draw("validation", 5).enrolled)
    again = retrained.score(embedded_draw("validation", 5))
    assert np.array_equal(again.mated_score, chosen.validation.mated_score)
    assert np.array_equal(again.non_mated_score, chosen.validation.non_mated_score)


def test_a_classifier_that_only_meets_fpir_1_percent_by_accepting_nothing_is_refused() -> None:
    # Every non-mated probe is one of the gallery's photos, so every k votes it in unanimously.
    embedded = embedded_draw("validation", 6)
    photos = np.stack(
        [photo for identity in sorted(embedded.enrolled) for photo in embedded.enrolled[identity]]
    )
    copies = dataclasses.replace(
        embedded,
        non_mated=Probes(np.arange(5000, 5000 + len(photos)), photos),
    )

    with pytest.raises(ValueError, match="knn cannot keep FPIR at or below 1% on validation"):
        choose_hyperparameter("knn", copies)


def test_a_classifier_scores_only_the_draw_whose_gallery_it_was_trained_on() -> None:
    validation, test = embedded_draw("validation", 3), embedded_draw("test", 4)
    chosen = choose_hyperparameter("knn", validation)

    with pytest.raises(ValueError, match="gallery it was trained on"):
        train_classifier(chosen, validation.enrolled).score(test)


def test_a_two_identity_gallery_gets_a_top_class_and_score_from_one_decision_value() -> None:
    embedded = embedded_draw("validation", 14, identities=2)
    trained = train_classifier(choose_hyperparameter("linear-svm", embedded), embedded.enrolled)

    scored = trained.score(embedded)

    # scikit-learn gives the second class's decision value alone: the first class's is minus it.
    decision = trained.estimator.decision_function(embedded.mated.embeddings.astype(np.float64))
    top = np.where(decision > 0, trained.identities[1], trained.identities[0])
    assert np.array_equal(scored.mated_correct, top == embedded.mated.identities)
    assert scored.mated_score == pytest.approx(np.abs(decision))


def test_a_classifier_needs_a_gallery_of_two_identities() -> None:
    embedded = embedded_draw("validation", 3)
    chosen = choose_hyperparameter("logistic-regression", embedded)
    first = min(embedded.enrolled)

    with pytest.raises(ValueError, match="two identities"):
        train_classifier(chosen, {first: embedded.enrolled[first]})


def test_a_method_that_cannot_be_frozen_is_named(
    draws: tuple[EmbeddedDraw, EmbeddedDraw],
) -> None:
    # Every non-mated probe is a gallery photo, so best-photo scores each of them 1.
    validation = draws[0]
    photos = np.stack([photo for photos in validation.enrolled.values() for photo in photos])
    copies = dataclasses.replace(
        validation, non_mated=Probes(np.arange(5000, 5000 + len(photos)), photos)
    )

    with pytest.raises(ValueError, match="best-photo: no threshold keeps FPIR"):
        compare(copies, draws[1], MODEL, seed=7)


def test_a_classifier_that_does_not_converge_is_an_error_not_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(learning, "_MAX_ITER", 1)

    with pytest.raises(RuntimeError, match="did not converge"):
        choose_hyperparameter("logistic-regression", embedded_draw("validation", 3))


class _Fits:
    """Every X each scikit-learn estimator is fitted on, with its parameters."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, Any], NDArray[np.float64]]] = []

    def spy(self, monkeypatch: pytest.MonkeyPatch, *classes: Any) -> None:
        for cls in classes:
            original = cls.fit

            def fit(
                estimator: Any, x: Any, y: Any, *args: Any, _original: Any = original, **kw: Any
            ) -> Any:
                self.seen.append(
                    (type(estimator).__name__, estimator.get_params(), np.array(x, copy=True))
                )
                return _original(estimator, x, y, *args, **kw)

            monkeypatch.setattr(cls, "fit", fit)

    def rows(self) -> Iterator[tuple[str, dict[str, Any], NDArray[np.float64]]]:
        yield from self.seen


def _in(rows: NDArray[np.floating[Any]], others: NDArray[np.floating[Any]]) -> NDArray[np.bool_]:
    """Whether each of `rows` equals some row of `others`, to float32 precision."""
    near = np.abs(rows[:, np.newaxis, :] - others[np.newaxis, :, :]).max(axis=2) < 1e-6
    return np.asarray(near.any(axis=1))


def test_the_learned_rule_cut_off_comes_from_identity_grouped_out_of_fold_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = TopTwoDraw.of(embedded_draw("validation", 7))
    features = np.concatenate(
        [
            np.column_stack([validation.mated.scores, validation.mated.gaps]),
            np.column_stack([validation.non_mated.scores, validation.non_mated.gaps]),
        ]
    )
    identities = np.concatenate([validation.mated_identity, validation.non_mated_identity])
    fits = _Fits()
    fits.spy(monkeypatch, LogisticRegression)

    fitted = fit_learned_rule(validation)

    assert [params["C"] for _, params, _ in fits.rows()] == [np.inf] * (learning.FOLDS + 1)
    held_out = []
    for _, _, x in fits.seen[: learning.FOLDS]:
        trained = _in(features, x)
        # No identity is on both sides of a fold.
        assert not set(identities[trained]) & set(identities[~trained])
        held_out.append(~trained)
    # Each probe is held out exactly once, and the final fit sees them all.
    assert (np.sum(held_out, axis=0) == 1).all()
    assert _in(features, fits.seen[-1][2]).all()
    assert fitted.rule.folds == learning.FOLDS
    assert fitted.threshold.value == freeze(fitted.out_of_fold).value
    in_sample = fitted.score(validation)
    assert not np.array_equal(in_sample.mated_score, fitted.out_of_fold.mated_score)


def test_the_learned_rule_is_a_logistic_regression_on_the_top_score_and_its_gap() -> None:
    validation, test = (
        TopTwoDraw.of(embedded_draw("validation", 8)),
        TopTwoDraw.of(embedded_draw("test", 9)),
    )
    fitted = fit_learned_rule(validation)
    rule = fitted.rule

    scored = fitted.score(test)

    z = rule.intercept + rule.top_score * test.mated.scores + rule.gap * test.mated.gaps
    assert scored.mated_score == pytest.approx(1 / (1 + np.exp(-z)))
    assert np.array_equal(scored.mated_correct, test.best_photo().mated_correct)
    non_mated = rule.probability(test.non_mated.scores, test.non_mated.gaps)
    assert np.array_equal(non_mated, scored.non_mated_score)
    # More top score and more gap both say "match".
    assert rule.top_score > 0
    assert rule.gap > 0


def test_a_learned_rule_that_would_rank_the_runner_up_above_the_top_is_not_fitted() -> None:
    # Right top candidates score low with small gaps and non-mated ones high with large gaps, so
    # the fit would score a runner-up above its top candidate, and live and evaluation would
    # disagree about who the top candidate is.
    rng = np.random.default_rng(3)
    mated_identity = np.repeat(np.arange(20), 5)
    non_mated_identity = np.repeat(np.arange(100, 120), 5)

    def top_two(identities: NDArray[np.int_], top: float, gap: float) -> TopTwo:
        scores = rng.normal(top, 0.1, identities.size)
        return TopTwo(identities, scores, scores - np.abs(rng.normal(gap, 0.1, identities.size)))

    validation = TopTwoDraw(
        draw="validation",
        mated_identity=mated_identity,
        mated=top_two(mated_identity, 0.4, 0.05),
        non_mated_identity=non_mated_identity,
        non_mated=top_two(np.zeros_like(non_mated_identity), 0.6, 0.2),
    )

    with pytest.raises(ValueError, match="runner-up above the top candidate"):
        fit_learned_rule(validation)


def test_no_test_draw_data_reaches_any_fitting_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test draw's probes never reach a fit, and nothing frozen depends on the test draw.

    The classifiers are retrained on the test draw's own gallery by design (#10): that is
    enrolment, as live, not fitting; they are the only fits that see the test draw, and only
    with the hyperparameter the validation draw chose."""
    validation, test = embedded_draw("validation", 10), embedded_draw("test", 11)
    fits = _Fits()
    fits.spy(monkeypatch, KNeighborsClassifier, LogisticRegression, LinearSVC)

    result = compare(validation, test, MODEL, seed=3).result

    test_probes = np.concatenate([test.mated.embeddings, test.non_mated.embeddings])
    test_gallery = np.stack([photo for photos in test.enrolled.values() for photo in photos])
    top_two = TopTwoDraw.of(test)
    test_features = np.concatenate(
        [
            np.column_stack([top_two.mated.scores, top_two.mated.gaps]),
            np.column_stack([top_two.non_mated.scores, top_two.non_mated.gaps]),
        ]
    )
    chosen = {
        m.method: m.hyperparameter.value for m in result.methods if m.hyperparameter is not None
    }
    names: dict[str, tuple[Method, str]] = {
        "KNeighborsClassifier": ("knn", "n_neighbors"),
        "LogisticRegression": ("logistic-regression", "C"),
        "LinearSVC": ("linear-svm", "C"),
    }
    validation_gallery = np.stack([p for photos in validation.enrolled.values() for p in photos])
    tuning, enrolments, learned = 0, 0, 0
    for name, params, x in fits.rows():
        if x.shape[1] == 2:
            assert not _in(x, test_features).any()
            learned += 1
            continue
        assert not _in(x, test_probes).any()
        if _in(x, test_gallery).any():
            method, key = names[name]
            assert _in(x, test_gallery).all()
            assert params[key] == chosen[method]
            enrolments += 1
        else:
            assert _in(x, validation_gallery).all()
            tuning += 1
    # Every candidate on the validation gallery, each chosen one on the test gallery, and the
    # learned rule's folds and final fit: the spy saw every fit.
    candidates = sum(len(m.hyperparameter.candidates) for m in result.methods if m.hyperparameter)
    assert (tuning, enrolments, learned) == (candidates, len(CLASSIFIERS), learning.FOLDS + 1)

    # Moving every test embedding moves nothing that was frozen on validation.
    rng = np.random.default_rng(12)

    def moved(photos: NDArray[np.float32]) -> NDArray[np.float32]:
        return unit_rows(photos + rng.normal(scale=0.1, size=photos.shape))

    perturbed = EmbeddedDraw(
        draw="test",
        enrolled={i: list(moved(np.stack(p))) for i, p in test.enrolled.items()},
        mated=Probes(test.mated.identities, moved(test.mated.embeddings)),
        non_mated=Probes(test.non_mated.identities, moved(test.non_mated.embeddings)),
    )
    again = compare(validation, perturbed, MODEL, seed=3).result

    def frozen(model: LearningModel) -> list[object]:
        return [(m.hyperparameter, m.threshold, m.validation) for m in model.methods]

    assert frozen(again) == frozen(result)
    assert again.learned_rule == result.learned_rule
    assert again.sample == result.sample
    assert again.method("best-photo").test != result.method("best-photo").test


def test_compare_scores_every_method_on_both_draws(
    draws: tuple[EmbeddedDraw, EmbeddedDraw], compared: LearningModel
) -> None:
    validation, test = draws

    assert LearningModel.model_validate(compared.model_dump(mode="json")) == compared
    assert compared.model == MODEL
    assert [m.method for m in compared.methods] == list(METHODS)
    families = {m.method: (m.family, m.needs_retraining) for m in compared.methods}
    assert families == {
        "best-photo": ("scoring-rule", False),
        "mean": ("scoring-rule", False),
        "knn": ("classifier", True),
        "logistic-regression": ("classifier", True),
        "linear-svm": ("classifier", True),
        "learned": ("learned-rule", False),
    }
    for m in compared.methods:
        assert (m.hyperparameter is not None) == (m.family == "classifier")
        assert (m.validation.draw, m.test.draw) == ("validation", "test")
        assert m.validation.at_threshold.threshold == m.test.at_threshold.threshold == m.threshold
    # The baseline threshold is identification's committed one, exactly.
    baseline = compared.method("best-photo")
    assert baseline.threshold == freeze(validation.score()).value
    assert baseline.test == draw_result(test.score(), freeze(validation.score()), seed=7)
    # The learned rule's is frozen on its out-of-fold output on validation.
    fitted = fit_learned_rule(TopTwoDraw.of(validation))
    assert compared.learned_rule == fitted.rule
    assert compared.method("learned").threshold == fitted.threshold.value
    assert compared.method("learned").validation == draw_result(
        fitted.out_of_fold, fitted.threshold, seed=7
    )


def test_each_gain_is_paired_against_the_baseline_on_the_test_draw(
    draws: tuple[EmbeddedDraw, EmbeddedDraw], compared: LearningModel
) -> None:
    _, test = draws
    mean = score_probes("test", Gallery.enrol(test.enrolled).averaged(), test.mated, test.non_mated)

    assert compared.method("mean").gain == paired_gain(test.score(), mean, 0.01, seed=7)
    baseline = compared.method("best-photo")
    for m in compared.methods:
        if m is baseline:
            continue
        assert m.gain is not None
        assert m.gain.target_fpir == 0.01
        assert m.gain.value == pytest.approx(
            m.test.operating_points[0].tpir.value - baseline.test.operating_points[0].tpir.value
        )


def test_the_live_rule_is_one_that_can_run_live(compared: LearningModel) -> None:
    assert compared.live_rule in {"best-photo", "mean", "learned"}
    assert live_rule(compared.methods) == (compared.live_rule, compared.live_reason)


def test_the_gap_histogram_bins_every_test_probe(draws: tuple[EmbeddedDraw, EmbeddedDraw]) -> None:
    top_two = TopTwoDraw.of(draws[1])

    histogram = gap_histogram(top_two)

    correct = top_two.mated_correct
    assert (sum(histogram.right), sum(histogram.wrong)) == (
        int(correct.sum()),
        int((~correct).sum()),
    )
    assert sum(histogram.non_mated) == top_two.non_mated_identity.size
    assert histogram.edges[:3] == [0.0, 0.02, 0.04]
    assert histogram.edges[-1] >= max(top_two.mated.gaps.max(), top_two.non_mated.gaps.max())
    assert histogram.edges[-2] < max(top_two.mated.gaps.max(), top_two.non_mated.gaps.max())


def test_the_last_gap_bin_holds_the_largest_gap_despite_rounding() -> None:
    # 0.7000000000000001 * 50 rounds to 35, yet the gap lies just past 35 / 50.
    largest = float(np.nextafter(0.7, 1))
    top_two = TopTwo(
        identities=np.array([1, 2]),
        scores=np.array([0.9, 0.8]),
        runner_up_scores=np.array([0.9 - largest, 0.7]),
    )
    draw = TopTwoDraw(
        draw="test",
        mated_identity=np.array([1, 3]),
        mated=top_two,
        non_mated_identity=np.array([9]),
        non_mated=TopTwo(np.array([1]), np.array([0.5]), np.array([0.49])),
    )

    histogram = gap_histogram(draw)

    assert len(histogram.edges) == 37
    assert (sum(histogram.right), sum(histogram.wrong), sum(histogram.non_mated)) == (1, 1, 1)
    assert histogram.right[-1] == 1


def test_the_sample_keeps_every_wrong_top_candidate_and_samples_the_rest() -> None:
    top_two = TopTwoDraw.of(embedded_draw("validation", 13, noise=0.3))
    wrong = int((~top_two.mated_correct).sum())

    sample = top_gap_sample(top_two, seed=1, size=10)

    assert wrong > 10
    assert [sample.kind.count(k) for k in ("right", "wrong", "non-mated")] == [10, wrong, 10]
    assert top_gap_sample(top_two, seed=1, size=10) == sample
    assert top_gap_sample(top_two, seed=2, size=10) != sample
    everything = top_gap_sample(top_two, seed=1, size=10_000)
    assert everything.kind.count("non-mated") == top_two.non_mated_identity.size


def _result(method: Method, gain: tuple[float, float, float] | None) -> MethodResult:
    rng = np.random.default_rng(0)
    scored = ScoredProbes(
        draw="validation",
        mated_identity=np.repeat(np.arange(20), 5),
        mated_score=rng.uniform(0.4, 0.9, 100),
        mated_correct=np.ones(100, dtype=np.bool_),
        non_mated_identity=np.repeat(np.arange(100, 120), 5),
        non_mated_score=rng.uniform(-0.2, 0.5, 100),
    )
    frozen = freeze(scored)
    outcome = draw_result(scored, frozen, seed=1)
    classifier = method in CLASSIFIERS
    return MethodResult(
        method=method,
        family="classifier" if classifier else "scoring-rule",
        needs_retraining=classifier,
        hyperparameter=None,
        threshold=frozen.value,
        validation=outcome,
        test=outcome,
        gain=None
        if gain is None
        else Gain(
            target_fpir=0.01,
            value=gain[0],
            ci=SignedInterval(low=gain[1], high=gain[2]),
            improves=gain[1] > 0,
        ),
    )


def test_best_photo_stays_live_when_no_live_capable_method_improves_on_it() -> None:
    methods = [
        _result("best-photo", None),
        _result("mean", (0.004, -0.002, 0.01)),
        _result("knn", (0.2, 0.1, 0.3)),
        _result("learned", (-0.01, -0.03, 0.01)),
    ]

    rule, reason = live_rule(methods)

    assert rule == "best-photo"
    assert "No method that needs no retraining measurably improves" in reason
    assert "the mean rule +0.40 points (95% CI -0.20 to +1.00)" in reason
    assert "kNN improves on it but needs retraining whenever the watchlist changes" in reason


def test_with_only_classifiers_compared_best_photo_stays_live_with_nothing_to_list() -> None:
    methods = [_result("best-photo", None), _result("knn", (-0.01, -0.02, 0.0))]

    assert live_rule(methods) == (
        "best-photo",
        "No method that needs no retraining measurably improves test TPIR at FPIR 1% on "
        "best-photo.",
    )


def test_the_largest_measurable_gain_among_live_capable_methods_goes_live() -> None:
    methods = [
        _result("best-photo", None),
        _result("mean", (0.01, 0.002, 0.02)),
        _result("logistic-regression", (0.3, 0.2, 0.4)),
        _result("learned", (0.02, 0.01, 0.03)),
    ]

    rule, reason = live_rule(methods)

    assert rule == "learned"
    assert reason.startswith("The learned rule improves test TPIR at FPIR 1% on best-photo by")
    assert "+2.00 points (95% CI +1.00 to +3.00)" in reason


def test_compare_takes_the_validation_draw_then_the_test_draw(
    draws: tuple[EmbeddedDraw, EmbeddedDraw],
) -> None:
    validation, test = draws

    with pytest.raises(ValueError, match="validation draw, then the test draw"):
        compare(test, validation, MODEL, seed=7)


def test_a_k_larger_than_the_gallery_is_not_tried() -> None:
    small = embedded_draw("validation", 5, identities=2, held_out=30)

    chosen = choose_hyperparameter("knn", small)

    # Two identities with five photos each: k = 15 is no neighbourhood of ten photos.
    assert chosen.hyperparameter.candidates == [3, 5, 7, 9]
