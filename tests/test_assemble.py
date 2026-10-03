"""The results file's derived blocks under each model's live rule (#10, #28), the same-person
thresholds (#49), and what of learning, bias and the live operating points survives a rerun."""

import pytest

from results_files import (
    synthetic_identification,
    synthetic_learning,
    synthetic_live,
    synthetic_verification,
)
from ryuk.evaluation.active import assemble, carried, frozen_thresholds
from ryuk.evaluation.results import (
    DrawDigest,
    LearnedRule,
    LearningModel,
    MethodResult,
    Results,
    SamePersonThreshold,
)


def test_without_a_winner_the_thresholds_are_identifications_own() -> None:
    identification = synthetic_identification()
    verification = synthetic_verification()

    plain = assemble(verification, identification)
    learned = assemble(verification, identification, synthetic_learning(identification))

    assert learned.thresholds == plain.thresholds
    assert learned.first_active_model == plain.first_active_model
    assert learned.learning is not None


def test_a_winning_mean_rule_is_the_threshold_the_service_reads() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification, {"facenet": "mean"})

    results = assemble(synthetic_verification(), identification, learning)

    facenet = results.thresholds[2]
    mean = learning.models[2].method("mean")
    assert (facenet.rule, facenet.threshold, facenet.learned_rule) == ("mean", mean.threshold, None)
    # Frozen by the learning run, not by identification's.
    assert facenet.commit == learning.provenance.commit
    assert [t.rule for t in results.thresholds[:2]] == ["best-photo", "best-photo"]
    assert results.thresholds[0].commit == identification.provenance.commit


def test_a_winning_learned_rule_carries_its_coefficients() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification, {"sface": "learned"})

    results = assemble(synthetic_verification(), identification, learning)

    assert results.thresholds[0].rule == "learned"
    assert results.thresholds[0].learned_rule == learning.models[0].learned_rule


def test_the_first_active_model_is_judged_under_the_live_rule() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification, {"arcface": "mean"})

    results = assemble(synthetic_verification(), identification, learning)

    assert results.first_active_model is not None
    arcface = results.first_active_model.eligibility[1]
    mean = learning.models[1].method("mean").test.at_threshold
    assert (arcface.test_tpir, arcface.test_fpir) == (mean.tpir, mean.fpir.value)


def test_a_learned_threshold_without_its_coefficients_is_refused() -> None:
    identification = synthetic_identification()
    results = assemble(
        synthetic_verification(),
        identification,
        synthetic_learning(identification, {"sface": "learned"}),
    )
    document = results.model_dump(mode="json")
    document["thresholds"][0]["learned_rule"] = None

    with pytest.raises(ValueError, match="coefficients"):
        Results.model_validate(document)


def test_a_live_rule_other_than_the_one_its_gains_choose_is_refused() -> None:
    # FaceNet's mean rule improves on best-photo, so the mean rule is its live rule.
    compared = synthetic_learning(synthetic_identification(), {"facenet": "mean"}).models[2]
    document = compared.model_dump(mode="json")

    with pytest.raises(ValueError, match="live rule"):
        LearningModel.model_validate({**document, "live_rule": "learned"})
    with pytest.raises(ValueError, match="live rule"):
        LearningModel.model_validate({**document, "live_rule": "best-photo"})


def test_a_live_rule_without_an_improving_gain_is_refused() -> None:
    # Nothing improves on SFace's best-photo, so nothing but best-photo can run live.
    compared = synthetic_learning(synthetic_identification()).models[0]

    with pytest.raises(ValueError, match="live rule"):
        LearningModel.model_validate({**compared.model_dump(mode="json"), "live_rule": "mean"})


def test_a_classifier_that_claims_to_need_no_retraining_is_refused() -> None:
    compared = synthetic_learning(synthetic_identification()).models[0]
    mean = compared.method("mean").model_dump(mode="json")

    with pytest.raises(ValueError, match="retraining"):
        MethodResult.model_validate({**mean, "method": "knn", "family": "classifier"})
    with pytest.raises(ValueError, match="retraining"):
        MethodResult.model_validate({**mean, "needs_retraining": True})


@pytest.mark.parametrize(("top_score", "gap"), [(-1.0, 0.0), (4.0, -2.5), (-30.0, 14.9)])
def test_a_learned_rule_that_would_rank_the_runner_up_above_the_top_is_refused(
    top_score: float, gap: float
) -> None:
    with pytest.raises(ValueError, match="runner-up"):
        LearnedRule(intercept=0.0, top_score=top_score, gap=gap, folds=5)


@pytest.mark.parametrize(("top_score", "gap"), [(0.0, 0.0), (4.0, -2.0), (-30.0, 15.0)])
def test_a_learned_rule_that_ties_the_top_two_at_worst_is_kept(
    top_score: float, gap: float
) -> None:
    assert LearnedRule(intercept=0.0, top_score=top_score, gap=gap, folds=5).gap == gap


def test_learning_on_other_draws_is_refused() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification).model_copy(
        update={"draws": [DrawDigest(draw="validation", selection_sha256="d" * 64)]}
    )

    with pytest.raises(ValueError, match="other draws"):
        assemble(synthetic_verification(), identification, learning)


def test_learning_needs_identification() -> None:
    learning = synthetic_learning(synthetic_identification())

    with pytest.raises(ValueError, match="no"):
        assemble(synthetic_verification(), None, learning)


def test_learning_still_describing_identification_is_carried() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification)

    kept = carried(identification, learning, None, None)

    assert (kept.learning, kept.bias, kept.live, kept.dropped) == (learning, None, None, ())


def test_learning_is_dropped_with_a_reason_when_identification_goes() -> None:
    learning = synthetic_learning(synthetic_identification())

    kept = carried(None, learning, None, None)

    assert kept.learning is None
    assert len(kept.dropped) == 1
    assert "ryuk evaluate learn" in kept.dropped[0]


def test_learning_is_dropped_when_a_baseline_threshold_moved() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification)
    moved = identification.model_copy(
        update={
            "models": [
                identification.models[0].model_copy(update={"threshold": 0.123}),
                *identification.models[1:],
            ]
        }
    )

    kept = carried(moved, learning, None, None)

    assert kept.learning is None
    assert "baseline threshold" in kept.dropped[0]


def test_without_the_live_operating_points_no_model_has_a_same_person_threshold() -> None:
    results = assemble(synthetic_verification(), synthetic_identification())

    assert [t.same_person for t in results.thresholds] == [None, None, None]


def test_each_same_person_threshold_is_the_one_live_froze() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification, {"facenet": "mean"})
    live = synthetic_live(identification, frozen_thresholds(identification, learning))

    results = assemble(synthetic_verification(), identification, learning, None, live)

    assert results.live == live
    assert [t.same_person for t in results.thresholds] == [
        SamePersonThreshold(
            threshold=m.same_person.threshold,
            target_far=m.same_person.target_far,
            far=m.same_person.validation_far,
            impostor_pairs=m.same_person.validation_impostor_pairs,
            commit=live.provenance.commit,
            date=live.provenance.generated_at.date(),
        )
        for m in live.models
    ]
    # The 1:N thresholds are untouched.
    assert [t.model_copy(update={"same_person": None}) for t in results.thresholds] == (
        assemble(synthetic_verification(), identification, learning).thresholds
    )


def test_a_same_person_threshold_other_than_lives_is_refused() -> None:
    identification = synthetic_identification()
    live = synthetic_live(identification, frozen_thresholds(identification, None))
    document = assemble(synthetic_verification(), identification, None, None, live).model_dump(
        mode="json"
    )
    document["thresholds"][1]["same_person"]["threshold"] = 0.9

    with pytest.raises(ValueError, match="same-person threshold is the one live measured"):
        Results.model_validate(document)


def test_live_operating_points_at_a_rule_no_longer_live_are_refused() -> None:
    identification = synthetic_identification()
    live = synthetic_live(identification, frozen_thresholds(identification, None))
    learning = synthetic_learning(identification, {"facenet": "mean"})

    with pytest.raises(ValueError, match="facenet now runs mean"):
        assemble(synthetic_verification(), identification, learning, None, live)


def test_live_operating_points_on_other_draws_are_refused() -> None:
    identification = synthetic_identification()
    live = synthetic_live(identification, frozen_thresholds(identification, None)).model_copy(
        update={"draws": [DrawDigest(draw="validation", selection_sha256="d" * 64)]}
    )

    with pytest.raises(ValueError, match="other draws"):
        assemble(synthetic_verification(), identification, None, None, live)


def test_live_operating_points_are_carried_while_every_live_rule_holds() -> None:
    identification = synthetic_identification()
    learning = synthetic_learning(identification)
    live = synthetic_live(identification, frozen_thresholds(identification, learning))

    kept = carried(identification, learning, None, live)

    assert (kept.live, kept.dropped) == (live, ())


def test_live_operating_points_are_dropped_with_a_reason_when_a_live_rule_moves() -> None:
    identification = synthetic_identification()
    live = synthetic_live(identification, frozen_thresholds(identification, None))

    learning = synthetic_learning(identification, {"sface": "mean"})

    kept = carried(identification, learning, None, live)

    assert kept.live is None
    assert kept.dropped == (
        f"sface now runs mean at {learning.models[0].method('mean').threshold:.3f}, not the "
        f"best-photo at {live.models[0].threshold:.3f} its small galleries were scored at; run "
        "`ryuk evaluate live` again",
    )
