"""The results file's derived blocks under each model's live rule (#10, #28), and what of learning
and bias survives a rerun of identification."""

import pytest

from results_files import synthetic_identification, synthetic_learning, synthetic_verification
from ryuk.evaluation.active import assemble, carried
from ryuk.evaluation.results import DrawDigest, Results


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

    kept = carried(identification, learning, None)

    assert (kept.learning, kept.bias, kept.dropped) == (learning, None, ())


def test_learning_is_dropped_with_a_reason_when_identification_goes() -> None:
    learning = synthetic_learning(synthetic_identification())

    kept = carried(None, learning, None)

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

    kept = carried(moved, learning, None)

    assert kept.learning is None
    assert "baseline threshold" in kept.dropped[0]
