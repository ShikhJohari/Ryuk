"""Every rule evaluation can choose to run live is one the live monitor computes, on the scale of
its threshold (#10, #28; the audit's trap T8)."""

import numpy as np
import pytest

from results_files import synthetic_identification, synthetic_learning, synthetic_verification
from ryuk.evaluation.active import assemble
from ryuk.evaluation.results import LearnedRule
from ryuk.recognition import Embedding
from ryuk.watchlist.live import WatchlistEmbeddings
from ryuk.watchlist.registry import Evaluation

RULE = LearnedRule(intercept=-10.0, top_score=12.0, gap=20.0, folds=5)


def unit(*values: float) -> Embedding:
    vector = np.asarray(values, dtype=np.float32)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def watchlist(*persons: tuple[str, list[Embedding]]) -> WatchlistEmbeddings:
    vectors = [vector for _, photos in persons for vector in photos]
    counts = [len(photos) for _, photos in persons]
    return WatchlistEmbeddings(
        np.stack(vectors),
        tuple((person_id, person_id.title()) for person_id, _ in persons),
        np.cumsum([0, *counts[:-1]], dtype=np.intp),
    )


def test_the_mean_rule_scores_the_cosine_to_each_persons_renormalised_mean() -> None:
    photos = [unit(1, 0, 0), unit(0, 1, 0)]
    probe = unit(1, 1, 0.2)

    ranking = watchlist(("a", photos), ("b", [unit(0, 0, 1)])).ranking(probe, "mean")

    assert ranking is not None
    mean = unit(1, 1, 0)
    assert ranking.top.person_id == "a"
    assert ranking.top.score == pytest.approx(float(mean @ probe), abs=1e-6)
    assert ranking.runner_up is not None
    assert ranking.runner_up.score == pytest.approx(float(unit(0, 0, 1) @ probe), abs=1e-6)


def test_the_learned_rule_scores_the_top_two_by_their_margin_over_each_other() -> None:
    people = watchlist(("a", [unit(1, 0, 0)]), ("b", [unit(0, 1, 0)]), ("c", [unit(0, 0, 1)]))
    probe = unit(0.9, 0.4, 0.1)
    top, runner_up = float(unit(1, 0, 0) @ probe), float(unit(0, 1, 0) @ probe)

    ranking = people.ranking(probe, "learned", RULE)

    assert ranking is not None
    assert ranking.runner_up is not None
    assert (ranking.top.person_id, ranking.runner_up.person_id) == ("a", "b")
    expected = RULE.probability(
        np.array([top, runner_up]), np.array([top - runner_up] * 2) * [1, -1]
    )
    assert [ranking.top.score, ranking.runner_up.score] == pytest.approx(list(expected))
    # Both are P(match), on the threshold's scale, and the runner-up never outranks the top.
    assert 0 <= ranking.runner_up.score < ranking.top.score <= 1


def test_with_one_person_the_learned_rule_takes_no_margin() -> None:
    probe = unit(0.8, 0.6, 0)

    ranking = watchlist(("a", [unit(1, 0, 0)])).ranking(probe, "learned", RULE)

    assert ranking is not None
    assert ranking.runner_up is None
    cosine = float(unit(1, 0, 0) @ probe)
    assert ranking.top.score == pytest.approx(
        float(RULE.probability(np.array(cosine), np.array(0.0)))
    )


def test_the_learned_rule_cannot_run_without_its_coefficients() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        watchlist(("a", [unit(1, 0, 0)])).ranking(unit(1, 0, 0), "learned")


@pytest.mark.parametrize("rule", ["mean", "learned"])
def test_a_model_whose_live_rule_is_not_best_photo_stays_evaluated(rule: str) -> None:
    identification = synthetic_identification()
    results = assemble(
        synthetic_verification(),
        identification,
        synthetic_learning(identification, {"facenet": rule}),  # type: ignore[dict-item]
    )

    evaluation = Evaluation.from_results(results)

    facenet = next(e for key, e in evaluation.models.items() if key.network == "facenet")
    assert len(evaluation.models) == 3
    assert (facenet.rule, facenet.threshold) == (rule, results.thresholds[2].threshold)
    assert facenet.learned_rule == results.thresholds[2].learned_rule
