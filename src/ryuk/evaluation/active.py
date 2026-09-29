"""The first active model (#9), and the results file's blocks that follow from identification.

A model is eligible if it reproduces its published LFW accuracy within 0.5 points, keeps its test
draw FPIR at the frozen threshold at or below 2%, and takes at most 30 ms per face end to end.
Among eligible models the highest test TPIR at the frozen threshold wins; if another eligible
model's interval overlaps the leader's, the fastest of those wins instead. Each model is judged
under the rule it would run live: best-photo, unless learning found a rule that needs no
retraining and improves on it (#10), in which case its threshold is a cut-off on that rule.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ryuk.evaluation.names import model_name
from ryuk.evaluation.results import (
    Bias,
    DrawResult,
    Eligibility,
    FirstActiveModel,
    Identification,
    LearnedRule,
    Learning,
    LearningModel,
    MatchRule,
    ModelThreshold,
    OpenSetModel,
    Provenance,
    Rate,
    RecognitionModelId,
    Results,
    Verification,
    bias_mismatch,
    learning_mismatch,
)
from ryuk.evaluation.verification import TOLERANCE_POINTS

MAX_TEST_FPIR: Final = 0.02
MAX_MS_PER_FACE: Final = 30.0


@dataclass(frozen=True, slots=True)
class Contender:
    model: RecognitionModelId
    lfw_gap_points: float | None
    """None if LFW has no result for this exact model."""
    test_tpir: Rate
    test_fpir: float
    ms_per_face: float


def first_active_model(contenders: Sequence[Contender]) -> FirstActiveModel:
    eligibility = [_judge(contender) for contender in contenders]
    eligible = [c for c in eligibility if c.eligible]
    if not eligible:
        return FirstActiveModel(
            model=None,
            reason=(
                "No model is eligible: each must reproduce its published LFW accuracy within "
                f"{TOLERANCE_POINTS:g} points, keep test FPIR at or below {MAX_TEST_FPIR:.0%} "
                f"and take at most {MAX_MS_PER_FACE:g} ms per face."
            ),
            eligibility=eligibility,
        )
    leader = max(eligible, key=lambda c: c.test_tpir.value)
    close = [c for c in eligible if _overlap(c.test_tpir, leader.test_tpir)]
    winner = min(close, key=lambda c: (c.ms_per_face, -c.test_tpir.value))
    tpir = f"{winner.test_tpir.value:.2%}"
    if winner is leader and len(close) == 1:
        reason = (
            f"{model_name(winner.model)} has the highest test TPIR at its frozen threshold "
            f"({tpir}) of the {len(eligible)} eligible models, and no other's interval overlaps it."
        )
    elif winner is leader:
        reason = (
            f"{model_name(winner.model)} has the highest test TPIR at its frozen threshold "
            f"({tpir}) and is also the fastest of the eligible models whose intervals overlap it."
        )
    else:
        reason = (
            f"{model_name(winner.model)}'s test TPIR interval overlaps "
            f"{model_name(leader.model)}'s ({tpir} against {leader.test_tpir.value:.2%}), and it "
            f"is faster ({winner.ms_per_face:.1f} against {leader.ms_per_face:.1f} ms per face)."
        )
    return FirstActiveModel(model=winner.model, reason=reason, eligibility=eligibility)


def _judge(contender: Contender) -> Eligibility:
    gap = contender.lfw_gap_points
    reproduces = gap is not None and abs(gap) <= TOLERANCE_POINTS
    within = contender.test_fpir <= MAX_TEST_FPIR
    fast = contender.ms_per_face <= MAX_MS_PER_FACE
    return Eligibility(
        model=contender.model,
        lfw_gap_points=gap,
        reproduces_lfw=reproduces,
        test_tpir=contender.test_tpir,
        test_fpir=contender.test_fpir,
        fpir_within_limit=within,
        ms_per_face=contender.ms_per_face,
        fast_enough=fast,
        eligible=reproduces and within and fast,
    )


def failures(judged: Eligibility) -> list[str]:
    """Why a model is not eligible, one clause per test it failed."""
    reasons = []
    if not judged.reproduces_lfw:
        reasons.append(
            "LFW did not score it"
            if judged.lfw_gap_points is None
            else f"its LFW accuracy is {judged.lfw_gap_points:+.2f} points from published"
        )
    if not judged.fpir_within_limit:
        reasons.append(f"its test FPIR is {judged.test_fpir:.2%}, over {MAX_TEST_FPIR:.0%}")
    if not judged.fast_enough:
        reasons.append(f"it takes {judged.ms_per_face:.1f} ms per face, over {MAX_MS_PER_FACE:g}")
    return reasons


def _overlap(first: Rate, second: Rate) -> bool:
    return first.ci.low <= second.ci.high and second.ci.low <= first.ci.high


def assemble(
    verification: Verification,
    identification: Identification | None,
    learning: Learning | None = None,
    bias: Bias | None = None,
) -> Results:
    """The whole results file: identification's thresholds block and first active model are
    derived here, each model under its live rule, so they always agree with the sections they
    come from."""
    if identification is None:
        return Results(verification=verification, learning=learning, bias=bias)
    lfw = {result.model: result for result in verification.models}
    live = [_live(result, identification, learning) for result in identification.models]
    return Results(
        verification=verification,
        identification=identification,
        thresholds=[
            ModelThreshold(
                model=rule.model,
                rule=rule.rule,
                threshold=rule.threshold,
                target_fpir=rule.target_fpir,
                learned_rule=rule.learned_rule,
                commit=rule.frozen_by.commit,
                date=rule.frozen_by.generated_at.date(),
            )
            for rule in live
        ],
        first_active_model=first_active_model(
            [
                Contender(
                    model=rule.model,
                    lfw_gap_points=lfw[rule.model].gap_points if rule.model in lfw else None,
                    test_tpir=rule.test.at_threshold.tpir,
                    test_fpir=rule.test.at_threshold.fpir.value,
                    ms_per_face=rule.ms_per_face,
                )
                for rule in live
            ]
        ),
        learning=learning,
        bias=bias,
    )


@dataclass(frozen=True, slots=True)
class Carried:
    """What of the learning and bias sections still applies after identification is rerun or
    dropped, and why the rest does not."""

    learning: Learning | None
    bias: Bias | None
    dropped: tuple[str, ...]


def carried(
    identification: Identification | None, learning: Learning | None, bias: Bias | None
) -> Carried:
    """Keep learning and bias where they still describe `identification`; each is regenerated
    from the cached embeddings by `ryuk evaluate learn` and `ryuk evaluate bias`, so a stale one is dropped."""
    dropped: list[str] = []
    if mismatch := learning_mismatch(identification, learning):
        dropped.append(f"{mismatch}; run `ryuk evaluate learn` again")
        learning = None
    if mismatch := bias_mismatch(identification, learning, bias):
        dropped.append(f"{mismatch}; run `ryuk evaluate bias` again")
        bias = None
    return Carried(learning, bias, tuple(dropped))


@dataclass(frozen=True, slots=True)
class _LiveRule:
    """What one model runs live, and the test draw it is judged on for the first active model."""

    model: RecognitionModelId
    rule: MatchRule
    threshold: float
    target_fpir: float
    learned_rule: LearnedRule | None
    test: DrawResult
    ms_per_face: float
    frozen_by: Provenance
    """The run that froze the threshold."""


def _live(
    rehearsed: OpenSetModel, identification: Identification, learning: Learning | None
) -> _LiveRule:
    """`rehearsed`'s live rule: best-photo as identification measured it, or learning's winner.

    A winner keeps identification's ms per face: the rules differ only in the gallery search,
    microseconds against the milliseconds of detection and embedding.
    """
    compared = _compared(rehearsed.model, learning)
    if learning is None or compared is None or compared.live_rule == "best-photo":
        return _LiveRule(
            model=rehearsed.model,
            rule=rehearsed.rule,
            threshold=rehearsed.threshold,
            target_fpir=rehearsed.target_fpir,
            learned_rule=None,
            test=rehearsed.test,
            ms_per_face=rehearsed.ms_per_face,
            frozen_by=identification.provenance,
        )
    winner = compared.method(compared.live_rule)
    return _LiveRule(
        model=rehearsed.model,
        rule=compared.live_rule,
        threshold=winner.threshold,
        target_fpir=learning.target_fpir,
        learned_rule=compared.learned_rule if compared.live_rule == "learned" else None,
        test=winner.test,
        ms_per_face=rehearsed.ms_per_face,
        frozen_by=learning.provenance,
    )


def _compared(model: RecognitionModelId, learning: Learning | None) -> LearningModel | None:
    if learning is None:
        return None
    return next((m for m in learning.models if m.model == model), None)
