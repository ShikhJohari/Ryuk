"""The first active model (#9), and the results file's blocks that follow from identification.

A model is eligible if it reproduces its published LFW accuracy within 0.5 points, keeps its test
draw FPIR at the frozen threshold at or below 2%, and takes at most 30 ms per face end to end.
Among eligible models the highest test TPIR at the frozen threshold wins; if another eligible
model's interval overlaps the leader's, the fastest of those wins instead. Each model is judged
under the rule it would run live, the best-photo rule until #10's comparison says otherwise.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ryuk.evaluation.results import (
    Eligibility,
    FirstActiveModel,
    Identification,
    ModelThreshold,
    Rate,
    RecognitionModelId,
    Results,
    Verification,
)
from ryuk.evaluation.tables import model_name
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
    candidates = [_judge(contender) for contender in contenders]
    eligible = [c for c in candidates if c.eligible]
    if not eligible:
        return FirstActiveModel(
            model=None,
            reason=(
                "No model is eligible: each must reproduce its published LFW accuracy within "
                f"{TOLERANCE_POINTS:g} points, keep test FPIR at or below {MAX_TEST_FPIR:.0%} "
                f"and take at most {MAX_MS_PER_FACE:g} ms per face."
            ),
            candidates=candidates,
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
    return FirstActiveModel(model=winner.model, reason=reason, candidates=candidates)


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


def _overlap(first: Rate, second: Rate) -> bool:
    return first.ci.low <= second.ci.high and second.ci.low <= first.ci.high


def assemble(verification: Verification, identification: Identification | None) -> Results:
    """The whole results file: identification's thresholds block and first active model are
    derived here, so they always agree with the sections they come from."""
    if identification is None:
        return Results(verification=verification)
    lfw = {result.model: result for result in verification.models}
    provenance = identification.provenance
    return Results(
        verification=verification,
        identification=identification,
        thresholds=[
            ModelThreshold(
                model=result.model,
                rule=result.rule,
                threshold=result.threshold,
                target_fpir=result.target_fpir,
                commit=provenance.commit,
                date=provenance.generated_at.date(),
            )
            for result in identification.models
        ],
        first_active_model=first_active_model(
            [
                Contender(
                    model=result.model,
                    lfw_gap_points=lfw[result.model].gap_points if result.model in lfw else None,
                    test_tpir=result.test.at_threshold.tpir,
                    test_fpir=result.test.at_threshold.fpir.value,
                    ms_per_face=result.ms_per_face,
                )
                for result in identification.models
            ]
        ),
    )
