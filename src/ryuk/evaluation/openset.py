"""Open-set identification on one CelebA draw under the best-photo rule (#9, #10, #27).

Each probe is scored against every gallery identity by its match score: the cosine to that
identity's best enrolled photo. Its top candidate is the highest-scoring identity. At a threshold
t, a probe is a match iff its top candidate scores at or above t, as in the live monitor.

- Rank-1: the share of mated probes whose top candidate is right, whatever the score.
- TPIR: the share of mated probes that match their own identity.
- Misidentification rate: the share of mated probes that match the wrong identity.
- FPIR: the share of non-mated probes that match anyone.

A model's threshold is frozen on the validation draw: the lowest threshold whose FPIR is at or
below 1% there. Only `freeze` makes a `FrozenThreshold`, and it refuses the test draw, so the
test draw can only ever be scored at a threshold the validation draw chose.

Two methods scored on the same probes are compared by `paired_gain`: both are read in the same
bootstrap resamples, so the interval of their difference carries only the noise they do not
share (#10).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ryuk.eda.summary import Draw
from ryuk.evaluation.bootstrap import (
    CONFIDENCE,
    RESAMPLES,
    WILSON_BELOW,
    adjusted_wilson,
    identity_weights,
    percentile_interval,
    ratio_interval,
)
from ryuk.evaluation.results import (
    AtThreshold,
    DrawResult,
    Gain,
    Interval,
    OpenSetCurve,
    OpenSetPoint,
    Rate,
    SignedInterval,
)
from ryuk.recognition import Embedding

TARGET_FPIR: Final = 0.01
"""The FPIR each model's threshold is frozen at on the validation draw (#9)."""

FPIR_TARGETS: Final[tuple[tuple[float, bool], ...]] = ((0.01, False), (0.001, True))
"""Operating points read off each draw's curve, and whether each is indicative: 0.1% of about
4,800 non-mated probes is about 5 false alarms."""

CURVE_POINTS: Final = 200


@dataclass(frozen=True, eq=False)
class Gallery:
    """The enrolled embeddings of a draw's gallery, rows grouped by identity in `identities`
    order; `starts[i]` is identity i's first row."""

    identities: NDArray[np.int_]
    embeddings: NDArray[np.float32]
    starts: NDArray[np.int_]

    @classmethod
    def enrol(cls, enrolled: Mapping[int, Sequence[Embedding]]) -> "Gallery":
        """A gallery from each identity's enrolled embeddings; every identity needs one."""
        if not enrolled or any(not photos for photos in enrolled.values()):
            raise ValueError("every gallery identity needs at least one enrolled embedding")
        identities = sorted(enrolled)
        sizes = [len(enrolled[identity]) for identity in identities]
        return cls(
            identities=np.array(identities, dtype=np.int_),
            embeddings=np.stack([e for i in identities for e in enrolled[i]]).astype(np.float32),
            starts=np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int_),
        )

    def averaged(self) -> "Gallery":
        """One row per identity, the renormalised mean of its enrolled embeddings, so that its
        best photo is its mean and `top_candidates` scores under the mean rule."""
        means = np.add.reduceat(self.embeddings, self.starts, axis=0)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("an identity's enrolled embeddings cancel out, so have no mean")
        return Gallery(
            identities=self.identities,
            embeddings=(means / norms).astype(np.float32),
            starts=np.arange(self.identities.size, dtype=np.int_),
        )

    def top_candidates(
        self, probes: NDArray[np.float32]
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Each probe's top candidate and its match score; a tie goes to the lower identity."""
        cosines = probes @ self.embeddings.T
        by_identity = np.maximum.reduceat(cosines, self.starts, axis=1)
        top = np.argmax(by_identity, axis=1)
        scores = by_identity[np.arange(len(top)), top]
        return self.identities[top], np.clip(scores.astype(np.float64), -1.0, 1.0)

    def top_two(self, probes: NDArray[np.float32]) -> "TopTwo":
        """Each probe's top candidate and match score, as `top_candidates` gives them, and the
        runner-up's: the best score of any other identity."""
        if self.identities.size < 2:
            raise ValueError("a runner-up needs a gallery of at least two identities")
        by_identity = np.maximum.reduceat(probes @ self.embeddings.T, self.starts, axis=1)
        rows = np.arange(by_identity.shape[0])
        top = np.argmax(by_identity, axis=1)
        scores = by_identity[rows, top]
        by_identity[rows, top] = -np.inf
        return TopTwo(
            identities=self.identities[top],
            scores=np.clip(scores.astype(np.float64), -1.0, 1.0),
            runner_up_scores=np.clip(by_identity.max(axis=1).astype(np.float64), -1.0, 1.0),
        )


@dataclass(frozen=True, eq=False)
class TopTwo:
    """Each probe's top candidate and match score under the best-photo rule, and the runner-up's
    match score: the second-best identity, never the top one's second photo."""

    identities: NDArray[np.int_]
    scores: NDArray[np.float64]
    runner_up_scores: NDArray[np.float64]

    @property
    def gaps(self) -> NDArray[np.float64]:
        """How far each top candidate scores above the runner-up; 0 on a tie."""
        return self.scores - self.runner_up_scores


@dataclass(frozen=True, eq=False)
class Probes:
    """Probe embeddings, one row each, and the identity each belongs to."""

    identities: NDArray[np.int_]
    embeddings: NDArray[np.float32]
    images: tuple[str, ...] = ()
    """Each probe's image, where the caller knows it: photo-condition groups need its labels."""


@dataclass(frozen=True, eq=False)
class EmbeddedDraw:
    """One draw's embeddings under one recognition model: each gallery identity's enrolled
    embeddings, and the mated and non-mated probes. Everything a method sees of a draw."""

    draw: Draw
    enrolled: Mapping[int, Sequence[Embedding]]
    mated: Probes
    non_mated: Probes

    def score(self) -> "ScoredProbes":
        """The draw scored under the best-photo rule."""
        return score_probes(self.draw, Gallery.enrol(self.enrolled), self.mated, self.non_mated)


@dataclass(frozen=True, eq=False)
class ScoredProbes:
    """Every probe of one draw reduced to its top candidate's score, and for a mated probe
    whether that candidate is right."""

    draw: Draw
    mated_identity: NDArray[np.int_]
    mated_score: NDArray[np.float64]
    mated_correct: NDArray[np.bool_]
    non_mated_identity: NDArray[np.int_]
    non_mated_score: NDArray[np.float64]


def score_probes(draw: Draw, gallery: Gallery, mated: Probes, non_mated: Probes) -> ScoredProbes:
    mated_top, mated_score = gallery.top_candidates(mated.embeddings)
    _, non_mated_score = gallery.top_candidates(non_mated.embeddings)
    return ScoredProbes(
        draw=draw,
        mated_identity=np.asarray(mated.identities, dtype=np.int_),
        mated_score=mated_score,
        mated_correct=mated_top == mated.identities,
        non_mated_identity=np.asarray(non_mated.identities, dtype=np.int_),
        non_mated_score=non_mated_score,
    )


def rank_1(scored: ScoredProbes) -> float:
    return float(np.mean(scored.mated_correct))


@dataclass(frozen=True, eq=False)
class Curve:
    """The full TPIR/FPIR curve, one point per distinct score, highest first, from +inf."""

    thresholds: NDArray[np.float64]
    fpir: NDArray[np.float64]
    tpir: NDArray[np.float64]
    misidentification: NDArray[np.float64]


def open_set_curve(scored: ScoredProbes) -> Curve:
    scores = np.concatenate([scored.mated_score, scored.non_mated_score])
    mated = scored.mated_score.size
    right = np.concatenate([scored.mated_correct, np.zeros(scored.non_mated_score.size, bool)])
    wrong = np.concatenate([~scored.mated_correct, np.zeros(scored.non_mated_score.size, bool)])
    alarm = np.concatenate([np.zeros(mated, bool), np.ones(scored.non_mated_score.size, bool)])
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    # The last probe of each run of equal scores, where the whole run has been accepted.
    ends = np.append(np.flatnonzero(np.diff(ranked)), ranked.size - 1)

    def rate(flags: NDArray[np.bool_], total: int) -> NDArray[np.float64]:
        return np.append(0.0, np.cumsum(flags[order])[ends] / total)

    return Curve(
        thresholds=np.append(np.inf, ranked[ends]),
        fpir=rate(alarm, scored.non_mated_score.size),
        tpir=rate(right, mated),
        misidentification=rate(wrong, mated),
    )


@dataclass(frozen=True, slots=True)
class TpirAtFpir:
    target_fpir: float
    tpir: float
    fpir: float
    threshold: float
    """+inf when only accepting nothing meets the target."""


def tpir_at_fpir(scored: ScoredProbes, target_fpir: float) -> TpirAtFpir:
    """The curve point with the lowest threshold whose FPIR is at or below `target_fpir`."""
    if not 0 <= target_fpir <= 1:
        raise ValueError(f"target_fpir must be between 0 and 1, got {target_fpir}")
    curve = open_set_curve(scored)
    point = np.flatnonzero(curve.fpir <= target_fpir)[-1]
    return TpirAtFpir(
        target_fpir=target_fpir,
        tpir=float(curve.tpir[point]),
        fpir=float(curve.fpir[point]),
        threshold=float(curve.thresholds[point]),
    )


_SEAL: Final = object()


class FrozenThreshold:
    """A model's threshold, chosen on the validation draw. Made only by `freeze`.

    Not a dataclass, so `dataclasses.replace` cannot copy one with another value, and its
    attributes are read-only.
    """

    __slots__ = ("_target_fpir", "_value")

    def __init__(self, value: float, target_fpir: float, seal: object = None) -> None:
        if seal is not _SEAL:
            raise TypeError("a threshold is only made by freeze(), from the validation draw")
        self._value = value
        self._target_fpir = target_fpir

    @property
    def value(self) -> float:
        return self._value

    @property
    def target_fpir(self) -> float:
        return self._target_fpir

    def __repr__(self) -> str:
        return f"FrozenThreshold(value={self._value!r}, target_fpir={self._target_fpir!r})"


def freeze(validation: ScoredProbes, target_fpir: float = TARGET_FPIR) -> FrozenThreshold:
    """The lowest threshold with FPIR at or below `target_fpir` on the validation draw."""
    if validation.draw != "validation":
        raise ValueError(f"only the validation draw sets a threshold, not the {validation.draw}")
    point = tpir_at_fpir(validation, target_fpir)
    if not np.isfinite(point.threshold):
        raise ValueError(f"no threshold keeps FPIR at or below {target_fpir} but accepting none")
    return FrozenThreshold(point.threshold, target_fpir, _SEAL)


@dataclass(frozen=True)
class IdentityGroups:
    """Probes grouped by identity for the bootstrap: each probe's identity index and weights."""

    mated: NDArray[np.int_]
    non_mated: NDArray[np.int_]
    mated_weights: NDArray[np.int_]
    non_mated_weights: NDArray[np.int_]

    @classmethod
    def of(cls, scored: ScoredProbes, seed: int) -> "IdentityGroups":
        return cls.of_identities(scored.mated_identity, scored.non_mated_identity, seed)

    @classmethod
    def of_identities(
        cls, mated_identity: NDArray[np.int_], non_mated_identity: NDArray[np.int_], seed: int
    ) -> "IdentityGroups":
        """Mated and non-mated probes grouped by each one's identity, resampled with `seed`."""
        # Gallery and held-out identities are different people, resampled independently.
        rng = np.random.default_rng(seed)
        _, mated = np.unique(mated_identity, return_inverse=True)
        _, non_mated = np.unique(non_mated_identity, return_inverse=True)
        return cls(
            mated=mated,
            non_mated=non_mated,
            mated_weights=identity_weights(int(mated.max()) + 1, RESAMPLES, rng),
            non_mated_weights=identity_weights(int(non_mated.max()) + 1, RESAMPLES, rng),
        )

    def mated_rate(self, flags: NDArray[np.bool_], *, error: bool) -> Rate:
        return probe_rate(flags, self.mated, self.mated_weights, error=error)

    def non_mated_rate(self, flags: NDArray[np.bool_], *, error: bool) -> Rate:
        return probe_rate(flags, self.non_mated, self.non_mated_weights, error=error)


def draw_result(scored: ScoredProbes, threshold: FrozenThreshold, *, seed: int) -> DrawResult:
    """A draw's rank-1, rates at the frozen threshold, operating points and curve, each rate with
    its identity-level interval."""
    groups = IdentityGroups.of(scored, seed)
    t = threshold.value
    accepted = scored.mated_score >= t
    return DrawResult(
        draw=scored.draw,
        mated_probes=scored.mated_score.size,
        non_mated_probes=scored.non_mated_score.size,
        rank_1=groups.mated_rate(scored.mated_correct, error=False),
        at_threshold=AtThreshold(
            threshold=t,
            tpir=groups.mated_rate(accepted & scored.mated_correct, error=False),
            fpir=groups.non_mated_rate(scored.non_mated_score >= t, error=True),
            misidentification=groups.mated_rate(accepted & ~scored.mated_correct, error=True),
        ),
        operating_points=[
            _operating_point(scored, groups, target, indicative)
            for target, indicative in FPIR_TARGETS
        ],
        curve=_downsampled(open_set_curve(scored), scored.non_mated_score.size),
    )


def probe_rate(
    flags: NDArray[np.bool_], group: NDArray[np.int_], weights: NDArray[np.int_], *, error: bool
) -> Rate:
    """The share of probes flagged, with its bootstrap interval, and the adjusted Wilson check
    when the error rate (the rate itself, or its complement for a success rate) is under 1%.

    `group[i]` is probe i's identity as an index into the columns of `weights`, which come from
    `identity_weights` over exactly those identities."""
    groups = weights.shape[1]
    flagged = np.bincount(group, weights=flags, minlength=groups).astype(np.int_)
    trials = np.bincount(group, minlength=groups).astype(np.int_)
    value = float(flagged.sum() / trials.sum())
    wilson = None
    if (value if error else 1 - value) < WILSON_BELOW:
        if error:
            wilson = adjusted_wilson(flagged, trials)
        else:
            errors = adjusted_wilson(trials - flagged, trials)
            wilson = Interval(low=1 - errors.high, high=1 - errors.low)
    return Rate(value=value, ci=ratio_interval(flagged, trials, weights), adjusted_wilson=wilson)


def paired_gain(
    baseline: ScoredProbes, method: ScoredProbes, target_fpir: float, *, seed: int
) -> Gain:
    """`method`'s TPIR at `target_fpir` minus `baseline`'s, each read off its own curve, with the
    percentile interval of that difference over resamples both are read in.

    Both must score the same probes of the same draw: the pairing is by identity."""
    if not (
        baseline.draw == method.draw
        and np.array_equal(baseline.mated_identity, method.mated_identity)
        and np.array_equal(baseline.non_mated_identity, method.non_mated_identity)
    ):
        raise ValueError("a gain compares two methods on the same probes of the same draw")
    groups = IdentityGroups.of(baseline, seed)
    differences = _resampled_tpir(method, groups, target_fpir) - _resampled_tpir(
        baseline, groups, target_fpir
    )
    tail = (1 - CONFIDENCE) / 2
    low, high = np.quantile(differences, [tail, 1 - tail])
    return Gain(
        target_fpir=target_fpir,
        value=tpir_at_fpir(method, target_fpir).tpir - tpir_at_fpir(baseline, target_fpir).tpir,
        ci=SignedInterval(low=float(low), high=float(high)),
        improves=bool(low > 0),
    )


def _operating_point(
    scored: ScoredProbes, groups: IdentityGroups, target: float, indicative: bool
) -> OpenSetPoint:
    point = tpir_at_fpir(scored, target)
    values = _resampled_tpir(scored, groups, target)
    return OpenSetPoint(
        target_fpir=target,
        fpir=point.fpir,
        tpir=Rate(value=point.tpir, ci=percentile_interval(values)),
        threshold=point.threshold if np.isfinite(point.threshold) else None,
        indicative=indicative,
    )


def _resampled_tpir(
    scored: ScoredProbes, groups: IdentityGroups, target: float
) -> NDArray[np.float64]:
    """TPIR at `target` in each of `groups`' resamples, each choosing its own threshold."""
    order = np.argsort(-scored.non_mated_score, kind="stable")
    ranked = scored.non_mated_score[order]
    right = scored.mated_correct
    values = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        alarms = groups.non_mated_weights[r, groups.non_mated[order]]
        over = np.flatnonzero(np.cumsum(alarms) > target * alarms.sum())
        # Every probe scoring above the first non-mated score that breaks the budget matches.
        floor = ranked[over[0]] if over.size else -np.inf
        weights = groups.mated_weights[r, groups.mated]
        values[r] = weights[right & (scored.mated_score > floor)].sum() / weights.sum()
    return values


def _downsampled(curve: Curve, non_mated: int) -> OpenSetCurve:
    """About `CURVE_POINTS` points of the curve, evenly spaced in log FPIR from one false alarm.

    For each FPIR on the grid, the point kept is the curve's last at or below it, so every point
    kept lies on the curve; the first point at FPIR 0 and the last at 1 are always kept.
    """
    grid = np.geomspace(1 / non_mated, 1.0, CURVE_POINTS)
    picks = np.searchsorted(curve.fpir, grid, side="right") - 1
    last_at_zero = np.flatnonzero(curve.fpir == 0)[-1]
    keep = np.unique(np.concatenate([[last_at_zero], picks, [curve.fpir.size - 1]]))
    return OpenSetCurve(
        fpir=[float(v) for v in curve.fpir[keep]], tpir=[float(v) for v in curve.tpir[keep]]
    )
