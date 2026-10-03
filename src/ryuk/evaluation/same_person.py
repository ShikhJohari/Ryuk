"""The same-person threshold, the 1:1 cut-off of enrollment's `may_not_be_same_person` warning
(#49, #47 Q6).

A photo added to a person of interest warns that it may not be them when its best cosine to their
enrolled photos is under the threshold. That is verification, one face against one person, so the
1:N threshold is the wrong cut-off for it: frozen so that 1% of strangers match anyone in a gallery
of 2,500 photos, it is far stricter than any one pair needs, and with one photo enrolled it warned
on a fifth or more of a person's own photos under SFace and FaceNet.

Each model's threshold is measured on CelebA pairs as the warning sees them. A mated pair is a
gallery identity's enrolled photo and one of its own mated probes; an impostor pair, the same photo
and a probe of any other identity, gallery or held out. The threshold is frozen on the validation
draw's impostor pairs with one photo enrolled: the lowest cosine whose false-accept rate, the share
of impostor pairs at or above it, is at or below `TARGET_FAR`, read off the curve exactly as an
open-set threshold is. The test draw is then scored once at it, with one photo enrolled and with
every enrolled photo: the warning rate is the share of mated pairs under the threshold, a person's
own photos that warn, and the false-accept rate the share of impostor pairs that do not.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ryuk.eda.summary import Draw
from ryuk.evaluation.bootstrap import pair_ratio_interval
from ryuk.evaluation.draws import ENROLLED_PER_IDENTITY
from ryuk.evaluation.openset import (
    EmbeddedDraw,
    Gallery,
    IdentityGroups,
    ScoredProbes,
    tpir_at_fpir,
)
from ryuk.evaluation.results import Rate, SamePerson, SamePersonRates

TARGET_FAR: Final = 0.001
"""The false-accept rate each same-person threshold is frozen at: one impostor pair in a thousand
scores at or above it. On about six million validation impostor pairs that is some six thousand
pairs, so the threshold is well pinned down."""

ENROLLED_PHOTOS: Final[tuple[int, ...]] = (1, ENROLLED_PER_IDENTITY)
"""The photos enrolled per identity the test draw reports: a person's first photo, and the
rehearsal's every photo."""


@dataclass(frozen=True, eq=False)
class Pairs:
    """Every mated and impostor pair of one draw, each scored by the probe's best cosine to the
    first `photos` enrolled photos of a gallery identity."""

    draw: Draw
    photos: int
    mated_identity: NDArray[np.int_]
    """The gallery identity whose photos each mated pair compares with."""
    mated_score: NDArray[np.float64]
    impostor_identity: NDArray[np.int_]
    """The identity of each impostor pair's probe, the person who is not who they are added as."""
    impostor_gallery_identity: NDArray[np.int_]
    """The gallery identity whose photos each impostor pair compares with."""
    impostor_score: NDArray[np.float64]


def pairs(draw: EmbeddedDraw, photos: int) -> Pairs:
    """Each gallery identity's first `photos` enrolled photos against every probe of the draw."""
    if any(len(enrolled) < photos for enrolled in draw.enrolled.values()):
        raise ValueError(f"pairs with {photos} enrolled photos need {photos} enrolled photos each")
    gallery = Gallery.enrol(
        {identity: list(enrolled)[:photos] for identity, enrolled in draw.enrolled.items()}
    )
    identities = np.concatenate([draw.mated.identities, draw.non_mated.identities])
    probes = np.concatenate([draw.mated.embeddings, draw.non_mated.embeddings])
    # Probes by gallery identity: each probe's best cosine to that identity's photos.
    scores = np.maximum.reduceat(probes @ gallery.embeddings.T, gallery.starts, axis=1)
    scores = np.clip(scores.astype(np.float64), -1.0, 1.0)
    own = identities[:, np.newaxis] == gallery.identities[np.newaxis, :]
    probe_identity = np.broadcast_to(identities[:, np.newaxis], own.shape)
    return Pairs(
        draw=draw.draw,
        photos=photos,
        mated_identity=np.broadcast_to(gallery.identities[np.newaxis, :], own.shape)[own],
        mated_score=scores[own],
        impostor_identity=probe_identity[~own],
        impostor_gallery_identity=np.broadcast_to(gallery.identities[np.newaxis, :], own.shape)[
            ~own
        ],
        impostor_score=scores[~own],
    )


_SEAL: Final = object()


class FrozenSamePerson:
    """A same-person threshold, chosen on the validation draw's one-photo impostor pairs. Made
    only by `freeze_same_person`.

    Not a dataclass, so `dataclasses.replace` cannot copy one with another value, and its
    attributes are read-only.
    """

    __slots__ = ("_far", "_impostor_pairs", "_target_far", "_value")

    def __init__(
        self, value: float, target_far: float, far: float, impostor_pairs: int, seal: object = None
    ) -> None:
        if seal is not _SEAL:
            raise TypeError(
                "a same-person threshold is only made by freeze_same_person(), from the "
                "validation draw"
            )
        self._value = value
        self._target_far = target_far
        self._far = far
        self._impostor_pairs = impostor_pairs

    @property
    def value(self) -> float:
        return self._value

    @property
    def target_far(self) -> float:
        return self._target_far

    @property
    def far(self) -> float:
        """The false-accept rate on the validation draw at the threshold."""
        return self._far

    @property
    def impostor_pairs(self) -> int:
        return self._impostor_pairs

    def __repr__(self) -> str:
        return f"FrozenSamePerson(value={self._value!r}, target_far={self._target_far!r})"


def freeze_same_person(validation: Pairs, target_far: float = TARGET_FAR) -> FrozenSamePerson:
    """The lowest threshold whose false-accept rate on the validation draw's one-photo impostor
    pairs is at or below `target_far`."""
    if validation.draw != "validation":
        raise ValueError(f"only the validation draw sets a threshold, not the {validation.draw}")
    if validation.photos != 1:
        raise ValueError("a same-person threshold is set on pairs with one enrolled photo")
    # Impostor pairs stand where non-mated probes do on an open-set curve, and a mated pair is
    # always "right": the threshold is read off the same curve, by the same rule.
    point = tpir_at_fpir(
        ScoredProbes(
            draw=validation.draw,
            mated_identity=validation.mated_identity,
            mated_score=validation.mated_score,
            mated_correct=np.ones(validation.mated_score.size, dtype=np.bool_),
            non_mated_identity=validation.impostor_identity,
            non_mated_score=validation.impostor_score,
        ),
        target_far,
    )
    if not np.isfinite(point.threshold):
        raise ValueError(f"no threshold keeps FAR at or below {target_far} but accepting none")
    return FrozenSamePerson(
        point.threshold, target_far, point.fpir, validation.impostor_score.size, _SEAL
    )


def same_person(
    validation: EmbeddedDraw,
    test: EmbeddedDraw,
    *,
    live_threshold: float | None,
    seed: int,
    target_far: float = TARGET_FAR,
) -> SamePerson:
    """A model's same-person threshold frozen on `validation`, and `test` scored once at it.

    `live_threshold` is the model's 1:N threshold where it is a cosine, for the warning rate the
    warning had before #49; None under the learned rule, whose threshold is a probability.
    """
    frozen = freeze_same_person(pairs(validation, photos=1), target_far)
    return SamePerson(
        threshold=frozen.value,
        target_far=frozen.target_far,
        validation_far=frozen.far,
        validation_impostor_pairs=frozen.impostor_pairs,
        test=[
            _rates(pairs(test, photos), frozen, live_threshold, seed) for photos in ENROLLED_PHOTOS
        ],
    )


def _rates(
    scored: Pairs, frozen: FrozenSamePerson, live_threshold: float | None, seed: int
) -> SamePersonRates:
    # Mated pairs are clustered by the gallery identity, impostor pairs by it and by the probe's
    # identity too; gallery identities get the same resamples in both.
    groups = IdentityGroups.of_identities(scored.mated_identity, scored.impostor_identity, seed)

    def warning_rate(threshold: float) -> Rate:
        return groups.mated_rate(scored.mated_score < threshold, error=True)

    return SamePersonRates(
        enrolled_photos=scored.photos,
        mated_pairs=scored.mated_score.size,
        impostor_pairs=scored.impostor_score.size,
        warning_rate=warning_rate(frozen.value),
        far=_far(scored, scored.impostor_score >= frozen.value, groups),
        warning_rate_at_live_threshold=None
        if live_threshold is None
        else warning_rate(live_threshold),
    )


def _far(scored: Pairs, accepted: NDArray[np.bool_], groups: IdentityGroups) -> Rate:
    """The false-accept rate with the pigeonhole bootstrap's interval, both identities of each
    impostor pair resampled. The adjusted Wilson check, where the rate is under 1%, clusters by
    the probe's identity alone, as it does for FPIR."""
    by_probe = groups.non_mated_rate(accepted, error=True)
    gallery = np.unique(scored.mated_identity)
    rows = np.searchsorted(gallery, scored.impostor_gallery_identity)
    if not np.array_equal(gallery[rows], scored.impostor_gallery_identity):
        raise ValueError("every impostor pair's gallery identity has mated pairs too")
    columns = int(groups.non_mated.max()) + 1
    cell = rows * columns + groups.non_mated

    def counts(weights: NDArray[np.bool_] | None) -> NDArray[np.int_]:
        totals = np.bincount(cell, weights=weights, minlength=gallery.size * columns)
        return totals.astype(np.int_).reshape(gallery.size, columns)

    return Rate(
        value=by_probe.value,
        ci=pair_ratio_interval(
            counts(accepted), counts(None), groups.mated_weights, groups.non_mated_weights
        ),
        adjusted_wilson=by_probe.adjusted_wilson,
    )
