"""The live operating point away from the rehearsal: smaller galleries and one enrolled photo
(#49, #47 Q6).

Every threshold was frozen on galleries of 500 identities with 5 enrolled photos each, but a live
watchlist starts with one photo per person and a handful of people. Under best-photo a stranger
has fewer photos to come close to on a smaller watchlist, so FPIR at the same threshold falls,
and a person with one photo enrolled has fewer chances to be matched, so TPIR falls too. Both are
measured here on the test draw, at each model's frozen threshold under the rule it runs live.

The test draw's gallery is split at random into disjoint galleries of one size, and each is
scored as a watchlist of its own: each mated probe against the gallery its identity is in, every
non-mated probe against every gallery. The split depends only on its seed and the size, so the
same galleries are scored with every enrolled photo and with each identity's first. Rates pool
the galleries; their intervals resample identities, gallery and held out separately, each
identity carrying its probes from every gallery it was scored in.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ryuk.evaluation.bootstrap import RESAMPLES, identity_weights
from ryuk.evaluation.draws import ENROLLED_PER_IDENTITY
from ryuk.evaluation.learning import score_rule
from ryuk.evaluation.openset import EmbeddedDraw, Probes, probe_rate
from ryuk.evaluation.results import ModelThreshold, SmallGallery

GALLERY_SIZES: Final[tuple[int, ...]] = (100, 20, 5)
"""The smaller galleries scored after the rehearsal's own: each divides its 500 identities."""

PARTITION_SEED: Final = 49
"""The committed seed of the split into smaller galleries."""

ENROLLED_PHOTOS: Final[tuple[int, ...]] = (ENROLLED_PER_IDENTITY, 1)
"""Every enrolled photo, as rehearsed, then each identity's first, as a watchlist starts."""


def partition(identities: Sequence[int], size: int, *, seed: int) -> NDArray[np.int_]:
    """The identities split at random into disjoint galleries of `size`, one row each."""
    if size < 1 or len(identities) % size:
        raise ValueError(f"cannot split {len(identities)} identities into galleries of {size} each")
    rng = np.random.default_rng([seed, size])
    shuffled = rng.permutation(np.array(sorted(identities), dtype=np.int_))
    return shuffled.reshape(-1, size)


def small_galleries(
    draw: EmbeddedDraw, frozen: ModelThreshold, *, seed: int, sizes: Sequence[int] = GALLERY_SIZES
) -> list[SmallGallery]:
    """`draw` under `frozen`'s live rule at its threshold, on its own gallery and then on
    galleries of each of `sizes`, each with every enrolled photo and then one."""
    identities = sorted(draw.enrolled)
    if frozen.rule == "learned" and any(size < 2 for size in sizes):
        raise ValueError(
            "the learned rule scores a runner-up: a gallery needs at least two identities"
        )
    return [
        _scored(draw, frozen, galleries, photos, seed)
        for galleries in [
            partition(identities, size, seed=PARTITION_SEED) for size in (len(identities), *sizes)
        ]
        for photos in ENROLLED_PHOTOS
    ]


def _scored(
    draw: EmbeddedDraw,
    frozen: ModelThreshold,
    galleries: NDArray[np.int_],
    photos: int,
    seed: int,
) -> SmallGallery:
    right, wrong, mated, alarms, non_mated = [], [], [], [], []
    for members in galleries:
        own = np.isin(draw.mated.identities, members)
        watchlist = EmbeddedDraw(
            draw=draw.draw,
            enrolled={int(i): list(draw.enrolled[int(i)])[:photos] for i in sorted(members)},
            mated=Probes(draw.mated.identities[own], draw.mated.embeddings[own]),
            non_mated=draw.non_mated,
        )
        scored = score_rule(frozen.rule, watchlist, frozen.learned_rule)
        accepted = scored.mated_score >= frozen.threshold
        right.append(accepted & scored.mated_correct)
        wrong.append(accepted & ~scored.mated_correct)
        mated.append(scored.mated_identity)
        alarms.append(scored.non_mated_score >= frozen.threshold)
        non_mated.append(scored.non_mated_identity)
    rng = np.random.default_rng(seed)
    _, mated_group = np.unique(np.concatenate(mated), return_inverse=True)
    _, non_mated_group = np.unique(np.concatenate(non_mated), return_inverse=True)
    mated_weights = identity_weights(int(mated_group.max()) + 1, RESAMPLES, rng)
    non_mated_weights = identity_weights(int(non_mated_group.max()) + 1, RESAMPLES, rng)
    return SmallGallery(
        identities=galleries.shape[1],
        enrolled_photos=photos,
        galleries=galleries.shape[0],
        mated_probes=mated_group.size,
        non_mated_probes=non_mated_group.size,
        tpir=probe_rate(np.concatenate(right), mated_group, mated_weights, error=False),
        fpir=probe_rate(np.concatenate(alarms), non_mated_group, non_mated_weights, error=True),
        misidentification=probe_rate(np.concatenate(wrong), mated_group, mated_weights, error=True),
    )
