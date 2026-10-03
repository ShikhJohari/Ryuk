"""Synthetic embedded draws: gallery and held-out identities around random centres, built
without data or weights."""

import numpy as np
from numpy.typing import NDArray

from ryuk.eda.summary import Draw
from ryuk.evaluation.openset import EmbeddedDraw, Probes


def unit_rows(values: NDArray[np.float64]) -> NDArray[np.float32]:
    return (values / np.linalg.norm(values, axis=-1, keepdims=True)).astype(np.float32)


def embedded_draw(
    draw: Draw, seed: int, *, identities: int = 24, held_out: int = 30, noise: float = 0.25
) -> EmbeddedDraw:
    """`identities` enrolled with 5 photos and 6 mated probes each, and `held_out` identities
    with 6 non-mated probes each, around random centres in 16 dimensions. The draws share no
    identities, as CelebA's do not."""
    rng = np.random.default_rng(seed)
    first = 0 if draw == "validation" else 10_000
    centres = unit_rows(rng.normal(size=(identities + held_out, 16)))

    def photos(centre: int, count: int) -> NDArray[np.float32]:
        return unit_rows(centres[centre] + rng.normal(scale=noise, size=(count, 16)))

    gallery = range(identities)
    others = range(identities, identities + held_out)
    return EmbeddedDraw(
        draw=draw,
        enrolled={first + i: list(photos(i, 5)) for i in gallery},
        mated=Probes(
            np.repeat([first + i for i in gallery], 6),
            np.concatenate([photos(i, 6) for i in gallery]),
        ),
        non_mated=Probes(
            np.repeat([first + 5000 + i for i in others], 6),
            np.concatenate([photos(i, 6) for i in others]),
        ),
    )
