"""The gallery and held-out identities of one CelebA open-set draw (#9, #10, #27).

A draw is made from one CelebA split's usable images, after the images with no usable face are
excluded (#17). Identities with at least 20 usable images can be enrolled; 500 of them are drawn
at random into the gallery, each with 5 enrolled photos and 15 mated probes. Every other identity
with a usable image is held out, with up to 10 of its images as non-mated probes: #10 counts
about 485 to 500 held-out identities per draw, which is every identity outside the gallery.

The draw is Ryuk's own design, fixed by its seed: identities and images are put in sorted order
before anything random happens, so the same usable images and seed give the same draw.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ryuk.eda.summary import Draw

GALLERY_SIZE: Final = 500
ENROLLED_PER_IDENTITY: Final = 5
MATED_PROBES_PER_IDENTITY: Final = 15
NON_MATED_PROBES_PER_IDENTITY: Final = 10
MIN_GALLERY_IMAGES: Final = ENROLLED_PER_IDENTITY + MATED_PROBES_PER_IDENTITY
DRAW_SEED: Final = 27
"""The committed seed both draws are made with."""


@dataclass(frozen=True, slots=True)
class GalleryIdentity:
    identity: int
    enrolled: tuple[str, ...]
    """The enrolled photos, by CelebA file name."""
    probes: tuple[str, ...]
    """The mated probes, never among the enrolled photos."""


@dataclass(frozen=True, slots=True)
class HeldOutIdentity:
    identity: int
    probes: tuple[str, ...]
    """The non-mated probes."""


@dataclass(frozen=True, slots=True)
class OpenSetDraw:
    """Who is enrolled and which images are probes, identities in ascending order."""

    draw: Draw
    seed: int
    gallery: tuple[GalleryIdentity, ...]
    held_out: tuple[HeldOutIdentity, ...]

    def images(self) -> list[str]:
        """Every image the draw uses: enrolled photos, mated probes, then non-mated probes."""
        return [
            *(image for g in self.gallery for image in g.enrolled),
            *(image for g in self.gallery for image in g.probes),
            *(image for h in self.held_out for image in h.probes),
        ]

    @property
    def selection_sha256(self) -> str:
        """A digest of every choice the draw made, to check a rebuilt draw is the same one."""
        selection = {
            "draw": self.draw,
            "gallery": [[g.identity, g.enrolled, g.probes] for g in self.gallery],
            "held_out": [[h.identity, h.probes] for h in self.held_out],
        }
        return hashlib.sha256(json.dumps(selection, separators=(",", ":")).encode()).hexdigest()


class DrawMismatchError(ValueError):
    """A rebuilt draw is not the one its committed `selection_sha256` records."""


def check_selection(selection: OpenSetDraw, expected_sha256: str) -> None:
    """Raise DrawMismatchError unless `selection` is the draw whose digest was committed.

    A draw is rebuilt from the images a scan finds usable, so another detector, minimum face
    size, seed or copy of CelebA can give another draw; one identity crossing the 20-image line
    reshuffles the whole gallery.
    """
    actual = selection.selection_sha256
    if actual != expected_sha256:
        raise DrawMismatchError(
            f"the rebuilt {selection.draw} draw is not the committed one: its selection_sha256 is "
            f"{actual}, the results record {expected_sha256}. The usable images, the detector, "
            "the minimum face size or the seed changed"
        )


def make_draw(
    draw: Draw,
    usable: Mapping[int, Sequence[str]],
    seed: int,
    *,
    gallery_size: int = GALLERY_SIZE,
) -> OpenSetDraw:
    """Draw the gallery and held-out identities from each identity's usable images.

    `usable` maps every identity of the draw's split to the file names of its images with a
    usable face; an identity with none is left out of the draw. Raises ValueError if fewer than
    `gallery_size` identities have enough usable images to be enrolled.
    """
    rng = np.random.default_rng(seed)
    images = {identity: sorted(names) for identity, names in sorted(usable.items()) if names}
    candidates = [i for i, names in images.items() if len(names) >= MIN_GALLERY_IMAGES]
    if len(candidates) < gallery_size:
        raise ValueError(
            f"{len(candidates)} identities have at least {MIN_GALLERY_IMAGES} usable images, "
            f"{gallery_size} needed"
        )
    chosen = sorted(int(i) for i in rng.choice(candidates, size=gallery_size, replace=False))

    gallery = []
    for identity in chosen:
        shuffled = _shuffled(images[identity], rng)
        gallery.append(
            GalleryIdentity(
                identity,
                enrolled=shuffled[:ENROLLED_PER_IDENTITY],
                probes=shuffled[ENROLLED_PER_IDENTITY:MIN_GALLERY_IMAGES],
            )
        )
    enrolled = set(chosen)
    held_out = [
        HeldOutIdentity(identity, _shuffled(names, rng)[:NON_MATED_PROBES_PER_IDENTITY])
        for identity, names in images.items()
        if identity not in enrolled
    ]
    return OpenSetDraw(draw, seed, tuple(gallery), tuple(held_out))


def _shuffled(names: Sequence[str], rng: np.random.Generator) -> tuple[str, ...]:
    return tuple(names[int(i)] for i in rng.permutation(len(names)))
