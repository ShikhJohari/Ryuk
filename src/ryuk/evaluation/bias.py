"""Who the system fails: per-group rates on the test draw at a model's single frozen threshold
(#10, #28).

No group gets its own threshold: that would be a fairness intervention, left to future work. At
the one threshold t, as in `openset.draw_result`, a mated probe counts for TPIR if its top
candidate is right and scores at least t, for misidentification if it is wrong and scores at
least t; a non-mated probe is a false alarm if its top candidate scores at least t.

Two kinds of group, each attribute split into with and without it:

- Identity groups (Male, Young): an identity's majority label over all its images in the split,
  usable or not, with the EDA's 80% agreement (`eda.aggregate.majority_label`), so the groups are
  Section 2's. An identity with no majority is left out of that attribute and counted. TPIR and
  misidentification are over the mated probes of the group's gallery identities, FPIR over the
  non-mated probes of its held-out identities. The Male and Young cells are indicative only, and
  leave out an identity with either label mixed.
- Photo conditions (Eyeglasses, Wearing_Hat, Blurry): each probe's own image label, so one
  identity's probes can fall in both groups.

Each rate's interval is the identity-level percentile bootstrap over the identities behind it,
with the adjusted Wilson check below 1% (`openset.probe_rate`). Every (attribute, group, side)
resamples from its own generator, seeded with the bootstrap seed and the attribute's position in
`BREAKDOWNS`, the group's position within it and the side (0 gallery, 1 held out). Those
positions are fixed by the attribute, not by which groups have probes, so a group's intervals
depend on its own probes alone. A side with fewer than `MIN_IDENTITIES` identities gets no rates,
but the group is still reported with its counts (#10). The gallery is drawn at random, not
stratified; the counts are its composition.

Group labels read as the attribute's phrase or its negation, capitalised: "Male" and "Not male",
"Young" and "Not young", "Eyeglasses" and "No eyeglasses", "Hat" and "No hat", "Blurry" and "Not
blurry"; a cell joins its two with a comma, as "Male, not young".
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ryuk.datasets.celeba import read_labels
from ryuk.eda.aggregate import majority_label
from ryuk.eda.build import RULES
from ryuk.eda.summary import (
    ATTRIBUTES,
    DRAW_SPLITS,
    IDENTITY_ATTRIBUTES,
    PHOTO_CONDITIONS,
    Attribute,
    Draw,
    IdentityAttribute,
    PhotoCondition,
)
from ryuk.evaluation.bootstrap import RESAMPLES, identity_weights
from ryuk.evaluation.openset import ScoredProbes, probe_rate
from ryuk.evaluation.results import (
    AttributeBreakdown,
    BiasAttribute,
    BiasModel,
    GroupRates,
    MatchRule,
    Rate,
    RecognitionModelId,
)

MIN_IDENTITIES: Final = 30
"""A rate needs at least this many identities behind it; fewer are too few to estimate (#10)."""

BREAKDOWNS: Final[tuple[tuple[BiasAttribute, tuple[Attribute, ...]], ...]] = (
    ("Male", ("Male",)),
    ("Young", ("Young",)),
    ("Male_and_Young", ("Male", "Young")),
    ("Eyeglasses", ("Eyeglasses",)),
    ("Wearing_Hat", ("Wearing_Hat",)),
    ("Blurry", ("Blurry",)),
)
"""Each reported attribute and the CelebA attributes its groups combine, in report order."""

_PHRASES: Final[Mapping[Attribute, tuple[str, str]]] = MappingProxyType(
    {
        "Male": ("male", "not male"),
        "Young": ("young", "not young"),
        "Eyeglasses": ("eyeglasses", "no eyeglasses"),
        "Wearing_Hat": ("hat", "no hat"),
        "Blurry": ("blurry", "not blurry"),
    }
)
"""Each attribute's phrase with it and without it, lower case."""

_MIXED: Final = -1
"""The label code of an identity with no majority; True is 1 and False 0."""
_GALLERY: Final = 0
_HELD_OUT: Final = 1


@dataclass(frozen=True)
class GroupLabels:
    """What the groups are made from: each identity's majority label for Male and Young (None
    when mixed), and each image's own label for the photo conditions."""

    identities: Mapping[int, Mapping[IdentityAttribute, bool | None]]
    photos: Mapping[str, Mapping[PhotoCondition, bool]]

    @classmethod
    def from_table(
        cls, labels: pa.Table, *, agreement: float = RULES.majority_agreement
    ) -> "GroupLabels":
        """The labels of every row of a `read_labels` table: all of a split's images, usable or
        not, as the EDA takes them."""
        flags = {attribute: _flags(labels, attribute) for attribute in ATTRIBUTES}
        identities, index, images = np.unique(
            labels.column("celeb_id").to_numpy(), return_inverse=True, return_counts=True
        )
        labelled = {
            attribute: np.bincount(index, weights=flags[attribute], minlength=identities.size)
            for attribute in IDENTITY_ATTRIBUTES
        }
        paths = [str(path) for path in labels.column("path").to_pylist()]
        return cls(
            identities={
                int(identity): {
                    attribute: majority_label(
                        int(labelled[attribute][i]), int(images[i]), agreement
                    )
                    for attribute in IDENTITY_ATTRIBUTES
                }
                for i, identity in enumerate(identities)
            },
            photos={
                path: {attribute: bool(flags[attribute][row]) for attribute in PHOTO_CONDITIONS}
                for row, path in enumerate(paths)
            },
        )


def read_group_labels(
    root: Path, draw: Draw, *, agreement: float = RULES.majority_agreement
) -> GroupLabels:
    """The group labels of every image in the draw's split, from the fetched label table."""
    return GroupLabels.from_table(
        read_labels(root, DRAW_SPLITS[draw], attributes=ATTRIBUTES), agreement=agreement
    )


@dataclass(frozen=True, eq=False)
class LabelledProbes:
    """The test draw's probes scored under one rule, each named by its image, with the labels
    their groups are made from. `mated_images` and `non_mated_images` follow `scored`'s order;
    `labels` must cover every probe's identity and image."""

    scored: ScoredProbes
    mated_images: Sequence[str]
    non_mated_images: Sequence[str]
    labels: GroupLabels

    def __post_init__(self) -> None:
        if self.scored.draw != "test":
            raise ValueError(
                f"the bias breakdown is reported on the test draw, not the {self.scored.draw}"
            )


def bias_model(
    probes: LabelledProbes,
    threshold: float,
    *,
    model: RecognitionModelId,
    rule: MatchRule,
    seed: int,
) -> BiasModel:
    """One model's per-group rates under `rule`, at the rule's frozen `threshold`."""
    return BiasModel(
        model=model,
        rule=rule,
        threshold=threshold,
        attributes=attribute_breakdowns(probes, threshold, seed=seed),
    )


def attribute_breakdowns(
    probes: LabelledProbes, threshold: float, *, seed: int, min_identities: int = MIN_IDENTITIES
) -> list[AttributeBreakdown]:
    """Every attribute of `BREAKDOWNS` split into its groups, at the single `threshold`."""
    if not math.isfinite(threshold):
        raise ValueError(f"the threshold must be finite, got {threshold}")
    if min_identities < 1:
        raise ValueError(f"min_identities must be at least 1, got {min_identities}")
    scored = probes.scored
    accepted = scored.mated_score >= threshold
    at_threshold = _AtThreshold(
        gallery=_Side.of(scored.mated_identity, probes.mated_images, probes.labels),
        held_out=_Side.of(scored.non_mated_identity, probes.non_mated_images, probes.labels),
        right=accepted & scored.mated_correct,
        wrong=accepted & ~scored.mated_correct,
        alarm=scored.non_mated_score >= threshold,
        seed=seed,
        min_identities=min_identities,
    )
    return [
        at_threshold.breakdown(position, attribute, combined)
        for position, (attribute, combined) in enumerate(BREAKDOWNS)
    ]


@dataclass(frozen=True, eq=False)
class _Side:
    """The gallery's mated probes or the held-out identities' non-mated ones: each probe's
    identity and, per attribute, its label code (1, 0, or `_MIXED`)."""

    identity: NDArray[np.int_]
    codes: Mapping[Attribute, NDArray[np.int8]]

    @classmethod
    def of(cls, identity: NDArray[np.int_], images: Sequence[str], labels: GroupLabels) -> "_Side":
        if len(images) != identity.size:
            raise ValueError(f"need one image per probe: {identity.size} probes, {len(images)}")
        people, index = np.unique(identity, return_inverse=True)
        if unlabelled := [int(i) for i in people if int(i) not in labels.identities]:
            raise ValueError(f"no labels for identity {', '.join(map(str, unlabelled[:5]))}")
        if unknown := [name for name in images if name not in labels.photos]:
            raise ValueError(f"no labels for image {', '.join(unknown[:5])}")
        codes: dict[Attribute, NDArray[np.int8]] = {}
        for attribute in IDENTITY_ATTRIBUTES:
            per_identity = [_code(labels.identities[int(i)][attribute]) for i in people]
            codes[attribute] = np.array(per_identity, dtype=np.int8)[index]
        for condition in PHOTO_CONDITIONS:
            per_image = [labels.photos[name][condition] for name in images]
            codes[condition] = np.array(per_image, dtype=np.int8)
        return cls(identity=np.asarray(identity, dtype=np.int_), codes=codes)

    def members(self, values: Mapping[Attribute, bool]) -> NDArray[np.bool_]:
        """Which probes belong to the group with these values."""
        member = np.ones(self.identity.size, dtype=np.bool_)
        for attribute, value in values.items():
            member &= self.codes[attribute] == int(value)
        return member

    def mixed(self, attributes: Sequence[Attribute]) -> int:
        """Identities left out because any of `attributes` has no majority."""
        left_out = np.zeros(self.identity.size, dtype=np.bool_)
        for attribute in attributes:
            left_out |= self.codes[attribute] == _MIXED
        return int(np.unique(self.identity[left_out]).size)

    def rates(
        self,
        member: NDArray[np.bool_],
        flags: Sequence[tuple[NDArray[np.bool_], bool]],
        *,
        stream: Sequence[int],
        min_identities: int,
    ) -> tuple[int, list[Rate | None]]:
        """The group's identity count and, per (flags, is an error rate), its rate over the
        group's probes; no rates when too few identities stand behind them."""
        people, index = np.unique(self.identity[member], return_inverse=True)
        if people.size < min_identities:
            return int(people.size), [None] * len(flags)
        weights = identity_weights(people.size, RESAMPLES, np.random.default_rng(list(stream)))
        return int(people.size), [
            probe_rate(flagged[member], index, weights, error=error) for flagged, error in flags
        ]


@dataclass(frozen=True, eq=False)
class _AtThreshold:
    """Both sides of the draw with what each probe counts for at the threshold: a mated probe
    `right` or `wrong` and accepted, a non-mated one an `alarm`."""

    gallery: _Side
    held_out: _Side
    right: NDArray[np.bool_]
    wrong: NDArray[np.bool_]
    alarm: NDArray[np.bool_]
    seed: int
    min_identities: int

    def breakdown(
        self, position: int, attribute: BiasAttribute, combined: tuple[Attribute, ...]
    ) -> AttributeBreakdown:
        """`attribute`'s groups, `position` being its place in `BREAKDOWNS`."""
        # (True, True), (True, False), ...: every group with an attribute before any without it.
        groups = [
            self._group(dict(zip(combined, values, strict=True)), (self.seed, position, group))
            for group, values in enumerate(product((True, False), repeat=len(combined)))
        ]
        return AttributeBreakdown(
            attribute=attribute,
            basis="identity" if all(a in IDENTITY_ATTRIBUTES for a in combined) else "photo",
            indicative=len(combined) > 1,
            mixed_gallery_identities=self.gallery.mixed(combined),
            mixed_held_out_identities=self.held_out.mixed(combined),
            groups=groups,
            fpir_ratio=_fpir_ratio(groups),
        )

    def _group(self, values: Mapping[Attribute, bool], stream: tuple[int, ...]) -> GroupRates:
        in_gallery, in_held_out = self.gallery.members(values), self.held_out.members(values)
        gallery_identities, (tpir, misidentification) = self.gallery.rates(
            in_gallery,
            [(self.right, False), (self.wrong, True)],
            stream=(*stream, _GALLERY),
            min_identities=self.min_identities,
        )
        held_out_identities, (fpir,) = self.held_out.rates(
            in_held_out,
            [(self.alarm, True)],
            stream=(*stream, _HELD_OUT),
            min_identities=self.min_identities,
        )
        return GroupRates(
            label=_label(values),
            values={str(attribute): value for attribute, value in values.items()},
            gallery_identities=gallery_identities,
            held_out_identities=held_out_identities,
            mated_probes=int(in_gallery.sum()),
            non_mated_probes=int(in_held_out.sum()),
            tpir=tpir,
            misidentification=misidentification,
            fpir=fpir,
        )


def _fpir_ratio(groups: Sequence[GroupRates]) -> float | None:
    """The worst group's FPIR over the best's, among groups with an FPIR (NIST FRVT style)."""
    fpirs = [group.fpir.value for group in groups if group.fpir is not None]
    if len(fpirs) < 2 or min(fpirs) == 0:
        return None
    return max(fpirs) / min(fpirs)


def _label(values: Mapping[Attribute, bool]) -> str:
    text = ", ".join(_PHRASES[a][0 if value else 1] for a, value in values.items())
    return text[0].upper() + text[1:]


def _code(label: bool | None) -> int:
    return _MIXED if label is None else int(label)


def _flags(labels: pa.Table, attribute: Attribute) -> NDArray[np.bool_]:
    column = labels.column(attribute)
    if column.null_count:
        raise ValueError(f"{column.null_count} images have no {attribute} label")
    return np.asarray(column.to_numpy(), dtype=np.bool_)
