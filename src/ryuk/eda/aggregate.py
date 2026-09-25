"""The EDA's arithmetic: every number in the dataset summary, from plain in-memory inputs.

Nothing here reads files or runs the detector, so each rule is unit tested on small inputs with
answers worked out by hand. Floats are rounded as they go into the summary (pixels and degrees
to two places, shares to six), so a rerun gives small diffs.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from typing import Final, NamedTuple

import numpy as np

from ryuk.datasets.lfw import LfwImage, PairsFile
from ryuk.eda.pose import Pose, head_pose
from ryuk.eda.scan import ImageScan
from ryuk.eda.summary import (
    ATTRIBUTES,
    DRAW_SPLITS,
    IDENTITY_ATTRIBUTES,
    Attribute,
    AttributePrevalence,
    CelebaDraw,
    DetectionStats,
    Distribution,
    Draw,
    GroupStats,
    HeadPose,
    Histogram,
    ImagesPerIdentity,
    ImagesPerIdentityRow,
    MinUsableFaceSize,
    PairList,
    Quantiles,
    Rules,
)

FACE_SIZE_EDGES: Final = tuple(float(edge) for edge in range(0, 301, 2))
"""Face short side, 0-300 px in 2 px bins. CelebA's aligned crops are 178 x 218."""
POSE_EDGES: Final = tuple(float(edge) for edge in range(-90, 91, 5))
"""Head pose angles, -90 to 90 degrees in 5 degree bins."""
SHARE_EDGES: Final = tuple(tenth / 10 for tenth in range(11))
"""A share in tenths. Built as i / 10, not by repeated addition, so a share of exactly 3 / 10
falls in [0.3, 0.4) rather than below an edge of 0.30000000000000004."""

PIXEL_DIGITS: Final = 2
DEGREE_DIGITS: Final = 2
# The minimum face size's shares keep more places, so a share just under `keep` never rounds up
# to it: with about 40,000 detections one face moves the share by 0.000025.
_KEPT_DIGITS: Final = 6


def histogram(values: Sequence[float], edges: Sequence[float]) -> Histogram:
    """Counts of `values` in the bins between `edges`; values outside them are not counted."""
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=np.asarray(edges))
    return Histogram(edges=list(edges), counts=[int(count) for count in counts])


def distribution(values: Sequence[float], edges: Sequence[float], *, digits: int) -> Distribution:
    """Mean, quantiles (linear interpolation) and histogram, with floats rounded to `digits`."""
    if not values:
        return Distribution(n=0, mean=None, quantiles=None, histogram=histogram(values, edges))
    array = np.asarray(values, dtype=np.float64)
    p1, p5, p25, p50, p75, p95, p99 = (
        round(float(q), digits) for q in np.percentile(array, [1, 5, 25, 50, 75, 95, 99])
    )
    return Distribution(
        n=len(values),
        mean=round(float(array.mean()), digits),
        quantiles=Quantiles(p1=p1, p5=p5, p25=p25, p50=p50, p75=p75, p95=p95, p99=p99),
        histogram=histogram(values, edges),
    )


def images_per_identity(counts: Iterable[int]) -> ImagesPerIdentity:
    """The frequency table of images per identity, from each identity's image count."""
    counts = list(counts)
    if any(count < 1 for count in counts):
        raise ValueError("every identity needs at least one image")
    frequency = Counter(counts)
    return ImagesPerIdentity(
        identities=len(counts),
        images=sum(counts),
        median=float(np.median(counts)) if counts else None,
        max=max(counts, default=None),
        table=[
            ImagesPerIdentityRow(images=images, identities=identities)
            for images, identities in sorted(frequency.items())
        ],
    )


def min_usable_face_size(
    short_sides: Sequence[float], *, step: int = 10, keep: float = 0.99
) -> MinUsableFaceSize:
    """#17's rule: the largest multiple of `step` px that at least `keep` of the faces reach.

    `short_sides` are the box short sides of CelebA's detections, one per detected image. A face
    exactly at a value reaches it, as `usable_faces` counts it. `step` must be positive and `keep`
    a share in (0, 1]; any other value would never stop the search.
    """
    if step <= 0:
        raise ValueError(f"step must be a positive number of pixels, got {step}")
    if not 0 < keep <= 1:
        raise ValueError(f"keep must be a share in (0, 1], got {keep}")
    if not short_sides:
        raise ValueError("no detections to measure a minimum face size from")
    ordered = np.sort(np.asarray(short_sides, dtype=np.float64))

    def share(value: int) -> float:
        reaching = len(ordered) - int(np.searchsorted(ordered, value, side="left"))
        return reaching / len(ordered)

    if share(step) < keep:
        raise ValueError(f"fewer than {keep:.2%} of detections reach {step} px")
    value = step
    while share(value + step) >= keep:
        value += step
    return MinUsableFaceSize(
        value=value,
        step=step,
        keep=keep,
        detections=len(ordered),
        kept=round(share(value), _KEPT_DIGITS),
        kept_at_next_step=round(share(value + step), _KEPT_DIGITS),
    )


def majority_label(labelled: int, images: int, agreement: float) -> bool | None:
    """#10's majority label: True if at least `agreement` of the identity's `images` carry the
    label, False if at least `agreement` do not, None (mixed) otherwise."""
    if images < 1:
        raise ValueError("an identity with no images has no majority label")
    if labelled / images >= agreement:
        return True
    if (images - labelled) / images >= agreement:
        return False
    return None


class LabelledImage(NamedTuple):
    """One CelebA image as the draw statistics see it: whose it is, its labels, and whether
    YuNet found a face and a usable one."""

    identity: int
    labels: Mapping[Attribute, bool]
    detected: bool
    usable: bool


def attribute_prevalence(
    attribute: Attribute, images: Sequence[LabelledImage], *, agreement: float
) -> AttributePrevalence:
    """How common `attribute` is, per image and per identity (#17)."""
    labelled, totals = _per_identity(images, lambda image: image.labels[attribute])
    majorities = [majority_label(labelled[i], totals[i], agreement) for i in totals]
    return AttributePrevalence(
        attribute=attribute,
        images_with=sum(labelled.values()),
        identities_with_any=sum(1 for count in labelled.values() if count),
        identity_share=histogram([labelled[i] / totals[i] for i in totals], SHARE_EDGES),
        majority_with=majorities.count(True),
        majority_without=majorities.count(False),
        mixed=majorities.count(None),
    )


def group_stats(
    attribute: Attribute,
    images: Sequence[LabelledImage],
    *,
    agreement: float,
    min_gallery_images: int,
) -> tuple[GroupStats, GroupStats]:
    """Detection and exclusion in the group with `attribute`, then the group without it.

    An identity attribute groups whole identities by their majority label, leaving mixed ones
    out; a photo condition groups images by their own label (#10).
    """
    value: Callable[[LabelledImage], bool | None]
    if attribute in IDENTITY_ATTRIBUTES:
        labelled, totals = _per_identity(images, lambda image: image.labels[attribute])
        majority = {i: majority_label(labelled[i], totals[i], agreement) for i in totals}
        value, minimum = (lambda image: majority[image.identity]), min_gallery_images
    else:
        value, minimum = (lambda image: image.labels[attribute]), None
    with_it = [image for image in images if value(image) is True]
    without = [image for image in images if value(image) is False]
    return _group(attribute, True, with_it, minimum), _group(attribute, False, without, minimum)


def _group(
    attribute: Attribute,
    value: bool,
    images: Sequence[LabelledImage],
    min_gallery_images: int | None,
) -> GroupStats:
    return GroupStats(
        attribute=attribute,
        value=value,
        identities=len({image.identity for image in images}),
        images=len(images),
        detected=sum(image.detected for image in images),
        usable=sum(image.usable for image in images),
        eligible_identities=None
        if min_gallery_images is None
        else eligible_identities(images, min_gallery_images),
    )


def gallery_candidates(images: Iterable[LabelledImage], min_gallery_images: int) -> int:
    """Identities with at least `min_gallery_images` images, before exclusion (#9)."""
    counts = Counter(image.identity for image in images)
    return sum(1 for count in counts.values() if count >= min_gallery_images)


def eligible_identities(images: Iterable[LabelledImage], min_gallery_images: int) -> int:
    """Identities with at least `min_gallery_images` usable images: #9's rule after #17's
    exclusion of images with no usable face."""
    return gallery_candidates((image for image in images if image.usable), min_gallery_images)


def celeba_draw(
    draw: Draw, images: Sequence[LabelledImage], detection: DetectionStats, rules: Rules
) -> CelebaDraw:
    """A draw's composition, attribute prevalence and bias groups, in `ATTRIBUTES` order."""
    agreement, minimum = rules.majority_agreement, rules.min_gallery_images
    return CelebaDraw(
        draw=draw,
        split=DRAW_SPLITS[draw],
        images_per_identity=images_per_identity(
            Counter(image.identity for image in images).values()
        ),
        gallery_candidates=gallery_candidates(images, minimum),
        eligible_identities=eligible_identities(images, minimum),
        attributes=[
            attribute_prevalence(attribute, images, agreement=agreement) for attribute in ATTRIBUTES
        ],
        detection=detection,
        groups=[
            group
            for attribute in ATTRIBUTES
            for group in group_stats(
                attribute, images, agreement=agreement, min_gallery_images=minimum
            )
        ],
    )


def pair_list(pairs: PairsFile, excluded: Set[LfwImage]) -> PairList:
    """One LFW pairs list's composition; `excluded` are the images with no usable face."""
    images = {image for pair in pairs.pairs for image in (pair.first, pair.second)}
    matched = sum(pair.matched for pair in pairs.pairs)
    return PairList(
        name=pairs.name,
        view=pairs.view,
        folds=pairs.folds,
        matched=matched,
        mismatched=len(pairs.pairs) - matched,
        identities=len({image.identity for image in images}),
        images=len(images),
        pairs_with_excluded_image=sum(
            pair.first in excluded or pair.second in excluded for pair in pairs.pairs
        ),
    )


def detection_stats(scans: Sequence[ImageScan], *, min_face_size: int) -> DetectionStats:
    """How YuNet did on a set of images, described by each image's centre-most detection.

    The centre-most detection is taken usable or not, so the face sizes show what the minimum
    cuts off. A detection whose landmarks fit no pose is left out of the pose distributions only.
    """
    faces = [(scan, face) for scan in scans if (face := scan.centre_most) is not None]
    poses = [p for scan, face in faces if (p := head_pose(face.landmarks, scan.shape))]
    return DetectionStats(
        images=len(scans),
        detected=len(faces),
        multiple_faces=sum(len(scan.detections) > 1 for scan in scans),
        usable=sum(scan.usable(min_face_size) for scan in scans),
        face_short_side=distribution(
            [face.box.short_side for _, face in faces], FACE_SIZE_EDGES, digits=PIXEL_DIGITS
        ),
        head_pose=_head_pose(poses),
    )


def _head_pose(poses: Sequence[Pose]) -> HeadPose:
    def angles(angle: Callable[[Pose], float]) -> Distribution:
        return distribution([angle(p) for p in poses], POSE_EDGES, digits=DEGREE_DIGITS)

    return HeadPose(
        yaw=angles(lambda p: p.yaw), pitch=angles(lambda p: p.pitch), roll=angles(lambda p: p.roll)
    )


def _per_identity(
    images: Iterable[LabelledImage], label: Callable[[LabelledImage], bool]
) -> tuple[dict[int, int], dict[int, int]]:
    """Per identity, in first-seen order: how many images carry `label`, and how many it has."""
    labelled: dict[int, int] = defaultdict(int)
    totals: dict[int, int] = defaultdict(int)
    for image in images:
        labelled[image.identity] += label(image)
        totals[image.identity] += 1
    return labelled, totals
