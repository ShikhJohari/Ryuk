"""Small, hand-made but valid `EdaSummary` values for the tests that only read a summary.

The numbers are consistent with each other where the figures rely on it (a draw's group counts
never exceed the draw's), but otherwise arbitrary. Tests vary them with `model_copy(update=...)`.
"""

import datetime
from collections.abc import Sequence

from ryuk.eda.summary import (
    AttributePrevalence,
    CelebaDraw,
    DetectionStats,
    Distribution,
    Draw,
    EdaSummary,
    GroupStats,
    HeadPose,
    Histogram,
    ImagesPerIdentity,
    ImagesPerIdentityRow,
    Lfw,
    MinUsableFaceSize,
    PairList,
    Provenance,
    Quantiles,
    Rules,
)

SIZE_EDGES = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
POSE_EDGES = [-30.0, -15.0, 0.0, 15.0, 30.0]
SHARE_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0]


def distribution(edges: Sequence[float], counts: Sequence[int], median: float) -> Distribution:
    """A distribution whose `n` is its binned counts, with rough but ordered quantiles."""
    low, high = edges[0], edges[-1]
    return Distribution(
        n=sum(counts),
        mean=median,
        quantiles=Quantiles(
            p1=low,
            p5=low,
            p25=(low + median) / 2,
            p50=median,
            p75=(median + high) / 2,
            p95=high,
            p99=high,
        ),
        histogram=Histogram(edges=list(edges), counts=list(counts)),
    )


def empty_distribution(edges: Sequence[float]) -> Distribution:
    """A distribution over nothing, as for a dataset with no detections."""
    return Distribution(
        n=0,
        mean=None,
        quantiles=None,
        histogram=Histogram(edges=list(edges), counts=[0] * (len(edges) - 1)),
    )


def detection(images: int, detected: int, usable: int, sizes: Sequence[int]) -> DetectionStats:
    """Detection stats with face sizes binned on `SIZE_EDGES` and a pose from `detected`."""
    if sum(sizes) != detected:
        raise ValueError("the face sizes must count every detected image")
    third = detected // 3
    pose = [third // 2, third, detected - 2 * third, third - third // 2]
    return DetectionStats(
        images=images,
        detected=detected,
        multiple_faces=detected // 10,
        usable=usable,
        face_short_side=distribution(SIZE_EDGES, sizes, 70.0),
        head_pose=HeadPose(
            yaw=distribution(POSE_EDGES, pose, 1.0),
            pitch=distribution(POSE_EDGES, pose, -4.0),
            roll=distribution(POSE_EDGES, pose, -0.2),
        ),
    )


def images_per_identity(table: dict[int, int]) -> ImagesPerIdentity:
    """A frequency table from {images: identities}."""
    rows = [ImagesPerIdentityRow(images=k, identities=n) for k, n in sorted(table.items())]
    identities = sum(table.values())
    images = sum(k * n for k, n in table.items())
    counts = sorted(k for k, n in table.items() for _ in range(n))
    return ImagesPerIdentity(
        identities=identities,
        images=images,
        median=float(counts[len(counts) // 2]) if counts else None,
        max=max(table) if table else None,
        table=rows,
    )


def prevalence(
    attribute: str, images_with: int, any_: int, majority: tuple[int, int, int]
) -> AttributePrevalence:
    """`majority` is (with, without, mixed) identities."""
    with_, without, mixed = majority
    return AttributePrevalence.model_validate(
        {
            "attribute": attribute,
            "images_with": images_with,
            "identities_with_any": any_,
            "identity_share": {"edges": SHARE_EDGES, "counts": [without, mixed, 0, with_]},
            "majority_with": with_,
            "majority_without": without,
            "mixed": mixed,
        }
    )


def group(
    attribute: str, value: bool, counts: tuple[int, int, int, int], eligible: int | None
) -> GroupStats:
    """`counts` is (identities, images, detected, usable)."""
    identities, images, detected, usable = counts
    return GroupStats.model_validate(
        {
            "attribute": attribute,
            "value": value,
            "identities": identities,
            "images": images,
            "detected": detected,
            "usable": usable,
            "eligible_identities": eligible,
        }
    )


def celeba_draw(draw: Draw, *, shift: int = 0) -> CelebaDraw:
    """A draw of 10 identities and about 180 images; `shift` sets its numbers off another draw's."""
    table = images_per_identity({5: 2, 18: 2, 20: 3 + shift, 25: 3 - shift})
    images = table.images
    return CelebaDraw(
        draw=draw,
        split="valid" if draw == "validation" else "test",
        images_per_identity=table,
        gallery_candidates=6,
        eligible_identities=5,
        attributes=[
            prevalence("Male", 80, 5, (4, 5, 1)),
            prevalence("Young", 140, 8, (7, 2, 1)),
            prevalence("Eyeglasses", 12 + shift, 4, (0, 9, 1)),
            prevalence("Wearing_Hat", 8, 3, (0, 10, 0)),
            prevalence("Blurry", 9, 5, (0, 10, 0)),
        ],
        detection=detection(images, images - 2, images - 4, [1, 1, 20, 80, 70, images - 174]),
        groups=[
            group("Male", True, (4, 80, 79, 78), 2),
            group("Male", False, (5, 100, 99, 97), 3),
            group("Young", True, (7, 140, 139, 137), 4),
            group("Young", False, (2, 40, 39, 39), 1),
            group("Eyeglasses", True, (4, 12, 11, 10), None),
            group("Eyeglasses", False, (10, images - 12, images - 13, images - 14), None),
            group("Wearing_Hat", True, (3, 8, 7, 7), None),
            group("Wearing_Hat", False, (10, images - 8, images - 9, images - 11), None),
            group("Blurry", True, (5, 9, 8, 6), None),
            group("Blurry", False, (10, images - 9, images - 10, images - 10), None),
        ],
    )


def eda_summary() -> EdaSummary:
    """A complete summary: LFW with its three pairs files, and both CelebA draws."""
    lfw_table = images_per_identity({1: 30, 2: 8, 3: 4, 7: 2, 40: 1})
    return EdaSummary(
        schema_version=1,
        provenance=Provenance(
            generated=datetime.date(2026, 9, 25),
            git_commit="0123456789abcdef0123456789abcdef01234567",
            git_dirty=False,
            celeba_revision="rev",
            celeba_labels_digest="sha256 00",
            lfw_archive="sha256 00",
            yunet_sha256="00",
            score_threshold=0.9,
            nms_threshold=0.3,
        ),
        rules=Rules(min_gallery_images=20, majority_agreement=0.8),
        min_usable_face_size=MinUsableFaceSize(
            value=40, step=5, keep=0.99, detections=372, kept=0.9946, kept_at_next_step=0.98
        ),
        lfw=Lfw(
            images_per_identity=lfw_table,
            pairs=[
                PairList(
                    name="pairs",
                    view=2,
                    folds=10,
                    matched=30,
                    mismatched=30,
                    identities=25,
                    images=50,
                    pairs_with_excluded_image=1,
                ),
                PairList(
                    name="pairsDevTrain",
                    view=1,
                    folds=1,
                    matched=11,
                    mismatched=11,
                    identities=20,
                    images=30,
                    pairs_with_excluded_image=0,
                ),
                PairList(
                    name="pairsDevTest",
                    view=1,
                    folds=1,
                    matched=5,
                    mismatched=5,
                    identities=10,
                    images=15,
                    pairs_with_excluded_image=0,
                ),
            ],
            detection=detection(lfw_table.images, 110, 109, [0, 1, 4, 30, 55, 20]),
        ),
        celeba=[celeba_draw("validation"), celeba_draw("test", shift=1)],
    )
