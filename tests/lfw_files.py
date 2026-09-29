"""A tiny LFW fetch root for tests: four identities of three shots, and a blank wall."""

from pathlib import Path

import cv2
import numpy as np

from ryuk.detector import Image
from synthetic import ASTRONAUT


def _person(identity: int, shot: int) -> Image:
    """A 250x250 LFW-like image. Identities differ in pixels; shots of one differ slightly."""
    head = cv2.resize(np.asarray(ASTRONAUT[0:300, 100:350]), (200, 240))
    if identity % 2:
        head = np.ascontiguousarray(head[:, ::-1])
    head = np.roll(head, identity * 40, axis=2) if identity >= 2 else head
    canvas = np.full((250, 250, 3), 127, dtype=np.uint8)
    canvas[5:245, 25:225] = head
    return np.clip(canvas.astype(np.int16) + 4 * shot, 0, 255).astype(np.uint8)


def _write(folder: Path, name: str, number: int, image: Image) -> None:
    path = folder / name / f"{name}_{number:04d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _pairs_file(path: Path, header: str, folds: list[tuple[list[str], list[str]]]) -> None:
    lines = [header]
    for matched, mismatched in folds:
        lines += matched + mismatched
    path.write_text("\n".join(lines) + "\n")


def write_lfw(data_dir: Path) -> None:
    """Four identities of three shots each, plus Blank_Wall, whose one image has no face, laid
    out under `data_dir` as `ryuk data fetch` leaves LFW, with two View 2 folds of six pairs."""
    root = data_dir / "lfw"
    images = root / "lfw_funneled"
    names = ["Ann", "Bob", "Cy", "Dee"]
    for identity, name in enumerate(names):
        for shot in (1, 2, 3):
            _write(images, name, shot, _person(identity, shot))
    _write(images, "Blank_Wall", 1, np.full((250, 250, 3), 127, dtype=np.uint8))

    fold = (
        ["Ann\t1\t2", "Bob\t1\t3", "Blank_Wall\t1\t1"],
        ["Ann\t1\tBob\t2", "Cy\t1\tDee\t1", "Ann\t2\tBlank_Wall\t1"],
    )
    other = (
        ["Cy\t1\t2", "Dee\t2\t3", "Ann\t2\t3"],
        ["Bob\t1\tCy\t3", "Dee\t1\tAnn\t3", "Bob\t2\tDee\t2"],
    )
    _pairs_file(root / "pairs.txt", "2\t3", [fold, other])
    _pairs_file(root / "pairsDevTrain.txt", "3", [other])
    _pairs_file(root / "pairsDevTest.txt", "3", [fold])
