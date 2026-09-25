"""The CelebA watchlist rehearsal (#27): draw, freeze each model's threshold, score the test draw.

For each draw, every image of its CelebA split is scanned with YuNet and kept only if it has a
usable face (#17); the gallery and held-out identities are drawn from what is left. Each model
then embeds the drawn images through the benchmark pipeline, reusing the scan's detection and the
crop LFW View 1 chose for the network. The validation draw is scored and the threshold frozen on
it before the test draw is embedded, and the test draw is scored once, at that threshold.
"""

import logging
import statistics
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ryuk.datasets.celeba import CelebaImage, iter_images, read_labels
from ryuk.detector import Detection, benchmark_face
from ryuk.eda.scan import Scanner
from ryuk.eda.summary import DRAW_SPLITS, Draw
from ryuk.evaluation.bootstrap import BOOTSTRAP_SEED, CONFIDENCE, RESAMPLES
from ryuk.evaluation.draws import (
    DRAW_SEED,
    GALLERY_SIZE,
    MIN_GALLERY_IMAGES,
    OpenSetDraw,
    make_draw,
)
from ryuk.evaluation.embeddings import EmbeddingCache, image_key
from ryuk.evaluation.openset import (
    Gallery,
    Probes,
    ScoredProbes,
    draw_result,
    freeze,
    score_probes,
)
from ryuk.evaluation.results import (
    Bootstrap,
    DetectorId,
    DrawSelection,
    Identification,
    OpenSetModel,
    Provenance,
)
from ryuk.evaluation.verification import Pipeline, model_id
from ryuk.recognition import Embedding, Network, RecognitionModel
from ryuk.recognition.faces import Crop

logger = logging.getLogger(__name__)

_TIMED_FACES: Final = 200
_WARM_UP: Final = 10


@dataclass(frozen=True)
class PreparedDraw:
    """A draw with the label rows of the images it uses and each one's benchmark face."""

    selection: OpenSetDraw
    record: DrawSelection
    labels: pa.Table
    faces: Mapping[str, Detection]


@dataclass(frozen=True)
class CelebaEvaluation:
    root: Path
    """The fetch root, `RYUK_DATA_DIR`."""
    pipeline: Pipeline
    cache: EmbeddingCache
    scanner: Scanner
    draw_seed: int = DRAW_SEED
    bootstrap_seed: int = BOOTSTRAP_SEED
    gallery_size: int = GALLERY_SIZE

    def run(
        self,
        models: Mapping[Network, Callable[[], RecognitionModel]],
        crops: Mapping[Network, Crop],
        provenance: Provenance,
    ) -> Identification:
        """Draw both draws, then rehearse each model; `crops` is LFW View 1's choice per network."""
        validation, test = (self.prepare(draw) for draw in DRAW_SPLITS)
        results = []
        for network, load in models.items():
            model = load()
            pipeline = self.pipeline.with_crop(crops[network])
            logger.info("rehearsing %s (%s) on CelebA", network, model.key.id)
            validation_scored, _ = self._score(model, pipeline, validation)
            threshold = freeze(validation_scored)
            test_scored, gallery = self._score(model, pipeline, test)
            results.append(
                OpenSetModel(
                    model=model_id(model),
                    crop=pipeline.crop,
                    rule="best-photo",
                    threshold=threshold.value,
                    target_fpir=threshold.target_fpir,
                    validation=draw_result(validation_scored, threshold, seed=self.bootstrap_seed),
                    test=draw_result(test_scored, threshold, seed=self.bootstrap_seed),
                    ms_per_face=self._ms_per_face(model, pipeline, test, gallery),
                )
            )
        return Identification(
            provenance=provenance,
            detector=DetectorId(
                weights_sha256=self.pipeline.detector_sha256,
                min_face_size=self.pipeline.min_face_size,
            ),
            bootstrap=Bootstrap(
                resamples=RESAMPLES, seed=self.bootstrap_seed, confidence=CONFIDENCE
            ),
            draws=[validation.record, test.record],
            models=results,
        )

    def prepare(self, draw: Draw) -> PreparedDraw:
        """Scan the draw's split, exclude images with no usable face, and make the draw."""
        split = DRAW_SPLITS[draw]
        labels = read_labels(self.root, split, attributes=[])
        paths = [str(path) for path in labels.column("path").to_pylist()]
        identities: list[int] = labels.column("celeb_id").to_numpy().tolist()
        rows: list[int] = []

        def images() -> Iterator[CelebaImage]:
            for image in iter_images(self.root, labels):
                rows.append(image.row)
                yield image

        scans = self.scanner.scan(
            images(), CelebaImage.decode, total=labels.num_rows, name=f"CelebA {split}"
        )
        usable: dict[int, list[str]] = {identity: [] for identity in identities}
        faces: dict[str, Detection] = {}
        for row, scan in zip(rows, scans, strict=True):
            face = benchmark_face(scan.detections, scan.shape, self.pipeline.min_face_size)
            if face is not None:
                usable[identities[row]].append(paths[row])
                faces[paths[row]] = face

        selection = make_draw(draw, usable, self.draw_seed, gallery_size=self.gallery_size)
        chosen = selection.images()
        mated = sum(len(g.probes) for g in selection.gallery)
        record = DrawSelection(
            draw=draw,
            split=split,
            seed=self.draw_seed,
            images=labels.num_rows,
            usable_images=len(faces),
            gallery_candidates=sum(len(names) >= MIN_GALLERY_IMAGES for names in usable.values()),
            gallery=[g.identity for g in selection.gallery],
            held_out=[h.identity for h in selection.held_out],
            enrolled_photos=sum(len(g.enrolled) for g in selection.gallery),
            mated_probes=mated,
            non_mated_probes=sum(len(h.probes) for h in selection.held_out),
            selection_sha256=selection.selection_sha256,
        )
        logger.info(
            "%s draw: %d gallery and %d held-out identities, %d images",
            draw,
            len(record.gallery),
            len(record.held_out),
            len(chosen),
        )
        return PreparedDraw(
            selection=selection,
            record=record,
            labels=labels.filter(pc.is_in(labels.column("path"), pa.array(chosen))),
            faces={path: faces[path] for path in chosen},
        )

    def _embed(
        self, model: RecognitionModel, pipeline: Pipeline, draw: PreparedDraw
    ) -> dict[str, Embedding]:
        """Each drawn image's embedding by file name, from the cache where it can be."""
        cached = self.cache.load(model.key, pipeline.id, model.dimension)
        computed: dict[str, Embedding | None] = {}
        embeddings: dict[str, Embedding] = {}
        for image in iter_images(self.root, draw.labels):
            key = image_key(image.png)
            if key in cached:
                embedding = cached[key]
            elif key in computed:
                embedding = computed[key]
            else:
                face = pipeline.cut(image.decode(), draw.faces[image.path], model.input_size)
                embedding = computed[key] = model.embed(face)
                if len(computed) % 2000 == 0:
                    logger.info("%s: %d CelebA images embedded", model.key.network, len(computed))
            if embedding is None:
                raise ValueError(f"{image.path} is cached as having no usable face, yet has one")
            embeddings[image.path] = embedding
        if computed:
            self.cache.save(model.key, pipeline.id, model.dimension, computed)
        return embeddings

    def _score(
        self, model: RecognitionModel, pipeline: Pipeline, draw: PreparedDraw
    ) -> tuple[ScoredProbes, Gallery]:
        embeddings = self._embed(model, pipeline, draw)
        selection = draw.selection
        gallery = Gallery.enrol(
            {g.identity: [embeddings[p] for p in g.enrolled] for g in selection.gallery}
        )

        def probes(identities: list[tuple[int, str]]) -> Probes:
            return Probes(
                np.array([identity for identity, _ in identities], dtype=np.int_),
                np.stack([embeddings[path] for _, path in identities]),
            )

        mated = probes([(g.identity, p) for g in selection.gallery for p in g.probes])
        non_mated = probes([(h.identity, p) for h in selection.held_out for p in h.probes])
        return score_probes(selection.draw, gallery, mated, non_mated), gallery

    def _ms_per_face(
        self, model: RecognitionModel, pipeline: Pipeline, draw: PreparedDraw, gallery: Gallery
    ) -> float:
        """Warm median time from a probe's pixels to its top candidate, over up to 200 probes.

        Detection runs afresh, as it does live; decoding is not timed, as frames arrive decoded.
        """
        wanted = [p for g in draw.selection.gallery for p in g.probes][: _WARM_UP + _TIMED_FACES]
        labels = draw.labels.filter(pc.is_in(draw.labels.column("path"), pa.array(wanted)))
        images = [image.decode() for image in iter_images(self.root, labels)]

        def search(index: int) -> float:
            start = time.perf_counter()
            face = pipeline.face(images[index], model.input_size)
            if face is None:
                raise ValueError("a drawn probe lost its usable face when detected again")
            gallery.top_candidates(model.embed(face)[np.newaxis, :])
            return (time.perf_counter() - start) * 1000

        for index in range(min(_WARM_UP, len(images))):
            search(index)
        return statistics.median(search(index) for index in range(len(images))[-_TIMED_FACES:])
