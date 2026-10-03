"""The CelebA watchlist rehearsal (#27): draw, freeze each model's threshold, score the test draw.

For each draw, every image of its CelebA split is scanned with YuNet and kept only if it has a
usable face (#17); the gallery and held-out identities are drawn from what is left. A draw rebuilt
where one was committed must be that same draw, checked by its `selection_sha256`. Each model
then embeds the drawn images through the benchmark pipeline, reusing the scan's detection and the
crop LFW View 1 chose for the network. The validation draw is scored and the threshold frozen on
it before the test draw is embedded, and the test draw is scored once, at that threshold.
"""

import logging
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ryuk.datasets.celeba import CelebaImage, iter_images, read_labels
from ryuk.detector import Detection, benchmark_face
from ryuk.eda.build import RULES
from ryuk.eda.scan import Scanner
from ryuk.eda.summary import DRAW_SPLITS, Draw
from ryuk.evaluation.bias import MIN_IDENTITIES, LabelledProbes, bias_model, read_group_labels
from ryuk.evaluation.bootstrap import BOOTSTRAP_SEED, CONFIDENCE, RESAMPLES
from ryuk.evaluation.draws import (
    DRAW_SEED,
    GALLERY_SIZE,
    MIN_GALLERY_IMAGES,
    OpenSetDraw,
    check_selection,
    make_draw,
)
from ryuk.evaluation.embeddings import EmbeddingCache, image_key
from ryuk.evaluation.learning import FOLDS, compare, score_rule
from ryuk.evaluation.openset import (
    TARGET_FPIR,
    EmbeddedDraw,
    Gallery,
    Probes,
    ScoredProbes,
    draw_result,
    freeze,
    score_probes,
)
from ryuk.evaluation.results import (
    Bias,
    BiasModel,
    Bootstrap,
    DetectorId,
    DrawDigest,
    DrawSelection,
    Identification,
    Learning,
    LearningModel,
    Live,
    LiveModel,
    ModelThreshold,
    OpenSetModel,
    Provenance,
)
from ryuk.evaluation.same_person import same_person
from ryuk.evaluation.scores import scores_path, scores_table, write_scores
from ryuk.evaluation.small_galleries import GALLERY_SIZES, PARTITION_SEED, small_galleries
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
    small_gallery_sizes: Sequence[int] = GALLERY_SIZES
    """The smaller galleries `live` splits the test draw's gallery into; each divides it."""
    selections: Mapping[Draw, str] = field(default_factory=dict)
    """The committed `selection_sha256` of each draw to rebuild exactly; a draw not named here
    is made afresh."""

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
            bootstrap=self._bootstrap(),
            draws=[validation.record, test.record],
            models=results,
        )

    def learn(
        self,
        models: Mapping[Network, Callable[[], RecognitionModel]],
        identification: Identification,
        provenance: Provenance,
        scores: Path,
    ) -> Learning:
        """#10's comparison of every method on identification's draws, model by model, with each
        draw's per-probe scores written under `scores` (the cache's `scores` folder).

        `selections` must name identification's draws, so the draws rebuilt are those.
        """
        validation, test = (self.prepare(draw) for draw in DRAW_SPLITS)
        compared: list[LearningModel] = []
        for rehearsed in identification.models:
            model, pipeline = self._rehearsed(models, rehearsed)
            logger.info("comparing methods on %s (%s)", rehearsed.model.network, model.key.id)
            embedded = {
                d.record.draw: self.embed_draw(model, pipeline, d) for d in (validation, test)
            }
            comparison = compare(
                embedded["validation"], embedded["test"], rehearsed.model, seed=self.bootstrap_seed
            )
            for prepared in (validation, test):
                draw = prepared.record.draw
                probes = embedded[draw]
                write_scores(
                    scores_path(scores, model.key, prepared.record.selection_sha256),
                    scores_table(
                        [*probes.mated.images, *probes.non_mated.images],
                        comparison.scores[draw],
                        comparison.runner_up[draw],
                    ),
                )
            compared.append(comparison.result)
        return Learning(
            provenance=provenance,
            bootstrap=self._bootstrap(),
            draws=[_digest(validation), _digest(test)],
            target_fpir=TARGET_FPIR,
            folds=FOLDS,
            models=compared,
        )

    def bias(
        self,
        models: Mapping[Network, Callable[[], RecognitionModel]],
        identification: Identification,
        learning: Learning,
        provenance: Provenance,
    ) -> Bias:
        """Per-group rates on the test draw at each model's single frozen threshold, under
        best-photo and under its live rule where that differs (#10)."""
        test = self.prepare("test")
        labels = read_group_labels(self.root, "test")
        compared = {m.model: m for m in learning.models}
        breakdowns: list[BiasModel] = []
        for rehearsed in identification.models:
            model, pipeline = self._rehearsed(models, rehearsed)
            logger.info("breaking down %s (%s) by group", rehearsed.model.network, model.key.id)
            embedded = self.embed_draw(model, pipeline, test)
            result = compared[rehearsed.model]
            for rule in result.bias_rules:
                scored = score_rule(rule, embedded, result.learned_rule)
                probes = LabelledProbes(
                    scored, embedded.mated.images, embedded.non_mated.images, labels
                )
                threshold = result.method(rule).threshold
                breakdowns.append(
                    bias_model(
                        probes,
                        threshold,
                        model=rehearsed.model,
                        rule=rule,
                        seed=self.bootstrap_seed,
                    )
                )
        return Bias(
            provenance=provenance,
            bootstrap=self._bootstrap(),
            draw=_digest(test),
            min_identities=MIN_IDENTITIES,
            agreement=RULES.majority_agreement,
            models=breakdowns,
        )

    def live(
        self,
        models: Mapping[Network, Callable[[], RecognitionModel]],
        identification: Identification,
        thresholds: Sequence[ModelThreshold],
        provenance: Provenance,
    ) -> Live:
        """Each model's same-person threshold, frozen on the validation draw's impostor pairs,
        and its live rule at `thresholds`' frozen threshold on smaller galleries of the test
        draw (#49). `thresholds` are the ones the service reads, one per identification model."""
        validation, test = (self.prepare(draw) for draw in DRAW_SPLITS)
        measured: list[LiveModel] = []
        for rehearsed, frozen in zip(identification.models, thresholds, strict=True):
            if frozen.model != rehearsed.model:
                raise ValueError("the thresholds must be identification's models, in order")
            model, pipeline = self._rehearsed(models, rehearsed)
            logger.info(
                "measuring %s (%s) away from the rehearsal", frozen.model.network, model.key.id
            )
            embedded = {
                d.record.draw: self.embed_draw(model, pipeline, d) for d in (validation, test)
            }
            measured.append(
                LiveModel(
                    model=frozen.model,
                    rule=frozen.rule,
                    threshold=frozen.threshold,
                    same_person=same_person(
                        embedded["validation"],
                        embedded["test"],
                        # A learned rule's threshold is a probability, not a cosine.
                        live_threshold=None if frozen.rule == "learned" else frozen.threshold,
                        seed=self.bootstrap_seed,
                    ),
                    small_galleries=small_galleries(
                        embedded["test"],
                        frozen,
                        seed=self.bootstrap_seed,
                        sizes=self.small_gallery_sizes,
                    ),
                )
            )
        return Live(
            provenance=provenance,
            bootstrap=self._bootstrap(),
            draws=[_digest(validation), _digest(test)],
            partition_seed=PARTITION_SEED,
            models=measured,
        )

    def _rehearsed(
        self, models: Mapping[Network, Callable[[], RecognitionModel]], rehearsed: OpenSetModel
    ) -> tuple[RecognitionModel, Pipeline]:
        """The model identification rehearsed, loaded, with the crop it was rehearsed with."""
        model = models[rehearsed.model.network]()
        if model_id(model) != rehearsed.model:
            raise ValueError(
                f"{rehearsed.model.network} loads as {model.key.id}, not the model identification "
                "rehearsed"
            )
        return model, self.pipeline.with_crop(rehearsed.crop)

    def _bootstrap(self) -> Bootstrap:
        return Bootstrap(resamples=RESAMPLES, seed=self.bootstrap_seed, confidence=CONFIDENCE)

    def prepare(self, draw: Draw) -> PreparedDraw:
        """Scan the draw's split, exclude images with no usable face, and make the draw.

        Raises DrawMismatchError, before anything is embedded, if `selections` names this draw
        and the draw made differs from it.
        """
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
        if (expected := self.selections.get(draw)) is not None:
            check_selection(selection, expected)
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

    def embed(
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

    def embed_draw(
        self, model: RecognitionModel, pipeline: Pipeline, draw: PreparedDraw
    ) -> EmbeddedDraw:
        """The draw's gallery and probes under `model`, each probe with its image."""
        embeddings = self.embed(model, pipeline, draw)
        selection = draw.selection

        def probes(identities: list[tuple[int, str]]) -> Probes:
            return Probes(
                np.array([identity for identity, _ in identities], dtype=np.int_),
                np.stack([embeddings[path] for _, path in identities]),
                tuple(path for _, path in identities),
            )

        return EmbeddedDraw(
            draw=selection.draw,
            enrolled={g.identity: [embeddings[p] for p in g.enrolled] for g in selection.gallery},
            mated=probes([(g.identity, p) for g in selection.gallery for p in g.probes]),
            non_mated=probes([(h.identity, p) for h in selection.held_out for p in h.probes]),
        )

    def _score(
        self, model: RecognitionModel, pipeline: Pipeline, draw: PreparedDraw
    ) -> tuple[ScoredProbes, Gallery]:
        embedded = self.embed_draw(model, pipeline, draw)
        gallery = Gallery.enrol(embedded.enrolled)
        return score_probes(embedded.draw, gallery, embedded.mated, embedded.non_mated), gallery

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


def _digest(draw: PreparedDraw) -> DrawDigest:
    return DrawDigest(draw=draw.record.draw, selection_sha256=draw.record.selection_sha256)
