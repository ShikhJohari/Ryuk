"""LFW verification: every recognition model scored on View 2 by the 10-fold recipe (#9).

Each image goes through the benchmark pipeline once per model: YuNet, the centre-most usable
face (#17), a crop, then the model. Embeddings are cached by image and model key. A pair with
an image that has no usable face is not scored, and the image is listed in the results.

Pipeline choices are made on View 1 only. The one choice so far is FaceNet's crop, the YuNet
five-point alignment against the box with a margin of 14 (#7); the crop with the higher
DevTest accuracy wins, a tie going to the five-point crop.
"""

import logging
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from numpy.typing import NDArray

from ryuk.detector import Detector, Image, benchmark_face
from ryuk.evaluation import metrics
from ryuk.evaluation.embeddings import Cached, EmbeddingCache, image_key
from ryuk.evaluation.lfw import IMAGES, VIEW_1_TEST, VIEW_1_TRAIN, VIEW_2, Pair, read_pairs
from ryuk.evaluation.results import (
    CropTrial,
    Curve,
    DetectorId,
    Fold,
    Int8Footnote,
    LfwModel,
    OperatingPoint,
    Provenance,
    Published,
    RecognitionModelId,
    Verification,
)
from ryuk.recognition import AlignedSize, Network, RecognitionModel
from ryuk.recognition.faces import CROPS, Crop, face_crop

logger = logging.getLogger(__name__)

PUBLISHED: Mapping[Network, Published] = {
    "sface": Published(
        accuracy=0.9940,
        source="https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md",
    ),
    "arcface": Published(
        accuracy=0.9983,
        source="https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md",
        note=(
            "The buffalo_l pack's figure. The same README's row for the standalone "
            "R50 WebFace600K model says 99.80."
        ),
    ),
    "facenet": Published(
        accuracy=0.9965,
        source="https://github.com/davidsandberg/facenet",
        note=(
            "99.65 ± 0.25, measured on MTCNN crops with a margin of 32 and flipped-image "
            "averaging. Ryuk uses a YuNet crop and no flip, so a small shortfall is expected."
        ),
    ),
}
TOLERANCE_POINTS = 0.5
"""A model further than this from its published accuracy is flagged (#9)."""

FAR_TARGETS: tuple[tuple[float, bool], ...] = ((1e-2, False), (1e-3, True))
"""FAR targets and whether each is indicative: 1e-3 of 3,000 negatives is 3 false accepts."""

CANDIDATE_CROPS: Mapping[Network, tuple[Crop, ...]] = {
    "sface": ("five-point",),
    "arcface": ("five-point",),
    "facenet": CROPS,
}

_TIMED_FACES = 200
_WARM_UP = 10


@dataclass(frozen=True)
class Pipeline:
    """Everything between an image file and the face a model sees."""

    detector: Detector
    detector_sha256: str
    min_face_size: int
    crop: Crop

    @property
    def id(self) -> str:
        return f"yunet-{self.detector_sha256[:12]}-min{self.min_face_size}-{self.crop}"

    def with_crop(self, crop: Crop) -> "Pipeline":
        return Pipeline(self.detector, self.detector_sha256, self.min_face_size, crop)

    def face(self, image: Image, size: AlignedSize) -> Image | None:
        """The image's benchmark face, cropped for a model, or None if it has no usable face."""
        detection = benchmark_face(self.detector.detect(image), image.shape, self.min_face_size)
        if detection is None:
            return None
        return face_crop(self.detector, image, detection, self.crop, size)


def read_image(path: Path) -> tuple[str, Image]:
    """An image file's cache key and its decoded BGR pixels."""
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path} is not an image OpenCV can decode")
    return image_key(data), np.asarray(image, dtype=np.uint8)


def embed_images(
    model: RecognitionModel,
    pipeline: Pipeline,
    folder: Path,
    images: Iterable[PurePosixPath],
    cache: EmbeddingCache,
) -> dict[PurePosixPath, Cached]:
    """Each image's embedding under this model and pipeline, from the cache where it can be."""
    cached = cache.load(model.key, pipeline.id, model.dimension)
    computed: dict[str, Cached] = {}
    embeddings: dict[PurePosixPath, Cached] = {}
    for image_path in dict.fromkeys(images):
        key, image = read_image(folder / image_path)
        if key in cached:
            embeddings[image_path] = cached[key]
            continue
        if key not in computed:
            face = pipeline.face(image, model.input_size)
            computed[key] = None if face is None else model.embed(face)
            if len(computed) % 1000 == 0:
                logger.info("%s: %d images embedded", model.key.network, len(computed))
        embeddings[image_path] = computed[key]
    if computed:
        cache.save(model.key, pipeline.id, model.dimension, computed)
    return embeddings


@dataclass(frozen=True)
class ScoredPairs:
    """The pairs that could be scored: cosine score, same person or not, fold."""

    scores: NDArray[np.float64]
    same: NDArray[np.bool_]
    folds: NDArray[np.int_]


def score_pairs(pairs: Sequence[Pair], embeddings: Mapping[PurePosixPath, Cached]) -> ScoredPairs:
    """Cosine similarity for every pair whose two images both have an embedding."""
    kept = [
        (float(first @ second), pair.same, pair.fold)
        for pair in pairs
        if (first := embeddings[pair.first]) is not None
        and (second := embeddings[pair.second]) is not None
    ]
    if not kept:
        raise ValueError("no pair has two usable faces")
    scores, same, folds = zip(*kept, strict=True)
    return ScoredPairs(
        np.clip(np.array(scores, dtype=np.float64), -1.0, 1.0),
        np.array(same, dtype=np.bool_),
        np.array(folds, dtype=np.int_),
    )


def lfw_result(
    model: RecognitionModelId, crop: Crop, scored: ScoredPairs, published: Published
) -> LfwModel:
    """A model's View 2 result: 10-fold accuracy, AUC, operating points and ROC."""
    kfold = metrics.kfold_accuracy(scored.scores, scored.same, scored.folds)
    pairs_per_fold = np.bincount(scored.folds)
    gap = (kfold.mean - published.accuracy) * 100
    return LfwModel(
        model=model,
        crop=crop,
        accuracy=kfold.mean,
        standard_error=kfold.standard_error,
        folds=[
            Fold(accuracy=accuracy, threshold=threshold, pairs=int(pairs))
            for accuracy, threshold, pairs in zip(
                kfold.fold_accuracies, kfold.thresholds, pairs_per_fold, strict=True
            )
        ],
        auc=metrics.roc_auc(scored.scores, scored.same),
        operating_points=[
            _operating_point(scored, target, indicative) for target, indicative in FAR_TARGETS
        ],
        roc=_curve(metrics.compact(metrics.roc(scored.scores, scored.same))),
        published=published,
        gap_points=round(gap, 4),
        reproduces_published=abs(gap) <= TOLERANCE_POINTS,
    )


def _operating_point(scored: ScoredPairs, target: float, indicative: bool) -> OperatingPoint:
    point = metrics.tar_at_far(scored.scores, scored.same, target)
    return OperatingPoint(
        target_far=target,
        far=point.far,
        tar=point.tar,
        threshold=point.threshold if np.isfinite(point.threshold) else None,
        indicative=indicative,
    )


def _curve(roc: metrics.Roc) -> Curve:
    return Curve(far=[float(v) for v in roc.far], tar=[float(v) for v in roc.tar])


def crop_trial(network: Network, crop: Crop, train: ScoredPairs, test: ScoredPairs) -> CropTrial:
    """A crop's View 1 score: threshold chosen on DevTrain, accuracy on DevTest."""
    threshold = metrics.best_threshold(train.scores, train.same)
    return CropTrial(
        network=network,
        crop=crop,
        threshold=threshold,
        accuracy=metrics.accuracy(test.scores, test.same, threshold),
        chosen=False,
    )


def choose(trials: Sequence[CropTrial]) -> list[CropTrial]:
    """The trials with the best-scoring one marked chosen; the earliest wins a tie."""
    best = max(range(len(trials)), key=lambda index: (trials[index].accuracy, -index))
    return [
        trial.model_copy(update={"chosen": index == best}) for index, trial in enumerate(trials)
    ]


def model_id(model: RecognitionModel) -> RecognitionModelId:
    key = model.key
    if key.network == "fake":
        raise ValueError("the fake recognition model has no place in the results")
    return RecognitionModelId(
        network=key.network,
        provider=key.provider,
        weights_sha256=key.weights_sha256,
        dimension=model.dimension,
    )


def median_ms_per_face(model: RecognitionModel, faces: Sequence[Image]) -> float:
    """Warm median time for one embedding, over up to 200 faces after 10 warm-up calls."""
    for face in faces[:_WARM_UP]:
        model.embed(face)
    timings = []
    for face in faces[:_TIMED_FACES]:
        start = time.perf_counter()
        model.embed(face)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


@dataclass(frozen=True)
class LfwData:
    folder: Path
    view_1_train: list[Pair]
    view_1_test: list[Pair]
    view_2: list[Pair]

    @classmethod
    def read(cls, data_dir: Path) -> "LfwData":
        return cls(
            folder=data_dir / IMAGES,
            view_1_train=read_pairs(data_dir / VIEW_1_TRAIN),
            view_1_test=read_pairs(data_dir / VIEW_1_TEST),
            view_2=read_pairs(data_dir / VIEW_2),
        )


def _images(pairs: Iterable[Pair]) -> list[PurePosixPath]:
    return list(dict.fromkeys(path for pair in pairs for path in (pair.first, pair.second)))


@dataclass(frozen=True)
class Models:
    """How to load each compared recognition model, and SFace int8 for its footnote."""

    compared: Mapping[Network, Callable[[], RecognitionModel]]
    sface_int8: Callable[[], RecognitionModel]


@dataclass(frozen=True)
class LfwEvaluation:
    lfw: LfwData
    pipeline: Pipeline
    cache: EmbeddingCache

    def run(self, models: Models, provenance: Provenance) -> Verification:
        """Tune on View 1, score View 2 once per model, and measure the SFace int8 footnote."""
        trials: list[CropTrial] = []
        results: list[LfwModel] = []
        excluded: set[PurePosixPath] = set()
        for network, load in models.compared.items():
            model = load()
            logger.info("evaluating %s (%s)", network, model.key.id)
            network_trials = self._tune(network, model)
            trials.extend(network_trials)
            crop = next((t.crop for t in network_trials if t.chosen), CANDIDATE_CROPS[network][0])
            embeddings = self._embed(model, crop, _images(self.lfw.view_2))
            excluded.update(path for path, embedding in embeddings.items() if embedding is None)
            scored = score_pairs(self.lfw.view_2, embeddings)
            results.append(lfw_result(model_id(model), crop, scored, PUBLISHED[network]))

        return Verification(
            provenance=provenance,
            detector=DetectorId(
                weights_sha256=self.pipeline.detector_sha256,
                min_face_size=self.pipeline.min_face_size,
            ),
            pairs=len(self.lfw.view_2),
            excluded_images=sorted(str(path) for path in excluded),
            view_1=trials,
            models=results,
            sface_int8=self._int8_footnote(models.sface_int8(), models.compared["sface"]()),
        )

    def _embed(
        self, model: RecognitionModel, crop: Crop, images: Iterable[PurePosixPath]
    ) -> dict[PurePosixPath, Cached]:
        return embed_images(
            model, self.pipeline.with_crop(crop), self.lfw.folder, images, self.cache
        )

    def _tune(self, network: Network, model: RecognitionModel) -> list[CropTrial]:
        """View 1 trials of each candidate crop, if the network has more than one."""
        candidates = CANDIDATE_CROPS[network]
        if len(candidates) == 1:
            return []
        images = _images([*self.lfw.view_1_train, *self.lfw.view_1_test])
        trials = []
        for crop in candidates:
            embeddings = self._embed(model, crop, images)
            train = score_pairs(self.lfw.view_1_train, embeddings)
            test = score_pairs(self.lfw.view_1_test, embeddings)
            trials.append(crop_trial(network, crop, train, test))
        return choose(trials)

    def _int8_footnote(self, int8: RecognitionModel, fp32: RecognitionModel) -> Int8Footnote:
        """SFace int8 against fp32: its own View 2 accuracy, drift from fp32, and speed."""
        images = _images(self.lfw.view_2)
        int8_embeddings = self._embed(int8, "five-point", images)
        fp32_embeddings = self._embed(fp32, "five-point", images)
        cosines = np.array(
            [
                float(first @ second)
                for path in images
                if (first := int8_embeddings[path]) is not None
                and (second := fp32_embeddings[path]) is not None
            ]
        )
        scored = score_pairs(self.lfw.view_2, int8_embeddings)
        kfold = metrics.kfold_accuracy(scored.scores, scored.same, scored.folds)
        faces = self._faces(images, _WARM_UP + _TIMED_FACES)
        return Int8Footnote(
            model=model_id(int8),
            accuracy=kfold.mean,
            standard_error=kfold.standard_error,
            cosine_to_fp32_mean=float(np.clip(cosines.mean(), -1.0, 1.0)),
            cosine_to_fp32_min=float(np.clip(cosines.min(), -1.0, 1.0)),
            faces_compared=len(cosines),
            ms_per_face_int8=median_ms_per_face(int8, faces),
            ms_per_face_fp32=median_ms_per_face(fp32, faces),
        )

    def _faces(self, images: Iterable[PurePosixPath], count: int) -> list[Image]:
        """The first `count` usable five-point 112-pixel faces among `images`."""
        pipeline = self.pipeline.with_crop("five-point")
        faces: list[Image] = []
        for path in images:
            face = pipeline.face(read_image(self.lfw.folder / path)[1], 112)
            if face is not None:
                faces.append(face)
                if len(faces) == count:
                    break
        return faces
