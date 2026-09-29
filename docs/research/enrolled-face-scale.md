# Enrolled-face scale and ICC handling

Issue: [ShikhJohari/Ryuk#53](https://github.com/ShikhJohari/Ryuk/issues/53), the 28 Sep audit's M5. Measured on 2026-09-28 on the M4 Mac, at `main` 7d9038c, with the committed weights, the frozen thresholds in `evaluation/results.json` and ArcFace on its default provider (CoreML). OpenCV 5.0.0, Pillow 12.3.0, onnxruntime 1.30.0.

## Question

M5 asked two things:

- **The crop shrink.** A face enrolled from a phone close-up is 400–800 px across. The five-point crop (`Detector.align`, OpenCV's `alignCrop`) warps it straight to 112 px with a bilinear `warpAffine` and no prefilter. The CelebA faces the thresholds were frozen on (median box 86 px) were warped at close to 1:1. Does the shrink move enrolled embeddings enough to matter?
- **ICC.** `prepare_photo` drops the ICC profile without converting to sRGB, so a Display P3 upload is read as if it were sRGB. Does that matter?

**Material** means one of these:

- a frozen threshold or the active-model choice would change;
- or more than about 1% of mated probes cross the threshold.

## Verdict

- **The shrink is not material.** Take the same landmarks, warp a 5× larger copy of the face, and the embedding moves no more than re-saving the photo as a JPEG at quality 95 moves it. The mean match-score shift is at most 0.002, with no consistent direction, so no threshold moves. Extrapolated to the 7,500 validation mated probes, the shrink moves 0.36% of SFace's probes across the threshold and 0.01% of ArcFace's. A JPEG re-encode moves 0.48% and 0.01%. **No code change for the crop.**
- **ICC is not material.** A P3 JPEG read as sRGB changes each pixel by 2.9/255 on average. Its embedding is as close to the sRGB control as a JPEG re-encode is (cosine 0.991–0.995). Converting to sRGB changes no decision. **No code change is required.** The conversion is still cheap and correct (see Recommendation).
- **What does move a close-up's embedding is where YuNet puts the landmarks at the new scale, not the warp.**
  - Warp the native image with the landmarks YuNet found on the 5× image, and the cosine falls to 0.955 (SFace), 0.972 (ArcFace) and 0.974 (FaceNet). The whole close-up path gives the same figures.
  - The landmarks move by 7.9% of the inter-ocular distance on average.
  - The proposed crop-from-a-downscaled-copy fix does not touch this. Re-detecting on a copy with a 128 px face does not help either.
  - The effect is symmetric: the mean score shift is between −0.004 and +0.006.
  - It crosses about 1.2% of SFace's mated probes in both directions, so net ≈ 0. FaceNet's figure is 4.2% (net −1.4 points). ArcFace, the active model, is at 0.03%.
  - This is a separate question from M5; see Recommendation.
- **Full-resolution detection of the 5× image found a usable face in 34 of 300 images.** This confirms the audit's B1 from another side. On `main` at 7d9038c, `_enrollable` still detects at full resolution. The close-up rows below therefore use the bounded detection that the B1 fix adds, condition (3).

## Method

The script is under Reproduction. Its steps:

1. **Draw.** Rebuild the CelebA validation draw with the repo's own `CelebaEvaluation.prepare` (YuNet scan, `benchmark_face`, seed 27). Assert that its `selection_sha256` matches the committed one (`75f843d5d2a1…`).
2. **Take 300 usable faces from it:**
   - a gallery of the first 50 gallery identities, one enrolled photo each;
   - 150 mated probes, 3 per gallery identity;
   - 100 non-mated probes, one per held-out identity, the first 100.

   All CelebA images here are 178×218 PNG.
3. **Embed each face under every condition below, with each model's committed crop.** SFace and ArcFace use `five-point`; FaceNet uses `box-margin-32`. Models load through `ryuk.recognition.load.load_model` and cut through `ryuk.recognition.faces.face_crop`, as enrollment does.
   - **Parity check.** The native embeddings match the evaluation's embedding cache (minimum cosine 0.9999999 for all 300 images under all three models). The cached validation TPIRs reproduce the committed ones exactly: 0.9483, 0.9817 and 0.8453.
4. **Upscaling.** Every upscale is `cv2.resize` ×5 with `INTER_LANCZOS4`, giving 890×1090 and a median face box of 428 px.

The conditions (FIX is the proposed fix: crop from an `INTER_AREA` copy shrunk so that the box's short side is 128 px, with the landmarks mapped onto it):

| Condition | What it isolates |
|---|---|
| `jpeg95` | Noise floor: the native image through `prepare_photo` as a quality-95 JPEG, re-detected |
| `shrink` | **The shrink alone**: the 5× image, cut with the native landmarks mapped onto it (pixel-centre corrected) |
| `shrink+FIX` | The same, cut from the 128 px copy |
| `shrink+noise`, `shrink+noise+FIX` | As above, with Gaussian noise σ = 4/255 added at 5× scale; see Caveats |
| `landmarks` | **The landmark change alone**: the native image, cut with the landmarks YuNet found on the 5× image mapped back |
| (2) `5x-full` | The 5× image detected at full resolution and cut from it: `main` today |
| (3) `5x-bounded` | The 5× image detected on a copy bounded to 640 px (`MAX_DETECTION_SIDE`), boxes scaled back, cut from the 5× image: the B1 fix |
| `5x-bounded+FIX` | (3), cut from the 128 px copy |
| `5x-redetect128` | (3), then detected again on the 128 px copy and cut from it |

**Decisions.** Each model is scored at its frozen threshold (SFace 0.4980, ArcFace 0.3417, FaceNet 0.7088) with the best-photo rule (`Gallery.top_candidates`). The gallery is 50 identities × 1 photo. A mated probe is a hit when its top candidate is right and scores at or above the threshold. Two set-ups:

- **A:** a native gallery, with the probes under the condition. This is the brief's set-up.
- **B:** a gallery under the condition, with native probes. This is the production case: a close-up enrolled, a small face seen live.

**Extrapolation.** 150 probes is too few to resolve 1%. So every measured A-set-up score change is applied to every one of the 7,500 validation mated probes, as scored from the cache with the real 5-photo gallery. The result is the expected share that would stop matching (lost) or start matching (gained).

## Results

### Cosine to the native embedding (mean / 5th percentile / min)

| Condition | n | SFace | ArcFace (CoreML) | FaceNet |
|---|---|---|---|---|
| `jpeg95` (noise floor) | 299 | 0.9912 / 0.9775 / 0.9364 | 0.9925 / 0.9832 / 0.9420 | 0.9939 / 0.9776 / 0.9432 |
| `shrink` | 300 | 0.9955 / 0.9922 / 0.9856 | 0.9960 / 0.9932 / 0.9903 | 0.9894 / 0.9788 / 0.9361 |
| `shrink+FIX` | 300 | 0.9970 / 0.9946 / 0.9923 | 0.9968 / 0.9944 / 0.9888 | 0.9949 / 0.9893 / 0.9527 |
| `shrink+noise` | 300 | 0.9901 / 0.9835 / 0.9664 | 0.9904 / 0.9846 / 0.9747 | 0.9893 / 0.9780 / 0.9375 |
| `shrink+noise+FIX` | 300 | 0.9964 / 0.9936 / 0.9895 | 0.9962 / 0.9938 / 0.9885 | 0.9948 / 0.9893 / 0.9532 |
| `landmarks` | 298 | 0.9545 / 0.8986 / 0.8073 | 0.9719 / 0.9420 / 0.8542 | 0.9739 / 0.9372 / 0.9076 |
| (2) `5x-full` | **34** | 0.9518 / 0.9210 / 0.9163 | 0.9718 / 0.9552 / 0.9453 | 0.9694 / 0.9408 / 0.9263 |
| (3) `5x-bounded` | 298 | 0.9533 / 0.8972 / 0.8014 | 0.9713 / 0.9419 / 0.8465 | 0.9754 / 0.9445 / 0.9037 |
| `5x-bounded+FIX` | 298 | 0.9551 / 0.8976 / 0.8059 | 0.9728 / 0.9436 / 0.8533 | 0.9750 / 0.9449 / 0.9064 |
| `5x-redetect128` | 298 | 0.9480 / 0.8761 / 0.7903 | 0.9698 / 0.9399 / 0.8789 | 0.9765 / 0.9555 / 0.9077 |

- (2) found a usable face in only 34 of the 300 images; (3) found one in 298. Its statistics cover only those 34.
- **Landmark error.** For the `landmarks` condition, the landmarks YuNet finds on the 5× image (mapped back) are on average 7.9% of the inter-ocular distance from the native ones (median 7.1%, 95th percentile 14.9%). For `5x-redetect128` the mean is 8.4%.

### Decisions at the frozen thresholds

On the native baseline, 118 (SFace), 145 (ArcFace) and 88 (FaceNet) of the 150 mated probes are hits, and none of the 100 non-mated probes matches anyone.

Each cell reads:

- **A:** lost / gained, of 150 probes;
- **B:** lost / gained, of 150 probes;
- **Validation:** the expected lost / gained share of the 7,500 validation mated probes;
- **Δ:** the mean change in the mated score.

| Condition | SFace | ArcFace | FaceNet |
|---|---|---|---|
| `jpeg95` | A 2/0 · B 1/3 · val 0.26%/0.22% · Δ +0.002 | A 1/0 · B 0/0 · val 0.01%/0.00% · Δ 0.000 | A 4/1 · B 2/1 · val 0.97%/0.66% · Δ 0.000 |
| `shrink` | A 3/2 · B 3/3 · val 0.19%/0.17% · Δ 0.000 | A 0/0 · B 0/0 · val 0.01%/0.00% · Δ +0.001 | A 5/1 · B 3/0 · val 1.25%/1.10% · Δ −0.001 |
| `shrink+FIX` | A 1/3 · B 1/2 · val 0.12%/0.16% · Δ +0.001 | A 0/0 · B 0/0 · val 0.01%/0.00% · Δ +0.001 | A 4/1 · B 3/0 · val 0.83%/0.78% · Δ −0.001 |
| `shrink+noise` | A 2/3 · B 4/3 · val 0.34%/0.20% · Δ 0.000 | A 0/0 · B 0/1 · val 0.01%/0.00% · Δ −0.001 | A 5/1 · B 3/0 · val 1.26%/1.07% · Δ 0.000 |
| `shrink+noise+FIX` | A 1/3 · B 2/1 · val 0.13%/0.16% · Δ +0.001 | A 0/0 · B 0/0 · val 0.01%/0.00% · Δ +0.001 | A 4/1 · B 4/0 · val 0.84%/0.77% · Δ −0.001 |
| `landmarks` | A 4/3 · B 5/5 · val 0.57%/0.54% · Δ +0.005 | A 1/0 · B 0/0 · val 0.02%/0.01% · Δ +0.002 | A 10/2 · B 3/3 · val 3.06%/1.18% · Δ +0.001 |
| (3) `5x-bounded` | A 4/6 · B 5/5 · val 0.62%/0.55% · Δ +0.005 | A 0/0 · B 0/0 · val 0.02%/0.01% · Δ +0.001 | A 6/2 · B 5/4 · val 2.77%/1.40% · Δ +0.001 |
| `5x-bounded+FIX` | A 4/5 · B 6/5 · val 0.57%/0.56% · Δ +0.006 | A 1/0 · B 0/0 · val 0.02%/0.01% · Δ +0.001 | A 6/3 · B 3/4 · val 2.58%/1.25% · Δ 0.000 |
| `5x-redetect128` | A 3/2 · B 6/5 · val 0.96%/0.45% · Δ +0.003 | A 0/0 · B 0/1 · val 0.02%/0.01% · Δ +0.001 | A 7/4 · B 5/1 · val 2.10%/1.41% · Δ −0.004 |

- **Non-mated probes.** None started or stopped matching in any condition or set-up.
- **Where Δ comes from.** It is set-up B's mean where B exists, otherwise A's.
- **Why FaceNet's crossings are large everywhere.** Its threshold sits in a dense part of its score distribution: validation TPIR is 84.5%. So even a JPEG re-encode crosses 1.6% of its probes. The `shrink` figure is not the missing-prefilter mechanism either: `box_crop` already resizes with `INTER_AREA`, which is a prefilter.
  - A plausible contributor, not verified: at 86 px the box edges are truncated to whole pixels (`int(...)` in `box_crop`), a larger relative jitter than at 5×.

### ICC (Display P3)

Each of the 50 gallery photos was uploaded twice through `prepare_photo`:

- **Control:** as an sRGB JPEG, quality 95.
- **P3 upload:** converted to Display P3 with LittleCMS (`ImageCms`, relative colorimetric) against macOS's `/System/Library/ColorSync/Profiles/Display P3.icc`, then saved as a quality-95 JPEG with that profile embedded. Both show the same colours in a colour-managed viewer.

Each P3 upload was then stored two ways:

- **As `main` stores it:** the profile is dropped, so P3 values are read as sRGB.
- **With the fix:** converted to sRGB (`ImageCms.profileToProfile`) and then re-encoded as `prepare_photo` does.

Reading P3 as sRGB changes each pixel by 2.9/255 on average.

| Against the sRGB control (n = 50) | SFace | ArcFace | FaceNet |
|---|---|---|---|
| P3 read as sRGB (`main`), cosine mean / p5 / min | 0.9914 / 0.9775 / 0.9713 | 0.9915 / 0.9815 / 0.9747 | 0.9946 / 0.9853 / 0.9825 |
| P3 converted to sRGB, cosine | 0.9937 / 0.9869 / 0.9835 | 0.9931 / 0.9878 / 0.9797 | 0.9959 / 0.9888 / 0.9774 |
| For scale: control vs native PNG, cosine | 0.9909 / 0.9777 / 0.9471 | 0.9922 / 0.9813 / 0.9608 | 0.9941 / 0.9812 / 0.9554 |
| Decisions, P3 read as sRGB (B lost/gained · val) | 3/0 · 0.28%/0.17% | 0/1 · 0.01%/0.00% | 1/1 · 0.80%/0.58% |
| Decisions, P3 converted | 3/0 · 0.21%/0.18% | 1/1 · 0.01%/0.00% | 1/2 · 0.82%/0.56% |

The P3-as-sRGB error is the size of a JPEG re-encode. Converting removes about a quarter of it (SFace 1 − cosine goes from 0.0086 to 0.0063), and changes no decision.

## Recommendation

1. **No crop change for M5.** Cropping from a copy downscaled so that the face is about 112–160 px does make the shrink cleaner:
   - It raises the `shrink` cosine from 0.9955 to 0.9970 for SFace.
   - It recovers the noise case, from 0.990 to 0.996.

   But the shrink was already below the JPEG noise floor, and the fix moves no decision. If it is ever wanted, it is a small change in `face_crop`: for `five-point`, when the box's short side is over 160 px, resize the image with `INTER_AREA` by 128 / short side, map the landmarks with the pixel-centre rule `(p + 0.5)·f − 0.5`, and call `alignCrop` on the copy. It would need a test that a 5× face and its native face embed within the JPEG floor. It is not warranted by these numbers.
2. **No ICC change is required by the criterion.** The conversion is still cheap and correct, and it also fixes how a P3 photo looks when the client shows the stored copy. If it is wanted, it belongs in `photos._decode`:
   - After `exif_transpose`, if `opened.info.get("icc_profile")` is present, apply `ImageCms.profileToProfile(..., ImageCms.createProfile("sRGB"), outputMode="RGB")`, then strip as now.
   - Test with a Pillow-written P3 JPEG whose stored pixels match the sRGB original within a few levels.

   That is a product call for #43's watchlist fix pack, not a finding of this measurement.
3. **Land B1's bounded detection first.** Full-resolution detection lost 266 of 300 close-ups here. Once it lands, enrollment works as condition (3) does.
4. **The effect that is large is YuNet's scale-dependent landmark placement.** The crop change will not fix it.
   - It is net-neutral for the active model: ArcFace is at 0.03% of mated probes, with Δ +0.001.
   - It crosses about 1.2% of SFace's mated probes in both directions, and 4.2% of FaceNet's, net −1.4 points.
   - Presumably it applies equally to live faces that are not about 86 px (not measured), so it is a question of how representative CelebA's single scale is. It is not specific to enrollment.
   - If anyone pursues it, it is a measurement for #28/#32's report (scale-stratified TPIR), not a code change here.

## Caveats

- **Synthetic close-ups.** A Lanczos 5× upscale holds no detail above the original's Nyquist frequency. A real 400 px face has pores, eyelashes and sensor noise, which a no-prefilter warp aliases. So the `shrink` rows understate the aliasing.
  - The σ = 4 noise rows are the stand-in: they show the mechanism is real (cosine −0.005) and that the fix removes it, but it stays below the JPEG floor.
  - Real close-ups also differ in pose, lens perspective and lighting. None of that is measured.
- **Sample size.** With 150 mated probes, 1% is 1.5 probes, so the per-probe counts are indicative only; the validation extrapolation is the better guide.
  - The extrapolation assumes that a probe-side score change measured against a one-photo gallery carries over to best-of-5 scores.
  - For ICC it uses the gallery-side (B) changes.
- **Non-mated probes.** A 50-identity gallery gives non-mated top scores far below the threshold, so FPIR effects are not resolved here.
- **ICC gamut.** The P3 photos were synthesised from sRGB content, so every colour lies inside sRGB. Real P3 captures can hold saturated colours outside sRGB. Skin rarely does, but the backgrounds a crop includes can.
- **Platform.** The P3 profile path is macOS's. On another OS, point `P3_ICC` at any Display P3 ICC file, or the ICC part is skipped. ArcFace was measured on CoreML only.

## Reproduction

From the repository root, with the weights fetched, CelebA fetched and the embedding cache populated by `ryuk evaluate celeba`, save the script below as `enrolled_face_scale.py` outside the repository and run:

```sh
uv run python /path/to/enrolled_face_scale.py /tmp/enrolled-face-scale.json
```

It takes about 2.5 minutes on the M4: the validation scan about 8 s, then all three models. It writes nothing but stdout and the JSON named by its argument, and it only reads the embedding cache. The noise uses seed 53.

<details>
<summary><code>enrolled_face_scale.py</code></summary>

```python
"""#53 (M5): does cutting an enrolled face out of a large photo move its embedding?

Run from the repository root: `uv run python enrolled_face_scale.py [out.json]`. Reads data/raw, models/weights,
evaluation/results.json and (read-only) data/cache/embeddings; writes nothing but stdout and
the JSON named by argv[1], if given.
"""

import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image as PILImage
from PIL import ImageCms

from ryuk.datasets.celeba import iter_images
from ryuk.detector import (
    MAX_DETECTION_SIDE,
    MIN_USABLE_FACE_SIZE,
    Detection,
    Detector,
    benchmark_face,
)
from ryuk.eda.scan import Scanner, default_workers
from ryuk.evaluation.celeba import CelebaEvaluation
from ryuk.evaluation.embeddings import EmbeddingCache, image_key
from ryuk.evaluation.openset import Gallery
from ryuk.evaluation.results import read_results
from ryuk.evaluation.verification import Pipeline
from ryuk.fetch.pinned import file_checksum
from ryuk.recognition.faces import face_crop
from ryuk.recognition.load import NETWORKS, load_model
from ryuk.watchlist.photos import prepare_photo
from ryuk.weights import YUNET

T0 = time.perf_counter()
ROOT = Path("data/raw")
WEIGHTS = Path("models/weights")
N_GALLERY, PROBES_EACH, N_NON_MATED = 50, 3, 100
SCALE = 5
FIX_BOX = 128  # the fix: crop from a copy shrunk so the box's short side is this many px
NOISE_SIGMA = 4.0
P3_ICC = Path("/System/Library/ColorSync/Profiles/Display P3.icc")


def log(*a):
    print(f"[{time.perf_counter() - T0:6.1f}s]", *a, flush=True)


results = read_results(Path("evaluation/results.json"))
assert results is not None and results.identification is not None
thresholds = {t.model.network: t for t in results.thresholds}
crops = {m.model.network: m.crop for m in results.identification.models}
committed = next(d for d in results.identification.draws if d.draw == "validation")

yunet = YUNET.path(WEIGHTS)
detector = Detector(yunet)
pipeline = Pipeline(detector, file_checksum(yunet, "sha256"), MIN_USABLE_FACE_SIZE, "five-point")
evaluation = CelebaEvaluation(
    root=ROOT,
    pipeline=pipeline,
    cache=EmbeddingCache(Path("data/cache/embeddings")),
    scanner=Scanner(yunet, default_workers()),
)
prepared = evaluation.prepare("validation")
assert prepared.record.selection_sha256 == committed.selection_sha256, "draw differs"
log("validation draw rebuilt, selection_sha256 matches", committed.selection_sha256[:12])

sel = prepared.selection
gallery_ids = sel.gallery[:N_GALLERY]
gallery_photo = {g.identity: g.enrolled[0] for g in gallery_ids}
mated = [(g.identity, p) for g in gallery_ids for p in g.probes[:PROBES_EACH]]
non_mated = [(h.identity, h.probes[0]) for h in sel.held_out[:N_NON_MATED]]
names = [*gallery_photo.values(), *(p for _, p in mated), *(p for _, p in non_mated)]
labels = prepared.labels.filter(pc.is_in(prepared.labels.column("path"), pa.array(names)))
images, pngs = {}, {}
for im in iter_images(ROOT, labels):
    images[im.path] = im.decode()
    pngs[im.path] = im.png
log(f"{len(images)} images ({N_GALLERY} gallery, {len(mated)} mated, {len(non_mated)} non-mated)")
shapes = {im.shape[:2] for im in images.values()}
log("image shapes:", shapes)


def upscale(image):
    h, w = image.shape[:2]
    return np.asarray(
        cv2.resize(image, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4), np.uint8
    )


def scaled(d: Detection, f: float) -> Detection:
    """`d` in the pixels of the image resized by `f`, pixel centres mapped as cv2.resize maps them."""
    from ryuk.detector import Box, Landmarks

    b = d.box
    return Detection(
        Box((b.x + 0.5) * f - 0.5, (b.y + 0.5) * f - 0.5, b.width * f, b.height * f),
        Landmarks(*(((x + 0.5) * f - 0.5, (y + 0.5) * f - 0.5) for x, y in d.landmarks)),
        d.score,
    )


def shrink_for_crop(image, d: Detection):
    """The fix: an INTER_AREA copy with the box's short side FIX_BOX px, and the detection on it."""
    f = FIX_BOX / d.box.short_side
    if f >= 1:
        return image, d
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA
    )
    fx, fy = small.shape[1] / w, small.shape[0] / h
    return np.asarray(small, np.uint8), scaled(d, (fx + fy) / 2)


def jpeg(pil, icc=None):
    buf = io.BytesIO()
    kwargs = {"quality": 95}
    if icc is not None:
        kwargs["icc_profile"] = icc
    pil.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


rng = np.random.default_rng(53)
# condition -> path -> (image, detection) or None
CONDITIONS = [
    "native",
    "native_jpeg95",
    "native_up5lm",
    "up5_nativelm",
    "up5_nativelm_fix",
    "up5n_nativelm",
    "up5n_nativelm_fix",
    "up5_full",
    "up5_bounded",
    "up5_bounded_fix",
    "up5_redetect128",
    "up5n_bounded",
    "up5n_bounded_fix",
]
landmark_error = {"native_up5lm": [], "up5_redetect128": []}


def nme(a: Detection, b: Detection) -> float:
    """Mean landmark distance between two detections of one face, over a's inter-ocular distance."""
    pa_, pb = np.array(a.landmarks), np.array(b.landmarks)
    return float(np.linalg.norm(pa_ - pb, axis=1).mean() / np.linalg.norm(pa_[0] - pa_[1]))


conditions: dict[str, dict[str, tuple | None]] = {c: {} for c in CONDITIONS}
boxes = {c: [] for c in conditions}
for path, image in images.items():
    native_det = prepared.faces[path]
    conditions["native"][path] = (image, native_det)
    boxes["native"].append(native_det.box.short_side)
    jp = prepare_photo(jpeg(PILImage.fromarray(image[:, :, ::-1].copy()))).pixels()
    dj = benchmark_face(detector.detect(jp), jp.shape)
    conditions["native_jpeg95"][path] = None if dj is None else (jp, dj)
    big = upscale(image)
    # Pure shrink: the upscaled image, cut with the native landmarks mapped onto it.
    conditions["up5_nativelm"][path] = (big, scaled(native_det, SCALE))
    conditions["up5_nativelm_fix"][path] = shrink_for_crop(big, scaled(native_det, SCALE))
    d = benchmark_face(detector.detect(big), big.shape)
    conditions["up5_full"][path] = None if d is None else (big, d)
    d = benchmark_face(detector.detect(big, max_side=MAX_DETECTION_SIDE), big.shape)
    conditions["up5_bounded"][path] = None if d is None else (big, d)
    conditions["up5_bounded_fix"][path] = None if d is None else shrink_for_crop(big, d)
    # Pure landmark change: the native image, cut with the upscaled image's landmarks mapped back.
    conditions["native_up5lm"][path] = None if d is None else (image, scaled(d, 1 / SCALE))
    if d is not None:
        landmark_error["native_up5lm"].append(nme(native_det, scaled(d, 1 / SCALE)))
    # Shrink so the box is FIX_BOX px, then detect again on that copy and crop from it.
    conditions["up5_redetect128"][path] = None
    if d is not None:
        small, _ = shrink_for_crop(big, d)
        d2 = benchmark_face(detector.detect(small), small.shape)
        if d2 is not None:
            conditions["up5_redetect128"][path] = (small, d2)
            landmark_error["up5_redetect128"].append(
                nme(native_det, scaled(d2, image.shape[1] / small.shape[1]))
            )
    noisy = np.clip(big.astype(np.float32) + rng.normal(0, NOISE_SIGMA, big.shape), 0, 255).astype(
        np.uint8
    )
    conditions["up5n_nativelm"][path] = (noisy, scaled(native_det, SCALE))
    conditions["up5n_nativelm_fix"][path] = shrink_for_crop(noisy, scaled(native_det, SCALE))
    dn = benchmark_face(detector.detect(noisy, max_side=MAX_DETECTION_SIDE), noisy.shape)
    conditions["up5n_bounded"][path] = None if dn is None else (noisy, dn)
    conditions["up5n_bounded_fix"][path] = None if dn is None else shrink_for_crop(noisy, dn)
    for c in conditions:
        if c != "native" and conditions[c][path] is not None:
            boxes[c].append(conditions[c][path][1].box.short_side)
for c, v in landmark_error.items():
    log(
        f"landmark NME vs native, {c}: mean {np.mean(v):.4f} median {np.median(v):.4f} p95 {np.percentile(v, 95):.4f}"
    )
for c in conditions:
    found = sum(v is not None for v in conditions[c].values())
    log(
        f"{c:18s} usable face found in {found}/{len(images)}; median box short side "
        f"{np.median(boxes[c]) if boxes[c] else float('nan'):.0f} px"
    )


# ICC: each gallery photo as an sRGB JPEG (control) and as the same colours encoded in Display P3.
def to_srgb(upload: bytes):
    """The fix: decode and convert the embedded profile to sRGB, as BGR."""
    with PILImage.open(io.BytesIO(upload)) as im:
        icc = im.info.get("icc_profile")
        rgb = im.convert("RGB")
        if icc:
            rgb = ImageCms.profileToProfile(
                rgb,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                SRGB,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB",
            )
    # Stored as prepare_photo stores it: re-encoded once at quality 95.
    with PILImage.open(io.BytesIO(jpeg(rgb))) as stored:
        return np.ascontiguousarray(np.asarray(stored.convert("RGB"), np.uint8)[:, :, ::-1])


icc_images: dict[str, dict[str, np.ndarray]] = {"srgb": {}, "p3_as_srgb": {}, "p3_converted": {}}
pixel_shift = []
if P3_ICC.is_file():
    SRGB = ImageCms.createProfile("sRGB")
    P3 = ImageCms.getOpenProfile(str(P3_ICC))
    p3_bytes = P3_ICC.read_bytes()
    to_p3 = ImageCms.buildTransform(
        SRGB, P3, "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC
    )
    for path in gallery_photo.values():
        rgb = PILImage.fromarray(images[path][:, :, ::-1].copy())
        srgb_upload = jpeg(rgb)
        p3_upload = jpeg(ImageCms.applyTransform(rgb, to_p3), p3_bytes)
        icc_images["srgb"][path] = prepare_photo(srgb_upload).pixels()
        icc_images["p3_as_srgb"][path] = prepare_photo(p3_upload).pixels()  # production today
        icc_images["p3_converted"][path] = to_srgb(p3_upload)
        pixel_shift.append(
            np.abs(
                icc_images["p3_as_srgb"][path].astype(int) - icc_images["srgb"][path].astype(int)
            ).mean()
        )
    log(f"ICC: mean |pixel| change P3-as-sRGB vs sRGB {np.mean(pixel_shift):.2f} / 255")
else:
    log("ICC: no Display P3 profile; skipped")


def cos_stats(a, b, keys):
    cs = np.array(
        [float(a[k] @ b[k]) for k in keys if a.get(k) is not None and b.get(k) is not None]
    )
    return {
        "n": int(cs.size),
        "mean": float(cs.mean()),
        "p5": float(np.percentile(cs, 5)),
        "min": float(cs.min()),
    }


def score(gallery_emb, probe_emb, probes, t):
    g = Gallery.enrol({i: [gallery_emb[p]] for i, p in gallery_photo.items()})
    keys = [(i, p) for i, p in probes if probe_emb.get(p) is not None]
    top, s = g.top_candidates(np.stack([probe_emb[p] for _, p in keys]))
    return {p: (float(sc), bool(tp == i), bool(sc >= t)) for (i, p), sc, tp in zip(keys, s, top)}


DELTAS: dict = {}


def flips(base, other, kind):
    lost = gained = 0
    deltas = DELTAS.setdefault(kind, [])
    deltas.clear()
    for p, (s0, c0, a0) in base.items():
        if p not in other:
            continue
        s1, c1, a1 = other[p]
        deltas.append(s1 - s0)
        ok0 = a0 and c0 if kind == "mated" else a0
        ok1 = a1 and c1 if kind == "mated" else a1
        lost += ok0 and not ok1
        gained += ok1 and not ok0
    base_ok = sum((a and c) if kind == "mated" else a for s, c, a in base.values())
    return {
        "n": len(deltas),
        "base_positive": base_ok,
        "lost": lost,
        "gained": gained,
        "mean_dscore": float(np.mean(deltas)),
        "p5_dscore": float(np.percentile(deltas, 5)),
        "p95_dscore": float(np.percentile(deltas, 95)),
    }


out = {
    "landmark_nme": {
        c: {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
        }
        for c, v in landmark_error.items()
    },
    "conditions": {c: sum(v is not None for v in conditions[c].values()) for c in conditions},
    "median_box": {c: float(np.median(v)) if v else None for c, v in boxes.items()},
    "icc_pixel_shift": float(np.mean(pixel_shift)) if pixel_shift else None,
    "models": {},
}
cache = EmbeddingCache(Path("data/cache/embeddings"))
draw_keys = {im.path: image_key(im.png) for im in iter_images(ROOT, prepared.labels)}
log(f"{len(draw_keys)} validation draw images keyed")
VAL: dict = {}


def expected(deltas):
    """Each validation mated probe's score moved by every measured delta in turn: the mean share
    of the 7,500 that would stop matching and start matching at the frozen threshold."""
    s, correct, t = VAL["score"], VAL["correct"], VAL["t"]
    moved = s[:, None] + deltas[None, :]
    before = (s >= t) & correct
    after = (moved >= t) & correct[:, None]
    return {
        "lost": float((before[:, None] & ~after).mean()),
        "gained": float((~before[:, None] & after).mean()),
    }


for network in NETWORKS:
    model = load_model(network, WEIGHTS)
    t = thresholds[network]
    assert (
        model.key.provider == t.model.provider
        and model.key.weights_sha256 == t.model.weights_sha256
    )
    crop, size = crops[network], model.input_size
    thr = t.threshold
    cached_draw = cache.load(model.key, pipeline.with_crop(crop).id, model.dimension)
    ev = {path: cached_draw[key] for path, key in draw_keys.items()}
    vg = Gallery.enrol({g.identity: [ev[p] for p in g.enrolled] for g in sel.gallery})
    vtop, vscore = vg.top_candidates(np.stack([ev[p] for g in sel.gallery for p in g.probes]))
    vids = np.array([g.identity for g in sel.gallery for p in g.probes])
    VAL.update(score=vscore, correct=vtop == vids, t=thr)
    log(network, f"validation TPIR from cache {np.mean((vscore >= thr) & (vtop == vids)):.4f}")
    emb = {}
    for c, items in conditions.items():
        emb[c] = {
            p: None if v is None else model.embed(face_crop(detector, v[0], v[1], crop, size))
            for p, v in items.items()
        }
    for c, items in icc_images.items():
        emb["icc_" + c] = {}
        for p, px in items.items():
            d = benchmark_face(detector.detect(px, max_side=MAX_DETECTION_SIDE), px.shape)
            emb["icc_" + c][p] = (
                None if d is None else model.embed(face_crop(detector, px, d, crop, size))
            )
    # Parity: native embeddings against the evaluation's cache.
    cached = cache.load(model.key, pipeline.with_crop(crop).id, model.dimension)
    parity = [
        float(emb["native"][p] @ cached[image_key(pngs[p])])
        for p in images
        if image_key(pngs[p]) in cached
    ]
    m = {
        "threshold": thr,
        "crop": crop,
        "provider": model.key.provider,
        "cache_parity": {"n": len(parity), "min_cos": min(parity) if parity else None},
        "cos": {},
        "flips": {},
    }
    for c in emb:
        if c == "native" or c.startswith("icc_"):
            continue
        m["cos"][c] = cos_stats(emb["native"], emb[c], list(images))
    if icc_images["srgb"]:
        g = list(gallery_photo.values())
        m["cos"]["icc_p3_as_srgb"] = cos_stats(emb["icc_srgb"], emb["icc_p3_as_srgb"], g)
        m["cos"]["icc_p3_converted"] = cos_stats(emb["icc_srgb"], emb["icc_p3_converted"], g)
        m["cos"]["icc_srgb_jpeg_vs_native"] = cos_stats(emb["native"], emb["icc_srgb"], g)
    # A: native gallery, probes under the condition. B: gallery under the condition, native probes.
    nat = emb["native"]
    base_m = score(nat, nat, mated, thr)
    base_n = score(nat, nat, non_mated, thr)
    m["base"] = {
        "mated_tp": sum(a and c for _, c, a in base_m.values()),
        "mated": len(base_m),
        "non_mated_fp": sum(a for _, _, a in base_n.values()),
        "non_mated": len(base_n),
    }
    for c in CONDITIONS[1:]:
        m["flips"][f"A:{c}"] = {
            "mated": flips(base_m, score(nat, emb[c], mated, thr), "mated"),
            "non_mated": flips(base_n, score(nat, emb[c], non_mated, thr), "non"),
        }
        m.setdefault("expected", {})[c] = expected(np.array(DELTAS["mated"]))
        if all(emb[c][p] is not None for p in gallery_photo.values()):
            m["flips"][f"B:{c}"] = {
                "mated": flips(base_m, score(emb[c], nat, mated, thr), "mated"),
                "non_mated": flips(base_n, score(emb[c], nat, non_mated, thr), "non"),
            }
        else:
            missing = sum(emb[c][p] is None for p in gallery_photo.values())
            m["flips"][f"B:{c}"] = f"{missing} gallery photos had no usable face"
    if icc_images["srgb"]:
        ref_m, ref_n = (
            score(emb["icc_srgb"], nat, mated, thr),
            score(emb["icc_srgb"], nat, non_mated, thr),
        )
        for c in ["icc_p3_as_srgb", "icc_p3_converted"]:
            m["flips"][f"B:{c}"] = {
                "mated": flips(ref_m, score(emb[c], nat, mated, thr), "mated"),
                "non_mated": flips(ref_n, score(emb[c], nat, non_mated, thr), "non"),
            }
            m["expected"][c] = expected(np.array(DELTAS["mated"]))
    out["models"][network] = m
    log(network, "base", m["base"], "parity", m["cache_parity"])
    for c, v in m["cos"].items():
        log(
            " ",
            network,
            f"{c:22s} n={v['n']:3d} mean={v['mean']:.4f} p5={v['p5']:.4f} min={v['min']:.4f}",
        )
    for c, v in m["expected"].items():
        log(
            " ",
            network,
            f"expected on validation {c:22s} lost={v['lost']:.4%} gained={v['gained']:.4%}",
        )
    for k, v in m["flips"].items():
        if isinstance(v, str):
            log(" ", network, k, v)
            continue
        mm, nn = v["mated"], v["non_mated"]
        log(
            " ",
            network,
            f"{k:24s} mated n={mm['n']} base={mm['base_positive']} lost={mm['lost']} gained={mm['gained']} "
            f"dmean={mm['mean_dscore']:+.4f} p5={mm['p5_dscore']:+.4f} p95={mm['p95_dscore']:+.4f} | "
            f"non-mated n={nn['n']} fp={nn['base_positive']} +{nn['gained']} -{nn['lost']}",
        )

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
log("done")
```

</details>
