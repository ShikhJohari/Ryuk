# Open-source recognition models and detectors for Ryuk

Research for #3 (part of #1). Target machine: Apple M4, 16 GB, no discrete GPU, macOS,
Python 3.12 managed by `uv`. Everything below was checked against primary sources
(package READMEs, source code, and license files) on 2026-09-22, and wheel availability
was confirmed by downloading the actual cp312/macosx-arm64 files with
`pip download --only-binary=:all: --python-version 3.12 --platform macosx_14_0_arm64`.

## Recommendation

Three recognition models, one shared detector.

**SFace** (OpenCV Zoo, `cv2.FaceRecognizerSF`). Apache 2.0, ships inside the plain
`opencv-python` wheel with no extra install, 128-d embedding, 0.9940 accuracy on the
zoo's own LFW-style pair evaluation. It is the only candidate with zero extra
dependencies beyond OpenCV itself, which matters on a laptop with no GPU: one wheel,
one runtime, no ONNX Runtime session to manage.

**InsightFace ArcFace, `buffalo_l` pack** (`insightface` + `onnxruntime`). MIT-licensed
library code, but the bundled ONNX weights are non-commercial research use only, a
restriction Ryuk should treat as a hard constraint until watchlist deployment terms are
settled. In exchange for that restriction, `buffalo_l` is the strongest model here:
99.83 LFW, 91.25 on the harder MR-ALL multi-racial benchmark, 512-d embeddings, and it
runs on `onnxruntime`'s CoreMLExecutionProvider on Apple Silicon out of the box. If the
non-commercial term turns out to be a blocker, `buffalo_s` is a drop-in same-package
fallback at lower accuracy (99.70 LFW) and lower cost.

**facenet-pytorch, `InceptionResnetV1(pretrained='vggface2')`**. MIT license, no
non-commercial strings attached, 512-d L2-normalized embeddings, 0.9965 on the LFW
number the repo itself cites from davidsandberg/facenet. This is the third model the
ADR asks research to pick, and it earns the slot mainly by having no licensing
asterisk and a plain `pip install`. AdaFace, MagFace, and GhostFaceNets were all
considered and rejected for now (see below): none ship ONNX weights from their
official repos, only PyTorch checkpoints or Keras `.h5` files on Google Drive, which
fails the "runs as it ships" bar this ticket is checking against.

**Detector: YuNet** (`cv2.FaceDetectorYN`, OpenCV Zoo, MIT). It ships inside
`opencv-python` too, so picking it costs nothing beyond what SFace already needs. It
outputs 5-point landmarks (right eye, left eye, nose tip, right mouth corner, left
mouth corner) per detected face, and, this is the key finding, the OpenCV source for
`FaceRecognizerSF::alignCrop` warps those 5 points to the exact same destination
template `{(38.29,51.70),(73.53,51.50),(56.03,71.74),(41.55,92.37),(70.73,92.20)}`
that InsightFace's `face_align.norm_crop` (used by both SCRFD and RetinaFace) warps to.
Same five magic numbers, same point order, in both codebases' source. That means
YuNet's landmarks can feed SFace's `alignCrop` and InsightFace's `norm_crop` without
running two detectors. facenet-pytorch is the odd one out: its bundled MTCNN detects
5 points too, but the library's own `extract_face` does a box-crop-and-resize, not a
landmark warp, so wiring YuNet's landmarks into the facenet-pytorch path is an
integration task for Ryuk to write and validate, not something either library already
does. SCRFD and RetinaFace (both bundled inside `insightface`) also produce compatible
5-point landmarks, but adopting either as the shared detector would mean running
`insightface` just to detect faces before ever calling SFace or facenet-pytorch, an
extra heavyweight dependency for no accuracy benefit over YuNet on a CPU-only laptop.
MediaPipe's Face Detector was ruled out for the shared role because it emits 6
keypoints (adds two ear tragions), not 5, so it does not slot into either alignment
template without a mapping step of its own.

## Comparison table

| Candidate | Type | Install | License | Embedding dim | Input size | LFW accuracy | Similarity metric |
|---|---|---|---|---|---|---|---|
| SFace | recognition | `opencv-python` (cp37-abi3, runs on cp312) | Apache 2.0 | 128 | 112x112 | 0.9940 | cosine (thr 0.363) or norm-L2 (thr 1.128) |
| ArcFace `buffalo_l` | recognition | `insightface` + `onnxruntime` (cp312 arm64) | non-commercial research only | 512 | 112x112 | 0.9983 | cosine |
| ArcFace `buffalo_s` | recognition | `insightface` + `onnxruntime` (cp312 arm64) | non-commercial research only | 512 | 112x112 | 0.9970 | cosine |
| facenet-pytorch (vggface2) | recognition | `facenet-pytorch` + `torch` (cp312 arm64) | MIT | 512 | 160x160 | 0.9965 | L2 distance on L2-normalized vectors (cosine-equivalent) |
| facenet-pytorch (casia-webface) | recognition | same package | MIT | 512 | 160x160 | 0.9905 | same |
| AdaFace (R100, WebFace12M) | recognition, not recommended | PyTorch checkpoint only, no ONNX, Google Drive | MIT | 512 (typical) | 112x112 | 0.9982 | cosine |
| MagFace (iResNet100) | recognition, not recommended | PyTorch checkpoint only, no ONNX, Google/Baidu Drive | Apache 2.0 | 512 (typical) | 112x112 | 0.9983 (paper) | cosine, magnitude-aware |
| GhostFaceNets (V1/V2) | recognition, not recommended | Keras `.h5` only, no ONNX, Google Drive | MIT | 512 | 112x112 | up to 0.9977 | cosine |
| YuNet | detector | `opencv-python` (same wheel as SFace) | MIT | n/a | dynamic, trained on 10x10-300x300 faces | n/a | n/a |
| SCRFD (`buffalo_l`'s detector) | detector | `insightface` + `onnxruntime` | non-commercial research only | n/a | 640x640 default | n/a | n/a |
| RetinaFace (insightface) | detector | `insightface` + `onnxruntime` | non-commercial research only | n/a | varies | n/a | n/a |
| MediaPipe Face Detector | detector | `mediapipe` (macosx_11 arm64 wheel, runs on 14) | Apache 2.0 | n/a | 128x128 | n/a | n/a |

## Recognition models: evidence

### SFace (OpenCV Zoo)

- Model files and license: three ONNX variants (fp32, int8, int8 block-quantized),
  "All files in this directory are licensed under Apache 2.0 License."
  ([README](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface))
- Accuracy on the zoo's own evaluation: SFace 0.9940, block-quantized 0.9942,
  quantized 0.9932 (same README).
- Input size and embedding dimension confirmed by loading the actual ONNX graph
  (`face_recognition_sface_2021dec.onnx`): input `data` is `[1, 3, 112, 112]`, output
  `fc1` is `[1, 128]`.
- Alignment: `FaceRecognizerSF::alignCrop` warps 5 input points to a fixed 112x112
  destination template via a similarity transform, then crops
  ([opencv/opencv face_recognize.cpp](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_recognize.cpp)).
- Similarity metric and thresholds, straight from the zoo's own `sface.py`: cosine
  similarity, threshold 0.363, match if score is at or above threshold; norm-L2
  distance, threshold 1.128, match if distance is at or below threshold
  ([sface.py](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py)).

### InsightFace ArcFace (`buffalo_l` / `buffalo_s`)

- Library license: MIT, "no limitation for both academic and commercial usage."
  Model license: "the pretrained models provided with this library are for
  non-commercial research purposes only," explicitly naming `buffalo_l` and directing
  commercial users to `recognition-oss-pack@insightface.ai`
  ([insightface README](https://raw.githubusercontent.com/deepinsight/insightface/master/README.md),
  lines 32-44).
- Pack composition and accuracy, from the model zoo table: `buffalo_l` pairs
  SCRFD-10GF detection with ResNet50@WebFace600K recognition, 326MB, 99.83 LFW, 91.25
  MR-ALL; `buffalo_s` pairs SCRFD-500MF with MBF@WebFace600K, 159MB, 99.70 LFW, 71.87
  MR-ALL
  ([python-package/docs/model_zoo.md](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/docs/model_zoo.md)).
- Embedding dimension: the recognition training config used for these backbones sets
  `config.embedding_size = 512` by default
  ([arcface_torch configs](https://raw.githubusercontent.com/deepinsight/insightface/master/recognition/arcface_torch/configs/wf42m_pfc03_32gpu_r100.py)).
- Alignment and similarity metric, from `arcface_onnx.py`: faces are cropped with
  `face_align.norm_crop(img, landmark=face.kps, image_size=112)`, and similarity is
  plain cosine similarity (`np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))`)
  ([arcface_onnx.py](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/model_zoo/arcface_onnx.py)).
  `norm_crop` warps 5 landmarks to `arcface_dst`
  `= [[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]]`
  at 112x112
  ([face_align.py](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/utils/face_align.py)).
  These are the same five points, in the same order, that OpenCV's `alignCrop` uses.
- Wheel availability: `pip download --only-binary=:all: --python-version 3.12
  --platform macosx_14_0_arm64 insightface` resolves `insightface-2.0-py3-none-any.whl`
  (pure Python, depends on `onnxruntime`, which resolves
  `onnxruntime-1.30.0-cp312-cp312-macosx_14_0_arm64.whl`). Checked directly against
  PyPI on 2026-09-22.
- Apple Silicon note: onnxruntime's official macOS wheels build in the CoreML
  execution provider (`pip install onnxruntime` on macOS builds with `--use_coreml`),
  requiring macOS 10.15+
  ([onnxruntime CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)).

### facenet-pytorch

- License: MIT
  ([LICENSE.md](https://raw.githubusercontent.com/timesler/facenet-pytorch/master/LICENSE.md)).
- Pretrained models: `20180402-114759` (VGGFace2, 107MB, LFW 0.9965) and
  `20180408-102900` (CASIA-Webface, 111MB, LFW 0.9905), both auto-downloaded on first
  use; the README attributes these LFW numbers to the original
  davidsandberg/facenet repo, not a facenet-pytorch re-evaluation
  ([README](https://raw.githubusercontent.com/timesler/facenet-pytorch/master/README.md)).
- Embedding dimension: `InceptionResnetV1` sets `self.last_linear =
  nn.Linear(1792, 512, bias=False)`, and the forward pass L2-normalizes the output
  (`F.normalize(x, p=2, dim=1)`) unless `classify=True`
  ([inception_resnet_v1.py](https://raw.githubusercontent.com/timesler/facenet-pytorch/master/models/inception_resnet_v1.py)).
- Input size: both pretrained models were trained on 160x160 images (README).
- Alignment: bundled `MTCNN` detects 5-point landmarks (`detect(img, landmarks=True)`
  returns boxes, probabilities, and points), but the default `extract_face` /
  `crop_resize` path does a box crop plus resize to 160x160, not a landmark-based
  similarity warp
  ([mtcnn.py](https://raw.githubusercontent.com/timesler/facenet-pytorch/master/models/mtcnn.py),
  [detect_face.py](https://raw.githubusercontent.com/timesler/facenet-pytorch/master/models/utils/detect_face.py)).
  This is a real difference from SFace and ArcFace's alignment convention, not an
  inference: it means facenet-pytorch's published accuracy numbers assume its own
  box-crop preprocessing, and swapping in a different alignment (e.g. YuNet's 5-point
  warp) is untested territory that Ryuk would need to validate before trusting it.
- Wheel availability: `facenet-pytorch-2.6.0-py3-none-any.whl` (pure Python) depends
  on `torch`, which resolves `torch-2.14.0-cp312-cp312-macosx_14_0_arm64.whl`. Both
  confirmed present on PyPI for cp312/macosx-arm64 on 2026-09-22.

### AdaFace, MagFace, GhostFaceNets (not recommended for now)

All three have credible published accuracy (AdaFace R100/WebFace12M: 0.9982 LFW;
MagFace iResNet100: 0.9983 LFW per the
[CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.pdf);
GhostFaceNetV1/V2: up to 0.9977 LFW per the
[repo README](https://raw.githubusercontent.com/HamadYA/GhostFaceNets/main/README.md)),
and permissive licenses (AdaFace MIT, MagFace Apache 2.0, GhostFaceNets MIT, all
confirmed via each repo's GitHub license API). None clears this ticket's practical bar
though:

- AdaFace ([mk-minchul/AdaFace](https://raw.githubusercontent.com/mk-minchul/AdaFace/master/README.md))
  distributes only PyTorch `.ckpt` files on Google Drive, no ONNX export in the
  official repo. Alignment uses MTCNN to 112x112 BGR crops.
- MagFace ([IrvingMeng/MagFace](https://raw.githubusercontent.com/IrvingMeng/MagFace/main/README.md))
  distributes PyTorch checkpoints on Google Drive/Baidu Drive, no official ONNX; a
  community conversion attempt exists but is an open, unresolved issue
  ([issue #32](https://github.com/IrvingMeng/MagFace/issues/32)). Alignment is
  explicitly the same insightface 5-point, 112x112 template.
- GhostFaceNets ([HamadYA/GhostFaceNets](https://raw.githubusercontent.com/HamadYA/GhostFaceNets/main/README.md))
  distributes Keras `.h5` weights on Google Drive/GitHub releases, no ONNX in the
  official repo.

None of these fail on licensing or accuracy. They fail on distribution: converting a
checkpoint to ONNX, or trusting a third-party conversion, is exactly the kind of
extra engineering effort the ADR rules out ("Ryuk uses open-source pretrained
recognition models as they ship... and never fine-tunes them"). Revisit if any of
these projects publish an official ONNX export.

## Detectors: evidence

### YuNet (`cv2.FaceDetectorYN`)

- License: MIT, all files in the OpenCV Zoo directory
  ([README](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).
- Ships inside the base `opencv-python` wheel: unzipping
  `opencv_python-5.0.0.93-cp37-abi3-macosx_13_0_arm64.whl` and reading
  `cv2/__init__.pyi` shows both `FaceDetectorYN` and `FaceRecognizerSF` are present,
  no `opencv-contrib-python` needed. The wheel's `cp37-abi3` tag is Python's stable
  ABI, so it installs and runs under Python 3.12 without a cp312-specific build.
- Output format: from the zoo's own `demo.py`, each detection row is `[bbox(4),
  landmarks(10), score(1)]`, i.e. 4 box values, 5 landmark points (right eye, left
  eye, nose tip, right mouth corner, left mouth corner) as x,y pairs, then a
  confidence score
  ([demo.py](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/demo.py)).
- Input handling: default input size 320x320, configurable per-frame via
  `setInputSize`; trained to detect faces roughly 10x10 to 300x300 pixels in that
  frame (zoo README).
- Compatibility with SFace: the SFace demo feeds YuNet's raw `bbox+landmarks` slice
  directly into `recognizer.match`, which calls `alignCrop` internally, with no
  intermediate reformatting
  ([demo.py](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/demo.py)).
  The two projects designed this pairing together, so it is safe to rely on.
- Compatibility with InsightFace ArcFace: not demonstrated in any official demo (the
  two projects don't reference each other), but the destination template is
  identical byte-for-byte to `arcface_dst`, see the SFace section above. This is
  strong circumstantial evidence, not a tested pairing; verify with a visual overlay
  before shipping.

### SCRFD and RetinaFace (via `insightface`)

Both are internal to the `insightface` package and produce 5-point landmarks through
the same `distance2kps` decoding path (`use_kps = True` branches in
[scrfd.py](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/model_zoo/scrfd.py)
and
[retinaface.py](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/model_zoo/retinaface.py)),
consumed by the same `face_align.norm_crop`. `buffalo_l` bundles SCRFD-10GF as its
detector by default. Using either as Ryuk's shared detector would mean depending on
`insightface` (and its non-commercial model license) just to detect faces before ever
calling SFace or facenet-pytorch, for no landmark-quality benefit over YuNet on CPU.

### MediaPipe Face Detector

Apache 2.0 license (confirmed via the `google-ai-edge/mediapipe` repo's GitHub
license API). The BlazeFace-based Face Detector task outputs 6 keypoints per face:
left eye, right eye, nose tip, mouth center, and left and right ear tragion, at a
128x128 model input
([MediaPipe Face Detector docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)).
Six points, not five, and a different point set (adds tragions, uses a single mouth
point instead of two corners), so it does not map onto either the OpenCV or
InsightFace 5-point template without an explicit remapping/subsetting step. Wheel
`mediapipe-1.0.1-py3-none-macosx_11_0_arm64.whl` is available for Python 3.12 arm64
(the `macosx_11_0` platform tag is forward-compatible with macOS 14).

## Install notes

Wheel availability was checked directly against PyPI, not inferred from
changelogs, using:

```
pip download --no-deps --only-binary=:all: --python-version 3.12 \
  --platform macosx_14_0_arm64 -d <scratch-dir> <package>
```

Results on 2026-09-22, all landed successfully in a scratch directory under
`/private/tmp`:

| Package | Resolved wheel |
|---|---|
| opencv-python | `opencv_python-5.0.0.93-cp37-abi3-macosx_13_0_arm64.whl` |
| onnxruntime | `onnxruntime-1.30.0-cp312-cp312-macosx_14_0_arm64.whl` |
| insightface | `insightface-2.0-py3-none-any.whl` |
| facenet-pytorch | `facenet_pytorch-2.6.0-py3-none-any.whl` |
| torch | `torch-2.14.0-cp312-cp312-macosx_14_0_arm64.whl` |
| mediapipe | `mediapipe-1.0.1-py3-none-macosx_11_0_arm64.whl` |
| onnx | `onnx-1.23.0-cp312-abi3-macosx_13_0_universal2.whl` |
| numpy | `numpy-2.5.3-cp312-cp312-macosx_14_0_arm64.whl` |
| scikit-image | `scikit_image-0.26.0-cp312-cp312-macosx_11_0_arm64.whl` |
| scipy | `scipy-1.18.1-cp312-cp312-macosx_14_0_arm64.whl` |

Everything Ryuk needs for SFace, InsightFace ArcFace, facenet-pytorch, and YuNet
installs from prebuilt wheels on this machine; no source builds required.

Model weight downloads are separate from the pip packages and worth planning for:

- SFace ONNX files come from the OpenCV Zoo git repository directly (small, a few
  tens of MB), not from PyPI.
- `insightface`'s `FaceAnalysis(name='buffalo_l')` auto-downloads the model pack
  (326MB) on first use from InsightFace's own model-zoo GitHub release; this download
  is the point at which the non-commercial license attaches.
- facenet-pytorch auto-downloads its `.pt` checkpoints (around 110MB) from GitHub
  releases on first `InceptionResnetV1(pretrained=...)` call.

None of the three recommended models require an account, an API key, or a manual
Google Drive click, which is one more mark against AdaFace, MagFace, and
GhostFaceNets: all three route through Google Drive links that are not guaranteed to
survive quota limits or link rot.
