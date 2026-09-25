# Data and detection session, 25 September 2026

The record of the seventh working session, the second build session. It merged PR #33 (scaffold, #23) at Shikhar's request, then implemented ticket #24 (Data fetch, weights and YuNet detection) with `/implement` and merged it as PR #34. Read the scaffold session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- #23 and #24 are closed; main holds both.
- #24 unblocks #25 (EDA, figures and report scaffold) and #26 (Recognition models and LFW verification). They share no files beyond `pyproject.toml` and can run in parallel sessions. #27 (CelebA open set) waits on both.
- On this machine `data/raw` and `models/weights` are populated and verified. `ryuk data fetch` and `ryuk weights fetch` report "0 fetched" on a rerun.

## What exists now

- **`ryuk.fetch`.**
  - `pinned.py`: `PinnedFile` (URL, relative path, size, checksum), `fetch_pinned`, and `write_into_place`, which fills a `.part` file and renames it only once the writer returns.
  - `http.py`: `open_url`, and `RangeFile`, a seekable remote file with prefetched spans.
  - `lfw.py` and `celeba.py` hold the pinned sources. `FetchError` lives in `ryuk.fetch`; every failure the CLI can hit is one.
- **`ryuk.weights`.** `YUNET`, `SFACE`, `ARCFACE`, `FACENET` and `WEIGHTS`. `Weights.path(settings.weights_dir)` is where each file lives. ArcFace is an `ExtractedFile` taken from `buffalo_l.zip`; the zip is deleted afterwards.
- **`ryuk.detector`.**
  - `Detector(weights).detect(bgr_image)` returns `Detection(box, landmarks, score)`, best score first.
  - `usable_faces`, `centre_most`, `benchmark_face`, and `Detector.align(image, detection, 112 | 160)`.
  - `MIN_USABLE_FACE_SIZE = 40` is provisional: #25 fixes it by #17's rule.
- **Settings.** `RYUK_DATA_DIR` (default `data/raw`) and `RYUK_WEIGHTS_DIR` (default `models/weights`). Both are relative to the working directory, so run from the repository root.
- **On-disk layout.**
  - LFW: `data/raw/lfw/{pairs.txt, pairsDevTrain.txt, pairsDevTest.txt, archives/lfw-funneled.tgz, lfw_funneled/<name>/<name>_NNNN.jpg}`. `lfw_funneled/` also holds 11 `pairs_*.txt` files, so list identities by folder.
  - CelebA images: `data/raw/celeba/parquet/img_align+identity+attr/{valid,test}-0000N-of-00003.parquet`, holding PNG bytes in `image.bytes`.
  - CelebA labels: `data/raw/celeba/metadata/celeba_meta.parquet`, one row per image with `split, shard, row_group, row_in_file, path, celeb_id` and the 40 attributes. `shard`, `row_group` and `row_in_file` locate each image without scanning.

## Patterns later tickets should copy

- **Pinned files.** Describe a new downloaded artefact as a `PinnedFile` and go through `fetch_pinned` or `write_into_place`. Never write a final path directly.
  - A derived file whose bytes depend on library versions is pinned by a content digest instead. `labels_digest` is the example.
- **Fetch tests.** Use the `file_server` fixture in `tests/conftest.py`: a loopback HTTP server with byte ranges, redirects, `honour_ranges` and `cut_short`. Build small tarballs, zips and Parquet files inside the test.
- **Detector tests.** Use the real YuNet at `tests/fixtures/face_detection_yunet_2026may.onnx` (its sha256 is tested against the pin) and `tests/fixtures/astronaut.jpg`, a public-domain face. Compose multi-face scenes in numpy; `tests/test_detector.py` shows sizes YuNet reliably finds.
- **CLI.** Fetch commands log through `configure_logging()` (JSON lines), like `serve`.

## Things that bit

- **`alignCrop` needs a recognizer.** It is a method of `cv2.FaceRecognizerSF`, and `create()` refuses an empty model path. It never runs the network, so the Detector builds one from the YuNet ONNX. Keep it that way, or unit tests need the 38 MB SFace file.
- **YuNet behaviour.**
  - `detect()` returns `None`, not an empty array, for no faces.
  - Landmarks are labelled by image position, not anatomy: "right eye" is the eye on the image's left, even in a mirrored frame. That matters for the live monitor if the client mirrors the webcam.
  - YuNet pulls landmarks towards upright on rotated faces: at 15° it reports about 1 px of eye tilt against 11 px true. #25's head-pose estimate from landmarks will be biased low.
  - At the 0.9 threshold it finds faces down to boxes of about 25-31 px, with scores barely over 0.9. Even a clean 512 px portrait scores only 0.93.
- **Hugging Face range reads cost about 0.6 s each** (xet CDN time to first byte). One request per column chunk took 56 s per shard. The fix:
  - Resolve the redirect once.
  - Fetch each row group's label span in parallel (16 threads).
  - Let pyarrow read from the prefetched spans.

  The full label rebuild takes about 5 minutes.
- **Coverage can't see pyarrow's reads.** pyarrow reads Python file objects from its own C++ I/O threads, so coverage doesn't show code run there. `RangeFile` has direct tests for that reason.
- **`http.client` doesn't raise on a dropped body.** A connection cut mid-body returns a short read, not `IncompleteRead`. The size check catches it as `ChecksumMismatchError`.
- **Media URLs.** GitHub's `raw.githubusercontent.com` serves opencv_zoo's LFS pointer, not the model. Use `media.githubusercontent.com`, pinned to commit `47534e2`.

## Decisions and open items

- **Not reproduced by the fetch:**
  - The optional deep-funneled LFW tarball (Wayback Machine). #24 names only figshare.
  - The spike's summary and log files (`shape.json`, `fetch_log.json`, `source.json`, CelebA `README.md`). They are still on disk locally.
- **Review suggestions declined** (easily reversible, noted in PR #34):
  - `Detector` keeps YuNet's `score_threshold`, `nms_threshold` and `top_k` as keyword arguments, in case #25 tunes them.
  - `Weights.name` and `Weights.path()` stay for the service's startup hash check.
- PR #33 was merged with a merge commit, not squashed, to keep its per-commit history. #34 followed the same pattern.

## Resuming

Start a fresh session with `/implement` on #25 or #26, or run both as two sessions. Working rules as before:
- one ticket per session, on a `ShikharJohari/<n>-<slug>` branch;
- a PR that says `Closes #N`;
- nothing merged without Shikhar;
- role agents only, never Haiku;
- never Playwright locally.

#26 should build the recognition model interface on `Detector.align` and `ryuk.weights`. #25 should replace `MIN_USABLE_FACE_SIZE` with the measured value.
