---
status: accepted
---

# Enrolled photos are the source of truth; embeddings are derived per model

Embeddings from different recognition models cannot be compared, and Ryuk compares three of them, with the live monitor switching between them. So Ryuk keeps every enrolled photo and computes its embedding under every recognition model at enrollment. Switching the active model is then instant, and adding a model later means recomputing embeddings from the photos. Enrollment costs roughly 20 ms per photo across all three models on an Apple M4 laptop, and more where every model runs on CPU, but it is paid once per photo, so there is no reason to compute embeddings lazily.

## Considered options

- Store embeddings only and block a model switch until every person of interest is re-enrolled from photos the operator supplies again. Rejected: a switch would be slow and manual.
- Store photos, but compute embeddings for a model only when it becomes active. Rejected: switching would stall the live monitor for a cost that is trivial up front.

## Consequences

Ryuk stores face photos, not just embeddings, which is the more sensitive data. Purge must erase the photos along with everything made from them, and the report's ethics section says so. Every sighting records the model and threshold that produced it, because sightings from different models are not comparable either.

## Amendment (2026-09-29, #47 Q9)

The enrolled photo Ryuk keeps is the stored copy, not the upload, and the stored copy is the source of truth. An upload is turned upright from its EXIF orientation, shrunk so that its long side is at most 2048 px, rebuilt from its pixels so that no metadata survives, and re-encoded: a JPEG or WebP at quality 95, a PNG losslessly. The upload itself is not kept. Every embedding, at enrollment and in every rebuild, is computed from the stored copy's pixels, so a rebuild reproduces the embeddings enrollment made.

The cost was measured (`docs/research/enrolled-face-scale.md`). Re-encoding a face at quality 95 moves an expected 0.01% of ArcFace's validation mated probes across its frozen threshold and 0.48% of SFace's; the mean match score moves by at most 0.002, and no non-mated probe started or stopped matching, though the note's small gallery puts non-mated scores too far below the threshold to resolve an effect on FPIR. FaceNet's figure, 1.6%, reflects its threshold sitting in a dense part of its score distribution, where even a change this small crosses many probes. Matching works on 112 px face crops (160 px for FaceNet), and the same note found that cutting one from a face five times larger is not material. Keeping the upload as sent, with its metadata stripped, was considered and not chosen.
