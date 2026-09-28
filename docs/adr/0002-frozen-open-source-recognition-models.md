---
status: accepted
---

# Frozen open-source recognition models, no fine-tuning

The available compute when this was decided was one Apple M4 laptop with 16 GB and no discrete GPU. Ryuk uses open-source pretrained recognition models as they ship (SFace from the OpenCV zoo, ArcFace from InsightFace, and FaceNet VGGFace2 from facenet-pytorch, chosen in research) and never fine-tunes them. The learning Ryuk does is on top of frozen embeddings: classifiers, threshold calibration, and open-set decision rules, all of which train in seconds on a CPU.

## Consequences

The innovation story is "which open model, and what to learn on top of it", not "how to train a better model". A GPU-backed fine-tuning experiment is out of scope and would be a new effort.
