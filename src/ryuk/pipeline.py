"""The version of the face pipeline, shared by every store of embeddings.

An embedding is keyed by its recognition model (network, weights, provider), the detector's
weights, the minimum usable face size and the crop. Anything else between an image and its
embedding that can move it has no key of its own: the normalisation in `embed`, the alignment,
YuNet's score and NMS thresholds, the format a model is compiled to. Bump PIPELINE_VERSION with any
such change. The benchmark embedding cache then misses and embeds again, and the app database
rebuilds every stored embedding from its enrolled photo at the next start.
"""

from typing import Final

PIPELINE_VERSION: Final = 1
