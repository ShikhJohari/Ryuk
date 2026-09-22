# Ryuk

Face recognition against an enrolled watchlist, with the recognition models measured on public benchmarks before they are trusted in the live monitor.

## Language

### Watchlist

**Person of interest**:
A named individual on the watchlist, with one or more enrolled photos.
_Avoid_: Criminal, suspect, target, subject

**Watchlist**:
The set of persons of interest the live monitor compares faces against.
_Avoid_: Database, gallery, registry

**Enrollment**:
Turning a person of interest's photos into stored embeddings so they can be matched.
_Avoid_: Registration, indexing, training

### Recognition

**Detection**:
A face found in a frame, with its bounding box and landmarks. Says nothing about who it is.
_Avoid_: Recognition, hit

**Embedding**:
A fixed-length vector produced by a recognition model from one aligned face. Two embeddings from the same model can be compared; embeddings from different models cannot.
_Avoid_: Feature vector, fingerprint, encoding

**Match**:
A detection whose embedding is similar enough to an enrolled embedding to name a person of interest.
_Avoid_: Hit, alert, positive

**Sighting**:
A logged match from the live monitor: who, when, which frame, and how similar.
_Avoid_: Detection log, event, alert

**Threshold**:
The similarity cut-off above which a comparison counts as a match. Chosen per model from evaluation, never copied from a model's README.

### Evaluation

**Verification**:
Deciding whether two faces are the same person. One-to-one. Measured on labelled pairs.
_Avoid_: Matching, authentication

**Identification**:
Deciding which enrolled person, if any, a face belongs to. One-to-many.
_Avoid_: Classification, search

**Open set**:
An identification setting where the face may belong to nobody on the watchlist, and saying "nobody" is a valid and measured answer.

**Recognition model**:
A pretrained network that maps an aligned face to an embedding. Ryuk compares several; none are trained here.
_Avoid_: Embedder, backbone, encoder
