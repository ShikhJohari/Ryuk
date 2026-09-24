# Ryuk

Face recognition against an enrolled watchlist, with the recognition models measured on public benchmarks before they are trusted in the live monitor.

## Language

### Watchlist

**Person of interest**:
A named individual Ryuk can match: a name and one or more enrolled photos, nothing else. One record per person; names need not be unique.
_Avoid_: Criminal, suspect, target, subject

**Watchlist**:
The persons of interest the live monitor currently compares faces against. A removed person of interest is not on it.
_Avoid_: Database, registry, gallery (gallery is an evaluation term)

**Enrolled photo**:
A photo of a person of interest, kept as the lasting reference from which that person's embeddings are made. Contains exactly one face. A person of interest's last enrolled photo cannot be deleted.
_Avoid_: Reference image, sample, template

**Enrollment**:
Accepting a photo for a person of interest and producing its embedding under every recognition model. A photo with no face or more than one face is rejected.
_Avoid_: Registration, indexing, training

**Removal**:
Taking a person of interest off the watchlist while keeping their enrolled photos and sightings. Reversible.
_Avoid_: Delete, archive, deactivate

**Purge**:
Permanently erasing a person of interest with their enrolled photos, embeddings and sightings. Not reversible.
_Avoid_: Delete, hard delete, remove

**Operator**:
The one person running the live monitor and managing the watchlist.
_Avoid_: User, admin, analyst

### Recognition

**Live monitor**:
The screen that runs recognition on the operator's webcam and reports sightings as they happen.
_Avoid_: Feed, surveillance, camera view

**Detection**:
A face found in a frame, with its bounding box and landmarks. Says nothing about who it is.
_Avoid_: Recognition, hit

**Embedding**:
A fixed-length vector produced by a recognition model from one aligned face. Two embeddings from the same model can be compared; embeddings from different models cannot.
_Avoid_: Feature vector, fingerprint, encoding

**Active model**:
The one recognition model the live monitor uses at a given time. Switching it needs no re-enrollment.
_Avoid_: Current model, default model

**Candidate**:
A person of interest ranked by similarity to a detection. Only the top candidate can become a match.
_Avoid_: Suspect, guess, nearest neighbour

**Match**:
A detection whose top candidate scores at or above the active model's threshold. Names exactly one person of interest.
_Avoid_: Hit, alert, positive

**No match**:
A detection whose top candidate scores below the threshold. Shown with its score but never with a name, and never logged.
_Avoid_: Unknown, stranger, miss

**Sighting**:
One person of interest seen continuously over a span of time in the live monitor under one active model, however many frames or detections that covers. Keeps the face crop of the best match, the runner-up candidate, and the model and threshold that produced it; never the whole frame. Switching the active model ends every open sighting.
_Avoid_: Detection log, event, alert, hit

**Threshold**:
The similarity cut-off at or above which a comparison counts as a match. Fixed per recognition model from evaluation; never copied from a model's README and never adjusted by the operator.
_Avoid_: Sensitivity, confidence

### Evaluation

**Identity**:
A labelled person in a benchmark dataset. Not a person of interest.
_Avoid_: Class, subject, label

**Gallery**:
The identities enrolled in one evaluation experiment. The evaluation counterpart of the watchlist.
_Avoid_: Watchlist, database

**Probe**:
A face checked against a gallery in an evaluation. Mated if its identity is in the gallery, non-mated if not.
_Avoid_: Query, test image

**Verification**:
Deciding whether two faces are the same person. One-to-one. Measured on labelled pairs.
_Avoid_: Matching, authentication

**Identification**:
Deciding which enrolled person, if any, a face belongs to. One-to-many.
_Avoid_: Classification, search

**Open set**:
An identification setting where the face may belong to nobody enrolled, and saying "nobody" is a valid and measured answer.

**Recognition model**:
A pretrained network that maps an aligned face to an embedding. Ryuk compares several; none are trained here.
_Avoid_: Embedder, backbone, encoder
