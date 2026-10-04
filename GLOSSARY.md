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
A photo of a person of interest, kept as the lasting reference from which that person's embeddings are made. Contains exactly one face large enough to use; smaller faces, such as a bystander in the background, are ignored. A person of interest's last enrolled photo cannot be deleted.
_Avoid_: Reference image, sample, template

**Enrollment**:
Accepting a photo for a person of interest and producing its embedding under every recognition model. A photo with no usable face, or more than one, is rejected.
_Avoid_: Registration, indexing, training

**Removal**:
Taking a person of interest off the watchlist while keeping their enrolled photos and sightings. Reversible.
_Avoid_: Delete, archive, deactivate

**Purge**:
Permanently erasing a person of interest with their enrolled photos, embeddings and sightings, and clearing them as runner-up on anyone else's sightings. Not reversible.
_Avoid_: Delete, hard delete, remove

**Operator**:
The one person running the live monitor and managing the watchlist.
_Avoid_: User, admin, analyst

### Recognition

**Live monitor**:
The screen that runs recognition on the operator's webcam and reports sightings as they happen. Only one runs at a time.
_Avoid_: Feed, surveillance, camera view

**Detection**:
A face found in a frame, with its bounding box and landmarks. Says nothing about who it is.
_Avoid_: Recognition, hit

**Usable face**:
A detection large enough for its embedding to be trusted. Only a usable face is ever scored against the watchlist; a smaller detection is shown live as too small, ignored at enrollment, and excluded from evaluation.
_Avoid_: Valid face, good face

**Embedding**:
A fixed-length vector produced by a recognition model from one aligned face. Two embeddings from the same model can be compared; embeddings from different models cannot.
_Avoid_: Feature vector, fingerprint, encoding

**Active model**:
The one recognition model the live monitor uses at a given time. Only a recognition model with a threshold from evaluation can be active. Switching it needs no re-enrollment.
_Avoid_: Current model, default model

**Candidate**:
A person of interest ranked by similarity to a detection. Only the top candidate can become a match.
_Avoid_: Suspect, guess, nearest neighbour

**Runner-up**:
The second-ranked candidate at a sighting's best match. Recorded with the sighting, never shown live beside a face.
_Avoid_: Second match, alternative

**Match**:
A usable face whose top candidate scores at or above the active model's threshold. Names exactly one person of interest.
_Avoid_: Hit, alert, positive

**No match**:
A usable face whose top candidate scores below the threshold, or that has no candidate because nobody is on the watchlist. Shown with its score (none when nobody is on the watchlist) but never with a name, and never logged.
_Avoid_: Unknown, stranger, miss

**Confirmation**:
The moment a person of interest's matches in the live monitor are steady enough to open a sighting: matched in at least half the frames processed over a short window. Matches before confirmation are shown live but not logged.
_Avoid_: Verification (an evaluation term), debounce

**Sighting**:
One person of interest seen continuously over a span of time in the live monitor under one active model, however many frames or detections that covers. Opens at confirmation and ends once the person has gone unmatched for a few seconds or the live monitor stops. Keeps the face crop of the best match, the runner-up candidate, and the model and threshold that produced it; never the whole frame. Switching the active model ends every open sighting.
_Avoid_: Detection log, event, alert, hit

**Match score**:
The single number that says how well a detection fits a candidate, computed from the detection's embedding and the candidate's enrolled embeddings by the recognition model's live rule. Cosine similarity to the best enrolled photo unless evaluation showed another rule is better.
_Avoid_: Confidence, probability, similarity (when a learned rule is live)

**Threshold**:
The cut-off on a recognition model's match score at or above which a detection counts as a match. Fixed per recognition model from evaluation; never copied from a model's README and never adjusted by the operator.
_Avoid_: Sensitivity, confidence

**Same-person threshold**:
The cut-off on a recognition model's cosine under which a photo added to a person of interest warns that it may not be them, the photo's best cosine to their enrolled photos being compared with it. A one-to-one cut-off, lower than the threshold, which is one-to-many; fixed per recognition model from evaluation at a stated false-accept rate on impostor pairs.
_Avoid_: 1:1 threshold, verification threshold, warning threshold

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

**Misidentification**:
A mated probe whose top candidate is the wrong identity yet scores at or above the threshold. The error that matters most on a watchlist; its live counterpart is a match naming the wrong person of interest.
_Avoid_: False positive, wrong match

**Held-out identity**:
An identity never enrolled in an evaluation's gallery, whose faces only ever appear as non-mated probes. The evaluation's stand-in for a stranger.
_Avoid_: Unknown, impostor, distractor

**Impostor pair**:
A gallery identity's enrolled photos and a probe of another identity, gallery or held out, compared one-to-one. The probe's identity is not a held-out identity as such, and is never called an impostor on its own.
_Avoid_: Non-mated pair, negative pair, distractor

**Mated pair**:
A gallery identity's enrolled photos and one of its own probes, compared one-to-one: the same person.
_Avoid_: Genuine pair, positive pair

**Draw**:
One set of gallery and held-out identities for an open-set evaluation. Draws share no identities: the validation draw sets thresholds, the test draw reports results.
_Avoid_: Split, fold (a fold is LFW's)

**Verification**:
Deciding whether two faces are the same person. One-to-one. Measured on labelled pairs.
_Avoid_: Matching, authentication

**Identification**:
Deciding which enrolled person, if any, a face belongs to. One-to-many.
_Avoid_: Classification, search

**Open set**:
An identification setting where the face may belong to nobody enrolled, and saying "nobody" is a valid and measured answer.

**Recognition model**:
A pretrained network that maps an aligned face to an embedding, identified by its exact weights, the execution provider it runs on and the crop that cuts its faces: the same network with other weights, on another provider or under another crop is a different recognition model. The model key names the network, provider and weights; the crop is recorded beside it, and a change of crop rebuilds the enrolled embeddings. Ryuk compares several; none are trained here.
_Avoid_: Embedder, backbone, encoder
