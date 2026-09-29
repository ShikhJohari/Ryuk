# CelebA identity-label noise in the open-set draws

Research for #54 (audit finding M7, PR #42). Checked 2026-09-28 against `evaluation/results.json` at `7d9038c`. This is an inspection only: no code, results, thresholds or labels were changed.

Every number here was recomputed from the committed embedding cache, not re-embedded. Both draws were rebuilt with `CelebaEvaluation.prepare` and their `selection_sha256` matched `results.json`. Each model's committed threshold and operating points (TPIR and FPIR at FPIR 1% and 0.1%, both draws) were reproduced exactly before any counterfactual was computed.

The CelebA licence forbids redistribution, so this note cites images only by their CelebA file name without `.png` (for example `181500`) and identities by `celeb_id`. The side-by-side grids used for the judgements were made in a scratch folder and are not committed. Real people are not named.

## Question

For every model, which non-mated probes score highest on the validation and test draws, and is each one a stranger who looks like someone enrolled, or the enrolled person under a second CelebA id? How many of each model's budgeted false alarms go to such duplicates, and how much do they move the frozen threshold and TPIR at FPIR 1% and 0.1%? In particular, do they explain ArcFace's 80.7% validation TPIR at FPIR 0.1% against 98.4% on test?

## Answer

**Yes, a few identity labels in the validation draw are wrong, and they explain M7.** The noise is small in count but sits at the very top of the non-mated scores.

- **Validation:** 7 of 3,929 non-mated probes (0.18%) are photos of a person who is enrolled in the gallery under another id. 5 are confirmed and 2 are probable.
- **Test:** 1 of 4,072 non-mated probes is such a photo (`188385`).
- **FPIR 1%:** the 7 take 7 of ArcFace's 39 budgeted validation alarms, 7 of SFace's 39 and 4 of FaceNet's 39. They barely move anything here. ArcFace's frozen threshold would be 0.3361 instead of 0.3417, and its validation TPIR stays at 98.17%.
- **FPIR 0.1%:** ArcFace's seven highest-scoring validation non-mated probes are exactly these seven. The 0.1% budget is 3 alarms, so the threshold has to clear the fourth of them (0.6036). About a fifth of mated probes fail to match at that threshold, so validation TPIR at FPIR 0.1% is 80.7%. Without the seven, validation TPIR at FPIR 0.1% would be **97.8%**, against 98.4% on test.
- **2594 and 4064 are not duplicates.** Held-out identity 2594 is 10 more of ArcFace's 39 validation alarms: all 10 of its probes match gallery 4064. The two ids are two different people who look very alike, so these are genuine false alarms and stay in.
- **Other alarms:** the other 21 ArcFace alarms, and most SFace and FaceNet alarms, are ordinary lookalikes. A few come from sunglasses, hats, blurred or shaded enrolled photos, or a profile photo.
- **Detection:** no alarm came from a detection or crop error. Every YuNet box inspected was on the face.

## Method

1. **Rebuild the draws.** `CelebaEvaluation.prepare("validation")` and `prepare("test")` were run with the committed YuNet weights, the minimum face size and seed 27. Each image's embedding was loaded from `data/cache/embeddings/<model-key>/` for the pipeline id of the crop that `results.json` records per model.
2. **Score as the pipeline does.** `Gallery.enrol` was run on the 5 enrolled photos per gallery identity, and each probe was scored with `Gallery.top_candidates`. A probe's match score is its cosine to the best-matching enrolled photo of its top candidate (the best-photo rule). The committed operating points were checked with `tpir_at_fpir` and the threshold with `freeze`.
3. **List the tails.** For each model and draw, every non-mated probe was listed in descending score order until both of these held: at least 15 rows, and every probe at or above the model's frozen threshold included. Each row records the probe's held-out id, its top-candidate gallery id and the enrolled photo that gave the score. That gave 230 alarms: 39, 39 and 39 on validation, and 27, 37 and 49 on test, for ArcFace, SFace and FaceNet. They cover 77 distinct (held-out id, gallery id) pairs on validation and 89 on test.
4. **Look at every alarm.** For each pair, one grid row showed the following, each image with its YuNet box drawn:
   - up to three of the held-out identity's probes, the alarmed ones first;
   - the best-matching enrolled photo;
   - the gallery identity's other four enrolled photos.

   The likely duplicates were then viewed again at 1.6x, next to further photos of both identities.
5. **Check with embeddings.** For each alarmed probe, two ArcFace medians were computed:
   - its median cosine to its own identity's other drawn probes;
   - its median cosine to all 20 drawn photos of the gallery identity (5 enrolled and 15 mated probes).

   A mislabelled photo sits with the other identity and not with its own. For scale, the median correct mated score on validation is 0.702, and the 99th percentile of non-mated scores is 0.341.
6. **Sweep the whole draw.** Independently of the alarms, every non-mated probe of both draws was checked (ArcFace) for the same signature: a median cosine of at least 0.40 to some gallery identity's 20 photos, and below 0.20 to its own identity's other probes. Every gallery photo was also checked for sitting below 0.20 with its own identity.
7. **Counterfactuals.** The suspected probes were removed from the non-mated set, and the threshold was re-frozen and both draws re-read. These counterfactuals were computed in memory only.

Judgement categories:

- **label noise**: the probe and the enrolled photo show the same person;
- **lookalike**: two different people;
- **detection or crop error**;
- **other**: an image condition or model error that drives the score more than resemblance does.

A pair was called **confirmed** label noise when both of these held:

- by eye, the two photos show the same person and the probe does not show its own identity's person;
- in the embeddings, the probe's median cosine to its own identity is at or below 0.04 and to the gallery identity is at or above 0.54.

Two confirmed cases need a different embedding test: `181031`, whose identity has no other image, and `172941`, whose match is a stray enrolled photo. Their notes below give it.

It was called **probable** when the embeddings agree but the visual call is less certain.

## What the alarms are

Every non-mated probe at or above the model's frozen threshold is counted, one judgement per probe. The FPIR 1% budget is 39 alarms on validation (3,929 non-mated probes) and 40 on test (4,072).

| Model | Draw | Alarms at frozen threshold | Label noise | 2594 to 4064 | Other lookalikes | Other (image condition, model error) |
|---|---|---:|---:|---:|---:|---:|
| ArcFace | validation | 39 | 7 | 10 | 21 | 1 |
| ArcFace | test | 27 | 1 | 0 | 25 | 1 |
| SFace | validation | 39 | 7 | 0 | 28 | 4 |
| SFace | test | 37 | 1 | 0 | 31 | 5 |
| FaceNet | validation | 39 | 4 | 5 | 26 | 4 |
| FaceNet | test | 49 | 1 | 0 | 45 | 3 |

"Other" covers sunglasses, hats, a shaded or blurred enrolled photo, a profile enrolled photo, a child gallery, and one male probe matched to a female gallery (SFace).

## The label-noise pairs

The two cosine columns are ArcFace medians. "Own id" is the probe against its own identity's other drawn probes. "Gallery id" is the probe against all 20 drawn photos of the gallery identity. Scores are each model's best-photo match score: A is ArcFace, S is SFace and F is FaceNet. FaceNet runs on a different scale, with its threshold at 0.709.

| Draw | Probe image | Held-out id | Gallery id | Alarmed by (score) | Cosine, own id | Cosine, gallery id | Judgement |
|---|---|---:|---:|---|---:|---:|---|
| validation | 181500 | 1532 | 1529 | A 0.705, F 0.751, S 0.707 | -0.09 | 0.60 | label noise: mislabelled image |
| validation | 176769 | 2490 | 2491 | A 0.672, F 0.774, S 0.693 | 0.04 | 0.54 | label noise: mislabelled image |
| validation | 182091 | 406 | 4537 | A 0.604, F 0.762, S 0.624 | 0.00 | 0.54 | label noise: mislabelled image |
| validation | 181031 | 2708 | 1403 | A 0.628, F 0.797, S 0.589 | n/a | 0.58 | label noise: same person under two ids |
| validation | 172941 | 193 | 2716 | A 0.547, S 0.517 | 0.00 | 0.09 | label noise: probe and enrolled photo both mislabelled |
| validation | 181133 | 2789 | 2790 | A 0.559, S 0.567 | 0.00 | 0.52 | label noise (probable): mislabelled image |
| validation | 164657 | 3218 | 787 | A 0.543, S 0.552 | 0.12 | 0.46 | label noise (probable): mislabelled image |
| test | 188385 | 8554 | 6748 | A 0.578, F 0.852, S 0.648 | -0.09 | 0.54 | label noise: mislabelled image |

Notes on individual pairs:

- **`181031` (2708 to 1403):** identity 2708 has exactly one image in all of CelebA, this one, and it shows gallery 1403's person. This is the one true "same person under two ids" case. Apart from `172941` (below), the others are single photos filed under the wrong id.
- **`176769` (2490 to 2491), `181500` (1532 to 1529), `182091` (406 to 4537), `188385` (8554 to 6748):**
  - each probe shows the gallery identity's person;
  - each probe's own identity's other photos (`166181` and `179171` for 2490, `175476` and `181381` for 1532, `164256` and `164183` for 406, `199258` for 8554) show someone else;
  - in `182091`, the gallery person's name appears printed on the image.
- **`172941` (193 to 2716):**
  - the probe and enrolled photo `177109` show the same man, and neither is his own identity's person;
  - the probe's median cosine to its own id is 0.00, and to gallery 2716 it is 0.09;
  - `177109`'s median cosine to the other 19 photos of 2716 is 0.11;
  - the score of 0.547 is a probe matching a stray enrolled photo, with the label wrong on both sides.
- **`181133` (2789 to 2790) and `164657` (3218 to 787):** the embedding evidence is as strong as for the confirmed cases: a median of 0.00 and 0.12 to their own ids, and 0.52 and 0.46 to the gallery. By eye they are very likely the same person, but lighting, hairstyle and age differ enough that they are marked probable.
- **The whole-draw sweep (method step 6):**
  - on validation it found exactly six held-out probes with this signature: `182091`, `181500`, `176769`, `181031`, `181133` and `164657`, which are the ones above minus `172941`, whose double error the sweep cannot see;
  - on test it found only `188385`;
  - so, as far as the sweep can see, no label-noise probe that stayed under the threshold was missed.

### Adjacent ids

Three of the eight pairs are nearby CelebA ids: 1532 and 1529, 2490 and 2491, and 2789 and 2790. On the gallery side, the sweep flagged 14 of the 20,000 drawn gallery photos (7 per draw) that sit with another gallery identity rather than their own, and three of those are also adjacent: 1906 and 1908, 4248 and 4243, and 6184 and 6185. A photo filed under the next id along is a plausible annotation slip. That is an inference: CelebA does not document how ids were assigned. Those 14 gallery photos were found by embedding only and were not viewed. They affect mated probes (rank-1 and misidentification), not FPIR. They are:

- validation: `173300` and `182600` (480, with 4552), `181619` and `170922` (1906, with 1908), `165313` (2289, with 4038), `178770` (3530, with 3464), `167029` (4248, with 4243);
- test: `188842` (5212, with 6306), `193420` (6184, with 6185), `200207` (7396, with 7245), `185259` (9102, with 5448), `189557` and `186279` (9128, with 9310), `201452` (9715, with 6709).

In passing, enrolled photo `182269` of gallery 1529 looked like a different person from 1529's other photos. The sweep did not flag it.

## Identity 2594 and gallery 4064

Every one of held-out 2594's 10 validation probes matches gallery 4064 above ArcFace's threshold, at 0.342 to 0.430. That is 10 of the 39 alarms. Five of them also alarm for FaceNet, and one (`171762`, against gallery 4078) for SFace. For ArcFace, every one of them is best matched by the same enrolled photo, `178222`.

These are two distinct people:

- **Side by side:** all 10 of 2594's photos and all 20 of 4064's show two consistent faces, each with its own features, that resemble each other strongly.
- **2594's probes:** their median cosine to each other is 0.64 to 0.71, but to 4064's 20 photos only 0.21 to 0.33.
- **`178222`:** it sits with its own identity 4064.

This is a genuine near-lookalike and a fair false alarm. It is not label noise and must stay in the non-mated set. It accounts for most of the small gap that remains once the label noise is removed (see the diagnostic rows below).

## Counterfactual operating points

> **These are not the committed results.** Nothing in `evaluation/results.json` was changed, and the committed thresholds stand. The rows below say what the numbers would have been had the suspected probes not been in the non-mated set.

For each scenario, the listed probes are removed from the non-mated set, and the rest is recomputed exactly as the pipeline does:

- the threshold is re-frozen on validation;
- TPIR is read off each draw's own curve at FPIR 1% and 0.1%;
- the test draw is read at the re-frozen threshold.

"+1 test" means `188385` is also removed from the test draw, which leaves 4,071 non-mated probes there. The last row per model also removes 2594's 10 genuine lookalike probes. It is there only to show where the remaining gap sits, not as a result anyone should use.

| Model | Scenario | Validation non-mated | Frozen threshold | Val TPIR @ FPIR 1% | Val TPIR @ FPIR 0.1% (threshold) | Test TPIR @ FPIR 0.1% | Test TPIR at frozen threshold | Test FPIR at frozen threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ArcFace | Committed (`results.json`) | 3,929 | 0.3417 | 98.17% | 80.73% (0.6036) | 98.36% | 98.47% | 0.66% |
| ArcFace | Without the 5 confirmed (+1 test) | 3,924 | 0.3379 | 98.17% | 97.64% (0.4260) | 98.36% | 98.49% | 0.71% |
| ArcFace | Without all 7 suspected (+1 test) | 3,922 | 0.3361 | 98.17% | 97.79% (0.4128) | 98.36% | 98.49% | 0.81% |
| ArcFace | Also without 2594's 10 probes (diagnostic only) | 3,912 | 0.3302 | 98.19% | 98.12% (0.3663) | 98.36% | 98.49% | 1.08% |
| SFace | Committed (`results.json`) | 3,929 | 0.4980 | 94.83% | 83.71% (0.5889) | 90.00% | 95.03% | 0.91% |
| SFace | Without the 5 confirmed (+1 test) | 3,924 | 0.4959 | 94.97% | 91.71% (0.5341) | 90.51% | 95.20% | 0.96% |
| SFace | Without all 7 suspected (+1 test) | 3,922 | 0.4950 | 95.07% | 92.13% (0.5295) | 90.51% | 95.21% | 0.96% |
| SFace | Also without 2594's 10 probes (diagnostic only) | 3,912 | 0.4940 | 95.11% | 92.13% (0.5295) | 90.51% | 95.31% | 1.03% |
| FaceNet | Committed (`results.json`) | 3,929 | 0.7088 | 84.53% | 65.72% (0.7792) | 72.92% | 85.13% | 1.20% |
| FaceNet | Without the 5 confirmed (+1 test) | 3,924 | 0.7072 | 84.81% | 67.05% (0.7749) | 73.04% | 85.41% | 1.20% |
| FaceNet | Without all 7 suspected (+1 test) | 3,922 | 0.7072 | 84.81% | 67.05% (0.7749) | 73.04% | 85.41% | 1.20% |
| FaceNet | Also without 2594's 10 probes (diagnostic only) | 3,912 | 0.7031 | 85.51% | 67.05% (0.7749) | 73.04% | 86.19% | 1.20% |

What the table says:

- **ArcFace at FPIR 1%:**
  - label noise costs 7 of the 39 alarms but moves the frozen threshold by only 0.006 (0.3417 to 0.3361);
  - validation TPIR is unchanged (98.17%), and test TPIR at the frozen threshold rises by 0.02 points;
  - test FPIR at the threshold would have been 0.81% instead of 0.66%, so the committed threshold is slightly conservative, not wrong.
- **ArcFace at FPIR 0.1%:** removing the seven suspected probes closes the gap from 80.7% (validation) and 98.4% (test) to 97.8% and 98.4%. Also removing 2594 brings validation to 98.1%.
  - The whole gap is therefore label noise plus one lookalike pair. The 0.1% point has a budget of 3 to 4 alarms, so single probes decide it. This is why #27 marked it indicative.
  - The diagnostic row also shows why 2594 must stay in: without it, test FPIR at the re-frozen threshold would be 1.08%, over the target.
- **SFace:**
  - six of the seven are its top six validation non-mated, and `172941` is at rank 15;
  - removing them lifts validation TPIR at 0.1% from 83.7% to 92.1%, now above test's 90.5%;
  - at 1%, validation TPIR moves from 94.83% to 95.07%.
- **FaceNet:**
  - only 4 of its 39 validation alarms are label noise; its top non-mated are lookalikes (4517 to 3547 at 0.809, and 2247 to 1133);
  - removing the noise moves validation TPIR at 0.1% only from 65.7% to 67.1%, against 73.0% on test;
  - FaceNet's own validation-test gap at 0.1% is not a label-noise effect; it is the weaker model's lookalike tail read off 3 to 4 alarms.

## Top non-mated pairs per model

Each model's 15 highest-scoring non-mated probes on each draw are all above its frozen threshold, and each was viewed. "Best enrolled image" is the gallery photo that gave the match score.

### ArcFace, validation draw (frozen threshold 0.3417)

| Rank | Score | Held-out id | Probe image | Gallery id | Best enrolled image | Judgement |
|---:|---:|---:|---|---:|---|---|
| 1 | 0.7053 | 1532 | 181500 | 1529 | 179987 | label noise: mislabelled image |
| 2 | 0.6719 | 2490 | 176769 | 2491 | 170030 | label noise: mislabelled image |
| 3 | 0.6277 | 2708 | 181031 | 1403 | 162954 | label noise: same person under two ids |
| 4 | 0.6035 | 406 | 182091 | 4537 | 166870 | label noise: mislabelled image |
| 5 | 0.5594 | 2789 | 181133 | 2790 | 165067 | label noise (probable): mislabelled image |
| 6 | 0.5471 | 193 | 172941 | 2716 | 177109 | label noise: probe and enrolled photo both mislabelled |
| 7 | 0.5432 | 3218 | 164657 | 787 | 177807 | label noise (probable): mislabelled image |
| 8 | 0.4300 | 2594 | 162771 | 4064 | 178222 | lookalike (2594 and 4064 are distinct people) |
| 9 | 0.4253 | 2594 | 167620 | 4064 | 178222 | lookalike (2594 and 4064 are distinct people) |
| 10 | 0.4133 | 2594 | 165726 | 4064 | 178222 | lookalike (2594 and 4064 are distinct people) |
| 11 | 0.4103 | 2594 | 163824 | 4064 | 178222 | lookalike (2594 and 4064 are distinct people) |
| 12 | 0.3866 | 1464 | 179130 | 1046 | 182211 | other: dark, shaded enrolled photo |
| 13 | 0.3864 | 2590 | 166496 | 2843 | 165098 | lookalike |
| 14 | 0.3845 | 2594 | 163876 | 4064 | 178222 | lookalike (2594 and 4064 are distinct people) |
| 15 | 0.3841 | 2594 | 169956 | 4064 | 178222 | lookalike (2594 and 4064 are distinct people) |

### ArcFace, test draw (frozen threshold 0.3417)

| Rank | Score | Held-out id | Probe image | Gallery id | Best enrolled image | Judgement |
|---:|---:|---:|---|---:|---|---|
| 1 | 0.5779 | 8554 | 188385 | 6748 | 194513 | label noise: mislabelled image |
| 2 | 0.3958 | 7036 | 194985 | 8945 | 183143 | lookalike |
| 3 | 0.3808 | 5743 | 193459 | 7888 | 202153 | lookalike |
| 4 | 0.3789 | 7489 | 189502 | 5228 | 196355 | lookalike |
| 5 | 0.3738 | 5322 | 191676 | 5935 | 193504 | lookalike |
| 6 | 0.3734 | 6267 | 188286 | 9798 | 198415 | lookalike |
| 7 | 0.3716 | 9928 | 197311 | 5301 | 194701 | other: glasses on both |
| 8 | 0.3690 | 8698 | 200840 | 10001 | 191855 | lookalike |
| 9 | 0.3673 | 7776 | 197608 | 9732 | 184507 | lookalike |
| 10 | 0.3667 | 8359 | 200074 | 5458 | 197647 | lookalike |
| 11 | 0.3658 | 10108 | 196762 | 4978 | 196836 | lookalike |
| 12 | 0.3614 | 6293 | 198167 | 9746 | 189372 | lookalike |
| 13 | 0.3592 | 7736 | 200274 | 9314 | 201301 | lookalike |
| 14 | 0.3590 | 7664 | 197478 | 10047 | 185776 | lookalike |
| 15 | 0.3564 | 7489 | 189376 | 5228 | 196355 | lookalike |

### SFace, validation draw (frozen threshold 0.4980)

| Rank | Score | Held-out id | Probe image | Gallery id | Best enrolled image | Judgement |
|---:|---:|---:|---|---:|---|---|
| 1 | 0.7068 | 1532 | 181500 | 1529 | 179987 | label noise: mislabelled image |
| 2 | 0.6928 | 2490 | 176769 | 2491 | 170030 | label noise: mislabelled image |
| 3 | 0.6239 | 406 | 182091 | 4537 | 162814 | label noise: mislabelled image |
| 4 | 0.5889 | 2708 | 181031 | 1403 | 162954 | label noise: same person under two ids |
| 5 | 0.5674 | 2789 | 181133 | 2790 | 171271 | label noise (probable): mislabelled image |
| 6 | 0.5525 | 3218 | 164657 | 787 | 173257 | label noise (probable): mislabelled image |
| 7 | 0.5467 | 761 | 179848 | 317 | 168118 | lookalike |
| 8 | 0.5340 | 893 | 177029 | 947 | 177822 | other: blurred enrolled photo |
| 9 | 0.5335 | 2919 | 175281 | 787 | 178194 | lookalike |
| 10 | 0.5295 | 200 | 172529 | 4038 | 171215 | lookalike |
| 11 | 0.5286 | 2659 | 171978 | 1403 | 175793 | lookalike |
| 12 | 0.5279 | 2203 | 167968 | 875 | 170734 | lookalike |
| 13 | 0.5185 | 736 | 175209 | 3246 | 175707 | lookalike |
| 14 | 0.5179 | 2003 | 164318 | 1210 | 172023 | lookalike |
| 15 | 0.5174 | 193 | 172941 | 2716 | 177109 | label noise: probe and enrolled photo both mislabelled |

### SFace, test draw (frozen threshold 0.4980)

| Rank | Score | Held-out id | Probe image | Gallery id | Best enrolled image | Judgement |
|---:|---:|---:|---|---:|---|---|
| 1 | 0.6480 | 8554 | 188385 | 6748 | 194513 | label noise: mislabelled image |
| 2 | 0.5527 | 9271 | 189625 | 6153 | 191108 | lookalike |
| 3 | 0.5482 | 7664 | 184036 | 10047 | 185776 | lookalike |
| 4 | 0.5469 | 6293 | 194187 | 6080 | 187744 | lookalike |
| 5 | 0.5466 | 9229 | 198285 | 7216 | 184149 | other: sunglasses and hats |
| 6 | 0.5417 | 7185 | 199022 | 9807 | 184279 | lookalike |
| 7 | 0.5400 | 5585 | 192082 | 6080 | 194888 | lookalike |
| 8 | 0.5358 | 4989 | 184094 | 5218 | 194846 | lookalike |
| 9 | 0.5299 | 5030 | 191846 | 6823 | 196953 | lookalike |
| 10 | 0.5270 | 9283 | 201362 | 9105 | 193960 | lookalike |
| 11 | 0.5255 | 9713 | 202327 | 8736 | 197206 | lookalike |
| 12 | 0.5248 | 7889 | 195841 | 5444 | 196225 | other: enrolled photo covered by hands |
| 13 | 0.5244 | 7189 | 189251 | 6763 | 199361 | lookalike |
| 14 | 0.5222 | 8814 | 183251 | 6937 | 194226 | lookalike |
| 15 | 0.5212 | 7862 | 197079 | 4998 | 198818 | lookalike |

### FaceNet, validation draw (frozen threshold 0.7088)

| Rank | Score | Held-out id | Probe image | Gallery id | Best enrolled image | Judgement |
|---:|---:|---:|---|---:|---|---|
| 1 | 0.8085 | 4517 | 179287 | 3547 | 164671 | lookalike |
| 2 | 0.7971 | 2708 | 181031 | 1403 | 174854 | label noise: same person under two ids |
| 3 | 0.7818 | 2247 | 180242 | 1133 | 171073 | lookalike |
| 4 | 0.7791 | 4460 | 172491 | 1590 | 174639 | lookalike |
| 5 | 0.7749 | 4517 | 169129 | 3547 | 164671 | lookalike |
| 6 | 0.7745 | 2490 | 176769 | 2491 | 170030 | label noise: mislabelled image |
| 7 | 0.7698 | 2594 | 165726 | 4064 | 178222 | lookalike (2594 and 4064 are distinct people) |
| 8 | 0.7648 | 1877 | 168208 | 2422 | 182544 | lookalike |
| 9 | 0.7616 | 406 | 182091 | 4537 | 166870 | label noise: mislabelled image |
| 10 | 0.7515 | 2594 | 167620 | 4064 | 163899 | lookalike (2594 and 4064 are distinct people) |
| 11 | 0.7510 | 1532 | 181500 | 1529 | 179987 | label noise: mislabelled image |
| 12 | 0.7491 | 2247 | 180603 | 1133 | 171073 | lookalike |
| 13 | 0.7486 | 2247 | 178336 | 1133 | 171444 | lookalike |
| 14 | 0.7468 | 2834 | 163655 | 1046 | 182211 | other: dark, shaded enrolled photo |
| 15 | 0.7466 | 2834 | 164253 | 1046 | 182211 | other: dark, shaded enrolled photo |

### FaceNet, test draw (frozen threshold 0.7088)

| Rank | Score | Held-out id | Probe image | Gallery id | Best enrolled image | Judgement |
|---:|---:|---:|---|---:|---|---|
| 1 | 0.8525 | 8554 | 188385 | 6748 | 194513 | label noise: mislabelled image |
| 2 | 0.8481 | 5667 | 188703 | 7045 | 187579 | lookalike |
| 3 | 0.8000 | 5743 | 202345 | 5433 | 201529 | lookalike |
| 4 | 0.7692 | 8633 | 191250 | 9898 | 196501 | lookalike |
| 5 | 0.7633 | 5945 | 195598 | 9232 | 190488 | lookalike |
| 6 | 0.7629 | 7510 | 187933 | 6920 | 184515 | lookalike |
| 7 | 0.7603 | 9229 | 200915 | 7216 | 193486 | other: sunglasses and hats |
| 8 | 0.7597 | 5945 | 201051 | 9232 | 190488 | lookalike |
| 9 | 0.7589 | 8359 | 200074 | 5458 | 197647 | lookalike |
| 10 | 0.7562 | 9761 | 198312 | 7789 | 201923 | lookalike |
| 11 | 0.7548 | 9824 | 187466 | 9286 | 194614 | lookalike |
| 12 | 0.7537 | 7841 | 196984 | 9898 | 196501 | lookalike |
| 13 | 0.7523 | 5244 | 196923 | 7789 | 184790 | lookalike |
| 14 | 0.7464 | 7841 | 192488 | 9898 | 196501 | lookalike |
| 15 | 0.7450 | 7664 | 197478 | 8918 | 189183 | lookalike |

## Other observations

- **No detection or crop errors.** In all 166 pairs, YuNet's box was on the face in both probe and gallery photo. Some CelebA images have smeared or stretched borders from CelebA's own alignment padding (for example the lower band of `181500`), but the face crop is unaffected.
- **Image condition drives a minority of the alarms**, mostly for SFace and FaceNet:
  - dark sunglasses on both sides (validation 4023 to 3211; test 7566 to 7848 and 9215 to 7848);
  - hats and sunglasses (test 9229 to 7216);
  - a shaded enrolled photo under a visor (`182211` of gallery 1046, matched by three probes of two held-out ids);
  - a blurred enrolled photo (`177822`), a profile enrolled photo (`198923`), an enrolled photo covered by hands (`196225`), and a gallery of childhood photos (9831);
  - one male probe matched to a female gallery at SFace 0.502 (3348 to 327).
- **Lookalike alarms cluster within apparent demographic groups:**
  - FaceNet's test alarms include several pairs of East Asian men (5743 to 5433, 8633 to 9898, 5945 to 9232, 7841 to 9898, 8152 to 9898) and East Asian women (7510 to 6920, 7020 to 6920);
  - there are pairs of Black men wearing hats or sunglasses (9229 to 7216, 9215 to 7216, 7566 to 7848) and Black women (5667 to 7045, 5315 to 7045);
  - on validation there are many pairs of young blonde women.

  This is an observation from viewing, not a measurement. CelebA has no skin-tone or race label (#10), so #28 cannot break FPIR down this way.

## Drafted text

### Limitations paragraph for #32

For the report's Limitations section, `report/sections/08-limitations.qmd`. It is Section 8 in `report.qmd`; Section 7 is the Prototype.

> **CelebA identity labels are not clean.** Some of CelebA's identity labels are wrong, and a few of the wrong ones land among the non-mated probes. Every non-mated probe that scored above a model's frozen threshold was inspected (#54). Seven of the validation draw's 3,929 non-mated probes (five certain, two probable) and one of the test draw's 4,072 turned out to be photos of a person enrolled in the gallery under another id. Six are single photos filed under the wrong id; one is a one-image identity that duplicates an enrolled person; in one, a probe and an enrolled photo of the same third person are both mislabelled. They were still counted as false alarms. At FPIR 1% they took 7 of ArcFace's 39 permitted validation alarms. This raised its frozen threshold from 0.336 to 0.342 and left validation TPIR at 98.2%; test FPIR at the frozen threshold would have been 0.81% rather than 0.66%. At FPIR 0.1% they are ArcFace's seven highest-scoring validation non-mated probes. This is why its validation TPIR there is 80.7% against 98.4% on test; without them it would be 97.8%. The frozen thresholds are therefore slightly conservative, and the FPIR 0.1% figures, already indicative, turn on single probes. Nothing was relabelled: the thresholds were frozen on the data as published, and a correction judged by eye on a handful of alarms could not be applied consistently to the whole dataset.

### Sentence for #28's bias section

> CelebA's identity labels are imperfect (#54). One test non-mated probe (`188385`, labelled not Male and Young) is a mislabelled photo of an enrolled person and counts as a false alarm in its groups. The threshold was frozen on a validation draw holding seven such probes. So a per-group FPIR that differs from another group's by one or two alarms should be read with that in mind before it is put down to the model.

## Appendix: every alarmed pair

Every (held-out id, gallery id) pair with at least one probe at or above some model's frozen threshold, in descending order of top score, with each alarmed probe's scores. The same A, S and F abbreviations are used.

<details>
<summary>Validation draw: 77 pairs</summary>

| Held-out id | Gallery id | Alarmed probe images (model: score) | Judgement |
|---:|---:|---|---|
| 4517 | 3547 | 179287 (F 0.809, S 0.506); 169129 (F 0.775) | lookalike |
| 2708 | 1403 | 181031 (A 0.628, F 0.797, S 0.589) | label noise: same person under two ids |
| 2247 | 1133 | 180242 (F 0.782); 180603 (F 0.749); 178336 (F 0.749) | lookalike |
| 4460 | 1590 | 172491 (F 0.779) | lookalike |
| 2490 | 2491 | 176769 (A 0.672, F 0.774, S 0.693) | label noise: mislabelled image |
| 2594 | 4064 | 162771 (A 0.430, F 0.711); 167620 (A 0.425, F 0.752); 165726 (A 0.413, F 0.770); 163824 (A 0.410, F 0.712); 163876 (A 0.385); 169956 (A 0.384); 175324 (A 0.375); 171907 (A 0.373, F 0.715); 165619 (A 0.354); 171762 (A 0.342) | lookalike (2594 and 4064 are distinct people) |
| 1877 | 2422 | 168208 (F 0.765) | lookalike |
| 406 | 4537 | 182091 (A 0.604, F 0.762, S 0.624) | label noise: mislabelled image |
| 1532 | 1529 | 181500 (A 0.705, F 0.751, S 0.707) | label noise: mislabelled image |
| 2834 | 1046 | 163655 (F 0.747); 164253 (F 0.747) | other: dark, shaded enrolled photo |
| 3639 | 510 | 165869 (F 0.742) | lookalike |
| 3113 | 2289 | 164879 (F 0.737) | lookalike |
| 2906 | 2283 | 168525 (F 0.733) | lookalike |
| 1464 | 1046 | 179130 (A 0.387, F 0.728) | other: dark, shaded enrolled photo |
| 2551 | 1631 | 182406 (F 0.728) | lookalike |
| 1675 | 4144 | 179284 (F 0.714, S 0.504); 179415 (F 0.725); 178232 (F 0.710) | lookalike |
| 2906 | 510 | 171655 (F 0.724) | lookalike |
| 776 | 2899 | 166209 (F 0.723) | lookalike |
| 916 | 2283 | 179255 (F 0.722) | lookalike |
| 2203 | 3685 | 169682 (S 0.510); 166919 (F 0.719, S 0.500); 166656 (A 0.344) | lookalike |
| 2919 | 3811 | 175281 (F 0.719); 176563 (F 0.713) | lookalike |
| 321 | 953 | 172059 (F 0.719) | other: probe in sunglasses and hat |
| 4557 | 678 | 180493 (F 0.717) | lookalike |
| 2919 | 2289 | 169701 (F 0.716) | lookalike |
| 3944 | 2422 | 169894 (F 0.716) | lookalike |
| 3059 | 3811 | 176392 (F 0.711) | lookalike |
| 479 | 1826 | 163444 (F 0.711) | lookalike |
| 4460 | 2283 | 179262 (F 0.709) | lookalike |
| 2789 | 2790 | 181133 (A 0.559, S 0.567) | label noise (probable): mislabelled image |
| 3218 | 787 | 164657 (A 0.543, S 0.552) | label noise (probable): mislabelled image |
| 761 | 317 | 179848 (S 0.547) | lookalike |
| 193 | 2716 | 172941 (A 0.547, S 0.517) | label noise: probe and enrolled photo both mislabelled |
| 893 | 947 | 177029 (S 0.534) | other: blurred enrolled photo |
| 2919 | 787 | 175281 (S 0.534) | lookalike |
| 200 | 4038 | 172529 (S 0.530) | lookalike |
| 2659 | 1403 | 171978 (S 0.529) | lookalike |
| 2203 | 875 | 167968 (S 0.528) | lookalike |
| 736 | 3246 | 175209 (S 0.518) | lookalike |
| 2003 | 1210 | 164318 (S 0.518) | lookalike |
| 3950 | 3521 | 164749 (S 0.514) | lookalike |
| 3033 | 2418 | 165595 (S 0.514) | lookalike |
| 109 | 4552 | 166261 (S 0.511) | other: probe in sunglasses |
| 3241 | 1469 | 172564 (S 0.511) | lookalike |
| 461 | 787 | 177478 (S 0.510) | lookalike |
| 3033 | 1018 | 179467 (S 0.509) | lookalike |
| 2077 | 2886 | 176837 (S 0.507) | lookalike |
| 321 | 2283 | 163600 (S 0.503) | lookalike |
| 3348 | 327 | 173897 (S 0.502) | other: male probe, female gallery |
| 2795 | 2348 | 166603 (S 0.501) | lookalike |
| 2594 | 4078 | 171762 (S 0.501) | lookalike |
| 2557 | 649 | 175703 (S 0.501) | lookalike |
| 2825 | 627 | 180920 (S 0.501) | lookalike |
| 4023 | 3211 | 170799 (S 0.501) | other: dark sunglasses on both |
| 1109 | 3782 | 167359 (S 0.501) | lookalike |
| 2247 | 3859 | 180242 (S 0.500) | lookalike |
| 893 | 1154 | 173944 (S 0.500) | lookalike |
| 4818 | 1212 | 180997 (S 0.499) | lookalike |
| 2919 | 2547 | 171380 (S 0.498) | lookalike |
| 2164 | 643 | 167277 (S 0.498) | lookalike |
| 2590 | 2843 | 166496 (A 0.386) | lookalike |
| 1004 | 2810 | 171904 (A 0.366); 177743 (A 0.352) | lookalike |
| 3194 | 4038 | 163903 (A 0.364) | lookalike |
| 3493 | 4432 | 163325 (A 0.360) | lookalike |
| 2923 | 1542 | 176724 (A 0.356) | lookalike |
| 2108 | 2745 | 169012 (A 0.355) | lookalike |
| 200 | 2283 | 166232 (A 0.355) | lookalike |
| 3224 | 2716 | 180207 (A 0.354); 177506 (A 0.349) | lookalike |
| 321 | 4038 | 180611 (A 0.354) | lookalike |
| 2860 | 2762 | 168740 (A 0.353) | lookalike |
| 761 | 3490 | 179848 (A 0.353) | lookalike |
| 3726 | 921 | 167166 (A 0.352) | lookalike |
| 461 | 984 | 170819 (A 0.352) | lookalike |
| 2860 | 2840 | 170809 (A 0.351) | lookalike |
| 2378 | 3385 | 170338 (A 0.350) | lookalike |
| 1257 | 3859 | 166322 (A 0.346) | lookalike |
| 1350 | 1884 | 172819 (A 0.344) | lookalike |
| 1109 | 4558 | 167359 (A 0.342) | lookalike |

</details>

<details>
<summary>Test draw: 89 pairs</summary>

| Held-out id | Gallery id | Alarmed probe images (model: score) | Judgement |
|---:|---:|---|---|
| 8554 | 6748 | 188385 (A 0.578, F 0.852, S 0.648) | label noise: mislabelled image |
| 5667 | 7045 | 188703 (F 0.848); 202483 (F 0.712) | lookalike |
| 5743 | 5433 | 202345 (F 0.800) | lookalike |
| 8633 | 9898 | 191250 (F 0.769) | lookalike |
| 5945 | 9232 | 195598 (F 0.763); 201051 (F 0.760); 183343 (F 0.729); 193892 (F 0.721) | lookalike |
| 7510 | 6920 | 187933 (F 0.763); 184392 (F 0.713) | lookalike |
| 9229 | 7216 | 198285 (S 0.547); 200915 (F 0.760) | other: sunglasses and hats |
| 8359 | 5458 | 200074 (A 0.367, F 0.759) | lookalike |
| 9761 | 7789 | 189707 (A 0.352); 198312 (F 0.756) | lookalike |
| 9824 | 9286 | 187466 (F 0.755) | lookalike |
| 7841 | 9898 | 196984 (F 0.754); 192488 (F 0.746) | lookalike |
| 5244 | 7789 | 196923 (F 0.752) | lookalike |
| 7664 | 8918 | 197478 (F 0.745) | lookalike |
| 9271 | 6153 | 189625 (F 0.743, S 0.553) | lookalike |
| 9870 | 9874 | 198397 (F 0.736) | lookalike |
| 9215 | 7216 | 188857 (F 0.736) | lookalike |
| 8505 | 7824 | 185900 (F 0.736) | other: probe in sunglasses |
| 7776 | 9732 | 197608 (A 0.367); 198508 (F 0.735); 197365 (F 0.715) | lookalike |
| 7488 | 5935 | 192514 (F 0.734) | lookalike |
| 5244 | 9864 | 200359 (F 0.733) | lookalike |
| 8152 | 9898 | 196545 (F 0.732); 199802 (F 0.716); 198765 (F 0.710) | lookalike |
| 7776 | 7251 | 199833 (F 0.730) | lookalike |
| 9428 | 5100 | 201558 (F 0.730) | lookalike |
| 5743 | 7888 | 193459 (A 0.381, F 0.729) | lookalike |
| 7020 | 6920 | 183448 (F 0.729); 196825 (F 0.718); 198685 (F 0.711) | lookalike |
| 7020 | 9102 | 185187 (F 0.728) | lookalike |
| 6221 | 7908 | 200477 (F 0.727) | lookalike |
| 7664 | 7789 | 188392 (F 0.727) | lookalike |
| 9215 | 5304 | 186412 (F 0.726) | lookalike |
| 8201 | 8572 | 183617 (F 0.723) | lookalike |
| 7020 | 8965 | 192190 (F 0.722) | lookalike |
| 7566 | 7848 | 184274 (F 0.717) | other: sunglasses on both |
| 9761 | 9275 | 199031 (F 0.714) | lookalike |
| 8296 | 7789 | 197926 (F 0.713) | lookalike |
| 6600 | 5987 | 184546 (F 0.711) | lookalike |
| 8052 | 8164 | 194304 (F 0.711) | lookalike |
| 5315 | 7045 | 183459 (F 0.710) | lookalike |
| 10132 | 7568 | 201580 (F 0.710) | lookalike |
| 7664 | 10047 | 184036 (S 0.548); 197478 (A 0.359) | lookalike |
| 6293 | 6080 | 194187 (S 0.547) | lookalike |
| 7185 | 9807 | 199022 (S 0.542) | lookalike |
| 5585 | 6080 | 192082 (S 0.540) | lookalike |
| 4989 | 5218 | 184094 (S 0.536) | lookalike |
| 5030 | 6823 | 191846 (S 0.530) | lookalike |
| 9283 | 9105 | 201362 (S 0.527) | lookalike |
| 9713 | 8736 | 202327 (S 0.526) | lookalike |
| 7889 | 5444 | 195841 (S 0.525) | other: enrolled photo covered by hands |
| 7189 | 6763 | 189251 (S 0.524) | lookalike |
| 8814 | 6937 | 183251 (S 0.522) | lookalike |
| 7862 | 4998 | 197079 (S 0.521) | lookalike |
| 5947 | 6919 | 187124 (S 0.519) | lookalike |
| 6253 | 8308 | 200585 (S 0.515) | lookalike |
| 9215 | 7848 | 195464 (S 0.514) | other: sunglasses on both |
| 9377 | 6093 | 201185 (A 0.347, S 0.514) | lookalike |
| 8296 | 6745 | 182878 (S 0.512) | other: enrolled photo in profile |
| 8255 | 8745 | 199279 (S 0.511) | lookalike |
| 5931 | 6703 | 195643 (S 0.510) | lookalike |
| 7334 | 8537 | 199840 (S 0.508) | lookalike |
| 9632 | 9831 | 195099 (S 0.506) | other: gallery photos are of a child |
| 6655 | 7057 | 184864 (S 0.505) | lookalike |
| 6047 | 5212 | 184033 (S 0.505) | lookalike |
| 8814 | 9717 | 191953 (S 0.505) | lookalike |
| 8499 | 6745 | 200036 (S 0.503) | lookalike |
| 5168 | 8495 | 199719 (S 0.502) | lookalike |
| 5328 | 8234 | 184039 (S 0.501); 190481 (S 0.501) | lookalike |
| 5535 | 5958 | 195311 (S 0.501) | lookalike |
| 8806 | 6494 | 183228 (S 0.500) | lookalike |
| 9713 | 6219 | 197650 (S 0.500) | lookalike |
| 9713 | 5223 | 187688 (S 0.499) | lookalike |
| 6323 | 5216 | 184013 (S 0.499) | lookalike |
| 9647 | 6194 | 189514 (S 0.498) | lookalike |
| 7036 | 8945 | 194985 (A 0.396) | lookalike |
| 7489 | 5228 | 189502 (A 0.379); 189376 (A 0.356) | lookalike |
| 5322 | 5935 | 191676 (A 0.374) | lookalike |
| 6267 | 9798 | 188286 (A 0.373) | lookalike |
| 9928 | 5301 | 197311 (A 0.372) | other: glasses on both |
| 8698 | 10001 | 200840 (A 0.369) | lookalike |
| 10108 | 4978 | 196762 (A 0.366) | lookalike |
| 6293 | 9746 | 198167 (A 0.361) | lookalike |
| 7736 | 9314 | 200274 (A 0.359) | lookalike |
| 9255 | 9764 | 201450 (A 0.356) | lookalike |
| 9268 | 5663 | 190881 (A 0.355) | lookalike |
| 6432 | 9010 | 188380 (A 0.353) | lookalike |
| 6297 | 6945 | 201373 (A 0.352) | lookalike |
| 8463 | 8786 | 197950 (A 0.351) | lookalike |
| 7334 | 10102 | 189626 (A 0.347) | lookalike |
| 5030 | 5084 | 191846 (A 0.346); 194440 (A 0.344) | lookalike |
| 10169 | 7179 | 187621 (A 0.343) | lookalike |
| 6221 | 6703 | 198235 (A 0.342) | lookalike |

</details>
