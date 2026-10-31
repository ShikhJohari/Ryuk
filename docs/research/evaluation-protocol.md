# Evaluation protocol for verification and open-set identification

This is a wayfinder for issue #4: what protocol and metrics make Ryuk's numbers comparable to the numbers published for SFace and ArcFace, and how to extend the same rigor to open-set identification, which LFW does not cover.

> **Corrections, 2026-09-25.** Later tickets overruled parts of this write-up; the body below is kept as researched.
> - The 99.77% quoted for ArcFace belongs to the ResNet100 / MS1MV2 checkpoint. Ryuk uses `buffalo_l` (`w600k_r50`, ResNet50 on WebFace600K), published at **99.83%** ([evaluation protocol ticket](https://github.com/ShikhJohari/Ryuk/issues/9#issuecomment-5822916457)). The published targets are SFace 99.40, ArcFace 99.83, FaceNet 99.65.
> - The adjusted Wilson check floors N* at G, not G/2, for every rate, including when there are no errors: all of Ryuk's open-set rates are per identity, like the paper's FRR, not over pairs of identities like its FAR (#44).

## Recommended metric set

**Verification (1:1, LFW-style pairs).** Compute, per fold and averaged over 10 folds: accuracy at a threshold chosen on the other 9 folds, the ROC curve and its AUC, and TAR (true accept rate, also called VAL) at FAR = 1e-2 and FAR = 1e-3. Report the mean accuracy and the standard error of the mean across folds, not a single pooled number.

**Open-set identification (1:many, gallery vs. probe).** Compute closed-set rank-1 identification rate over probes that do have a match on the watchlist, and, at a chosen score threshold, the open-set pair DIR (detection and identification rate) and FAR, or equivalently TPIR and FPIR. Report these as a curve (TPIR vs. FPIR, or DIR vs. FAR) rather than a single point, because the interesting operating range is FPIR between 0.1% and 10%.

**Threshold selection.** Pick the operating threshold on a validation split, using the same rule that will later score the watchlist in production, then freeze it before touching the test split. Never copy a threshold out of a model's README, per `GLOSSARY.md`.

**Confidence intervals.** Use identity-level (not pair-level) resampling, because genuine and impostor pairs sharing an identity are correlated and pair-level bootstrap understates variance. A percentile bootstrap over identities is the practical default; Fogliato et al. (2023) show a Wilson interval with variance corrected for identity-level dependence covers more reliably at the extremes, and is worth cross-checking when an error rate is very low. Section "Threshold and confidence interval procedure" below gives both.

## Verification protocol: LFW's 10-fold pairs

### What pairs.txt contains

The database has two views. View 1 (`pairsDevTrain.txt` / `pairsDevTest.txt`) is for model selection and should never be used for the number you report. View 2 (`pairs.txt`) is for performance reporting only, and is organized as 10 subsets of 600 pairs each (300 matched, 300 mismatched), for 6,000 pairs total (Huang, Ramesh, Berg, Learned-Miller, "Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments," UMass Amherst TR 07-49, 2007, https://people.cs.umass.edu/~elm/papers/lfw.pdf, sections III and VI-F).

The file format, quoted from the technical report:

- First line: the number of sets, then N, the number of matched pairs per set (equal to the number of mismatched pairs per set).
- A matched pair line: `name n1 n2`, meaning the pair is images `n1` and `n2` of that person (for example `George_W_Bush 10 24`).
- A mismatched pair line: `name1 n1 name2 n2`, meaning image `n1` of `name1` paired with image `n2` of `name2`.
- The next 2N lines give set 1's matched then mismatched pairs, repeated for the other nine sets.

Matched pairs were sampled by picking a person with at least two images uniformly at random, then two of their images; mismatched pairs by picking two different people uniformly at random, then one image of each. Both processes reject and retry on a duplicate pair (same report, section VI-F).

### How the 10-fold accuracy and its error bar are computed

The report specifies leave-one-subset-out cross-validation: for each of the 10 experiments, combine the other 9 subsets into a training set, choose the classifier's threshold and any other parameters on that training set only, then test on the held-out subset. Let `p_i` be the percent correct on subset `i`. Report:

```
mean accuracy   μ = (Σ_{i=1..10} p_i) / 10
sample std dev  σ = sqrt( Σ_{i=1..10} (p_i - μ)^2 / 9 )
standard error  SE = σ / sqrt(10)
```

(same report, section III). The report is explicit that a threshold must not be chosen by looking at the test fold, and that View 2 should be used sparingly, ideally exactly once per model.

### How OpenCV zoo and InsightFace actually implement this

The OpenCV Zoo evaluation code for face recognition models, including SFace, is adapted directly from InsightFace's evaluator (`tools/eval/README.md`, https://github.com/opencv/opencv_zoo/blob/main/tools/eval/README.md, and `tools/eval/datasets/lfw.py`, https://github.com/opencv/opencv_zoo/blob/main/tools/eval/datasets/lfw.py). Both implementations, and InsightFace's own `verification.py` (https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval/verification.py), share the same core:

- Embeddings are L2-normalized, and the pair distance is squared Euclidean distance: `dist = ||e1 - e2||^2` on unit vectors, which is a monotonic function of cosine similarity (`dist = 2 - 2*cos_sim`).
- Thresholds are swept over a fixed grid (`np.arange(0, 4, 0.01)` for accuracy, a finer `np.arange(0, 4, 0.001)` for TAR@FAR).
- Folding uses `sklearn.model_selection.KFold` with `shuffle=False` over the 6,000 pairs in file order, which lines up with the 10 official LFW subsets since the file already groups pairs by subset.
- Per fold: pick the threshold that maximizes accuracy on the 9 training folds (`argmax` over the grid), then apply that one threshold to the held-out fold. `accuracy = (tp + tn) / n_pairs` on the held-out fold; the reported number is the mean over the 10 held-out-fold accuracies.
- TAR@FAR is computed the same way but the per-fold threshold is chosen by linear interpolation of the training fold's FAR-vs-threshold curve to hit a target FAR (1e-3 by default in both codebases), then that threshold's TAR and FAR are measured on the held-out fold and averaged over folds, with the standard deviation of the per-fold TAR also reported.

This is the exact recipe Ryuk should reproduce for LFW: same distance function, same per-fold threshold selection, same averaging. Matching it is what lets Ryuk's number be compared to a README number at all.

## Open-set identification protocol: gallery, probes, and DIR/TPIR/FPIR

LFW only measures verification. IJB-C adds an open-set 1:N protocol, which is what Ryuk's watchlist scenario actually looks like: a probe face may or may not belong to anyone on the watchlist, and saying "nobody" has to be a scored outcome (see `GLOSSARY.md`'s definition of open set).

### Definitions

Given a gallery (enrolled templates) and a set of probes, some of which are mated (the probe's true identity is in the gallery) and some non-mated (it is not):

- **Rank-N identification rate (closed-set)**: over mated probes only, the fraction whose correct gallery template appears in the top N ranked candidates by score.
- **DIR(θ, N)**, detection and identification rate: over mated probes, the fraction whose correct match is in the top N ranks *and* whose top score is at least θ. Formally, with `K` the set of mated probes, `rank(p)` the rank of the correct match for probe `p`, and `sim(p, gallery)` its top similarity score:

  ```
  DIR(θ, N) = |{ p in K : rank(p) <= N and sim(p, gallery) >= θ }| / |K|
  ```

  This is the metric introduced for the Janus series in Klare et al., "Pushing the Frontiers of Unconstrained Face Detection and Recognition: IARPA Janus Benchmark A," CVPR 2015 (https://openaccess.thecvf.com/content_cvpr_2015/papers/Klare_Pushing_the_Frontiers_2015_CVPR_paper.pdf), reported as a DIR-vs-FAR curve at a fixed rank (usually rank 1).
- **FAR(θ)** in the same open-set curve: over non-mated probes `U`, the fraction whose top score against the gallery is still at least θ, i.e. a false alarm on someone who is not enrolled: `FAR(θ) = |{ p in U : max_sim(p, gallery) >= θ }| / |U|`.
- **TPIR / FPIR / FNIR**, the pairing IJB-B and IJB-C report directly: FNIR(θ) is the fraction of mated searches that fail to return the correct gallery template at or above score θ; TPIR(θ) = 1 - FNIR(θ); FPIR(θ) is the fraction of non-mated searches that return some candidate at or above θ. Wang and Deng's survey states it compactly: "FPIR measures what fraction of comparisons between probe templates and non-mate gallery templates result in a match score exceeding T. At the same time, FNIR measures what fraction of probe searches will fail to match a mated gallery template above a score of T" (Wang & Deng, "Deep Face Recognition: A Survey," arXiv:1804.06655, section IV.B). The Maze et al. IJB-C paper uses the same FNIR definition and a related quantity FPI (a raw count, not yet normalized to FPIR) for its weighted end-to-end metric, and is explicit that this one variant deliberately does not divide by the number of non-mated searches, unlike the closed-form FPIR used elsewhere in the series (Maze, Adams, Duncan, et al., "IARPA Janus Benchmark – C: Face Dataset and Protocol," ICB 2018, http://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf, section 2.3.4). Ryuk should use the normalized FPIR (divide by the number of non-mated probes) since that is what makes the metric comparable across probe-set sizes, and note the IJB-C paper's own footnote as a caveat rather than silently picking a convention.

### How IJB-C builds mated and non-mated probes

IJB-C ships two disjoint galleries, G1 and G2, each built by randomly assigning half of a subject's media to that gallery as their enrollment template and reserving the rest as probes. Because the galleries are disjoint, a probe drawn from a subject enrolled only in G1 is automatically non-mated when searched against G2, giving open-set negatives without needing a separate pool of strangers (Maze et al. 2018, section 2.2).

### Simulating open-set identification on a CelebA subset

CelebA has identity labels for 202,599 images across 10,177 identities (`identity_CelebA.txt`; Liu, Luo, Wang, Tang, "Deep Learning Face Attributes in the Wild," ICCV 2015, https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which is enough to build an IJB-C-style split without needing a licensed benchmark. In Ryuk's own vocabulary:

1. Pick a subset of identities with at least, say, 5 images each (CelebA averages roughly 20 images per identity but this varies).
2. Split those identities into two disjoint groups: **known** identities, who will be enrolled as persons of interest, and **held-out** identities, who never appear in the watchlist. A held-out identity's images only ever appear as probes; that is what stands in for "the face belongs to nobody on the watchlist."
3. For each known identity, enroll one image (or a few, if testing multi-image enrollment) as their watchlist embedding. The identity's remaining images become mated probes: sightings that should match.
4. Every image of a held-out identity becomes a non-mated probe: a sighting that should not match anyone on the watchlist.
5. Run every probe against the watchlist, take the best-matching enrolled identity and its score, and compute rank-1 accuracy on mated probes (closed-set), plus DIR(θ, 1) on mated probes and FAR(θ) or FPIR(θ) on held-out probes as θ sweeps the full score range, to get a DIR-vs-FAR or TPIR-vs-FPIR curve.
6. Keep a second, disjoint draw of known/held-out identities as a validation split for threshold selection (see below), so the reported curve is not tuned on the same data it is measured on.

## Threshold and confidence interval procedure

### Choosing the operating threshold

`GLOSSARY.md` already states the rule: the threshold is "chosen per model from evaluation, never copied from a model's README." Concretely, mirror the LFW View 1 / View 2 separation: pick the threshold on a validation split (or 9 of the 10 folds, for LFW), by whatever business rule matters, for example "the highest threshold that keeps FAR at or under 1%" or "the threshold that maximizes accuracy," and only then apply that frozen threshold to the held-out test split or fold. For open-set identification, the same idea applies to θ in DIR/FPIR: choose θ on a validation draw of known/held-out identities to hit a target FPIR, then freeze it before scoring the test draw.

### Reporting uncertainty

Fogliato, Patil, and Perona, "Confidence Intervals for Error Rates in 1:1 Matching Tasks: Critical Statistical Analysis and Recommendations" (arXiv:2306.01198, https://arxiv.org/abs/2306.01198), is the primary source the ticket asks for on this point. Their central finding: because a face verification test set has dependence (the same identity appears in many pairs), a naive bootstrap that resamples individual pairs, or even one that resamples identities but pools all their comparisons ("subsets bootstrap"), can under-cover, sometimes badly, for the false accept rate. Their recommendation (R1) is a Wilson interval with variance corrected for that dependence:

```
I_FAR = [ (F̂AR + z²/(2N*) ∓ z·sqrt( z²/(4N*²) + F̂AR(1-F̂AR)/N* ) ) / (1 + z²/N*) ]
```

where `z = z_{1-α/2}`, `F̂AR` is the estimated false accept rate, and `N* = max( F̂AR(1-F̂AR) / Var(F̂AR), G/2 )` is an effective sample size that shrinks the naive pair count down to account for how few independent identities `G` actually generated those pairs (same paper, section 4.1, equation 5; the FRR interval mirrors it). They explicitly advise against the naive Wilson interval (which assumes independent pairs) and against the subsets/two-level bootstrap for FAR, both of which they show fail to hit nominal coverage in their experiments; vertex and double-or-nothing bootstrap are their fallback when the error rate itself is not too close to zero.

For Ryuk, given CelebA-subset-sized test sets rather than IJB-C-sized ones, the pragmatic default is an **identity-level percentile bootstrap**: resample identities with replacement, include all of that identity's pairs or probes in the resample, recompute the metric, repeat 1,000+ times, and take the 2.5th/97.5th percentiles as the 95% interval. This is simple to implement and matches the direction of Fogliato et al.'s critique (resample identities, not pairs), even though their paper shows it can still under-cover FAR at very low error rates. When an error rate of interest (FAR at a fixed threshold, or FPIR) comes out very small, run the Wilson-with-adjusted-variance check above as well, and report both if they disagree meaningfully.

## Published numbers to expect

### SFace on LFW

Two different numbers exist for "SFace on LFW," from two different training runs, and Ryuk should know which one it is reproducing:

- The SFace paper itself reports a ResNet50 model trained on CASIA-WebFace, evaluated with the standard LFW protocol above: its best configuration (a=0.90, b=1.30) gets 99.57% on LFW, with the other sigmoid-loss configurations in the 99.48-99.57% range, alongside ArcFace-with-the-same-backbone baselines at 99.52-99.57% and plain softmax at 99.25% (Zhong, Deng, Hu, Zhao, Li, Wen, "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition," arXiv:2205.12010, Table I).
- The OpenCV Zoo's shipped, downloadable SFace ONNX model reports 0.9940 (99.40%) accuracy in its own results table, with the block-quantized and int8 variants at 0.9942 and 0.9932 (https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md). This is the number Ryuk should expect to land near if it evaluates the actual zoo weights, since that model is a different training run than the one in the paper's ablation tables, packaged with 5-landmark alignment.

A few tenths of a percent of drift from either number is normal and comes from crop/alignment differences, not from a broken protocol; multiple percentage points of drift means something in the pipeline (alignment, distance metric, or the fold split) does not match the recipe above.

### ArcFace on LFW

Deng, Guo, Yang, Xue, Kotsia, Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (arXiv:1801.07698, IEEE TPAMI 2021), reports its best ResNet100 models, trained on either MS1MV3 or the IBUG-500K cleaned dataset, at **99.83%** on LFW and 98.0-98.02% on YTF (paper's verification performance table). InsightFace's own model zoo lists a slightly lower 99.77% for the widely-distributed ResNet100/MS1MV2 pretrained model (per the InsightFace wiki's model zoo page), which is the more likely number for Ryuk to reproduce if it downloads that specific public checkpoint rather than retraining. Both numbers use the same 10-fold LFW protocol and `verification.py` implementation described above.

## Open questions and caveats

- IJB-C's own paper (Maze et al. 2018) documents FPI/FNIR for its end-to-end "weighted IET" metric in detail, but does not itself spell out the DIR formula; that traces to the original IJB-A paper (Klare et al., CVPR 2015). Direct extraction of the IJB-A PDF failed in this session (both the CVF and cv-foundation copies parsed as corrupt), so the DIR formula above is reconstructed from a secondary description of that paper plus the standard form used consistently across the FRVT/IJB literature; it should be treated as very likely correct but not verified against the primary PDF text.
- The ticket's phrase "TAR at FAR of 1e-2 and 1e-3" for LFW is a stronger operating point than LFW's own pair count comfortably supports: with 6,000 pairs total and roughly 3,000 negative pairs, a FAR of 1e-3 is only about 3 expected false accepts per fold, so the TAR@FAR=1e-3 number will be noisy. IJB-C exists specifically because LFW "cannot be reliably estimated at lower, operationally relevant FAR values" (Maze et al. 2018, section 1.1); treat LFW's TAR@1e-3 as indicative, not precise, and lean on the confidence interval to say so.
- CelebA does not ship an official open-set identification split the way IJB-C does; the construction above is Ryuk's own design, informed by IJB-C's gallery-splitting principle, not a protocol with its own paper to cite. It should be documented as such (with the exact identity list and seed used) wherever Ryuk reports numbers from it, so results are reproducible even though the split itself is not a published benchmark.
