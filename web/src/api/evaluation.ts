import { Effect, Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import { ApiClient } from "./api-client";
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

/*
 * The evaluation page's view of the committed outputs (#32), as
 * `GET /api/evaluation` serves it. Rates and accuracies are fractions in
 * [0, 1]; a gap to a published figure is in percentage points. The service's
 * `evaluation_models.py` documents every field.
 */

export const Network = Schema.Literal("arcface", "facenet", "sface");
export type Network = typeof Network.Type;
export type NetworkMatchesContract = Assert<
  Equals<Network, Schemas["Network"]>
>;

export const Provider = Schema.Literal("cpu", "coreml");
export type ProviderMatchesContract = Assert<
  Equals<typeof Provider.Type, Schemas["Provider"]>
>;

export const Crop = Schema.Literal(
  "five-point",
  "box-margin-14",
  "box-margin-32",
);
export type CropMatchesContract = Assert<
  Equals<typeof Crop.Type, Schemas["Crop"]>
>;

export const Draw = Schema.Literal("validation", "test");
export type Draw = typeof Draw.Type;
export type DrawMatchesContract = Assert<Equals<Draw, Schemas["Draw"]>>;

export const Split = Schema.Literal("valid", "test");
export type SplitMatchesContract = Assert<
  Equals<typeof Split.Type, Schemas["Split"]>
>;

export const MatchRule = Schema.Literal("best-photo", "mean", "learned");
export type MatchRule = typeof MatchRule.Type;
export type MatchRuleMatchesContract = Assert<
  Equals<MatchRule, Schemas["MatchRule"]>
>;

export const Method = Schema.Literal(
  "best-photo",
  "mean",
  "knn",
  "logistic-regression",
  "linear-svm",
  "learned",
);
export type Method = typeof Method.Type;
export type MethodMatchesContract = Assert<Equals<Method, Schemas["Method"]>>;

export const MethodFamily = Schema.Literal(
  "scoring-rule",
  "classifier",
  "learned-rule",
);
export type MethodFamilyMatchesContract = Assert<
  Equals<typeof MethodFamily.Type, Schemas["MethodFamily"]>
>;

export const BiasAttribute = Schema.Literal(
  "Male",
  "Young",
  "Male_and_Young",
  "Eyeglasses",
  "Wearing_Hat",
  "Blurry",
);
export type BiasAttribute = typeof BiasAttribute.Type;
export type BiasAttributeMatchesContract = Assert<
  Equals<BiasAttribute, Schemas["BiasAttribute"]>
>;

/** `identity`: each identity's majority label; `photo`: each probe's own. */
export const GroupBasis = Schema.Literal("identity", "photo");
export type GroupBasisMatchesContract = Assert<
  Equals<typeof GroupBasis.Type, Schemas["GroupBasis"]>
>;

/** A 95% interval; on a gain, a difference of rates, so `low` may be negative. */
export const Interval = Schema.Struct({
  low: Schema.Number,
  high: Schema.Number,
}).annotations({ identifier: "Interval" });
export type Interval = typeof Interval.Type;
export type IntervalMatchesContract = Assert<
  Equals<Interval, Schemas["Interval"]>
>;

/**
 * A rate with its 95% identity-level bootstrap interval, and the
 * dependence-adjusted Wilson interval for a rate within 1% of 0 or 100%.
 */
export const Rate = Schema.Struct({
  value: Schema.Number,
  ci: Interval,
  adjustedWilson: Schema.NullOr(Interval),
}).annotations({ identifier: "Rate" });
export type Rate = typeof Rate.Type;
export type RateMatchesContract = Assert<Equals<Rate, Schemas["Rate"]>>;

/** A recognition model as evaluation measured it; `id` is `GET /api/models`' `id`. */
export const ModelRef = Schema.Struct({
  id: Schema.String,
  network: Network,
  provider: Provider,
  weightsSha256: Schema.String,
  dimension: Schema.Int,
  name: Schema.String,
}).annotations({ identifier: "ModelRef" });
export type ModelRef = typeof ModelRef.Type;
export type ModelRefMatchesContract = Assert<
  Equals<ModelRef, Schemas["ModelRef"]>
>;

// The datasets.

export const DetectionCounts = Schema.Struct({
  images: Schema.Int,
  detected: Schema.Int,
  multipleFaces: Schema.Int,
  usable: Schema.Int,
}).annotations({ identifier: "DetectionCounts" });
export type DetectionCounts = typeof DetectionCounts.Type;
export type DetectionCountsMatchesContract = Assert<
  Equals<DetectionCounts, Schemas["DetectionCounts"]>
>;

export const LfwPairList = Schema.Struct({
  name: Schema.Literal("pairsDevTrain", "pairsDevTest", "pairs"),
  view: Schema.Int,
  folds: Schema.Int,
  matched: Schema.Int,
  mismatched: Schema.Int,
  identities: Schema.Int,
  images: Schema.Int,
  pairsWithExcludedImage: Schema.Int,
}).annotations({ identifier: "LfwPairList" });
export type LfwPairList = typeof LfwPairList.Type;
export type LfwPairListMatchesContract = Assert<
  Equals<LfwPairList, Schemas["LfwPairList"]>
>;

export const LfwDataset = Schema.Struct({
  identities: Schema.Int,
  images: Schema.Int,
  pairs: Schema.Array(LfwPairList),
  detection: DetectionCounts,
}).annotations({ identifier: "LfwDataset" });
export type LfwDatasetMatchesContract = Assert<
  Equals<typeof LfwDataset.Type, Schemas["LfwDataset"]>
>;

export const CelebaDataset = Schema.Struct({
  draw: Draw,
  split: Split,
  identities: Schema.Int,
  images: Schema.Int,
  galleryCandidates: Schema.Int,
  eligibleIdentities: Schema.Int,
  detection: DetectionCounts,
}).annotations({ identifier: "CelebaDataset" });
export type CelebaDataset = typeof CelebaDataset.Type;
export type CelebaDatasetMatchesContract = Assert<
  Equals<CelebaDataset, Schemas["CelebaDataset"]>
>;

export const DatasetSummary = Schema.Struct({
  minUsableFaceSize: Schema.Int,
  minGalleryImages: Schema.Int,
  lfw: LfwDataset,
  celeba: Schema.Array(CelebaDataset),
}).annotations({ identifier: "DatasetSummary" });
export type DatasetSummary = typeof DatasetSummary.Type;
export type DatasetSummaryMatchesContract = Assert<
  Equals<DatasetSummary, Schemas["DatasetSummary"]>
>;

// LFW verification.

export const TarAtFar = Schema.Struct({
  targetFar: Schema.Number,
  far: Schema.Number,
  tar: Schema.Number,
  threshold: Schema.NullOr(Schema.Number),
  indicative: Schema.Boolean,
}).annotations({ identifier: "TarAtFar" });
export type TarAtFar = typeof TarAtFar.Type;
export type TarAtFarMatchesContract = Assert<
  Equals<TarAtFar, Schemas["TarAtFar"]>
>;

/** FAR against TAR from (0, 0) to (1, 1), paired by index. */
export const RocCurve = Schema.Struct({
  far: Schema.Array(Schema.Number),
  tar: Schema.Array(Schema.Number),
}).annotations({ identifier: "RocCurve" });
export type RocCurveMatchesContract = Assert<
  Equals<typeof RocCurve.Type, Schemas["RocCurve"]>
>;

/** A published accuracy; any spread quoted with it is in `note`, as its source states it. */
export const PublishedAccuracy = Schema.Struct({
  accuracy: Schema.Number,
  source: Schema.String,
  note: Schema.NullOr(Schema.String),
}).annotations({ identifier: "PublishedAccuracy" });
export type PublishedAccuracyMatchesContract = Assert<
  Equals<typeof PublishedAccuracy.Type, Schemas["PublishedAccuracy"]>
>;

export const LfwResult = Schema.Struct({
  model: ModelRef,
  crop: Crop,
  accuracy: Schema.Number,
  standardError: Schema.Number,
  accuracyIfExcludedWereErrors: Schema.Number,
  auc: Schema.Number,
  published: PublishedAccuracy,
  gapPoints: Schema.Number,
  reproducesPublished: Schema.Boolean,
  operatingPoints: Schema.Array(TarAtFar),
  roc: RocCurve,
}).annotations({ identifier: "LfwResult" });
export type LfwResult = typeof LfwResult.Type;
export type LfwResultMatchesContract = Assert<
  Equals<LfwResult, Schemas["LfwResult"]>
>;

/** SFace int8 on the same pairs; its ms per face times the embedding alone. */
export const SfaceInt8 = Schema.Struct({
  accuracy: Schema.Number,
  standardError: Schema.Number,
  cosineToFp32Mean: Schema.Number,
  cosineToFp32Min: Schema.Number,
  facesCompared: Schema.Int,
  msPerFaceInt8: Schema.Number,
  msPerFaceFp32: Schema.Number,
}).annotations({ identifier: "SfaceInt8" });
export type SfaceInt8 = typeof SfaceInt8.Type;
export type SfaceInt8MatchesContract = Assert<
  Equals<SfaceInt8, Schemas["SfaceInt8"]>
>;

export const VerificationReport = Schema.Struct({
  pairs: Schema.Int,
  scoredPairs: Schema.Int,
  tolerancePoints: Schema.Number,
  models: Schema.Array(LfwResult),
  sfaceInt8: SfaceInt8,
}).annotations({ identifier: "VerificationReport" });
export type VerificationReport = typeof VerificationReport.Type;
export type VerificationReportMatchesContract = Assert<
  Equals<VerificationReport, Schemas["VerificationReport"]>
>;

// CelebA open-set identification.

export const OpenSetDraw = Schema.Struct({
  draw: Draw,
  split: Split,
  galleryIdentities: Schema.Int,
  heldOutIdentities: Schema.Int,
  enrolledPhotos: Schema.Int,
  matedProbes: Schema.Int,
  nonMatedProbes: Schema.Int,
}).annotations({ identifier: "OpenSetDraw" });
export type OpenSetDraw = typeof OpenSetDraw.Type;
export type OpenSetDrawMatchesContract = Assert<
  Equals<OpenSetDraw, Schemas["OpenSetDraw"]>
>;

/** TPIR read off the test curve at a target FPIR, not at the frozen threshold. */
export const TpirAtFpir = Schema.Struct({
  targetFpir: Schema.Number,
  fpir: Schema.Number,
  tpir: Rate,
  threshold: Schema.NullOr(Schema.Number),
  indicative: Schema.Boolean,
}).annotations({ identifier: "TpirAtFpir" });
export type TpirAtFpir = typeof TpirAtFpir.Type;
export type TpirAtFpirMatchesContract = Assert<
  Equals<TpirAtFpir, Schemas["TpirAtFpir"]>
>;

/** TPIR against FPIR on the test draw, paired by index; the first FPIR may be 0. */
export const OpenSetCurve = Schema.Struct({
  fpir: Schema.Array(Schema.Number),
  tpir: Schema.Array(Schema.Number),
}).annotations({ identifier: "OpenSetCurve" });
export type OpenSetCurveMatchesContract = Assert<
  Equals<typeof OpenSetCurve.Type, Schemas["OpenSetCurve"]>
>;

export const AtThreshold = Schema.Struct({
  tpir: Rate,
  fpir: Rate,
  misidentification: Rate,
}).annotations({ identifier: "AtThreshold" });
export type AtThresholdMatchesContract = Assert<
  Equals<typeof AtThreshold.Type, Schemas["AtThreshold"]>
>;

export const OpenSetResult = Schema.Struct({
  model: ModelRef,
  crop: Crop,
  rule: MatchRule,
  threshold: Schema.Number,
  targetFpir: Schema.Number,
  rank1: Rate,
  atThreshold: AtThreshold,
  operatingPoints: Schema.Array(TpirAtFpir),
  curve: OpenSetCurve,
  msPerFace: Schema.Number,
}).annotations({ identifier: "OpenSetResult" });
export type OpenSetResult = typeof OpenSetResult.Type;
export type OpenSetResultMatchesContract = Assert<
  Equals<OpenSetResult, Schemas["OpenSetResult"]>
>;

export const IdentificationReport = Schema.Struct({
  bootstrapResamples: Schema.Int,
  draws: Schema.Array(OpenSetDraw),
  models: Schema.Array(OpenSetResult),
}).annotations({ identifier: "IdentificationReport" });
export type IdentificationReport = typeof IdentificationReport.Type;
export type IdentificationReportMatchesContract = Assert<
  Equals<IdentificationReport, Schemas["IdentificationReport"]>
>;

// The first active model.

export const LfwGate = Schema.Struct({
  accuracy: Schema.Literal("scored-pairs"),
  scoredPairs: Schema.Int,
  pairs: Schema.Int,
  tolerancePoints: Schema.Number,
}).annotations({ identifier: "LfwGate" });
export type LfwGateMatchesContract = Assert<
  Equals<typeof LfwGate.Type, Schemas["LfwGate"]>
>;

export const Eligibility = Schema.Struct({
  model: ModelRef,
  lfwGapPoints: Schema.NullOr(Schema.Number),
  reproducesLfw: Schema.Boolean,
  testTpir: Rate,
  testFpir: Schema.Number,
  fpirWithinLimit: Schema.Boolean,
  msPerFace: Schema.Number,
  fastEnough: Schema.Boolean,
  eligible: Schema.Boolean,
}).annotations({ identifier: "Eligibility" });
export type Eligibility = typeof Eligibility.Type;
export type EligibilityMatchesContract = Assert<
  Equals<Eligibility, Schemas["Eligibility"]>
>;

export const FirstActiveModel = Schema.Struct({
  model: Schema.NullOr(ModelRef),
  reason: Schema.String,
  lfwGate: LfwGate,
  maxTestFpir: Schema.Number,
  maxMsPerFace: Schema.Number,
  eligibility: Schema.Array(Eligibility),
}).annotations({ identifier: "FirstActiveModel" });
export type FirstActiveModel = typeof FirstActiveModel.Type;
export type FirstActiveModelMatchesContract = Assert<
  Equals<FirstActiveModel, Schemas["FirstActiveModel"]>
>;

// Learning on embeddings.

export const Hyperparameter = Schema.Struct({
  name: Schema.Literal("k", "C"),
  value: Schema.Number,
}).annotations({ identifier: "Hyperparameter" });
export type HyperparameterMatchesContract = Assert<
  Equals<typeof Hyperparameter.Type, Schemas["Hyperparameter"]>
>;

/** A method's TPIR at `targetFpir` minus the baseline's; `improves` iff `ci` is wholly above 0. */
export const Gain = Schema.Struct({
  targetFpir: Schema.Number,
  value: Schema.Number,
  ci: Interval,
  improves: Schema.Boolean,
}).annotations({ identifier: "Gain" });
export type Gain = typeof Gain.Type;
export type GainMatchesContract = Assert<Equals<Gain, Schemas["Gain"]>>;

export const MethodComparison = Schema.Struct({
  method: Method,
  family: MethodFamily,
  needsRetraining: Schema.Boolean,
  hyperparameter: Schema.NullOr(Hyperparameter),
  threshold: Schema.Number,
  rank1: Rate,
  atThreshold: AtThreshold,
  operatingPoints: Schema.Array(TpirAtFpir),
  gain: Schema.NullOr(Gain),
}).annotations({ identifier: "MethodComparison" });
export type MethodComparison = typeof MethodComparison.Type;
export type MethodComparisonMatchesContract = Assert<
  Equals<MethodComparison, Schemas["MethodComparison"]>
>;

export const LearningComparison = Schema.Struct({
  model: ModelRef,
  liveRule: MatchRule,
  liveReason: Schema.String,
  methods: Schema.Array(MethodComparison),
}).annotations({ identifier: "LearningComparison" });
export type LearningComparison = typeof LearningComparison.Type;
export type LearningComparisonMatchesContract = Assert<
  Equals<LearningComparison, Schemas["LearningComparison"]>
>;

export const LearningReport = Schema.Struct({
  targetFpir: Schema.Number,
  models: Schema.Array(LearningComparison),
}).annotations({ identifier: "LearningReport" });
export type LearningReport = typeof LearningReport.Type;
export type LearningReportMatchesContract = Assert<
  Equals<LearningReport, Schemas["LearningReport"]>
>;

// The bias breakdown.

/** One group's rates; a rate is null when too few identities stand behind it. */
export const GroupRates = Schema.Struct({
  label: Schema.String,
  galleryIdentities: Schema.Int,
  heldOutIdentities: Schema.Int,
  matedProbes: Schema.Int,
  nonMatedProbes: Schema.Int,
  tpir: Schema.NullOr(Rate),
  misidentification: Schema.NullOr(Rate),
  fpir: Schema.NullOr(Rate),
}).annotations({ identifier: "GroupRates" });
export type GroupRates = typeof GroupRates.Type;
export type GroupRatesMatchesContract = Assert<
  Equals<GroupRates, Schemas["GroupRates"]>
>;

export const AttributeBreakdown = Schema.Struct({
  attribute: BiasAttribute,
  basis: GroupBasis,
  indicative: Schema.Boolean,
  mixedGalleryIdentities: Schema.Int,
  mixedHeldOutIdentities: Schema.Int,
  fpirRatio: Schema.NullOr(Schema.Number),
  groups: Schema.Array(GroupRates),
}).annotations({ identifier: "AttributeBreakdown" });
export type AttributeBreakdown = typeof AttributeBreakdown.Type;
export type AttributeBreakdownMatchesContract = Assert<
  Equals<AttributeBreakdown, Schemas["AttributeBreakdown"]>
>;

export const BiasBreakdown = Schema.Struct({
  model: ModelRef,
  rule: MatchRule,
  threshold: Schema.Number,
  attributes: Schema.Array(AttributeBreakdown),
}).annotations({ identifier: "BiasBreakdown" });
export type BiasBreakdown = typeof BiasBreakdown.Type;
export type BiasBreakdownMatchesContract = Assert<
  Equals<BiasBreakdown, Schemas["BiasBreakdown"]>
>;

export const BiasReport = Schema.Struct({
  minIdentities: Schema.Int,
  agreement: Schema.Number,
  models: Schema.Array(BiasBreakdown),
}).annotations({ identifier: "BiasReport" });
export type BiasReport = typeof BiasReport.Type;
export type BiasReportMatchesContract = Assert<
  Equals<BiasReport, Schemas["BiasReport"]>
>;

// The live operating points.

export const SamePersonRates = Schema.Struct({
  enrolledPhotos: Schema.Int,
  matedPairs: Schema.Int,
  impostorPairs: Schema.Int,
  warningRate: Rate,
  far: Rate,
  warningRateAtLiveThreshold: Schema.NullOr(Rate),
}).annotations({ identifier: "SamePersonRates" });
export type SamePersonRates = typeof SamePersonRates.Type;
export type SamePersonRatesMatchesContract = Assert<
  Equals<SamePersonRates, Schemas["SamePersonRates"]>
>;

export const SamePersonThreshold = Schema.Struct({
  threshold: Schema.Number,
  targetFar: Schema.Number,
  validationFar: Schema.Number,
  validationImpostorPairs: Schema.Int,
  test: Schema.Array(SamePersonRates),
}).annotations({ identifier: "SamePersonThreshold" });
export type SamePersonThresholdMatchesContract = Assert<
  Equals<typeof SamePersonThreshold.Type, Schemas["SamePersonThreshold"]>
>;

export const SmallGallery = Schema.Struct({
  identities: Schema.Int,
  enrolledPhotos: Schema.Int,
  galleries: Schema.Int,
  matedProbes: Schema.Int,
  nonMatedProbes: Schema.Int,
  tpir: Rate,
  fpir: Rate,
  misidentification: Rate,
}).annotations({ identifier: "SmallGallery" });
export type SmallGallery = typeof SmallGallery.Type;
export type SmallGalleryMatchesContract = Assert<
  Equals<SmallGallery, Schemas["SmallGallery"]>
>;

export const LiveOperatingPoints = Schema.Struct({
  model: ModelRef,
  rule: MatchRule,
  threshold: Schema.Number,
  samePerson: SamePersonThreshold,
  smallGalleries: Schema.Array(SmallGallery),
}).annotations({ identifier: "LiveOperatingPoints" });
export type LiveOperatingPoints = typeof LiveOperatingPoints.Type;
export type LiveOperatingPointsMatchesContract = Assert<
  Equals<LiveOperatingPoints, Schemas["LiveOperatingPoints"]>
>;

export const LiveReport = Schema.Struct({
  models: Schema.Array(LiveOperatingPoints),
}).annotations({ identifier: "LiveReport" });
export type LiveReport = typeof LiveReport.Type;
export type LiveReportMatchesContract = Assert<
  Equals<LiveReport, Schemas["LiveReport"]>
>;

/** Everything the evaluation page shows. A section is null until the command that measures it has run. */
export const EvaluationReport = Schema.Struct({
  dataset: DatasetSummary,
  verification: VerificationReport,
  identification: Schema.NullOr(IdentificationReport),
  firstActiveModel: Schema.NullOr(FirstActiveModel),
  learning: Schema.NullOr(LearningReport),
  bias: Schema.NullOr(BiasReport),
  live: Schema.NullOr(LiveReport),
}).annotations({ identifier: "EvaluationReport" });
export type EvaluationReport = typeof EvaluationReport.Type;
export type EvaluationReportMatchesContract = Assert<
  Equals<EvaluationReport, Schemas["EvaluationReport"]>
>;

/** `503 evaluation_unavailable` when the service started without the results. */
export const getEvaluation = Effect.flatMap(ApiClient, (api) =>
  api.get("/api/evaluation", EvaluationReport),
);
