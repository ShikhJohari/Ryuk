import { HttpResponse, http } from "msw";
import type {
  AttributeBreakdown,
  BiasBreakdown,
  EvaluationReport,
  Interval,
  LfwResult,
  LiveOperatingPoints,
  MatchRule,
  MethodComparison,
  ModelRef,
  Network,
  OpenSetResult,
  Rate,
  SmallGallery,
  TpirAtFpir,
} from "@/api/evaluation";
import { problemResponse } from "./api-server";

/** A model as evaluation measured it, keyed as `recognitionModel` keys it in ./monitor. */
function modelRef(network: Network, name: string, dimension: number): ModelRef {
  return {
    id: `${network}-cpu-${"0".repeat(64)}`,
    network,
    provider: "cpu",
    weightsSha256: "0".repeat(64),
    dimension,
    name,
  };
}

export const sfaceRef = modelRef("sface", "SFace", 128);
export const arcfaceRef = modelRef("arcface", "ArcFace (CPU)", 512);
export const facenetRef = modelRef("facenet", "FaceNet", 512);

function interval(low: number, high: number): Interval {
  return { low, high };
}

/** A rate with its bootstrap interval, and an adjusted Wilson interval if given. */
export function rate(
  value: number,
  low: number,
  high: number,
  wilson: Interval | null = null,
): Rate {
  return { value, ci: interval(low, high), adjustedWilson: wilson };
}

function lfwResult(
  model: ModelRef,
  accuracy: number,
  published: LfwResult["published"],
  tar: readonly [number, number],
  changes: Partial<LfwResult> = {},
): LfwResult {
  return {
    model,
    crop: model.network === "facenet" ? "box-margin-32" : "five-point",
    accuracy,
    standardError: 0.0012,
    accuracyIfExcludedWereErrors: accuracy - 0.0138,
    auc: 0.9977,
    published,
    gapPoints: (accuracy - published.accuracy) * 100,
    reproducesPublished: true,
    operatingPoints: [
      {
        targetFar: 0.01,
        far: 0.0098,
        tar: tar[0],
        threshold: 0.289,
        indicative: false,
      },
      {
        targetFar: 0.001,
        far: 0.00068,
        tar: tar[1],
        threshold: 0.369,
        indicative: true,
      },
    ],
    roc: {
      far: [0, 0, 0.00034, 0.00068, 0.0098, 0.1, 1],
      tar: [0, tar[1] - 0.01, tar[1] - 0.005, tar[1], tar[0], 0.999, 1],
    },
    ...changes,
  };
}

function operatingPoints(
  atOne: number,
  atTenth: number,
): ReadonlyArray<TpirAtFpir> {
  return [
    {
      targetFpir: 0.01,
      fpir: 0.0098,
      tpir: rate(atOne, atOne - 0.009, atOne + 0.008),
      threshold: 0.495,
      indicative: false,
    },
    {
      targetFpir: 0.001,
      fpir: 0.00098,
      tpir: rate(atTenth, atTenth - 0.015, atTenth + 0.026),
      threshold: 0.547,
      indicative: true,
    },
  ];
}

function openSetResult(
  model: ModelRef,
  threshold: number,
  tpir: number,
  fpir: number,
  msPerFace: number,
): OpenSetResult {
  return {
    model,
    crop: model.network === "facenet" ? "box-margin-32" : "five-point",
    rule: "best-photo",
    threshold,
    targetFpir: 0.01,
    rank1: rate(0.9804, 0.9759, 0.9848),
    atThreshold: {
      tpir: rate(tpir, tpir - 0.0085, tpir + 0.0077),
      fpir: rate(fpir, 0.0062, 0.0124, interval(0.0064, 0.0129)),
      misidentification: rate(0.0016, 0.0005, 0.0031),
    },
    operatingPoints: operatingPoints(tpir + 0.002, tpir - 0.05),
    curve: {
      fpir: [0, 0.00025, 0.001, 0.01, 0.1, 1],
      tpir: [tpir - 0.3, tpir - 0.06, tpir - 0.05, tpir, tpir + 0.03, 1],
    },
    msPerFace,
  };
}

function method(
  name: MethodComparison["method"],
  tpir: number,
  changes: Partial<MethodComparison> = {},
): MethodComparison {
  return {
    method: name,
    family:
      name === "best-photo" || name === "mean"
        ? "scoring-rule"
        : name === "learned"
          ? "learned-rule"
          : "classifier",
    needsRetraining:
      name !== "best-photo" && name !== "mean" && name !== "learned",
    hyperparameter: null,
    threshold: 0.4943,
    rank1: rate(0.981, 0.976, 0.985),
    atThreshold: {
      tpir: rate(tpir, tpir - 0.008, tpir + 0.008),
      fpir: rate(0.0091, 0.0062, 0.0124),
      misidentification: rate(0.0016, 0.0005, 0.0031),
    },
    operatingPoints: operatingPoints(tpir, tpir - 0.05),
    gain: null,
    ...changes,
  };
}

function gain(value: number, low: number, high: number) {
  return {
    targetFpir: 0.01,
    value,
    ci: interval(low, high),
    improves: low > 0,
  };
}

function attributes(): ReadonlyArray<AttributeBreakdown> {
  return [
    {
      attribute: "Male",
      basis: "identity",
      indicative: false,
      mixedGalleryIdentities: 0,
      mixedHeldOutIdentities: 7,
      fpirRatio: 3.868,
      groups: [
        {
          label: "Male",
          galleryIdentities: 177,
          heldOutIdentities: 240,
          matedProbes: 2655,
          nonMatedProbes: 1956,
          tpir: rate(0.9782, 0.968, 0.987),
          misidentification: rate(0.0011, 0.0003, 0.0023),
          fpir: rate(0.0036, 0.0011, 0.0068),
        },
        {
          label: "Not male",
          galleryIdentities: 323,
          heldOutIdentities: 253,
          matedProbes: 4845,
          nonMatedProbes: 2095,
          tpir: rate(0.935, 0.922, 0.947),
          misidentification: rate(0.0019, 0.0006, 0.0035),
          fpir: rate(0.0138, 0.0089, 0.0195),
        },
      ],
    },
    {
      attribute: "Male_and_Young",
      basis: "identity",
      indicative: true,
      mixedGalleryIdentities: 39,
      mixedHeldOutIdentities: 33,
      fpirRatio: null,
      groups: [
        {
          label: "Male, young",
          galleryIdentities: 112,
          heldOutIdentities: 123,
          matedProbes: 1680,
          nonMatedProbes: 1001,
          tpir: rate(0.9798, 0.968, 0.99),
          misidentification: rate(0.0012, 0.0002, 0.0026),
          fpir: rate(0.0057, 0.0014, 0.0112),
        },
        {
          label: "Not male, not young",
          galleryIdentities: 29,
          heldOutIdentities: 23,
          matedProbes: 435,
          nonMatedProbes: 180,
          tpir: null,
          misidentification: null,
          fpir: null,
        },
      ],
    },
    {
      attribute: "Eyeglasses",
      basis: "photo",
      indicative: false,
      mixedGalleryIdentities: 0,
      mixedHeldOutIdentities: 0,
      fpirRatio: 1.6354,
      groups: [
        {
          label: "Eyeglasses",
          galleryIdentities: 150,
          heldOutIdentities: 121,
          matedProbes: 396,
          nonMatedProbes: 348,
          tpir: rate(0.9217, 0.893, 0.948),
          misidentification: rate(0.0051, 0.0, 0.0127),
          fpir: rate(0.0057, 0.0, 0.0144),
        },
        {
          label: "No eyeglasses",
          galleryIdentities: 496,
          heldOutIdentities: 484,
          matedProbes: 7104,
          nonMatedProbes: 3724,
          tpir: rate(0.9519, 0.944, 0.959),
          misidentification: rate(0.0015, 0.0006, 0.0027),
          fpir: rate(0.0094, 0.0064, 0.0127),
        },
      ],
    },
  ];
}

function biasBreakdown(
  model: ModelRef,
  rule: MatchRule,
  threshold: number,
): BiasBreakdown {
  return { model, rule, threshold, attributes: attributes() };
}

function smallGalleries(
  tpir5: number,
  tpir1: number,
): ReadonlyArray<SmallGallery> {
  return [500, 100, 20, 5].flatMap((identities) => {
    const galleries = 500 / identities;
    const nonMatedProbes = 4072 * galleries;
    const fpir5 = 0.0074 / galleries;
    const fpir1 = 0.0028 / galleries;
    return [
      {
        identities,
        enrolledPhotos: 5,
        galleries,
        matedProbes: 7500,
        nonMatedProbes,
        tpir: rate(tpir5, tpir5 - 0.007, tpir5 + 0.007),
        fpir: rate(fpir5, fpir5 * 0.6, fpir5 * 1.4),
        misidentification: rate(0.0016, 0.0005, 0.0031),
      },
      {
        identities,
        enrolledPhotos: 1,
        galleries,
        matedProbes: 7500,
        nonMatedProbes,
        tpir: rate(tpir1, tpir1 - 0.012, tpir1 + 0.012),
        fpir: rate(fpir1, fpir1 * 0.5, fpir1 * 1.6),
        misidentification: rate(0.0021, 0.0008, 0.0037),
      },
    ];
  });
}

function live(
  model: ModelRef,
  rule: MatchRule,
  threshold: number,
  samePersonThreshold: number,
  atLiveThreshold: readonly [Rate, Rate] | null,
): LiveOperatingPoints {
  return {
    model,
    rule,
    threshold,
    samePerson: {
      threshold: samePersonThreshold,
      targetFar: 0.001,
      validationFar: 0.001,
      validationImpostorPairs: 5707000,
      test: [
        {
          enrolledPhotos: 1,
          matedPairs: 7500,
          impostorPairs: 5778500,
          warningRate: rate(0.0764, 0.0698, 0.0832),
          far: rate(0.00093, 0.00081, 0.00107, interval(0.0008, 0.0011)),
          warningRateAtLiveThreshold: atLiveThreshold?.[0] ?? null,
        },
        {
          enrolledPhotos: 5,
          matedPairs: 7500,
          impostorPairs: 5778500,
          warningRate: rate(0.0165, 0.0131, 0.0203),
          far: rate(0.00356, 0.0032, 0.0039),
          warningRateAtLiveThreshold: atLiveThreshold?.[1] ?? null,
        },
      ],
    },
    smallGalleries: smallGalleries(0.9619, 0.7834),
  };
}

/** A complete evaluation, every section measured, for three models. */
export const evaluationReport: EvaluationReport = {
  dataset: {
    minUsableFaceSize: 70,
    minGalleryImages: 20,
    lfw: {
      identities: 5749,
      images: 13233,
      pairs: [
        {
          name: "pairsDevTrain",
          view: 1,
          folds: 1,
          matched: 1100,
          mismatched: 1100,
          identities: 2132,
          images: 3443,
          pairsWithExcludedImage: 32,
        },
        {
          name: "pairsDevTest",
          view: 1,
          folds: 1,
          matched: 500,
          mismatched: 500,
          identities: 963,
          images: 1549,
          pairsWithExcludedImage: 16,
        },
        {
          name: "pairs",
          view: 2,
          folds: 10,
          matched: 3000,
          mismatched: 3000,
          identities: 4281,
          images: 7701,
          pairsWithExcludedImage: 83,
        },
      ],
      detection: {
        images: 13233,
        detected: 13161,
        multipleFaces: 654,
        usable: 13144,
      },
    },
    celeba: [
      {
        draw: "validation",
        split: "valid",
        identities: 985,
        images: 19867,
        galleryCandidates: 629,
        eligibleIdentities: 584,
        detection: {
          images: 19867,
          detected: 19175,
          multipleFaces: 7,
          usable: 19070,
        },
      },
      {
        draw: "test",
        split: "test",
        identities: 1000,
        images: 19962,
        galleryCandidates: 616,
        eligibleIdentities: 572,
        detection: {
          images: 19962,
          detected: 19314,
          multipleFaces: 20,
          usable: 19228,
        },
      },
    ],
  },
  verification: {
    pairs: 6000,
    scoredPairs: 5917,
    tolerancePoints: 0.5,
    models: [
      lfwResult(
        sfaceRef,
        0.99376,
        {
          accuracy: 0.994,
          source: "https://github.com/opencv/opencv_zoo",
          note: null,
        },
        [0.99326, 0.98349],
      ),
      lfwResult(
        arcfaceRef,
        0.9978,
        {
          accuracy: 0.9983,
          source: "https://github.com/deepinsight/insightface",
          note: "The buffalo_l pack's figure.",
        },
        [0.99697, 0.99629],
      ),
      lfwResult(
        facenetRef,
        0.99358,
        {
          accuracy: 0.9965,
          source: "https://github.com/davidsandberg/facenet",
          note: "99.65 ± 0.25, measured on MTCNN crops with a margin of 32.",
        },
        [0.9936, 0.97944],
      ),
    ],
    sfaceInt8: {
      accuracy: 0.99121,
      standardError: 0.00132,
      cosineToFp32Mean: 0.97125,
      cosineToFp32Min: 0.93046,
      facesCompared: 7643,
      msPerFaceInt8: 11.44,
      msPerFaceFp32: 4.07,
    },
  },
  identification: {
    bootstrapResamples: 2000,
    draws: [
      {
        draw: "validation",
        split: "valid",
        galleryIdentities: 500,
        heldOutIdentities: 485,
        enrolledPhotos: 2500,
        matedProbes: 7500,
        nonMatedProbes: 3929,
      },
      {
        draw: "test",
        split: "test",
        galleryIdentities: 500,
        heldOutIdentities: 500,
        enrolledPhotos: 2500,
        matedProbes: 7500,
        nonMatedProbes: 4072,
      },
    ],
    models: [
      openSetResult(sfaceRef, 0.498, 0.9503, 0.0091, 5.37),
      openSetResult(arcfaceRef, 0.3417, 0.9847, 0.0066, 8.35),
      openSetResult(facenetRef, 0.7088, 0.8836, 0.0096, 11.7),
    ],
  },
  firstActiveModel: {
    model: arcfaceRef,
    reason:
      "ArcFace (CPU) has the highest test TPIR at its frozen threshold (98.47%) of the 2 eligible models.",
    lfwGate: {
      accuracy: "scored-pairs",
      scoredPairs: 5917,
      pairs: 6000,
      tolerancePoints: 0.5,
    },
    maxTestFpir: 0.02,
    maxMsPerFace: 30,
    eligibility: [
      {
        model: sfaceRef,
        lfwGapPoints: -0.0248,
        reproducesLfw: true,
        testTpir: rate(0.9619, 0.9544, 0.9688),
        testFpir: 0.0074,
        fpirWithinLimit: true,
        msPerFace: 5.37,
        fastEnough: true,
        eligible: true,
      },
      {
        model: arcfaceRef,
        lfwGapPoints: -0.0498,
        reproducesLfw: true,
        testTpir: rate(0.9847, 0.9809, 0.9883),
        testFpir: 0.0066,
        fpirWithinLimit: true,
        msPerFace: 8.35,
        fastEnough: true,
        eligible: true,
      },
      {
        model: facenetRef,
        lfwGapPoints: -0.2919,
        reproducesLfw: true,
        testTpir: rate(0.8836, 0.8696, 0.8969),
        testFpir: 0.0251,
        fpirWithinLimit: false,
        msPerFace: 11.7,
        fastEnough: true,
        eligible: false,
      },
    ],
  },
  learning: {
    targetFpir: 0.01,
    models: [
      {
        model: sfaceRef,
        liveRule: "mean",
        liveReason:
          "The mean rule improves test TPIR at FPIR 1% on best-photo by +1.36 points.",
        methods: [
          method("best-photo", 0.9527, { threshold: 0.498 }),
          method("mean", 0.9663, { gain: gain(0.0136, 0.0061, 0.0196) }),
          method("knn", 0.9133, {
            hyperparameter: { name: "k", value: 5 },
            gain: gain(-0.0393, -0.0519, -0.0267),
          }),
          method("learned", 0.954, { gain: gain(0.0013, -0.0013, 0.0051) }),
        ],
      },
      {
        model: arcfaceRef,
        liveRule: "best-photo",
        liveReason:
          "No method that needs no retraining measurably improves test TPIR at FPIR 1% on best-photo.",
        methods: [
          method("best-photo", 0.9849, { threshold: 0.3417 }),
          method("mean", 0.9831, { gain: gain(-0.0019, -0.0039, 0.0001) }),
        ],
      },
      {
        model: facenetRef,
        liveRule: "learned",
        liveReason: "The learned rule improves test TPIR at FPIR 1% the most.",
        methods: [
          method("best-photo", 0.8836, { threshold: 0.7088 }),
          method("learned", 0.9021, { gain: gain(0.0186, 0.009, 0.027) }),
        ],
      },
    ],
  },
  bias: {
    minIdentities: 30,
    agreement: 0.8,
    models: [
      biasBreakdown(sfaceRef, "best-photo", 0.498),
      biasBreakdown(sfaceRef, "mean", 0.4943),
      biasBreakdown(arcfaceRef, "best-photo", 0.3417),
      biasBreakdown(facenetRef, "best-photo", 0.7088),
    ],
  },
  live: {
    models: [
      live(sfaceRef, "mean", 0.4943, 0.3773, [
        rate(0.2152, 0.2051, 0.2256),
        rate(0.0464, 0.0412, 0.0519),
      ]),
      live(arcfaceRef, "best-photo", 0.3417, 0.2161, [
        rate(0.0528, 0.0471, 0.0589),
        rate(0.0149, 0.0118, 0.0183),
      ]),
      live(facenetRef, "learned", 0.6347, 0.5213, null),
    ],
  },
};

/** `GET /api/evaluation` answering with `report`. */
export function evaluationHandler(report: EvaluationReport = evaluationReport) {
  return http.get("*/api/evaluation", () => HttpResponse.json(report));
}

/** `GET /api/evaluation` as the service answers when it started without the results. */
export const evaluationUnavailable = http.get("*/api/evaluation", () =>
  problemResponse({
    type: "about:blank",
    title: "Service Unavailable",
    status: 503,
    detail: "The service was started without the evaluation results.",
    code: "evaluation_unavailable",
  }),
);
