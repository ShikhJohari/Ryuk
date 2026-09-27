import { Effect, Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import { ApiClient } from "./api-client";
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

export const ModelState = Schema.Literal(
  "active",
  "available",
  "not_evaluated",
  "unavailable",
);
export type ModelState = typeof ModelState.Type;
export type ModelStateMatchesContract = Assert<
  Equals<ModelState, Schemas["ModelState"]>
>;

export const RecognitionModelInfo = Schema.Struct({
  id: Schema.String,
  network: Schema.Literal("arcface", "facenet", "sface"),
  provider: Schema.Literal("cpu", "coreml"),
  weightsSha256: Schema.String,
  name: Schema.String,
  state: ModelState,
  threshold: Schema.NullOr(Schema.Number),
  dimension: Schema.Int,
  msPerFace: Schema.NullOr(Schema.Number),
}).annotations({ identifier: "RecognitionModelInfo" });
export type RecognitionModelInfo = typeof RecognitionModelInfo.Type;
export type RecognitionModelInfoMatchesContract = Assert<
  Equals<RecognitionModelInfo, Schemas["RecognitionModelInfo"]>
>;

export type ActiveModelChoiceMatchesContract = Assert<
  Equals<{ readonly modelKey: string }, Schemas["ActiveModelChoice"]>
>;

/** Only an evaluated model with its weights present can be active. */
export const canBeActive = (model: RecognitionModelInfo) =>
  model.state === "active" || model.state === "available";

export const listModels = Effect.flatMap(ApiClient, (api) =>
  api.get("/api/models", Schema.Array(RecognitionModelInfo)),
);

export const setActiveModel = (modelKey: string) =>
  Effect.flatMap(ApiClient, (api) =>
    api.put("/api/active-model", { modelKey }, RecognitionModelInfo),
  );
