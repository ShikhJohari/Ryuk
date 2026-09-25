import { Effect, Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import { ApiClient } from "./api-client";
import type { components } from "./schema.gen";

export const Health = Schema.Struct({
  status: Schema.Literal("ok"),
  version: Schema.String,
}).annotations({ identifier: "Health" });

export type Health = typeof Health.Type;

export type HealthMatchesContract = Assert<
  Equals<Health, components["schemas"]["Health"]>
>;

export const getHealth = Effect.flatMap(ApiClient, (api) =>
  api.get("/api/health", Health),
);
