import { Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import type { components } from "./schema.gen";

/**
 * RFC 9457 problem detail, served as application/problem+json on every
 * non-2xx response. `code` is the stable machine code (e.g. `not_found`).
 */
export const Problem = Schema.Struct({
  type: Schema.String,
  title: Schema.String,
  status: Schema.Int,
  detail: Schema.String,
  code: Schema.String,
}).annotations({ identifier: "Problem" });

export type Problem = typeof Problem.Type;

export type ProblemMatchesContract = Assert<
  Equals<Problem, components["schemas"]["Problem"]>
>;
