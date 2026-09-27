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

export const WarningCode = Schema.Literal(
  "duplicate_name",
  "looks_like_other",
  "may_not_be_same_person",
).annotations({ identifier: "WarningCode" });

export type WarningCode = typeof WarningCode.Type;

export type WarningCodeMatchesContract = Assert<
  Equals<WarningCode, components["schemas"]["WarningCode"]>
>;

/** One enrollment warning; `personId` names the other person it is about. */
export const EnrollmentWarning = Schema.Struct({
  code: WarningCode,
  detail: Schema.String,
  personId: Schema.NullOr(Schema.String),
}).annotations({ identifier: "EnrollmentWarning" });

export type EnrollmentWarning = typeof EnrollmentWarning.Type;

export type EnrollmentWarningMatchesContract = Assert<
  Equals<EnrollmentWarning, components["schemas"]["EnrollmentWarning"]>
>;

/**
 * `409 warnings`: resend with every listed code acknowledged to proceed.
 * Decoded into `ApiProblem.warnings`.
 */
export const WarningsProblem = Schema.Struct({
  ...Problem.fields,
  warnings: Schema.Array(EnrollmentWarning),
}).annotations({ identifier: "WarningsProblem" });

export type WarningsProblemMatchesContract = Assert<
  Equals<typeof WarningsProblem.Type, components["schemas"]["WarningsProblem"]>
>;
