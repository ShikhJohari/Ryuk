import { ApiProblem } from "@/api/api-client";
import type { EnrollmentWarning } from "@/api/problem";

/** What to tell the operator about a failed call. */
export function problemMessage(error: unknown): string {
  return error instanceof ApiProblem
    ? error.detail
    : "The service could not be reached. Check that it is running, then try again.";
}

/** The warnings of a `409 warnings`, or null for any other failure. */
export function warningsOf(
  error: unknown,
): ReadonlyArray<EnrollmentWarning> | null {
  return error instanceof ApiProblem && error.code === "warnings"
    ? (error.warnings ?? [])
    : null;
}

/** True when the service answered that nothing has this ID. */
export function isNotFound(error: unknown): boolean {
  return error instanceof ApiProblem && error.code === "not_found";
}
