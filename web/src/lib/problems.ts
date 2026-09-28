import { HttpClientError } from "@effect/platform";
import { ParseResult } from "effect";
import { ApiProblem } from "@/api/api-client";
import type { EnrollmentWarning } from "@/api/problem";

/**
 * What to tell the operator about each problem `code` the service sends.
 * The service's `detail` is for logs and developers, never shown: the copy
 * here is the client's, in the glossary's terms.
 */
const problemCopy: Readonly<Record<string, string>> = {
  no_face:
    "No face was found in the photo. Choose one that shows their face clearly.",
  multiple_faces:
    "The photo has more than one face large enough to use. Choose a photo of only this person.",
  face_too_small:
    "The face in the photo is too small to use. Choose a closer photo.",
  unsupported_image:
    "The file is not a JPEG, PNG or WebP image that can be read.",
  photo_too_large:
    "The photo is too large. Choose one under 10 MB and 40 megapixels.",
  invalid_name: "Enter a name of at most 200 characters.",
  last_photo:
    "A person of interest's last enrolled photo cannot be deleted. Add another first.",
  not_found:
    "It is no longer there; it may have changed in another tab. Reload the page.",
  cannot_be_active:
    "That model cannot be active: its weights are missing or it has not been evaluated.",
  no_detector:
    "Enrollment needs the face detector's weights. Fetch them with `ryuk weights fetch`, then restart the service.",
  watchlist_unavailable:
    "The watchlist is not running. Restart the service, then try again.",
  cross_origin:
    "The service only accepts changes from a page on this machine. Open Ryuk at 127.0.0.1 or localhost.",
  invalid_host:
    "The service only answers requests addressed to this machine. Open Ryuk at 127.0.0.1 or localhost.",
  invalid_request:
    "The service could not read the request. The client and the service may be different versions: reload the page.",
  internal_error:
    "The service hit an unexpected error. Try again; if it keeps happening, check the service's log.",
  warnings: "The service raised warnings to confirm first.",
};

const UNREACHABLE =
  "The service could not be reached. Check that it is running, then try again.";
const UNREADABLE =
  "The service's answer could not be read. The client and the service may be different versions: rebuild the client or restart the service, then reload the page.";

/** What to tell the operator about a failed call. */
export function problemMessage(error: unknown): string {
  if (error instanceof ApiProblem) {
    return (
      problemCopy[error.code] ??
      `The service refused that (HTTP ${error.status}). Try again; if it keeps happening, check the service's log.`
    );
  }
  if (HttpClientError.isHttpClientError(error)) {
    switch (error.reason) {
      case "Transport":
        return UNREACHABLE;
      case "InvalidUrl":
        return "The service's address is not a valid URL. Check VITE_API_BASE_URL, then rebuild the client.";
      case "Encode":
        return "The request could not be sent. Reload the page, then try again.";
      case "StatusCode":
        // No problem body: the answer came from something between here and
        // the service, such as the dev server's proxy when the service is down.
        return `The service could not be reached or did not explain its answer (HTTP ${error.response.status}). Check that it is running, then try again.`;
      case "Decode":
      case "EmptyBody":
        return UNREADABLE;
    }
  }
  if (ParseResult.isParseError(error)) {
    return UNREADABLE;
  }
  return "Something went wrong. Reload the page, then try again.";
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
