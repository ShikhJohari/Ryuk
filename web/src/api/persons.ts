import { Effect, Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import { ApiClient } from "./api-client";
import { API_BASE_URL } from "./base-url";
import type { WarningCode } from "./problem";
import type { components, operations } from "./schema.gen";

type Schemas = components["schemas"];

export const PersonStatus = Schema.Literal("on_watchlist", "removed");
export type PersonStatus = typeof PersonStatus.Type;
export type PersonStatusMatchesContract = Assert<
  Equals<PersonStatus, Schemas["PersonStatus"]>
>;

/** The watchlist filter: a status, or everyone. */
export const StatusFilter = Schema.Literal("on_watchlist", "removed", "all");
export type StatusFilter = typeof StatusFilter.Type;
export type StatusFilterMatchesContract = Assert<
  Equals<
    StatusFilter,
    NonNullable<operations["listPersons"]["parameters"]["query"]>["status"] &
      string
  >
>;

/** The image types the service accepts for enrolled photos, up to 10 MB. */
export const PHOTO_TYPES = "image/jpeg,image/png,image/webp";

export const EnrolledPhoto = Schema.Struct({
  id: Schema.String,
  mediaType: Schema.String,
  width: Schema.Int,
  height: Schema.Int,
  createdAt: Schema.String,
}).annotations({ identifier: "EnrolledPhoto" });
export type EnrolledPhoto = typeof EnrolledPhoto.Type;
export type EnrolledPhotoMatchesContract = Assert<
  Equals<EnrolledPhoto, Schemas["EnrolledPhoto"]>
>;

export const PersonOfInterestSummary = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  status: PersonStatus,
  photoCount: Schema.Int,
  coverPhotoId: Schema.String,
  createdAt: Schema.String,
  statusChangedAt: Schema.String,
}).annotations({ identifier: "PersonOfInterestSummary" });
export type PersonOfInterestSummary = typeof PersonOfInterestSummary.Type;
export type PersonOfInterestSummaryMatchesContract = Assert<
  Equals<PersonOfInterestSummary, Schemas["PersonOfInterestSummary"]>
>;

export const PersonOfInterest = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  status: PersonStatus,
  createdAt: Schema.String,
  statusChangedAt: Schema.String,
  photos: Schema.Array(EnrolledPhoto),
}).annotations({ identifier: "PersonOfInterest" });
export type PersonOfInterest = typeof PersonOfInterest.Type;
export type PersonOfInterestMatchesContract = Assert<
  Equals<PersonOfInterest, Schemas["PersonOfInterest"]>
>;

/** What a PATCH changes: a field left out is left as it is. */
export type PersonOfInterestChanges = {
  readonly name?: string;
  readonly status?: PersonStatus;
};
export type PersonOfInterestChangesMatchContract = Assert<
  Equals<PersonOfInterestChanges, Schemas["PersonOfInterestChanges"]>
>;

const persons = "/api/persons";
const person = (personId: string) =>
  `${persons}/${encodeURIComponent(personId)}`;
const photo = (personId: string, photoId: string) =>
  `${person(personId)}/photos/${encodeURIComponent(photoId)}`;

/** Where the browser loads an enrolled photo from; never cached by the service. */
export const photoImageUrl = (personId: string, photoId: string) =>
  `${API_BASE_URL}${photo(personId, photoId)}/image`;

export const listPersons = (status: StatusFilter) =>
  Effect.flatMap(ApiClient, (api) =>
    api.get(
      `${persons}?${new URLSearchParams({ status })}`,
      Schema.Array(PersonOfInterestSummary),
    ),
  );

export const getPerson = (personId: string) =>
  Effect.flatMap(ApiClient, (api) =>
    api.get(person(personId), PersonOfInterest),
  );

/**
 * A photo upload, with the warnings the operator has already acknowledged:
 * the multipart fields of `POST /api/persons/{id}/photos`, by the contract's
 * names. `acknowledgedWarnings` is sent as one field per code.
 */
export type PhotoUpload = {
  readonly photo: File;
  readonly acknowledgedWarnings: ReadonlyArray<WarningCode>;
};
export type PhotoUploadMatchesContract = Assert<
  Equals<keyof PhotoUpload, keyof Schemas["Body_addPhoto"]>
>;
export type AcknowledgedWarningsMatchContract = Assert<
  Equals<
    PhotoUpload["acknowledgedWarnings"],
    NonNullable<Schemas["Body_addPhoto"]["acknowledgedWarnings"]>
  >
>;

/** The multipart fields of `POST /api/persons`, by the contract's names. */
export type Enrollment = PhotoUpload & { readonly name: string };
export type EnrollmentMatchesContract = Assert<
  Equals<keyof Enrollment, keyof Schemas["Body_createPerson"]>
>;

/** Multipart fields by name, a list as one field per item. */
function formData(
  fields: Readonly<Record<string, string | Blob | ReadonlyArray<string>>>,
): FormData {
  const form = new FormData();
  for (const [name, value] of Object.entries(fields)) {
    if (typeof value === "string" || value instanceof Blob) {
      form.append(name, value);
    } else {
      for (const item of value) {
        form.append(name, item);
      }
    }
  }
  return form;
}

export const createPerson = (enrollment: Enrollment) =>
  Effect.flatMap(ApiClient, (api) =>
    api.postForm(persons, formData(enrollment), PersonOfInterest),
  );

export const addPhoto = (personId: string, upload: PhotoUpload) =>
  Effect.flatMap(ApiClient, (api) =>
    api.postForm(`${person(personId)}/photos`, formData(upload), EnrolledPhoto),
  );

const updatePerson = (personId: string, changes: PersonOfInterestChanges) =>
  Effect.flatMap(ApiClient, (api) =>
    api.patch(person(personId), changes, PersonOfInterest),
  );

export const renamePerson = (personId: string, name: string) =>
  updatePerson(personId, { name });

/**
 * Removal takes them off the watchlist, keeping their photos and sightings;
 * restoring puts them back. Either way their open sighting ends.
 */
export const setPersonStatus = (personId: string, status: PersonStatus) =>
  updatePerson(personId, { status });

/**
 * Purge: erases them with their enrolled photos, embeddings and sightings,
 * and clears them as runner-up on anyone else's. Cannot be undone.
 */
export const purgePerson = (personId: string) =>
  Effect.flatMap(ApiClient, (api) => api.delete(person(personId)));

export const deletePhoto = (personId: string, photoId: string) =>
  Effect.flatMap(ApiClient, (api) => api.delete(photo(personId, photoId)));
