import { Effect, Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import { ApiClient } from "./api-client";
import type { WarningCode } from "./problem";
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

export const PersonStatus = Schema.Literal("on_watchlist", "removed");
export type PersonStatus = typeof PersonStatus.Type;
export type PersonStatusMatchesContract = Assert<
  Equals<PersonStatus, Schemas["PersonStatus"]>
>;

/** The watchlist filter: a status, or everyone. */
export const StatusFilter = Schema.Literal("on_watchlist", "removed", "all");
export type StatusFilter = typeof StatusFilter.Type;

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

export type PersonOfInterestChangesMatchContract = Assert<
  Equals<{ readonly name: string }, Schemas["PersonOfInterestChanges"]>
>;

const persons = "/api/persons";
const person = (personId: string) =>
  `${persons}/${encodeURIComponent(personId)}`;
const photo = (personId: string, photoId: string) =>
  `${person(personId)}/photos/${encodeURIComponent(photoId)}`;

/** Where the browser loads an enrolled photo from; never cached by the service. */
export const photoImageUrl = (personId: string, photoId: string) =>
  `${import.meta.env.VITE_API_BASE_URL ?? ""}${photo(personId, photoId)}/image`;

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

/** A photo upload, with the warnings the operator has already acknowledged. */
export type PhotoUpload = {
  readonly photo: File;
  readonly acknowledgedWarnings: ReadonlyArray<WarningCode>;
};

const photoForm = ({ photo, acknowledgedWarnings }: PhotoUpload) => {
  const form = new FormData();
  form.append("photo", photo);
  for (const code of acknowledgedWarnings) {
    form.append("acknowledgedWarnings", code);
  }
  return form;
};

export const createPerson = ({
  name,
  ...upload
}: PhotoUpload & { readonly name: string }) =>
  Effect.flatMap(ApiClient, (api) => {
    const form = photoForm(upload);
    form.append("name", name);
    return api.postForm(persons, form, PersonOfInterest);
  });

export const addPhoto = (personId: string, upload: PhotoUpload) =>
  Effect.flatMap(ApiClient, (api) =>
    api.postForm(
      `${person(personId)}/photos`,
      photoForm(upload),
      EnrolledPhoto,
    ),
  );

export const renamePerson = (personId: string, name: string) =>
  Effect.flatMap(ApiClient, (api) =>
    api.patch(person(personId), { name }, PersonOfInterest),
  );

export const deletePhoto = (personId: string, photoId: string) =>
  Effect.flatMap(ApiClient, (api) => api.delete(photo(personId, photoId)));
