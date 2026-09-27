import type {
  EnrolledPhoto,
  PersonOfInterest,
  PersonOfInterestSummary,
} from "@/api/persons";

export function enrolledPhoto(id: string): EnrolledPhoto {
  return {
    id,
    mediaType: "image/jpeg",
    width: 400,
    height: 400,
    createdAt: "2026-09-27T10:00:00Z",
  };
}

export function personOfInterest(
  id: string,
  name: string,
  photoIds: ReadonlyArray<string> = [`${id}-photo-1`],
): PersonOfInterest {
  return {
    id,
    name,
    status: "on_watchlist",
    createdAt: "2026-09-27T10:00:00Z",
    statusChangedAt: "2026-09-27T10:00:00Z",
    photos: photoIds.map(enrolledPhoto),
  };
}

export function summary(person: PersonOfInterest): PersonOfInterestSummary {
  return {
    id: person.id,
    name: person.name,
    status: person.status,
    photoCount: person.photos.length,
    coverPhotoId: person.photos[0]?.id ?? "",
    createdAt: person.createdAt,
    statusChangedAt: person.statusChangedAt,
  };
}
