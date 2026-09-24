---
status: accepted
---

# Face data lives inside SQLite, erased with secure_delete

Enrolled photos, embeddings and sighting crops are stored as BLOBs in the same SQLite database as the records that own them, not as files beside it, and the database runs with `PRAGMA secure_delete` on. Purge has to erase a person of interest and everything derived from them completely; with every byte in one database, a purge is one transaction that either happens entirely or not at all, and secure_delete overwrites the freed pages instead of leaving the faces recoverable in the file. At Ryuk's scale (hundreds of persons of interest, a few thousand photos, embeddings of 0.5 to 2 KB) BLOB storage costs nothing noticeable.

## Considered options

- Photos and crops as files, embeddings as sidecar `.npy`, the database holding paths. Rejected: a purge interrupted between the database commit and the file deletions leaves face photos on disk that nothing references, which is the worst place for a privacy failure to hide.
- Embeddings as BLOBs, photos as files. Rejected for the same reason; the photos are the more sensitive half.

## Consequences

The database file holds face images, so backing it up or copying it copies them; the report's ethics section says so. The live monitor never searches the database directly: the active model's embeddings for persons on the watchlist are loaded into memory at startup and after each watchlist change.
